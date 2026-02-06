# self_guide.py
import json
import time
import os
import re
import sys
import uuid
from glob import glob
from typing import Optional, Tuple, List, Any, Dict



from pathlib import Path
import subprocess

try:
    from src.llm_client import chat_complete, resolve_model
    from src.utils.dataset_io import load_dataset, resolve_data_path, validate_openai_api_key
    from src.utils.scoring import (
        extract_gsm8k_final_number as extract_gsm8k_number,
        postprocess_pred as postprocess_gsm8k_pred,
        judge_correctness as judge_dataset_correctness,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.llm_client import chat_complete, resolve_model
    from src.utils.dataset_io import load_dataset, resolve_data_path, validate_openai_api_key
    from src.utils.scoring import (
        extract_gsm8k_final_number as extract_gsm8k_number,
        postprocess_pred as postprocess_gsm8k_pred,
        judge_correctness as judge_dataset_correctness,
    )

from src.abs.common_entry import create_base_parser, run_main
from src.metrics import compute_change_type, normalize_answer
from src.common import (
    PROMPT_CONTRACT_VERSION,
    PROMPT_VERSIONS,
    PROLOG_CONTRACT_VERSION,
    build_config_hash,
    normalize_error_code,
    validate_output_format,
    validate_prolog_contract,
    validate_prompt_inputs,
)
# 解析 Guideline 里的 task_type（Yes/No/Partial）
# 构造 “只输出 Prolog” 的提示词（给 Round C 的 Prolog 生成用）
# 把 Prolog 输出清洗成 clauses，并 subprocess 调 CaRing 的 call_swipl.py
# ====== CaRing tool path ======
CARING_TASK_DIR = {
    "gsm8k": Path(__file__).resolve().parents[1] / "caring" / "tasks" / "gsm8k",
    "prontoqa": Path(__file__).resolve().parents[1] / "caring" / "tasks" / "prontoqa",
    "proofwriter": Path(__file__).resolve().parents[1] / "caring" / "tasks" / "proofwriter",
}

SWIPL_OUT_SCHEMA_VERSION = "1.0"
SWIPL_REQUIRED_KEYS = {
    "schema_version",
    "ok",
    "answer",
    "proof",
    "error_code",
}

ANSWER_PLACEHOLDERS = {"unknown", "none", "null", "n/a", "na"}


def is_nonempty_answer(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text.lower() not in ANSWER_PLACEHOLDERS


def is_nonempty_proof(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and text != "NO_PROOF_RETURNED")


def parse_caring_swipl_answer(raw: str) -> Dict[str, Any]:
    """严格按 schema_version 解析 CaRing call_swipl 的 out.json。"""
    parsed: Dict[str, Any] = {
        "schema_version": SWIPL_OUT_SCHEMA_VERSION,
        "ok": False,
        "answer": None,
        "proof": None,
        "error_code": "SCHEMA_INVALID_JSON",
        "raw": raw or "",
        "legacy": False,
        "validation_error": None,
        "solution_count": None,
    }
    body = (raw or "").strip()
    if not body:
        parsed["error_code"] = "EMPTY_SWIPL_OUTPUT"
        return parsed

    try:
        obj = json.loads(body)
    except Exception:
        parsed["error_code"] = "SCHEMA_INVALID_JSON"
        return parsed

    if not isinstance(obj, dict):
        parsed["error_code"] = "SCHEMA_EXPECTED_OBJECT"
        return parsed

    schema_version = obj.get("schema_version")
    if schema_version is None:
        parsed["schema_version"] = "LEGACY_SCHEMA"
        parsed["error_code"] = "LEGACY_SCHEMA"
        parsed["legacy"] = True
        return parsed

    parsed["schema_version"] = schema_version
    if schema_version != SWIPL_OUT_SCHEMA_VERSION:
        parsed["error_code"] = "SCHEMA_VERSION_UNSUPPORTED"
        return parsed

    missing = sorted(k for k in SWIPL_REQUIRED_KEYS if k not in obj)
    if missing:
        parsed["error_code"] = "SCHEMA_MISSING_REQUIRED_KEYS"
        parsed["validation_error"] = f"missing keys: {', '.join(missing)}"
        return parsed

    if not isinstance(obj.get("ok"), bool):
        parsed["error_code"] = "SCHEMA_TYPE_MISMATCH"
        parsed["validation_error"] = "ok must be bool"
        return parsed

    answer = obj.get("answer")
    if answer is not None and not isinstance(answer, (str, int, float)):
        parsed["error_code"] = "SCHEMA_TYPE_MISMATCH"
        parsed["validation_error"] = "answer must be string/number/null"
        return parsed

    proof = obj.get("proof")
    if proof is not None and not isinstance(proof, str):
        parsed["error_code"] = "SCHEMA_TYPE_MISMATCH"
        parsed["validation_error"] = "proof must be string/null"
        return parsed

    error_code = obj.get("error_code")
    if error_code is not None and not isinstance(error_code, str):
        parsed["error_code"] = "SCHEMA_TYPE_MISMATCH"
        parsed["validation_error"] = "error_code must be string/null"
        return parsed

    parsed["ok"] = bool(obj["ok"])
    parsed["answer"] = answer
    parsed["proof"] = proof
    parsed["error_code"] = error_code
    parsed["raw"] = obj.get("raw") if "raw" in obj else parsed["raw"]
    solution_count = obj.get("solution_count")
    if solution_count is not None:
        if isinstance(solution_count, int) and solution_count >= 0:
            parsed["solution_count"] = solution_count
        else:
            parsed["error_code"] = "SCHEMA_TYPE_MISMATCH"
            parsed["validation_error"] = "solution_count must be non-negative int/null"
            return parsed
    return parsed


def parse_task_type_from_guideline(guideline: str) -> str:
    """
    Expect guideline contains a line like:
      task_type: Yes/No/Partial
    fallback: if not found -> "No"
    """
    m = re.search(r"(?im)^\s*task_type\s*:\s*(yes|no|partial)\s*$", guideline or "")
    return (m.group(1).capitalize() if m else "No")

def build_prolog_gen_prompt(qblock: str, guideline: str) -> str:
    """
    Round C 子步骤：生成可执行 Prolog（只输出 Prolog，最后一行为 query）
    """
    return (
        "Each fact/rule MUST be exactly ONE line and end with a period '.' (no multi-line rules). The final line MUST be the query and also end with '.'. Output Prolog ONLY."
        "You will write SWI-Prolog code for the problem.\n"
        "STRICT RULES:\n"
        "1) Output ONLY Prolog code. No markdown. No explanation.\n"
        "2) Use facts/rules for context. The LAST LINE must be the query (NOT commented).\n"
        "3) Keep variable/constant names consistent.\n"
        "4) Do NOT include symbols like '$' in numbers.\n\n"
        "[Guideline]\n"
        f"{guideline}\n\n"
        "[Problem]\n"
        f"{qblock}\n\n"
        "Prolog code (last line is the query):\n"
    )

def extract_prolog_clauses(code: str) -> list:
    """
    Turn Prolog code into a list of single-line clauses, each ending with '.'.
    Supports multi-line rules (:- ... , ... .) by buffering until '.'.
    """
    code = (code or "").strip()

    # strip ```prolog / ``` fences if any
    code = re.sub(r"^\s*```(?:prolog)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```\s*$", "", code)

    lines = []
    for ln in code.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # drop % comments (keep code before %)
        ln = re.split(r"%", ln, maxsplit=1)[0].strip()
        if ln:
            lines.append(ln)

    clauses = []
    buf = ""
    for ln in lines:
        # accumulate until we see a terminating '.'
        buf = (buf + " " + ln).strip() if buf else ln

        if buf.endswith("."):
            # normalize whitespace in the completed clause
            clauses.append(re.sub(r"\s+", " ", buf))
            buf = ""

    # if something left without '.', force terminate (better than losing it)
    if buf:
        buf = buf if buf.endswith(".") else (buf + ".")
        clauses.append(re.sub(r"\s+", " ", buf))

    return clauses

def _resolve_tmp_root(tmp_dir: Optional[str]) -> Path:
    if tmp_dir:
        return Path(tmp_dir).expanduser().resolve()
    env_dir = os.getenv("TMP_PROLOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path(os.getcwd()) / "tmp_prolog").resolve()

def run_caring_call_swipl(
    dataset_key: str,
    clauses: list,
    max_result: int = 20,
    meta_interpreter: str = "iter_deep_with_proof",
    max_depth: int = 25,
    debug: bool = False,
    keep_tmp: bool = False,
    tmp_dir: Optional[str] = None,
) -> dict:
    caring_root = (Path(__file__).resolve().parent.parent / "caring").resolve()

    name_map = {"gsm8k": "gsm8k", "prontoqa": "prontoqa", "proofwriter": "proofwriter"}
    task_name = name_map.get(dataset_key, dataset_key)

    task_dir = caring_root / "tasks" / task_name
    call_py = task_dir / "call_swipl.py"
    mi_pl   = task_dir / "meta_interpreter.pl"

    if not call_py.exists():
        raise FileNotFoundError(f"call_swipl.py not found: {call_py}")
    if not mi_pl.exists():
        raise FileNotFoundError(f"meta_interpreter.pl not found: {mi_pl}")

    tmp_root = _resolve_tmp_root(tmp_dir)
    run_id = f"{dataset_key}_{os.getpid()}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    run_dir = (tmp_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    assert_path = run_dir / f"{run_id}_assert.pl"
    out_path = run_dir / f"{run_id}_out.json"

    assert_path.write_text("\n".join(clauses) + "\n", encoding="utf-8")

    cmd = [
        sys.executable, str(call_py),
        "--assert_path", str(assert_path),
        "--mi_path", str(mi_pl),
        "--output_path", str(out_path),
        "--max_result", str(max_result),
        "--meta_interpreter", str(meta_interpreter),
        "--max_depth", str(max_depth),
    ]
    if debug:
        cmd.append("--debug")
    if keep_tmp:
        cmd.append("--keep_tmp")
    if tmp_dir:
        cmd.extend(["--tmp_dir", str(tmp_dir)])

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        result = {
            "ok": False,
            "error": "call_swipl timeout",
            "error_code": "SWIPL_TIMEOUT",
            "raw": "",
            "stdout": (e.stdout or "")[-2000:] if hasattr(e, "stdout") else "",
            "stderr": (e.stderr or "")[-2000:] if hasattr(e, "stderr") else "",
            "returncode": -1,
            "cmd": " ".join(cmd),
            "out_path": str(out_path),
            "assert_path": str(assert_path),
            "kept_tmp": debug or keep_tmp,
        }
        if not (debug or keep_tmp):
            try:
                assert_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            if tmp_dir:
                cmd.extend(["--tmp_dir", str(tmp_dir)])
        return result

    raw = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""

    ok = (p.returncode == 0) and (raw != "")
    error_code = None
    if not ok:
        if not out_path.exists():
            err = f"out.json not created: {out_path}\n" + (p.stderr or p.stdout or "")
            error_code = "OUT_JSON_NOT_CREATED"
        elif raw == "":
            err = f"out.json is empty: {out_path}\n" + (p.stderr or p.stdout or "")
            error_code = "OUT_JSON_NOT_CREATED"
        else:
            err = (p.stderr or p.stdout or "unknown error").strip()
            error_code = "SWIPL_CALL_FAILED"
    else:
        err = None
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                error_code = payload.get("error_code")
                if payload.get("ok") is False:
                    ok = False
        except Exception:
            error_code = "SCHEMA_INVALID_JSON"
            ok = False

    result = {
        "ok": ok,
        "error": err,
        "error_code": error_code,
        "raw": raw,
        "stdout": (p.stdout or "")[-2000:],
        "stderr": (p.stderr or "")[-2000:],
        "returncode": p.returncode,
        "cmd": " ".join(cmd),
        "out_path": str(out_path),
        "assert_path": str(assert_path),
        "kept_tmp": debug or keep_tmp,
    }
    if not (debug or keep_tmp):
        try:
            assert_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            run_dir.rmdir()
        except Exception:
            pass
    return result





SOLVE_MODEL = resolve_model(None, purpose="solve")
GUIDE_MODEL = resolve_model(None, purpose="guide")

# 默认每题生成 1 份 guideline；想更像“完整版本”（多采样+合并）可改成 3
N_GUIDE_CANDIDATES = int(os.getenv("N_GUIDE_CANDIDATES", "1"))

# 可选：两轮之间 sleep，避免触发限流
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.5"))
PROLOG_MAX_RESULT = 20
PROLOG_META_INTERPRETER = "iter_deep_with_proof"
PROLOG_MAX_DEPTH = 25
SOLVE_TEMPERATURE = 0.2
GUIDE_TEMPERATURE = 0.7
MAX_RETRIES = 3

def postprocess_pred(dataset_key: str, pred: str) -> str:
    pred = (pred or "").strip()

    if dataset_key == "mmlu":
        c = extract_mmlu_choice(pred)
        return c if c else pred[:1].upper()

    if dataset_key == "sqa":
        w = pred.lower().split()[0] if pred.split() else ""
        return "Yes" if w == "yes" else ("No" if w == "no" else pred)

    if dataset_key == "date":
        d = normalize_date_mmddyyyy(pred)
        return d if d else pred

    if dataset_key == "gsm8k":
        return postprocess_gsm8k_pred(pred)

    # free text: 取第一行更干净
    return pred.splitlines()[0].strip() if pred else pred


def build_cot_draft_prompt(qblock: str, format_rule: str) -> str:
    return (
        f"{qblock}\n\n"
        "Think step by step, then give your final answer.\n"
        f"{format_rule}\n"
        "Final answer:"
    )


def add_message(role: str, content: str, history: list):
    history.append({"role": role, "content": content})

def ai_request(
    history: list,
    model: str,
    t: float = 0.2,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    mock_llm: bool = False,
    mock_profile: Optional[str] = None,
    dataset_key: Optional[str] = None,
    prompt_type: Optional[str] = None,
) -> str:
    return chat_complete(
        history,
        model=model,
        temperature=t,
        max_retries=max_retries,
        timeout=timeout,
        mock_llm=mock_llm,
        mock_profile=mock_profile,
        dataset_key=dataset_key,
        prompt_type=prompt_type,
    )

def build_mock_swipl_output(dataset_key: str, llm_candidate_norm: str) -> dict:
    if dataset_key == "gsm8k":
        digits = re.findall(r"-?\d+", str(llm_candidate_norm or ""))
        if digits:
            answer = str(int(digits[-1]) + 1)
        else:
            answer = "1"
    else:
        answer = llm_candidate_norm or ("True" if dataset_key in ("prontoqa", "proofwriter") else "0")
    raw_payload = json.dumps({
        "schema_version": SWIPL_OUT_SCHEMA_VERSION,
        "ok": True,
        "answer": answer,
        "proof": "mock_proof",
        "error_code": None,
        "raw": "",
    })
    return {
        "ok": True,
        "error": None,
        "error_code": None,
        "raw": raw_payload,
        "stdout": "",
        "stderr": "",
        "returncode": 0,
        "cmd": "mock_swipl",
    }



# ======================
# Dataset helpers
# ======================
#造题面与输出格式约束
def build_question_block(dataset_key: str, data: dict) -> Tuple[str, str]:
    """
    Returns:
      qblock: question block (includes options for MMLU)
      format_rule: strict output rule for Round2
    """
    question = data.get("question", "")
    context = data.get("context", "")

    if dataset_key == "mmlu":
        choice = data["choices"]  # {"A": "...", "B": "...", ...}
        options = "\n".join([f"{k}: {v}" for k, v in sorted(choice.items())])
        qblock = f"Question: {question}\n{options}"
        format_rule = "Output ONLY one letter: A/B/C/D. No other text."
        return qblock, format_rule

    if dataset_key == "sqa":
        qblock = f"Question: {question}"
        format_rule = "Output ONLY: Yes or No. No other text."
        return qblock, format_rule

    if dataset_key == "date":
        qblock = f"Question: {question}"
        format_rule = "Output ONLY the date in MM/DD/YYYY format. No other text."
        return qblock, format_rule

    if dataset_key in ("prontoqa", "proofwriter"):
        if context and question:
            qblock = f"Context: {context}\nQuestion: {question}"
        else:
            qblock = question or context
        format_rule = "Output ONLY the final answer. Do NOT add explanation."
        return qblock, format_rule

    # clutrr / other free text
    qblock = f"{question}"
    format_rule = "Output ONLY the final answer. Do NOT add explanation."
    return qblock, format_rule


# ======================
# Round 1: Guideline Generator (Self-Guide)
# ======================
#给 Round1 生成 guideline 的 prompt 模板。
# def build_guideline_prompt(dataset_key: str, qblock: str, format_rule: str) -> str:
#     """
#     MUST NOT solve the question. Only produce a checklist/guideline.
#     """
#     return (
#         "You are a GuidelineGenerator for solving the given task.\n"
#         "Your goal: write a short, actionable guideline/checklist to help solve this question.\n\n"
#         "Hard rules:\n"
#         "1) DO NOT solve the question.\n"
#         "2) DO NOT output any final answer.\n"
#         "3) Provide 6-10 bullet points.\n"
#         "4) Include verification steps and common pitfalls.\n"
#         f"5) Enforce output format constraint for the solver: {format_rule}\n\n"
#         f"{qblock}\n\n"
#         "Guideline (bullets only):"
#     )
def build_guideline_prompt(dataset_key: str, qblock: str, format_rule: str) -> str:
    """
    Output MUST be YAML with 4 sections:
      task_type: Yes/No/Partial
      schema: ...
      query_goal: ...
      fallback: ...
    """
    return (
        "You are a GuidelineGenerator.\n"
        "You MUST NOT solve the question.\n\n"
        "Output YAML with EXACT keys:\n"
        "task_type: Yes/No/Partial  # whether to use Prolog\n"
        "schema: |                 # naming, predicates, negation/unknown conventions\n"
        "query_goal: |             # what the Prolog query should prove/return\n"
        "fallback: |               # what to do on parse error/timeout/multiple answers\n\n"
        "Rules:\n"
        "1) DO NOT output any final answer.\n"
        "2) schema MUST mention: constants naming, predicate set, negation/Unknown convention.\n"
        "3) query_goal MUST be specific (what to prove/return).\n"
        f"4) solver output constraint: {format_rule}\n\n"
        f"{qblock}\n\n"
        "YAML:"
    )



def consolidate_guidelines_prompt(guides: List[str], format_rule: str) -> str:
    return (
        "You are a GuidelineEditor.\n"
        "Merge multiple guideline candidates into ONE best guideline.\n"
        "Rules:\n"
        "1) Remove duplicates and contradictions.\n"
        "2) Keep it 6-10 bullets.\n"
        "3) DO NOT solve the question or output any final answer.\n"
        f"4) Must enforce: {format_rule}\n\n"
        "Candidates:\n"
        + "\n\n".join([f"[Candidate {i+1}]\n{g}" for i, g in enumerate(guides)])
        + "\n\nFinal merged guideline (bullets only):"
    )


def postprocess_guideline(s: str) -> str:
    s = (s or "").strip()
    # 防止模型偷偷给答案：粗暴去除类似“Answer:”行（可按需增强）
    s = re.sub(r"(?im)^\s*(answer|choice)\s*:\s*.*$", "", s).strip()
    return s if s else "- Follow the problem requirements strictly.\n- Double-check your work."


#核心：生成 guideline
def generate_guideline_from_prompt(
    g_prompt: str,
    format_rule: str,
    dataset_key: str,
    mock_llm: bool = False,
    mock_profile: Optional[str] = None,
) -> str:
    candidates = []
    for _ in range(max(1, N_GUIDE_CANDIDATES)):
        h = [{"role": "user", "content": g_prompt}]
        g = ai_request(
            h,
            model=GUIDE_MODEL,
            t=GUIDE_TEMPERATURE,
            mock_llm=mock_llm,
            mock_profile=mock_profile,
            dataset_key=dataset_key,
            prompt_type="guideline",
        )  # guideline 适当高温度
        candidates.append(postprocess_guideline(g))
        time.sleep(0.2)

    if len(candidates) == 1:
        return candidates[0]
    # 新增：如果 guideline 是 YAML 结构（含 task_type/schema/query_goal/fallback），不要 merge
    if re.search(r"(?im)^\s*task_type\s*:\s*(yes|no|partial)\s*$", candidates[0]):
        return candidates[0]

    merge_p = consolidate_guidelines_prompt(candidates, format_rule)
    merge_h = [{"role": "user", "content": merge_p}]
    merged = ai_request(
        merge_h,
        model=GUIDE_MODEL,
        t=0.2,
        mock_llm=mock_llm,
        mock_profile=mock_profile,
        dataset_key=dataset_key,
        prompt_type="guideline_merge",
    )
    return postprocess_guideline(merged)



# ======================
# Round 2: Solve with guideline
# ======================
def build_solve_prompt(dataset_key, qblock, guideline, format_rule, method_key, draft_answer: str = "") -> str:
    guide_block = f"Guideline/checklist:\n{guideline}\n\n"
    draft_block = f"Draft answer (may be wrong):\n{draft_answer}\n\n" if draft_answer else ""

    if method_key == "cot_selfguide":
        reasoning_instr = "Think step by step internally, then give the final answer.\n"
    elif method_key == "sd_selfguide":
        reasoning_instr = ""
    else:
        raise ValueError("Invalid method.")

    return (
        f"{guide_block}"
        f"{draft_block}"
        f"{qblock}\n\n"
        f"{reasoning_instr}"
        "The draft may be entirely wrong; do NOT copy it.\n"
        "Solve the problem independently first, then use the guideline to check and revise.\n"
        "IMPORTANT: Do NOT output explanation.\n"
        "After solving, compare your solution with the Draft and identify any mistakes in the Draft internally.\n"
        "Then revise your final answer if needed.\n"

        f"{format_rule}\n"
        "Final answer:"
    )


# ======================
# Optional scoring (same as你之前)
# ======================
def normalize_date_mmddyyyy(s: str) -> Optional[str]:
    s = str(s)
    m = re.search(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b", s)
    if not m:
        return None
    mm = int(m.group(1))
    dd = int(m.group(2))
    yyyy = m.group(3)
    return f"{mm:02d}/{dd:02d}/{yyyy}"


def extract_mmlu_choice(s: str) -> str:
    s = str(s)
    matches = re.findall(r"\b([A-Da-d])\b", s)
    return matches[-1].upper() if matches else ""

def extract_gsm8k_final_number(s: str) -> str:
    return extract_gsm8k_number(s)

def judge_correctness(dataset_key: str, gold: str, pred: str) -> str:
    return judge_dataset_correctness(dataset_key, gold, pred)


# ======================
# Main: Self-Guide (Round1 guideline -> Round2 solve)
# ======================
def self_guide_run(
    dataset: str,
    method: str,
    start_index: int = 0,
    num_samples: int = 1,
    force_task_type: Optional[str] = None,
    data_path: Optional[str] = None,
    log_dir_override: Optional[str] = None,
    mock_llm: bool = False,
    mock_profile: Optional[str] = None,
    mock_prolog: bool = False,
    prolog_role: str = "off",
    meta_interpreter: str = PROLOG_META_INTERPRETER,
    max_depth: int = PROLOG_MAX_DEPTH,
    prolog_max_result: int = PROLOG_MAX_RESULT,
    debug: bool = False,
    keep_tmp: bool = False,
    tmp_dir: Optional[str] = None,
):
    validate_openai_api_key(mock_llm)


    dataset_key = dataset.lower()
    method_key = method.lower()

    try:
        data_path = resolve_data_path(dataset_key, data_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    log_dir = log_dir_override or f"log/{method_key}/{dataset_key}"
    os.makedirs(log_dir, exist_ok=True)
    if mock_llm:
        for log_path in glob(os.path.join(log_dir, "*.json")):
            os.remove(log_path)

    samples = load_dataset(data_path, dataset_key)
    if not samples:
        print(
            f"Dataset file is empty or invalid after mapping: {data_path}.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Self-Guide running: dataset={dataset_key}, method={method_key}, start_index={start_index}")
    end = min(len(samples), start_index + num_samples)
    for i, line in enumerate(samples[start_index:end], start=start_index):
        data = line
        sample_id = data.get("id", i)
        gold = data.get("answer", "")

        qblock, format_rule = build_question_block(dataset_key, data)

        base_run_config = {
            "dataset": dataset_key,
            "method": method_key,
            "solve_model": SOLVE_MODEL,
            "guide_model": GUIDE_MODEL,
            "solve_temperature": SOLVE_TEMPERATURE,
            "guide_temperature": GUIDE_TEMPERATURE,
            "max_retries": MAX_RETRIES,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "prompt_versions": PROMPT_VERSIONS,
            "prolog_contract_version": PROLOG_CONTRACT_VERSION,
            "swipl_schema_version": SWIPL_OUT_SCHEMA_VERSION,
            "meta_interpreter": meta_interpreter,
            "max_depth": max_depth,
            "prolog_max_result": prolog_max_result,
            "force_task_type": force_task_type,
            "mock_profile": mock_profile,
            "mock_llm": mock_llm,
            "mock_prolog": mock_prolog,
            "prolog_role": prolog_role,
        }
        config_hash = build_config_hash(base_run_config)
        sample_error_code = "OK"

        ## ---------- Round 0: CoT draft (A) ----------
        contract_round_a = validate_prompt_inputs("round_a_draft", {"qblock": qblock, "format_rule": format_rule})
        if not contract_round_a.ok:
            sample_error_code = normalize_error_code(contract_round_a.error_code)
        cot_prompt = build_cot_draft_prompt(qblock, format_rule)
        draft_history = [{"role": "user", "content": cot_prompt}]
        draft_raw = ai_request(
            draft_history,
            model=SOLVE_MODEL,
            t=SOLVE_TEMPERATURE,
            mock_llm=mock_llm,
            mock_profile=mock_profile,
            dataset_key=dataset_key,
            prompt_type="draft",
        )
        time.sleep(SLEEP_SEC)

        # 截断草稿，避免太长（更稳、更省 token）
        draft_raw_short = "\n".join(draft_raw.splitlines()[-20:]).strip()
        draft = postprocess_pred(dataset_key, draft_raw)  # 统计/日志用

        # ---------- Round 1: generate guideline (B) ----------
        contract_round_b = validate_prompt_inputs("round_b_guideline", {"dataset_key": dataset_key, "qblock": qblock,
                                                                        "format_rule": format_rule})
        if sample_error_code == "OK" and not contract_round_b.ok:
            sample_error_code = normalize_error_code(contract_round_b.error_code)
        g_prompt = build_guideline_prompt(dataset_key, qblock, format_rule)
        guideline = generate_guideline_from_prompt(
            g_prompt,
            format_rule,
            dataset_key=dataset_key,
            mock_llm=mock_llm,
            mock_profile=mock_profile,
        )
        time.sleep(SLEEP_SEC)

        # ---------- Round 2: solve with guideline + draft (C) ----------
        role_mode = (prolog_role or "off").strip().lower()
        contract_round_c = validate_prompt_inputs(
            "round_c_solve",
            {
                "dataset_key": dataset_key,
                "qblock": qblock,
                "guideline": guideline,
                "format_rule": format_rule,
                "method_key": method_key,
            },
        )
        if sample_error_code == "OK" and not contract_round_c.ok:
            sample_error_code = normalize_error_code(contract_round_c.error_code)
        solve_prompt = build_solve_prompt(
            dataset_key, qblock, guideline, format_rule, method_key, draft_answer=draft_raw_short
        )

        history = []
        add_message("user", solve_prompt, history)
        pred_raw = ai_request(
            history,
            model=SOLVE_MODEL,
            t=SOLVE_TEMPERATURE,
            mock_llm=mock_llm,
            mock_profile=mock_profile,
            dataset_key=dataset_key,
            prompt_type="solve",
        )
        pred = postprocess_pred(dataset_key, pred_raw)
        output_contract = validate_output_format(dataset_key, pred)
        if sample_error_code == "OK" and not output_contract.ok:
            sample_error_code = normalize_error_code(output_contract.error_code)


        # ======== Round C': optional Prolog verification/execution (CaRing) ========
        route = "llm_only"
        prolog_pack = {
            "enabled": False,
            "task_type": "No",
            "prolog_max_result": prolog_max_result,
            "meta_interpreter": meta_interpreter,
            "max_depth": max_depth,
            "role_mode": role_mode,
        }
        llm_candidate = pred
        llm_candidate_norm = postprocess_pred(dataset_key, llm_candidate)
        final_answer = llm_candidate_norm
        route_reason = None

        if dataset_key in ("gsm8k", "prontoqa", "proofwriter"):
            task_type = parse_task_type_from_guideline(guideline)

            if force_task_type:
                task_type = force_task_type

            prolog_pack["task_type"] = task_type
            mode_allows_prolog = role_mode in ("verifier", "executor")
            prolog_pack["enabled"] = mode_allows_prolog and (task_type in ("Yes", "Partial"))

            if prolog_pack["enabled"]:
                prolog_prompt = build_prolog_gen_prompt(qblock, guideline)
                prolog_history = [{"role": "user", "content": prolog_prompt}]
                prolog_raw = ai_request(
                    prolog_history,
                    model=SOLVE_MODEL,
                    t=SOLVE_TEMPERATURE,
                    mock_llm=mock_llm,
                    mock_profile=mock_profile,
                    dataset_key=dataset_key,
                    prompt_type="prolog",
                )
                prolog_pack["prolog_raw"] = prolog_raw

                try:
                    clauses = extract_prolog_clauses(prolog_raw)
                    prolog_pack["clauses"] = clauses
                    prolog_contract = validate_prolog_contract(clauses)
                    prolog_pack["contract"] = {
                        "version": PROLOG_CONTRACT_VERSION,
                        "ok": prolog_contract.ok,
                        "error_code": prolog_contract.error_code,
                        "message": prolog_contract.message,
                    }
                    if sample_error_code == "OK" and not prolog_contract.ok:
                        sample_error_code = normalize_error_code(prolog_contract.error_code)

                    if mock_prolog:
                        swipl_out = build_mock_swipl_output(dataset_key, llm_candidate_norm)
                    else:
                        swipl_out = run_caring_call_swipl(
                            dataset_key,
                            clauses,
                            max_result=prolog_max_result,
                            meta_interpreter=meta_interpreter,
                            max_depth=max_depth,
                            debug=debug,
                            keep_tmp=keep_tmp,
                            tmp_dir=tmp_dir,
                        )

                    if "raw" not in swipl_out:
                        swipl_out["raw"] = ""

                    prolog_pack["swipl"] = swipl_out
                    prolog_pack["clauses_count"] = len(clauses)

                    swipl_contract = parse_caring_swipl_answer(swipl_out.get("raw"))
                    prolog_pack["swipl_contract"] = swipl_contract

                    prolog_answer_raw = swipl_contract.get("answer")
                    prolog_answer = str(prolog_answer_raw) if prolog_answer_raw is not None else None
                    prolog_answer_norm = (
                        postprocess_pred(dataset_key, prolog_answer)
                        if prolog_answer is not None
                        else None
                    )
                    prolog_pack["proof"] = swipl_contract.get("proof")
                    prolog_pack["error_code"] = swipl_contract.get("error_code")
                    raw_meta = swipl_contract.get("raw") if isinstance(swipl_contract.get("raw"), dict) else {}
                    solution_count = swipl_contract.get("solution_count")
                    if solution_count is None and isinstance(raw_meta, dict):
                        raw_count = raw_meta.get("results_count")
                        if isinstance(raw_count, int) and raw_count >= 0:
                            solution_count = raw_count
                    prolog_pack["solution_count"] = solution_count
                    prolog_pack["prolog_answer_raw"] = prolog_answer
                    prolog_pack["prolog_answer_norm"] = prolog_answer_norm

                    if sample_error_code == "OK" and prolog_pack["error_code"]:
                        sample_error_code = normalize_error_code(prolog_pack["error_code"])

                    if meta_interpreter in ("with_proof", "iter_deep_with_proof") and swipl_contract.get("ok"):
                        if not prolog_pack["proof"]:
                            prolog_pack["proof"] = "NO_PROOF_RETURNED"

                    prolog_schema_ok = (
                            swipl_contract.get("schema_version") == SWIPL_OUT_SCHEMA_VERSION
                            and swipl_contract.get("validation_error") is None
                            and swipl_contract.get("legacy") is False
                    )
                    prolog_ok = bool(swipl_contract.get("ok") and prolog_answer_norm is not None and prolog_schema_ok)

                    if prolog_ok:
                        if role_mode == "executor":
                            route = "executor"
                            final_answer = prolog_answer_norm
                            route_reason = "Executor mode: final answer forced from Prolog."
                        else:
                            route = "verifier"
                            if normalize_answer(llm_candidate_norm) != normalize_answer(prolog_answer_norm):
                                final_answer = prolog_answer_norm
                                llm_prolog_conflict = normalize_answer(llm_candidate_norm) != normalize_answer(
                                    prolog_answer_norm)
                                if llm_prolog_conflict:
                                    answer_nonempty = is_nonempty_answer(prolog_answer_norm)
                                    proof_nonempty = is_nonempty_proof(prolog_pack.get("proof"))
                                    allow_override = bool(
                                        prolog_ok
                                        and answer_nonempty
                                        and (proof_nonempty or solution_count == 1)
                                    )
                                    prolog_pack["answer_nonempty"] = answer_nonempty
                                    prolog_pack["proof_nonempty"] = proof_nonempty
                                    prolog_pack["verifier_allow_override"] = allow_override
                                    if allow_override:
                                        final_answer = prolog_answer_norm
                                        route_reason = "Verifier override accepted by trust gate."
                                        prolog_pack["verifier_gate"] = "override"
                                    else:
                                        gate = (
                                            "multi_solution_conflict"
                                            if (not proof_nonempty and isinstance(solution_count,
                                                                                  int) and solution_count > 1)
                                            else "prolog_inconclusive"
                                        )
                                        route_reason = f"Verifier kept LLM final: {gate}."
                                        prolog_pack["verifier_gate"] = gate
                                else:
                                    route_reason = "Verifier mode: no conflict between LLM and Prolog answers."
                            else:
                                final_answer = llm_candidate_norm
                                route_reason = "Verifier mode: LLM matches Prolog; keep LLM final."
                    else:
                        route = "llm_only"
                        final_answer = llm_candidate_norm

                        error_hint = ""
                        if prolog_pack["error_code"]:
                            error_hint = str(prolog_pack["error_code"])
                        elif swipl_out.get("error"):
                            error_hint = str(swipl_out.get("error"))
                        elif swipl_out.get("stderr"):
                            error_hint = str(swipl_out.get("stderr"))
                        error_hint = error_hint[:200]

                        route_reason = "Prolog failed, schema invalid, or no answer parsed."
                        if error_hint:
                            route_reason = f"{route_reason} {error_hint}"
                        prolog_pack["fallback_reason"] = route_reason

                except Exception as e:
                    prolog_pack["swipl"] = {
                        "ok": False,
                        "error": str(e),
                        "error_code": "SELF_GUIDE_SWIPL_EXCEPTION",
                        "raw": "",
                        "cmd": None,
                        "returncode": None,
                    }
                    route = "llm_only"
                    final_answer = llm_candidate_norm
                    if sample_error_code == "OK":
                        sample_error_code = "SELF_GUIDE_SWIPL_EXCEPTION"
                    route_reason = f"Prolog parse/execute exception: {e}"
                    prolog_pack["fallback_reason"] = route_reason
                    prolog_pack["clauses_count"] = None
            else:
                if role_mode == "off":
                    route_reason = "Prolog role is off."
                else:
                    route_reason = "Prolog disabled by task_type."
                prolog_pack["fallback_reason"] = route_reason
                prolog_pack["clauses_count"] = None
        else:
            route_reason = "Dataset not supported for Prolog."
            prolog_pack["fallback_reason"] = route_reason
            prolog_pack["clauses_count"] = None

        add_message("assistant", final_answer, history)
        time.sleep(SLEEP_SEC)

        correctness = judge_correctness(dataset_key, gold, final_answer)

        has_gold = bool(str(gold).strip())
        draft_correct = judge_correctness(dataset_key, gold, draft) if has_gold else None
        final_correct = judge_correctness(dataset_key, gold, final_answer) if has_gold else None
        draft_final_same = normalize_answer(draft) == normalize_answer(final_answer)
        prolog_answer = prolog_pack.get("prolog_answer_norm") if isinstance(prolog_pack, dict) else None
        prolog_ok = bool(prolog_pack.get("swipl_contract", {}).get("ok")) if isinstance(
            prolog_pack.get("swipl_contract"), dict) else False
        prolog_used = bool(route in ("verifier", "executor"))
        prolog_overruled = bool(route in ("verifier", "executor") and prolog_answer is not None and normalize_answer(
            llm_candidate_norm) != normalize_answer(prolog_answer))
        final_modified_by_prolog = bool(
            route in ("verifier", "executor") and prolog_answer is not None and normalize_answer(
                final_answer) == normalize_answer(prolog_answer))
        draft_to_final_change_type = compute_change_type(
            draft=draft,
            final=final_answer,
            prolog_used=prolog_used,
            prolog_overruled=prolog_overruled,
        )

        draft_prolog_conflict = None
        if prolog_answer is not None:
            draft_prolog_conflict = normalize_answer(draft) != normalize_answer(prolog_answer)


        # ---------- save log ----------
        # ---------- save log ----------
        log_path = os.path.join(log_dir, f"{dataset_key}_{i}.json")

        payload = {
            "id": sample_id,
            "dataset": dataset_key,
            "method": method_key,
            "models": {"guide_model": GUIDE_MODEL, "solve_model": SOLVE_MODEL},
            "format_rule": format_rule,
            "run_config": {
                "dataset": dataset_key,
                "method": method_key,
                "start_index": start_index,
                "num_samples": num_samples,
                "force_task_type": force_task_type,
                "solve_model": SOLVE_MODEL,
                "guide_model": GUIDE_MODEL,
                "solve_temperature": SOLVE_TEMPERATURE,
                "guide_temperature": GUIDE_TEMPERATURE,
                "max_retries": MAX_RETRIES,
                "prompt_contract_version": PROMPT_CONTRACT_VERSION,
                "prompt_versions": PROMPT_VERSIONS,
                "prolog_contract_version": PROLOG_CONTRACT_VERSION,
                "swipl_schema_version": SWIPL_OUT_SCHEMA_VERSION,
                "mock_profile": mock_profile,
                "mock_llm": mock_llm,
                "mock_prolog": mock_prolog,
                "prolog_role": prolog_role,
                "prolog_max_result": prolog_max_result,
                "prolog_enabled": prolog_pack.get("enabled", False),
                "prolog_task_type": prolog_pack.get("task_type"),
                "meta_interpreter": meta_interpreter,
                "max_depth": max_depth,
                "debug": debug,
                "keep_tmp": keep_tmp,
                "config_hash": config_hash,
            },
            "guideline": guideline,
            "gold": gold,
            "draft_raw": draft_raw,
            "draft": draft,
            "draft_answer": draft,
            "pred_raw": pred_raw,
            "pred": pred,
            "llm_candidate": llm_candidate,
            "llm_candidate_norm": llm_candidate_norm,
            "final_answer": final_answer,
            "draft_final_same": draft_final_same,
            "final_modified_by_prolog": final_modified_by_prolog,
            "prolog_used": prolog_used,
            "prolog_ok": prolog_ok,
            "prolog_answer": prolog_answer,
            "draft_to_final_change_type": draft_to_final_change_type,
            "draft_correct": draft_correct,
            "final_correct": final_correct,
            "prolog_overruled": prolog_overruled,
            "draft_prolog_conflict": draft_prolog_conflict,
            "correctness": correctness,
            "round1_guideline_prompt": g_prompt,
            "round2_solve_prompt": solve_prompt,
            "log": history,
            "route": route,
            "prolog": prolog_pack,
            "route_reason": route_reason,
            "error_code": normalize_error_code(sample_error_code),
            "config_hash": config_hash,
            "contracts": {
                "prompt": {"version": PROMPT_CONTRACT_VERSION, "round_versions": PROMPT_VERSIONS},
                "prolog": {"version": PROLOG_CONTRACT_VERSION, "schema_version": SWIPL_OUT_SCHEMA_VERSION},
            },
        }

        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(payload, lf, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    def _main():
        parser = create_base_parser(
            "Self-guide baseline entrypoint",
            dataset_help="gsm8k / prontoqa / proofwriter / mmlu / clutrr / sqa / date",
            method_help="sd_selfguide or cot_selfguide",
            include_num_samples=True,
            include_log_dir=True,
        )
        parser.add_argument(
            "--force_task_type",
            choices=("Yes", "No", "Partial"),
            default=None,
        )
        parser.add_argument("--mock_prolog", action="store_true", help="mock Prolog execution (no SWI-Prolog call)")
        parser.add_argument("--prolog_role", choices=("off", "verifier", "executor"), default="off")
        parser.add_argument("--meta_interpreter", default=PROLOG_META_INTERPRETER)
        parser.add_argument("--max_depth", type=int, default=PROLOG_MAX_DEPTH)
        parser.add_argument("--prolog_max_result", type=int, default=PROLOG_MAX_RESULT)
        parser.add_argument("--debug", action="store_true", help="enable debug logging and keep tmp files")
        parser.add_argument("--keep_tmp", action="store_true", help="keep Prolog temp files")
        parser.add_argument("--tmp_dir", default=None, help="root dir for Prolog temp files")

        args = parser.parse_args()

        self_guide_run(
            args.dataset,
            args.method,
            args.start_index,
            num_samples=args.num_samples,
            force_task_type=args.force_task_type,
            data_path=args.data_path,
            log_dir_override=args.log_dir,
            mock_llm=args.mock_llm,
            mock_profile=args.mock_profile,
            mock_prolog=args.mock_prolog,
            prolog_role=args.prolog_role,
            meta_interpreter=args.meta_interpreter,
            max_depth=args.max_depth,
            prolog_max_result=args.prolog_max_result,
            debug=args.debug,
            keep_tmp=args.keep_tmp,
            tmp_dir=args.tmp_dir,
        )

    run_main(_main)
