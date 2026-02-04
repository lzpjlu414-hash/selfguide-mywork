# self_guide.py
import json
import time
import os
import re
from glob import glob
from typing import Optional, Tuple, List

from openai import OpenAI

from tqdm import tqdm
from argparse import ArgumentParser

from pathlib import Path
import subprocess

from src.utils.dataset_io import load_jsonl
from src.utils.scoring import (
    extract_gsm8k_final_number as extract_gsm8k_number,
    postprocess_pred as postprocess_gsm8k_pred,
    judge_correctness as judge_gsm8k_correctness,
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

def parse_caring_swipl_answer(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    尝试从 CaRing call_swipl 输出(out.json内容)里提取:
      - ans: Prolog 结论/答案（字符串）
      - proof: proof/trace（字符串，可为空）
    注意：不同 CaRing 版本 out.json 字段名可能不同，所以这里做“容错式”提取。
    你跑一条样本后，看 prolog.swipl.raw 实际 keys，再把 KEY_CANDIDATES 精简/改准即可。
    """
    raw = (raw or "").strip()
    if not raw:
        return None, None

    # 1) 先尝试当 JSON 解析
    try:
        obj = json.loads(raw)
    except Exception:
        # 不是 JSON：就把 raw 当答案（proof 空）
        return raw.splitlines()[0].strip(), None

    if isinstance(obj, list):
        obj = obj[0] if obj else {}

    # 2) 从 JSON 里找答案字段
    KEY_CANDIDATES_ANS = ["answer", "pred", "result", "final_answer", "output", "answers"]
    ans = None
    for k in KEY_CANDIDATES_ANS:
        if k in obj and obj[k] is not None:
            v = obj[k]
            if isinstance(v, list):
                ans = str(v[0]) if v else None
            else:
                ans = str(v)
            break

    # 3) 找 proof/trace 字段
    KEY_CANDIDATES_PROOF = ["proof", "trace", "Proof", "proofs", "traces"]
    proof = None
    for k in KEY_CANDIDATES_PROOF:
        if k in obj and obj[k] is not None:
            v = obj[k]
            if isinstance(v, list):
                if not v:
                    proof = None
                else:
                    proof = "\n".join(str(item) for item in v if str(item).strip())
            else:
                proof = str(v)
            break

    return (ans.strip() if ans else None), (proof.strip() if proof else None)


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



import subprocess
import sys


def run_caring_call_swipl(
    dataset_key: str,
    clauses: list,
    max_result: int = 20,
    meta_interpreter: str = "iter_deep_with_proof",
    max_depth: int = 25,
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

    tmp_dir = (Path(os.getcwd()) / "tmp_prolog").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    assert_path = tmp_dir / f"{dataset_key}_assert.pl"
    out_path    = tmp_dir / f"{dataset_key}_out.json"

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

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": "call_swipl timeout",
            "raw": "",
            "stdout": (e.stdout or "")[-2000:] if hasattr(e, "stdout") else "",
            "stderr": (e.stderr or "")[-2000:] if hasattr(e, "stderr") else "",
            "returncode": -1,
            "cmd": " ".join(cmd),
            "out_path": str(out_path),
        }

    raw = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""

    ok = (p.returncode == 0) and (raw != "")
    if not ok:
        if not out_path.exists():
            err = f"out.json not created: {out_path}\n" + (p.stderr or p.stdout or "")
        elif raw == "":
            err = f"out.json is empty: {out_path}\n" + (p.stderr or p.stdout or "")
        else:
            err = (p.stderr or p.stdout or "unknown error").strip()
    else:
        err = None

    return {
        "ok": ok,
        "error": err,
        "raw": raw,
        "stdout": (p.stdout or "")[-2000:],
        "stderr": (p.stderr or "")[-2000:],
        "returncode": p.returncode,
        "cmd": " ".join(cmd),
        "out_path": str(out_path),
    }





# ======================
# OpenAI config
# ======================
_CLIENT = None


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", None) or None,
        )
    return _CLIENT


SOLVE_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-1106")
GUIDE_MODEL = os.getenv("OPENAI_GUIDE_MODEL", SOLVE_MODEL)

# 默认每题生成 1 份 guideline；想更像“完整版本”（多采样+合并）可改成 3
N_GUIDE_CANDIDATES = int(os.getenv("N_GUIDE_CANDIDATES", "1"))

# 可选：两轮之间 sleep，避免触发限流
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.5"))
PROLOG_MAX_RESULT = 20
PROLOG_META_INTERPRETER = "iter_deep_with_proof"
PROLOG_MAX_DEPTH = 25

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

def _mock_ai_answer(prompt: str) -> str:
    # guideline YAML
    if "Output YAML with EXACT keys" in prompt and "YAML:" in prompt:
        return (
            "task_type: Yes\n"
            "schema: |\n"
            "  - predicates: ans/1\n"
            "  - constants: use integers only\n"
            "  - negation/unknown: not used\n"
            "query_goal: |\n"
            "  - return the final numeric answer as Ans\n"
            "fallback: |\n"
            "  - if Prolog fails, use the LLM numeric result\n"
        )
    # prolog generation
    if "Prolog code (last line is the query)" in prompt:
        return "ans(810).\nans(Ans).\n"
    # draft / solve
    return "810"

def ai_request(history: list, model: str, t: float = 0.2, max_retries: int = 3, mock_llm: bool = False) -> str:
    if mock_llm:
        prompt = history[-1]["content"] if history else ""
        return _mock_ai_answer(prompt)

    """Retry wrapper for client.chat.completions.create (openai>=1.x)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=history,
                temperature=t,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"Request failed after {max_retries} retries: {last_err}")



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

    if dataset_key == "mmlu":
        choice = data["choices"]  # {"A": "...", "B": "...", ...}
        options = "\n".join([f"{k}: {v}" for k, v in choice.items()])
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
def generate_guideline_from_prompt(g_prompt: str, format_rule: str, mock_llm: bool = False) -> str:
    candidates = []
    for _ in range(max(1, N_GUIDE_CANDIDATES)):
        h = [{"role": "user", "content": g_prompt}]
        g = ai_request(h, model=GUIDE_MODEL, t=0.7, mock_llm=mock_llm)  # guideline 适当高温度
        candidates.append(postprocess_guideline(g))
        time.sleep(0.2)

    if len(candidates) == 1:
        return candidates[0]
    # 新增：如果 guideline 是 YAML 结构（含 task_type/schema/query_goal/fallback），不要 merge
    if re.search(r"(?im)^\s*task_type\s*:\s*(yes|no|partial)\s*$", candidates[0]):
        return candidates[0]

    merge_p = consolidate_guidelines_prompt(candidates, format_rule)
    merge_h = [{"role": "user", "content": merge_p}]
    merged = ai_request(merge_h, model=GUIDE_MODEL, t=0.2, mock_llm=mock_llm)
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
    gold_s = str(gold).strip().lower()
    pred_s = str(pred).strip()

    if dataset_key == "mmlu":
        gold_choice = gold_s[:1].upper()
        pred_choice = extract_mmlu_choice(pred_s)
        return "True" if pred_choice == gold_choice else "False"

    if dataset_key == "sqa":
        g = gold_s
        first = pred_s.strip().lower().split()[0] if pred_s.strip().split() else ""
        return "True" if first in ("yes", "no") and first == g else "False"

    if dataset_key == "date":
        gold_norm = normalize_date_mmddyyyy(gold_s) or gold_s
        pred_norm = normalize_date_mmddyyyy(pred_s)
        return "True" if (pred_norm is not None and pred_norm.lower() == gold_norm.lower()) else "False"
        # ✅ 新增：gsm8k 判分只比最终数字
    if dataset_key == "gsm8k":
        return judge_gsm8k_correctness(gold, pred)

    return "True" if gold_s in pred_s.lower() else "False"


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
    meta_interpreter: str = PROLOG_META_INTERPRETER,
    max_depth: int = PROLOG_MAX_DEPTH,
    prolog_max_result: int = PROLOG_MAX_RESULT,
):
    if (not mock_llm) and (not os.getenv("OPENAI_API_KEY", "")):
       raise ValueError("OPENAI_API_KEY is empty. Please set env OPENAI_API_KEY.")


    dataset_key = dataset.lower()
    method_key = method.lower()

    # data_path = f"log_guideline/{dataset}.jsonl"
    data_path = data_path or f"log/{dataset_key}.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    log_dir = log_dir_override or f"log/{method_key}/{dataset_key}"
    os.makedirs(log_dir, exist_ok=True)
    if mock_llm:
        for log_path in glob(os.path.join(log_dir, "*.json")):
            os.remove(log_path)

    samples = load_jsonl(data_path)
    if not samples:
        raise ValueError(f"Dataset file is empty or invalid: {data_path}")

    print(f"Self-Guide running: dataset={dataset_key}, method={method_key}, start_index={start_index}")
    end = min(len(samples), start_index + num_samples)
    for i, line in tqdm(
            enumerate(samples[start_index:end], start=start_index),
            total=end - start_index,
    ):
        data = line
        sample_id = data.get("id", i)
        gold = data.get("answer", "")

        qblock, format_rule = build_question_block(dataset_key, data)

        ## ---------- Round 0: CoT draft (A) ----------
        cot_prompt = build_cot_draft_prompt(qblock, format_rule)
        draft_history = [{"role": "user", "content": cot_prompt}]
        draft_raw = ai_request(draft_history, model=SOLVE_MODEL, t=0.2, mock_llm=mock_llm)
        time.sleep(SLEEP_SEC)

        # 截断草稿，避免太长（更稳、更省 token）
        draft_raw_short = "\n".join(draft_raw.splitlines()[-20:]).strip()
        draft = postprocess_pred(dataset_key, draft_raw)  # 统计/日志用

        # ---------- Round 1: generate guideline (B) ----------
        g_prompt = build_guideline_prompt(dataset_key, qblock, format_rule)
        guideline = generate_guideline_from_prompt(g_prompt, format_rule, mock_llm=mock_llm)
        time.sleep(SLEEP_SEC)

        # ---------- Round 2: solve with guideline + draft (C) ----------
        solve_prompt = build_solve_prompt(
            dataset_key, qblock, guideline, format_rule, method_key, draft_answer=draft_raw_short
        )

        history = []
        add_message("user", solve_prompt, history)
        pred_raw = ai_request(history, model=SOLVE_MODEL, t=0.2, mock_llm=mock_llm)
        pred = postprocess_pred(dataset_key, pred_raw)


        # ======== Round C': optional Prolog verification/execution (CaRing) ========
        route = "llm_only"
        prolog_pack = {
            "enabled": False,
            "task_type": "No",
            "prolog_max_result": prolog_max_result,
            "meta_interpreter": meta_interpreter,
            "max_depth": max_depth,
        }  # 默认不启用
        llm_candidate = pred
        llm_candidate_norm = postprocess_pred(dataset_key, llm_candidate)
        final_answer = llm_candidate_norm
        route_reason = None

        # 只对 CaRing 三任务启用 Prolog（你现在 mmlu/sqa/date/clutrr 就不要碰）
        if dataset_key in ("gsm8k", "prontoqa", "proofwriter"):
            task_type = parse_task_type_from_guideline(guideline)  # 期望返回 Yes/No/Partial

            if force_task_type:
                task_type = force_task_type

            prolog_pack["task_type"] = task_type
            prolog_pack["enabled"] = (task_type in ("Yes", "Partial"))  # ✅ 只有 Yes/Partial 才算启用

            if prolog_pack["enabled"]:
                # (1) 生成 Prolog（严格：只输出 Prolog）
                prolog_prompt = build_prolog_gen_prompt(qblock, guideline)
                prolog_history = [{"role": "user", "content": prolog_prompt}]
                prolog_raw = ai_request(prolog_history, model=SOLVE_MODEL, t=0.2, mock_llm=mock_llm)
                prolog_pack["prolog_raw"] = prolog_raw

                try:
                    clauses = extract_prolog_clauses(prolog_raw)
                    prolog_pack["clauses"] = clauses

                    # (2) 执行 CaRing 的 call_swipl.py
                    swipl_out = run_caring_call_swipl(
                        dataset_key,
                        clauses,
                        max_result=prolog_max_result,
                        meta_interpreter=meta_interpreter,
                        max_depth=max_depth,
                    )

                    # ✅ 强制保证 raw 字段存在（避免你之后解析时抓不到）
                    if "raw" not in swipl_out:
                        swipl_out["raw"] = ""

                    prolog_pack["swipl"] = swipl_out
                    prolog_pack["clauses_count"] = len(clauses)
                    prolog_pack["prolog_max_result"] = prolog_max_result
                    prolog_pack["meta_interpreter"] = meta_interpreter
                    prolog_pack["max_depth"] = max_depth

                    # 解析 Prolog answer + proof
                    prolog_answer, prolog_proof = parse_caring_swipl_answer(swipl_out.get("raw"))
                    prolog_pack["prolog_answer_raw"] = prolog_answer
                    prolog_pack["prolog_answer_norm"] = (
                        postprocess_pred(dataset_key, prolog_answer) if prolog_answer else None
                    )
                    proof_list = []
                    raw_obj = None
                    try:
                        raw_obj = json.loads(swipl_out.get("raw", "") or "")
                    except Exception:
                        raw_obj = None
                    if isinstance(raw_obj, dict) and isinstance(raw_obj.get("proofs"), list):
                        proof_list = [str(p) for p in raw_obj.get("proofs") if str(p).strip()]
                    prolog_pack["proofs"] = proof_list
                    prolog_pack["proof"] = prolog_proof

                    if meta_interpreter in ("with_proof", "iter_deep_with_proof") and swipl_out.get("ok"):
                        if not prolog_pack["proof"]:
                            if proof_list:
                                prolog_pack["proof"] = "\n".join(proof_list)
                            else:
                                prolog_pack["proof"] = "NO_PROOF_RETURNED"

                    if swipl_out.get("ok") and prolog_answer:
                        prolog_answer_norm = prolog_pack["prolog_answer_norm"]
                        if prolog_answer_norm == llm_candidate_norm:
                            route = "executor"
                            final_answer = prolog_answer_norm
                            route_reason = "Prolog answer matches LLM after normalization."
                            prolog_pack["route_reason"] = route_reason
                        else:
                            route = "verifier"
                            final_answer = prolog_answer_norm
                            route_reason = "Prolog answer differs from LLM after normalization."
                            prolog_pack["route_reason"] = route_reason
                            prolog_pack["audit"] = {
                                "llm_candidate_raw": llm_candidate,
                                "llm_candidate_norm": llm_candidate_norm,
                                "prolog_answer_raw": prolog_answer,
                                "prolog_answer_norm": prolog_answer_norm,
                                "reason": "Prolog answer differs from LLM; trust Prolog for correction.",
                            }
                    else:
                        route = "llm_only"
                        final_answer = llm_candidate_norm

                        error_hint = ""
                        if swipl_out.get("error"):
                            error_hint = str(swipl_out.get("error"))
                        elif swipl_out.get("stderr"):
                            error_hint = str(swipl_out.get("stderr"))
                        error_hint = error_hint[:200]

                        route_reason = "Prolog failed or no answer parsed."
                        if error_hint:
                            route_reason = f"{route_reason} {error_hint}"
                        prolog_pack["fallback_reason"] = route_reason

                except Exception as e:
                    prolog_pack["swipl"] = {
                        "ok": False,
                        "error": str(e),
                        "raw": "",
                        "cmd": None,
                        "returncode": None,
                    }
                    route = "llm_only"
                    final_answer = llm_candidate_norm
                    route_reason = f"Prolog parse/execute exception: {e}"
                    prolog_pack["fallback_reason"] = route_reason
                    prolog_pack["clauses_count"] = None
                    prolog_pack["prolog_max_result"] = prolog_max_result
                    prolog_pack["meta_interpreter"] = meta_interpreter
                    prolog_pack["max_depth"] = max_depth
            else:
                route_reason = "Prolog disabled by task_type."
                prolog_pack["fallback_reason"] = route_reason
                prolog_pack["clauses_count"] = None
                prolog_pack["prolog_max_result"] = prolog_max_result
                prolog_pack["meta_interpreter"] = meta_interpreter
                prolog_pack["max_depth"] = max_depth
        else:
            route_reason = "Dataset not supported for Prolog."
            prolog_pack["fallback_reason"] = route_reason
            prolog_pack["clauses_count"] = None
            prolog_pack["prolog_max_result"] = prolog_max_result
            prolog_pack["meta_interpreter"] = meta_interpreter
            prolog_pack["max_depth"] = max_depth

        add_message("assistant", final_answer, history)
        time.sleep(SLEEP_SEC)

        correctness = judge_correctness(dataset_key, gold, final_answer)


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
                "prolog_max_result": prolog_max_result,
                "prolog_enabled": prolog_pack.get("enabled", False),
                "prolog_task_type": prolog_pack.get("task_type"),
                "meta_interpreter": meta_interpreter,
                "max_depth": max_depth,
            },
            "guideline": guideline,
            "gold": gold,
            "draft_raw": draft_raw,
            "draft": draft,
            "pred_raw": pred_raw,
            "pred": pred,
            "llm_candidate": llm_candidate,
            "llm_candidate_norm": llm_candidate_norm,
            "final_answer": final_answer,
            "correctness": correctness,
            "round1_guideline_prompt": g_prompt,
            "round2_solve_prompt": solve_prompt,
            "log": history,
            "route": route,
            "prolog": prolog_pack,
            "route_reason": route_reason,
        }

        with open(log_path, "w", encoding="utf-8") as lf:
            json.dump(payload, lf, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="gsm8k / prontoqa / proofwriter / mmlu / clutrr / sqa / date")
    parser.add_argument("--method", required=True, help="sd_selfguide or cot_selfguide")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--force_task_type",
        choices=("Yes", "No", "Partial"),
        default=None,
    )
    parser.add_argument("--data_path", default=None, help="override dataset jsonl path")
    parser.add_argument("--log_dir", default=None, help="override log directory")
    parser.add_argument("--mock_llm", action="store_true", help="use deterministic mock outputs (no API call)")
    parser.add_argument("--meta_interpreter", default=PROLOG_META_INTERPRETER)
    parser.add_argument("--max_depth", type=int, default=PROLOG_MAX_DEPTH)
    parser.add_argument("--prolog_max_result", type=int, default=PROLOG_MAX_RESULT)

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
        meta_interpreter=args.meta_interpreter,
        max_depth=args.max_depth,
        prolog_max_result=args.prolog_max_result,
    )
