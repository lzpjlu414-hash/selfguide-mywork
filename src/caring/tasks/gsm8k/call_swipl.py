import os
import re
import json
import sys
import time
import tempfile
import subprocess
from argparse import ArgumentParser
from pathlib import Path


def _read_lines_utf8_sig(path: str):
    # 吃掉 BOM（你之前已经踩过 \ufeff）
    with open(path, "r", encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def _safe_remove(path: str, retries: int = 8, sleep_sec: float = 0.1):
    if not path or not os.path.exists(path):
        return
    for _ in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            time.sleep(sleep_sec)
        except Exception:
            return


def _wrap_query(query_string: str, meta_interpreter: str, max_depth: int):
    q = (query_string or "").strip()
    if q.endswith("."):
        q = q[:-1].strip()

    templ = {
        "raw": "{}",
        "with_proof": "mi_tree(g({}), Proof)",
        "iter_deep_with_proof": "mi_id_limit(g({}), Proof, {})",
        "iter_deep_no_proof": "mi_id_limit_no_proof(g({}), {})",
    }
    if meta_interpreter not in templ:
        raise ValueError(f"Unknown meta_interpreter: {meta_interpreter}")

    if "iter_deep" in meta_interpreter:
        return templ[meta_interpreter].format(q, max_depth)
    return templ[meta_interpreter].format(q)


def _run_individual_prologging(assert_path: str, mi_path: str, out_path: str, max_result: int, debug: bool):
    # 定位 src/caring/individual_prologging.py
    task_dir = Path(__file__).resolve().parent          # .../tasks/gsm8k
    caring_dir = task_dir.parent.parent                 # .../caring
    individual_py = caring_dir / "individual_prologging.py"
    if not individual_py.exists():
        raise FileNotFoundError(f"individual_prologging.py not found: {individual_py}")

    cmd = [
        sys.executable, str(individual_py),
        "--assert_path", str(assert_path),
        "--mi_path", str(mi_path),
        "--output_path", str(out_path),
        "--max_result", str(max_result),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if debug:
        print("[call_swipl] CMD:", " ".join(cmd), file=sys.stderr)
        print("[call_swipl] returncode:", p.returncode, file=sys.stderr)
        if p.stdout:
            print("[call_swipl] stdout:\n" + p.stdout, file=sys.stderr)
        if p.stderr:
            print("[call_swipl] stderr:\n" + p.stderr, file=sys.stderr)

    return p


def _read_jsonl(path: str):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return []
    txt = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        return []

    # 兼容：有的实现写的是单行 JSON（非 jsonl）
    if "\n" not in txt:
        try:
            obj = json.loads(txt)
            return [obj] if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
        except Exception:
            return []

    items = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            items.append(json.loads(ln))
        except Exception:
            pass
    return items


def main():
    parser = ArgumentParser()
    parser.add_argument("--assert_path", type=str, required=True)
    parser.add_argument("--mi_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_result", type=int, default=20)
    parser.add_argument("--debug", action="store_true")

    # 可选参数：如果你后续要 proof / iter deep，就用它们
    parser.add_argument("--meta_interpreter", type=str, default="raw",
                        choices=["raw", "with_proof", "iter_deep_with_proof", "iter_deep_no_proof"])
    parser.add_argument("--max_depth", type=int, default=25)

    args = parser.parse_args()

    lines = _read_lines_utf8_sig(args.assert_path)
    if not lines:
        raise ValueError(f"assert_path is empty: {args.assert_path}")

    orig_query = lines[-1]
    facts = lines[:-1]

    wrapped_query = _wrap_query(orig_query, args.meta_interpreter, args.max_depth)

    # 写一个“只替换 query 行”的临时 assert 文件（facts 原样保留）
    tmp_assert = tempfile.NamedTemporaryFile(suffix=".pl", delete=False)
    try:
        with open(tmp_assert.name, "w", encoding="utf-8", newline="\n") as f:
            for ln in facts:
                f.write(ln + "\n")
            # individual_prologging 会 strip '.'，这里确保最多 1 个 '.'
            uq = wrapped_query.strip()
            if not uq.endswith("."):
                uq += "."
            f.write(uq + "\n")
    finally:
        try:
            tmp_assert.close()
        except Exception:
            pass

    tmp_out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    try:
        tmp_out.close()
    except Exception:
        pass

    # 调 individual_prologging
    p = _run_individual_prologging(tmp_assert.name, args.mi_path, tmp_out.name, args.max_result, args.debug)

    results = _read_jsonl(tmp_out.name)

    # 从 orig_query 里抓变量名：daily_profit(Profit) -> Profit
    output = {"answer": [""], "proofs": [""]}
    target_key = re.findall(r"\((.*?)\)", (orig_query or "").strip().rstrip("."), re.DOTALL)
    if len(target_key) == 1:
        key = target_key[0].strip()
        answers, proofs = [], []
        for r in results:
            if isinstance(r, dict) and key in r:
                answers.append(r[key])
            if isinstance(r, dict) and "Proof" in r:
                proofs.append(r["Proof"])
        if answers:
            output["answer"] = list({str(a) for a in answers})
        if proofs:
            output["proofs"] = list({str(pr) for pr in proofs})

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    # 清理临时文件（debug 时也清理，避免你目录堆垃圾；要保留就自己注释掉）
    _safe_remove(tmp_assert.name)
    _safe_remove(tmp_out.name)

    # 让上层能感知失败：individual_prologging 非 0 就非 0
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
