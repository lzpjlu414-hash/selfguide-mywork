import os
import re
import json
import sys
import time
import tempfile
import uuid
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "1.0"


def _build_schema_output(ok: bool, answer=None, proof=None, error_code=None, raw=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(ok),
        "answer": answer,
        "proof": proof,
        "error_code": error_code,
        "raw": raw,
    }


def _write_schema_output(path: str, payload: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)



def _read_lines_utf8_sig(path: str):
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

def _resolve_tmp_root(tmp_dir: Optional[str]) -> Path:
    if tmp_dir:
        return Path(tmp_dir).expanduser().resolve()
    env_dir = os.getenv("TMP_PROLOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.cwd() / "tmp_prolog").resolve()


def _make_run_dir(tmp_dir: Optional[str], task_prefix: str) -> Path:
    root = _resolve_tmp_root(tmp_dir)
    run_id = f"{task_prefix}_{os.getpid()}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    run_dir = (root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    task_dir = Path(__file__).resolve().parent
    caring_dir = task_dir.parent.parent
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


def _collect_proofs(results: list) -> list:
    proofs = []
    for r in results:
        if not isinstance(r, dict):
            continue
        if "Proof" in r and r["Proof"] is not None:
            proofs.append(r["Proof"])
        elif "proof" in r and r["proof"] is not None:
            proofs.append(r["proof"])
        elif "proofs" in r and r["proofs"] is not None:
            if isinstance(r["proofs"], list):
                proofs.extend(r["proofs"])
            else:
                proofs.append(r["proofs"])
    return list(dict.fromkeys(str(pr) for pr in proofs if str(pr).strip()))

def consult_prolog(
    prolog_string,
    query_string,
    meta_interpreter="raw",
    max_depth=5,
    debug=False,
    dataset_name="vanilla",
    max_result=20,
    keep_tmp=False,
    tmp_dir=None,
):
    run_dir = _make_run_dir(tmp_dir, task_prefix="gsm8k")
    tmp_assert = tempfile.NamedTemporaryFile(suffix=".pl", delete=False, dir=run_dir)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=run_dir)
    try:
        with open(tmp_assert.name, "w", encoding="utf-8", newline="\n") as fobj:
            payload_lines = [ln.strip() for ln in (prolog_string or "").splitlines() if ln.strip()]
            for ln in payload_lines:
                fobj.write(ln.rstrip(".") + ".\n")
            query = _wrap_query(query_string, meta_interpreter, max_depth)
            fobj.write(query.rstrip(".") + ".\n")

        p = _run_individual_prologging(
            tmp_assert.name,
            str(Path(__file__).resolve().parent / "meta_interpreter.pl"),
            tmp_out.name,
            max_result,
            debug,
        )
        results = _read_jsonl(tmp_out.name) if p.returncode == 0 else []
        proof = None
        if meta_interpreter in ("with_proof", "iter_deep_with_proof"):
            proofs = _collect_proofs(results)
            proof = "\n".join(proofs) if proofs else None
        return {
            "answer": bool(results),
            "proofs": [proof] if proof else [],
            "ok": p.returncode == 0,
            "error_code": None if p.returncode == 0 else "SWIPL_CALL_FAILED",
        }
    finally:
        if not (debug or keep_tmp):
            _safe_remove(tmp_assert.name)
            _safe_remove(tmp_out.name)
            try:
                run_dir.rmdir()
            except Exception:
                pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--assert_path", type=str, required=True)
    parser.add_argument("--mi_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_result", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    parser.add_argument("--tmp_dir", default=None)

    parser.add_argument("--meta_interpreter", type=str, default="raw",
                        choices=["raw", "with_proof", "iter_deep_with_proof", "iter_deep_no_proof"])
    parser.add_argument("--max_depth", type=int, default=25)

    args = parser.parse_args()

    returncode = 1
    run_dir = _make_run_dir(args.tmp_dir, task_prefix="gsm8k")
    tmp_assert = None
    tmp_out = None
    try:
        lines = _read_lines_utf8_sig(args.assert_path)
        if not lines:
            _write_schema_output(args.output_path, _build_schema_output(False, error_code="EMPTY_ASSERT_PATH"))
            sys.exit(1)

        orig_query = lines[-1]
        facts = lines[:-1]
        wrapped_query = _wrap_query(orig_query, args.meta_interpreter, args.max_depth)

        tmp_assert = tempfile.NamedTemporaryFile(suffix=".pl", delete=False, dir=run_dir)
        with open(tmp_assert.name, "w", encoding="utf-8", newline="\n") as f:
            for ln in facts:
                f.write(ln + "\n")
            uq = wrapped_query.strip()
            if not uq.endswith("."):
                uq += "."
            f.write(uq + "\n")
        tmp_assert.close()

        tmp_out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, dir=run_dir)
        tmp_out.close()
        p = _run_individual_prologging(tmp_assert.name, args.mi_path, tmp_out.name, args.max_result, args.debug)

        returncode = p.returncode

        results = _read_jsonl(tmp_out.name)

        q0 = (orig_query or "").strip().rstrip(".")
        vars_in_q = [v for v in re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", q0) if v != "Proof"]
        key = vars_in_q[0] if vars_in_q else None

        answers = []
        for r in results:
            if not isinstance(r, dict):
                continue

            if key and key in r:
                answers.append(r[key])
            elif not key:
                for k in ("Answer", "Ans", "Result", "X"):
                    if k in r:
                        answers.append(r[k])
                        break

        if not key and not answers and results:
            answers = ["True"]

        answer = None
        if answers:
            answer = list(dict.fromkeys(str(a) for a in answers))[0]

        proof = None
        if args.meta_interpreter in ("with_proof", "iter_deep_with_proof"):
            proofs = _collect_proofs(results)
            if proofs:
                proof = "\n".join(proofs)
            elif returncode == 0 and results:
                proof = "NO_PROOF_RETURNED"

        if returncode != 0:
                error_code = "SWIPL_CALL_FAILED"
        elif not results:
                error_code = "NO_SOLUTION"
        elif len(set(str(a) for a in answers)) > 1:
                error_code = "MULTIPLE_ANSWERS_CONFLICT"
        elif args.meta_interpreter in ("with_proof", "iter_deep_with_proof") and proof is None:
                error_code = "PROOF_MISSING"
        else:
                error_code = None

        payload = _build_schema_output(
                ok=(error_code is None),
                answer=answer,
                proof=proof,
                error_code=error_code,
                raw={
                    "results_count": len(results),
                    "returncode": returncode,
                    "stdout_tail": (p.stdout or "")[-2000:],
                    "stderr_tail": (p.stderr or "")[-2000:],
                    "meta_interpreter": args.meta_interpreter,
                    "max_depth": args.max_depth,
                },
            )
        _write_schema_output(args.output_path, payload)
    except subprocess.TimeoutExpired:
        _write_schema_output(args.output_path, _build_schema_output(False, error_code="SWIPL_TIMEOUT"))
        returncode = 124
    except Exception as e:
        _write_schema_output(args.output_path,
                             _build_schema_output(False, error_code="SWIPL_EXEC_EXCEPTION", raw=str(e)))
        returncode = 1
    finally:
        if not (args.debug or args.keep_tmp):
            if tmp_assert is not None:
                _safe_remove(tmp_assert.name)
            if tmp_out is not None:
                _safe_remove(tmp_out.name)
            try:
                run_dir.rmdir()
            except Exception:
                pass


    sys.exit(returncode)


if __name__ == "__main__":
    main()
