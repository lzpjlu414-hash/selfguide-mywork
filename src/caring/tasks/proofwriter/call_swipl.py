import tempfile
import os
import re
import subprocess
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

curr_dir = os.path.dirname(os.path.abspath(__file__))

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

def extract_clauses_from_code(prolog_code: str):
    lines = prolog_code.split("\n")
    clauses = []

    continue_signal = False
    for i, line in enumerate(lines):
        if line.startswith("%") or line.startswith("/*") or line.strip() == '':
            continue_signal = False
            continue
        
        if "%" in line:
            line = line.split("%")[0].strip()
        if "/*" in line:
            line = line.split("/*")[0].strip()

        if continue_signal and line.startswith(" "):
            clauses[-1] += ' ' + line.strip()
        else:
            clauses.append(line)
            continue_signal = True
    clauses = [_.strip().rstrip('.') for _ in clauses]
    
    predicates = []
    for clause in clauses:
        if clause.startswith(":-"):
            continue
        if ':-' in clause:
            head, body = clause.split(":-")
            predicates.extend(
                [_.strip() for _ in head.split("(")[:-1]]
            )
    return clauses, set(predicates)

##############################################
#              Main Function                 #
#            To call SWI-Prolog              #
##############################################
def consult_prolog(
        prolog_string,
        query_string,
        meta_interpreter="raw",
        max_depth=5,
        debug=False,
        dataset_name="vanilla",
        max_result=20,
):
    
    """
    Args:
        prolog_string:
            string, the string of Prolog knwoledge base to be consulted
        query_string:
            string, the string of Prolog query to be executed
        consult_raw_query:
            bool, whether to consult the raw query, i.e., **NO** special meta-interpreter is used.
        generate_proof_tree:
            bool, whether to generate the proof tree for the query
        max_depth:
            int, the maximum depth of the iterative deepening search
        debug:
            bool, whether to print all the inputs and outputs when interacting with SWI-Prolog
        dataset_name:
            string, the name of the dataset, determines which meta-interpreter_*.pl to use
    """




    ########################################
    clauses, predicates = extract_clauses_from_code(prolog_string)
    ########################################

    user_query = _wrap_query(query_string, meta_interpreter, max_depth)

    # import pdb; pdb.set_trace()

    # Write the Prolog knowledge base to a temporary file.
    tmp_clause_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    with open(tmp_clause_file.name, 'w') as f:
        f.writelines(
            [clause.strip() + '\n' for clause in clauses] + [user_query + '\n']
        )
    tmp_output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    mi_path = os.path.join(file_path, "meta_interpreter.pl")
    tmp_clause_path = os.path.abspath(tmp_clause_file.name)
    tmp_output_path = os.path.abspath(tmp_output_file.name)

    ###### Execute Prolog ######
    response = _run_individual_prologging(
        tmp_clause_path,
        mi_path,
        tmp_output_path,
        max_result=max_result,
        debug=debug,
    )

    exec_meta = {
        "returncode": response.returncode,
        "stderr_tail": (response.stderr or "")[-2000:],
        "stdout_tail": (response.stdout or "")[-2000:],
        "meta_interpreter": meta_interpreter,
        "max_depth": max_depth,
    }

    results = _read_jsonl(tmp_output_file.name) if response.returncode == 0 else []

    output = {
        "answer": None,
        "proofs": [],
        "exec": exec_meta,
    }

    if results:
        output["answer"] = True
    if meta_interpreter in ("with_proof", "iter_deep_with_proof"):
        output["proofs"] = _collect_proofs(results)
    os.remove(tmp_clause_file.name)
    os.remove(tmp_output_file.name)
    
    # import pdb; pdb.set_trace()
    
    return output
def main():
    parser = ArgumentParser()
    parser.add_argument("--assert_path", type=str, required=True)
    parser.add_argument("--mi_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_result", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--meta_interpreter",
        type=str,
        default="raw",
        choices=["raw", "with_proof", "iter_deep_with_proof", "iter_deep_no_proof"],
    )
    parser.add_argument("--max_depth", type=int, default=25)
    args = parser.parse_args()

    lines = _read_lines_utf8_sig(args.assert_path)
    if not lines:
        raise ValueError(f"assert_path is empty: {args.assert_path}")

    orig_query = lines[-1]
    facts = lines[:-1]

    wrapped_query = _wrap_query(orig_query, args.meta_interpreter, args.max_depth)

    tmp_assert = tempfile.NamedTemporaryFile(suffix=".pl", delete=False)
    try:
        with open(tmp_assert.name, "w", encoding="utf-8", newline="\n") as f:
            for ln in facts:
                f.write(ln + "\n")
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

    p = _run_individual_prologging(tmp_assert.name, args.mi_path, tmp_out.name, args.max_result, args.debug)
    exec_meta = {
        "returncode": p.returncode,
        "stderr_tail": (p.stderr or "")[-2000:],
        "stdout_tail": (p.stdout or "")[-2000:],
        "meta_interpreter": args.meta_interpreter,
        "max_depth": args.max_depth,
    }

    results = _read_jsonl(tmp_out.name)

    output = {"answer": [], "proofs": [], "exec": exec_meta}

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

    if answers:
        output["answer"] = list(dict.fromkeys(str(a) for a in answers))
    if args.meta_interpreter in ("with_proof", "iter_deep_with_proof"):
        output["proofs"] = _collect_proofs(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    _safe_remove(tmp_assert.name)
    _safe_remove(tmp_out.name)

    sys.exit(p.returncode)


if __name__ == "__main__":
    main()