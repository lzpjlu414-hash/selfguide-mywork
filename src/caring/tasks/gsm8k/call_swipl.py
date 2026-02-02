import tempfile
import os
import re
import subprocess #在 Python 里启动外部命令（这里是启动另一个 python 脚本，脚本内部再调用 SWI-Prolog）。
import json
import shutil
import sys
from interruptingcow import timeout

curr_dir = os.path.dirname(os.path.abspath(__file__))


#从一段 Prolog 文本里提取“干净的子句列表 clauses”（去注释、拼续行、去末尾点号）
#从这些子句里抽出“谓词名集合 predicates”（例如 parent/2 的谓词名是 parent）
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
        else:
            predicates.append(clause.split('(')[0].strip())
    return clauses, set(predicates)

##############################################
#              Main Function                 #
#            To call SWI-Prolog              #
##############################################
#这个函数把一段 Prolog 文本 + query，交给 SWI-Prolog 执行，并返回答案/证明。
def consult_prolog(
        prolog_string,
        query_string,
        meta_interpreter="raw", #选择不同“元解释器”包装查询（是否返回 Proof，是否迭代加深）
        max_depth=5,
        debug=False,
        dataset_name="vanilla",
        output_path=None,
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


    prolog_meta_interpreters = {
        "raw": "{}",
        "with_proof": "mi_tree(g({}), Proof)",  # With proof generation. One argument: Goal
        "iter_deep_with_proof": "mi_id_limit(g({}), Proof, {})",   # Iterative deepening search, with proof generation. Two arguments: Goal, MaxDepth
        # "iter_deep_no_proof": prolog_output_all_answers.format("mi_id_limit_no_proof(g({}), {})"),  # Iterative deepening search. Two arguments: Goal, MaxDepth
        "iter_deep_no_proof": "mi_id_limit_no_proof(g({}), {})",  # Iterative deepening search. Two arguments: Goal, MaxDepth
    }

    ########################################
    clauses, predicates = extract_clauses_from_code(prolog_string)
    ########################################

    if query_string.endswith('.'):
        query_string = query_string[:-1].strip()
    # which types of meta-interpreters to use for querying Prolog
    #生成最终要丢给 Prolog 执行的“查询语句 user_query”
    if "iter_deep" not in meta_interpreter:
        user_query = prolog_meta_interpreters[meta_interpreter].format(query_string)
    else:
        user_query = prolog_meta_interpreters[meta_interpreter].format(query_string, max_depth)

    # Write the Prolog knowledge base to a temporary file.
    tmp_clause_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    with open(tmp_clause_file.name, 'w') as f:
        f.writelines(
            [clause.strip() + '\n' for clause in clauses] + [user_query + '\n']
        )
    tmp_output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    # Windows 上要先 close，否则后续写入/复制可能被占用
    try:
        tmp_output_file.close()
    except Exception:
        pass



    file_path = os.path.dirname(os.path.abspath(__file__))
    mi_path = os.path.join(file_path, "meta_interpreter.pl")
    tmp_clause_path = os.path.abspath(tmp_clause_file.name)
    tmp_output_path = os.path.abspath(tmp_output_file.name)

    # print("*** Prolog Code ***")
    # print(prolog_string)
    # print()
    # print("*** Clauses ***")
    # print('\n'.join(clauses))
    # print()
    # print("*** Query Code ***")
    # print(query_string)
    # print()
    # import pdb; pdb.set_trace()

    ###### Execute Prolog ######
    command = [
            "python",
            f"{curr_dir.split('/src/')[0]}/src/individual_prologging.py",
            "--assert_path",
            tmp_clause_path,
            "--mi_path",   #meta_interpreter.pl
            mi_path,
            "--output_path",
            tmp_output_path,
        ]
    try:
        with timeout(10.0, RuntimeError):
            response= subprocess.run(
                command,
            )
    except RuntimeError as e:
        response = None

    if response and response.returncode == 0:
        with open(tmp_output_file.name, 'r', encoding='utf-8') as f:
            results = [json.loads(_) for _ in f.readlines() if _.strip()]
    else:
        results = []

    output = {
        'answer': None,
        'proofs': None,
    }

    # Extract the query(Key). For example, given "query(Salary)"", we extract "Salary".
    target_key = re.findall(r'\((.*?)\)', query_string, re.DOTALL)
    assert len(target_key) == 1


    num_results = 0
    for r in results:
        num_results += 1
        if target_key[0] in r:
            if output["answer"] is None:
                output["answer"] = [r[target_key[0]]]
            else:
                output["answer"].append(r[target_key[0]])
        if "Proof" in r:
            if output["proofs"] is None:
                output["proofs"] = [r['Proof']]
            else:
                output["proofs"].append(r['Proof'])
    output['answer'] = list(set(output['answer'])) if output['answer'] is not None else [""]
    output['proofs'] = list(set(output['proofs'])) if output['proofs'] is not None else [""]

    if output_path:
        final_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        if (not os.path.exists(tmp_output_path)) or (os.path.getsize(tmp_output_path) == 0):
            print(f"[call_swipl] ERROR: temp output not created or empty: {tmp_output_path}", file=sys.stderr)
            if response:
                print(f"[call_swipl] stdout:\n{getattr(response, 'stdout', '')}", file=sys.stderr)
                print(f"[call_swipl] stderr:\n{getattr(response, 'stderr', '')}", file=sys.stderr)
            sys.exit(1)
        try:
            shutil.copyfile(tmp_output_path, final_output_path)
        except Exception as e:
            print(f"[call_swipl] ERROR: failed to copy temp output -> output_path: {e}", file=sys.stderr)
            print(f"[call_swipl] temp:  {tmp_output_path}", file=sys.stderr)
            print(f"[call_swipl] final: {final_output_path}", file=sys.stderr)
            sys.exit(1)
        try:
            os.remove(tmp_output_path)
        except Exception:
            pass

    os.remove(tmp_clause_file.name)
    if os.path.exists(tmp_output_file.name):
        os.remove(tmp_output_file.name)
    # del prolog
    # prolog.query("halt")

    # for i in range(10):
    #     for predicate in predicates:
    #         prolog.retractall("{}".format(predicate))
    #         prolog.retractall("{}({})".format(predicate, ",".join(['_' for _ in range(i)])))
    #         # prolog.query(f"abolish({predicate}/{i})")

    return output
