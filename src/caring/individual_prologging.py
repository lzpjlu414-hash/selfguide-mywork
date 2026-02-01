import pyswip   #连接 SWI-Prolog
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert_path", type=str, required=True, help="")
    parser.add_argument("--mi_path", type=str, required=True, help="")
    parser.add_argument('--output_path', type=str, required=True)
    # parser.add_argument('--query', type=str, required=True)
    parser.add_argument("--max_result", type=int, default=20)

    args = parser.parse_args()

    #这相当于启动了一个 SWI-Prolog 会话，后面 consult/assertz/query 都在这个会话里执行。
    prolog = pyswip.Prolog() #初始化 Prolog 引擎
    
    with open(args.assert_path, 'r', encoding='utf-8') as f:
        _clauses = [_.strip() for _ in f.readlines() if _.strip()]

    if not _clauses:
        raise ValueError(f"assert_path is empty: {args.assert_path}")

    #把最后一行当query，其余当clauses（要assert 的规则 / 事实）
    query = _clauses[-1]
    clauses = _clauses[:-1]

    if query.endswith('.'):
        query = query[:-1]
    
    try:
        prolog.consult(args.mi_path)
        for clause in clauses:
            if clause.endswith('.'):
                clause = clause[:-1]
            prolog.assertz(clause)
            
        results = prolog.query(query, maxresult=args.max_result)

        with open(args.output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r).strip() + '\n')
    except pyswip.prolog.PrologError as e:
                print("PrologError:", e)
