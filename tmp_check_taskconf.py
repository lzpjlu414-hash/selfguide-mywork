import json
from src.abs.self_guide_myself import parse_task_type_with_confidence

path = r"runs\debug_taskconf\20260206-225433\gsm8k_0.json"
with open(path, "r", encoding="utf-8") as f:
    j = json.load(f)

g = j.get("guideline")
print("guideline_type:", type(g))
if isinstance(g, dict):
    print("guideline_keys:", sorted(g.keys()))
print("parsed:", parse_task_type_with_confidence(g))

# 构造降置信版本：删掉关键字段，看看 confidence 是否降到 <0.4
if isinstance(g, dict):
    g_low = dict(g)
    for k in ["symbolization_schema", "query_goal", "fallback_policy", "fallback", "schema", "query"]:
        g_low.pop(k, None)
    print("parsed_low:", parse_task_type_with_confidence(g_low))
