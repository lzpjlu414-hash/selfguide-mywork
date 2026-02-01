import os
import json
import re
from typing import Any, Optional, Set, List, Tuple


# -------------------------
# 工具：把 dict/list 转成文本
# -------------------------
def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# -------------------------
# 从 student_log/teacher_log 抽“模型最终输出”
# 返回 (raw_text, is_fallback_dump)
# -------------------------
def extract_raw_from_log(log_obj: Any) -> Tuple[str, bool]:
    if log_obj is None:
        return "", False

    # 情况 A：log 已经是字符串
    if isinstance(log_obj, str):
        s = log_obj.strip()
        return s, False

    # 情况 B：log 是 dict
    if isinstance(log_obj, dict):
        candidate_keys = [
            "final_answer", "final", "answer", "prediction", "pred",
            "model_output", "output", "response", "completion",
            "assistant_response", "assistant", "content", "text"
        ]
        for k in candidate_keys:
            if k in log_obj and log_obj[k]:
                return _to_text(log_obj[k]).strip(), False

        # messages 列表
        if "messages" in log_obj and isinstance(log_obj["messages"], list):
            for m in reversed(log_obj["messages"]):
                if isinstance(m, dict) and m.get("role") in ("assistant", "model"):
                    c = m.get("content")
                    if c:
                        return _to_text(c).strip(), False

        # 类 OpenAI choices
        if "choices" in log_obj and isinstance(log_obj["choices"], list) and log_obj["choices"]:
            ch0 = log_obj["choices"][0]
            if isinstance(ch0, dict):
                msg = ch0.get("message")
                if isinstance(msg, dict) and msg.get("content"):
                    return _to_text(msg["content"]).strip(), False
                if ch0.get("text"):
                    return _to_text(ch0["text"]).strip(), False

        # 保底 dump
        return json.dumps(log_obj, ensure_ascii=False), True

    # 情况 C：log 是 list（你现在就是这个）
    if isinstance(log_obj, list):
        # 找最后一个 assistant/model 的 content
        for m in reversed(log_obj):
            if isinstance(m, dict) and m.get("role") in ("assistant", "model"):
                c = m.get("content")
                if c:
                    return _to_text(c).strip(), False
        return json.dumps(log_obj, ensure_ascii=False), True

    return str(log_obj).strip(), True


def extract_raw(d: dict) -> Tuple[str, bool]:
    raw, fb = extract_raw_from_log(d.get("student_log"))
    if raw:
        return raw, fb
    raw, fb = extract_raw_from_log(d.get("teacher_log"))
    if raw:
        return raw, fb
    return "", False


# -------------------------
# 解析：从 raw 输出里抽关系标签（修复 granddaughter->daughter 的 bug）
# -------------------------
def _last_label_in_text(text: str, label_set: Set[str]) -> Optional[str]:
    """
    在文本里找 label_set 里的标签，返回“最后一次完整单词出现”的标签。
    关键：用 finditer 的 start 位置，避免 rfind 在 'granddaughter' 里匹配到 'daughter' 子串。
    同时：长标签优先（granddaughter > daughter）
    """
    if not text or not label_set:
        return None

    low = text.lower()
    norm_map = {lab.lower(): lab for lab in label_set}
    labels_sorted = sorted(norm_map.keys(), key=len, reverse=True)

    best_lab = None
    best_pos = -1
    for lab_low in labels_sorted:
        for m in re.finditer(rf"\b{re.escape(lab_low)}\b", low):
            if m.start() > best_pos:
                best_pos = m.start()
                best_lab = lab_low

    return norm_map.get(best_lab) if best_lab else None


def parse_pred(raw: str, label_set: Set[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None

    # 0) raw 本身就是标签
    if label_set:
        norm_map = {lab.lower(): lab for lab in label_set}
        if s.lower() in norm_map:
            return norm_map[s.lower()]

    # 1) 优先从“最后几行”的 Answer/Final Answer 行抽（最稳）
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    tail = lines[-12:]  # 只看末尾，避免过程里提到的标签干扰
    for ln in reversed(tail):
        m = re.search(r"(Final\s*Answer|Answer)\s*:\s*(.*)$", ln, flags=re.I)
        if m:
            rest = m.group(2).strip()
            lab = _last_label_in_text(rest, label_set)
            if lab:
                return lab

    # 2) 再找全文的 Final Answer: <label>
    m = re.search(r'Final\s*Answer\s*:\s*([A-Za-z_]+)', s, flags=re.I)
    if m and label_set:
        cand = m.group(1).strip().lower()
        norm_map = {lab.lower(): lab for lab in label_set}
        if cand in norm_map:
            return norm_map[cand]

    # 3) 最后：在全文中找最后出现的标签
    return _last_label_in_text(s, label_set)


def is_strict_format_ok(raw: str) -> bool:
    if raw is None:
        return False
    s = raw.strip()
    if "\n" in s:
        return False
    return re.match(r"^Final\s*Answer\s*:\s*[A-Za-z_]+\s*$", s, flags=re.I) is not None


# -------------------------
# 主流程
# -------------------------
log_dir = "log/CLUTRR"
files = sorted([fn for fn in os.listdir(log_dir) if fn.endswith(".json")])

all_data = []
for fname in files:
    with open(os.path.join(log_dir, fname), "r", encoding="utf-8") as f:
        all_data.append(json.load(f))

print("示例 keys:", list(all_data[0].keys()))
print("student_log type:", type(all_data[0].get("student_log")))
print("student_log preview:", _to_text(all_data[0].get("student_log"))[:500])

# gold 就是顶层 answer
label_set: Set[str] = set()
missing_gold = 0
for d in all_data:
    gold = str(d.get("answer", "")).strip()
    if gold:
        label_set.add(gold)
    else:
        missing_gold += 1
print(f"label_set 大小: {len(label_set)}, missing_gold: {missing_gold}")

total = len(all_data)

baseline_correct = 0
parsed_correct = 0

parse_fail_count = 0
format_violation_count = 0
missing_raw_count = 0
missing_gold_count = 0

fail_cases = []
wrong_cases = []

for i, d in enumerate(all_data):
    # 基线（guide_correctness）
    if str(d.get("guide_correctness", "")).lower() == "true":
        baseline_correct += 1

    gold = str(d.get("answer", "")).strip()
    raw, is_fallback_dump = extract_raw(d)

    if not gold:
        missing_gold_count += 1
    if not raw:
        missing_raw_count += 1

    parsed_pred = parse_pred(raw, label_set)

    # 前10条 debug
    if i < 10:
        print("=" * 70)
        print(f"[{i}] GOLD: {gold}")
        print(f"[{i}] PARSED_PRED: {parsed_pred}")
        print(f"[{i}] RAW_MODEL_OUTPUT:\n{raw[:1200]}")
        if is_fallback_dump:
            print(f"[{i}] NOTE: RAW 是从 log dump 保底提取（字段里没找到明确 final 输出）")

    # 解析失败
    if parsed_pred is None or (label_set and parsed_pred not in label_set):
        parse_fail_count += 1
        if raw and (not is_fallback_dump) and (not is_strict_format_ok(raw)):
            format_violation_count += 1
        fail_cases.append((d.get("id"), gold, raw[-400:]))
        continue

    # 解析后准确率
    if gold and parsed_pred == gold:
        parsed_correct += 1
    else:
        wrong_cases.append((d.get("id"), gold, parsed_pred, raw[-300:]))

print("\n===== 汇总 =====")
print(f"总样本数: {total}")
print(f"基线(guide_correctness) 正确数: {baseline_correct}, 准确率: {baseline_correct/total:.2%}")
print(f"解析后 正确数: {parsed_correct}, 准确率: {parsed_correct/total:.2%}")
print(f"parse_fail_count: {parse_fail_count}")
print(f"format_violation_count: {format_violation_count}")
print(f"missing_raw_count: {missing_raw_count}")
print(f"missing_gold_count: {missing_gold_count}")

print("\n===== parse_fail 样例（最多10条）=====")
for x in fail_cases[:10]:
    print("-" * 50)
    print("id:", x[0], "gold:", x[1])
    print("raw_tail:", x[2])

print("\n===== 答错样例（最多10条）=====")
for x in wrong_cases[:10]:
    print("-" * 50)
    print("id:", x[0], "gold:", x[1], "pred:", x[2])
    print("raw_tail:", x[3])
