# LEGACY: One-off encoding fix utility; not a primary pipeline entry point.
from pathlib import Path

p = Path(r"src\caring\tasks\gsm8k\meta_interpreter.pl")
b = p.read_bytes()

# 不要把 latin-1 放进候选：它“永远能解码成功”，但很可能解错
encs = ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "gb18030", "gbk"]

text = None
used = None
for enc in encs:
    try:
        text = b.decode(enc)
        used = enc
        break
    except UnicodeDecodeError:
        pass

if text is None:
    raise SystemExit("decode failed for all candidate encodings")

# 统一：去 BOM + 统一换行 + 清掉常见不可见空格
text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
text = text.replace("\u00a0", " ").replace("\u3000", " ")

with p.open("w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("decoded_with =", used)
print("rewritten_as_utf8 =", p)
