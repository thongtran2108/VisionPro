"""
tools/i18n_scan.py — Quét chuỗi GUI cần dịch (AST-based).

Tìm string literal là ĐỐI SỐ của các "UI sink" (QLabel, setText, setToolTip,
QMessageBox.*, addItem(s), addAction, addTab, setWindowTitle, setPlaceholderText,
showMessage…) + field của ToolDef / P / ParamDef (name, description, label,
tooltip, choices). Bỏ qua print(), stylesheet, mã kỹ thuật.

Xuất:
  • /tmp/i18n_strings.json  — list unique chuỗi plain (target dịch) + meta.
  • /tmp/i18n_dynamic.json  — f-string ở UI sink (cần convert tay).
  • In thống kê per-file + tổng.
"""
from __future__ import annotations
import ast, json, os, re, sys

ROOTS = ["core", "ui", "main.py"]

# Hàm/widget nhận text user-facing.
UI_SINKS = {
    "QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QGroupBox",
    "QAction", "QToolButton", "QCommandLinkButton",
    "setText", "setToolTip", "setPlaceholderText", "setWindowTitle",
    "setStatusTip", "setTitle", "setInformativeText", "setItemText",
    "addTab", "addItem", "addItems", "insertItem", "addAction", "addMenu",
    "showMessage", "setLabelText", "setButtonText",
    "information", "warning", "critical", "question", "about",
}
TOOLDEF = {"ToolDef"}
PARAMDEF = {"P", "ParamDef"}

VN = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
                "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]")


def is_styleish(s: str) -> bool:
    """Loại stylesheet / mã kỹ thuật / format code."""
    if not s or not s.strip():
        return True
    low = s.lower()
    if "{" in s and any(k in low for k in
                        ("background", "color:", "border", "padding", "font-",
                         "margin", "qpushbutton", "qlabel", "qwidget", "qcombobox",
                         "qlineedit", "qframe", "qtabbar", "qmenu", "qscrollbar")):
        return True
    if s.startswith("#") and len(s) <= 9:           # hex color
        return True
    if re.fullmatch(r"[\s\d.,:;/_\-+=%°\[\](){}<>|]*", s):  # chỉ ký hiệu/số
        return True
    if re.fullmatch(r"[a-z_]+", s) and len(s) <= 3:  # mã ngắn 'abc'
        return True
    return False


def funcname(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


results = {}   # str -> list "file:line"
dynamic = []   # f-string sites
files = {}


def add(s, where):
    if s is None or is_styleish(s):
        return
    results.setdefault(s, []).append(where)
    f = where.split(":")[0]
    files[f] = files.get(f, 0) + 1


def scan_file(path):
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        print("  parse fail", path, e); return
    rel = os.path.relpath(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = funcname(node)
        if fn is None:
            continue
        where = f"{rel}:{node.lineno}"
        if fn == "print":
            continue
        if fn in UI_SINKS:
            for a in node.args:
                s = const_str(a)
                if s is not None:
                    add(s, where)
                elif isinstance(a, ast.List):       # addItems([...])
                    for el in a.elts:
                        add(const_str(el), where)
                elif isinstance(a, ast.JoinedStr):  # f-string
                    parts = [p.value for p in a.values
                             if isinstance(p, ast.Constant) and isinstance(p.value, str)]
                    txt = "".join(parts)
                    if txt.strip() and not is_styleish(txt):
                        dynamic.append({"where": where, "sink": fn, "parts": txt})
        elif fn in TOOLDEF:
            if len(node.args) >= 2:
                add(const_str(node.args[1]), where)   # name
            if len(node.args) >= 4:
                add(const_str(node.args[3]), where)   # description
        elif fn in PARAMDEF:
            if len(node.args) >= 2:
                add(const_str(node.args[1]), where)   # label
            for kw in node.keywords:
                if kw.arg == "tooltip":
                    add(const_str(kw.value), where)
                elif kw.arg == "choices" and isinstance(kw.value, ast.List):
                    for el in kw.value.elts:
                        add(const_str(el), where)


def walk_root(r):
    if r.endswith(".py"):
        scan_file(r); return
    for dp, _, fns in os.walk(r):
        if "__pycache__" in dp or "/vendor" in dp.replace(os.sep, "/"):
            continue
        for fn in fns:
            if fn.endswith(".py"):
                scan_file(os.path.join(dp, fn))


for r in ROOTS:
    if os.path.exists(r):
        walk_root(r)

uniq = sorted(results.keys())
vi_src = [s for s in uniq if VN.search(s)]
en_src = [s for s in uniq if not VN.search(s)]

meta = [{"s": s, "src": "vi" if VN.search(s) else "en", "n": len(results[s]),
         "at": results[s][0]} for s in uniq]
json.dump(meta, open("/tmp/i18n_strings.json", "w"), ensure_ascii=False, indent=1)
json.dump(dynamic, open("/tmp/i18n_dynamic.json", "w"), ensure_ascii=False, indent=1)

print(f"UNIQUE GUI strings: {len(uniq)}  (vi-src={len(vi_src)}  en-src={len(en_src)})")
print(f"Dynamic f-string UI sites: {len(dynamic)}")
print("\nPer-file occurrences (top):")
for f, c in sorted(files.items(), key=lambda x: -x[1]):
    print(f"  {c:4}  {f}")
