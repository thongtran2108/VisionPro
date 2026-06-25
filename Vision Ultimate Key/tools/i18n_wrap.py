"""
tools/i18n_wrap.py — Auto-wrap chuỗi UI bằng tr() (AST-guided, byte-accurate).

Chỉ bọc tr() quanh string literal là ĐỐI SỐ TRỰC TIẾP của UI sink an toàn
(QLabel, QPushButton, setText, setToolTip, setWindowTitle, addTab, addAction,
QMessageBox.*…). KHÔNG đụng addItem/addItems (combo đọc currentText làm logic),
setStyleSheet, print, f-string (JoinedStr). Bỏ chuỗi đã wrap (arg là Call) và
chuỗi style/kỹ thuật. Tự thêm import tr nếu thiếu.

Dùng:  python tools/i18n_wrap.py <file1> <file2> ...        # apply
       python tools/i18n_wrap.py --dry <file...>            # chỉ in kế hoạch
"""
from __future__ import annotations
import ast, sys, re

SINKS = {
    "QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QGroupBox",
    "QAction", "QToolButton", "QCommandLinkButton",
    "setText", "setToolTip", "setPlaceholderText", "setWindowTitle",
    "setStatusTip", "setTitle", "setInformativeText", "setLabelText",
    "addTab", "addAction", "addMenu", "showMessage",
    "information", "warning", "critical", "question", "about",
}
# Cố tình KHÔNG wrap: addItem(s)/insertItem/setItemText/setButtonText (combo
# logic), setStyleSheet (CSS), setObjectName, print…

VN = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


def is_styleish(s: str) -> bool:
    if not s or not s.strip():
        return True
    low = s.lower()
    if "{" in s and any(k in low for k in
                        ("background", "color:", "border", "padding", "font-",
                         "margin", "qpushbutton", "qlabel", "qwidget", "qcombobox",
                         "qlineedit", "qframe", "qtabbar", "qmenu", "qscrollbar",
                         "qslider", "qprogress", "qcheckbox", "qheader", "qtable")):
        return True
    if s.startswith("#") and len(s) <= 9:
        return True
    if re.fullmatch(r"[\s\d.,:;/_\-+=%°\[\](){}<>|*#]*", s):
        return True
    return False


def funcname(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def collect_spans(tree):
    """Trả list (start_line, start_col, end_line, end_col) byte-offset của các
    Constant-str cần wrap."""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if funcname(node) not in SINKS:
            continue
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                if is_styleish(a.value):
                    continue
                spans.append((a.lineno, a.col_offset, a.end_lineno, a.end_col_offset))
    return spans


def apply_to_file(path, dry=False):
    raw = open(path, "rb").read()
    src = raw.decode("utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  SKIP {path}: parse error {e}"); return 0

    # line byte-offsets (ast col_offset = byte offset within line, utf-8)
    blines = src.encode("utf-8").split(b"\n")
    # absolute byte start of each line
    starts = [0]
    for bl in blines:
        starts.append(starts[-1] + len(bl) + 1)  # +1 cho '\n'

    def abspos(line, col):
        return starts[line - 1] + col

    spans = collect_spans(tree)
    edits = []
    data = src.encode("utf-8")
    for (sl, sc, el, ec) in spans:
        a = abspos(sl, sc); b = abspos(el, ec)
        orig = data[a:b]
        # skip nếu đã nằm trong tr( (phòng hờ) — orig là literal, kiểm tra ký
        # tự ngay trước có phải 'tr(' không (đã loại vì arg sẽ là Call, nhưng
        # double-check khi literal đứng ngay sau 'tr(')
        before = data[max(0, a-3):a]
        if before.endswith(b"tr("):
            continue
        edits.append((a, b, orig))

    if not edits:
        return 0
    # apply desc
    edits.sort(key=lambda x: -x[0])
    for a, b, orig in edits:
        data = data[:a] + b"tr(" + orig + b")" + data[b:]

    new = data.decode("utf-8")
    # ensure import — chèn sau import top-level CUỐI (dùng end_lineno để
    # không rơi vào giữa import ngoặc nhiều dòng).
    if "from core.i18n import" not in new:
        try:
            t2 = ast.parse(new)
            last = 0
            for n in t2.body:
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    last = max(last, getattr(n, "end_lineno", n.lineno))
            lines = new.split("\n")
            lines.insert(last, "from core.i18n import tr")
            new = "\n".join(lines)
        except Exception:
            new = "from core.i18n import tr\n" + new

    if dry:
        print(f"  {path}: would wrap {len(edits)} strings")
        return len(edits)
    open(path, "w", encoding="utf-8").write(new)
    print(f"  {path}: wrapped {len(edits)} strings")
    return len(edits)


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry" in args
    files = [a for a in args if a != "--dry"]
    total = sum(apply_to_file(f, dry) for f in files)
    print(f"TOTAL: {total}")
