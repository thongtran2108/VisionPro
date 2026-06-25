"""
core/xlsx_io.py — Đọc/ghi XLSX tối giản CHỈ bằng thư viện chuẩn (zipfile + xml).

Mục đích: CSV Logger xuất .xlsx mà KHÔNG cần openpyxl/pandas → giữ app
self-contained khi đóng gói (PyInstaller). Ghi cell dạng inlineStr (string)
hoặc numeric (<v>). Đọc hỗ trợ inlineStr, sharedStrings (file do Excel/
openpyxl tạo) và numeric.

API:
    write_xlsx(path, headers, rows, sheet_name="Log")
    read_xlsx(path) -> (headers, rows)      # headers = hàng đầu; rows = phần còn lại
    append_row(path, headers, row, sheet_name="Log")
"""
from __future__ import annotations
from typing import List, Tuple, Any
import os
import re
import zipfile
from xml.sax.saxutils import escape


# ── Column letter ↔ index ─────────────────────────────────────────
def _col_letter(idx: int) -> str:
    """0 → A, 25 → Z, 26 → AA…"""
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def _col_index(letters: str) -> int:
    """A → 0, Z → 25, AA → 26…"""
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    '</Types>'
)

_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    '</Relationships>'
)

_WB_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    '</Relationships>'
)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _cell_xml(ref: str, value: Any) -> str:
    if value is None:
        return f'<c r="{ref}"/>'
    if _is_number(value):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return (f'<c r="{ref}" t="inlineStr"><is>'
            f'<t xml:space="preserve">{text}</t></is></c>')


def _sheet_xml(all_rows: List[List[Any]]) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetData>']
    for ri, row in enumerate(all_rows, start=1):
        cells = "".join(_cell_xml(f"{_col_letter(ci)}{ri}", v)
                        for ci, v in enumerate(row))
        parts.append(f'<row r="{ri}">{cells}</row>')
    parts.append('</sheetData></worksheet>')
    return "".join(parts)


def _workbook_xml(sheet_name: str) -> str:
    name = escape(sheet_name)[:31] or "Log"
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{name}" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>')


def write_xlsx(path: str, headers: List[Any], rows: List[List[Any]],
               sheet_name: str = "Log") -> None:
    """Ghi MỚI file .xlsx. headers = hàng đầu (có thể rỗng); rows = các hàng
    dữ liệu (list các list)."""
    all_rows: List[List[Any]] = []
    if headers:
        all_rows.append(list(headers))
    all_rows.extend(list(r) for r in rows)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("xl/workbook.xml", _workbook_xml(sheet_name))
        z.writestr("xl/_rels/workbook.xml.rels", _WB_RELS)
        z.writestr("xl/worksheets/sheet1.xml", _sheet_xml(all_rows))


# ── Reader ────────────────────────────────────────────────────────
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _num(raw: str):
    """'12' → 12, '1.5' → 1.5, còn lại giữ nguyên string. Giúp số giữ kiểu
    qua mỗi vòng đọc-ghi (append) thay vì biến thành text."""
    if raw == "":
        return ""
    try:
        return int(raw) if not any(ch in raw for ch in ".eE") else float(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _parse_shared_strings(data: bytes) -> List[str]:
    import xml.etree.ElementTree as ET
    out: List[str] = []
    root = ET.fromstring(data)
    for si in root:
        if _strip_ns(si.tag) != "si":
            continue
        # <si> có thể chứa nhiều <r><t>… (rich text) → nối lại
        buf = []
        for t in si.iter():
            if _strip_ns(t.tag) == "t":
                buf.append(t.text or "")
        out.append("".join(buf))
    return out


def read_xlsx(path: str) -> Tuple[List[str], List[List[str]]]:
    """Đọc sheet đầu tiên → (headers, rows). headers = hàng đầu, rows = phần
    còn lại. Hỗ trợ inlineStr / sharedStrings / numeric. Lỗi → ([], [])."""
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = set(z.namelist())
            shared: List[str] = []
            if "xl/sharedStrings.xml" in names:
                shared = _parse_shared_strings(z.read("xl/sharedStrings.xml"))
            sheet_name = "xl/worksheets/sheet1.xml"
            if sheet_name not in names:
                cands = sorted(n for n in names
                               if n.startswith("xl/worksheets/")
                               and n.endswith(".xml"))
                if not cands:
                    return [], []
                sheet_name = cands[0]
            root = ET.fromstring(z.read(sheet_name))
    except Exception:
        return [], []

    grid: List[List[str]] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "row":
            continue
        row_vals: List[str] = []
        max_ci = -1
        cell_map = {}
        for c in el:
            if _strip_ns(c.tag) != "c":
                continue
            ref = c.attrib.get("r", "")
            m = re.match(r"([A-Z]+)(\d+)", ref)
            ci = _col_index(m.group(1)) if m else len(cell_map)
            ctype = c.attrib.get("t", "")
            val = ""
            if ctype == "inlineStr":
                buf = [t.text or "" for t in c.iter()
                       if _strip_ns(t.tag) == "t"]
                val = "".join(buf)
            else:
                v_el = next((t for t in c if _strip_ns(t.tag) == "v"), None)
                raw = v_el.text if v_el is not None and v_el.text else ""
                if ctype == "s":
                    try:
                        val = shared[int(raw)]
                    except (ValueError, IndexError):
                        val = ""
                elif ctype in ("str", "b"):
                    val = raw
                else:
                    val = _num(raw)   # numeric cell → giữ kiểu số
            cell_map[ci] = val
            max_ci = max(max_ci, ci)
        for i in range(max_ci + 1):
            row_vals.append(cell_map.get(i, ""))
        grid.append(row_vals)

    if not grid:
        return [], []
    headers = grid[0]
    return headers, grid[1:]


def append_row(path: str, headers: List[Any], row: List[Any],
               sheet_name: str = "Log") -> None:
    """Thêm 1 hàng vào .xlsx. File chưa có → tạo mới với header. File đã có →
    đọc lại, nối hàng, ghi đè (full rewrite — đủ nhanh cho logger AOI)."""
    existing_rows: List[List[Any]] = []
    if os.path.exists(path):
        old_headers, old_rows = read_xlsx(path)
        if old_headers:
            existing_rows = old_rows
            headers = old_headers   # giữ header cũ để cột thẳng hàng
    existing_rows.append(list(row))
    write_xlsx(path, headers, existing_rows, sheet_name=sheet_name)
