"""
core/i18n.py — Hệ thống đa ngôn ngữ (i18n) cho VisionPro.

Thiết kế tối giản, runtime, KHÔNG dùng .ts/.qm của Qt:
  • Chuỗi nguồn trong code (đang trộn Anh/Việt) chính là "key" tra cứu.
  • Mỗi ngôn ngữ có 1 bảng {key_nguồn: bản_dịch}. Thiếu key → trả NGUYÊN
    chuỗi nguồn (fallback an toàn — app luôn chạy được dù chưa dịch đủ).
  • Ngôn ngữ chọn 1 lần lúc khởi động (đọc QSettings). Đổi ngôn ngữ trong
    app chỉ lưu lựa chọn + nhắc khởi động lại để áp dụng (đơn giản, chắc
    chắn, không phải retranslate từng widget lúc chạy).

Cách dùng:
    from core.i18n import tr
    QLabel(tr("File"))          # vi→"Tệp", zh→"文件", en→"File"
    QLabel(tr("Đang nạp dự án…"))   # en→"Loading project…", zh→"正在加载项目…"

Thêm bản dịch: sửa core/translations.py (không cần đụng file này).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Mã ngôn ngữ hỗ trợ + nhãn hiển thị (luôn show theo bản ngữ để dễ nhận ra).
LANGUAGES: List[Tuple[str, str]] = [
    ("vi", "Tiếng Việt"),
    ("en", "English"),
    ("zh", "中文"),
]
LANGUAGE_CODES: List[str] = [code for code, _ in LANGUAGES]
DEFAULT_LANG = "vi"          # App vốn viết tiếng Việt → mặc định 'vi'.
_SETTINGS_KEY = "language"   # Khoá QSettings lưu lựa chọn ngôn ngữ.

_current: str = DEFAULT_LANG
_tables: Dict[str, Dict[str, str]] = {}


def _load_tables() -> None:
    """Nạp bảng dịch (lazy, 1 lần). Lỗi → bảng rỗng → tr() trả chuỗi gốc."""
    global _tables
    if _tables:
        return
    try:
        from core.translations import TRANSLATIONS
        _tables = TRANSLATIONS
    except Exception as e:  # pragma: no cover - phòng thủ
        print(f"[i18n] load translations failed: {e}")
        _tables = {}


def set_language(code: str, persist: bool = False) -> None:
    """Đặt ngôn ngữ hiện tại. persist=True → ghi vào QSettings (áp dụng ở
    lần khởi động sau)."""
    global _current
    _current = code if code in LANGUAGE_CODES else DEFAULT_LANG
    if persist:
        try:
            from PySide6.QtCore import QSettings
            QSettings().setValue(_SETTINGS_KEY, _current)
        except Exception:
            pass


def current_language() -> str:
    return _current


def language_label(code: str) -> str:
    for c, lbl in LANGUAGES:
        if c == code:
            return lbl
    return code


def init_language() -> str:
    """Đọc ngôn ngữ đã lưu từ QSettings → set làm ngôn ngữ hiện tại.
    Gọi 1 lần trong main() ngay sau khi tạo QApplication (trước mọi dialog).
    Trả về mã ngôn ngữ đang dùng."""
    _load_tables()
    code = DEFAULT_LANG
    try:
        from PySide6.QtCore import QSettings
        saved = QSettings().value(_SETTINGS_KEY, DEFAULT_LANG)
        if saved:
            code = str(saved)
    except Exception:
        pass
    set_language(code, persist=False)
    return _current


def localize_tool_registry() -> None:
    """Dịch TẠI CHỖ các field hiển thị của TOOL_REGISTRY theo ngôn ngữ hiện
    tại: tool.name, tool.description, và label/tooltip của mỗi param.

    KHÔNG đụng: category (là key tra màu/icon ở tool_library), choices (giá
    trị logic proc đọc/so sánh), tool_id, tên port — các thứ này dịch ở chỗ
    HIỂN THỊ (qua tr()) chứ không mutate dữ liệu.

    Gọi 1 lần trong main() sau init_language(). Đổi ngôn ngữ cần khởi động
    lại → mỗi lần chạy registry được import mới rồi localize sang ngôn ngữ
    đã lưu (không bị dịch chồng)."""
    try:
        from core.tool_registry import TOOL_REGISTRY
    except Exception as e:  # pragma: no cover
        print(f"[i18n] localize_tool_registry skip: {e}")
        return
    for t in TOOL_REGISTRY:
        try:
            t.name = tr(t.name)
            t.description = tr(t.description)
            for p in t.params:
                p.label = tr(p.label)
                if p.tooltip:
                    p.tooltip = tr(p.tooltip)
        except Exception:
            pass


def tr(text: str) -> str:
    """Dịch chuỗi nguồn sang ngôn ngữ hiện tại. Thiếu bản dịch (hoặc đang
    ở ngôn ngữ gốc của chuỗi) → trả NGUYÊN chuỗi nguồn."""
    if not text:
        return text
    if not _tables:
        _load_tables()
    table = _tables.get(_current)
    if not table:
        return text
    return table.get(text, text)
