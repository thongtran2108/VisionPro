"""
ui/sfc_dialog.py — Dialog cấu hình SFC/MES Integration

Một dialog duy nhất, ba section (giống `PLCDialog`):
1. Scanner — chọn COM port (auto-detect), baudrate, trigger, test scan.
2. API GET — URL template với {SN}, headers, test request.
3. API POST — URL, headers, body template JSON (placeholder
   {SN}, {<node_id>.<port>} hoặc {<tool_id>.<port>}), test request.

Settings persist qua QSettings, lưu cùng config app.
"""
from __future__ import annotations

import json
from typing import Optional, List, Callable

from PySide6.QtCore import Qt, QSettings, Signal, QStringListModel
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QPushButton, QCheckBox,
    QPlainTextEdit, QMessageBox, QWidget, QFrame, QSizePolicy,
    QScrollArea, QCompleter,
)
from PySide6.QtGui import QFont, QTextCursor, QKeyEvent

from core.sfc import (
    SfcManager, ScannerConfig, ApiGetConfig, ApiPostConfig,
    list_serial_ports,
)


# Separator giữa identifier (tool_id / node_id) và human label trong popup
# suggestion. Khi insert, strip phần phía sau để chỉ chèn identifier.
_LABEL_SEP = "   —   "


# ── Body-template editor with {placeholder} autocomplete ──────────
class _TemplateEdit(QPlainTextEdit):
    """QPlainTextEdit có autocomplete cho placeholder `{...}`.

    - Gõ `{` → popup gợi ý builtins (SN, api_get.value) + tool_id của mọi
      node hiện có trong graph.
    - Sau `{<tool>.` → popup gợi ý các output port của tool đó.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._provider: Optional[Callable[[str], List[str]]] = None
        self._completer = QCompleter([], self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseInsensitive)
        # Substring match — gõ "patmax" sẽ khớp "patmax_align", gõ "pat" cũng
        # ra cả "patmax_align" và "patfind". User không cần nhớ tiền tố.
        try:
            self._completer.setFilterMode(Qt.MatchContains)
        except Exception:
            pass    # Qt < 5.2 fallback (PySide6 luôn có)
        self._completer.activated.connect(self._insert_completion)
        self._model = QStringListModel([], self._completer)
        self._completer.setModel(self._model)

    def set_provider(self, fn: Callable[[str], List[str]]) -> None:
        """fn(prefix) -> list of suggestions theo context hiện tại."""
        self._provider = fn

    # ── context helpers ───────────────────────────────────────
    def _ref_context(self):
        """Return (token_before_cursor, has_dot) nếu cursor đang trong
        ``{...}``; else None."""
        c = self.textCursor()
        pos = c.position()
        text = self.toPlainText()
        # Tìm `{` gần nhất phía trước cursor, không bị `}` chen vào
        open_idx = text.rfind("{", 0, pos)
        close_idx = text.rfind("}", 0, pos)
        if open_idx == -1 or open_idx < close_idx:
            return None
        # Nếu có dấu '}' hoặc xuống dòng giữa { và cursor → không phải ref
        between = text[open_idx + 1:pos]
        if "}" in between or "\n" in between:
            return None
        return between

    def _current_prefix(self):
        """Phần đang gõ sau `{` hoặc sau `.`. Trả ('', '') nếu không ở ref."""
        token = self._ref_context()
        if token is None:
            return None
        if "." in token:
            head, _, tail = token.rpartition(".")
            return ("port", head, tail)
        return ("name", "", token)

    # ── popup ─────────────────────────────────────────────────
    def _trigger_popup(self):
        ctx = self._current_prefix()
        if ctx is None or self._provider is None:
            self._completer.popup().hide()
            return
        kind, head, prefix = ctx
        # Provider trả full list theo kind (head dùng cho kind="port")
        if kind == "port":
            items = self._provider(f"@port:{head}")
        else:
            items = self._provider("@name:")
        if not items:
            self._completer.popup().hide()
            return
        self._model.setStringList(items)
        self._completer.setCompletionPrefix(prefix)
        # Hiển thị popup gần cursor
        rect = self.cursorRect()
        rect.setWidth(self._completer.popup().sizeHintForColumn(0)
                      + self._completer.popup().verticalScrollBar().sizeHint().width()
                      + 20)
        self._completer.complete(rect)

    def _insert_completion(self, completion: str):
        ctx = self._current_prefix()
        if ctx is None:
            return
        kind, head, prefix = ctx
        # Strip phần human label (sau separator) — chỉ chèn tool_id / node_id.
        if _LABEL_SEP in completion:
            completion = completion.split(_LABEL_SEP, 1)[0]
        c = self.textCursor()
        # Xoá phần prefix đang gõ rồi chèn completion
        for _ in range(len(prefix)):
            c.deletePreviousChar()
        c.insertText(completion)
        # Sau khi chọn tool name → gợi ý chèn dấu '.' để tiếp port
        if kind == "name":
            # Builtins (SN, api_get.value, api_get.text) đã đầy đủ → đóng `}`
            if completion in ("SN", "api_get.value", "api_get.text"):
                c.insertText("}")
                self.setTextCursor(c)
                self._completer.popup().hide()
                return
            c.insertText(".")
            self.setTextCursor(c)
            # Trigger popup port ngay (giờ context có dấu '.')
            self._trigger_popup()
        else:
            # Chọn xong port → đóng bằng '}'
            c.insertText("}")
            self.setTextCursor(c)
            self._completer.popup().hide()

    # ── events ────────────────────────────────────────────────
    def keyPressEvent(self, e: QKeyEvent):
        popup = self._completer.popup()
        if popup.isVisible():
            # Delegate navigation/accept tới popup; activated → _insert_completion.
            if e.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape,
                            Qt.Key_Tab, Qt.Key_Backtab):
                e.ignore()
                return
        super().keyPressEvent(e)
        # Sau khi insert/move → re-check context
        ctx = self._current_prefix()
        if ctx is None:
            popup.hide()
            return
        # Trigger popup khi gõ ký tự liên quan hoặc khi cursor di chuyển trong ref
        text = e.text()
        if text in ("{", "."):
            self._trigger_popup()
        elif text and (text.isalnum() or text == "_"):
            self._trigger_popup()
        elif not text:
            # Arrow / Home / End → refresh nếu vẫn trong ref
            self._trigger_popup()
        else:
            popup.hide()


class SfcDialog(QDialog):
    """Dialog cấu hình SFC. Snapshot cấu hình → SfcManager khi user nhấn Save."""

    sn_scanned = Signal(str)   # phát ra mỗi khi scan thành công

    def __init__(self, manager: SfcManager, graph=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("SFC / MES Integration")
        # Cho window có nút minimize/maximize + tự maximize khi mở.
        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(1024, 900)
        self._mgr = manager
        self._graph = graph
        if graph is not None:
            self._mgr.set_graph(graph)
        self._build_ui()
        self._load_settings()
        self._refresh_ports()
        self._refresh_node_picker()
        self._wire_body_autocomplete()

    def showEvent(self, event):
        """Tự maximize lần đầu mở để không bị cắt nội dung trên màn nhỏ."""
        super().showEvent(event)
        if not getattr(self, "_did_first_maximize", False):
            self._did_first_maximize = True
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def set_graph(self, graph) -> None:
        self._graph = graph
        self._mgr.set_graph(graph)
        self._refresh_node_picker()
        # Provider closure capture self._graph qua self → tự cập nhật, nhưng
        # rewire để đảm bảo editor lấy bản mới.
        self._wire_body_autocomplete()

    # ════════════════════════════════════════════════════════════════
    #  UI build
    # ════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # Outer layout wraps a scroll area so all 3 sections luôn truy cập
        # được trên màn hình nhỏ — vẫn cuộn được kể cả khi minimize.
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body_w = QWidget(); scroll.setWidget(body_w)
        outer.addWidget(scroll)
        root = QVBoxLayout(body_w); root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── 1. Scanner ────────────────────────────────────────────
        gb_scan = QGroupBox("1.  Hand Scanner  (Serial / COM)")
        g = QGridLayout(gb_scan)

        self.cb_port = QComboBox()
        self.cb_port.setMinimumWidth(280)
        self.btn_refresh_ports = QPushButton("🔄  Refresh")
        self.btn_refresh_ports.clicked.connect(self._refresh_ports)

        self.sp_baud = QSpinBox(); self.sp_baud.setRange(300, 921600); self.sp_baud.setValue(9600)
        self.cb_bytesize = QComboBox(); self.cb_bytesize.addItems(["5","6","7","8"]); self.cb_bytesize.setCurrentText("8")
        self.cb_parity   = QComboBox(); self.cb_parity.addItems(["N","E","O","M","S"])
        self.cb_stopbits = QComboBox(); self.cb_stopbits.addItems(["1","1.5","2"])
        self.sp_timeout  = QSpinBox(); self.sp_timeout.setRange(10, 60000); self.sp_timeout.setValue(1000); self.sp_timeout.setSuffix(" ms")

        self.le_trigger  = QLineEdit("02 F4 03")
        self.le_trigger.setPlaceholderText("Hex bytes (vd 02 F4 03). Để trống nếu scanner auto-trigger.")
        self.sp_readsize = QSpinBox(); self.sp_readsize.setRange(1, 4096); self.sp_readsize.setValue(128); self.sp_readsize.setSuffix(" bytes")
        self.cb_encoding = QComboBox(); self.cb_encoding.addItems(["utf-8","ascii","latin-1","utf-16"])
        self.sp_expected_len = QSpinBox(); self.sp_expected_len.setRange(0, 4096); self.sp_expected_len.setValue(0); self.sp_expected_len.setSpecialValueText("any")
        self.le_contains = QLineEdit("")
        self.le_contains.setPlaceholderText("(tuỳ chọn) data phải chứa chuỗi này")

        row = 0
        g.addWidget(QLabel("COM port:"), row, 0); g.addWidget(self.cb_port, row, 1, 1, 2); g.addWidget(self.btn_refresh_ports, row, 3); row += 1
        g.addWidget(QLabel("Baudrate:"), row, 0); g.addWidget(self.sp_baud, row, 1)
        g.addWidget(QLabel("Byte:"), row, 2); g.addWidget(self.cb_bytesize, row, 3); row += 1
        g.addWidget(QLabel("Parity:"), row, 0); g.addWidget(self.cb_parity, row, 1)
        g.addWidget(QLabel("Stop bits:"), row, 2); g.addWidget(self.cb_stopbits, row, 3); row += 1
        g.addWidget(QLabel("Timeout:"), row, 0); g.addWidget(self.sp_timeout, row, 1)
        g.addWidget(QLabel("Read size:"), row, 2); g.addWidget(self.sp_readsize, row, 3); row += 1
        g.addWidget(QLabel("Trigger (hex):"), row, 0); g.addWidget(self.le_trigger, row, 1, 1, 3); row += 1
        g.addWidget(QLabel("Encoding:"), row, 0); g.addWidget(self.cb_encoding, row, 1)
        g.addWidget(QLabel("Expected len:"), row, 2); g.addWidget(self.sp_expected_len, row, 3); row += 1
        g.addWidget(QLabel("Contains:"), row, 0); g.addWidget(self.le_contains, row, 1, 1, 3); row += 1

        self.btn_scan_test = QPushButton("📟  Test Scan")
        self.btn_scan_test.clicked.connect(self._on_scan_test)
        self.lbl_last_sn = QLabel("Last SN: (chưa scan)")
        self.lbl_last_sn.setStyleSheet("color:#39ff14; font-family:'Courier New'; font-size:11px;")
        g.addWidget(self.btn_scan_test, row, 0, 1, 2)
        g.addWidget(self.lbl_last_sn, row, 2, 1, 2); row += 1

        root.addWidget(gb_scan)

        # ── 2. API GET ────────────────────────────────────────────
        gb_get = QGroupBox("2.  API GET  (lookup SN trên SFC)")
        gg = QGridLayout(gb_get)
        self.chk_get_enabled = QCheckBox("Enable API GET")
        gg.addWidget(self.chk_get_enabled, 0, 0, 1, 4)

        self.le_get_url = QLineEdit("")
        self.le_get_url.setPlaceholderText(
            "Vd: http://10.222.48.213:8888/v2/pass/mes/tsc/check/TSC-VN/TSC_VN1/OPLEUATTACHMENT?sn={SN}&station_id=OPLEUATTACHMENT")
        gg.addWidget(QLabel("URL template:"), 1, 0)
        gg.addWidget(self.le_get_url, 1, 1, 1, 3)

        self.le_get_headers = QLineEdit("{}")
        self.le_get_headers.setPlaceholderText('{"token":"abc","X-API-Key":"xxx"}')
        gg.addWidget(QLabel("Headers (JSON):"), 2, 0)
        gg.addWidget(self.le_get_headers, 2, 1, 1, 3)

        self.sp_get_timeout = QSpinBox(); self.sp_get_timeout.setRange(100, 60000); self.sp_get_timeout.setValue(5000); self.sp_get_timeout.setSuffix(" ms")
        self.sp_get_status  = QSpinBox(); self.sp_get_status.setRange(100, 599); self.sp_get_status.setValue(200)
        gg.addWidget(QLabel("Timeout:"), 3, 0); gg.addWidget(self.sp_get_timeout, 3, 1)
        gg.addWidget(QLabel("Expected status:"), 3, 2); gg.addWidget(self.sp_get_status, 3, 3)

        self.le_get_expected = QLineEdit("")
        self.le_get_expected.setPlaceholderText("(tuỳ chọn) response.text phải chứa chuỗi này → pass")
        gg.addWidget(QLabel("Must contain:"), 4, 0)
        gg.addWidget(self.le_get_expected, 4, 1, 1, 3)

        self.chk_get_parse_json = QCheckBox("Parse JSON")
        self.le_get_json_path   = QLineEdit("")
        self.le_get_json_path.setPlaceholderText('Dot path, vd "data.token"')
        gg.addWidget(self.chk_get_parse_json, 5, 0)
        gg.addWidget(QLabel("JSON path:"), 5, 1); gg.addWidget(self.le_get_json_path, 5, 2, 1, 2)

        self.chk_get_bypass_proxy = QCheckBox("Bypass system proxy (cho IP nội bộ LAN)")
        self.chk_get_bypass_proxy.setChecked(True)
        self.chk_get_bypass_proxy.setToolTip(
            "Bật để bỏ qua HTTP_PROXY / corporate Squid proxy — tránh "
            "request về IP LAN (10.x, 192.168.x) bị proxy chặn / trả HTML lỗi.")
        gg.addWidget(self.chk_get_bypass_proxy, 6, 0, 1, 4)

        self.btn_get_test = QPushButton("🌐  Test GET")
        self.btn_get_test.clicked.connect(self._on_get_test)
        gg.addWidget(self.btn_get_test, 7, 0, 1, 4)

        self.txt_get_resp = QPlainTextEdit()
        self.txt_get_resp.setReadOnly(True)
        self.txt_get_resp.setMaximumHeight(120)
        self.txt_get_resp.setStyleSheet("background:#0a0e1a; color:#94a3b8; font-family:'Courier New'; font-size:10px;")
        self.txt_get_resp.setPlaceholderText("Response sẽ hiện ở đây sau khi Test GET")
        gg.addWidget(self.txt_get_resp, 8, 0, 1, 4)
        root.addWidget(gb_get)

        # ── 3. API POST ───────────────────────────────────────────
        gb_post = QGroupBox("3.  API POST  (gửi kết quả sau pipeline)")
        gp = QGridLayout(gb_post)
        self.chk_post_enabled = QCheckBox("Enable API POST")
        gp.addWidget(self.chk_post_enabled, 0, 0, 1, 4)

        self.le_post_url = QLineEdit("")
        self.le_post_url.setPlaceholderText("Vd: http://10.222.48.213:8888/v2/sfc/result")
        gp.addWidget(QLabel("URL:"), 1, 0)
        gp.addWidget(self.le_post_url, 1, 1, 1, 3)

        self.cb_post_method = QComboBox(); self.cb_post_method.addItems(["POST","PUT"])
        self.sp_post_timeout = QSpinBox(); self.sp_post_timeout.setRange(100, 60000); self.sp_post_timeout.setValue(5000); self.sp_post_timeout.setSuffix(" ms")
        self.sp_post_status  = QSpinBox(); self.sp_post_status.setRange(100, 599); self.sp_post_status.setValue(200)
        gp.addWidget(QLabel("Method:"), 2, 0); gp.addWidget(self.cb_post_method, 2, 1)
        gp.addWidget(QLabel("Timeout:"), 2, 2); gp.addWidget(self.sp_post_timeout, 2, 3)

        self.le_post_headers = QLineEdit('{"Content-Type":"application/json"}')
        self.le_post_headers.setPlaceholderText('Vd: {"Content-Type":"application/json","token":"abc"}')
        gp.addWidget(QLabel("Headers (JSON):"), 3, 0)
        gp.addWidget(self.le_post_headers, 3, 1, 1, 3)

        gp.addWidget(QLabel("Expected status:"), 4, 0); gp.addWidget(self.sp_post_status, 4, 1)
        self.le_post_expected = QLineEdit("")
        self.le_post_expected.setPlaceholderText("(tuỳ chọn) response phải chứa chuỗi này")
        gp.addWidget(QLabel("Must contain:"), 4, 2); gp.addWidget(self.le_post_expected, 4, 3)

        self.chk_post_bypass_proxy = QCheckBox("Bypass system proxy (cho IP nội bộ LAN)")
        self.chk_post_bypass_proxy.setChecked(True)
        self.chk_post_bypass_proxy.setToolTip(
            "Bật để bỏ qua HTTP_PROXY / corporate Squid proxy.")
        gp.addWidget(self.chk_post_bypass_proxy, 5, 0, 1, 4)

        # Body template — JSON editor + insert reference helper
        hdr_body = QHBoxLayout()
        hdr_body.addWidget(QLabel("Body template (JSON):"))
        hdr_body.addStretch()
        self.cb_node_pick = QComboBox(); self.cb_node_pick.setMinimumWidth(220)
        self.cb_port_pick = QComboBox(); self.cb_port_pick.setMinimumWidth(140)
        self.cb_node_pick.currentIndexChanged.connect(self._on_node_pick_changed)
        self.btn_insert_ref = QPushButton("➕  Insert {ref}")
        self.btn_insert_ref.clicked.connect(self._on_insert_ref)
        hdr_body.addWidget(QLabel("Reference:")); hdr_body.addWidget(self.cb_node_pick)
        hdr_body.addWidget(self.cb_port_pick); hdr_body.addWidget(self.btn_insert_ref)
        gp.addLayout(hdr_body, 6, 0, 1, 4)

        self.txt_post_body = _TemplateEdit()
        self.txt_post_body.setFont(QFont("Courier New", 10))
        self.txt_post_body.setMinimumHeight(220)
        self.txt_post_body.setPlaceholderText(
            'Vd:\n'
            '{\n'
            '  "sn": "{SN}",\n'
            '  "result": "{patmax.found}",\n'
            '  "stationName": "OPLEUATTACHMENT"\n'
            '}\n\n'
            "Gõ `{` để xem gợi ý các tool đang có trong pipeline.\n"
            "Sau khi chọn tool, gõ `.` (tự thêm) để xem các port output.\n"
            "Tab / Enter để chấp nhận lựa chọn, Esc để đóng popup.")
        gp.addWidget(self.txt_post_body, 7, 0, 1, 4)

        self.btn_post_test = QPushButton("📤  Test POST")
        self.btn_post_test.clicked.connect(self._on_post_test)
        self.btn_post_preview = QPushButton("👁  Preview body")
        self.btn_post_preview.clicked.connect(self._on_post_preview)
        gp.addWidget(self.btn_post_preview, 8, 0, 1, 2)
        gp.addWidget(self.btn_post_test, 8, 2, 1, 2)

        self.txt_post_resp = QPlainTextEdit()
        self.txt_post_resp.setReadOnly(True)
        self.txt_post_resp.setMaximumHeight(120)
        self.txt_post_resp.setStyleSheet("background:#0a0e1a; color:#94a3b8; font-family:'Courier New'; font-size:10px;")
        self.txt_post_resp.setPlaceholderText("Response sẽ hiện ở đây sau khi Test POST")
        gp.addWidget(self.txt_post_resp, 9, 0, 1, 4)
        root.addWidget(gb_post)

        # ── Save / Close ──────────────────────────────────────────
        bar = QHBoxLayout()
        self.btn_save  = QPushButton("💾  Save")
        self.btn_close = QPushButton("Close")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_close.clicked.connect(self.close)
        bar.addStretch(); bar.addWidget(self.btn_save); bar.addWidget(self.btn_close)
        root.addLayout(bar)

    # ════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════
    def _refresh_ports(self):
        cur = self.cb_port.currentData() or self.cb_port.currentText()
        self.cb_port.clear()
        ports = list_serial_ports()
        if not ports:
            self.cb_port.addItem("(không phát hiện cổng — cắm scanner rồi Refresh)", "")
        for dev, desc in ports:
            self.cb_port.addItem(f"{dev}  —  {desc}", dev)
        # Restore previous selection if still present
        if cur:
            for i in range(self.cb_port.count()):
                if self.cb_port.itemData(i) == cur:
                    self.cb_port.setCurrentIndex(i); break

    def _refresh_node_picker(self):
        self.cb_node_pick.blockSignals(True)
        self.cb_node_pick.clear()
        self.cb_node_pick.addItem("(builtin) SN — mã scan", "SN")
        self.cb_node_pick.addItem("(builtin) api_get.value", "api_get.value")
        self.cb_node_pick.addItem("(builtin) api_get.text", "api_get.text")
        if self._graph is not None:
            for nid, n in self._graph.nodes.items():
                label = f"{n.tool.name}  [{nid}]"
                self.cb_node_pick.addItem(label, nid)
        self.cb_node_pick.blockSignals(False)
        self._on_node_pick_changed()

    def _wire_body_autocomplete(self):
        """Cung cấp gợi ý cho editor body template — chạy lại mỗi lần
        graph đổi (set_graph) để list tool/node luôn cập nhật."""
        if not hasattr(self, "txt_post_body"):
            return
        self.txt_post_body.set_provider(self._completion_provider)

    def _completion_provider(self, ctx: str) -> List[str]:
        """Trả về list suggestion cho editor.

        ctx:
            "@name:"            → list tool_id + builtin + (label hiển thị).
            "@port:<head>"      → list port của tool ứng với head.
        """
        if ctx == "@name:":
            items: List[str] = [
                "SN" + _LABEL_SEP + "Mã scan gần nhất",
                "api_get.value" + _LABEL_SEP + "Value extract từ API GET",
                "api_get.text" + _LABEL_SEP + "Response text của API GET",
            ]
            seen_tools = set()
            if self._graph is not None:
                # tool_id của các node đang có (vd "patmax_align — PatMax Align Tool")
                for n in self._graph.nodes.values():
                    if n.tool_id in seen_tools:
                        continue
                    seen_tools.add(n.tool_id)
                    items.append(f"{n.tool_id}{_LABEL_SEP}{n.tool.name}")
                # node_id (8-char) — phòng trường hợp pipeline có >1 node cùng tool_id
                for nid, n in self._graph.nodes.items():
                    items.append(f"{nid}{_LABEL_SEP}{n.tool.name} [{nid}]")
            return items

        if ctx.startswith("@port:"):
            head = ctx[len("@port:"):]
            ports = self._lookup_ports_for(head)
            return sorted(ports)
        return []

    def _lookup_ports_for(self, head: str) -> List[str]:
        """Tìm output port của tool theo head (node_id hoặc tool_id)."""
        if self._graph is None:
            return []
        # Match node_id trước
        n = self._graph.nodes.get(head)
        if n is None:
            for nn in self._graph.nodes.values():
                if nn.tool_id == head:
                    n = nn
                    break
        if n is None:
            return []
        return [p.name for p in n.tool.outputs]

    def _on_node_pick_changed(self):
        self.cb_port_pick.clear()
        data = self.cb_node_pick.currentData() or ""
        if data in ("SN", "api_get.value", "api_get.text"):
            self.cb_port_pick.addItem("(no port)", "")
            return
        if self._graph and data in self._graph.nodes:
            for p in self._graph.nodes[data].tool.outputs:
                self.cb_port_pick.addItem(p.name, p.name)

    def _on_insert_ref(self):
        node_data = self.cb_node_pick.currentData() or ""
        port_data = self.cb_port_pick.currentData() or ""
        if node_data in ("SN", "api_get.value", "api_get.text"):
            ref = "{" + node_data + "}"
        else:
            if not port_data:
                return
            ref = "{" + f"{node_data}.{port_data}" + "}"
        cursor = self.txt_post_body.textCursor()
        cursor.insertText(ref)

    # ════════════════════════════════════════════════════════════════
    #  Snapshot UI ↔ Manager
    # ════════════════════════════════════════════════════════════════
    def _gather_scanner(self) -> ScannerConfig:
        return ScannerConfig(
            port=self.cb_port.currentData() or self.cb_port.currentText().split(" ")[0] or "",
            baudrate=int(self.sp_baud.value()),
            bytesize=int(self.cb_bytesize.currentText()),
            parity=self.cb_parity.currentText(),
            stopbits=float(self.cb_stopbits.currentText()),
            timeout_ms=int(self.sp_timeout.value()),
            trigger_hex=self.le_trigger.text(),
            read_size=int(self.sp_readsize.value()),
            encoding=self.cb_encoding.currentText(),
            strip_chars=self._mgr.scanner.strip_chars,    # not exposed
            expected_length=int(self.sp_expected_len.value()),
            contains=self.le_contains.text(),
        )

    def _gather_get(self) -> ApiGetConfig:
        return ApiGetConfig(
            enabled=self.chk_get_enabled.isChecked(),
            url_template=self.le_get_url.text(),
            headers_json=self.le_get_headers.text() or "{}",
            timeout_ms=int(self.sp_get_timeout.value()),
            expected_status=int(self.sp_get_status.value()),
            expected_text=self.le_get_expected.text(),
            parse_json=self.chk_get_parse_json.isChecked(),
            json_path=self.le_get_json_path.text(),
            bypass_proxy=self.chk_get_bypass_proxy.isChecked(),
        )

    def _gather_post(self) -> ApiPostConfig:
        return ApiPostConfig(
            enabled=self.chk_post_enabled.isChecked(),
            url=self.le_post_url.text(),
            method=self.cb_post_method.currentText(),
            headers_json=self.le_post_headers.text() or "{}",
            body_template=self.txt_post_body.toPlainText(),
            timeout_ms=int(self.sp_post_timeout.value()),
            expected_status=int(self.sp_post_status.value()),
            expected_text=self.le_post_expected.text(),
            bypass_proxy=self.chk_post_bypass_proxy.isChecked(),
        )

    def _apply_to_manager(self):
        self._mgr.scanner = self._gather_scanner()
        self._mgr.api_get_cfg = self._gather_get()
        self._mgr.api_post_cfg = self._gather_post()

    # ════════════════════════════════════════════════════════════════
    #  Button handlers
    # ════════════════════════════════════════════════════════════════
    def _on_scan_test(self):
        self._apply_to_manager()
        sn, err = self._mgr.scan_once()
        if err:
            QMessageBox.warning(self, "Scan", f"Lỗi: {err}")
            return
        if not sn:
            QMessageBox.information(self, "Scan", "Không nhận được data (timeout?)")
            return
        self.lbl_last_sn.setText(f"Last SN: {sn}")
        self.sn_scanned.emit(sn)

    def _on_get_test(self):
        self._apply_to_manager()
        text, ok, err = self._mgr.api_get()
        if err:
            self.txt_get_resp.setPlainText(f"[ERROR] {err}")
            return
        self.txt_get_resp.setPlainText(
            f"[{'PASS' if ok else 'FAIL'}]\n{text}")

    def _on_post_preview(self):
        self._apply_to_manager()
        body_str, missing = self._mgr.resolve_placeholders(
            self._mgr.api_post_cfg.body_template)
        msg = body_str
        if missing:
            msg += f"\n\n⚠ Placeholder không resolve: {missing}"
        try:
            parsed = json.loads(body_str)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            msg = pretty + ("\n\n⚠ Missing: " + str(missing) if missing else "")
        except Exception as e:
            msg += f"\n\n⚠ JSON parse fail: {e}"
        self.txt_post_resp.setPlainText("[PREVIEW BODY]\n" + msg)

    def _on_post_test(self):
        self._apply_to_manager()
        text, ok, err = self._mgr.api_post()
        if err:
            self.txt_post_resp.setPlainText(f"[ERROR] {err}")
            return
        self.txt_post_resp.setPlainText(
            f"[{'PASS' if ok else 'FAIL'}]\n{text}")

    def _on_save(self):
        self._apply_to_manager()
        self._save_settings()
        QMessageBox.information(self, "Saved", "Đã lưu cấu hình SFC.")

    # ════════════════════════════════════════════════════════════════
    #  Persistence — QSettings group "sfc"
    # ════════════════════════════════════════════════════════════════
    def _save_settings(self):
        s = QSettings()
        s.beginGroup("sfc")
        s.setValue("scanner", json.dumps(self._mgr.scanner.to_dict()))
        s.setValue("api_get", json.dumps(self._mgr.api_get_cfg.to_dict()))
        s.setValue("api_post", json.dumps(self._mgr.api_post_cfg.to_dict()))
        s.endGroup()

    def _load_settings(self):
        s = QSettings()
        s.beginGroup("sfc")
        try:
            sc = json.loads(s.value("scanner", "") or "{}")
            self._mgr.scanner = ScannerConfig(**{
                **self._mgr.scanner.to_dict(), **sc})
        except Exception:
            pass
        try:
            gc = json.loads(s.value("api_get", "") or "{}")
            self._mgr.api_get_cfg = ApiGetConfig(**{
                **self._mgr.api_get_cfg.to_dict(), **gc})
        except Exception:
            pass
        try:
            pc = json.loads(s.value("api_post", "") or "{}")
            self._mgr.api_post_cfg = ApiPostConfig(**{
                **self._mgr.api_post_cfg.to_dict(), **pc})
        except Exception:
            pass
        s.endGroup()
        self._apply_manager_to_ui()

    def _apply_manager_to_ui(self):
        # Scanner
        sc = self._mgr.scanner
        if sc.port:
            for i in range(self.cb_port.count()):
                if self.cb_port.itemData(i) == sc.port:
                    self.cb_port.setCurrentIndex(i); break
            else:
                self.cb_port.addItem(sc.port, sc.port)
                self.cb_port.setCurrentIndex(self.cb_port.count() - 1)
        self.sp_baud.setValue(sc.baudrate)
        self.cb_bytesize.setCurrentText(str(sc.bytesize))
        self.cb_parity.setCurrentText(sc.parity)
        self.cb_stopbits.setCurrentText(str(int(sc.stopbits)) if sc.stopbits == int(sc.stopbits) else str(sc.stopbits))
        self.sp_timeout.setValue(sc.timeout_ms)
        self.le_trigger.setText(sc.trigger_hex)
        self.sp_readsize.setValue(sc.read_size)
        self.cb_encoding.setCurrentText(sc.encoding)
        self.sp_expected_len.setValue(sc.expected_length)
        self.le_contains.setText(sc.contains)
        if self._mgr.last_sn:
            self.lbl_last_sn.setText(f"Last SN: {self._mgr.last_sn}")
        # API GET
        g = self._mgr.api_get_cfg
        self.chk_get_enabled.setChecked(g.enabled)
        self.le_get_url.setText(g.url_template)
        self.le_get_headers.setText(g.headers_json)
        self.sp_get_timeout.setValue(g.timeout_ms)
        self.sp_get_status.setValue(g.expected_status)
        self.le_get_expected.setText(g.expected_text)
        self.chk_get_parse_json.setChecked(g.parse_json)
        self.le_get_json_path.setText(g.json_path)
        self.chk_get_bypass_proxy.setChecked(getattr(g, "bypass_proxy", True))
        # API POST
        p = self._mgr.api_post_cfg
        self.chk_post_enabled.setChecked(p.enabled)
        self.le_post_url.setText(p.url)
        self.cb_post_method.setCurrentText(p.method)
        self.le_post_headers.setText(p.headers_json)
        self.sp_post_timeout.setValue(p.timeout_ms)
        self.sp_post_status.setValue(p.expected_status)
        self.le_post_expected.setText(p.expected_text)
        self.chk_post_bypass_proxy.setChecked(getattr(p, "bypass_proxy", True))
        self.txt_post_body.setPlainText(p.body_template)

    def closeEvent(self, event):
        # Lưu auto khi đóng
        self._apply_to_manager()
        self._save_settings()
        super().closeEvent(event)
