"""
ui/plc_dialog.py — Dialog cấu hình & giám sát kết nối PLC

- Chọn loại PLC: Omron CP2E / Omron NX1P2 / Inovance H3U-H5U
- Cấu hình IP, port, polling interval
- Cấu hình vùng nhớ trigger (DM word / CIO bit / DM bit)
- Cấu hình vùng nhớ result (PASS/FAIL) + data
- Test connection, Read/Write thủ công, Start/Stop monitor
"""
from __future__ import annotations
from typing import Optional

import json

from PySide6.QtCore import Qt, Signal, QObject, QSettings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QCheckBox,
    QPlainTextEdit, QMessageBox, QWidget, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea,
)

from core.plc import (
    PLCManager, PLCConfig, MemoryArea, DRIVER_BY_MODEL, DataMapping,
    TriggerRoute,
)
from core.tool_registry import TOOL_BY_ID


_DATA_TYPES = ["int16", "int32", "float32", "scaled_int16", "scaled_int32"]


_AREA_LABELS = {
    "DM Word (D)":  MemoryArea.DM_WORD,
    "DM Bit":       MemoryArea.DM_BIT,
    "CIO Word":     MemoryArea.CIO_WORD,
    "CIO Bit":      MemoryArea.CIO_BIT,
    "Work Word":    MemoryArea.W_WORD,
    "Holding Word": MemoryArea.H_WORD,
}
_AREA_NAME_BY_ENUM = {v: k for k, v in _AREA_LABELS.items()}


class _ManagerBridge(QObject):
    """Bridge để callback từ thread monitor về Qt main thread.
    `trigger_fired` carries acquire_node_id ("" = run all pipeline)."""
    trigger_fired = Signal(str)
    error_occured = Signal(str)


class PLCDialog(QDialog):
    """Dialog cấu hình PLC. Phát signal ``trigger_fired`` khi PLC kích hoạt.
    Argument = acquire_node_id để route đến nhánh pipeline cụ thể; "" =
    chạy toàn pipeline (legacy single-trigger)."""

    trigger_fired = Signal(str)   # Forward ra MainWindow để chạy pipeline

    def __init__(self, manager: PLCManager, graph=None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("PLC Connection")
        # Window controls: minimize/maximize/close + tự maximize lần đầu mở
        # (giống SfcDialog) để không bị cắt nội dung trên màn nhỏ.
        self.setWindowFlags(
            Qt.Window
            | Qt.CustomizeWindowHint
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(900, 900)
        self._mgr = manager
        self._graph = graph
        self._bridge = _ManagerBridge()
        self._bridge.trigger_fired.connect(self._on_trigger_fired)
        self._bridge.error_occured.connect(self._log_error)

        self._build_ui()
        self._load_settings()
        self._refresh_status()

    def showEvent(self, event):
        """Tự maximize lần đầu mở."""
        super().showEvent(event)
        if not getattr(self, "_did_first_maximize", False):
            self._did_first_maximize = True
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def set_graph(self, graph) -> None:
        """Cập nhật reference đến FlowGraph hiện hành & refresh combo node."""
        self._graph = graph
        self._refresh_mapping_rows()
        self._refresh_route_rows()
        self._refresh_judge_combo()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Wrap nội dung trong QScrollArea — content nhiều section dài,
        # đảm bảo cuộn được trên màn hình bất kỳ.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body_w = QWidget(); scroll.setWidget(body_w)
        outer.addWidget(scroll)
        root = QVBoxLayout(body_w)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # — Connection group —
        gb_conn = QGroupBox("Connection")
        g = QGridLayout(gb_conn)

        self.cb_model = QComboBox()
        self.cb_model.addItems(list(DRIVER_BY_MODEL.keys()))
        self.cb_model.currentTextChanged.connect(self._on_model_changed)

        self.le_ip = QLineEdit("192.168.250.1")
        self.sp_port = QSpinBox(); self.sp_port.setRange(1, 65535); self.sp_port.setValue(9600)
        self.sp_poll = QSpinBox(); self.sp_poll.setRange(10, 5000); self.sp_poll.setValue(100); self.sp_poll.setSuffix(" ms")

        g.addWidget(QLabel("Model:"),    0, 0); g.addWidget(self.cb_model, 0, 1, 1, 3)
        g.addWidget(QLabel("IP:"),       1, 0); g.addWidget(self.le_ip,    1, 1)
        g.addWidget(QLabel("Port:"),     1, 2); g.addWidget(self.sp_port,  1, 3)
        g.addWidget(QLabel("Poll:"),     2, 0); g.addWidget(self.sp_poll,  2, 1)

        # FINS node addresses (chỉ ý nghĩa với NX1P2 — sẽ ẩn cho các model khác)
        self.sp_dest_node = QSpinBox(); self.sp_dest_node.setRange(0, 255); self.sp_dest_node.setValue(1)
        self.sp_src_node  = QSpinBox(); self.sp_src_node.setRange(0, 255); self.sp_src_node.setValue(25)
        self.lbl_dest_node = QLabel("FINS dest node:")
        self.lbl_src_node  = QLabel("FINS src node:")
        g.addWidget(self.lbl_dest_node, 2, 2); g.addWidget(self.sp_dest_node, 2, 3)
        g.addWidget(self.lbl_src_node,  3, 2); g.addWidget(self.sp_src_node,  3, 3)

        self.btn_connect = QPushButton("🔌  Connect")
        self.btn_disconnect = QPushButton("✖  Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        g.addWidget(self.btn_connect,    4, 0, 1, 2)
        g.addWidget(self.btn_disconnect, 4, 2, 1, 2)
        root.addWidget(gb_conn)

        # — Triggers group — bảng các trigger PLC độc lập, mỗi row có
        # Area/Address/Bit/Value/Acquire/Auto-clear riêng. Monitor poll
        # từng row độc lập, fire khi đọc khớp giá trị → chạy nhánh acquire
        # của row đó.
        gb_trig = QGroupBox(
            "Triggers (PLC → AOI) — mỗi row 1 trigger độc lập")
        gtl = QVBoxLayout(gb_trig)
        self.tbl_routes = QTableWidget(0, 7)
        self.tbl_routes.setHorizontalHeaderLabels(
            ["Area", "Address", "Bit", "Value", "Acquire Node",
             "Auto-clear", ""])
        hdr = self.tbl_routes.horizontalHeader()
        hdr.setStretchLastSection(False)
        for col in (0, 1, 2, 3, 5, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.tbl_routes.verticalHeader().setVisible(False)
        # Trong QScrollArea, table không tự expand → set min height + giữ
        # default row height đủ cao để cell widget (combo + spin) visible.
        self.tbl_routes.setMinimumHeight(180)
        self.tbl_routes.verticalHeader().setDefaultSectionSize(34)
        gtl.addWidget(self.tbl_routes, 1)

        rb = QHBoxLayout()
        self.btn_add_route = QPushButton("+ Add trigger")
        self.btn_add_route.clicked.connect(lambda: self._add_route_row())
        rb.addWidget(self.btn_add_route)
        rb.addStretch()
        self.btn_start = QPushButton("▶  Start monitor")
        self.btn_stop  = QPushButton("■  Stop monitor")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start_monitor)
        self.btn_stop.clicked.connect(self._on_stop_monitor)
        rb.addWidget(self.btn_start)
        rb.addWidget(self.btn_stop)
        gtl.addLayout(rb)
        root.addWidget(gb_trig)

        # — Result group —
        gb_res = QGroupBox("Result (AOI → PLC)")
        gr = QGridLayout(gb_res)

        self.cb_res_area = QComboBox(); self.cb_res_area.addItems(list(_AREA_LABELS.keys()))
        self.sp_res_addr = QSpinBox(); self.sp_res_addr.setRange(0, 0xFFFF); self.sp_res_addr.setValue(101)
        self.sp_pass_val = QSpinBox(); self.sp_pass_val.setRange(0, 0xFFFF); self.sp_pass_val.setValue(1)
        self.sp_fail_val = QSpinBox(); self.sp_fail_val.setRange(0, 0xFFFF); self.sp_fail_val.setValue(2)

        gr.addWidget(QLabel("Area:"),         0, 0); gr.addWidget(self.cb_res_area, 0, 1)
        gr.addWidget(QLabel("Address:"),      0, 2); gr.addWidget(self.sp_res_addr, 0, 3)
        gr.addWidget(QLabel("PASS value:"),   1, 0); gr.addWidget(self.sp_pass_val, 1, 1)
        gr.addWidget(QLabel("FAIL value:"),   1, 2); gr.addWidget(self.sp_fail_val, 1, 3)

        self.cb_data_area = QComboBox(); self.cb_data_area.addItems(list(_AREA_LABELS.keys()))
        self.sp_data_addr = QSpinBox(); self.sp_data_addr.setRange(0, 0xFFFF); self.sp_data_addr.setValue(110)
        gr.addWidget(QLabel("Default data area:"),  2, 0); gr.addWidget(self.cb_data_area, 2, 1)
        gr.addWidget(QLabel("Default start:"),      2, 2); gr.addWidget(self.sp_data_addr, 2, 3)

        self.cb_word_order = QComboBox(); self.cb_word_order.addItems(["ABCD (high word first)", "CDAB (low word first)"])
        gr.addWidget(QLabel("Float/int32 word order:"), 3, 0); gr.addWidget(self.cb_word_order, 3, 1, 1, 3)

        # Judge node — quyết định OK/NG dựa vào port "pass" của node nào.
        # "(Auto: all nodes pass)" = legacy: pass khi MỌI node trong pipeline pass.
        self.cb_judge_node = QComboBox()
        self.cb_judge_node.setToolTip(
            "Chọn node có port 'pass' để quyết định OK/NG gửi PLC. "
            "Auto = pass khi mọi node trong pipeline pass.")
        gr.addWidget(QLabel("Judge node:"), 4, 0)
        gr.addWidget(self.cb_judge_node, 4, 1, 1, 3)

        self.btn_send_pass = QPushButton("Send PASS (test)")
        self.btn_send_fail = QPushButton("Send FAIL (test)")
        self.btn_send_pass.clicked.connect(lambda: self._send_test_result(True))
        self.btn_send_fail.clicked.connect(lambda: self._send_test_result(False))
        gr.addWidget(self.btn_send_pass, 5, 0, 1, 2)
        gr.addWidget(self.btn_send_fail, 5, 2, 1, 2)
        root.addWidget(gb_res)

        # — Data mapping table (length, area, count… → PLC) —
        gb_map = QGroupBox("Data mappings (output của node → vùng nhớ PLC)")
        gm = QVBoxLayout(gb_map)

        self.tbl_map = QTableWidget(0, 7)
        self.tbl_map.setHorizontalHeaderLabels(
            ["Node", "Output", "Area", "Address", "Type", "Scale", ""])
        hdr = self.tbl_map.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (2, 3, 4, 5, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.tbl_map.verticalHeader().setVisible(False)
        self.tbl_map.setFixedHeight(180)
        gm.addWidget(self.tbl_map)

        hb_map = QHBoxLayout()
        self.btn_add_map  = QPushButton("➕  Add mapping")
        self.btn_test_map = QPushButton("⇡  Test send mappings")
        self.btn_refresh_map = QPushButton("↻  Refresh node list")
        self.btn_add_map.clicked.connect(lambda: self._add_mapping_row())
        self.btn_test_map.clicked.connect(self._test_send_mappings)
        self.btn_refresh_map.clicked.connect(self._refresh_mapping_rows)
        hb_map.addWidget(self.btn_add_map)
        hb_map.addWidget(self.btn_test_map)
        hb_map.addWidget(self.btn_refresh_map)
        hb_map.addStretch()
        gm.addLayout(hb_map)
        root.addWidget(gb_map)

        # — SFC trigger sequence — kết nối PLC trigger ↔ SFC workflow —
        gb_sfc = QGroupBox("SFC Trigger Sequence (PLC trigger → Scan → GET → Pipeline → POST)")
        gs = QGridLayout(gb_sfc)

        self.chk_sfc_seq_enabled = QCheckBox(
            "Enable SFC sequence khi PLC trigger")
        self.chk_sfc_seq_enabled.setToolTip(
            "Bật: PLC fire trigger DM → tự động chạy Scan → API GET → "
            "Pipeline → API POST.\n"
            "Tắt: PLC trigger chỉ chạy Pipeline (behaviour cũ).")
        gs.addWidget(self.chk_sfc_seq_enabled, 0, 0, 1, 4)

        self.chk_sfc_step_scan = QCheckBox("• Bước 1: Scan barcode/QR (Scanner section của SFC)")
        self.chk_sfc_step_scan.setChecked(True)
        self.chk_sfc_step_scan.setToolTip(
            "Bỏ tích nếu PLC trigger không cần scan (vd SN đã có sẵn từ camera/QR khác).")
        gs.addWidget(self.chk_sfc_step_scan, 1, 0, 1, 4)

        self.chk_sfc_step_get = QCheckBox("• Bước 2: API GET (check SN trên SFC)")
        self.chk_sfc_step_get.setChecked(True)
        self.chk_sfc_step_get.setToolTip(
            "Gọi API GET với {SN} từ bước scan. Tắt nếu không cần lookup trước "
            "khi chạy pipeline.")
        gs.addWidget(self.chk_sfc_step_get, 2, 0, 1, 4)

        self.chk_sfc_step_post = QCheckBox("• Bước 4: API POST (gửi kết quả sau pipeline)")
        self.chk_sfc_step_post.setChecked(True)
        self.chk_sfc_step_post.setToolTip(
            "Sau khi pipeline run xong → tự động POST kết quả với body template "
            "đã cấu hình. Tắt nếu chỉ muốn lưu local.")
        gs.addWidget(self.chk_sfc_step_post, 3, 0, 1, 4)

        self.chk_sfc_abort_on_fail = QCheckBox(
            "Dừng sequence nếu Scan/GET fail (không chạy pipeline & POST)")
        self.chk_sfc_abort_on_fail.setChecked(True)
        self.chk_sfc_abort_on_fail.setToolTip(
            "Bật: scan timeout / GET trả status ≠ expected → abort, không "
            "chạy pipeline.\n"
            "Tắt: vẫn chạy pipeline kể cả khi scan/GET fail.")
        gs.addWidget(self.chk_sfc_abort_on_fail, 4, 0, 1, 4)

        btn_open_sfc = QPushButton("⚙  Open SFC configuration…")
        btn_open_sfc.setToolTip("Mở dialog cấu hình Scanner / API GET / API POST")
        btn_open_sfc.clicked.connect(self._on_open_sfc_dialog)
        gs.addWidget(btn_open_sfc, 5, 0, 1, 4)

        root.addWidget(gb_sfc)

        # — Log —
        self.lbl_status = QLabel("● Disconnected")
        self.lbl_status.setStyleSheet("color:#ff3860;font-weight:700;")
        root.addWidget(self.lbl_status)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(140)
        root.addWidget(self.log)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        root.addWidget(btn_close)

    # ── Mapping table ─────────────────────────────────────────────
    def _node_options(self) -> list:
        """Trả về list (display_text, node_id) cho combo Node."""
        if not self._graph:
            return []
        opts = []
        for nid, node in self._graph.nodes.items():
            tool = TOOL_BY_ID.get(node.tool_id)
            label = tool.name if tool else node.tool_id
            opts.append((f"{label}  [{nid}]", nid))
        return opts

    def _acquire_node_options(self) -> list:
        """List (label, node_id) cho route picker — chỉ node category
        'Acquire Image' (acquire_image / camera_acquire / ...)."""
        if not self._graph:
            return []
        opts = []
        for nid, node in self._graph.nodes.items():
            tool = TOOL_BY_ID.get(node.tool_id)
            if not tool or getattr(tool, "category", "") != "Acquire Image":
                continue
            opts.append((f"{tool.name}  [{nid}]", nid))
        return opts

    def _judge_node_options(self) -> list:
        """List (label, node_id) cho judge combo — chỉ node có output port
        'pass' (judge / compare / blob / find_circle / ...)."""
        if not self._graph:
            return []
        opts = []
        for nid, node in self._graph.nodes.items():
            tool = TOOL_BY_ID.get(node.tool_id)
            if not tool:
                continue
            has_pass = any(p.name == "pass" for p in tool.outputs)
            if has_pass:
                opts.append((f"{tool.name}  [{nid}]", nid))
        return opts

    def _output_options(self, node_id: str) -> list:
        """Trả về list output name của node hiện tại."""
        if not self._graph or node_id not in self._graph.nodes:
            return []
        node = self._graph.nodes[node_id]
        tool = TOOL_BY_ID.get(node.tool_id)
        names = []
        if tool:
            for port in tool.outputs:
                if port.data_type != "image":   # bỏ image, chỉ numeric
                    names.append(port.name)
        # Bổ sung key đang có trong outputs runtime mà tool không khai báo
        for k, v in node.outputs.items():
            if k not in names and isinstance(v, (int, float)) and not isinstance(v, bool):
                names.append(k)
        return names

    def _add_mapping_row(self, mapping: Optional[DataMapping] = None):
        row = self.tbl_map.rowCount()
        self.tbl_map.insertRow(row)

        cb_node = QComboBox()
        cb_out  = QComboBox()
        cb_area = QComboBox(); cb_area.addItems(list(_AREA_LABELS.keys()))
        sp_addr = QSpinBox(); sp_addr.setRange(0, 0xFFFF)
        cb_type = QComboBox(); cb_type.addItems(_DATA_TYPES)
        sp_scale = QDoubleSpinBox(); sp_scale.setDecimals(4); sp_scale.setRange(-1e6, 1e6); sp_scale.setValue(1.0)
        btn_del = QPushButton("✖")

        # Populate node combo
        for txt, nid in self._node_options():
            cb_node.addItem(txt, userData=nid)

        def _refresh_outputs():
            nid = cb_node.currentData()
            current = cb_out.currentText()
            cb_out.clear()
            cb_out.addItems(self._output_options(nid) if nid else [])
            idx = cb_out.findText(current)
            if idx >= 0:
                cb_out.setCurrentIndex(idx)

        cb_node.currentIndexChanged.connect(lambda _: _refresh_outputs())
        _refresh_outputs()

        # Apply saved mapping values if provided
        if mapping:
            idx = cb_node.findData(mapping.node_id)
            if idx >= 0:
                cb_node.setCurrentIndex(idx)
            _refresh_outputs()
            # Cho phép giữ output_key cũ kể cả khi node hiện tại không có
            if cb_out.findText(mapping.output_key) < 0 and mapping.output_key:
                cb_out.addItem(mapping.output_key)
            cb_out.setCurrentText(mapping.output_key)
            cb_area.setCurrentText(_AREA_NAME_BY_ENUM.get(mapping.area, list(_AREA_LABELS.keys())[0]))
            sp_addr.setValue(mapping.address)
            cb_type.setCurrentText(mapping.data_type)
            sp_scale.setValue(mapping.scale)

        btn_del.setFixedWidth(28)
        btn_del.clicked.connect(lambda: self._remove_mapping_row(btn_del))

        self.tbl_map.setCellWidget(row, 0, cb_node)
        self.tbl_map.setCellWidget(row, 1, cb_out)
        self.tbl_map.setCellWidget(row, 2, cb_area)
        self.tbl_map.setCellWidget(row, 3, sp_addr)
        self.tbl_map.setCellWidget(row, 4, cb_type)
        self.tbl_map.setCellWidget(row, 5, sp_scale)
        self.tbl_map.setCellWidget(row, 6, btn_del)

    def _remove_mapping_row(self, btn: QPushButton):
        for row in range(self.tbl_map.rowCount()):
            if self.tbl_map.cellWidget(row, 6) is btn:
                self.tbl_map.removeRow(row)
                return

    # ── Triggers (multi-row) ──
    def _add_route_row(self, route: Optional[TriggerRoute] = None):
        row = self.tbl_routes.rowCount()
        self.tbl_routes.insertRow(row)
        self.tbl_routes.setRowHeight(row, 34)
        # Area combo
        cb_area = QComboBox(); cb_area.addItems(list(_AREA_LABELS.keys()))
        if route:
            cb_area.setCurrentText(_AREA_NAME_BY_ENUM.get(
                route.area, list(_AREA_LABELS.keys())[0]))
        # Address
        sp_addr = QSpinBox(); sp_addr.setRange(0, 0xFFFF)
        sp_addr.setValue(route.address if route else 100)
        # Bit (chỉ ý nghĩa khi area là *_BIT)
        sp_bit = QSpinBox(); sp_bit.setRange(0, 15)
        sp_bit.setValue(route.bit if route else 0)
        # Value (word so sánh; bit thì luôn so 1)
        sp_val = QSpinBox(); sp_val.setRange(0, 0xFFFF)
        sp_val.setValue(route.value if route else 1)
        # Acquire combo
        cb_acq = QComboBox()
        cb_acq.addItem("(All Acquire — toàn pipeline)", userData="")
        for txt, nid in self._acquire_node_options():
            cb_acq.addItem(txt, userData=nid)
        if route and route.acquire_node_id:
            idx = cb_acq.findData(route.acquire_node_id)
            if idx >= 0:
                cb_acq.setCurrentIndex(idx)
            else:
                cb_acq.addItem(f"(missing) [{route.acquire_node_id}]",
                               userData=route.acquire_node_id)
                cb_acq.setCurrentIndex(cb_acq.count() - 1)
        # Auto-clear
        chk_auto = QCheckBox()
        chk_auto.setChecked(bool(route.auto_clear) if route else True)
        # Delete
        btn_del = QPushButton("✖"); btn_del.setFixedWidth(28)
        btn_del.clicked.connect(lambda: self._remove_route_row(btn_del))

        self.tbl_routes.setCellWidget(row, 0, cb_area)
        self.tbl_routes.setCellWidget(row, 1, sp_addr)
        self.tbl_routes.setCellWidget(row, 2, sp_bit)
        self.tbl_routes.setCellWidget(row, 3, sp_val)
        self.tbl_routes.setCellWidget(row, 4, cb_acq)
        self.tbl_routes.setCellWidget(row, 5, chk_auto)
        self.tbl_routes.setCellWidget(row, 6, btn_del)

    def _remove_route_row(self, btn: QPushButton):
        for row in range(self.tbl_routes.rowCount()):
            if self.tbl_routes.cellWidget(row, 6) is btn:
                self.tbl_routes.removeRow(row)
                return

    def _read_routes(self) -> list:
        out = []
        for row in range(self.tbl_routes.rowCount()):
            cb_area = self.tbl_routes.cellWidget(row, 0)
            sp_addr = self.tbl_routes.cellWidget(row, 1)
            sp_bit  = self.tbl_routes.cellWidget(row, 2)
            sp_val  = self.tbl_routes.cellWidget(row, 3)
            cb_acq  = self.tbl_routes.cellWidget(row, 4)
            chk     = self.tbl_routes.cellWidget(row, 5)
            if not (cb_area and sp_addr and sp_val and cb_acq):
                continue
            out.append(TriggerRoute(
                area=_AREA_LABELS[cb_area.currentText()],
                address=int(sp_addr.value()),
                bit=int(sp_bit.value()) if sp_bit else 0,
                value=int(sp_val.value()),
                auto_clear=bool(chk.isChecked()) if chk else True,
                acquire_node_id=str(cb_acq.currentData() or ""),
            ))
        return out

    def _refresh_route_rows(self):
        existing = self._read_routes()
        self.tbl_routes.setRowCount(0)
        for r in existing:
            self._add_route_row(r)

    def _refresh_judge_combo(self):
        """Populate cb_judge_node với list node có port 'pass'. Giữ
        selection hiện tại nếu node còn trong graph."""
        prev = self.cb_judge_node.currentData() if self.cb_judge_node.count() else ""
        self.cb_judge_node.clear()
        self.cb_judge_node.addItem("(Auto: all nodes pass)", userData="")
        for txt, nid in self._judge_node_options():
            self.cb_judge_node.addItem(txt, userData=nid)
        if prev:
            idx = self.cb_judge_node.findData(prev)
            if idx >= 0:
                self.cb_judge_node.setCurrentIndex(idx)
            else:
                self.cb_judge_node.addItem(f"(missing) [{prev}]", userData=prev)
                self.cb_judge_node.setCurrentIndex(self.cb_judge_node.count() - 1)

    def _read_mappings(self) -> list:
        """Đọc bảng → list[DataMapping]."""
        out = []
        for row in range(self.tbl_map.rowCount()):
            cb_node = self.tbl_map.cellWidget(row, 0)
            cb_out  = self.tbl_map.cellWidget(row, 1)
            cb_area = self.tbl_map.cellWidget(row, 2)
            sp_addr = self.tbl_map.cellWidget(row, 3)
            cb_type = self.tbl_map.cellWidget(row, 4)
            sp_scale = self.tbl_map.cellWidget(row, 5)
            nid = cb_node.currentData() if cb_node else None
            okey = cb_out.currentText() if cb_out else ""
            if not nid or not okey:
                continue
            out.append(DataMapping(
                node_id=nid,
                output_key=okey,
                area=_AREA_LABELS[cb_area.currentText()],
                address=sp_addr.value(),
                data_type=cb_type.currentText(),
                scale=sp_scale.value(),
            ))
        return out

    def _refresh_mapping_rows(self):
        """Refresh combo node ở mỗi row khi graph thay đổi (giữ giá trị hiện tại)."""
        existing = self._read_mappings()
        self.tbl_map.setRowCount(0)
        for m in existing:
            self._add_mapping_row(m)

    def _test_send_mappings(self):
        """Đọc graph hiện tại + gửi mappings về PLC để kiểm tra."""
        if not self._mgr.is_connected:
            QMessageBox.warning(self, "Not connected", "Connect PLC trước đã.")
            return
        if self._graph is None:
            QMessageBox.warning(self, "No graph", "Không có pipeline.")
            return
        self._mgr.config = self._gather_config()
        # results dict: {node_id: outputs_dict}
        results = {nid: dict(n.outputs) for nid, n in self._graph.nodes.items()}
        try:
            report = self._mgr.write_data_mappings(results)
        except Exception as e:
            QMessageBox.critical(self, "Write failed", str(e))
            return
        for r in report:
            if "error" in r:
                self._log(f"✗ {r['node_id']}.{r['output_key']} @ {r['address']} — {r['error']}")
            else:
                self._log(f"✓ {r['node_id']}.{r['output_key']} = {r['value']} → {r['address']} [{r['data_type']}]")

    # ── Helpers ───────────────────────────────────────────────────
    def _on_model_changed(self, model: str):
        is_nx = "NX1P2" in model
        for w in (self.lbl_dest_node, self.sp_dest_node, self.lbl_src_node, self.sp_src_node):
            w.setVisible(is_nx)
        if "NX1P2" in model:
            self.sp_port.setValue(9600)
        elif "CP2E" in model:
            self.sp_port.setValue(9600)
        elif "H3U" in model or "H5U" in model:
            self.sp_port.setValue(502)

    def _gather_config(self) -> PLCConfig:
        return PLCConfig(
            model=self.cb_model.currentText(),
            ip=self.le_ip.text().strip(),
            port=self.sp_port.value(),
            poll_interval_ms=self.sp_poll.value(),
            trigger_routes=self._read_routes(),
            result_area=_AREA_LABELS[self.cb_res_area.currentText()],
            result_address=self.sp_res_addr.value(),
            result_pass_value=self.sp_pass_val.value(),
            result_fail_value=self.sp_fail_val.value(),
            result_judge_node_id=str(self.cb_judge_node.currentData() or ""),
            data_area=_AREA_LABELS[self.cb_data_area.currentText()],
            data_start_address=self.sp_data_addr.value(),
            float_word_order="ABCD" if self.cb_word_order.currentIndex() == 0 else "CDAB",
            data_mappings=self._read_mappings(),
            fins_dest_node=self.sp_dest_node.value(),
            fins_src_node=self.sp_src_node.value(),
        )

    def _apply_config_to_ui(self, cfg: PLCConfig):
        idx = self.cb_model.findText(cfg.model)
        if idx >= 0: self.cb_model.setCurrentIndex(idx)
        self.le_ip.setText(cfg.ip)
        self.sp_port.setValue(cfg.port)
        self.sp_poll.setValue(cfg.poll_interval_ms)
        self.cb_res_area.setCurrentText(_AREA_NAME_BY_ENUM[cfg.result_area])
        self.sp_res_addr.setValue(cfg.result_address)
        self.sp_pass_val.setValue(cfg.result_pass_value)
        self.sp_fail_val.setValue(cfg.result_fail_value)
        self.cb_data_area.setCurrentText(_AREA_NAME_BY_ENUM[cfg.data_area])
        self.sp_data_addr.setValue(cfg.data_start_address)
        self.cb_word_order.setCurrentIndex(0 if cfg.float_word_order == "ABCD" else 1)
        self.sp_dest_node.setValue(cfg.fins_dest_node)
        self.sp_src_node.setValue(cfg.fins_src_node)
        self._on_model_changed(cfg.model)
        self.tbl_map.setRowCount(0)
        for m in cfg.data_mappings:
            self._add_mapping_row(m)
        # Trigger routes + judge node
        self.tbl_routes.setRowCount(0)
        for r in cfg.trigger_routes:
            self._add_route_row(r)
        self._refresh_judge_combo()
        if cfg.result_judge_node_id:
            idx = self.cb_judge_node.findData(cfg.result_judge_node_id)
            if idx >= 0:
                self.cb_judge_node.setCurrentIndex(idx)
            else:
                self.cb_judge_node.addItem(
                    f"(missing) [{cfg.result_judge_node_id}]",
                    userData=cfg.result_judge_node_id)
                self.cb_judge_node.setCurrentIndex(self.cb_judge_node.count() - 1)

    def _refresh_status(self):
        if self._mgr.is_connected:
            self.lbl_status.setText(
                f"● Connected to {self._mgr.config.model} @ {self._mgr.config.ip}"
                + ("  [monitoring]" if self._mgr.is_monitoring else ""))
            self.lbl_status.setStyleSheet("color:#39ff14;font-weight:700;")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.btn_start.setEnabled(not self._mgr.is_monitoring)
            self.btn_stop.setEnabled(self._mgr.is_monitoring)
        else:
            self.lbl_status.setText("● Disconnected")
            self.lbl_status.setStyleSheet("color:#ff3860;font-weight:700;")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(False)

    def _log(self, msg: str):
        from datetime import datetime
        self.log.appendPlainText(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _log_error(self, msg: str):
        self._log("ERROR: " + msg)

    # ── Actions ───────────────────────────────────────────────────
    def _on_connect(self):
        self._mgr.config = self._gather_config()
        try:
            self._mgr.connect()
            self._log(f"Connected to {self._mgr.config.model} @ {self._mgr.config.ip}:{self._mgr.config.port}")
        except Exception as e:
            QMessageBox.critical(self, "Connect failed", str(e))
            self._log(f"Connect failed: {e}")
        self._refresh_status()
        self._save_settings()

    def _on_disconnect(self):
        self._mgr.disconnect()
        self._log("Disconnected")
        self._refresh_status()

    def _on_start_monitor(self):
        # Cập nhật config (đặc biệt là trigger fields) trước khi start
        self._mgr.config = self._gather_config()
        try:
            self._mgr.start_monitor(
                on_trigger=self._bridge.trigger_fired.emit,
                on_error=self._bridge.error_occured.emit,
            )
            cfg = self._mgr.config
            n = len(cfg.trigger_routes)
            if n == 0:
                self._log("Monitor started — no triggers configured "
                          "(add triggers in table to receive PLC events).")
            else:
                self._log(f"Monitoring {n} trigger(s) — poll every "
                          f"{cfg.poll_interval_ms}ms")
        except Exception as e:
            QMessageBox.critical(self, "Start monitor failed", str(e))
        self._refresh_status()
        self._save_settings()

    def _on_stop_monitor(self):
        self._mgr.stop_monitor()
        self._log("Monitor stopped")
        self._refresh_status()

    def _on_trigger_fired(self, acquire_node_id: str = ""):
        if acquire_node_id:
            self._log(f"⚡ Trigger received → run acquire [{acquire_node_id}]")
        else:
            self._log("⚡ Trigger received → run pipeline")
        self.trigger_fired.emit(acquire_node_id)

    def _send_test_result(self, passed: bool):
        self._mgr.config = self._gather_config()
        try:
            self._mgr.write_result(passed=passed)
            self._log(f"Sent {'PASS' if passed else 'FAIL'} → "
                      f"{self._mgr.config.result_area.value}{self._mgr.config.result_address}")
        except Exception as e:
            QMessageBox.critical(self, "Write failed", str(e))

    # ── SFC integration ───────────────────────────────────────────
    def _on_open_sfc_dialog(self):
        """Yêu cầu MainWindow mở SfcDialog. Phát signal lên parent — parent
        đã có _open_sfc_dialog wired ở menu Tools."""
        parent = self.parent()
        if parent is not None and hasattr(parent, "_open_sfc_dialog"):
            parent._open_sfc_dialog()
        else:
            QMessageBox.information(self, "SFC",
                "Mở Tools → SFC / MES Integration… từ main window để cấu hình.")

    def get_sfc_sequence_settings(self) -> dict:
        """Trả về settings SFC sequence cho MainWindow đọc khi PLC trigger fire."""
        return {
            "enabled":       self.chk_sfc_seq_enabled.isChecked(),
            "step_scan":     self.chk_sfc_step_scan.isChecked(),
            "step_get":      self.chk_sfc_step_get.isChecked(),
            "step_post":     self.chk_sfc_step_post.isChecked(),
            "abort_on_fail": self.chk_sfc_abort_on_fail.isChecked(),
        }

    # ── Persistence ───────────────────────────────────────────────
    def _save_settings(self):
        s = QSettings()
        s.beginGroup("plc")
        cfg = self._gather_config()
        s.setValue("model", cfg.model)
        s.setValue("ip", cfg.ip)
        s.setValue("port", cfg.port)
        s.setValue("poll", cfg.poll_interval_ms)
        s.setValue("res_area", cfg.result_area.value)
        s.setValue("res_addr", cfg.result_address)
        s.setValue("pass_val", cfg.result_pass_value)
        s.setValue("fail_val", cfg.result_fail_value)
        s.setValue("data_area", cfg.data_area.value)
        s.setValue("data_addr", cfg.data_start_address)
        s.setValue("word_order", cfg.float_word_order)
        s.setValue("mappings", json.dumps([m.to_dict() for m in cfg.data_mappings]))
        s.setValue("trig_routes",
                   json.dumps([r.to_dict() for r in cfg.trigger_routes]))
        s.setValue("judge_node", cfg.result_judge_node_id)
        s.setValue("fins_dest", cfg.fins_dest_node)
        s.setValue("fins_src",  cfg.fins_src_node)
        # SFC sequence settings (lưu cùng group "plc")
        seq = self.get_sfc_sequence_settings()
        s.setValue("sfc_seq_enabled",     seq["enabled"])
        s.setValue("sfc_seq_step_scan",   seq["step_scan"])
        s.setValue("sfc_seq_step_get",    seq["step_get"])
        s.setValue("sfc_seq_step_post",   seq["step_post"])
        s.setValue("sfc_seq_abort_fail",  seq["abort_on_fail"])
        s.endGroup()

    def _load_settings(self):
        s = QSettings()
        s.beginGroup("plc")
        try:
            # Read trig_routes; bỏ entry old-format (chỉ có trigger_value
            # + acquire_node_id, thiếu "area") — đó là test data từ schema
            # cũ trước khi multi-trigger có Area/Address/Bit. Tránh user
            # thấy "nhiều trigger lạ" khi mới mở dialog.
            try:
                _raw_routes = json.loads(s.value("trig_routes", "[]") or "[]")
                if not isinstance(_raw_routes, list):
                    _raw_routes = []
                _raw_routes = [d for d in _raw_routes
                               if isinstance(d, dict) and "area" in d]
            except Exception:
                _raw_routes = []
            cfg = PLCConfig(
                model=s.value("model", "Omron CP2E"),
                ip=s.value("ip", "192.168.250.1"),
                port=int(s.value("port", 9600)),
                poll_interval_ms=int(s.value("poll", 100)),
                result_area=MemoryArea(s.value("res_area", MemoryArea.DM_WORD.value)),
                result_address=int(s.value("res_addr", 101)),
                result_pass_value=int(s.value("pass_val", 1)),
                result_fail_value=int(s.value("fail_val", 2)),
                data_area=MemoryArea(s.value("data_area", MemoryArea.DM_WORD.value)),
                data_start_address=int(s.value("data_addr", 110)),
                float_word_order=s.value("word_order", "ABCD"),
                data_mappings=[
                    DataMapping.from_dict(d)
                    for d in json.loads(s.value("mappings", "[]") or "[]")
                ],
                trigger_routes=[
                    TriggerRoute.from_dict(d) for d in _raw_routes
                ],
                result_judge_node_id=str(s.value("judge_node", "") or ""),
                fins_dest_node=int(s.value("fins_dest", 1)),
                fins_src_node=int(s.value("fins_src", 25)),
            )
            # Migration: nếu setting cũ có trig_addr/trig_val (single
            # trigger) mà chưa có trig_routes → tạo 1 row từ giá trị cũ
            # để không mất config khi update lên multi-trigger.
            if not cfg.trigger_routes and s.contains("trig_addr"):
                try:
                    cfg.trigger_routes = [TriggerRoute(
                        area=MemoryArea(s.value("trig_area",
                                                MemoryArea.DM_WORD.value)),
                        address=int(s.value("trig_addr", 100)),
                        bit=int(s.value("trig_bit", 0)),
                        value=int(s.value("trig_val", 1)),
                        auto_clear=s.value("auto_clr", True, type=bool),
                        acquire_node_id="",
                    )]
                except Exception:
                    pass
            # Empty sau cùng (fresh install hoặc old-format đã bị filter
            # ở bước parse) → khởi tạo 1 row default để user có template
            # sẵn, không phải click "+ Add trigger" mới có chỗ nhập.
            if not cfg.trigger_routes:
                cfg.trigger_routes = [TriggerRoute()]
            self._apply_config_to_ui(cfg)
        except Exception:
            pass
        # SFC sequence settings
        try:
            self.chk_sfc_seq_enabled.setChecked(
                s.value("sfc_seq_enabled", False, type=bool))
            self.chk_sfc_step_scan.setChecked(
                s.value("sfc_seq_step_scan", True, type=bool))
            self.chk_sfc_step_get.setChecked(
                s.value("sfc_seq_step_get", True, type=bool))
            self.chk_sfc_step_post.setChecked(
                s.value("sfc_seq_step_post", True, type=bool))
            self.chk_sfc_abort_on_fail.setChecked(
                s.value("sfc_seq_abort_fail", True, type=bool))
        except Exception:
            pass
        s.endGroup()

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)
