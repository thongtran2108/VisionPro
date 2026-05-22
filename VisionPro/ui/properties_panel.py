"""
ui/properties_panel.py  — FIXED
Fix: QScrollArea.setWidget() xóa widget cũ → không tái dùng placeholder.
     Thay bằng _make_placeholder() tạo mới mỗi lần cần.
"""
from __future__ import annotations
from typing import Optional, Any, Tuple

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QLineEdit, QSpinBox, QDoubleSpinBox,
                                QComboBox, QCheckBox, QPushButton, QSlider,
                                QScrollArea, QFrame, QTabWidget, QFileDialog,
                                QDialog, QDialogButtonBox)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

from core.flow_graph import FlowGraph, NodeInstance
from core.tool_registry import ToolDef, ParamDef


# ── ROI param set detection ───────────────────────────────────────────
def _detect_roi_kind(tool: ToolDef) -> Optional[str]:
    """Return ROI kind cho tool dựa trên tên params:
       - "rect_xywh"      : params có x, y, w, h
       - "rect_x1y1x2y2"  : params có x1, y1, x2, y2
       - "point_pick"     : params có pick_x, pick_y
       - None             : không phải tool có ROI vẽ được
    """
    names = {p.name for p in tool.params}
    if {"x", "y", "w", "h"} <= names:
        return "rect_xywh"
    if {"x1", "y1", "x2", "y2"} <= names:
        return "rect_x1y1x2y2"
    if {"pick_x", "pick_y"} <= names:
        return "point_pick"
    return None


def _make_placeholder(text: str) -> QLabel:
    """Tạo QLabel placeholder MỚI mỗi lần — tránh C++ deleted object crash."""
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color:#1e2d45; font-size:13px; padding:20px;")
    return lbl


class ParamRow(QWidget):
    value_changed = Signal(str, object)

    def __init__(self, param: ParamDef, current_value: Any, parent=None):
        super().__init__(parent)
        self.param = param
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        lbl = QLabel(param.label)
        lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(110)
        lbl.setMaximumWidth(130)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self._editor = self._build_editor(param, current_value)
        lay.addWidget(self._editor, 1)

    def _build_editor(self, p: ParamDef, val: Any) -> QWidget:
        if p.ptype == "bool":
            w = QCheckBox()
            w.setChecked(bool(val))
            w.stateChanged.connect(lambda s: self.value_changed.emit(p.name, bool(s)))
            return w

        if p.ptype == "enum":
            w = QComboBox()
            w.addItems(p.choices)
            if str(val) in p.choices:
                w.setCurrentText(str(val))
            w.currentTextChanged.connect(lambda t: self.value_changed.emit(p.name, t))
            return w

        if p.ptype == "int":
            sb = QSpinBox()
            sb.setMinimum(int(p.min_val) if p.min_val is not None else -999999)
            sb.setMaximum(int(p.max_val) if p.max_val is not None else  999999)
            sb.setSingleStep(int(p.step) if p.step else 1)
            sb.setValue(int(val) if val is not None else 0)
            sb.valueChanged.connect(lambda v: self.value_changed.emit(p.name, v))
            if getattr(p, "use_slider", False) and p.min_val is not None and p.max_val is not None:
                return self._wrap_slider(sb, int(p.min_val), int(p.max_val), is_float=False)
            return sb

        if p.ptype == "float":
            sb = QDoubleSpinBox()
            sb.setMinimum(float(p.min_val) if p.min_val is not None else -1e9)
            sb.setMaximum(float(p.max_val) if p.max_val is not None else  1e9)
            sb.setSingleStep(float(p.step) if p.step else 0.1)
            sb.setDecimals(4)
            sb.setValue(float(val) if val is not None else 0.0)
            sb.valueChanged.connect(lambda v: self.value_changed.emit(p.name, v))
            if getattr(p, "use_slider", False) and p.min_val is not None and p.max_val is not None:
                return self._wrap_slider(sb, float(p.min_val), float(p.max_val), is_float=True)
            return sb

        # str
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        le = QLineEdit(str(val) if val else "")
        le.setPlaceholderText(p.label)
        le.textChanged.connect(lambda t: self.value_changed.emit(p.name, t))
        hl.addWidget(le)
        if "path" in p.name.lower():
            is_folder = "folder" in p.name.lower() or "dir" in p.name.lower()
            btn = QPushButton("📂" if is_folder else "…")
            btn.setFixedWidth(28)
            btn.setStyleSheet(
                "QPushButton{background:#1e2d45;border:none;border-radius:3px;color:#e2e8f0;}"
                "QPushButton:hover{background:#00d4ff;color:#000;}")
            if is_folder:
                btn.clicked.connect(lambda: self._browse_folder(le))
            else:
                file_filter = getattr(p, "file_filter", None)
                btn.clicked.connect(lambda _=False, f=file_filter:
                                     self._browse_file(le, f))
            hl.addWidget(btn)
        return w

    def _wrap_slider(self, spin: QWidget, lo, hi, is_float: bool) -> QWidget:
        """Combine slider + spinbox; expose setValue/value on the wrapper
        so callers using `pr._editor.setValue()` keep working."""
        FLOAT_SCALE = 1000
        sl = QSlider(Qt.Horizontal)
        if is_float:
            sl.setRange(int(lo * FLOAT_SCALE), int(hi * FLOAT_SCALE))
            sl.setValue(int(spin.value() * FLOAT_SCALE))
        else:
            sl.setRange(int(lo), int(hi))
            sl.setValue(int(spin.value()))
        sl.setMinimumWidth(80)

        def _slider_to_spin(v):
            spin.blockSignals(True)
            spin.setValue(v / FLOAT_SCALE if is_float else v)
            spin.blockSignals(False)
            spin.valueChanged.emit(spin.value())

        def _spin_to_slider(v):
            sl.blockSignals(True)
            sl.setValue(int(v * FLOAT_SCALE) if is_float else int(v))
            sl.blockSignals(False)

        sl.valueChanged.connect(_slider_to_spin)
        spin.valueChanged.connect(_spin_to_slider)

        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        hl.addWidget(sl, 1)
        spin.setMinimumWidth(72 if is_float else 56)
        spin.setMaximumWidth(96 if is_float else 70)
        hl.addWidget(spin)

        # Proxy setValue/value so external code (sync spinbox) still works.
        w.setValue = spin.setValue
        w.value = spin.value
        w.blockSignals = lambda b: (sl.blockSignals(b), spin.blockSignals(b))
        return w

    def _browse_file(self, le: QLineEdit, file_filter: Optional[str] = None):
        flt = file_filter or (
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", le.text() or "", flt)
        if path:
            le.setText(path)
            # Prefetch chỉ cho file ảnh (không áp dụng cho model .pt/.onnx).
            if file_filter is None:
                try:
                    from core.tool_registry import acquire_prefetch
                    acquire_prefetch(path)
                except Exception:
                    pass

    def _browse_folder(self, le: QLineEdit):
        path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", le.text() or "")
        if path:
            le.setText(path)
            # Prefetch vài file đầu tiên trong folder để Run đầu nhanh.
            try:
                import os
                from core.tool_registry import (acquire_prefetch,
                                                  _PREFETCH_DEPTH)
                exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                files = sorted(
                    f for f in os.listdir(path)
                    if f.lower().endswith(exts)
                    and os.path.isfile(os.path.join(path, f))
                )[:_PREFETCH_DEPTH + 1]
                acquire_prefetch([os.path.join(path, f) for f in files])
            except Exception:
                pass


class NodeInfoWidget(QWidget):
    def __init__(self, node: NodeInstance, parent=None):
        super().__init__(parent)
        tool: ToolDef = node.tool
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        icon_lbl = QLabel(tool.icon)
        icon_lbl.setStyleSheet(
            f"font-size:24px; background:{tool.color};"
            f"border-radius:6px; padding:4px 8px;")
        icon_lbl.setFixedSize(44, 44)
        icon_lbl.setAlignment(Qt.AlignCenter)
        top.addWidget(icon_lbl)

        info_lay = QVBoxLayout()
        name_lbl = QLabel(tool.name)
        name_lbl.setStyleSheet("color:#e2e8f0; font-size:14px; font-weight:700;")
        cat_lbl = QLabel(f"Category: {tool.category}")
        cat_lbl.setStyleSheet("color:#64748b; font-size:11px;")
        id_lbl = QLabel(f"Node ID: {node.node_id}")
        id_lbl.setStyleSheet("color:#1e2d45; font-size:10px; font-family:'Courier New';")
        info_lay.addWidget(name_lbl)
        info_lay.addWidget(cat_lbl)
        info_lay.addWidget(id_lbl)
        top.addLayout(info_lay, 1)
        lay.addLayout(top)

        desc = QLabel(tool.description)
        desc.setStyleSheet("color:#94a3b8; font-size:11px; padding:6px 0;")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)

        ports_lbl = QLabel("PORTS")
        ports_lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:1.5px;")
        lay.addWidget(ports_lbl)

        for p in tool.inputs:
            r = QLabel(f"  ⬤  IN  •  {p.name}  [{p.data_type}]"
                       f"{'  (opt)' if not p.required else ''}")
            r.setStyleSheet("color:#00b4d8; font-size:11px; font-family:'Courier New';")
            lay.addWidget(r)
        for p in tool.outputs:
            r = QLabel(f"  ⬤  OUT  •  {p.name}  [{p.data_type}]")
            r.setStyleSheet("color:#ff8c42; font-size:11px; font-family:'Courier New';")
            lay.addWidget(r)

        lay.addStretch()


class OutputsWidget(QWidget):
    def __init__(self, node: NodeInstance, parent=None):
        super().__init__(parent)
        self._node = node
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(12, 10, 12, 10)
        self._lay.setSpacing(6)
        self._build()

    def _build(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        node = self._node
        status_colors = {
            "idle": "#64748b", "running": "#ffd700",
            "pass": "#39ff14", "fail": "#ff3860", "error": "#ff3860"
        }
        sc = status_colors.get(node.status, "#64748b")
        st = QLabel(f"Status: {node.status.upper()}")
        st.setStyleSheet(f"color:{sc}; font-size:13px; font-weight:700;")
        self._lay.addWidget(st)

        if node.error_msg:
            err = QLabel(f"Error: {node.error_msg}")
            err.setStyleSheet("color:#ff3860; font-size:11px;")
            err.setWordWrap(True)
            self._lay.addWidget(err)

        if not node.outputs:
            lbl = QLabel("No output yet.\nRun the pipeline first.")
            lbl.setStyleSheet("color:#1e2d45; font-size:12px;")
            lbl.setAlignment(Qt.AlignCenter)
            self._lay.addWidget(lbl)
            self._lay.addStretch()
            return

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        self._lay.addWidget(sep)

        for key, val in node.outputs.items():
            if key == "image":
                continue
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)

            k = QLabel(key)
            k.setStyleSheet("color:#64748b; font-size:11px; font-family:'Courier New';")
            k.setMinimumWidth(100)
            rl.addWidget(k)

            if isinstance(val, bool):
                v_txt = "✔ TRUE" if val else "✖ FALSE"
                v_col = "#39ff14" if val else "#ff3860"
            elif isinstance(val, float):
                v_txt = f"{val:.5f}"; v_col = "#00d4ff"
            elif isinstance(val, int):
                v_txt = str(val); v_col = "#00d4ff"
            elif isinstance(val, str):
                v_txt = val[:60]; v_col = "#e2e8f0"
            elif isinstance(val, list):
                v_txt = f"[list: {len(val)} items]"; v_col = "#ff8c42"
            else:
                v_txt = type(val).__name__; v_col = "#64748b"

            v = QLabel(v_txt)
            v.setStyleSheet(f"color:{v_col}; font-size:11px; font-weight:600;")
            v.setWordWrap(True)
            rl.addWidget(v, 1)
            self._lay.addWidget(row)

        self._lay.addStretch()


class ImagePreviewWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background:#000; border:1px solid #1e2d45; border-radius:6px;"
            "color:#1e2d45; font-size:12px;")
        self.setMinimumHeight(180)
        self.setText("No Image")
        self._current_image = None

    def set_image(self, img_array):
        if img_array is None:
            self.setText("No Image")
            self._current_image = None
            return
        import cv2
        arr = img_array.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

        h, w, ch = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pw = max(1, self.width() - 4)
        ph = max(1, self.height() - 4)
        scaled = pix.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self._current_image = img_array

    def resizeEvent(self, event):
        if self._current_image is not None:
            self.set_image(self._current_image)
        super().resizeEvent(event)


class PropertiesPanel(QWidget):
    params_changed = Signal(str)
    # Phát khi param thay đổi ảnh hưởng cấu trúc port (vd input_count
    # của Pass/Fail Judge). Listener (MainWindow) sẽ gọi refresh_ports
    # trên NodeItem tương ứng.
    ports_need_refresh = Signal(str)
    # Tên các param khi đổi → trigger rebuild port (số lượng port phụ thuộc).
    _PORT_PARAMS = {"input_count"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)
        self._current_node_id: Optional[str] = None
        self._graph: Optional[FlowGraph] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._title = QLabel("⚙  PROPERTIES")
        self._title.setStyleSheet("""
            background:#060a14; color:#00d4ff;
            font-size:11px; font-weight:700; letter-spacing:2px;
            padding:10px 12px; border-bottom:1px solid #1e2d45;
        """)
        lay.addWidget(self._title)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border:none; background:#0d1220; }
            QTabBar::tab {
                background:#0a0e1a; color:#64748b;
                padding:6px 10px; border:none;
                font-size:11px; font-weight:600;
            }
            QTabBar::tab:selected { color:#00d4ff; border-bottom:2px solid #00d4ff; }
            QTabBar::tab:hover { color:#e2e8f0; }
        """)
        lay.addWidget(self._tabs)

        # ── Tab: Info ──────────────────────────────────────────────
        self._info_scroll = QScrollArea()
        self._info_scroll.setWidgetResizable(True)
        self._info_scroll.setFrameShape(QFrame.NoFrame)
        self._info_scroll.setWidget(_make_placeholder("Select a node\nto view properties."))
        self._tabs.addTab(self._info_scroll, "Info")

        # ── Tab: Params ────────────────────────────────────────────
        self._params_scroll = QScrollArea()
        self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.NoFrame)
        self._params_scroll.setWidget(_make_placeholder("Select a node\nto edit parameters."))
        self._tabs.addTab(self._params_scroll, "Params")

        # ── Tab: Output ────────────────────────────────────────────
        self._out_scroll = QScrollArea()
        self._out_scroll.setWidgetResizable(True)
        self._out_scroll.setFrameShape(QFrame.NoFrame)
        self._out_scroll.setWidget(_make_placeholder("Run pipeline to\nsee outputs."))
        self._tabs.addTab(self._out_scroll, "Output")

        # ── Tab: Preview ───────────────────────────────────────────
        preview_tab = QWidget()
        pl = QVBoxLayout(preview_tab)
        pl.setContentsMargins(8, 8, 8, 8)
        self._img_preview = ImagePreviewWidget()
        pl.addWidget(self._img_preview)
        self._prev_info = QLabel("")
        self._prev_info.setStyleSheet(
            "color:#64748b; font-size:10px; font-family:'Courier New';")
        self._prev_info.setAlignment(Qt.AlignCenter)
        self._prev_info.setWordWrap(True)
        pl.addWidget(self._prev_info)
        pl.addStretch()
        self._tabs.addTab(preview_tab, "Preview")

        # ── Bottom hint ────────────────────────────────────────────
        self._empty_lbl = QLabel("Click a node to inspect it")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color:#1e2d45; font-size:11px; padding:8px;")
        lay.addWidget(self._empty_lbl)

    # ── Public API ────────────────────────────────────────────────
    def set_graph(self, graph: FlowGraph):
        self._graph = graph

    def show_node(self, node_id: str):
        if not self._graph or node_id not in self._graph.nodes:
            return
        self._current_node_id = node_id
        node = self._graph.nodes[node_id]
        self._title.setText(f"⚙  {node.tool.name}")
        self._empty_lbl.hide()

        self._info_scroll.setWidget(NodeInfoWidget(node))
        self._build_params_tab(node)
        self._refresh_outputs_tab(node)
        self._refresh_preview(node)

    def refresh_outputs(self):
        if not self._current_node_id or not self._graph:
            return
        if self._current_node_id not in self._graph.nodes:
            return
        node = self._graph.nodes[self._current_node_id]
        self._refresh_outputs_tab(node)
        self._refresh_preview(node)

    def clear(self):
        """Reset về trạng thái rỗng — luôn tạo placeholder MỚI."""
        self._current_node_id = None
        self._title.setText("⚙  PROPERTIES")
        self._empty_lbl.show()
        # Tạo widget mới mỗi lần — KHÔNG tái dùng object cũ đã bị Qt xóa
        self._info_scroll.setWidget(
            _make_placeholder("Select a node\nto view properties."))
        self._params_scroll.setWidget(
            _make_placeholder("Select a node\nto edit parameters."))
        self._out_scroll.setWidget(
            _make_placeholder("Run pipeline to\nsee outputs."))
        self._img_preview.set_image(None)
        self._prev_info.setText("")

    # ── Internal ──────────────────────────────────────────────────
    def _build_params_tab(self, node: NodeInstance):
        tool = node.tool
        if not tool.params:
            self._params_scroll.setWidget(
                _make_placeholder("No parameters\nfor this tool."))
            return

        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        # Nếu tool có ROI params → nút Draw ROI bằng mouse drag
        roi_kind = _detect_roi_kind(tool)
        if roi_kind is not None:
            btn = QPushButton({
                "rect_xywh":     "🖱  Draw ROI (drag rectangle)",
                "rect_x1y1x2y2": "🖱  Draw ROI band (drag rectangle)",
                "point_pick":    "🖱  Pick point on image (click)",
            }[roi_kind])
            btn.setStyleSheet(
                "QPushButton{background:#0f3460;color:#00d4ff;"
                "border:1px solid #1e2d45;border-radius:4px;"
                "padding:8px;font-weight:600;font-size:12px;}"
                "QPushButton:hover{background:#00d4ff;color:#000;}")
            btn.clicked.connect(
                lambda: self._open_roi_dialog(node, roi_kind))
            cl.addWidget(btn)
            sep_top = QFrame(); sep_top.setFrameShape(QFrame.HLine)
            sep_top.setStyleSheet("color:#1e2d45;")
            cl.addWidget(sep_top)

        for param in tool.params:
            # Conditional visibility
            if getattr(param, "visible_if", None):
                ok = True
                for k, v in param.visible_if.items():
                    if node.params.get(k) != v:
                        ok = False; break
                if not ok:
                    continue
            pr = ParamRow(param, node.params.get(param.name, param.default))
            pr.value_changed.connect(
                lambda name, val, nid=node.node_id:
                    self._on_param_changed(nid, name, val))
            cl.addWidget(pr)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        cl.addWidget(sep)
        note = QLabel("Changes apply on next run  ▶")
        note.setStyleSheet("color:#1e2d45; font-size:10px;")
        note.setAlignment(Qt.AlignCenter)
        cl.addWidget(note)
        cl.addStretch()
        self._params_scroll.setWidget(container)

    def _open_roi_dialog(self, node: NodeInstance, kind: str):
        """Mở ROIDrawDialog với ảnh upstream → user kéo chuột vẽ ROI →
        ghi kết quả vào params. Tìm ảnh upstream từ connected input port
        'image' của node; fallback acquire image node trong graph nếu node
        chưa có upstream image."""
        img = self._find_upstream_image(node)
        if img is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Draw ROI",
                "Chưa có ảnh upstream. Hãy Run pipeline trước để load ảnh.")
            return
        dlg = ROIDrawDialog(img, kind, dict(node.params), self)
        if dlg.exec() != QDialog.Accepted:
            return
        result = dlg.get_result()
        # Cập nhật từng param + emit signal (trigger refresh UI + persist)
        for k, v in result.items():
            node.params[k] = v
        self.params_changed.emit(node.node_id)
        # Rebuild params tab để spinbox phản ánh giá trị mới
        QTimer.singleShot(0, lambda nid=node.node_id: self._refresh_node_safe(nid))

    def _refresh_node_safe(self, node_id: str):
        """Re-render params tab cho node_id nếu nó còn đang được show."""
        if not self._graph or node_id not in self._graph.nodes:
            return
        if self._current_node_id == node_id:
            self._build_params_tab(self._graph.nodes[node_id])

    def _find_upstream_image(self, node: NodeInstance):
        """Tìm numpy image gần nhất cho node:
          1. Output của upstream node connect tới input 'image' của node này
          2. Output của node tự nó (nếu đã chạy rồi)
          3. Bất kỳ Acquire Image node nào trong graph (last-resort)
        """
        import numpy as np
        if self._graph is None:
            return None
        # 1. Upstream qua image port
        for c in self._graph.connections:
            if c.dst_id == node.node_id and c.dst_port == "image":
                src = self._graph.nodes.get(c.src_id)
                if src and isinstance(src.outputs.get(c.src_port), np.ndarray):
                    return src.outputs[c.src_port]
        # 2. Output của chính node
        out = node.outputs.get("image")
        if isinstance(out, np.ndarray):
            return out
        # 3. Bất kỳ acquire image nào
        for n in self._graph.nodes.values():
            if n.tool.tool_id in ("acquire_image", "camera_acquire"):
                im = n.outputs.get("image")
                if isinstance(im, np.ndarray):
                    return im
        return None

    def _on_param_changed(self, node_id: str, name: str, value: Any):
        if not (self._graph and node_id in self._graph.nodes):
            return
        node = self._graph.nodes[node_id]
        node.params[name] = value
        self.params_changed.emit(node_id)
        if name in self._PORT_PARAMS:
            self.ports_need_refresh.emit(node_id)
        # Nếu có param khác phụ thuộc vào tên này → rebuild để cập nhật visibility.
        # Defer qua event loop: nếu rebuild ngay trong slot sẽ xoá chính widget
        # đang phát signal (vd QComboBox source_mode) → crash C++ deleted object.
        if any(getattr(p, "visible_if", None) and name in p.visible_if
                for p in node.tool.params):
            QTimer.singleShot(0, lambda nid=node_id: self._rebuild_params_if_current(nid))

    def _rebuild_params_if_current(self, node_id: str):
        if (self._graph and self._current_node_id == node_id
                and node_id in self._graph.nodes):
            self._build_params_tab(self._graph.nodes[node_id])

    def _refresh_outputs_tab(self, node: NodeInstance):
        self._out_scroll.setWidget(OutputsWidget(node))

    def _refresh_preview(self, node: NodeInstance):
        img = node.outputs.get("image")
        if img is not None:
            import numpy as np
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                self._prev_info.setText(f"{w} × {h} px  |  {img.dtype}")
                self._img_preview.set_image(img)
                self._tabs.setCurrentIndex(3)
                return
        self._img_preview.set_image(None)
        self._prev_info.setText("")


# ── ROI Draw Dialog ───────────────────────────────────────────────────
class ROIDrawDialog(QDialog):
    """Dialog cho phép kéo chuột vẽ ROI trên ảnh upstream.

    kind:
      - "rect_xywh"     → rectangle drag, trả {x, y, w, h}
      - "rect_x1y1x2y2" → rectangle drag, trả {x1, y1, x2, y2}
                           (top-left / bottom-right corners)
      - "point_pick"    → single click, trả {pick_x, pick_y}
    """

    def __init__(self, image, kind: str, current: dict, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._result: dict = {}
        self.setWindowTitle("🖱  Draw ROI on Image")
        self.resize(1000, 700)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)
        self.setStyleSheet("QDialog{background:#0a0e1a;color:#e2e8f0;}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        # Hint label
        hint = {
            "rect_xywh":     "Kéo chuột để vẽ hình chữ nhật. Esc huỷ.",
            "rect_x1y1x2y2": "Kéo chuột vẽ hình chữ nhật — 2 góc map sang (x1,y1)→(x2,y2).",
            "point_pick":    "Click chuột để chọn pixel.",
        }.get(kind, "Kéo chuột để vẽ.")
        hl = QLabel(hint)
        hl.setStyleSheet("color:#00d4ff;font-size:11px;padding:4px 6px;")
        lay.addWidget(hl)

        # Image label (reuse InteractiveImageLabel)
        from ui.node_detail_dialog import InteractiveImageLabel
        mode = "pick" if kind == "point_pick" else "roi"
        self._label = InteractiveImageLabel(mode=mode)
        self._scroll = _QScroll()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            "QScrollArea{background:#050810;border:1px solid #1e2d45;}")
        self._label.set_scroll_area(self._scroll)
        if image is not None:
            self._label.set_image(image)
            # Pre-populate hiện ROI nếu có
            if kind == "rect_xywh":
                x = int(current.get("x", 0)); y = int(current.get("y", 0))
                w = int(current.get("w", 50)); h = int(current.get("h", 50))
                if w > 0 and h > 0:
                    self._label.set_rect_from_params(x, y, w, h)
            elif kind == "rect_x1y1x2y2":
                x1 = int(current.get("x1", 0)); y1 = int(current.get("y1", 0))
                x2 = int(current.get("x2", 100)); y2 = int(current.get("y2", 100))
                x = min(x1, x2); y = min(y1, y2)
                w = abs(x2 - x1); h = abs(y2 - y1)
                if w > 0 and h > 0:
                    self._label.set_rect_from_params(x, y, w, h)
            elif kind == "point_pick":
                # InteractiveImageLabel pick mode — pre-pos chưa support
                pass
        lay.addWidget(self._scroll, 1)

        # Buttons
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.setStyleSheet(
            "QPushButton{background:#1e2d45;color:#e2e8f0;border:none;"
            "border-radius:4px;padding:6px 14px;font-weight:600;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        # Track latest rect/pick từ label signals
        self._latest_rect: Optional[Tuple[int, int, int, int]] = None
        if kind == "point_pick":
            self._label.pixel_picked.connect(self._on_picked)
            self._picked: Optional[Tuple[int, int]] = None
            # Initialize từ current params nếu có
            if current.get("pick_x") is not None and current.get("pick_y") is not None:
                self._picked = (int(current["pick_x"]), int(current["pick_y"]))
        else:
            self._label.roi_changed.connect(self._on_roi_changed)
            # Initialize từ current params
            if kind == "rect_xywh":
                self._latest_rect = (
                    int(current.get("x", 0)), int(current.get("y", 0)),
                    int(current.get("w", 0)), int(current.get("h", 0)))
            elif kind == "rect_x1y1x2y2":
                x1 = int(current.get("x1", 0)); y1 = int(current.get("y1", 0))
                x2 = int(current.get("x2", 0)); y2 = int(current.get("y2", 0))
                self._latest_rect = (
                    min(x1, x2), min(y1, y2),
                    abs(x2 - x1), abs(y2 - y1))

    def _on_picked(self, x: int, y: int):
        self._picked = (int(x), int(y))

    def _on_roi_changed(self, x: int, y: int, w: int, h: int):
        self._latest_rect = (int(x), int(y), int(w), int(h))

    def _on_accept(self):
        if self._kind == "point_pick":
            if self._picked is None:
                self.reject(); return
            self._result = {"pick_x": int(self._picked[0]),
                            "pick_y": int(self._picked[1])}
            self.accept(); return
        if self._latest_rect is None or self._latest_rect[2] <= 0:
            self.reject(); return
        x, y, w, h = self._latest_rect
        if self._kind == "rect_xywh":
            self._result = {"x": int(x), "y": int(y),
                            "w": int(w), "h": int(h)}
        else:  # rect_x1y1x2y2
            self._result = {"x1": int(x), "y1": int(y),
                            "x2": int(x + w), "y2": int(y + h)}
        self.accept()

    def get_result(self) -> dict:
        return dict(self._result)
