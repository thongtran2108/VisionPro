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

from core.i18n import tr
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

        lbl = QLabel(tr(param.label))
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
            for c in p.choices:
                w.addItem(tr(c), c)        # hiển thị bản dịch, lưu giá trị gốc
            ix = w.findData(str(val))
            if ix >= 0:
                w.setCurrentIndex(ix)
            w.currentIndexChanged.connect(
                lambda _i, ww=w: self.value_changed.emit(p.name, ww.currentData()))
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
        le.setPlaceholderText(tr(p.label))
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


class OverlaySelectRow(QWidget):
    """Param kiểu 'overlay_select' — picker chọn tool để overlay annotation
    lên ảnh, mô phỏng mục 'Results' của Image Viewer. Lưu list node_id vào
    param. Dùng cho Save Image: tick tool nào thì ảnh lưu hiện annotation đó.
    Emit value_changed(name, list) y như ParamRow để panel xử lý chung."""
    value_changed = Signal(str, object)

    def __init__(self, param: ParamDef, current: Any,
                 graph: Optional[FlowGraph], node: NodeInstance, parent=None):
        super().__init__(parent)
        self._param = param
        self._graph = graph
        self._node = node
        self._selected = set(current or [])

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        lbl = QLabel(tr(param.label))
        lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(110)
        lbl.setMaximumWidth(130)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        from PySide6.QtWidgets import QToolButton, QMenu
        self._btn = QToolButton()
        self._btn.setPopupMode(QToolButton.InstantPopup)
        if param.tooltip:
            self._btn.setToolTip(tr(param.tooltip))
        self._btn.setStyleSheet(
            "QToolButton{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:3px 8px;border-radius:4px;font-size:11px;"
            "text-align:left;}"
            "QToolButton:hover{background:#1a2236;color:#00d4ff;}"
            "QToolButton::menu-indicator{image:none;}")
        self._menu = QMenu(self._btn)
        self._menu.setStyleSheet(
            "QMenu{background:#0d1220;border:1px solid #1e2d45;"
            "padding:4px;color:#e2e8f0;}"
            "QMenu::separator{height:1px;background:#1e2d45;margin:4px 6px;}")
        self._menu.aboutToShow.connect(self._rebuild_menu)
        self._btn.setMenu(self._menu)
        lay.addWidget(self._btn, 1)
        self._update_text()

    def _update_text(self):
        n = len(self._selected)
        self._btn.setText(tr("📊 Chọn kết quả ▾") if n == 0
                          else tr("📊 {n} kết quả ▾").format(n=n))

    def _rebuild_menu(self):
        from PySide6.QtWidgets import QWidgetAction, QCheckBox
        from PySide6.QtGui import QAction
        menu = self._menu
        menu.clear()
        if self._graph is None:
            self._add_note(menu, tr("  (No pipeline)  "))
            return
        from core.overlay_render import overlayable_nodes
        items = overlayable_nodes(self._graph, exclude_id=self._node.node_id)
        # Drop node_id đã chọn nhưng không còn trong graph (file đổi schema).
        valid = {nid for nid, _ in items}
        self._selected = {nid for nid in self._selected if nid in valid}

        # Header — nhắc đây là overlay lên ảnh gốc, giống Image Viewer.
        wa_hdr = QWidgetAction(menu)
        hdr = QLabel(tr("  Base: ảnh gốc + overlay tool đã tick  "))
        hdr.setStyleSheet("color:#00d4ff; font-size:10px; font-weight:700; "
                          "letter-spacing:1px; padding:6px 8px;")
        wa_hdr.setDefaultWidget(hdr)
        menu.addAction(wa_hdr)
        menu.addSeparator()

        if not items:
            self._add_note(menu, tr("  (Chưa có tool nào trong pipeline)  "))
        else:
            for nid, node in items:
                wa = QWidgetAction(menu)
                cb = QCheckBox(f"  {node.tool.icon}  {node.name}  "
                               f"({node.tool.tool_id})")
                cb.setChecked(nid in self._selected)
                cb.setStyleSheet(
                    "QCheckBox{color:#e2e8f0; font-size:11px; padding:4px 8px;}"
                    "QCheckBox::indicator{width:14px; height:14px;}")
                cb.toggled.connect(
                    lambda on, _nid=nid: self._toggle(_nid, on))
                wa.setDefaultWidget(cb)
                menu.addAction(wa)

        menu.addSeparator()
        act_clear = QAction(tr("✗  Clear All"), menu)
        act_clear.triggered.connect(self._clear)
        menu.addAction(act_clear)

    def _add_note(self, menu, text: str):
        from PySide6.QtWidgets import QWidgetAction
        wa = QWidgetAction(menu)
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#64748b; padding:8px;")
        wa.setDefaultWidget(lbl)
        menu.addAction(wa)

    def _toggle(self, nid: str, on: bool):
        if on:
            self._selected.add(nid)
        else:
            self._selected.discard(nid)
        self._update_text()
        self.value_changed.emit(self._param.name, sorted(self._selected))

    def _clear(self):
        self._selected.clear()
        self._update_text()
        self.value_changed.emit(self._param.name, [])


class CsvColumnsRow(QWidget):
    """Param kiểu 'csv_columns' — mở trình soạn bảng (CsvColumnsDialog) để định
    nghĩa cột header + link giá trị cho CSV/XLSX Logger. Lưu list dict cột.
    Emit value_changed(name, list) như ParamRow."""
    value_changed = Signal(str, object)

    def __init__(self, param: ParamDef, current: Any,
                 graph: Optional[FlowGraph], node: NodeInstance, parent=None):
        super().__init__(parent)
        self._param = param
        self._graph = graph
        self._node = node
        self._cols = [dict(c) for c in (current or []) if isinstance(c, dict)]

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        lbl = QLabel(tr(param.label))
        lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(110)
        lbl.setMaximumWidth(130)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self._btn = QPushButton()
        if param.tooltip:
            self._btn.setToolTip(tr(param.tooltip))
        self._btn.setStyleSheet(
            "QPushButton{background:#0f3460;color:#00d4ff;border:1px solid #1e2d45;"
            "border-radius:4px;padding:6px 8px;font-weight:600;font-size:11px;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        self._btn.clicked.connect(self._open)
        lay.addWidget(self._btn, 1)
        self._update_text()

    def _update_text(self):
        n = len(self._cols)
        self._btn.setText(tr("🧾 Soạn bảng…") if n == 0
                          else tr("🧾 Soạn bảng ({n} cột)").format(n=n))

    def _open(self):
        dlg = CsvColumnsDialog(self._cols, self._graph, self._node, self)
        if dlg.exec() == QDialog.Accepted:
            self._cols = dlg.get_columns()
            self._update_text()
            self.value_changed.emit(self._param.name, self._cols)


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
        name_lbl = QLabel(node.name)
        name_lbl.setStyleSheet("color:#e2e8f0; font-size:14px; font-weight:700;")
        cat_txt = (f"{tr(tool.name)}  ·  {tr(tool.category)}"
                   if node.name != tool.name
                   else f"{tr('Category')}: {tr(tool.category)}")
        cat_lbl = QLabel(cat_txt)
        cat_lbl.setStyleSheet("color:#64748b; font-size:11px;")
        id_lbl = QLabel(f"Node ID: {node.node_id}")
        id_lbl.setStyleSheet("color:#1e2d45; font-size:10px; font-family:'Courier New';")
        info_lay.addWidget(name_lbl)
        info_lay.addWidget(cat_lbl)
        info_lay.addWidget(id_lbl)
        top.addLayout(info_lay, 1)
        lay.addLayout(top)

        desc = QLabel(tr(tool.description))
        desc.setStyleSheet("color:#94a3b8; font-size:11px; padding:6px 0;")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)

        ports_lbl = QLabel(tr("PORTS"))
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
            lbl = QLabel(tr("No output yet.\nRun the pipeline first."))
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)
        self._current_node_id: Optional[str] = None
        self._graph: Optional[FlowGraph] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._title = QLabel(tr("⚙  PROPERTIES"))
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
        self._info_scroll.setWidget(_make_placeholder(tr("Select a node\nto view properties.")))
        self._tabs.addTab(self._info_scroll, tr("Info"))

        # ── Tab: Params ────────────────────────────────────────────
        self._params_scroll = QScrollArea()
        self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.NoFrame)
        self._params_scroll.setWidget(_make_placeholder(tr("Select a node\nto edit parameters.")))
        self._tabs.addTab(self._params_scroll, tr("Params"))

        # ── Tab: Output ────────────────────────────────────────────
        self._out_scroll = QScrollArea()
        self._out_scroll.setWidgetResizable(True)
        self._out_scroll.setFrameShape(QFrame.NoFrame)
        self._out_scroll.setWidget(_make_placeholder(tr("Run pipeline to\nsee outputs.")))
        self._tabs.addTab(self._out_scroll, tr("Output"))

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
        self._tabs.addTab(preview_tab, tr("Preview"))

        # ── Bottom hint ────────────────────────────────────────────
        self._empty_lbl = QLabel(tr("Click a node to inspect it"))
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
        self._title.setText(f"⚙  {node.name}")
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
        self._title.setText(tr("⚙  PROPERTIES"))
        self._empty_lbl.show()
        # Tạo widget mới mỗi lần — KHÔNG tái dùng object cũ đã bị Qt xóa
        self._info_scroll.setWidget(
            _make_placeholder(tr("Select a node\nto view properties.")))
        self._params_scroll.setWidget(
            _make_placeholder(tr("Select a node\nto edit parameters.")))
        self._out_scroll.setWidget(
            _make_placeholder(tr("Run pipeline to\nsee outputs.")))
        self._img_preview.set_image(None)
        self._prev_info.setText("")

    # ── Internal ──────────────────────────────────────────────────
    def _build_params_tab(self, node: NodeInstance):
        tool = node.tool
        if not tool.params:
            self._params_scroll.setWidget(
                _make_placeholder(tr("No parameters\nfor this tool.")))
            return
        # Backfill default cho param còn thiếu + chuẩn hoá giá trị enum lạ
        # (pipeline cũ / tool thêm param mới / file lỗi) → visible_if đọc đúng
        # giá trị, không ẩn nhầm các param phụ thuộc (vd slider màu color_segment).
        for _p in tool.params:
            if _p.name not in node.params:
                node.params[_p.name] = _p.default
            elif _p.ptype == "enum" and _p.choices \
                    and node.params[_p.name] not in _p.choices:
                node.params[_p.name] = _p.default

        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        # Nếu tool có ROI params → nút Draw ROI bằng mouse drag
        roi_kind = _detect_roi_kind(tool)
        if roi_kind is not None:
            btn = QPushButton(tr({
                "rect_xywh":     "🖱  Draw ROI (drag rectangle)",
                "rect_x1y1x2y2": "🖱  Draw ROI band (drag rectangle)",
                "point_pick":    "🖱  Pick point on image (click)",
            }[roi_kind]))
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
            if param.ptype == "overlay_select":
                pr = OverlaySelectRow(
                    param, node.params.get(param.name, param.default),
                    self._graph, node)
            elif param.ptype == "csv_columns":
                pr = CsvColumnsRow(
                    param, node.params.get(param.name, param.default),
                    self._graph, node)
            else:
                pr = ParamRow(param, node.params.get(param.name, param.default))
            pr.value_changed.connect(
                lambda name, val, nid=node.node_id:
                    self._on_param_changed(nid, name, val))
            cl.addWidget(pr)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1e2d45;")
        cl.addWidget(sep)
        note = QLabel(tr("Changes apply on next run  ▶"))
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
                self, tr("Draw ROI"),
                tr("Chưa có ảnh upstream. Hãy Run pipeline trước để load ảnh."))
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


# ── CSV/XLSX Columns Editor ───────────────────────────────────────────
class CsvColumnsDialog(QDialog):
    """Trình soạn bảng cho CSV/XLSX Logger — mỗi dòng = 1 CỘT trong file log:
    đặt Header rồi link Nguồn giá trị (thời gian / PASS-FAIL / mã scan / output
    của node bất kỳ / hằng số). Có preview 1 dòng mẫu như Excel."""

    def __init__(self, columns, graph: Optional[FlowGraph],
                 node: NodeInstance, parent=None):
        super().__init__(parent)
        self._graph = graph
        self._node = node
        self._cols = [dict(c) for c in (columns or []) if isinstance(c, dict)]
        self._source_opts = self._build_source_options()

        self.setWindowTitle(tr("🧾  Cấu hình bảng dữ liệu (CSV/XLSX Logger)"))
        self.resize(780, 560)
        self.setStyleSheet("QDialog{background:#0a0e1a;color:#e2e8f0;}"
                           "QLabel{color:#e2e8f0;}")

        from PySide6.QtWidgets import QTableWidget, QHeaderView
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10); lay.setSpacing(8)

        hint = QLabel(tr("Mỗi dòng = 1 CỘT trong file log. Viết Header rồi chọn "
                         "Nguồn giá trị để link. Mỗi lần Run ghi 1 dòng mới."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#00d4ff;font-size:11px;padding:2px;")
        lay.addWidget(hint)

        tb = QHBoxLayout(); tb.setSpacing(6)

        def _mk(txt, tip, fn):
            b = QPushButton(txt); b.setToolTip(tip); b.setFixedHeight(28)
            b.setStyleSheet(
                "QPushButton{background:#1e2d45;color:#e2e8f0;border:none;"
                "border-radius:4px;padding:4px 10px;font-weight:600;}"
                "QPushButton:hover{background:#00d4ff;color:#000;}")
            b.clicked.connect(fn); return b

        tb.addWidget(_mk(tr("＋ Thêm cột"), tr("Thêm 1 cột mới"), self._add))
        tb.addWidget(_mk(tr("✖ Xoá cột"), tr("Xoá cột đang chọn"), self._remove))
        tb.addWidget(_mk(tr("↑"), tr("Di chuyển lên"), lambda: self._move(-1)))
        tb.addWidget(_mk(tr("↓"), tr("Di chuyển xuống"), lambda: self._move(1)))
        tb.addStretch()
        lay.addLayout(tb)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(
            [tr("Header (tên cột)"), tr("Nguồn giá trị"),
             tr("Hằng số (khi chọn 'cố định')")])
        hh = self._table.horizontalHeader()
        for ci in range(3):
            hh.setSectionResizeMode(ci, QHeaderView.Stretch)
        self._table.setStyleSheet(
            "QTableWidget{background:#0d1220;color:#e2e8f0;gridline-color:#1e2d45;"
            "border:1px solid #1e2d45;}"
            "QTableWidget::item:selected{background:#1a2236;}"
            "QHeaderView::section{background:#11182a;color:#94a3b8;border:none;"
            "padding:4px;font-weight:600;}")
        self._table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self._table, 1)

        pv_lbl = QLabel(tr("Xem trước (1 dòng mẫu từ output hiện tại):"))
        pv_lbl.setStyleSheet("color:#64748b;font-size:10px;")
        lay.addWidget(pv_lbl)
        self._preview = QTableWidget(0, 0)
        self._preview.setFixedHeight(90)
        self._preview.setEditTriggers(QTableWidget.NoEditTriggers)
        self._preview.verticalHeader().setVisible(False)
        self._preview.setStyleSheet(
            "QTableWidget{background:#0d1220;color:#e2e8f0;gridline-color:#1e2d45;"
            "border:1px solid #1e2d45;}"
            "QHeaderView::section{background:#11182a;color:#00d4ff;border:none;"
            "padding:3px;font-weight:600;}")
        lay.addWidget(self._preview)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.setStyleSheet(
            "QPushButton{background:#1e2d45;color:#e2e8f0;border:none;"
            "border-radius:4px;padding:6px 14px;font-weight:600;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        if not self._cols:
            # Seed vài cột hay dùng để user bắt đầu nhanh.
            self._cols = [
                {"header": "Thời gian", "kind": "timestamp"},
                {"header": "Kết quả", "kind": "pass"},
                {"header": "Mã scan", "kind": "sn"},
            ]
        self._rebuild_table()

    # ── Sources ───────────────────────────────────────────────────
    def _build_source_options(self):
        """List (label, source_dict): built-in + output port (non-image) của
        mọi node khác trong graph."""
        opts = [
            (tr("🕒 Thời gian (timestamp)"), {"kind": "timestamp"}),
            (tr("✔ Kết quả PASS/FAIL (port pass)"), {"kind": "pass"}),
            (tr("🔖 Mã scan (SFC)"), {"kind": "sn"}),
            (tr("✎ Giá trị cố định"), {"kind": "const", "value": ""}),
        ]
        if self._graph is not None:
            for nid, n in self._graph.nodes.items():
                if nid == self._node.node_id:
                    continue
                for p in n.tool.outputs:
                    if p.data_type == "image":
                        continue
                    opts.append((f"{n.tool.icon} {n.name} · {p.name}",
                                 {"kind": "node", "node": nid, "port": p.name}))
        return opts

    @staticmethod
    def _same_source(a, b) -> bool:
        if a.get("kind") != b.get("kind"):
            return False
        if a.get("kind") == "node":
            return a.get("node") == b.get("node") and a.get("port") == b.get("port")
        return True

    # ── Table build / edit ────────────────────────────────────────
    def _rebuild_table(self):
        from PySide6.QtWidgets import QTableWidgetItem
        t = self._table
        t.blockSignals(True)
        t.setRowCount(0)
        for r, col in enumerate(self._cols):
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(str(col.get("header", ""))))
            combo = QComboBox()
            combo.setStyleSheet(
                "QComboBox{background:#0a0e1a;color:#e2e8f0;"
                "border:1px solid #1e2d45;padding:2px 6px;}"
                "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
                "selection-background-color:#1a2236;}")
            sel = -1
            for i, (label, src) in enumerate(self._source_opts):
                combo.addItem(label, src)
                if sel < 0 and self._same_source(src, col):
                    sel = i
            if sel >= 0:
                combo.setCurrentIndex(sel)
            combo.currentIndexChanged.connect(
                lambda _i, rr=r: self._on_source(rr))
            t.setCellWidget(r, 1, combo)
            is_const = col.get("kind") == "const"
            citem = QTableWidgetItem(str(col.get("value", "")) if is_const else "")
            if not is_const:
                citem.setFlags(citem.flags() & ~Qt.ItemIsEnabled)
            t.setItem(r, 2, citem)
        t.blockSignals(False)
        self._rebuild_preview()

    def _on_source(self, r):
        if not (0 <= r < len(self._cols)):
            return
        combo = self._table.cellWidget(r, 1)
        if combo is None:
            return
        src = dict(combo.currentData() or {})
        src["header"] = self._cols[r].get("header", "")
        from PySide6.QtWidgets import QTableWidgetItem
        self._table.blockSignals(True)
        it = self._table.item(r, 2) or QTableWidgetItem("")
        if src.get("kind") == "const":
            it.setFlags(it.flags() | Qt.ItemIsEnabled | Qt.ItemIsEditable
                        | Qt.ItemIsSelectable)
            src["value"] = it.text()
        else:
            it.setText("")
            it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
        self._table.setItem(r, 2, it)
        self._table.blockSignals(False)
        self._cols[r] = src
        self._rebuild_preview()

    def _on_item_changed(self, item):
        r, c = item.row(), item.column()
        if not (0 <= r < len(self._cols)):
            return
        if c == 0:
            self._cols[r]["header"] = item.text()
        elif c == 2 and self._cols[r].get("kind") == "const":
            self._cols[r]["value"] = item.text()
        self._rebuild_preview()

    def _current_row(self) -> int:
        r = self._table.currentRow()
        if r is not None and r >= 0:
            return r
        return len(self._cols) - 1 if self._cols else -1

    def _add(self):
        self._cols.append(
            {"header": tr("Cột {n}").format(n=len(self._cols) + 1),
             "kind": "timestamp"})
        self._rebuild_table()
        self._table.setCurrentCell(len(self._cols) - 1, 0)

    def _remove(self):
        r = self._current_row()
        if 0 <= r < len(self._cols):
            self._cols.pop(r)
            self._rebuild_table()

    def _move(self, delta: int):
        r = self._current_row()
        j = r + delta
        if 0 <= r < len(self._cols) and 0 <= j < len(self._cols):
            self._cols[r], self._cols[j] = self._cols[j], self._cols[r]
            self._rebuild_table()
            self._table.setCurrentCell(j, 0)

    # ── Preview ───────────────────────────────────────────────────
    def _preview_value(self, col):
        kind = col.get("kind")
        if kind == "timestamp":
            import datetime
            fmt = self._node.params.get("timestamp_format") or "%Y-%m-%d %H:%M:%S"
            try:
                return datetime.datetime.now().strftime(fmt)
            except Exception:
                return "(thời gian)"
        if kind == "pass":
            return "PASS / FAIL"
        if kind == "sn":
            try:
                from core.tool_registry import EXEC_CTX
                return (getattr(EXEC_CTX.sfc, "last_sn", "") or "(mã scan)")
            except Exception:
                return "(mã scan)"
        if kind == "const":
            return col.get("value", "")
        if kind == "node":
            nid, port = col.get("node"), col.get("port")
            if self._graph is not None and nid in self._graph.nodes:
                v = self._graph.nodes[nid].outputs.get(port)
                if v is not None:
                    from core.tool_registry import _csv_fmt_value
                    return _csv_fmt_value(v)
                return f"({port})"
            return "(node đã xoá?)"
        return ""

    def _rebuild_preview(self):
        from PySide6.QtWidgets import QTableWidgetItem
        headers = [str(c.get("header", "")) for c in self._cols]
        self._preview.clear()
        self._preview.setColumnCount(len(headers))
        self._preview.setRowCount(1 if headers else 0)
        self._preview.setHorizontalHeaderLabels(headers)
        for ci, col in enumerate(self._cols):
            self._preview.setItem(0, ci,
                                  QTableWidgetItem(str(self._preview_value(col))))
        self._preview.resizeColumnsToContents()

    def get_columns(self):
        """Trả list cột — bỏ cột header rỗng cho file gọn."""
        return [dict(c) for c in self._cols
                if str(c.get("header", "")).strip()]
