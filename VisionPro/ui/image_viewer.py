"""
ui/image_viewer.py
Panel xem ảnh chính — hiển thị ảnh kết quả với zoom/pan, overlay info,
chọn node để xem output image.
"""
from __future__ import annotations
from typing import Optional, Dict, List
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QComboBox, QPushButton, QScrollArea,
                                QSizePolicy, QFrame, QSlider, QCheckBox,
                                QToolBar, QSplitter, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, Signal, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import (QPixmap, QImage, QColor, QPainter, QPen, QBrush,
                            QFont, QWheelEvent, QMouseEvent, QTransform,
                            QPainterPath)

from core.flow_graph import FlowGraph


# ── Zoomable image widget ─────────────────────────────────────────
class ZoomableImageWidget(QWidget):
    """Widget hiển thị ảnh có zoom/pan bằng mouse."""
    pixel_info = Signal(int, int, tuple)   # x, y, (r,g,b)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#050810;")

        self._pixmap:   Optional[QPixmap] = None
        self._arr:      Optional[np.ndarray] = None
        self._scale     = 1.0
        self._offset    = QPointF(0, 0)
        self._panning   = False
        self._pan_start = QPointF(0, 0)
        self._show_grid = False
        self._zoom_text = ""

        # Pan/zoom mượt với ảnh lớn (20MP): vẽ nhanh (FastTransformation)
        # trong lúc tương tác, hẹn 1 lần repaint smooth sau khi dừng ~120ms.
        self._interacting = False
        self._smooth_timer = QTimer(self)
        self._smooth_timer.setSingleShot(True)
        self._smooth_timer.setInterval(120)
        self._smooth_timer.timeout.connect(self._end_interaction)

    def _begin_interaction(self):
        self._interacting = True
        self._smooth_timer.start()   # restart → smooth lại sau khi dừng

    def _end_interaction(self):
        self._interacting = False
        self.update()

    # ── Image ───────────────────────────────────────────────────
    def set_image(self, arr: Optional[np.ndarray]):
        if arr is None:
            self._arr = None
            self._pixmap = None
            self.update()
            return
        # Đảm bảo contiguous để Qt dùng buffer trực tiếp (không copy).
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        self._arr = arr   # giữ alive cho QImage buffer reference
        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, arr.strides[0],
                           QImage.Format_Grayscale8)
        elif arr.shape[2] == 3:
            h, w, _ = arr.shape
            # Format_BGR888 dùng trực tiếp BGR của OpenCV → skip cvtColor.
            qimg = QImage(arr.data, w, h, arr.strides[0],
                           QImage.Format_BGR888)
        elif arr.shape[2] == 4:
            import cv2
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            self._arr = arr
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, arr.strides[0],
                           QImage.Format_RGBA8888)
        else:
            self._pixmap = None
            self.update()
            return
        # QPixmap.fromImage copy data sang pixmap format → arr có thể GC sau
        # call này. Nhưng giữ self._arr để pixel-pick / hover còn truy cập.
        self._pixmap = QPixmap.fromImage(qimg)
        self._fit_to_window()
        self.update()

    def _fit_to_window(self):
        if not self._pixmap:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if ww < 1 or wh < 1:
            return
        self._scale  = min(ww / pw, wh / ph) * 0.95
        self._offset = QPointF(
            (ww - pw * self._scale) / 2,
            (wh - ph * self._scale) / 2)

    def fit(self):
        self._fit_to_window()
        self.update()

    def set_zoom(self, factor: float):
        if not self._pixmap:
            return
        cx = self.width() / 2
        cy = self.height() / 2
        img_cx = (cx - self._offset.x()) / self._scale
        img_cy = (cy - self._offset.y()) / self._scale
        self._scale = max(0.05, min(20.0, factor))
        self._offset = QPointF(cx - img_cx * self._scale,
                               cy - img_cy * self._scale)
        self._zoom_text = f"{self._scale * 100:.0f}%"
        self._begin_interaction()
        self.update()

    # ── Paint ───────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        # Smooth chỉ khi đứng yên; lúc pan/zoom dùng fast → không giật ảnh 20MP.
        painter.setRenderHint(QPainter.SmoothPixmapTransform,
                              not self._interacting)
        painter.fillRect(self.rect(), QColor(5, 8, 16))

        if not self._pixmap:
            painter.setPen(QPen(QColor(30, 45, 69)))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "No Image\n\nRun pipeline to see results")
            return

        # Draw image
        dx = self._offset.x()
        dy = self._offset.y()
        pw = self._pixmap.width()  * self._scale
        ph = self._pixmap.height() * self._scale
        painter.drawPixmap(int(dx), int(dy), int(pw), int(ph), self._pixmap)

        # Grid overlay at high zoom
        if self._scale > 8 and self._arr is not None:
            painter.setPen(QPen(QColor(0, 212, 255, 40), 0.5))
            # Vertical lines
            x0 = int(dx % self._scale)
            while x0 < self.width():
                painter.drawLine(x0, 0, x0, self.height())
                x0 += int(self._scale)
            # Horizontal lines
            y0 = int(dy % self._scale)
            while y0 < self.height():
                painter.drawLine(0, y0, self.width(), y0)
                y0 += int(self._scale)

        # Zoom indicator
        if self._zoom_text:
            painter.setPen(QPen(QColor(0, 212, 255, 180)))
            painter.setFont(QFont("Courier New", 11, QFont.Bold))
            painter.drawText(self.rect().adjusted(10, 8, -10, -8),
                             Qt.AlignTop | Qt.AlignRight, self._zoom_text)

    # ── Mouse ───────────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap:
            return
        pos = event.position()
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_scale = max(0.05, min(20.0, self._scale * factor))

        # Zoom toward cursor
        img_x = (pos.x() - self._offset.x()) / self._scale
        img_y = (pos.y() - self._offset.y()) / self._scale
        self._scale  = new_scale
        self._offset = QPointF(pos.x() - img_x * self._scale,
                               pos.y() - img_y * self._scale)
        self._zoom_text = f"{self._scale * 100:.0f}%"
        self._begin_interaction()
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning   = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._panning:
            d = pos - self._pan_start
            self._pan_start = pos
            self._offset += QPointF(d.x(), d.y())
            self._begin_interaction()
            self.update()

        # Pixel info
        if self._arr is not None and self._pixmap:
            ix = int((pos.x() - self._offset.x()) / self._scale)
            iy = int((pos.y() - self._offset.y()) / self._scale)
            h, w = self._arr.shape[:2]
            if 0 <= ix < w and 0 <= iy < h:
                px = self._arr[iy, ix]
                if len(self._arr.shape) == 2:
                    rgb = (int(px), int(px), int(px))
                else:
                    rgb = (int(px[2]), int(px[1]), int(px[0]))  # BGR→RGB
                self.pixel_info.emit(ix, iy, rgb)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

    def resizeEvent(self, event):
        # Luôn refit khi widget thay đổi kích thước (vd Full Image View
        # toggle, window resize) — tránh ảnh tràn ra ngoài viewport.
        if self._pixmap is not None:
            self._fit_to_window()
        super().resizeEvent(event)


# ── Main ImageViewer panel ────────────────────────────────────────
class ImageViewerPanel(QWidget):
    """
    Panel xem ảnh chính — hiển thị output image của node được chọn.
    Có thể chọn node từ dropdown, zoom/pan, xem pixel info.
    """
    # Emit khi user thao tác state cần persist (overlay tick, multi-view
    # mode, custom views, ...). MainWindow connect signal này để mark
    # pipeline dirty → prompt save khi close.
    state_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: Optional[FlowGraph] = None
        self._current_node_id: Optional[str] = None
        self._node_map: Dict[str, str] = {}   # display_name → node_id
        # True trong khoảng set_graph đang khôi phục state từ ui_state →
        # các setter (setChecked / setCurrentIndex) không emit state_changed
        # và không ghi ngược lại ui_state (tránh dirty false-positive).
        self._restoring_state: bool = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Toolbar ────────────────────────────────────────────────
        tb = QWidget()
        tb.setFixedHeight(40)
        tb.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.setSpacing(8)

        view_lbl = QLabel("👁  IMAGE VIEWER")
        view_lbl.setStyleSheet(
            "color:#00d4ff; font-size:11px; font-weight:700; letter-spacing:2px;")
        tl.addWidget(view_lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#1e2d45;")
        tl.addWidget(sep)

        node_lbl = QLabel("Node:")
        node_lbl.setStyleSheet("color:#64748b; font-size:11px;")
        tl.addWidget(node_lbl)
        self._tb_node_lbl = node_lbl   # ref để ẩn khi multi-view ON

        self._node_combo = QComboBox()
        self._node_combo.setMinimumWidth(180)
        self._node_combo.setStyleSheet("""
            QComboBox{background:#0a0e1a;border:1px solid #1e2d45;
                      color:#e2e8f0;padding:2px 8px;border-radius:4px;font-size:11px;}
            QComboBox::drop-down{border:none;}
            QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;
                                         border:1px solid #1e2d45;
                                         selection-background-color:#1a2236;}
        """)
        self._node_combo.currentIndexChanged.connect(self._on_node_selected)
        tl.addWidget(self._node_combo)

        tl.addStretch()

        # Zoom controls
        def tb_btn(txt, tip):
            b = QPushButton(txt)
            b.setFixedSize(32, 28)
            b.setToolTip(tip)
            b.setStyleSheet("""
                QPushButton{background:#111827;border:1px solid #1e2d45;
                            border-radius:4px;color:#94a3b8;font-size:12px;}
                QPushButton:hover{background:#00d4ff;color:#000;}
            """)
            return b

        btn_fit  = tb_btn("⊡", "Fit to window (F)")
        btn_1to1 = tb_btn("1:1", "Actual pixels (cap ở fit nếu ảnh > viewport)")
        btn_in   = tb_btn("+", "Zoom in")
        btn_out  = tb_btn("−", "Zoom out")
        btn_fit.clicked.connect(self._fit)
        btn_1to1.clicked.connect(self._zoom_actual)
        btn_in.clicked.connect(lambda: self._img_view.set_zoom(self._img_view._scale * 1.5))
        btn_out.clicked.connect(lambda: self._img_view.set_zoom(self._img_view._scale / 1.5))
        for b in (btn_out, btn_in, btn_1to1, btn_fit):
            tl.addWidget(b)
        # Refs để ẩn cùng node combo khi vào multi-view (zoom + Results global
        # không apply cho grid — mỗi cell có combo + zoom riêng).
        self._tb_zoom_btns = (btn_out, btn_in, btn_1to1, btn_fit)

        # Results dropdown — chọn tool nào để overlay annotation lên ảnh gốc.
        # Menu rebuilt động khi mở: list tất cả node có image output, mỗi
        # node 1 checkbox. Ít nhất 1 cái tick → composite mode, base = ảnh gốc
        # (Acquire Image), overlay = diff(input, output) của các node ticked.
        from PySide6.QtWidgets import QToolButton, QMenu, QWidgetAction, QCheckBox
        self._selected_overlays: Dict[str, bool] = {}   # node_id → bật/tắt
        self._btn_results = QToolButton()
        self._btn_results.setText("📊 Results ▾")
        self._btn_results.setPopupMode(QToolButton.InstantPopup)
        self._btn_results.setFixedHeight(28)
        self._btn_results.setToolTip(
            "Results — pick tools để composite annotation lên ảnh gốc "
            "Acquire/Camera. Single view: 1 ảnh composite tất cả ticked. "
            "Multi-view (⊞): 1 ô per Acquire/Camera root; overlay tick "
            "composite trong mỗi ô. 1 input → 1 ô; 2 cameras → 2 ô.")
        self._btn_results.setStyleSheet("""
            QToolButton{background:#111827;border:1px solid #1e2d45;
                        border-radius:4px;color:#94a3b8;font-size:11px;
                        padding:0 10px;font-weight:600;}
            QToolButton:hover{background:#1a2236;color:#00d4ff;}
            QToolButton::menu-indicator{image:none;}
        """)

        self._results_menu = QMenu(self._btn_results)
        self._results_menu.setStyleSheet(
            "QMenu{background:#0d1220;border:1px solid #1e2d45;"
            "padding:4px;color:#e2e8f0;}"
            "QMenu::separator{height:1px;background:#1e2d45;margin:4px 6px;}")
        self._results_menu.aboutToShow.connect(self._rebuild_results_menu)
        self._btn_results.setMenu(self._results_menu)
        tl.addWidget(self._btn_results)

        # Multi-view toggle — 2 modes:
        # • Auto: 1 ô per Acquire/Camera root (mặc định).
        # • Custom: user thêm view (right-click canvas), mỗi view link
        #   tới 1 node cụ thể qua combo. Mode + view list + overlay ticks
        #   per cell đều lưu trong graph.ui_state.
        from PySide6.QtWidgets import QToolButton, QMenu as _QMenu
        from PySide6.QtGui import QActionGroup
        self._multi_mode: str = "auto"
        self._custom_views: List[str] = []   # node_id list cho Custom mode
        # Persisted overlay ticks per cell (composite results trong cell):
        # • _multi_overlays_auto: dict[root_id] = {overlay_nid: True}
        # • _multi_overlays_custom: list parallel với _custom_views,
        #   _multi_overlays_custom[i] = {overlay_nid: True}
        self._multi_overlays_auto: Dict[str, Dict[str, bool]] = {}
        self._multi_overlays_custom: List[Dict[str, bool]] = []
        self._btn_multi = tb_btn(
            "⊞",
            "Toggle multi-view. Mode chọn ở nút bên cạnh (Auto/Custom).\n"
            "Custom mode: right-click vào canvas để thêm view, "
            "✖ trên header ô để xóa.")
        self._btn_multi.setCheckable(True)
        self._btn_multi.toggled.connect(self._on_multi_toggled)
        tl.addWidget(self._btn_multi)

        # Mode selector — pill button hiển thị mode hiện tại + ▾ rõ ràng.
        self._btn_multi_mode = QToolButton()
        self._btn_multi_mode.setText("Auto ▾")
        self._btn_multi_mode.setPopupMode(QToolButton.InstantPopup)
        self._btn_multi_mode.setFixedHeight(28)
        self._btn_multi_mode.setStyleSheet("""
            QToolButton{background:#0d1220;border:1px solid #1e2d45;
                        border-radius:4px;color:#94a3b8;font-size:11px;
                        padding:0 8px;font-weight:600;}
            QToolButton:hover{background:#1a2236;color:#00d4ff;
                              border-color:#00d4ff;}
            QToolButton::menu-indicator{image:none;width:0px;}
        """)
        self._btn_multi_mode.setToolTip(
            "Chọn layout mode cho multi-view:\n"
            "• Auto — 1 ô per Acquire/Camera root (auto theo input).\n"
            "• Custom — tự thêm view qua right-click canvas, mỗi view "
            "link tới node bất kỳ.")
        _mm = _QMenu(self._btn_multi_mode)
        _mm.setStyleSheet(
            "QMenu{background:#0d1220;border:1px solid #1e2d45;"
            "padding:4px;color:#e2e8f0;}"
            "QMenu::item{padding:6px 16px;}"
            "QMenu::item:selected{background:#1a2236;color:#00d4ff;}")
        _mg = QActionGroup(self._btn_multi_mode)
        self._act_mode_auto   = _mm.addAction("Auto — 1 ô per Acquire/Camera")
        self._act_mode_custom = _mm.addAction("Custom — tự thêm view (right-click)")
        for _a in (self._act_mode_auto, self._act_mode_custom):
            _a.setCheckable(True)
            _mg.addAction(_a)
        self._act_mode_auto.setChecked(True)
        self._act_mode_auto.triggered.connect(lambda: self._set_multi_mode("auto"))
        self._act_mode_custom.triggered.connect(lambda: self._set_multi_mode("custom"))
        self._btn_multi_mode.setMenu(_mm)
        tl.addWidget(self._btn_multi_mode)

        lay.addWidget(tb)

        # ── Image view ─────────────────────────────────────────────
        # Stack: index 0 = single, index 1 = multi-grid
        from PySide6.QtWidgets import QStackedWidget, QGridLayout
        self._view_stack = QStackedWidget()
        self._img_view = ZoomableImageWidget()
        self._img_view.pixel_info.connect(self._on_pixel_info)
        self._view_stack.addWidget(self._img_view)

        # Multi-view grid (built lazily — refresh khi toggle / graph changes)
        self._multi_container = QWidget()
        self._multi_grid = QGridLayout(self._multi_container)
        self._multi_grid.setContentsMargins(0, 0, 0, 0)
        self._multi_grid.setSpacing(2)
        # Right-click canvas → context menu (Add View trong Custom mode).
        # Cells và inner views cũng được wire trong _rebuild_multi_grid để
        # right-click trên bất kỳ phần nào của vùng grid đều mở menu.
        self._multi_container.setContextMenuPolicy(Qt.CustomContextMenu)
        self._multi_container.customContextMenuRequested.connect(
            lambda pos: self._on_multi_canvas_menu(self._multi_container, pos))
        # Mỗi entry là dict {root, cell_widget, view, combo, status, root_lbl}.
        # `root` = default Acquire/Camera root cho ô (dùng để fallback combo
        # khi node đang chọn bị xóa); user có thể switch combo sang node thuộc
        # bất kỳ pipeline nào, header sẽ cập nhật theo.
        self._multi_views: List[dict] = []
        self._view_stack.addWidget(self._multi_container)

        lay.addWidget(self._view_stack, 1)

        # ── Status bar ─────────────────────────────────────────────
        status = QWidget()
        status.setFixedHeight(24)
        status.setStyleSheet("background:#060a14; border-top:1px solid #1e2d45;")
        sl = QHBoxLayout(status)
        sl.setContentsMargins(10, 0, 10, 0)
        sl.setSpacing(16)

        self._lbl_size   = QLabel("—")
        self._lbl_pixel  = QLabel("Hover over image for pixel info")
        self._lbl_status = QLabel("IDLE")
        for lbl in (self._lbl_size, self._lbl_pixel, self._lbl_status):
            lbl.setStyleSheet(
                "color:#1e2d45; font-size:10px; font-family:'Courier New';")
        sl.addWidget(self._lbl_size)
        sl.addWidget(QLabel("|"))
        sl.addWidget(self._lbl_pixel, 1)
        sl.addWidget(QLabel("|"))
        sl.addWidget(self._lbl_status)
        lay.addWidget(status)

    # ── Public API ────────────────────────────────────────────────
    def set_graph(self, graph: FlowGraph):
        self._graph = graph
        self._restoring_state = True
        # Restore Results toggles từ pipeline file. Chỉ giữ entry còn
        # node tương ứng trong graph (graph có thể đã đổi schema giữa
        # save & load → drop stale node_id).
        saved: Dict[str, bool] = {}
        ui = getattr(graph, "ui_state", None) if graph else None
        if isinstance(ui, dict):
            raw = ui.get("selected_overlays")
            if isinstance(raw, dict):
                saved = {nid: bool(v) for nid, v in raw.items()
                         if nid in graph.nodes}
        self._selected_overlays = saved
        # Restore multi-view mode + custom view list + overlay ticks
        self._multi_mode = "auto"
        self._custom_views = []
        self._multi_overlays_auto = {}
        self._multi_overlays_custom = []
        if isinstance(ui, dict):
            mode = ui.get("multi_view_mode")
            if mode in ("auto", "custom"):
                self._multi_mode = mode
            cv = ui.get("multi_custom_views")
            if isinstance(cv, list):
                self._custom_views = [nid for nid in cv
                                      if isinstance(nid, str)
                                      and nid in graph.nodes]
            oa = ui.get("multi_overlays_auto")
            if isinstance(oa, dict):
                for root, nids in oa.items():
                    if not isinstance(root, str) or root not in graph.nodes:
                        continue
                    if isinstance(nids, list):
                        self._multi_overlays_auto[root] = {
                            n: True for n in nids
                            if isinstance(n, str) and n in graph.nodes}
            oc = ui.get("multi_overlays_custom")
            if isinstance(oc, list):
                for nids in oc:
                    if isinstance(nids, list):
                        self._multi_overlays_custom.append({
                            n: True for n in nids
                            if isinstance(n, str) and n in graph.nodes})
                    else:
                        self._multi_overlays_custom.append({})
        # Pad/trim overlays_custom để match _custom_views length
        while len(self._multi_overlays_custom) < len(self._custom_views):
            self._multi_overlays_custom.append({})
        del self._multi_overlays_custom[len(self._custom_views):]
        try:
            self._act_mode_auto.setChecked(self._multi_mode == "auto")
            self._act_mode_custom.setChecked(self._multi_mode == "custom")
            self._btn_multi_mode.setText(
                "Custom ▾" if self._multi_mode == "custom" else "Auto ▾")
        except Exception:
            pass
        # Restore multi-view enabled state (Multi-view ON/OFF). Set qua
        # blockSignals để khỏi trigger _on_multi_toggled emit state_changed
        # (đang trong khoảng _restoring_state vẫn được _on_multi_toggled
        # respect, nhưng cẩn thận hai lớp cho chắc).
        want_multi = False
        if isinstance(ui, dict):
            want_multi = bool(ui.get("multi_view_enabled", False))
        if self._btn_multi.isChecked() != want_multi:
            self._btn_multi.blockSignals(True)
            self._btn_multi.setChecked(want_multi)
            self._btn_multi.blockSignals(False)
            # Đồng bộ visibility + view_stack vì _on_multi_toggled bị block
            self._on_multi_toggled(want_multi)
        self._update_results_btn_text()
        if self._btn_multi.isChecked():
            self._rebuild_multi_grid()
        self._restoring_state = False

    def _persist_overlays(self):
        """Ghi `_selected_overlays` vào graph.ui_state để save pipeline
        sẽ lưu lại. Chỉ giữ entry True để file gọn."""
        if not self._graph or self._restoring_state:
            return
        ui = getattr(self._graph, "ui_state", None)
        if not isinstance(ui, dict):
            self._graph.ui_state = {}
            ui = self._graph.ui_state
        on = {nid: True for nid, v in self._selected_overlays.items() if v}
        if on:
            ui["selected_overlays"] = on
        else:
            ui.pop("selected_overlays", None)
        self.state_changed.emit()

    def refresh_node_list(self):
        """Cập nhật dropdown list các node có output image."""
        if not self._graph:
            return
        self._node_combo.blockSignals(True)
        prev = self._current_node_id
        self._node_combo.clear()
        self._node_map = {}
        self._node_combo.addItem("— Select node —", None)

        for nid, node in self._graph.nodes.items():
            has_img_output = any(p.name == "image" for p in node.tool.outputs)
            if has_img_output:
                label = f"{node.tool.icon} {node.tool.name}  [{nid}]"
                self._node_combo.addItem(label, nid)
                self._node_map[label] = nid

        # Restore selection — ưu tiên prev (node user đang xem trước đó).
        # Nếu prev đã biến mất hoặc chưa từng chọn → fallback về Acquire
        # Image (hoặc Camera Acquire) để mở app/pipeline lên là thấy ảnh
        # nguồn ngay, không phải pick từ "— Select node —".
        target_idx = -1
        if prev:
            for i in range(self._node_combo.count()):
                if self._node_combo.itemData(i) == prev:
                    target_idx = i
                    break
        if target_idx < 0:
            # Tìm node đầu tiên thuộc category "Acquire Image", ưu tiên
            # tool_id "acquire_image" → "camera_acquire" → khác.
            def _rank(nid):
                n = self._graph.nodes.get(nid)
                if not n or getattr(n.tool, "category", "") != "Acquire Image":
                    return 99
                return {"acquire_image": 0, "camera_acquire": 1}.get(
                    n.tool.tool_id, 2)
            best = (99, -1)
            for i in range(1, self._node_combo.count()):
                nid = self._node_combo.itemData(i)
                r = _rank(nid)
                if r < best[0]:
                    best = (r, i)
            if best[1] > 0:
                target_idx = best[1]
        if target_idx > 0:
            self._node_combo.setCurrentIndex(target_idx)
            self._current_node_id = self._node_combo.itemData(target_idx)

        self._node_combo.blockSignals(False)
        if self._btn_multi.isChecked():
            self._rebuild_multi_grid()
        elif self._current_node_id:
            self._display_node(self._current_node_id)

    def show_node(self, node_id: str):
        """Hiển thị output image của node_id."""
        for i in range(self._node_combo.count()):
            if self._node_combo.itemData(i) == node_id:
                self._node_combo.setCurrentIndex(i)
                break
        self._display_node(node_id)

    def refresh_current(self):
        """Refresh ảnh của node đang xem."""
        if self._current_node_id:
            self._display_node(self._current_node_id)
        if self._btn_multi.isChecked():
            self._refresh_multi_views()

    # ── Multi-view ────────────────────────────────────────────────
    def _on_multi_toggled(self, checked: bool):
        """Toggle giữa single view và multi-view grid. Ẩn các widget single-
        view (node combo + zoom + global Results) khi vào grid — mỗi ô trong
        grid đã có combo + zoom + 📊 riêng nên top toolbar bị thừa."""
        single_only = not checked
        self._tb_node_lbl.setVisible(single_only)
        self._node_combo.setVisible(single_only)
        for b in self._tb_zoom_btns:
            b.setVisible(single_only)
        self._btn_results.setVisible(single_only)
        if checked:
            self._rebuild_multi_grid()
            self._view_stack.setCurrentIndex(1)
        else:
            self._view_stack.setCurrentIndex(0)
        # Persist enabled state để reopen pipeline khôi phục đúng view mode.
        if self._graph and not self._restoring_state:
            ui = getattr(self._graph, "ui_state", None)
            if not isinstance(ui, dict):
                self._graph.ui_state = {}
                ui = self._graph.ui_state
            ui["multi_view_enabled"] = bool(checked)
            self.state_changed.emit()

    # Tool IDs nhận diện 2 loại "Acquire Image" pipeline:
    #   acquire_image  → file-based (folder/file load) → header "Acquire Image"
    #   camera_acquire → camera-based (OpenCV/HikRobot) → header "Camera Image"
    _FILE_ACQUIRE_ID = "acquire_image"
    _CAMERA_ACQUIRE_ID = "camera_acquire"

    def _enumerate_branch_roots(self) -> List[str]:
        """Pipeline roots = node category 'Acquire Image' (tool_id
        `acquire_image` cho file, `camera_acquire` cho camera). Sort: file
        trước, camera sau, stable theo node_id trong mỗi nhóm.

        Lọc category thay vì 'has image-out & no image-in' để tránh leak các
        tool có image-input optional (vd Area Measure nối qua `mask`/
        `contours` thay vì port `image`) thành false-positive root.
        """
        if self._graph is None:
            return []
        roots = [nid for nid, node in self._graph.nodes.items()
                 if getattr(node.tool, "category", "") == "Acquire Image"]

        def _order(nid: str) -> int:
            tid = self._graph.nodes[nid].tool.tool_id
            if tid == self._FILE_ACQUIRE_ID:
                return 0
            if tid == self._CAMERA_ACQUIRE_ID:
                return 1
            return 2

        roots.sort(key=lambda nid: (_order(nid), nid))
        return roots

    def _root_pipeline_label(self, root_id: str) -> str:
        """Header label cho pipeline gốc — 'Acquire Image' (file) hoặc
        'Camera Image' (camera). Dùng cho cell header trong multi-view và
        section trong combo dropdown."""
        node = self._graph.nodes.get(root_id) if self._graph else None
        if node is None:
            return "?"
        if node.tool.tool_id == self._CAMERA_ACQUIRE_ID:
            return "Camera Image"
        if node.tool.tool_id == self._FILE_ACQUIRE_ID:
            return "Acquire Image"
        return node.tool.name

    def _node_pipeline_root(self, node_id: str) -> Optional[str]:
        """Tìm Acquire/Camera root mà node thuộc về (đi ngược upstream theo
        port `image`). Trả None nếu node không thuộc pipeline Acquire/Camera
        nào (vd dangling tool)."""
        if self._graph is None or node_id not in self._graph.nodes:
            return None
        roots = set(self._enumerate_branch_roots())
        if node_id in roots:
            return node_id
        visited = set()
        cur = node_id
        for _ in range(64):
            if cur in visited:
                break
            visited.add(cur)
            if cur in roots:
                return cur
            upstream = None
            for c in self._graph.connections:
                if c.dst_id == cur and c.dst_port == "image":
                    upstream = c.src_id
                    break
            if upstream is None:
                return None
            cur = upstream
        return None

    def _branch_terminal(self, root_id: str) -> str:
        """BFS xuôi dòng từ root theo image connections → trả về node cuối
        cùng (xa nhất từ root) có image output. Khi pipeline rẽ nhánh
        → chọn nhánh dài nhất.
        """
        if self._graph is None:
            return root_id
        img_outs: Dict[str, List[str]] = {}
        for c in self._graph.connections:
            if c.src_port == "image" and c.dst_port == "image":
                img_outs.setdefault(c.src_id, []).append(c.dst_id)
        # BFS với depth tracking → terminal = node depth lớn nhất có image out
        best = (0, root_id)
        visited = {root_id}
        queue = [(root_id, 0)]
        while queue:
            cur, depth = queue.pop(0)
            for dst in img_outs.get(cur, []):
                if dst in visited:
                    continue
                visited.add(dst)
                node = self._graph.nodes.get(dst)
                if node and "image" in {p.name for p in node.tool.outputs}:
                    if depth + 1 > best[0]:
                        best = (depth + 1, dst)
                queue.append((dst, depth + 1))
        return best[1]

    def _branch_image_nodes(self, root_id: str) -> List[str]:
        """BFS xuôi dòng từ root → list mọi node có image output trong branch.
        Dùng để populate dropdown trong mỗi ô của multi-view.
        """
        if self._graph is None:
            return []
        img_outs: Dict[str, List[str]] = {}
        for c in self._graph.connections:
            if c.src_port == "image" and c.dst_port == "image":
                img_outs.setdefault(c.src_id, []).append(c.dst_id)
        result = []
        seen = set()
        queue = [root_id]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            node = self._graph.nodes.get(cur)
            if node and "image" in {p.name for p in node.tool.outputs}:
                result.append(cur)
            for dst in img_outs.get(cur, []):
                if dst not in seen:
                    queue.append(dst)
        return result

    def _plan_multi_cells(self, roots: List[str]) -> List[tuple]:
        """Quyết định cells = list (root_default, node_default).
        1 ô per Acquire/Camera root, default = terminal node trong branch.
        Số ô = số đầu vào ảnh (1 input → 1 ô, 2 inputs → 2 ô, ...).
        Cap ở 9 cells để tránh grid quá đông.
        """
        if not roots:
            return []
        return [(r, self._branch_terminal(r)) for r in roots][:9]

    def _multi_cells_plan(self) -> List[tuple]:
        """Cells trong multi-view = list (root_id, node_id).
        Mode:
          • auto:   1 ô per Acquire/Camera root (default behavior).
          • custom: 1 ô per entry trong self._custom_views (user thêm/xóa).
        Cap 9 ô. Pipeline root của ô dùng để hiện label + group menu.
        """
        if self._graph is None:
            return []
        if self._multi_mode == "custom":
            cells: List[tuple] = []
            for nid in self._custom_views[:9]:
                if nid not in self._graph.nodes:
                    # Node bị xoá → giữ placeholder (root = self) để cell
                    # hiển thị "missing" thay vì crash.
                    cells.append((nid, nid))
                    continue
                root = self._node_pipeline_root(nid) or nid
                cells.append((root, nid))
            return cells
        return self._plan_multi_cells(self._enumerate_branch_roots())

    def _set_multi_mode(self, mode: str):
        """Switch giữa "auto" và "custom". Khi vào custom lần đầu mà
        _custom_views rỗng → seed bằng cells của auto mode để user thấy
        ngay layout quen thuộc, sau đó tự edit."""
        if mode not in ("auto", "custom"):
            return
        prev = self._multi_mode
        self._multi_mode = mode
        if mode == "custom" and not self._custom_views:
            # Seed từ auto plan để user có điểm bắt đầu
            for _root, nid in self._plan_multi_cells(
                    self._enumerate_branch_roots()):
                if nid:
                    self._custom_views.append(nid)
                    self._multi_overlays_custom.append({})
        # Sync menu radio + pill label
        try:
            self._act_mode_auto.setChecked(mode == "auto")
            self._act_mode_custom.setChecked(mode == "custom")
            self._btn_multi_mode.setText(
                "Custom ▾" if mode == "custom" else "Auto ▾")
        except Exception:
            pass
        # Multi-view chưa bật → bật luôn khi user pick mode (UX: click vào
        # selector mà không cần nhấn ⊞ trước)
        if not self._btn_multi.isChecked():
            self._btn_multi.setChecked(True)   # → triggers _on_multi_toggled
        elif prev != mode:
            self._rebuild_multi_grid()
        self._persist_multi_state()

    def _persist_multi_state(self):
        """Ghi multi_mode + custom_views + per-cell overlay ticks vào
        graph.ui_state để save pipeline lưu lại."""
        if not self._graph or self._restoring_state:
            return
        ui = getattr(self._graph, "ui_state", None)
        if not isinstance(ui, dict):
            self._graph.ui_state = {}
            ui = self._graph.ui_state
        ui["multi_view_mode"] = self._multi_mode
        # Chỉ lưu node_ids còn tồn tại để file gọn (xóa node ngoài UI sẽ
        # auto-rớt ra khỏi state lần save kế).
        nids = self._graph.nodes
        ui["multi_custom_views"] = [n for n in self._custom_views if n in nids]
        # Per-cell overlay ticks. Chỉ keep True keys + node còn tồn tại.
        ui["multi_overlays_auto"] = {
            root: [nid for nid, on in ov.items() if on and nid in nids]
            for root, ov in self._multi_overlays_auto.items()
            if root in nids and any(on and nid in nids
                                    for nid, on in ov.items())
        }
        ui["multi_overlays_custom"] = [
            [nid for nid, on in ov.items() if on and nid in nids]
            for ov in self._multi_overlays_custom
        ]
        self.state_changed.emit()

    def _rebuild_multi_grid(self):
        """Detect Acquire/Camera branches và build grid ZoomableImageWidget.
        Mỗi ô có combo chọn node từ BẤT KỲ Acquire/Camera branch nào — user
        có thể tự chọn 'view nào' (Acquire Image hoặc Camera Image) cho từng
        ô độc lập."""
        # Clear old widgets
        for cell in self._multi_views:
            cell["cell_widget"].setParent(None)
            cell["cell_widget"].deleteLater()
        self._multi_views = []

        cells_plan = self._multi_cells_plan()
        n = len(cells_plan)
        if n == 0:
            return

        # Grid layout: 1 → 1×1; 2 → 1×2; 3-4 → 2×2; 5-6 → 2×3; 7-9 → 3×3
        if n <= 1:    cols = 1
        elif n <= 2:  cols = 2
        elif n <= 6:  cols = (n + 1) // 2
        else:         cols = 3

        from PySide6.QtWidgets import (QVBoxLayout as _QV, QHBoxLayout as _QH,
                                       QToolButton, QMenu)
        for i, (root_id, default_nid) in enumerate(cells_plan):
            cell = QWidget()
            cell_lay = _QV(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(0)

            # Header: pipeline label + node combo + per-cell Results button + status
            hdr = QWidget()
            hdr.setStyleSheet(
                "background:#060a14;border-bottom:1px solid #1e2d45;")
            hl = _QH(hdr)
            hl.setContentsMargins(6, 3, 6, 3); hl.setSpacing(6)
            root_lbl = QLabel("")
            root_lbl.setTextFormat(Qt.RichText)
            hl.addWidget(root_lbl)

            cb = QComboBox()
            cb.setStyleSheet("""
                QComboBox{background:#0a0e1a;border:1px solid #1e2d45;
                          color:#e2e8f0;padding:1px 6px;border-radius:3px;
                          font-size:10px;}
                QComboBox::drop-down{border:none;}
                QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;
                                             border:1px solid #1e2d45;
                                             selection-background-color:#1a2236;}
            """)
            cb.setToolTip(
                "Chọn base node hiển thị trong ô — list gom cả Acquire Image "
                "và Camera Image branches.")
            self._populate_cell_combo(cb, default_nid)
            cb.currentIndexChanged.connect(
                lambda _idx, idx=i: self._on_multi_cell_changed(idx))
            hl.addWidget(cb, 1)

            # Per-cell Results button: pick overlay results để composite lên
            # base của ô này. Menu group theo pipeline (Acquire/Camera) như
            # global Results, nhưng selection độc lập cho từng ô.
            cell_results_btn = QToolButton()
            cell_results_btn.setText("📊")
            cell_results_btn.setPopupMode(QToolButton.InstantPopup)
            cell_results_btn.setFixedHeight(22)
            cell_results_btn.setToolTip(
                "Pick result(s) để composite lên ô này. Tick 1+ item → "
                "overlay annotation lên base image của pipeline. Không tick "
                "gì → chỉ hiện base node (combo bên trái).")
            cell_results_btn.setStyleSheet("""
                QToolButton{background:#111827;border:1px solid #1e2d45;
                            border-radius:3px;color:#94a3b8;font-size:11px;
                            padding:0 6px;font-weight:600;}
                QToolButton:hover{background:#1a2236;color:#00d4ff;}
                QToolButton::menu-indicator{image:none;}
            """)
            cell_menu = QMenu(cell_results_btn)
            cell_menu.setStyleSheet(
                "QMenu{background:#0d1220;border:1px solid #1e2d45;"
                "padding:4px;color:#e2e8f0;}"
                "QMenu::separator{height:1px;background:#1e2d45;margin:4px 6px;}")
            cell_results_btn.setMenu(cell_menu)
            hl.addWidget(cell_results_btn)

            status_lbl = QLabel("●")
            status_lbl.setStyleSheet("color:#64748b;font-size:11px;")
            hl.addWidget(status_lbl)

            # Custom mode: thêm "×" remove button → xóa view khỏi
            # _custom_views. Auto mode không cần (cell count auto theo root).
            if self._multi_mode == "custom":
                btn_x = QToolButton()
                btn_x.setText("✖")
                btn_x.setFixedSize(18, 18)
                btn_x.setToolTip("Remove view này")
                btn_x.setStyleSheet(
                    "QToolButton{background:transparent;border:none;"
                    "color:#94a3b8;font-size:11px;font-weight:700;}"
                    "QToolButton:hover{color:#ff3860;}")
                btn_x.clicked.connect(
                    lambda _checked=False, idx=i: self._remove_custom_view(idx))
                hl.addWidget(btn_x)

            cell_lay.addWidget(hdr)

            view = ZoomableImageWidget()
            cell_lay.addWidget(view, 1)

            r, c = divmod(i, cols)
            self._multi_grid.addWidget(cell, r, c)

            # Seed cell_overlays từ persisted state (theo mode).
            if self._multi_mode == "custom":
                if i < len(self._multi_overlays_custom):
                    saved_ov = dict(self._multi_overlays_custom[i])
                else:
                    saved_ov = {}
            else:
                saved_ov = dict(self._multi_overlays_auto.get(root_id, {}))
            # Loại nids đã bị xoá khỏi graph để khỏi composite stale
            saved_ov = {k: v for k, v in saved_ov.items()
                        if v and k in self._graph.nodes}

            entry = {
                "root": root_id, "cell_widget": cell,
                "view": view, "combo": cb, "status": status_lbl,
                "root_lbl": root_lbl,
                "cell_overlays": saved_ov,
                "cell_results_btn": cell_results_btn,
                "cell_menu": cell_menu,
                "cell_index": i,
            }
            self._multi_views.append(entry)
            # Rebuild menu khi mở → reflect graph hiện tại + checked state
            cell_menu.aboutToShow.connect(
                lambda _entry=entry: self._rebuild_cell_results_menu(_entry))
            # Right-click trên cell / inner view → cùng canvas menu để user
            # add view ngay cả khi grid đã đầy hint area.
            for _w in (cell, view):
                _w.setContextMenuPolicy(Qt.CustomContextMenu)
                _w.customContextMenuRequested.connect(
                    lambda pos, _src=_w:
                        self._on_multi_canvas_menu(_src, pos))
            # Đồng bộ overlay state về persistent store ngay lần đầu (để
            # nếu user vừa rebuild xong là save thì state vẫn lưu được).
            self._sync_cell_overlays_to_store(entry)

        self._refresh_multi_views()

    def _populate_cell_combo(self, cb: QComboBox, default_nid: Optional[str]):
        """Fill combo với tất cả image nodes từ MỌI Acquire/Camera branch,
        group theo pipeline. Đặt mặc định ở `default_nid` nếu có."""
        cb.blockSignals(True)
        cb.clear()
        roots = self._enumerate_branch_roots()
        for root in roots:
            label = self._root_pipeline_label(root)
            # Section separator để user phân biệt pipelines trong dropdown
            if cb.count() > 0:
                cb.insertSeparator(cb.count())
            head_idx = cb.count()
            cb.addItem(f"── {label} ──", None)
            # Disable head row (visual section header only)
            model = cb.model()
            from PySide6.QtCore import Qt as _Qt
            item = model.item(head_idx)
            if item is not None:
                item.setFlags(item.flags() & ~_Qt.ItemIsEnabled
                              & ~_Qt.ItemIsSelectable)
                item.setData("color:#64748b;font-style:italic;",
                             _Qt.ToolTipRole)
            for nid in self._branch_image_nodes(root):
                node = self._graph.nodes.get(nid)
                if not node:
                    continue
                cb.addItem(f"  {node.tool.icon} {node.tool.name}", nid)
        # Default select
        if default_nid is not None:
            for j in range(cb.count()):
                if cb.itemData(j) == default_nid:
                    cb.setCurrentIndex(j); break
        cb.blockSignals(False)

    def _on_multi_cell_changed(self, cell_idx: int):
        """User pick node khác cho ô `cell_idx` → load image + sync header.
        Custom mode: cập nhật _custom_views[idx] và persist."""
        if 0 <= cell_idx < len(self._multi_views):
            entry = self._multi_views[cell_idx]
            self._push_multi_cell(entry)
            if self._multi_mode == "custom":
                nid = entry["combo"].currentData()
                if nid and cell_idx < len(self._custom_views):
                    self._custom_views[cell_idx] = nid
                    self._persist_multi_state()

    def _add_custom_view(self):
        """Custom mode: thêm 1 view mới. Mặc định pick acquire root đầu tiên
        nếu có; nếu không có pipeline → image node đầu tiên."""
        if not self._graph:
            return
        roots = self._enumerate_branch_roots()
        default_nid = roots[0] if roots else None
        if default_nid is None:
            # Fallback: bất kỳ node nào có output image
            for nid, n in self._graph.nodes.items():
                if any(p.name == "image" for p in n.tool.outputs):
                    default_nid = nid
                    break
        if default_nid is None:
            return
        if len(self._custom_views) >= 9:
            return  # cap 9 cells
        self._custom_views.append(default_nid)
        self._multi_overlays_custom.append({})
        self._persist_multi_state()
        if self._btn_multi.isChecked():
            self._rebuild_multi_grid()

    def _remove_custom_view(self, cell_idx: int):
        """Custom mode: xóa view ở vị trí cell_idx (đồng thời rớt overlay
        ticks tương ứng)."""
        if 0 <= cell_idx < len(self._custom_views):
            self._custom_views.pop(cell_idx)
            if cell_idx < len(self._multi_overlays_custom):
                self._multi_overlays_custom.pop(cell_idx)
            self._persist_multi_state()
            if self._btn_multi.isChecked():
                self._rebuild_multi_grid()

    def _cell_pipeline_root(self, entry) -> Optional[str]:
        """Pipeline gốc của ô = pipeline của base node đang chọn ở combo
        (nếu node thuộc Acquire/Camera branch), fallback về entry['root']."""
        nid = entry["combo"].currentData() if "combo" in entry else None
        if nid:
            r = self._node_pipeline_root(nid)
            if r is not None:
                return r
        return entry.get("root")

    def _cell_branch_tools(self, entry) -> List[tuple]:
        """List (nid, node) các tool thuộc pipeline của ô — exclude Acquire
        root (đó là base, không phải tool overlay)."""
        if self._graph is None:
            return []
        root = self._cell_pipeline_root(entry)
        if root is None:
            return []
        tools = []
        for nid in self._branch_image_nodes(root):
            node = self._graph.nodes.get(nid)
            if (node is None
                    or getattr(node.tool, "category", "") == "Acquire Image"):
                continue
            tools.append((nid, node))
        return tools

    def _rebuild_cell_results_menu(self, entry):
        """Menu Results của 1 cell — CHỈ list tool thuộc pipeline của ô đó
        (Acquire Image branch HOẶC Camera Image branch, tùy combo base
        node). Tránh user phải scroll tìm tool giữa nhiều pipeline."""
        from PySide6.QtWidgets import QWidgetAction, QCheckBox, QLabel
        from PySide6.QtGui import QAction
        menu = entry['cell_menu']
        menu.clear()
        if not self._graph:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (No pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
            return

        root = self._cell_pipeline_root(entry)
        section_label = (self._root_pipeline_label(root) if root
                         else "—")

        # Header — ghi rõ ô đang là pipeline nào
        wa_hdr = QWidgetAction(menu)
        hdr = QLabel(f"  Overlay cho {section_label}  ")
        hdr.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; "
            "letter-spacing:1px; padding:6px 8px;")
        wa_hdr.setDefaultWidget(hdr)
        menu.addAction(wa_hdr)
        menu.addSeparator()

        tools = self._cell_branch_tools(entry)
        # Loại node combo đang chọn khỏi menu: combo đã render trực tiếp
        # annotation của node đó qua _vis_of → tick lại trong overlay sẽ
        # KHÔNG có hiệu ứng visible (composite layer cùng annotation lên
        # chính nó). Tránh user confused "tick mà không thấy gì".
        current_nid = entry["combo"].currentData() if "combo" in entry else None
        if current_nid:
            tools = [t for t in tools if t[0] != current_nid]
        if not tools:
            wa = QWidgetAction(menu)
            msg = ("(Pipeline chưa có tool nào)" if root
                   else "(Pick base node ở combo trước)")
            lbl = QLabel(f"  {msg}  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
        else:
            for nid, node in tools:
                wa = QWidgetAction(menu)
                cb = QCheckBox(f"  {node.tool.icon}  {node.tool.name}  "
                                f"({node.tool.tool_id})")
                cb.setChecked(entry['cell_overlays'].get(nid, False))
                cb.setStyleSheet(
                    "QCheckBox{color:#e2e8f0; font-size:11px; padding:4px 8px;}"
                    "QCheckBox::indicator{width:14px; height:14px;}")
                cb.toggled.connect(
                    lambda on, _nid=nid, _entry=entry:
                        self._on_cell_overlay_toggled(_entry, _nid, on))
                wa.setDefaultWidget(cb)
                menu.addAction(wa)

        menu.addSeparator()
        act_clear = QAction("✗  Clear All", menu)
        act_clear.triggered.connect(
            lambda _checked=False, _entry=entry: self._clear_cell_overlays(_entry))
        menu.addAction(act_clear)

    def _active_cell_overlays(self, entry) -> List[str]:
        """Overlay node_ids đang ACTIVE cho ô — merge:
        • Global ticks (Results dropdown chính của ImageViewerPanel)
        • Cell-level ticks (📊 button trong từng ô)
        Cả hai chỉ tính item thuộc pipeline hiện tại của ô (drop stale từ
        pipeline khác). Composite lên base image của combo trong ô."""
        if not self._graph:
            return []
        tool_ids = {nid for nid, _ in self._cell_branch_tools(entry)}
        cell_ticks = entry.get('cell_overlays', {})
        merged: List[str] = []
        for nid in tool_ids:
            on = cell_ticks.get(nid) or self._selected_overlays.get(nid)
            if on and nid not in merged:
                merged.append(nid)
        return merged

    def _on_cell_overlay_toggled(self, entry, node_id: str, on: bool):
        entry['cell_overlays'][node_id] = on
        self._sync_cell_overlays_to_store(entry)
        self._persist_multi_state()
        self._update_cell_results_btn(entry)
        self._push_multi_cell(entry)

    def _sync_cell_overlays_to_store(self, entry):
        """Mirror entry['cell_overlays'] → persistent store theo mode.
        Auto: keyed by root_id; Custom: keyed by cell index. Chỉ giữ True
        keys + node hiện còn trong graph để khỏi rác."""
        if not self._graph:
            return
        nodes = self._graph.nodes
        cleaned = {nid: True for nid, on in entry['cell_overlays'].items()
                   if on and nid in nodes}
        if self._multi_mode == "custom":
            idx = entry.get("cell_index", -1)
            if idx < 0:
                return
            # Mở rộng list nếu chưa đủ
            while len(self._multi_overlays_custom) <= idx:
                self._multi_overlays_custom.append({})
            self._multi_overlays_custom[idx] = cleaned
        else:
            root_id = entry.get("root")
            if root_id:
                self._multi_overlays_auto[root_id] = cleaned

    def _on_multi_canvas_menu(self, sender, pos):
        """Right-click canvas → Add View (custom mode) hoặc hint switch
        sang custom mode (auto mode)."""
        from PySide6.QtWidgets import QMenu as _QMenu
        menu = _QMenu(sender)
        menu.setStyleSheet(
            "QMenu{background:#0d1220;border:1px solid #1e2d45;"
            "padding:4px;color:#e2e8f0;}"
            "QMenu::item{padding:6px 14px;}"
            "QMenu::item:selected{background:#1a2236;color:#00d4ff;}"
            "QMenu::item:disabled{color:#475569;}")
        if self._multi_mode == "custom":
            if len(self._custom_views) >= 9:
                a = menu.addAction("Đã đạt cap 9 view")
                a.setEnabled(False)
            else:
                a = menu.addAction("＋  Add View")
                a.triggered.connect(self._add_custom_view)
        else:
            a = menu.addAction("Chuyển sang Custom mode để thêm view")
            a.triggered.connect(lambda: self._set_multi_mode("custom"))
        menu.exec(sender.mapToGlobal(pos))

    def _clear_cell_overlays(self, entry):
        """Clear chỉ overlay của pipeline hiện tại — selection các pipeline
        khác (nếu user đã switch qua-lại) được giữ."""
        active = self._active_cell_overlays(entry)
        for nid in active:
            entry['cell_overlays'][nid] = False
        self._update_cell_results_btn(entry)
        self._push_multi_cell(entry)

    def _update_cell_results_btn(self, entry):
        n = len(self._active_cell_overlays(entry))
        btn = entry['cell_results_btn']
        btn.setText("📊" if n == 0 else f"📊 ({n})")

    def _push_multi_cell(self, entry):
        """Load ảnh + status của node đang chọn trong ô vào view; cập nhật
        header để reflect pipeline gốc. Nếu cell có overlay items ticked
        (📊 button), composite chúng lên base image của pipeline."""
        nid = entry["combo"].currentData()
        node = self._graph.nodes.get(nid) if self._graph and nid else None

        pipeline_root = self._node_pipeline_root(nid) if nid else None
        if pipeline_root:
            label = self._root_pipeline_label(pipeline_root)
        else:
            label = self._root_pipeline_label(entry["root"])
        entry["root_lbl"].setText(f"<b>{label}</b>  →")
        entry["root_lbl"].setStyleSheet("color:#64748b;font-size:10px;")

        if node is None:
            return

        def _vis_of(n):
            v = n.outputs.get("_display_image")
            if v is None:
                v = n.outputs.get("image")
            return v

        # Chỉ apply overlays thuộc pipeline hiện tại của ô — bỏ stale items
        # nếu user đã switch combo qua pipeline khác. Loại luôn combo node
        # khỏi danh sách active vì combo đang RENDER trực tiếp annotation
        # của nó qua _vis_of → tick lại sẽ no-op (redundant).
        active = [oid for oid in self._active_cell_overlays(entry)
                  if oid != nid]

        img = None
        if active:
            # Base = render hiện tại của combo (có annotation của node đó),
            # rồi layer thêm annotation của các overlay khác bên trên. Cho
            # phép user xem "node X output + thêm marker của node Y" — UX
            # rõ ràng hơn so với reset về Acquire image làm base.
            base = _vis_of(node)
            if base is not None and isinstance(base, np.ndarray):
                import cv2
                comp = base.copy()
                if comp.ndim == 2:
                    comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
                for oid in active:
                    on_node = self._graph.nodes[oid]
                    before = self._node_input_image(on_node)
                    after = _vis_of(on_node)
                    if before is not None and after is not None:
                        comp = self._overlay_diff(comp, before, after)
                img = comp
        if img is None:
            img = _vis_of(node)
        if img is not None:
            entry["view"].set_image(img)

        status = getattr(node, "status", "—") or "—"
        color = {"pass": "#39ff14", "fail": "#ff3860",
                 "error": "#ff3860", "running": "#ffd700"}.get(status, "#64748b")
        entry["status"].setStyleSheet(
            f"color:{color};font-size:13px;font-weight:bold;")
        entry["status"].setToolTip(status.upper())
        self._update_cell_results_btn(entry)

    def _refresh_multi_views(self):
        """Push ảnh mới nhất lên từng ô. Cũng resync combo items khi graph
        thay đổi (node mới được thêm/xóa)."""
        if self._graph is None:
            return
        # Snapshot tất cả image nodes hiện tại từ mọi Acquire/Camera root
        roots = self._enumerate_branch_roots()
        current_ids = []
        for r in roots:
            for nid in self._branch_image_nodes(r):
                if nid not in current_ids:
                    current_ids.append(nid)

        for entry in self._multi_views:
            existing_ids = [entry["combo"].itemData(j)
                             for j in range(entry["combo"].count())
                             if entry["combo"].itemData(j) is not None]
            if list(existing_ids) != list(current_ids):
                prev = entry["combo"].currentData()
                self._populate_cell_combo(entry["combo"], prev)
                # Nếu prev không còn tồn tại, fallback về terminal của root
                if entry["combo"].currentData() != prev:
                    term = self._branch_terminal(entry["root"])
                    for j in range(entry["combo"].count()):
                        if entry["combo"].itemData(j) == term:
                            entry["combo"].setCurrentIndex(j); break
            self._push_multi_cell(entry)

    # ── Internal ─────────────────────────────────────────────────
    def _on_node_selected(self, idx: int):
        node_id = self._node_combo.itemData(idx)
        if node_id:
            self._display_node(node_id)
        else:
            self._img_view.set_image(None)
            self._lbl_size.setText("—")
            self._lbl_status.setText("IDLE")
            self._current_node_id = None

    def _rebuild_results_menu(self):
        """Rebuild menu mỗi khi mở → reflect graph hiện tại."""
        from PySide6.QtWidgets import QWidgetAction, QCheckBox, QLabel
        menu = self._results_menu
        menu.clear()
        if not self._graph:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (No pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
            return

        # Header — mode-dependent caption: single view = composite mode,
        # multi-view = mỗi item ticked thành 1 ô riêng.
        wa_hdr = QWidgetAction(menu)
        if self._btn_multi.isChecked():
            hdr_text = "  Multi-view: 1 ô / root + composite overlay  "
        else:
            hdr_text = "  Base: ảnh gốc (Acquire Image) + overlay  "
        hdr = QLabel(hdr_text)
        hdr.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; "
            "letter-spacing:1px; padding:6px 8px;")
        wa_hdr.setDefaultWidget(hdr)
        menu.addAction(wa_hdr)
        menu.addSeparator()

        # Group nodes theo pipeline (Acquire Image / Camera Image) — dễ tìm
        # tool nào thuộc flow nào. Trong mỗi group sort theo BFS order
        # (xuôi dòng từ root), phản ánh đúng flow execution.
        roots = self._enumerate_branch_roots()
        groups: List[tuple] = []   # [(section_label, [(nid, node), ...])]
        accounted: set = set()
        for root in roots:
            section_label = self._root_pipeline_label(root)
            tools = []
            for nid in self._branch_image_nodes(root):
                if nid in accounted:
                    continue
                accounted.add(nid)
                node = self._graph.nodes.get(nid)
                if (node is None
                        or getattr(node.tool, "category", "") == "Acquire Image"):
                    # Skip root acquire/camera node (đó là base, không phải tool)
                    continue
                tools.append((nid, node))
            if tools:
                groups.append((section_label, tools))

        # Nodes không thuộc Acquire/Camera flow nào (vd dangling tool) → group
        # "Other" để vẫn cho user pick được. Check tool output port (static)
        # thay vì n.outputs (chỉ có sau khi pipeline run).
        others = [(nid, n) for nid, n in self._graph.nodes.items()
                  if any(p.name == "image" for p in n.tool.outputs)
                  and nid not in accounted
                  and getattr(n.tool, "category", "") != "Acquire Image"]
        others.sort(key=lambda x: x[0])
        if others:
            groups.append(("Other", others))

        total_tools = sum(len(items) for _, items in groups)
        if total_tools == 0:
            wa = QWidgetAction(menu)
            lbl = QLabel("  (Chưa có tool nào trong pipeline)  ")
            lbl.setStyleSheet("color:#64748b; padding:8px;")
            wa.setDefaultWidget(lbl)
            menu.addAction(wa)
        else:
            for gi, (section_label, items) in enumerate(groups):
                if gi > 0:
                    menu.addSeparator()
                # Section header — pipeline gốc (Acquire Image / Camera Image)
                wa_sec = QWidgetAction(menu)
                sec_lbl = QLabel(f"  ── {section_label} ──  ")
                sec_lbl.setStyleSheet(
                    "color:#94a3b8; font-size:10px; font-weight:600; "
                    "padding:4px 8px; background:#0d1220;")
                wa_sec.setDefaultWidget(sec_lbl)
                menu.addAction(wa_sec)
                for nid, node in items:
                    wa = QWidgetAction(menu)
                    cb = QCheckBox(f"  {node.tool.icon}  {node.tool.name}  "
                                    f"({node.tool.tool_id})")
                    cb.setChecked(self._selected_overlays.get(nid, False))
                    cb.setStyleSheet(
                        "QCheckBox{color:#e2e8f0; font-size:11px; padding:4px 8px;}"
                        "QCheckBox::indicator{width:14px; height:14px;}")
                    cb.toggled.connect(
                        lambda on, _nid=nid: self._on_overlay_toggled(_nid, on))
                    wa.setDefaultWidget(cb)
                    menu.addAction(wa)

        menu.addSeparator()
        # Quick actions
        from PySide6.QtGui import QAction
        act_all = QAction("✓  Select All", menu)
        act_none = QAction("✗  Clear All", menu)
        act_all.triggered.connect(lambda: self._set_all_overlays(True))
        act_none.triggered.connect(lambda: self._set_all_overlays(False))
        menu.addAction(act_all)
        menu.addAction(act_none)

    def _on_overlay_toggled(self, node_id: str, on: bool):
        self._selected_overlays[node_id] = on
        self._persist_overlays()
        self._update_results_btn_text()
        if self._btn_multi.isChecked():
            # Multi-view: số ô không đổi theo tick (1 ô / root). Chỉ cần
            # refresh nội dung từng ô để re-composite overlay mới.
            self._refresh_multi_views()
        else:
            self.refresh_current()

    def _set_all_overlays(self, on: bool):
        if not self._graph:
            return
        for nid, n in self._graph.nodes.items():
            if "image" in n.outputs \
                    and getattr(n.tool, "category", "") != "Acquire Image":
                self._selected_overlays[nid] = on
        self._persist_overlays()
        self._update_results_btn_text()
        if self._btn_multi.isChecked():
            self._refresh_multi_views()
        else:
            self.refresh_current()

    def _update_results_btn_text(self):
        n = sum(1 for v in self._selected_overlays.values() if v)
        if n == 0:
            self._btn_results.setText("📊 Results ▾")
        else:
            self._btn_results.setText(f"📊 Results ({n}) ▾")

    def _find_acquire_root_image(self):
        """Trả output 'image' của node đầu chuỗi (Acquire Image)."""
        if not self._graph:
            return None
        for nid, n in self._graph.nodes.items():
            if getattr(n.tool, "category", "") == "Acquire Image" \
                    and "image" in n.outputs:
                return n.outputs["image"]
        return None

    def _overlay_diff(self, base: np.ndarray, before: np.ndarray,
                      after: np.ndarray) -> np.ndarray:
        """Compose pixel khác biệt (before→after) lên base. Dùng cho Shared
        Graphics: lấy annotation upstream-tool đã vẽ rồi áp lên ảnh hiển thị.

        before/after có thể khác channel count (vd PatMax nhận gray HxW, output
        BGR HxWx3 do `_bgr(img.copy())`); ta chỉ cần match H,W rồi up-convert
        gray → BGR trước khi absdiff. Trước đây check `before.shape != after.shape`
        cứng nên overlay PatMax + tool gray-input bị skip silent."""
        if (before is None or after is None
                or before.shape[:2] != after.shape[:2]
                or before.shape[:2] != base.shape[:2]):
            return base
        import cv2
        b = before if before.ndim == 3 else cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
        a = after  if after.ndim  == 3 else cv2.cvtColor(after,  cv2.COLOR_GRAY2BGR)
        diff = cv2.absdiff(a, b)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Pixel coi như annotation nếu lệch ≥ 20 (loại noise nhỏ)
        mask = (gray > 20)
        if not mask.any():
            return base
        out = base.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        out[mask] = a[mask]
        return out

    def _node_input_image(self, node):
        """Output 'image' của upstream gần nhất của node (input của node)."""
        if not self._graph:
            return None
        for c in self._graph.connections:
            if c.dst_id == node.node_id and c.dst_port == "image":
                src = self._graph.nodes.get(c.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    def _get_source_image(self, node):
        """Tìm ảnh gốc (raw) của pipeline — traverse ngược về node category
        'Acquire Image' đầu chuỗi. Cho phép Show Result OFF hiện ảnh thô,
        không phải output đã annotate của upstream gần nhất."""
        if not self._graph:
            return None
        visited = set()
        cur = node
        # Walk upstream qua port "image" để tìm root source
        for _ in range(64):    # an toàn: pipeline khó dài hơn 64 node
            if cur is None or cur.node_id in visited:
                break
            visited.add(cur.node_id)
            cat = getattr(cur.tool, "category", "")
            if cat == "Acquire Image" and "image" in cur.outputs:
                return cur.outputs["image"]
            # Tìm upstream nối vào port "image"
            upstream = None
            for c in self._graph.connections:
                if c.dst_id == cur.node_id and c.dst_port == "image":
                    upstream = self._graph.nodes.get(c.src_id)
                    break
            cur = upstream
        # Fallback: upstream gần nhất nếu không tìm thấy Acquire root
        for c in self._graph.connections:
            if c.dst_id == node.node_id and c.dst_port == "image":
                src = self._graph.nodes.get(c.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    def _display_node(self, node_id: str):
        if not self._graph or node_id not in self._graph.nodes:
            return
        self._current_node_id = node_id
        node = self._graph.nodes[node_id]

        active_overlays = [nid for nid, on in self._selected_overlays.items()
                            if on and nid in self._graph.nodes]

        def _vis_of(n):
            """Pick the annotated frame to render for node `n`:
            `_display_image` (private overlay) first, fall back to `image`
            (clean port). `or` is unsafe on numpy arrays — use is-None."""
            v = n.outputs.get("_display_image")
            if v is None:
                v = n.outputs.get("image")
            return v

        if active_overlays:
            # Composite mode: base = ảnh gốc Acquire, overlay = các tool đã tick.
            # Tool annotation lấy từ `_display_image` (ảnh + overlay) thay vì
            # port `image` (đã đổi sang clean pass-through).
            base = self._find_acquire_root_image()
            if base is None:
                img = _vis_of(node)
            else:
                import cv2
                comp = base.copy()
                if comp.ndim == 2:
                    comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
                for nid in active_overlays:
                    n = self._graph.nodes[nid]
                    before = self._node_input_image(n)
                    after  = _vis_of(n)
                    if before is not None and after is not None:
                        comp = self._overlay_diff(comp, before, after)
                img = comp
        else:
            # Mode bình thường: hiển thị output của node đang chọn — ưu tiên
            # `_display_image` (có overlay) rồi fall back `image` clean.
            img = _vis_of(node)

        if img is not None and isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            ch = img.shape[2] if len(img.shape) == 3 else 1
            tag = f"  •  Composite ({len(active_overlays)} overlays)" \
                if active_overlays else ""
            self._lbl_size.setText(
                f"{w}×{h}  ch:{ch}  dtype:{img.dtype}{tag}")
            self._img_view.set_image(img)
        else:
            self._img_view.set_image(None)
            self._lbl_size.setText("No image output yet")

        status_colors = {
            "pass": "#39ff14", "fail": "#ff3860",
            "error": "#ff3860", "idle": "#64748b", "running": "#ffd700"
        }
        sc = status_colors.get(node.status, "#64748b")
        self._lbl_status.setText(node.status.upper())
        self._lbl_status.setStyleSheet(
            f"color:{sc}; font-size:10px; font-family:'Courier New'; font-weight:700;")

    def _fit(self):
        self._img_view.fit()

    def _zoom_actual(self):
        """1:1 button — cap zoom ở fit_scale để ảnh không tràn viewport.
        Ảnh nhỏ hơn viewport: scale = 1.0 (actual pixels).
        Ảnh lớn hơn viewport: scale = fit_scale (vẫn hiển thị đầy đủ,
        không cần pan). User muốn zoom thật > fit có thể dùng wheel hoặc +.
        """
        v = self._img_view
        if v._pixmap is None:
            return
        pw, ph = v._pixmap.width(), v._pixmap.height()
        ww, wh = v.width(), v.height()
        if ww < 1 or wh < 1 or pw < 1 or ph < 1:
            return
        fit_scale = min(ww / pw, wh / ph) * 0.95
        v.set_zoom(min(1.0, fit_scale))

    def _on_pixel_info(self, x: int, y: int, rgb: tuple):
        r, g, b = rgb
        self._lbl_pixel.setText(
            f"X:{x:4d}  Y:{y:4d}    R:{r:3d}  G:{g:3d}  B:{b:3d}"
            f"    #{r:02X}{g:02X}{b:02X}")
        self._lbl_pixel.setStyleSheet(
            f"color:rgb({r},{g},{b}); font-size:10px; font-family:'Courier New';"
            f"background:#111827; border-radius:3px; padding:0 6px;")
