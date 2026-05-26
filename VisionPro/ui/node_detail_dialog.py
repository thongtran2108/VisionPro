"""
ui/node_detail_dialog.py — v5
Fix crop_roi:
  - Không có port kết nối → kéo chuột vẽ thủ công → lưu _drawn_roi
  - Có port kết nối → hiển thị readonly, vẽ rect từ port value
  - Mode hint thông minh hiển thị đang dùng mode nào
"""
from __future__ import annotations
from typing import Optional, Any, List, Tuple
import numpy as np

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                QTabWidget, QWidget, QScrollArea, QFrame,
                                QSplitter, QPushButton, QGroupBox, QCheckBox,
                                QSizePolicy, QApplication, QFileDialog,
                                QMessageBox, QComboBox)
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import (QPixmap, QImage, QFont, QColor, QPainter,
                            QPen, QBrush, QCursor, QMouseEvent)

from core.flow_graph import NodeInstance, FlowGraph
# PatMaxDialog imported lazily to avoid circular import
from core.tool_registry import ToolDef, ParamDef
from ui.properties_panel import ParamRow


# ════════════════════════════════════════════════════════════════════
#  Interactive image label
# ════════════════════════════════════════════════════════════════════
class InteractiveImageLabel(QLabel):
    """
    Hiển thị ảnh + overlay tương tác.
    mode="roi"      → kéo chuột chọn vùng (màu cyan)
    mode="template" → kéo chuột chọn template (màu cam)
    mode="pick"     → click lấy pixel color
    mode="view"     → chỉ xem, không tương tác
    mode="readonly" → xem + hiển thị rect cố định (port connected)
    """
    roi_changed    = Signal(int, int, int, int)
    pixel_picked   = Signal(int, int)
    pixel_hovered  = Signal(int, int)        # image coords, fires on mouse move
    mouse_left     = Signal()                # con trỏ rời khỏi widget
    template_drawn = Signal(int, int, int, int)
    origin_changed       = Signal(float, float)   # image coords (float)
    origin_angle_changed = Signal(float)          # degrees
    extra_origin_changed       = Signal(int, float, float)  # idx, x, y
    extra_origin_angle_changed = Signal(int, float)         # idx, degrees
    # Per-object origin markers (result view) — obj_idx tách rời với
    # extra_refs index để PatMaxDialog phân biệt nguồn drag.
    obj_origin_changed = Signal(int, float, float)  # obj_idx, x, y
    shape_drawn    = Signal(str, dict)      # shape_type, data (image coords)
    shapes_changed = Signal(list)           # multi-mode: list of {"type", **data}
    label_offset_changed = Signal(int, int) # drag label trên ảnh → (dx, dy) image px

    def __init__(self, mode="view", parent=None):
        super().__init__(parent)
        self.mode        = mode
        self._arr        = None
        self._scale      = 1.0      # = fit_scale * user_zoom (effective)
        self._fit_scale  = 1.0      # fit-to-widget scale (no user zoom)
        self._user_zoom  = 1.0      # multiplier ≥ 1 = phóng to, <1 = thu nhỏ
        self._pan_dx     = 0        # pan offset (widget coords) cộng vào _off_x
        self._pan_dy     = 0
        self._off_x      = 0
        self._off_y      = 0
        self._rect: Optional[QRect] = None
        self._drag_start: Optional[QPoint] = None
        self._dragging   = False
        self._pick_pos: Optional[Tuple[int,int]] = None
        # Pick-once: ép click kế tiếp emit pixel_picked + tự huỷ. Cho phép
        # tool ở mode "roi" (color_segment) cũng pick được màu.
        self._pick_once: bool = False
        self._readonly_rect: Optional[Tuple[int,int,int,int]] = None
        # Origin marker (PatMax pattern reference point) — image coords
        self._origin_xy: Optional[Tuple[float, float]] = None
        self._show_origin: bool = False
        self._dragging_origin: bool = False
        # Hệ trục XY tại origin — xoay được quanh tâm
        self._origin_angle: float = 0.0   # độ, 0 = X→phải, Y↓
        self._dragging_origin_rot: bool = False
        # Extra origins (multi reference markers — all draggable + rotatable)
        # mỗi entry: {"x": float, "y": float, "angle": float, "name": str}
        self._extras: List[dict] = []
        self._dragging_extra_center_idx: int = -1
        self._dragging_extra_rot_idx: int = -1
        self._extras_highlight_idx: int = -1
        # Per-object origin markers (PatMax result view).
        # mỗi entry: {"x", "y", "angle", "name", "obj_idx"} — image coords.
        # Tách khỏi _extras vì semantics khác: drag → ghi override per-object
        # vào node.params, không ảnh hưởng model pattern.
        self._obj_origins: List[dict] = []
        self._dragging_obj_origin_idx: int = -1
        # Khoá vẽ ROI mới khi True (cho phép edit shape hiện tại + kéo
        # origin/extras, không cho click empty area tạo shape mới)
        self._roi_locked: bool = False
        # Shape ROI ("rect" | "circle" | "ellipse" | "polygon")
        self._shape: str = "rect"
        self._shape_data: dict = {}                                # toạ độ ảnh
        self._poly_drawing: list = []                              # [(x,y), ...] đang vẽ
        # Edit-mode state cho shape đã vẽ xong (move / resize qua corner handles)
        self._edit_action: Optional[str] = None    # "move" | "tl" | "tr" | "bl" | "br"
        self._edit_anchor_w: Optional[QPoint] = None
        self._edit_orig_data: dict = {}
        # Multi-shape (opt-in qua set_multi_shape(True)). Khi tắt, behaviour
        # giống single-shape (chỉ dùng _shape_data). Khi bật, _shapes là
        # nguồn lưu trữ chính, _shape_data + _active_idx là shape đang active.
        self._multi: bool = False
        self._shapes: List[dict] = []     # mỗi entry: {"type": str, "data": dict}
        self._active_idx: Optional[int] = None

        # Draggable labels (vd Blob Analysis output) — list rect (x,y,w,h)
        # trong toạ độ ảnh + tâm anchor (cx, cy) tương ứng.
        self._label_rects: List[Tuple[int, int, int, int]] = []
        self._label_anchors: List[Tuple[float, float]] = []
        self._dragging_label: bool = False
        self._label_drag_start_img: Optional[Tuple[float, float]] = None
        self._label_drag_start_off: Tuple[int, int] = (0, 0)
        self._current_label_off: Tuple[int, int] = (0, 0)

        self.setAlignment(Qt.AlignCenter)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Hover color picker cần move event không cần nhấn nút
        self.setMouseTracking(True)
        # Optional QScrollArea parent — biết viewport size khi zoom > 1.
        self._scroll_area = None
        self._base_min_size = (400, 300)
        self.setStyleSheet(
            "background:#050810; border:1px solid #1e2d45; border-radius:6px;")

        cur_map = {
            "roi": Qt.CrossCursor, "template": Qt.CrossCursor,
            "pick": Qt.PointingHandCursor, "view": Qt.ArrowCursor,
            "readonly": Qt.ArrowCursor,
        }
        self.setCursor(QCursor(cur_map.get(mode, Qt.ArrowCursor)))

    def set_scroll_area(self, area):
        """Liên kết label với QScrollArea cha → zoom > 1 hiện scrollbars."""
        self._scroll_area = area
        self._base_min_size = (400, 300)

    # ── Image ──────────────────────────────────────────────────────
    def set_image(self, arr: Optional[np.ndarray]):
        # Reset zoom/pan khi đổi sang ảnh có kích thước khác (fit lại từ đầu)
        if arr is None or self._arr is None or arr.shape[:2] != self._arr.shape[:2]:
            self._user_zoom = 1.0
            self._pan_dx = 0
            self._pan_dy = 0
        self._arr = arr
        self._render()

    def set_rect_from_params(self, x, y, w, h):
        """Hiển thị rect khởi tạo từ params/port (image coords).
        Đồng thời populate _shape_data để corner-resize / body-drag hoạt
        động ngay (không cần user vẽ lại để kích hoạt edit handles)."""
        if self._arr is None:
            return
        ih, iw = self._arr.shape[:2]
        x = max(0, min(int(x), iw-1))
        y = max(0, min(int(y), ih-1))
        w = max(1, min(int(w), iw-x))
        h = max(1, min(int(h), ih-y))
        wx, wy = self._img_to_widget(x, y)
        ww = int(w * self._scale)
        wh = int(h * self._scale)
        self._rect = QRect(wx, wy, ww, wh)
        # _shape_data + _shape là source-of-truth cho edit/drag — đồng bộ luôn
        self._shape = "rect"
        self._shape_data = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        self._render()

    def set_readonly_rect(self, x, y, w, h):
        """Hiển thị rect từ port (không cho phép kéo thay đổi)."""
        self._readonly_rect = (x, y, w, h)
        self._render()

    def set_shape_mode(self, shape: str):
        """Đặt loại shape: 'rect' | 'circle' | 'ellipse' | 'polygon'.
        XOÁ shape_data hiện tại (single-mode reset). Multi-mode dùng
        set_next_shape_type() để giữ list."""
        if shape not in ("rect", "circle", "ellipse", "polygon"):
            shape = "rect"
        self._shape = shape
        self._poly_drawing = []
        self._rect = None
        self._shape_data = {}
        self._dragging = False
        self._render()

    def set_next_shape_type(self, shape: str):
        """Multi-mode: chỉ đổi loại shape sẽ vẽ tiếp, KHÔNG xoá list/active."""
        if shape not in ("rect", "circle", "ellipse", "polygon"):
            shape = "rect"
        self._shape = shape
        self._poly_drawing = []
        self._dragging = False
        self._render()

    def set_shape_data(self, shape: str, data: dict):
        """Khôi phục shape đã train (toạ độ ảnh)."""
        self._shape = shape
        self._shape_data = dict(data) if data else {}
        self._poly_drawing = []
        # Cập nhật _rect (bbox widget) cho rendering tham chiếu
        if shape == "rect" and data:
            wx, wy = self._img_to_widget(data["x"], data["y"])
            self._rect = QRect(wx, wy,
                                int(data["w"] * self._scale),
                                int(data["h"] * self._scale))
        else:
            self._rect = None
        self._render()

    def get_shape(self) -> Tuple[str, dict]:
        return self._shape, dict(self._shape_data)

    def cancel_polygon(self):
        if self._poly_drawing:
            self._poly_drawing = []
            self._render()

    # ── Multi-shape API ────────────────────────────────────────────
    def set_multi_shape(self, enable: bool):
        """Bật/tắt multi-shape. Tắt → xoá list, giữ shape hiện tại."""
        if bool(enable) == self._multi:
            return
        self._multi = bool(enable)
        if not self._multi:
            self._shapes = []
            self._active_idx = None
        elif self._shape_data:
            # Bật — đẩy shape hiện tại (nếu có) vào list để hiển thị nhất quán
            self._shapes = [{"type": self._shape, "data": dict(self._shape_data)}]
            self._active_idx = 0
        self._render()

    def get_shapes(self) -> List[dict]:
        """List shapes cho caller. Mỗi entry: {"type", **data}."""
        if self._multi:
            out = []
            for s in self._shapes:
                e = {"type": s["type"]}
                e.update(s["data"])
                out.append(e)
            return out
        if self._shape_data:
            e = {"type": self._shape}
            e.update(self._shape_data)
            return [e]
        return []

    def set_shapes(self, shapes: List[dict]):
        """Khôi phục danh sách shapes. Mỗi entry: {"type", **data}."""
        self._shapes = []
        for s in shapes or []:
            t = s.get("type", "rect")
            d = {k: v for k, v in s.items() if k != "type"}
            self._shapes.append({"type": t, "data": d})
        if self._shapes:
            self._active_idx = len(self._shapes) - 1
            last = self._shapes[self._active_idx]
            self._shape = last["type"]
            self._shape_data = dict(last["data"])
        else:
            self._active_idx = None
            self._shape_data = {}
        self._render()

    def clear_shapes(self):
        self._shapes = []
        self._active_idx = None
        self._shape_data = {}
        self._poly_drawing = []
        self._rect = None
        self._render()
        self._emit_shapes_changed()

    def delete_active_shape(self):
        if not self._multi:
            if self._shape_data:
                self._shape_data = {}
                self._render()
                self._emit_shapes_changed()
            return
        if self._active_idx is None or not (0 <= self._active_idx < len(self._shapes)):
            return
        self._shapes.pop(self._active_idx)
        if self._shapes:
            self._active_idx = min(self._active_idx, len(self._shapes) - 1)
            last = self._shapes[self._active_idx]
            self._shape = last["type"]
            self._shape_data = dict(last["data"])
        else:
            self._active_idx = None
            self._shape_data = {}
        self._render()
        self._emit_shapes_changed()

    def _emit_shapes_changed(self):
        self.shapes_changed.emit(self.get_shapes())

    def _commit_active_to_list(self):
        """Đồng bộ _shape_data → _shapes[_active_idx] (multi-mode only)."""
        if not self._multi or self._active_idx is None:
            return
        if 0 <= self._active_idx < len(self._shapes):
            self._shapes[self._active_idx] = {
                "type": self._shape, "data": dict(self._shape_data)}

    def _bbox_of(self, stype: str, sd: dict) -> Optional[QRect]:
        if not sd or stype not in ("rect", "ellipse", "circle"):
            return None
        if stype == "circle":
            wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
            wr = int(sd["r"] * self._scale)
            return QRect(wx - wr, wy - wr, 2 * wr, 2 * wr)
        wx, wy = self._img_to_widget(sd["x"], sd["y"])
        return QRect(wx, wy,
                      int(sd["w"] * self._scale),
                      int(sd["h"] * self._scale))

    def _hit_test_shapes_list(self, wx: int, wy: int
                                ) -> Optional[Tuple[int, str]]:
        """Hit-test trên _shapes (multi-mode). Trả (idx, action)."""
        if not self._multi:
            return None
        for i in reversed(range(len(self._shapes))):
            entry = self._shapes[i]
            if entry["type"] not in ("rect", "ellipse", "circle"):
                continue
            bb = self._bbox_of(entry["type"], entry["data"])
            if bb is None:
                continue
            for name, (cx, cy) in (
                ("tl", (bb.left(),  bb.top())),
                ("tr", (bb.right(), bb.top())),
                ("bl", (bb.left(),  bb.bottom())),
                ("br", (bb.right(), bb.bottom())),
            ):
                if abs(wx - cx) <= 8 and abs(wy - cy) <= 8:
                    return i, name
            if bb.contains(wx, wy):
                return i, "move"
        return None

    def _draw_one_shape(self, p: QPainter, stype: str, sd: dict,
                         label_num: int = 0):
        """Vẽ 1 shape không có handles — dùng cho các shape không-active trong list."""
        if not sd:
            return
        if stype == "rect":
            wx, wy = self._img_to_widget(sd["x"], sd["y"])
            ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
            p.drawRect(wx, wy, ww, wh)
            if label_num:
                p.drawText(wx + 4, wy + 14, f"#{label_num}")
        elif stype == "ellipse":
            wx, wy = self._img_to_widget(sd["x"], sd["y"])
            ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
            p.drawEllipse(wx, wy, ww, wh)
            if label_num:
                p.drawText(wx + 4, wy + 14, f"#{label_num}")
        elif stype == "circle":
            wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
            wr = int(sd["r"] * self._scale)
            p.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
            if label_num:
                p.drawText(wx - wr + 4, wy - wr + 14, f"#{label_num}")
        elif stype == "polygon" and sd.get("pts"):
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF
            pts_w = [QPointF(*self._img_to_widget(px, py)) for px, py in sd["pts"]]
            p.drawPolygon(QPolygonF(pts_w))
            if label_num and pts_w:
                p.drawText(int(pts_w[0].x()) + 4,
                            int(pts_w[0].y()) + 14, f"#{label_num}")

    def set_origin(self, x: Optional[float], y: Optional[float]):
        """Đặt điểm tham chiếu (origin) trên ảnh. Truyền (None,None) để ẩn."""
        if x is None or y is None:
            self._origin_xy = None
            self._show_origin = False
        else:
            self._origin_xy = (float(x), float(y))
            self._show_origin = True
        self._render()

    def set_origin_angle(self, angle: float):
        """Đặt góc xoay trục XY tại origin (độ)."""
        self._origin_angle = float(angle) % 360.0
        self._render()

    # ── Extra origins (multi marker, all draggable + rotatable) ─────
    def set_extras(self, extras: List[dict]):
        """Set list extra origin markers (image coords).
        Mỗi entry: {"x", "y", "angle", "name"}.
        """
        self._extras = [dict(e) for e in (extras or [])]
        # Reset drag indices nếu out of range
        if self._dragging_extra_center_idx >= len(self._extras):
            self._dragging_extra_center_idx = -1
        if self._dragging_extra_rot_idx >= len(self._extras):
            self._dragging_extra_rot_idx = -1
        self._render()

    def clear_extras(self):
        self._extras = []
        self._dragging_extra_center_idx = -1
        self._dragging_extra_rot_idx = -1
        self._extras_highlight_idx = -1
        self._render()

    # ── Per-object origin markers (PatMax result view) ──────────────
    def set_obj_origins(self, items: List[dict]):
        """Set per-object draggable origin markers. Mỗi item cần keys:
        x, y (image coords), angle (deg), name (str hiển thị), obj_idx (int).
        """
        self._obj_origins = [dict(e) for e in (items or [])]
        if self._dragging_obj_origin_idx >= len(self._obj_origins):
            self._dragging_obj_origin_idx = -1
        self._render()

    def clear_obj_origins(self):
        self._obj_origins = []
        self._dragging_obj_origin_idx = -1
        self._render()

    def _obj_origin_widget_pos(self, idx: int) -> Optional[Tuple[int, int]]:
        if 0 <= idx < len(self._obj_origins):
            e = self._obj_origins[idx]
            return self._img_to_widget(float(e.get("x", 0)),
                                        float(e.get("y", 0)))
        return None

    def _hit_obj_origin_center(self, wx: int, wy: int) -> int:
        for i in range(len(self._obj_origins)):
            c = self._obj_origin_widget_pos(i)
            if c is None:
                continue
            dx = wx - c[0]; dy = wy - c[1]
            if (dx * dx + dy * dy) <= self.CENTER_HIT_RADIUS ** 2:
                return i
        return -1

    def _update_obj_origin_pos_from_widget(self, idx: int, wx: int, wy: int):
        if not (0 <= idx < len(self._obj_origins)):
            return
        ix, iy = self._widget_to_img(wx, wy)
        self._obj_origins[idx]["x"] = float(ix)
        self._obj_origins[idx]["y"] = float(iy)
        self._render()
        obj_idx = int(self._obj_origins[idx].get("obj_idx", idx))
        self.obj_origin_changed.emit(obj_idx, float(ix), float(iy))

    def set_roi_locked(self, locked: bool):
        """Khoá vẽ ROI mới (chỉ áp khi mode='roi'/'template').
        Khi True: click empty area không tạo shape mới; vẫn cho edit
        shape hiện có + kéo origin/extras."""
        self._roi_locked = bool(locked)

    def set_label_rects(self, rects: List[Tuple[int, int, int, int]],
                         anchors: List[Tuple[float, float]],
                         current_offset: Tuple[int, int]):
        """Đăng ký vùng draggable cho text labels (vd: Blob output).
        rects/anchors cùng độ dài. current_offset là (dx, dy) hiện tại
        trong params — cần để khôi phục anchor lúc drag."""
        self._label_rects = list(rects or [])
        self._label_anchors = list(anchors or [])
        self._current_label_off = tuple(current_offset or (0, 0))

    def _hit_label(self, wx: int, wy: int) -> int:
        """Trả về index label trúng tại (wx, wy) widget coords, -1 nếu miss."""
        for i, (rx, ry, rw, rh) in enumerate(self._label_rects):
            wx0, wy0 = self._img_to_widget(rx, ry)
            ww = max(1, int(rw * self._scale))
            wh = max(1, int(rh * self._scale))
            # Mở rộng tolerance 4px để dễ trúng
            if wx0 - 4 <= wx <= wx0 + ww + 4 and wy0 - 4 <= wy <= wy0 + wh + 4:
                return i
        return -1

    def set_extras_highlight(self, idx: int):
        """Highlight extra marker idx (-1 = không highlight)."""
        self._extras_highlight_idx = int(idx)
        self._render()

    def _extra_widget_pos(self, idx: int) -> Optional[Tuple[int, int]]:
        if 0 <= idx < len(self._extras):
            e = self._extras[idx]
            return self._img_to_widget(float(e.get("x", 0)),
                                        float(e.get("y", 0)))
        return None

    def _extra_axis_endpoints(self, idx: int):
        c = self._extra_widget_pos(idx)
        if c is None:
            return None
        import math as _m
        a = _m.radians(float(self._extras[idx].get("angle", 0.0)))
        cx, cy = c
        L = self.AXIS_LEN
        x_end = (int(cx + _m.cos(a) * L), int(cy + _m.sin(a) * L))
        y_end = (int(cx - _m.sin(a) * L), int(cy + _m.cos(a) * L))
        return (cx, cy), x_end, y_end

    def _extra_rot_pos(self, idx: int):
        eps = self._extra_axis_endpoints(idx)
        if eps is None:
            return None
        import math as _m
        a = _m.radians(float(self._extras[idx].get("angle", 0.0)))
        cx, cy = eps[0]
        d = self.AXIS_LEN + self.ROT_HANDLE_OFFSET
        return (int(cx + _m.cos(a) * d), int(cy + _m.sin(a) * d))

    def _hit_extra_center(self, wx: int, wy: int) -> int:
        for i in range(len(self._extras)):
            c = self._extra_widget_pos(i)
            if c is None:
                continue
            dx = wx - c[0]; dy = wy - c[1]
            if (dx * dx + dy * dy) <= self.CENTER_HIT_RADIUS ** 2:
                return i
        return -1

    def _hit_extra_rot(self, wx: int, wy: int, radius: int = 10) -> int:
        for i in range(len(self._extras)):
            r = self._extra_rot_pos(i)
            if r is None:
                continue
            dx = wx - r[0]; dy = wy - r[1]
            if (dx * dx + dy * dy) <= radius ** 2:
                return i
        return -1

    def _update_extra_pos_from_widget(self, idx: int, wx: int, wy: int):
        if not (0 <= idx < len(self._extras)):
            return
        ix, iy = self._widget_to_img(wx, wy)
        self._extras[idx]["x"] = float(ix)
        self._extras[idx]["y"] = float(iy)
        self._render()
        self.extra_origin_changed.emit(idx, float(ix), float(iy))

    def _update_extra_angle_from_widget(self, idx: int, wx: int, wy: int):
        if not (0 <= idx < len(self._extras)):
            return
        c = self._extra_widget_pos(idx)
        if c is None:
            return
        import math as _m
        a = _m.degrees(_m.atan2(wy - c[1], wx - c[0]))
        self._extras[idx]["angle"] = a % 360.0
        self._render()
        self.extra_origin_angle_changed.emit(idx, a % 360.0)

    # ── Origin helpers (toạ độ widget của tâm + 2 đầu trục) ──────
    AXIS_LEN  = 32
    ROT_HANDLE_OFFSET = 8        # bán kính vòng tròn xoay sau X-arrow tip
    CENTER_HIT_RADIUS = 8

    def _origin_widget_pos(self) -> Optional[Tuple[int, int]]:
        if not self._show_origin or self._origin_xy is None:
            return None
        ox, oy = self._origin_xy
        return self._img_to_widget(ox, oy)

    def _origin_axis_endpoints(self) -> Optional[Tuple[Tuple[int,int],
                                                        Tuple[int,int],
                                                        Tuple[int,int]]]:
        """Trả (center, x_end, y_end) widget coords; None nếu không hiển thị."""
        c = self._origin_widget_pos()
        if c is None:
            return None
        import math as _m
        a = _m.radians(self._origin_angle)
        cx, cy = c
        L = self.AXIS_LEN
        # X-axis: pointing along angle (image-space: y-axis downwards)
        x_end = (int(cx + _m.cos(a) * L), int(cy + _m.sin(a) * L))
        # Y-axis: 90° clockwise (theo chuẩn image: Y xuống)
        y_end = (int(cx - _m.sin(a) * L), int(cy + _m.cos(a) * L))
        return (cx, cy), x_end, y_end

    def _rot_handle_pos(self) -> Optional[Tuple[int, int]]:
        eps = self._origin_axis_endpoints()
        if eps is None:
            return None
        import math as _m
        a = _m.radians(self._origin_angle)
        cx, cy = eps[0]
        d = self.AXIS_LEN + self.ROT_HANDLE_OFFSET
        return (int(cx + _m.cos(a) * d), int(cy + _m.sin(a) * d))

    def _hit_origin_center(self, wx: int, wy: int) -> bool:
        c = self._origin_widget_pos()
        if c is None:
            return False
        dx = wx - c[0]; dy = wy - c[1]
        return (dx * dx + dy * dy) <= self.CENTER_HIT_RADIUS ** 2

    def _hit_origin_rot(self, wx: int, wy: int, radius: int = 10) -> bool:
        r = self._rot_handle_pos()
        if r is None:
            return False
        dx = wx - r[0]; dy = wy - r[1]
        return (dx * dx + dy * dy) <= radius ** 2

    def _hit_origin_handle(self, wx: int, wy: int, radius: int = 12) -> bool:
        # Backwards-compat: gộp center + rotate handle.
        return self._hit_origin_center(wx, wy) or self._hit_origin_rot(wx, wy)

    def _img_to_widget(self, ix, iy):
        return int(ix * self._scale + self._off_x), int(iy * self._scale + self._off_y)

    def _widget_to_img(self, wx, wy):
        if self._scale == 0:
            return 0, 0
        return int((wx - self._off_x) / self._scale), int((wy - self._off_y) / self._scale)

    # ── Wheel zoom (giữ pixel dưới chuột cố định) ──────────────────
    def wheelEvent(self, event):
        if self._arr is None or self._fit_scale <= 0:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(0.2, min(20.0, self._user_zoom * factor))
        if abs(new_zoom - self._user_zoom) < 1e-4:
            return
        try:
            mp = event.position()
            mx = float(mp.x()); my = float(mp.y())
        except AttributeError:
            mx = float(event.x()); my = float(event.y())
        scale_old = self._scale
        if scale_old <= 0:
            return
        # Pixel ảnh dưới con trỏ trước khi zoom (toạ độ ảnh)
        ix = (mx - self._off_x) / scale_old
        iy = (my - self._off_y) / scale_old
        self._user_zoom = new_zoom

        if self._scroll_area is not None:
            # Render lại để label resize theo zoom mới
            self._render()
            # Sau render: tính vị trí pixel cũ trên label mới, scroll để đặt
            # nó dưới con trỏ (mx, my là toạ độ trên label cũ — vẫn ổn vì
            # con trỏ đang ở vị trí widget-space của label).
            new_scale = self._scale
            new_px = ix * new_scale + self._off_x
            new_py = iy * new_scale + self._off_y
            # Lệch so với mouse → cuộn theo
            hbar = self._scroll_area.horizontalScrollBar()
            vbar = self._scroll_area.verticalScrollBar()
            hbar.setValue(hbar.value() + int(new_px - mx))
            vbar.setValue(vbar.value() + int(new_py - my))
        else:
            # Pan-offset mode (không scroll area) — giữ pixel dưới chuột
            h, w = self._arr.shape[:2]
            new_scale = self._fit_scale * new_zoom
            new_dw = int(w * new_scale); new_dh = int(h * new_scale)
            base_off_x = (self.width()  - new_dw) // 2
            base_off_y = (self.height() - new_dh) // 2
            target_off_x = mx - ix * new_scale
            target_off_y = my - iy * new_scale
            self._pan_dx = int(round(target_off_x - base_off_x))
            self._pan_dy = int(round(target_off_y - base_off_y))
            self._render()
        event.accept()

    def reset_zoom(self):
        """Reset về fit-to-widget (zoom=1, pan=0)."""
        self._user_zoom = 1.0
        self._pan_dx = 0
        self._pan_dy = 0
        self._render()

    def _apply_zoom_centered(self, new_zoom: float):
        """Đặt user_zoom mới, giữ tâm ảnh cố định. Dùng cho zoom button
        (không có vị trí chuột như wheelEvent)."""
        if self._arr is None or self._fit_scale <= 0:
            return
        new_zoom = max(0.2, min(20.0, float(new_zoom)))
        if abs(new_zoom - self._user_zoom) < 1e-4:
            return
        self._user_zoom = new_zoom
        if self._scroll_area is not None:
            vp = self._scroll_area.viewport()
            cx_widget = vp.width()  / 2.0
            cy_widget = vp.height() / 2.0
            hbar = self._scroll_area.horizontalScrollBar()
            vbar = self._scroll_area.verticalScrollBar()
            # Pixel ảnh dưới tâm viewport TRƯỚC khi render lại
            old_scale = self._scale
            ix = (hbar.value() + cx_widget - self._off_x) / max(old_scale, 1e-6)
            iy = (vbar.value() + cy_widget - self._off_y) / max(old_scale, 1e-6)
            self._render()
            new_scale = self._scale
            # Cuộn để pixel ảnh đó vẫn ở tâm viewport
            hbar.setValue(int(ix * new_scale + self._off_x - cx_widget))
            vbar.setValue(int(iy * new_scale + self._off_y - cy_widget))
        else:
            self._pan_dx = 0
            self._pan_dy = 0
            self._render()

    def zoom_in(self):
        self._apply_zoom_centered(self._user_zoom * 1.25)

    def zoom_out(self):
        self._apply_zoom_centered(self._user_zoom / 1.25)

    def zoom_actual(self):
        """Hiển thị pixel ảnh 1:1 trên màn hình."""
        if self._fit_scale > 0:
            self._apply_zoom_centered(1.0 / self._fit_scale)

    def mouseDoubleClickEvent(self, event):
        """Double-click giữa khoảng trống (không trên shape) → reset zoom."""
        if (self._arr is not None and self._user_zoom != 1.0
                and event.button() == Qt.MiddleButton):
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ── Edit-mode helpers (move / resize shape đã vẽ xong) ─────────
    def _shape_widget_bbox(self) -> Optional[QRect]:
        """Bbox (widget coords) của shape đang lưu trong _shape_data."""
        sd = self._shape_data
        if not sd or self._shape not in ("rect", "ellipse", "circle"):
            return None
        if self._shape == "circle":
            wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
            wr = int(sd["r"] * self._scale)
            return QRect(wx - wr, wy - wr, 2 * wr, 2 * wr)
        wx, wy = self._img_to_widget(sd["x"], sd["y"])
        ww = int(sd["w"] * self._scale)
        wh = int(sd["h"] * self._scale)
        return QRect(wx, wy, ww, wh)

    def _hit_corner(self, wx: int, wy: int, tol: int = 8) -> Optional[str]:
        bb = self._shape_widget_bbox()
        if bb is None:
            return None
        for name, (cx, cy) in (
            ("tl", (bb.left(),  bb.top())),
            ("tr", (bb.right(), bb.top())),
            ("bl", (bb.left(),  bb.bottom())),
            ("br", (bb.right(), bb.bottom())),
        ):
            if abs(wx - cx) <= tol and abs(wy - cy) <= tol:
                return name
        return None

    def _hit_body(self, wx: int, wy: int) -> bool:
        bb = self._shape_widget_bbox()
        return bb is not None and bb.contains(wx, wy)

    def _apply_edit(self, pos: QPoint):
        """Áp delta widget→ảnh, cập nhật _shape_data theo _edit_action."""
        if self._edit_anchor_w is None or not self._edit_orig_data \
                or self._scale <= 0:
            return
        dx_i = (pos.x() - self._edit_anchor_w.x()) / self._scale
        dy_i = (pos.y() - self._edit_anchor_w.y()) / self._scale
        sd = dict(self._edit_orig_data)
        if self._arr is not None:
            H, W = self._arr.shape[:2]
        else:
            H = W = 10 ** 9

        if self._shape in ("rect", "ellipse"):
            x = sd["x"]; y = sd["y"]; w = sd["w"]; h = sd["h"]
            act = self._edit_action
            if act == "move":
                x = max(0, min(int(round(x + dx_i)), W - w))
                y = max(0, min(int(round(y + dy_i)), H - h))
            elif act == "tl":
                nx = int(round(x + dx_i)); ny = int(round(y + dy_i))
                nw = w + (x - nx); nh = h + (y - ny)
                if nw >= 4 and nh >= 4 and nx >= 0 and ny >= 0:
                    x, y, w, h = nx, ny, nw, nh
            elif act == "tr":
                ny = int(round(y + dy_i))
                nh = h + (y - ny); nw = int(round(w + dx_i))
                if nw >= 4 and nh >= 4 and ny >= 0 and (x + nw) <= W:
                    y, w, h = ny, nw, nh
            elif act == "bl":
                nx = int(round(x + dx_i))
                nw = w + (x - nx); nh = int(round(h + dy_i))
                if nw >= 4 and nh >= 4 and nx >= 0 and (y + nh) <= H:
                    x, w, h = nx, nw, nh
            elif act == "br":
                nw = int(round(w + dx_i)); nh = int(round(h + dy_i))
                if nw >= 4 and nh >= 4 and (x + nw) <= W and (y + nh) <= H:
                    w, h = nw, nh
            sd.update({"x": x, "y": y, "w": w, "h": h})

        elif self._shape == "circle":
            cx = sd["cx"]; cy = sd["cy"]; r = sd["r"]
            if self._edit_action == "move":
                cx = max(r, min(int(round(cx + dx_i)), W - r))
                cy = max(r, min(int(round(cy + dy_i)), H - r))
            else:
                # Bất kỳ corner nào → resize bán kính theo khoảng cách tới tâm
                wx_c, wy_c = self._img_to_widget(cx, cy)
                d_w = ((pos.x() - wx_c) ** 2 + (pos.y() - wy_c) ** 2) ** 0.5
                r = max(4, int(round(d_w / self._scale)))
                r = min(r, cx, cy, W - cx, H - cy)
            x = max(0, cx - r); y = max(0, cy - r)
            sd.update({"cx": cx, "cy": cy, "r": r,
                        "x": x, "y": y, "w": 2 * r, "h": 2 * r})

        self._shape_data = sd
        self._render()

    def _render(self):
        if self._arr is None:
            self.setText("No Image\nRun pipeline first or load image source.")
            return
        import cv2
        arr = self._arr.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        h, w, ch = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        # Khi có scroll area: dùng viewport size làm tham chiếu fit, không phải
        # self.width()/height() (chính label có thể đã grow theo zoom).
        if self._scroll_area is not None:
            vp = self._scroll_area.viewport()
            pw = max(1, vp.width() - 4)
            ph = max(1, vp.height() - 4)
        else:
            pw = max(1, self.width() - 4)
            ph = max(1, self.height() - 4)
        sx = pw / w; sy = ph / h
        self._fit_scale = min(sx, sy)
        self._scale = self._fit_scale * self._user_zoom
        dw = int(w * self._scale); dh = int(h * self._scale)

        if self._scroll_area is not None and self._user_zoom > 1.0:
            # Zoom > 1 → label rộng hơn viewport → QScrollArea hiện scrollbars.
            need_w = max(self._base_min_size[0], dw + 8)
            need_h = max(self._base_min_size[1], dh + 8)
            if self.width() != need_w or self.height() != need_h:
                self.setMinimumSize(need_w, need_h)
                self.resize(need_w, need_h)
            self._off_x = (self.width() - dw) // 2
            self._off_y = (self.height() - dh) // 2
        else:
            # Fit-to-viewport: kéo label về kích thước viewport, center image.
            if self._scroll_area is not None:
                vp = self._scroll_area.viewport()
                if self.width() != vp.width() or self.height() != vp.height():
                    self.setMinimumSize(self._base_min_size[0],
                                        self._base_min_size[1])
                    self.resize(vp.width(), vp.height())
            self._off_x = (self.width() - dw) // 2 + self._pan_dx
            self._off_y = (self.height() - dh) // 2 + self._pan_dy

        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(5, 8, 16))
        p = QPainter(canvas)
        # Anti-alias cho shape; SmoothPixmapTransform giúp nét cả khi zoom in/out.
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawPixmap(self._off_x, self._off_y, dw, dh, pix)

        # Readonly rect (port connected) — màu xanh lá
        if self._readonly_rect:
            rx, ry, rw, rh = self._readonly_rect
            wx, wy = self._img_to_widget(rx, ry)
            ww = int(rw * self._scale); wh = int(rh * self._scale)
            col = QColor(57, 255, 20)  # bright green
            p.setPen(QPen(col, 2, Qt.SolidLine))
            p.setBrush(QBrush(QColor(57, 255, 20, 25)))
            p.drawRect(wx, wy, ww, wh)
            p.setPen(QPen(col))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(wx + 4, wy - 6, f"[PORT] ({rx},{ry}) {rw}×{rh}")

        # Interactive / drawn shape — màu cyan (roi) hoặc cam (template)
        if self.mode == "template":
            col = QColor(255, 140, 50)
        else:
            col = QColor(0, 212, 255)
        fill = QColor(col.red(), col.green(), col.blue(), 28)

        # Multi-mode: vẽ tất cả shapes đã commit (ngoài active đang edit) — dim hơn
        if self._multi and self._shapes:
            other_col = QColor(col.red(), col.green(), col.blue(), 200)
            other_fill = QColor(col.red(), col.green(), col.blue(), 14)
            p.setPen(QPen(other_col, 2, Qt.DotLine))
            p.setBrush(QBrush(other_fill))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            for i, entry in enumerate(self._shapes):
                if i == self._active_idx:
                    continue
                self._draw_one_shape(p, entry["type"], entry["data"], i + 1)

        # Shape đã hoàn tất (saved)
        sd = self._shape_data
        if sd:
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(fill))
            if self._shape == "rect":
                wx, wy = self._img_to_widget(sd["x"], sd["y"])
                ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
                p.drawRect(wx, wy, ww, wh)
            elif self._shape == "ellipse":
                wx, wy = self._img_to_widget(sd["x"], sd["y"])
                ww = int(sd["w"] * self._scale); wh = int(sd["h"] * self._scale)
                p.drawEllipse(wx, wy, ww, wh)
            elif self._shape == "circle":
                wx, wy = self._img_to_widget(sd["cx"], sd["cy"])
                wr = int(sd["r"] * self._scale)
                p.drawEllipse(wx - wr, wy - wr, wr * 2, wr * 2)
                p.setPen(QPen(col, 1))
                p.drawLine(wx - 5, wy, wx + 5, wy)
                p.drawLine(wx, wy - 5, wx, wy + 5)
            elif self._shape == "polygon" and sd.get("pts"):
                from PySide6.QtCore import QPointF
                from PySide6.QtGui import QPolygonF
                pts_w = [QPointF(*self._img_to_widget(px, py)) for px, py in sd["pts"]]
                p.drawPolygon(QPolygonF(pts_w))

            # Corner handles cho rect/ellipse/circle — chỉ hiện khi
            # KHÔNG đang vẽ rubber-band, để user biết có thể move/resize.
            if (self.mode in ("roi", "template")
                    and self._shape in ("rect", "ellipse", "circle")
                    and not self._dragging):
                bb = self._shape_widget_bbox()
                if bb is not None:
                    p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
                    for cx, cy in ((bb.left(),  bb.top()),
                                    (bb.right(), bb.top()),
                                    (bb.left(),  bb.bottom()),
                                    (bb.right(), bb.bottom())):
                        p.drawRect(cx - 4, cy - 4, 8, 8)

        # Đang drag rect/circle/ellipse
        draw_rect = self._rect
        if draw_rect and not draw_rect.isNull() and self._dragging:
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(fill))
            if self._shape == "ellipse":
                p.drawEllipse(draw_rect)
            elif self._shape == "circle":
                p.drawEllipse(draw_rect)
            else:
                p.drawRect(draw_rect)
            # Corner handles + label
            p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
            for cx, cy in [(draw_rect.left(), draw_rect.top()),
                           (draw_rect.right(), draw_rect.top()),
                           (draw_rect.left(), draw_rect.bottom()),
                           (draw_rect.right(), draw_rect.bottom())]:
                p.drawRect(cx - 4, cy - 4, 8, 8)
            ix, iy = self._widget_to_img(draw_rect.left(), draw_rect.top())
            iw2 = int(draw_rect.width() / self._scale)
            ih2 = int(draw_rect.height() / self._scale)
            p.setPen(QPen(col)); p.setFont(QFont("Courier New", 9, QFont.Bold))
            p.drawText(draw_rect.left() + 4, draw_rect.top() - 6,
                       f"({ix},{iy})  {iw2}×{ih2}")
        elif draw_rect and not draw_rect.isNull() and self._shape == "rect" and not sd:
            # ROI rect đã set qua set_rect_from_params (legacy)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(QBrush(fill))
            p.drawRect(draw_rect)

        # Polygon đang vẽ
        if self._shape == "polygon" and self._poly_drawing:
            p.setPen(QPen(col, 2, Qt.DashLine)); p.setBrush(Qt.NoBrush)
            pts_w = [self._img_to_widget(px, py) for px, py in self._poly_drawing]
            for i in range(len(pts_w) - 1):
                p.drawLine(pts_w[i][0], pts_w[i][1],
                           pts_w[i+1][0], pts_w[i+1][1])
            p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
            for px, py in pts_w:
                p.drawEllipse(px - 4, py - 4, 8, 8)
            p.setPen(QPen(QColor(255, 215, 0)))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            if pts_w:
                p.drawText(pts_w[0][0] + 8, pts_w[0][1] - 6,
                           f"Polygon: {len(pts_w)} pt — double-click để đóng")

        # Origin marker — hệ trục XY xoay được quanh tâm
        if self._show_origin and self._origin_xy is not None:
            eps = self._origin_axis_endpoints()
            if eps is not None:
                (cx, cy), x_end, y_end = eps
                rot = self._rot_handle_pos()
                # X axis — đỏ
                col_x = QColor(255, 70, 70)
                p.setPen(QPen(col_x, 2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(cx, cy, x_end[0], x_end[1])
                # Mũi tên X
                import math as _m
                a = _m.radians(self._origin_angle)
                ah = 6.0
                ax1 = (int(x_end[0] - ah * _m.cos(a - _m.radians(25))),
                        int(x_end[1] - ah * _m.sin(a - _m.radians(25))))
                ax2 = (int(x_end[0] - ah * _m.cos(a + _m.radians(25))),
                        int(x_end[1] - ah * _m.sin(a + _m.radians(25))))
                p.drawLine(x_end[0], x_end[1], ax1[0], ax1[1])
                p.drawLine(x_end[0], x_end[1], ax2[0], ax2[1])
                p.setPen(QPen(col_x))
                p.setFont(QFont("Segoe UI", 9, QFont.Bold))
                p.drawText(x_end[0] + 4, x_end[1] + 4, "X")
                # Y axis — xanh lá
                col_y = QColor(80, 220, 100)
                p.setPen(QPen(col_y, 2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(cx, cy, y_end[0], y_end[1])
                b = a + _m.radians(90)  # góc Y
                ay1 = (int(y_end[0] - ah * _m.cos(b - _m.radians(25))),
                        int(y_end[1] - ah * _m.sin(b - _m.radians(25))))
                ay2 = (int(y_end[0] - ah * _m.cos(b + _m.radians(25))),
                        int(y_end[1] - ah * _m.sin(b + _m.radians(25))))
                p.drawLine(y_end[0], y_end[1], ay1[0], ay1[1])
                p.drawLine(y_end[0], y_end[1], ay2[0], ay2[1])
                p.setPen(QPen(col_y))
                p.drawText(y_end[0] + 4, y_end[1] + 4, "Y")
                # Rotate handle
                if rot is not None:
                    rh_col = QColor(255, 215, 0)
                    p.setPen(QPen(rh_col, 2))
                    p.setBrush(QBrush(QColor(255, 215, 0, 80)))
                    p.drawEllipse(rot[0] - 5, rot[1] - 5, 10, 10)
                    # Vòng cung gợi ý xoay
                    p.setPen(QPen(rh_col, 1, Qt.DotLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawArc(rot[0] - 8, rot[1] - 8, 16, 16, 0, 270 * 16)
                # Label toạ độ + góc
                p.setPen(QPen(QColor(0, 212, 255)))
                p.setFont(QFont("Courier New", 9, QFont.Bold))
                p.drawText(cx + 18, cy - 12,
                           f"O ({self._origin_xy[0]:.1f},{self._origin_xy[1]:.1f})  "
                           f"{self._origin_angle:+.1f}deg")

        # Extra origins — vẽ từng marker (tất cả luôn hiển thị + kéo được)
        for ei, ex in enumerate(self._extras):
            eps = self._extra_axis_endpoints(ei)
            if eps is None:
                continue
            (ecx, ecy), x_end, y_end = eps
            erot = self._extra_rot_pos(ei)
            is_hi = (ei == self._extras_highlight_idx)
            import math as _m2
            a = _m2.radians(float(ex.get("angle", 0.0)))
            ah = 6.0

            # X axis (đỏ; sáng hơn khi highlight)
            col_ex = QColor(255, 100, 100) if is_hi else QColor(220, 60, 60)
            p.setPen(QPen(col_ex, 2 if is_hi else 1, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(ecx, ecy, x_end[0], x_end[1])
            ax1 = (int(x_end[0] - ah * _m2.cos(a - _m2.radians(25))),
                    int(x_end[1] - ah * _m2.sin(a - _m2.radians(25))))
            ax2 = (int(x_end[0] - ah * _m2.cos(a + _m2.radians(25))),
                    int(x_end[1] - ah * _m2.sin(a + _m2.radians(25))))
            p.drawLine(x_end[0], x_end[1], ax1[0], ax1[1])
            p.drawLine(x_end[0], x_end[1], ax2[0], ax2[1])
            p.setPen(QPen(col_ex))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(x_end[0] + 4, x_end[1] + 4, "X")

            # Y axis (xanh lá)
            col_ey = QColor(120, 230, 120) if is_hi else QColor(70, 180, 80)
            p.setPen(QPen(col_ey, 2 if is_hi else 1, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(ecx, ecy, y_end[0], y_end[1])
            b = a + _m2.radians(90)
            ay1 = (int(y_end[0] - ah * _m2.cos(b - _m2.radians(25))),
                    int(y_end[1] - ah * _m2.sin(b - _m2.radians(25))))
            ay2 = (int(y_end[0] - ah * _m2.cos(b + _m2.radians(25))),
                    int(y_end[1] - ah * _m2.sin(b + _m2.radians(25))))
            p.drawLine(y_end[0], y_end[1], ay1[0], ay1[1])
            p.drawLine(y_end[0], y_end[1], ay2[0], ay2[1])
            p.setPen(QPen(col_ey))
            p.drawText(y_end[0] + 4, y_end[1] + 4, "Y")

            # Rotate handle (vàng — to hơn khi highlight)
            if erot is not None:
                rh_col = QColor(255, 215, 0)
                p.setPen(QPen(rh_col, 2))
                p.setBrush(QBrush(QColor(255, 215, 0, 80 if not is_hi else 160)))
                rr = 6 if is_hi else 4
                p.drawEllipse(erot[0] - rr, erot[1] - rr, rr * 2, rr * 2)

            # Label "Name (x, y) +deg"
            nm = str(ex.get("name", f"Ref {ei+1}"))
            label_col = QColor(0, 212, 255) if is_hi else QColor(100, 180, 220)
            p.setPen(QPen(label_col))
            p.setFont(QFont("Courier New", 9, QFont.Bold if is_hi else QFont.Normal))
            p.drawText(ecx + 14, ecy - 10,
                       f"{nm} ({float(ex.get('x', 0)):.1f},"
                       f"{float(ex.get('y', 0)):.1f})  "
                       f"{float(ex.get('angle', 0)):+.1f}deg")

        # Per-object origin markers (PatMax result view) — draggable centers,
        # vẽ trục XY xoay theo angle, label cyan giống style cv2 origin marker.
        import math as _m_oo
        for oi, oo in enumerate(self._obj_origins):
            wpos = self._obj_origin_widget_pos(oi)
            if wpos is None:
                continue
            ocx, ocy = wpos
            a_oo = _m_oo.radians(float(oo.get("angle", 0.0)))
            L = self.AXIS_LEN
            x_end = (int(ocx + _m_oo.cos(a_oo) * L),
                     int(ocy + _m_oo.sin(a_oo) * L))
            y_end = (int(ocx - _m_oo.sin(a_oo) * L),
                     int(ocy + _m_oo.cos(a_oo) * L))

            # X axis (đỏ)
            p.setPen(QPen(QColor(220, 60, 60), 2, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(ocx, ocy, x_end[0], x_end[1])
            p.setPen(QPen(QColor(220, 60, 60)))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(x_end[0] + 4, x_end[1] + 4, "X")
            # Y axis (xanh lá)
            p.setPen(QPen(QColor(70, 180, 80), 2, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(ocx, ocy, y_end[0], y_end[1])
            p.setPen(QPen(QColor(70, 180, 80)))
            p.drawText(y_end[0] + 4, y_end[1] + 4, "Y")
            # Center dot (cyan filled) — hint kéo được
            cdot_col = QColor(0, 212, 255)
            p.setPen(QPen(cdot_col, 1))
            p.setBrush(QBrush(cdot_col))
            p.drawEllipse(ocx - 4, ocy - 4, 8, 8)
            # Label
            nm_oo = str(oo.get("name", f"Obj {oi+1}"))
            p.setPen(QPen(QColor(0, 212, 255)))
            p.setFont(QFont("Courier New", 9, QFont.Bold))
            ang_oo = float(oo.get("angle", 0.0))
            lbl = (f"{nm_oo}: O ({float(oo.get('x', 0)):.1f},"
                   f"{float(oo.get('y', 0)):.1f})")
            if abs(ang_oo) > 0.05:
                lbl += f"  {ang_oo:+.1f}deg"
            p.drawText(ocx + 14, ocy - 10, lbl)

        # Pixel pick marker
        if self._pick_pos and self.mode == "pick":
            px2, py2 = self._img_to_widget(*self._pick_pos)
            p.setPen(QPen(QColor(255, 255, 0), 2))
            p.drawLine(px2 - 10, py2, px2 + 10, py2)
            p.drawLine(px2, py2 - 10, px2, py2 + 10)
            p.drawEllipse(px2 - 6, py2 - 6, 12, 12)

        p.end()
        self.setPixmap(canvas)

    # ── Mouse ──────────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()
        # Pick-once override (vd nút "Pick Color" trong color_segment): chỉ
        # emit pixel_picked rồi tự huỷ, không kích hoạt ROI/template drag.
        if self._pick_once and self._arr is not None:
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            h2, w2 = self._arr.shape[:2]
            if 0 <= ix < w2 and 0 <= iy < h2:
                self._pick_pos = (ix, iy)
                self._render()
                self.pixel_picked.emit(int(ix), int(iy))
            self.disarm_pick()
            event.accept()
            return
        # Label drag (Blob/AreaMeasure…): bắt trước mọi mode khác.
        if self._label_rects and self._hit_label(pos.x(), pos.y()) >= 0:
            self._dragging_label = True
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            self._label_drag_start_img = (float(ix), float(iy))
            self._label_drag_start_off = tuple(self._current_label_off)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        if self.mode == "pick":
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            if self._arr is not None:
                h2, w2 = self._arr.shape[:2]
                ix = max(0, min(ix, w2-1)); iy = max(0, min(iy, h2-1))
                self._pick_pos = (ix, iy)
                self._render()
                self.pixel_picked.emit(ix, iy)
        elif self.mode in ("roi", "template"):
            # Ưu tiên: main origin rot/center → extras rot/center → ROI
            if self._show_origin and self._hit_origin_rot(pos.x(), pos.y()):
                self._dragging_origin_rot = True
                self._update_origin_angle_from_widget(pos.x(), pos.y())
                return
            if self._show_origin and self._hit_origin_center(pos.x(), pos.y()):
                self._dragging_origin = True
                self._update_origin_from_widget(pos.x(), pos.y())
                return
            # Extra origins (multi refs)
            ehit = self._hit_extra_rot(pos.x(), pos.y())
            if ehit >= 0:
                self._dragging_extra_rot_idx = ehit
                self._update_extra_angle_from_widget(ehit, pos.x(), pos.y())
                return
            ehit = self._hit_extra_center(pos.x(), pos.y())
            if ehit >= 0:
                self._dragging_extra_center_idx = ehit
                self._update_extra_pos_from_widget(ehit, pos.x(), pos.y())
                return
            # Per-object origin markers (drag center, không có rotate handle)
            ohit = self._hit_obj_origin_center(pos.x(), pos.y())
            if ohit >= 0:
                self._dragging_obj_origin_idx = ohit
                self._update_obj_origin_pos_from_widget(
                    ohit, pos.x(), pos.y())
                return
            if event.button() == Qt.RightButton and self._shape == "polygon":
                # Right-click: huỷ polygon đang vẽ
                self.cancel_polygon()
                return
            if self._shape == "polygon":
                ix_f = (pos.x() - self._off_x) / self._scale if self._scale else 0.0
                iy_f = (pos.y() - self._off_y) / self._scale if self._scale else 0.0
                if self._arr is not None:
                    H2, W2 = self._arr.shape[:2]
                    ix_f = max(0.0, min(ix_f, W2 - 1.0))
                    iy_f = max(0.0, min(iy_f, H2 - 1.0))
                self._poly_drawing.append((ix_f, iy_f))
                self._render()
                return
            # rect / circle / ellipse — multi: scan list trước; single: shape_data
            self.setFocus()
            if self._shape in ("rect", "ellipse", "circle"):
                if self._multi:
                    hit = self._hit_test_shapes_list(pos.x(), pos.y())
                    if hit is not None:
                        idx, action = hit
                        # Commit active hiện tại trước khi đổi sang shape khác
                        self._commit_active_to_list()
                        entry = self._shapes[idx]
                        self._active_idx = idx
                        self._shape = entry["type"]
                        self._shape_data = dict(entry["data"])
                        self._edit_action = action  # "move"|"tl"|"tr"|"bl"|"br"
                        self._edit_anchor_w = pos
                        self._edit_orig_data = dict(entry["data"])
                        self._render()
                        return
                # Single-mode (hoặc multi không hit) → check edit trên _shape_data
                if self._shape_data:
                    corner = self._hit_corner(pos.x(), pos.y())
                    if corner:
                        self._edit_action = corner
                        self._edit_anchor_w = pos
                        self._edit_orig_data = dict(self._shape_data)
                        return
                    if self._hit_body(pos.x(), pos.y()):
                        self._edit_action = "move"
                        self._edit_anchor_w = pos
                        self._edit_orig_data = dict(self._shape_data)
                        return
            # Click ngoài shape (hoặc chưa có shape) → vẽ mới
            if self._roi_locked:
                # ROI khoá — không tạo shape mới
                return
            if self._multi:
                # Commit active hiện tại; multi-mode KHÔNG xoá list — append mới
                self._commit_active_to_list()
                self._active_idx = None
            self._drag_start = pos
            self._dragging   = True
            self._rect = QRect(pos, QSize(0, 0))
            self._shape_data = {}    # xoá shape cũ (single mode); multi → tạo entry mới khi release

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.mode in ("roi", "template") and self._shape == "polygon" \
                and len(self._poly_drawing) >= 3:
            pts = list(self._poly_drawing)
            self._poly_drawing = []
            xs = [px for px, _ in pts]; ys = [py for _, py in pts]
            x = int(min(xs)); y = int(min(ys))
            w = max(1, int(max(xs) - min(xs)))
            h = max(1, int(max(ys) - min(ys)))
            if self._arr is not None:
                H2, W2 = self._arr.shape[:2]
                x = max(0, min(x, W2 - 1)); y = max(0, min(y, H2 - 1))
                w = max(1, min(w, W2 - x)); h = max(1, min(h, H2 - y))
            self._shape_data = {"pts": pts, "x": x, "y": y, "w": w, "h": h}
            if self._multi:
                self._shapes.append({"type": "polygon",
                                      "data": dict(self._shape_data)})
                self._active_idx = len(self._shapes) - 1
            self.shape_drawn.emit("polygon", dict(self._shape_data))
            self.roi_changed.emit(x, y, w, h)
            self._emit_shapes_changed()
            self._render()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        # Hover color picker: phát toạ độ pixel ảnh khi con trỏ ở trong ảnh.
        if self._arr is not None and self._scale > 0:
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            h2, w2 = self._arr.shape[:2]
            if 0 <= ix < w2 and 0 <= iy < h2:
                self.pixel_hovered.emit(int(ix), int(iy))
            else:
                self.mouse_left.emit()
        # Label drag (Blob output) — sync offset về dialog/slider qua signal.
        if self._dragging_label and self._label_drag_start_img is not None:
            ix, iy = self._widget_to_img(pos.x(), pos.y())
            dx0, dy0 = self._label_drag_start_off
            sx, sy = self._label_drag_start_img
            new_dx = int(round(dx0 + (ix - sx)))
            new_dy = int(round(dy0 + (iy - sy)))
            # Cập nhật state local trước khi emit để hit-test mượt
            self._current_label_off = (new_dx, new_dy)
            self.label_offset_changed.emit(new_dx, new_dy)
            return
        # Đổi cursor khi hover trên label
        if (not self._dragging and not self._dragging_origin
                and not self._dragging_origin_rot
                and self._label_rects):
            if self._hit_label(pos.x(), pos.y()) >= 0:
                self.setCursor(QCursor(Qt.OpenHandCursor))
            else:
                cur_map = {"roi": Qt.CrossCursor, "template": Qt.CrossCursor,
                           "pick": Qt.PointingHandCursor, "view": Qt.ArrowCursor,
                           "readonly": Qt.ArrowCursor}
                self.setCursor(QCursor(cur_map.get(self.mode, Qt.ArrowCursor)))
        if self._dragging_origin_rot:
            self._update_origin_angle_from_widget(pos.x(), pos.y())
            return
        if self._dragging_origin:
            self._update_origin_from_widget(pos.x(), pos.y())
            return
        if self._dragging_extra_rot_idx >= 0:
            self._update_extra_angle_from_widget(
                self._dragging_extra_rot_idx, pos.x(), pos.y())
            return
        if self._dragging_extra_center_idx >= 0:
            self._update_extra_pos_from_widget(
                self._dragging_extra_center_idx, pos.x(), pos.y())
            return
        if self._dragging_obj_origin_idx >= 0:
            self._update_obj_origin_pos_from_widget(
                self._dragging_obj_origin_idx, pos.x(), pos.y())
            return
        if self._edit_action:
            self._apply_edit(pos)
            return
        if not self._dragging or self._drag_start is None:
            return
        if self._shape == "circle":
            cx = self._drag_start.x(); cy = self._drag_start.y()
            dx = pos.x() - cx; dy = pos.y() - cy
            r = int(max(1, (dx * dx + dy * dy) ** 0.5))
            self._rect = QRect(cx - r, cy - r, 2 * r, 2 * r)
        else:
            self._rect = QRect(self._drag_start, pos).normalized()
        self._render()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_label:
            self._dragging_label = False
            self._label_drag_start_img = None
            cur_map = {"roi": Qt.CrossCursor, "template": Qt.CrossCursor,
                       "pick": Qt.PointingHandCursor, "view": Qt.ArrowCursor,
                       "readonly": Qt.ArrowCursor}
            self.setCursor(QCursor(cur_map.get(self.mode, Qt.ArrowCursor)))
            return
        if self._dragging_origin_rot:
            self._dragging_origin_rot = False
            return
        if self._dragging_origin:
            self._dragging_origin = False
            return
        if self._dragging_extra_rot_idx >= 0:
            self._dragging_extra_rot_idx = -1
            return
        if self._dragging_extra_center_idx >= 0:
            self._dragging_extra_center_idx = -1
            return
        if self._dragging_obj_origin_idx >= 0:
            self._dragging_obj_origin_idx = -1
            return
        if self._edit_action:
            # Kết thúc move/resize — phát signal cho dialog cha cập nhật ROI
            self._edit_action = None
            self._edit_anchor_w = None
            self._edit_orig_data = {}
            sd = self._shape_data
            if sd and self._shape in ("rect", "ellipse", "circle") \
                    and "x" in sd and "y" in sd:
                if self._multi:
                    self._commit_active_to_list()
                self.shape_drawn.emit(self._shape, dict(sd))
                self.roi_changed.emit(int(sd["x"]), int(sd["y"]),
                                       int(sd["w"]), int(sd["h"]))
                if self.mode == "template":
                    self.template_drawn.emit(int(sd["x"]), int(sd["y"]),
                                              int(sd["w"]), int(sd["h"]))
                self._emit_shapes_changed()
            self._render()
            return
        if not self._dragging:
            return
        self._dragging = False
        if not (self._rect and self._rect.width() > 4 and self._rect.height() > 4):
            return
        ix, iy = self._widget_to_img(self._rect.left(), self._rect.top())
        iw2 = max(1, int(self._rect.width() / self._scale))
        ih2 = max(1, int(self._rect.height() / self._scale))
        if self._arr is not None:
            H2, W2 = self._arr.shape[:2]
            ix  = max(0, min(ix, W2-1)); iy  = max(0, min(iy, H2-1))
            iw2 = max(1, min(iw2, W2-ix)); ih2 = max(1, min(ih2, H2-iy))

        if self._shape == "rect":
            self._shape_data = {"x": ix, "y": iy, "w": iw2, "h": ih2}
        elif self._shape == "ellipse":
            self._shape_data = {"x": ix, "y": iy, "w": iw2, "h": ih2}
        elif self._shape == "circle":
            cx_w = self._drag_start.x() if self._drag_start else 0
            cy_w = self._drag_start.y() if self._drag_start else 0
            cx_i, cy_i = self._widget_to_img(cx_w, cy_w)
            r_i = max(1, int(self._rect.width() / 2 / self._scale))
            ix = max(0, cx_i - r_i); iy = max(0, cy_i - r_i)
            iw2 = 2 * r_i; ih2 = 2 * r_i
            if self._arr is not None:
                H2, W2 = self._arr.shape[:2]
                iw2 = min(iw2, W2 - ix); ih2 = min(ih2, H2 - iy)
            self._shape_data = {"cx": cx_i, "cy": cy_i, "r": r_i,
                                 "x": ix, "y": iy, "w": iw2, "h": ih2}

        if self._multi and self._shape_data:
            # Append shape mới vào list, set active = last
            self._shapes.append({"type": self._shape,
                                  "data": dict(self._shape_data)})
            self._active_idx = len(self._shapes) - 1

        if self._shape in ("rect", "ellipse", "circle"):
            self.shape_drawn.emit(self._shape, dict(self._shape_data))
        self.roi_changed.emit(ix, iy, iw2, ih2)
        if self.mode == "template":
            self.template_drawn.emit(ix, iy, iw2, ih2)
        self._emit_shapes_changed()
        self._render()

    def leaveEvent(self, event):
        self.mouse_left.emit()
        super().leaveEvent(event)

    def arm_pick(self):
        """Bật pick-once: click kế tiếp trên ảnh emit pixel_picked, dù
        widget đang ở mode 'roi' / 'view'. Tự huỷ sau khi pick."""
        self._pick_once = True
        self.setCursor(QCursor(Qt.CrossCursor))

    def disarm_pick(self):
        self._pick_once = False
        cur_map = {"roi": Qt.CrossCursor, "template": Qt.CrossCursor,
                   "pick": Qt.PointingHandCursor, "view": Qt.ArrowCursor,
                   "readonly": Qt.ArrowCursor}
        self.setCursor(QCursor(cur_map.get(self.mode, Qt.ArrowCursor)))

    def _update_origin_from_widget(self, wx: int, wy: int):
        """Convert widget pos → image coords, clamp ảnh, emit signal.
        Cho phép kéo origin ra ngoài ROI rect (chỉ clamp vào ảnh)."""
        ix_f = (wx - self._off_x) / self._scale if self._scale else 0.0
        iy_f = (wy - self._off_y) / self._scale if self._scale else 0.0
        if self._arr is not None:
            H2, W2 = self._arr.shape[:2]
            ix_f = max(0.0, min(ix_f, W2 - 1.0))
            iy_f = max(0.0, min(iy_f, H2 - 1.0))
        self._origin_xy = (ix_f, iy_f)
        self._show_origin = True
        self._render()
        self.origin_changed.emit(ix_f, iy_f)

    def _update_origin_angle_from_widget(self, wx: int, wy: int):
        """Tính góc xoay axes theo vị trí con trỏ so với tâm origin."""
        c = self._origin_widget_pos()
        if c is None:
            return
        import math as _m
        dx = wx - c[0]; dy = wy - c[1]
        if dx == 0 and dy == 0:
            return
        angle = _m.degrees(_m.atan2(dy, dx))
        self._origin_angle = angle % 360.0
        self._render()
        self.origin_angle_changed.emit(self._origin_angle)

    def keyPressEvent(self, event):
        if self._multi and event.key() in (Qt.Key_Delete, Qt.Key_Backspace) \
                and self._active_idx is not None:
            self.delete_active_shape()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)


# ════════════════════════════════════════════════════════════════════
#  Main Dialog
# ════════════════════════════════════════════════════════════════════
class NodeDetailDialog(QDialog):
    run_requested = Signal(str)

    def __init__(self, node: NodeInstance, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self._node  = node
        self._graph = graph
        tool: ToolDef = node.tool

        self.setWindowTitle(f"{tool.icon}  {node.name}  —  Detail")
        self.setMinimumSize(1000, 650)
        self.resize(1120, 740)
        self.setModal(False)
        # Title bar: thêm minimize + maximize button (mặc định QDialog chỉ
        # có close trên Windows). Giữ Qt.Dialog bằng cách OR thêm hint thay
        # vì replace flags.
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)
        self.setStyleSheet("""
            QDialog { background:#0a0e1a; color:#e2e8f0; }
            QGroupBox { border:1px solid #1e2d45; border-radius:6px;
                        margin-top:8px; padding-top:8px;
                        color:#64748b; font-size:11px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QScrollArea { border:none; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        hdr = QWidget(); hdr.setFixedHeight(54)
        hdr.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {tool.color},stop:1 #0a0e1a);"
            f"border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16, 0, 16, 0)

        icon_l = QLabel(tool.icon)
        icon_l.setStyleSheet("font-size:28px; background:transparent;")
        hl.addWidget(icon_l)

        tc = QVBoxLayout()
        self._title_lbl = QLabel(node.name)
        t1 = self._title_lbl
        t1.setStyleSheet("color:#fff; font-size:16px; font-weight:700; background:transparent;")
        T = f"  {tool.T_equiv}" if tool.T_equiv else ""
        name_pfx = f"{tool.name}  •  " if node.name != tool.name else ""
        t2 = QLabel(f"{name_pfx}{tool.category}{T}  •  {tool.description}")
        t2.setStyleSheet("color:#ffffff88; font-size:11px; background:transparent;")
        tc.addWidget(t1); tc.addWidget(t2)
        hl.addLayout(tc, 1)

        self._auto_run_cb = QCheckBox("Auto Run")
        self._auto_run_cb.setChecked(bool(node.params.get("_auto_run", False)))
        self._auto_run_cb.setStyleSheet(
            "QCheckBox{color:#fff;font-size:12px;font-weight:600;"
            "background:transparent;padding:0 8px;}"
            "QCheckBox::indicator{width:14px;height:14px;"
            "border:1px solid #00d4ff;border-radius:3px;background:#0a0e1a;}"
            "QCheckBox::indicator:checked{background:#00d4ff;}")
        self._auto_run_cb.setToolTip("Tự động chạy node khi thay đổi tham số")
        self._auto_run_cb.toggled.connect(self._on_auto_run_toggled)
        hl.addWidget(self._auto_run_cb)

        self._auto_run_timer = QTimer(self)
        self._auto_run_timer.setSingleShot(True)
        self._auto_run_timer.setInterval(40)
        self._auto_run_timer.timeout.connect(self._on_run)

        self._run_btn = QPushButton("▶  Run Node")
        self._run_btn.setFixedSize(120, 34)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#00d4ff;border:none;border-radius:5px;"
            "color:#000;font-weight:700;font-size:13px;}"
            "QPushButton:hover{background:#33ddff;}"
            "QPushButton:pressed{background:#0099bb;}")
        self._run_btn.clicked.connect(self._on_run)
        hl.addWidget(self._run_btn)
        root.addWidget(hdr)

        # ── Mode hint bar ─────────────────────────────────────────
        self._mode_hint = QLabel("")
        self._mode_hint.setStyleSheet(
            "background:#0d1a2a; color:#ffd700; font-size:11px;"
            "padding:5px 16px; border-bottom:1px solid #1e2d45;")
        self._mode_hint.hide()
        root.addWidget(self._mode_hint)

        # ── Splitter ──────────────────────────────────────────────
        spl = QSplitter(Qt.Horizontal)
        spl.setHandleWidth(1)
        spl.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(spl, 1)

        # LEFT — params
        left = QWidget(); left.setMaximumWidth(300); left.setMinimumWidth(240)
        ll   = QVBoxLayout(left); ll.setContentsMargins(10, 10, 10, 10); ll.setSpacing(8)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0d1220;}
            QTabBar::tab{background:#0a0e1a;color:#64748b;padding:6px 10px;
                         border:none;font-size:11px;font-weight:600;}
            QTabBar::tab:selected{color:#00d4ff;border-bottom:2px solid #00d4ff;}
        """)
        self._params_scroll = QScrollArea(); self._params_scroll.setWidgetResizable(True)
        self._params_scroll.setFrameShape(QFrame.NoFrame)
        self._params_scroll.setWidget(self._build_params_widget())
        tabs.addTab(self._params_scroll, "⚙ Params")
        tabs.addTab(self._build_ports_widget(), "🔌 Ports")
        ll.addWidget(tabs)

        self._out_group = QGroupBox("Output Values")
        og = QVBoxLayout(self._out_group)
        og.setContentsMargins(8, 12, 8, 8); og.setSpacing(4)
        self._out_labels = {}
        self._build_output_labels(og)
        ll.addWidget(self._out_group)

        self._status_lbl = QLabel("Status: IDLE")
        self._status_lbl.setStyleSheet("color:#64748b; font-size:11px; padding:2px;")
        ll.addWidget(self._status_lbl)
        spl.addWidget(left)

        # RIGHT — image preview HOẶC code editor (Script Tool)
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(6, 6, 6, 6); rl.setSpacing(4)

        # Script Tool: thay image panel bằng Python code editor có syntax
        # highlight + autocomplete. Các attr image-related set None/dummy để
        # refresh_outputs không cần if/else khắp nơi.
        if tool.tool_id == "script":
            self._img_label = None
            self._img_info = QLabel("")
            self._hover_bar = QLabel("")
            self._pixel_bar = QLabel("")
            self._build_script_editor_panel(rl)
            spl.addWidget(right)
            spl.setSizes([300, 800])

            cb = QPushButton("Close")
            cb.setFixedHeight(30)
            cb.setStyleSheet(
                "QPushButton{background:#1e2d45;border:none;border-radius:4px;"
                "color:#94a3b8;font-size:12px;margin:4px 12px;}"
                "QPushButton:hover{background:#00d4ff;color:#000;}")
            cb.clicked.connect(self.close)
            root.addWidget(cb)
            self.refresh_outputs()
            return

        img_hdr = QHBoxLayout()
        img_lbl = QLabel("OUTPUT IMAGE")
        img_lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:2px;")
        img_hdr.addWidget(img_lbl); img_hdr.addStretch()

        # Reset Image button — chỉ hiện cho Crop ROI: xoá drawn_roi + đưa
        # x/y/w/h về full ảnh nguồn để vẽ lại từ đầu.
        if tool.tool_id == "crop_roi":
            btn_reset = QPushButton("🔄  Reset Image")
            btn_reset.setFixedHeight(26)
            btn_reset.setStyleSheet(
                "QPushButton{background:#1e2d45;border:1px solid #3a4b6a;"
                "color:#94a3b8;font-size:11px;padding:0 12px;border-radius:4px;}"
                "QPushButton:hover{background:#2c3e60;color:#00d4ff;}")
            btn_reset.clicked.connect(self._on_reset_crop_image)
            img_hdr.addWidget(btn_reset)

        # Zoom buttons (−/+/1:1/fit) — wheel zoom đã có sẵn, nhưng user
        # cần nút bấm rõ ràng. Style giống main Image Viewer.
        def _zb(txt, tip):
            b = QPushButton(txt)
            b.setFixedSize(28, 26)
            b.setToolTip(tip)
            b.setStyleSheet(
                "QPushButton{background:#111827;border:1px solid #1e2d45;"
                "border-radius:4px;color:#94a3b8;font-size:11px;}"
                "QPushButton:hover{background:#00d4ff;color:#000;}")
            return b
        self._btn_zoom_out = _zb("−",   "Zoom out")
        self._btn_zoom_in  = _zb("+",   "Zoom in")
        self._btn_zoom_1   = _zb("1:1", "Actual pixels")
        self._btn_zoom_fit = _zb("⊡",   "Fit to window")
        for b in (self._btn_zoom_out, self._btn_zoom_in,
                  self._btn_zoom_1,   self._btn_zoom_fit):
            img_hdr.addWidget(b)

        # Color tools header — combobox + nút Pick Color.
        # • color_segment: combobox bound vào param `color_space` (HSV/RGB/
        #   HSL/Lab/Gray) → đổi không gian lọc + rebuild params + rerun.
        #   Nút Pick Color: click → arm pick-once → click ảnh kế tiếp set
        #   ngưỡng kênh quanh màu pixel ± tolerance.
        # • color_picker / color_match: combobox display-only, choices có
        #   thêm "All"/"HEX" cho hover bar.
        self._color_mode_cb: Optional[QComboBox] = None
        self._pick_btn: Optional[QPushButton] = None
        if tool.tool_id in ("color_segment", "color_picker", "color_match"):
            mode_lbl_text = "Filter:" if tool.tool_id == "color_segment" else "Color:"
            mode_lbl = QLabel(mode_lbl_text)
            mode_lbl.setStyleSheet("color:#94a3b8; font-size:11px; padding:0 4px 0 12px;")
            img_hdr.addWidget(mode_lbl)
            self._color_mode_cb = QComboBox()
            if tool.tool_id == "color_segment":
                self._color_mode_cb.addItems(
                    ["HSV", "RGB", "HSL", "Lab", "Gray"])
                saved_mode = node.params.get("color_space", "HSV")
            else:
                self._color_mode_cb.addItems(
                    ["All", "RGB", "HSV", "HSL", "Lab", "Gray", "HEX"])
                saved_mode = node.params.get("_color_pick_mode", "All")
            idx = self._color_mode_cb.findText(saved_mode)
            if idx >= 0:
                self._color_mode_cb.setCurrentIndex(idx)
            self._color_mode_cb.setFixedHeight(26)
            self._color_mode_cb.setMinimumWidth(80)
            if tool.tool_id == "color_segment":
                self._color_mode_cb.setToolTip(
                    "Không gian màu dùng để lọc. Đổi mode → params channel "
                    "thresholds cũng đổi theo (HSV/RGB/HSL/Lab/Gray).")
                self._color_mode_cb.currentTextChanged.connect(
                    self._on_color_space_changed)
            else:
                self._color_mode_cb.setToolTip(
                    "Định dạng giá trị màu khi rê chuột trên ảnh.")
                self._color_mode_cb.currentTextChanged.connect(
                    self._on_color_mode_changed)
            self._color_mode_cb.setStyleSheet(
                "QComboBox{background:#111827;border:1px solid #1e2d45;"
                "border-radius:4px;color:#e2e8f0;font-size:11px;padding:2px 8px;}"
                "QComboBox:hover{border-color:#00d4ff;}"
                "QComboBox QAbstractItemView{background:#111827;"
                "color:#e2e8f0;selection-background-color:#00d4ff;"
                "selection-color:#000;}")
            img_hdr.addWidget(self._color_mode_cb)

            if tool.tool_id == "color_segment":
                self._pick_btn = QPushButton("🎯 Pick Color")
                self._pick_btn.setCheckable(True)
                self._pick_btn.setFixedHeight(26)
                self._pick_btn.setToolTip(
                    "Click rồi click vào ảnh để lấy màu pixel làm màu lọc. "
                    "Ngưỡng kênh sẽ được set quanh giá trị pick ± Tolerance.")
                self._pick_btn.setStyleSheet(
                    "QPushButton{background:#111827;border:1px solid #1e2d45;"
                    "border-radius:4px;color:#94a3b8;font-size:11px;"
                    "padding:0 10px;}"
                    "QPushButton:hover{background:#1e2d45;color:#00d4ff;}"
                    "QPushButton:checked{background:#00d4ff;color:#000;"
                    "border-color:#00d4ff;}")
                self._pick_btn.toggled.connect(self._on_pick_color_toggled)
                img_hdr.addWidget(self._pick_btn)

        # ── Chọn mode interactive theo tool ──────────────────────
        self._roi_port_connected = False

        if tool.tool_id == "crop_roi":
            self._roi_port_connected = self._check_roi_ports_connected()
            # Luôn dùng mode "roi" để cho phép kéo/resize ROI.
            # Khi port kết nối: drag chỉ override một frame; port sẽ thắng
            # trong proc_crop ở lần Run kế tiếp. Báo cho user qua mode_hint.
            mode_str = "roi"
            if self._roi_port_connected:
                self._mode_hint.setText(
                    "🔗  Port x/y/w/h kết nối — ROI auto-tracking. Có thể "
                    "kéo override nhưng Run kế tiếp sẽ snap về port value.")
            else:
                self._mode_hint.setText(
                    "✏  Kéo vẽ ROI; sau khi vẽ kéo body để di chuyển hoặc "
                    "4 góc để resize.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode=mode_str)
            self._img_label.roi_changed.connect(self._on_roi_changed)
            drawn = node.params.get("_drawn_roi")
            if drawn:
                x2, y2, w2, h2 = drawn
            else:
                x2 = node.params.get("x", 0); y2 = node.params.get("y", 0)
                w2 = node.params.get("crop_w", 320); h2 = node.params.get("crop_h", 240)
            QTimer.singleShot(120, lambda: self._img_label.set_rect_from_params(x2, y2, w2, h2))

        elif tool.tool_id in ("patmax", "patmax_align", "patfind"):
            # PatMax/PatFind → mở PatMaxDialog chuyên dụng
            self._mode_hint.setText(
                "🎯  PatMax/PatFind — Cửa sổ Train & Search chuyên dụng đang mở...")
            self._mode_hint.setStyleSheet(
                "background:#0d1a2a; color:#00d4ff; font-size:11px;"
                "padding:5px 16px; border-bottom:1px solid #1e2d45;")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="view")
            # Mở PatMaxDialog sau khi dialog này xuất hiện
            QTimer.singleShot(150, self._open_patmax_dialog)

        elif tool.tool_id == "color_picker":
            self._mode_hint.setText(
                "🎨  Click chuột vào ảnh để lấy màu tại điểm đó. "
                "Rê chuột để xem màu live theo mode (RGB/HSV/Lab/HEX…) ở thanh dưới ảnh.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="pick")
            self._img_label.pixel_picked.connect(self._on_pixel_picked)

        elif tool.tool_id == "color_segment":
            # ROI shape: vẽ vùng cần phân tích (rect/circle/ellipse/polygon)
            self._shape_port_connected = self._check_shape_ports_connected()
            if self._shape_port_connected:
                self._mode_hint.setText(
                    "🔗  Port x/y/w/h kết nối — ROI auto-tracking theo "
                    "upstream. Shape vẽ tay làm template, mỗi lần Run sẽ "
                    "dịch/scale theo port. Click 'Pick Color' rồi click ảnh "
                    "để lấy màu lọc.")
            else:
                self._mode_hint.setText(
                    "🎨  Chọn ROI Shape ở params → vẽ vùng cần lọc trên ảnh "
                    "(Full Image = không giới hạn). Click 'Pick Color' rồi "
                    "click ảnh để lấy màu lọc theo Color Space đang chọn.")
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="roi")
            self._img_label.shape_drawn.connect(self._on_color_segment_shape)
            self._img_label.pixel_picked.connect(self._on_color_segment_pick)
            roi_type = node.params.get("roi_shape", "Full Image")
            shape_key = self._color_seg_shape_key(roi_type)
            if shape_key:
                self._img_label.set_shape_mode(shape_key)
                saved = node.params.get("_roi_shape_data")
                saved_t = node.params.get("_roi_shape_type")
                if saved and saved_t == shape_key:
                    QTimer.singleShot(120,
                        lambda s=shape_key, d=dict(saved):
                            self._img_label.set_shape_data(s, d))
            else:
                # Full Image → khoá vẽ shape
                self._img_label.set_roi_locked(True)

        elif tool.tool_id in ("create_rectangle", "create_circle",
                              "create_ellipse", "create_trapezoid",
                              "create_polygon"):
            # Create Shape tools — vẽ hình bằng kéo chuột (rect/circle/ellipse/
            # polygon). Trapezoid vẽ bằng bbox (shape "rect") → proc_ dựng hình
            # thang theo top_ratio. Vẽ xong → ghi geometry vào params + rerun.
            self._create_shape_key = {
                "create_rectangle": "rect", "create_circle": "circle",
                "create_ellipse": "ellipse", "create_trapezoid": "rect",
                "create_polygon": "polygon"}[tool.tool_id]
            if self._create_shape_key == "polygon":
                hint = ("✏  Click từng đỉnh trên ảnh, double-click để chốt "
                        "(≥3 điểm). Hoặc nhập cx/cy/r/Sides ở Params.")
            else:
                hint = ("✏  Kéo chuột vẽ hình trên ảnh — vẽ xong kéo thân để "
                        "di chuyển, kéo góc để resize. Hoặc nhập toạ độ ở Params.")
            self._mode_hint.setText(hint)
            self._mode_hint.show()
            self._img_label = InteractiveImageLabel(mode="roi")
            self._img_label.shape_drawn.connect(self._on_create_shape_drawn)
            self._img_label.set_shape_mode(self._create_shape_key)
            # Nền + overlay handle dựng ở 120ms (sau refresh_outputs cuối __init__,
            # vốn clear ảnh khi node chưa chạy).
            QTimer.singleShot(120, self._restore_create_shape_overlay)

        else:
            self._img_label = InteractiveImageLabel(mode="view")

        # Tool nào xuất _label_rects (vd Blob) → cho drag label sync về slider.
        self._img_label.label_offset_changed.connect(self._on_label_dragged)

        # Wire zoom buttons → InteractiveImageLabel methods (label vừa khởi
        # tạo xong nên giờ mới connect được).
        self._btn_zoom_out.clicked.connect(self._img_label.zoom_out)
        self._btn_zoom_in.clicked.connect(self._img_label.zoom_in)
        self._btn_zoom_1.clicked.connect(self._img_label.zoom_actual)
        self._btn_zoom_fit.clicked.connect(self._img_label.reset_zoom)

        img_hdr.addWidget(QWidget())   # spacer placeholder
        rl.addLayout(img_hdr)
        rl.addWidget(self._img_label, 1)

        self._img_info = QLabel("")
        self._img_info.setStyleSheet(
            "color:#64748b; font-size:10px; font-family:'Courier New'; padding:2px;")
        self._img_info.setAlignment(Qt.AlignCenter)
        rl.addWidget(self._img_info)

        # Hover color bar — hiển thị màu pixel dưới con trỏ chuột cho
        # các tool màu. Tự ẩn khi tool khác hoặc khi chuột rời ảnh.
        self._hover_bar = QLabel("")
        self._hover_bar.setTextFormat(Qt.RichText)
        self._hover_bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._hover_bar.setStyleSheet(
            "color:#e2e8f0; font-size:11px; font-family:'Courier New';"
            "background:#0d1a2a; border:1px solid #1e2d45; border-radius:4px;"
            "padding:3px 10px;")
        self._hover_bar.hide()
        rl.addWidget(self._hover_bar)
        # Cache ảnh nguồn để sample màu chính xác (không bị overlay ảnh hưởng).
        self._source_bgr: Optional[np.ndarray] = None
        if tool.tool_id in ("color_segment", "color_picker", "color_match"):
            self._img_label.pixel_hovered.connect(self._on_pixel_hovered)
            self._img_label.mouse_left.connect(self._on_image_mouse_left)

        self._pixel_bar = QLabel("")
        self._pixel_bar.setStyleSheet(
            "color:#ffd700; font-size:11px; font-family:'Courier New';"
            "background:#111827; border-radius:4px; padding:3px 10px;")
        self._pixel_bar.hide()
        rl.addWidget(self._pixel_bar)

        spl.addWidget(right)
        spl.setSizes([270, 830])

        cb = QPushButton("Close")
        cb.setFixedHeight(30)
        cb.setStyleSheet(
            "QPushButton{background:#1e2d45;border:none;border-radius:4px;"
            "color:#94a3b8;font-size:12px;margin:4px 12px;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        cb.clicked.connect(self.close)
        root.addWidget(cb)

        self.refresh_outputs()

    # ════════════════════════════════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════════════════════════════════
    def _check_roi_ports_connected(self) -> bool:
        """Trả về True nếu ít nhất một port x/y/w/h được kết nối."""
        node = self._node
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id and conn.dst_port in ("x","y","w","h"):
                return True
        return False

    def _check_shape_ports_connected(self) -> bool:
        """Dùng cho color_segment — giống _check_roi_ports_connected nhưng
        tên rõ ràng hơn (shape có thể là rect/circle/ellipse/polygon)."""
        return self._check_roi_ports_connected()

    def _get_input_image(self) -> Optional[np.ndarray]:
        for conn in self._graph.connections:
            if conn.dst_id == self._node.node_id and conn.dst_port == "image":
                src = self._graph.nodes.get(conn.src_id)
                if src and "image" in src.outputs:
                    return src.outputs["image"]
        return None

    # ════════════════════════════════════════════════════════════════
    #  Script Tool: code editor panel
    # ════════════════════════════════════════════════════════════════
    def _build_script_editor_panel(self, layout: QVBoxLayout):
        """Right-pane cho Script Tool: header + Python code editor + hint.
        Editor 2-way sync với node.params['expression']."""
        from ui.code_editor import CodeEditor
        node = self._node

        # Header
        hdr = QHBoxLayout()
        lbl = QLabel("🐍  PYTHON SCRIPT")
        lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:2px;")
        hdr.addWidget(lbl); hdr.addStretch()
        layout.addLayout(hdr)

        # Editor
        self._code_editor = CodeEditor()
        self._code_editor.setPlainText(node.params.get("expression", "") or "")
        self._refresh_script_completions()
        layout.addWidget(self._code_editor, 1)

        # Save on change — debounce 250ms để giảm dirty-mark spam khi gõ.
        self._script_save_timer = QTimer(self)
        self._script_save_timer.setSingleShot(True)
        self._script_save_timer.setInterval(250)
        self._script_save_timer.timeout.connect(self._save_script_to_params)
        self._code_editor.textChanged.connect(self._script_save_timer.start)

        # Hint bar — quick reference
        hint = QLabel(
            "<span style='color:#94a3b8'>Inputs:</span> "
            "<span style='color:#4ec9b0'>a, b, c</span> "
            "(lowercase của port name) hoặc <span style='color:#4ec9b0'>inputs['A']</span>"
            "<br><span style='color:#94a3b8'>Outputs:</span> gán biến cùng tên port "
            "(<span style='color:#dcdcaa'>result = 100</span>), "
            "<span style='color:#dcdcaa'>pass_value = True</span> cho port pass, "
            "hoặc <span style='color:#dcdcaa'>outputs['name'] = ...</span>"
            "<br><span style='color:#94a3b8'>Add port:</span> right-click node → "
            "Add Input / Add Output. <span style='color:#94a3b8'>Autocomplete:</span> "
            "<span style='color:#569cd6'>Ctrl+Space</span>")
        hint.setTextFormat(Qt.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "color:#94a3b8; font-size:11px; font-family:'Segoe UI';"
            "background:#0d1a2a; border:1px solid #1e2d45; border-radius:4px;"
            "padding:6px 10px;")
        layout.addWidget(hint)

    def _save_script_to_params(self):
        """Flush editor content → node.params['expression']. Mark dirty
        thông qua _on_run cycle hoặc params_changed nếu có."""
        if not hasattr(self, "_code_editor"):
            return
        text = self._code_editor.toPlainText()
        if self._node.params.get("expression") != text:
            self._node.params["expression"] = text
            if self._node.params.get("_auto_run", False):
                self._auto_run_timer.start()

    def _refresh_script_completions(self):
        """Reload danh sách autocomplete = port input/output hiện tại + keywords."""
        if not hasattr(self, "_code_editor"):
            return
        node = self._node
        extra: list = ["inputs", "outputs", "params", "pass_value"]
        # Input port names (uppercase + lowercase)
        for p in node.tool.inputs:
            extra.append(p.name)
            if p.name != p.name.lower():
                extra.append(p.name.lower())
        for name in (node.params.get("_extra_inputs") or []):
            extra.append(name)
            if name != name.lower():
                extra.append(name.lower())
        # Output port names
        for p in node.tool.outputs:
            extra.append(p.name)
        for name in (node.params.get("_extra_outputs") or []):
            extra.append(name)
        self._code_editor.set_completions(extra)

    # ════════════════════════════════════════════════════════════════
    #  Build sub-widgets
    # ════════════════════════════════════════════════════════════════
    def _build_params_widget(self) -> QWidget:
        node = self._node; tool = node.tool
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(5)
        if not tool.params:
            lbl = QLabel("No parameters.")
            lbl.setStyleSheet("color:#1e2d45; font-size:12px;")
            lbl.setAlignment(Qt.AlignCenter); lay.addWidget(lbl)
        else:
            self._param_rows = {}
            for param in tool.params:
                # Conditional visibility (visible_if)
                if getattr(param, "visible_if", None):
                    ok = True
                    for k, v in param.visible_if.items():
                        if node.params.get(k) != v:
                            ok = False; break
                    if not ok:
                        continue
                pr = ParamRow(param, node.params.get(param.name, param.default))
                if param.tooltip:
                    pr.setToolTip(param.tooltip)
                pr.value_changed.connect(
                    lambda name, val, nid=node.node_id: self._on_param(nid, name, val))
                lay.addWidget(pr)
                self._param_rows[param.name] = pr

        # YOLO Detect: panel hiển thị metadata model (classes/task/imgsz)
        # khi user add file. Update khi model_path param đổi.
        if tool.tool_id == "yolo_detect":
            self._yolo_info_lbl = QLabel("📂  Add file .pt hoặc .onnx để bắt đầu")
            self._yolo_info_lbl.setWordWrap(True)
            self._yolo_info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._yolo_info_lbl.setStyleSheet(
                "color:#94a3b8; font-size:11px; background:#0a0e1a;"
                "border:1px solid #1e2d45; border-radius:4px;"
                "padding:8px; font-family:'Courier New';")
            lay.addWidget(self._yolo_info_lbl)
            self._refresh_yolo_info()

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet("color:#1e2d45;")
        lay.addWidget(sep)
        note = QLabel("▶ Run Node để áp dụng thay đổi")
        note.setStyleSheet("color:#1e2d45; font-size:10px;")
        note.setAlignment(Qt.AlignCenter)
        lay.addWidget(note); lay.addStretch()
        return w

    def _refresh_yolo_info(self):
        """Đọc metadata model file → đổ vào panel info."""
        lbl = getattr(self, "_yolo_info_lbl", None)
        if lbl is None:
            return
        path = self._node.params.get("model_path", "")
        if not path:
            lbl.setText("📂  Add file .pt hoặc .onnx để bắt đầu")
            lbl.setStyleSheet(
                "color:#94a3b8; font-size:11px; background:#0a0e1a;"
                "border:1px solid #1e2d45; border-radius:4px;"
                "padding:8px; font-family:'Courier New';")
            return
        try:
            from core.tool_registry import yolo_inspect_model
            info = yolo_inspect_model(path)
        except Exception as e:
            info = {"ok": False, "error": str(e)}

        if not info.get("ok"):
            lbl.setText(f"⚠  {info.get('error', 'không đọc được model')}\n"
                        f"   {path}")
            lbl.setStyleSheet(
                "color:#ff9966; font-size:11px; background:#1a0a0a;"
                "border:1px solid #4a1a1a; border-radius:4px;"
                "padding:8px; font-family:'Courier New';")
            return

        import os as _os
        fmt = info.get("format", "?").upper()
        task = info.get("task", "detect")
        imgsz = info.get("imgsz", "—")
        size_mb = info.get("size_mb", 0)
        names = info.get("names") or []
        nc = info.get("num_classes", len(names))
        onnx_sib = info.get("onnx_sibling")

        names_preview = ", ".join(names[:8])
        if len(names) > 8:
            names_preview += f", … (+{len(names) - 8})"
        if not names:
            names_preview = "(metadata không có class names)"

        backend_hint = ""
        if fmt == "ONNX":
            backend_hint = "  • Fast path: ONNX Runtime → cv2.dnn"
        elif onnx_sib:
            backend_hint = f"  • Fast path: dùng {_os.path.basename(onnx_sib)} cạnh đó"
        else:
            backend_hint = "  • Chậm: PyTorch CPU (export .onnx để tăng tốc 3-4x)"

        text = (f"📦  {_os.path.basename(path)}  ({fmt}, {size_mb} MB)\n"
                f"   Task:    {task}\n"
                f"   Imgsz:   {imgsz}\n"
                f"   Classes ({nc}): {names_preview}\n"
                f"{backend_hint}")
        lbl.setText(text)
        lbl.setToolTip(path + (f"\nONNX sibling: {onnx_sib}" if onnx_sib else ""))
        lbl.setStyleSheet(
            "color:#39ff14; font-size:11px; background:#0a0e1a;"
            "border:1px solid #1b4332; border-radius:4px;"
            "padding:8px; font-family:'Courier New';")

    def _build_ports_widget(self) -> QWidget:
        tool = self._node.tool; w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)

        def section(label, color):
            h = QLabel(label)
            h.setStyleSheet(f"color:{color}; font-size:10px; font-weight:700; "
                            f"letter-spacing:1.5px; margin-top:4px;")
            lay.addWidget(h)

        if tool.inputs:
            section("INPUTS", "#00d4ff")
            for p in tool.inputs:
                connected = any(c.dst_id == self._node.node_id and c.dst_port == p.name
                                for c in self._graph.connections)
                status = "🔗" if connected else ("○" if not p.required else "●")
                r = QLabel(f"  {status}  {p.name}  [{p.data_type}]"
                           f"{'  (opt)' if not p.required else ''}")
                col = "#39ff14" if connected else "#00b4d8"
                r.setStyleSheet(
                    f"color:{col}; font-size:11px; font-family:'Courier New';"
                    f"background:#0a0e1a; border-radius:3px; padding:3px 6px;")
                lay.addWidget(r)

        if tool.outputs:
            section("OUTPUTS", "#ff8c42")
            for p in tool.outputs:
                r = QLabel(f"  ⬤  {p.name}  [{p.data_type}]")
                r.setStyleSheet(
                    "color:#ff8c42; font-size:11px; font-family:'Courier New';"
                    "background:#0a0e1a; border-radius:3px; padding:3px 6px;")
                lay.addWidget(r)

        # Crop ROI special: show connection status
        if tool.tool_id == "crop_roi":
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color:#1e2d45;"); lay.addWidget(sep)
            ports_status = []
            for pname in ("x","y","w","h"):
                conn = any(c.dst_id == self._node.node_id and c.dst_port == pname
                           for c in self._graph.connections)
                ports_status.append(f"{pname}:{'🔗' if conn else '✏'}")
            note = QLabel("  ".join(ports_status))
            note.setStyleSheet(
                "color:#ffd700; font-size:11px; font-family:'Courier New';"
                "padding:4px 6px; background:#0d1a2a; border-radius:4px;")
            lay.addWidget(note)

        lay.addStretch()
        return w

    def _build_output_labels(self, layout):
        self._out_labels = {}
        for port in self._node.tool.outputs:
            if port.name == "image":
                continue
            row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
            k = QLabel(port.name)
            k.setStyleSheet("color:#64748b; font-size:11px; font-family:'Courier New';")
            k.setMinimumWidth(80)
            v = QLabel("—")
            v.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            rl.addWidget(k); rl.addWidget(v, 1)
            layout.addWidget(row)
            self._out_labels[port.name] = v

    # ════════════════════════════════════════════════════════════════
    #  Interactive callbacks
    # ════════════════════════════════════════════════════════════════
    def _on_roi_changed(self, x, y, w, h):
        """
        Kéo chuột vẽ ROI thủ công → lưu vào _drawn_roi.
        Sync luôn x/y/crop_w/crop_h vào params + spinbox để hiển thị
        đúng vùng đang được cắt.
        """
        node = self._node
        node.params["_drawn_roi"] = (x, y, w, h)
        node.params["x"]      = int(x)
        node.params["y"]      = int(y)
        node.params["crop_w"] = int(w)
        node.params["crop_h"] = int(h)

        # Cập nhật spinbox params để hiển thị (nhưng _drawn_roi là nguồn truth)
        for name, val in [("x", x), ("y", y), ("crop_w", w), ("crop_h", h)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)

        # Nếu port x/y nhận từ PatMax Ref → đẩy ngược drag về PatMax model.extra_refs
        # → next Run sẽ tính ra đúng vị trí drag (không snap về cũ).
        self._propagate_drag_to_patmax_ref(int(x), int(y))

        # Trigger rerun để output x/y/w/h cập nhật → downstream tools
        # đang nhận x,y từ Crop ROI cũng tự re-process trong pipeline run.
        if getattr(self, "_auto_run_cb", None) and self._auto_run_cb.isChecked():
            self._auto_run_timer.start()

    def _propagate_drag_to_patmax_ref(self, new_x: int, new_y: int):
        """Tìm upstream PatMax feed x/y port → update model.extra_refs để drag
        Crop ROI persist (ngược dòng vào source). Chỉ áp dụng khi:
        - Cả x và y đến từ cùng 1 PatMax node
        - Source port là dạng 'refN_x' / 'refN_y' (cùng index N)
        - PatMax đã có model + có kết quả detect (cần origin + angle để
          đảo ngược pose transform về pattern-local).
        """
        import re, math
        if not self._graph:
            return
        node = self._node
        upstream = {}   # {"x" or "y": (src_node, src_port)}
        for c in self._graph.connections:
            if c.dst_id == node.node_id and c.dst_port in ("x", "y"):
                src = self._graph.nodes.get(c.src_id)
                if src and src.tool.tool_id in ("patmax", "patmax_align"):
                    upstream[c.dst_port] = (src, c.src_port)
        if "x" not in upstream or "y" not in upstream:
            return
        src_x, port_x = upstream["x"]
        src_y, port_y = upstream["y"]
        if src_x.node_id != src_y.node_id:
            return
        src = src_x

        # Parse refN_x / refN_y (cùng N). "x"/"y" trần = object origin (ref_idx=0).
        m_x = re.match(r"^ref(\d+)_(x|y)$", port_x)
        m_y = re.match(r"^ref(\d+)_(x|y)$", port_y)
        if m_x and m_y:
            if m_x.group(1) != m_y.group(1):
                return
            if m_x.group(2) != "x" or m_y.group(2) != "y":
                return
            ref_idx = int(m_x.group(1))   # 1-based: ref1 → extras[0]
        elif port_x == "x" and port_y == "y":
            ref_idx = 0   # origin (training pattern center)
        else:
            return

        # Cần result detect hiện tại để inverse pose. Lấy từ src.outputs.
        # Anchor cho ref transform = BBOX CENTER (r.x, r.y) — không phải
        # 'x'/'y' top-level (đó là pattern-origin transformed cho obj 0).
        objs = src.outputs.get("objects") or []
        if not objs:
            return
        obj0 = objs[0]
        obj_cx = float(obj0.get("center_x", obj0.get("x", 0.0)))
        obj_cy = float(obj0.get("center_y", obj0.get("y", 0.0)))
        obj_ang_deg = float(src.outputs.get("angle", 0.0))
        sc = float(src.outputs.get("scale", 1.0) or 1.0)
        if sc == 0:
            sc = 1.0

        # Engine forward: image = anchor + sc * R(-angle) * (local - pattern_center)
        # Inverse:       local = pattern_center + (1/sc) * R(+angle) * (image - anchor)
        # R(+a) * (vx, vy) = (vx*cos(a) - vy*sin(a), vx*sin(a) + vy*cos(a))
        import math as _m
        rad = _m.radians(obj_ang_deg)
        ca = _m.cos(rad); sa = _m.sin(rad)
        dx = float(new_x) - obj_cx
        dy = float(new_y) - obj_cy
        edx = (dx * ca - dy * sa) / sc
        edy = (dx * sa + dy * ca) / sc

        # Update model
        model = src.params.get("_patmax_model")
        if model is None:
            return
        pw = float(getattr(model, "pattern_w", 0) or 0)
        ph = float(getattr(model, "pattern_h", 0) or 0)
        new_local_x = edx + pw / 2.0
        new_local_y = edy + ph / 2.0

        if ref_idx == 0:
            # Origin (trained pattern center). SET (không cộng dồn).
            if hasattr(model, "origin_x"):
                model.origin_x = float(new_local_x)
                model.origin_y = float(new_local_y)
        else:
            extras = getattr(model, "extra_refs", None)
            if not isinstance(extras, list):
                return
            idx = ref_idx - 1
            if 0 <= idx < len(extras):
                extras[idx]["x"] = float(new_local_x)
                extras[idx]["y"] = float(new_local_y)
        # Đánh dấu PatMax node dirty → graph run sẽ pick up
        src.params["_patmax_model"] = model

        # Re-run upstream PatMax NGAY để outputs có giá trị ref mới — nếu
        # không, Crop ROI re-run kế tiếp sẽ đọc PatMax outputs CŨ → snap-back.
        try:
            up_inputs = {}
            for c2 in self._graph.connections:
                if c2.dst_id == src.node_id:
                    src2 = self._graph.nodes.get(c2.src_id)
                    if src2 and c2.src_port in src2.outputs:
                        up_inputs[c2.dst_port] = src2.outputs[c2.src_port]
            for port in src.tool.inputs:
                up_inputs.setdefault(port.name, port.default)
            out = src.tool.process_fn(up_inputs, src.params)
            src.outputs = out if out else {}
        except Exception:
            pass

        # Refresh open PatMax dialog nếu có (để Ref list + canvas marker update)
        if self.parent() is not None:
            for w in self.parent().children() if hasattr(self.parent(), "children") else []:
                try:
                    if w.__class__.__name__ == "PatMaxDialog" and getattr(w, "_node", None) is src:
                        if hasattr(w, "_refresh_references_list"):
                            w._refresh_references_list()
                        if hasattr(w, "_refresh_canvas_extras"):
                            w._refresh_canvas_extras()
                except Exception:
                    pass

    def _on_reset_crop_image(self):
        """Reset Crop ROI về full ảnh nguồn — xoá _drawn_roi + set
        x=y=0, w/h = kích thước input image."""
        node = self._node
        # Tìm input image: ưu tiên upstream output, fallback node.outputs
        src_img = None
        if self._graph:
            for c in self._graph.connections:
                if c.dst_id == node.node_id and c.dst_port == "image":
                    s = self._graph.nodes.get(c.src_id)
                    if s and "image" in s.outputs:
                        src_img = s.outputs["image"]; break
        if src_img is None:
            src_img = node.outputs.get("image")
        if src_img is None:
            QMessageBox.information(self, "Reset Image",
                "Chưa có ảnh nguồn — chạy pipeline trước rồi reset.")
            return
        h, w = src_img.shape[:2]
        node.params.pop("_drawn_roi", None)
        node.params["x"]      = 0
        node.params["y"]      = 0
        node.params["crop_w"] = int(w)
        node.params["crop_h"] = int(h)
        node.params["_crop_initialized"] = True
        # Sync spinbox
        for name, val in [("x", 0), ("y", 0), ("crop_w", w), ("crop_h", h)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        # Clear vẽ trên ảnh
        if hasattr(self, "_img_label"):
            self._img_label.set_rect_from_params(0, 0, w, h)
        self._img_info.setText(f"🔄 Reset → full image ({w}x{h})")

        # Cập nhật info label
        self._img_info.setText(
            f"Manual ROI: ({x},{y})  {w}×{h} px  —  ▶ Run Node để crop")
        self._img_info.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-family:'Courier New'; padding:2px;")

    def _open_patmax_dialog(self):
        """Mở cửa sổ PatMax chuyên dụng."""
        from ui.patmax_dialog import PatMaxDialog
        # Nếu đã mở rồi, bring to front
        existing = getattr(self, '_patmax_dlg', None)
        if existing and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        dlg = PatMaxDialog(self._node, self._graph, self)
        dlg.run_requested.connect(self.run_requested)
        dlg.model_trained.connect(self._on_patmax_model_trained)
        self._patmax_dlg = dlg
        dlg.show()
        # Update mode hint
        self._mode_hint.setText("🎯  PatMax dialog mở — Train & Search tại đó.")

    def _on_patmax_model_trained(self):
        """Callback khi PatMax train xong — refresh image preview."""
        self.refresh_outputs()

    def _on_template_drawn(self, x, y, w, h):
        """Vẽ ROI → cắt template → lưu vào params."""
        node = self._node
        img = node.outputs.get("image") or self._get_input_image()
        if img is None:
            QMessageBox.warning(self, "Template",
                                "Cần ảnh để cắt template.\n"
                                "Kết nối node Image Source và Run Node trước.")
            return
        H2, W2 = img.shape[:2]
        x = max(0, min(x, W2-1)); y = max(0, min(y, H2-1))
        w = max(1, min(w, W2-x)); h = max(1, min(h, H2-y))
        templ = img[y:y+h, x:x+w]
        node.params["_template_array"] = templ
        node.params["_template_rect"]  = (x, y, w, h)
        self._img_info.setText(
            f"✔ Template saved: ({x},{y})  {w}×{h} px  —  ▶ Run Node")
        self._img_info.setStyleSheet(
            "color:#ffd700; font-size:10px; font-family:'Courier New'; padding:2px;")

    def _sample_color_bgr(self, x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """Lấy (B, G, R) tại (x, y) từ ảnh nguồn (input). Fallback sang ảnh
        đang hiển thị nếu chưa có nguồn cache."""
        img = self._source_bgr
        if img is None or not isinstance(img, np.ndarray):
            img = getattr(self._img_label, "_arr", None)
        if img is None or not isinstance(img, np.ndarray):
            return None
        if img.ndim == 2:
            v = int(img[y, x])
            return (v, v, v)
        if img.ndim != 3 or img.shape[2] < 3:
            return None
        h, w = img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        b, g, r = img[y, x, 0], img[y, x, 1], img[y, x, 2]
        return (int(b), int(g), int(r))

    def _format_color_text(self, x: int, y: int,
                            bgr: Tuple[int, int, int]) -> str:
        """Format chuỗi HTML hiển thị màu theo mode hiện tại."""
        import cv2
        b, g, r = bgr
        mode = (self._color_mode_cb.currentText()
                if self._color_mode_cb is not None else "All")
        pix = np.uint8([[[b, g, r]]])
        H, S, V = (int(v) for v in cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0])
        L_, A_, B_ = (int(v) for v in cv2.cvtColor(pix, cv2.COLOR_BGR2LAB)[0, 0])
        # HSL tính tay (cv2 không có HSL trực tiếp)
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        mx = max(rn, gn, bn); mn = min(rn, gn, bn); d = mx - mn
        L = (mx + mn) / 2.0
        if d == 0:
            h_deg = 0.0; s_pct = 0.0
        else:
            s_pct = d / (1.0 - abs(2 * L - 1.0)) if (1.0 - abs(2 * L - 1.0)) > 1e-6 else 0.0
            if mx == rn:
                h_deg = ((gn - bn) / d) % 6.0
            elif mx == gn:
                h_deg = ((bn - rn) / d) + 2.0
            else:
                h_deg = ((rn - gn) / d) + 4.0
            h_deg *= 60.0
        hsl_h = int(round(h_deg)) % 360
        hsl_s = int(round(s_pct * 100))
        hsl_l = int(round(L * 100))
        gray = int(round(0.114 * b + 0.587 * g + 0.299 * r))
        hex_str = f"#{r:02X}{g:02X}{b:02X}"
        swatch = (f"<span style='background:rgb({r},{g},{b});"
                  f"color:rgb({r},{g},{b});border:1px solid #1e2d45;"
                  f"padding:0 8px;margin-right:6px;'>███</span>")
        pos = f"<b style='color:#94a3b8;'>({x},{y})</b>"
        parts = [swatch, pos]
        if mode == "All":
            parts += [
                f"<span style='color:#ffd700;'>RGB</span> {r},{g},{b}",
                f"<span style='color:#94a3b8;'>{hex_str}</span>",
                f"<span style='color:#ffd700;'>HSV</span> {H},{S},{V}",
                f"<span style='color:#ffd700;'>Lab</span> {L_},{A_},{B_}",
            ]
        elif mode == "RGB":
            parts += [f"<span style='color:#ffd700;'>RGB</span> {r}, {g}, {b}",
                      f"<span style='color:#94a3b8;'>{hex_str}</span>"]
        elif mode == "HSV":
            parts += [f"<span style='color:#ffd700;'>HSV</span> {H}, {S}, {V}",
                      f"<span style='color:#64748b;'>(OpenCV: H 0–180)</span>"]
        elif mode == "HSL":
            parts += [f"<span style='color:#ffd700;'>HSL</span> "
                      f"{hsl_h}°, {hsl_s}%, {hsl_l}%"]
        elif mode == "Lab":
            parts += [f"<span style='color:#ffd700;'>Lab</span> "
                      f"L={L_} a={A_} b={B_}"]
        elif mode == "Gray":
            parts += [f"<span style='color:#ffd700;'>Gray</span> {gray}"]
        elif mode == "HEX":
            parts += [f"<span style='color:#ffd700;'>HEX</span> {hex_str}"]
        return "  ".join(parts)

    def _on_pixel_hovered(self, x: int, y: int):
        """Rê chuột trên ảnh → cập nhật hover bar màu."""
        bgr = self._sample_color_bgr(x, y)
        if bgr is None:
            self._hover_bar.hide()
            return
        self._hover_bar.setText(self._format_color_text(x, y, bgr))
        self._hover_bar.show()

    def _on_image_mouse_left(self):
        self._hover_bar.hide()

    def _on_color_mode_changed(self, mode: str):
        """color_picker/color_match: combobox display-only."""
        self._node.params["_color_pick_mode"] = mode

    def _on_color_space_changed(self, space: str):
        """color_segment: combobox bound vào param color_space."""
        self._node.params["color_space"] = space
        # Rebuild params tab (visible_if đổi)
        QTimer.singleShot(0, self._rebuild_params_tab)
        if getattr(self, "_auto_run_cb", None) and self._auto_run_cb.isChecked():
            self._auto_run_timer.start()
        else:
            self._on_run()

    def _on_pick_color_toggled(self, checked: bool):
        """Toggle pick-once: arm/disarm InteractiveImageLabel."""
        if checked:
            self._img_label.arm_pick()
        else:
            self._img_label.disarm_pick()

    def _bgr_to_space(self, bgr: Tuple[int, int, int],
                       space: str) -> Tuple[int, int, int]:
        """Chuyển BGR pixel → tuple 3 channel theo color space của
        color_segment. Gray trả về (gray, 0, 0); chỉ kênh đầu có nghĩa."""
        import cv2
        b, g, r = bgr
        pix = np.uint8([[[b, g, r]]])
        if space == "HSV":
            H, S, V = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0, 0]
            return int(H), int(S), int(V)
        if space == "RGB":
            return int(r), int(g), int(b)
        if space == "HSL":
            H, L, S = cv2.cvtColor(pix, cv2.COLOR_BGR2HLS)[0, 0]
            return int(H), int(L), int(S)
        if space == "Lab":
            L, A, B = cv2.cvtColor(pix, cv2.COLOR_BGR2LAB)[0, 0]
            return int(L), int(A), int(B)
        if space == "Gray":
            gray = int(round(0.114 * b + 0.587 * g + 0.299 * r))
            return gray, 0, 0
        return int(r), int(g), int(b)

    # Mapping (color_space) → list of (param_prefix, channel_max). Dùng cho
    # pick handler set channel low/high. Gray chỉ có 1 channel.
    _COLOR_SPACE_CHANNELS = {
        "HSV":  [("h", 180), ("s", 255), ("v", 255)],
        "RGB":  [("r", 255), ("g", 255), ("b", 255)],
        "HSL":  [("hsl_h", 180), ("hsl_l", 255), ("hsl_s", 255)],
        "Lab":  [("lab_l", 255), ("lab_a", 255), ("lab_b", 255)],
        "Gray": [("gray", 255)],
    }

    def _on_color_segment_pick(self, x: int, y: int):
        """Click sau khi arm Pick Color → set thresholds quanh màu pixel."""
        bgr = self._sample_color_bgr(x, y)
        if bgr is None:
            return
        node = self._node
        space = node.params.get("color_space", "HSV")
        channels = self._COLOR_SPACE_CHANNELS.get(space, [])
        if not channels:
            return
        vals = self._bgr_to_space(bgr, space)
        pick_tol = int(node.params.get("tolerance", 0) or 0)
        if pick_tol <= 0:
            pick_tol = 20    # default rộng vừa phải khi tolerance=0
        for i, (prefix, vmax) in enumerate(channels):
            v = vals[i]
            lo = max(0, v - pick_tol)
            hi = min(vmax, v + pick_tol)
            for nm, val in ((f"{prefix}_low", lo), (f"{prefix}_high", hi)):
                node.params[nm] = val
                pr = getattr(self, "_param_rows", {}).get(nm)
                if pr:
                    ed = pr._editor
                    if hasattr(ed, "setValue"):
                        ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        # Lưu vị trí pick để hiển thị + uncheck button
        node.params["_last_pick"] = (int(x), int(y))
        if self._pick_btn is not None:
            self._pick_btn.blockSignals(True)
            self._pick_btn.setChecked(False)
            self._pick_btn.blockSignals(False)
        self._img_label.disarm_pick()
        self._on_run()

    def _on_pixel_picked(self, x, y):
        """Click → lấy màu pixel."""
        node = self._node
        node.params["pick_x"] = x; node.params["pick_y"] = y
        for name, val in [("pick_x", x), ("pick_y", y)]:
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True); ed.setValue(val); ed.blockSignals(False)
        self._pixel_bar.show()
        self._on_run()

    # ── Create Shape tools ─────────────────────────────────────────
    def _restore_create_shape_overlay(self):
        """Dựng nền + overlay shape (handle chỉnh sửa) từ params hiện tại.
        Params là source-of-truth; overlay chỉ để xem/kéo."""
        if not getattr(self, "_create_shape_key", None) or self._img_label is None:
            return
        node = self._node
        # Nền: nếu node đã chạy → giữ ảnh output (đã có hình); else dùng ảnh
        # upstream hoặc canvas đen để có chỗ vẽ.
        if not isinstance(node.outputs.get("image"), np.ndarray):
            base = self._get_input_image()
            if base is None or not isinstance(base, np.ndarray):
                base = np.zeros((480, 640, 3), dtype=np.uint8)
            self._img_label.set_image(base)
        key = self._create_shape_key
        if key == "circle":
            d = {"cx": int(node.params.get("cx", 320)),
                 "cy": int(node.params.get("cy", 240)),
                 "r":  int(node.params.get("r", 100))}
        elif key == "polygon":
            pts = node.params.get("_poly_pts")
            if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                d = {"pts": [(int(px), int(py)) for px, py in pts]}
            else:
                # Chưa vẽ tay → preview đa giác đều từ (cx,cy,r,n_sides).
                cx0 = int(node.params.get("cx", 320))
                cy0 = int(node.params.get("cy", 240))
                r0 = max(1, int(node.params.get("r", 100)))
                n0 = max(3, int(node.params.get("n_sides", 6)))
                ang = np.arange(n0) * 2 * np.pi / n0 - np.pi / 2
                xs = cx0 + r0 * np.cos(ang); ys = cy0 + r0 * np.sin(ang)
                d = {"pts": [(int(x), int(y)) for x, y in zip(xs, ys)]}
        else:  # rect / ellipse / trapezoid-bbox
            d = {"x": int(node.params.get("x", 100)),
                 "y": int(node.params.get("y", 100)),
                 "w": int(node.params.get("w", 200)),
                 "h": int(node.params.get("h", 150))}
        if d:
            self._img_label.set_shape_data(key, d)

    def _on_create_shape_drawn(self, shape_type: str, data: dict):
        """Vẽ xong shape → ghi geometry vào params + rerun (vẽ hình cố định)."""
        node = self._node
        d = dict(data or {})
        updates = {}
        if shape_type in ("rect", "ellipse"):
            for k in ("x", "y", "w", "h"):
                if k in d:
                    updates[k] = int(d[k])
        elif shape_type == "circle":
            for k in ("cx", "cy", "r"):
                if k in d:
                    updates[k] = int(d[k])
        elif shape_type == "polygon":
            pts = d.get("pts") or []
            node.params["_poly_pts"] = [[int(px), int(py)] for px, py in pts]
        for nm, v in updates.items():
            node.params[nm] = v
            pr = getattr(self, "_param_rows", {}).get(nm)
            if pr is not None and hasattr(pr._editor, "setValue"):
                pr._editor.blockSignals(True)
                pr._editor.setValue(v)
                pr._editor.blockSignals(False)
        if getattr(self, "_auto_run_cb", None) and self._auto_run_cb.isChecked():
            self._auto_run_timer.start()
        else:
            self._on_run()

    def _color_seg_shape_key(self, label: str) -> Optional[str]:
        return {"Rectangle": "rect", "Circle": "circle",
                "Ellipse": "ellipse", "Polygon": "polygon"}.get(label)

    def _on_color_segment_shape(self, shape_type: str, data: dict):
        """User vẽ xong shape → lưu vào params và chạy lại nếu Auto Run."""
        node = self._node
        node.params["_roi_shape_type"] = shape_type
        node.params["_roi_shape_data"] = dict(data) if data else None
        # Sync drawn w, h sang roi_w / roi_h params + spinbox → khi sau này
        # user nối port x/y, kích thước ROI sẽ giữ nguyên thay vì về default.
        if data and "w" in data and "h" in data:
            for nm, v in (("roi_w", int(data["w"])), ("roi_h", int(data["h"]))):
                node.params[nm] = v
                pr = getattr(self, "_param_rows", {}).get(nm)
                if pr:
                    ed = pr._editor
                    if hasattr(ed, "setValue"):
                        ed.blockSignals(True); ed.setValue(v); ed.blockSignals(False)
        if getattr(self, "_auto_run_cb", None) and self._auto_run_cb.isChecked():
            self._auto_run_timer.start()
        else:
            # Auto Run tắt → vẫn rerun để user thấy ngay vùng ROI mới
            self._on_run()

    def _on_param(self, node_id, name, value):
        if not (self._graph and node_id in self._graph.nodes):
            return
        node = self._graph.nodes[node_id]
        node.params[name] = value

        # YOLO Detect: model_path đổi → refresh info panel (lazy đọc metadata)
        if (node.tool.tool_id == "yolo_detect" and name == "model_path"
                and hasattr(self, "_yolo_info_lbl")):
            self._refresh_yolo_info()

        # Create Shape: chỉnh toạ độ tay → đồng bộ overlay (handle) ngay, kể cả
        # khi Auto Run tắt (ảnh output cập nhật khi Run).
        if (getattr(self, "_create_shape_key", None)
                and name in ("x", "y", "w", "h", "cx", "cy", "r",
                             "top_ratio", "n_sides")):
            QTimer.singleShot(0, self._restore_create_shape_overlay)

        # color_segment: đổi Color Space qua params panel → đồng bộ
        # combobox trên header (giữ 2 widget cùng giá trị).
        if (node.tool.tool_id == "color_segment" and name == "color_space"
                and self._color_mode_cb is not None):
            idx = self._color_mode_cb.findText(str(value))
            if idx >= 0 and self._color_mode_cb.currentIndex() != idx:
                self._color_mode_cb.blockSignals(True)
                self._color_mode_cb.setCurrentIndex(idx)
                self._color_mode_cb.blockSignals(False)

        # color_segment: đổi ROI Shape dropdown → đổi shape vẽ trên ảnh
        if (node.tool.tool_id == "color_segment" and name == "roi_shape"
                and hasattr(self, "_img_label")):
            key = self._color_seg_shape_key(value)
            if key is None:
                # Full Image → xoá shape đã lưu, khoá vẽ
                node.params.pop("_roi_shape_type", None)
                node.params.pop("_roi_shape_data", None)
                self._img_label.clear_shapes()
                self._img_label.set_roi_locked(True)
            else:
                self._img_label.set_roi_locked(False)
                self._img_label.set_shape_mode(key)
                # Nếu shape đã lưu cùng loại thì khôi phục
                saved = node.params.get("_roi_shape_data")
                saved_t = node.params.get("_roi_shape_type")
                if saved and saved_t == key:
                    self._img_label.set_shape_data(key, dict(saved))
                else:
                    node.params.pop("_roi_shape_data", None)
                    node.params.pop("_roi_shape_type", None)

        # Nếu có param khác phụ thuộc tên này → rebuild Params tab để cập nhật.
        # Defer qua event loop: rebuild ngay trong slot sẽ xoá chính widget
        # đang phát signal (vd QComboBox source_mode toggle Folder ↔ File
        # nhiều lần) → next toggle hit C++ deleted object → crash app.
        if any(getattr(p, "visible_if", None) and name in p.visible_if
                for p in node.tool.params):
            QTimer.singleShot(0, self._rebuild_params_tab)
        # Auto Run: debounce qua timer để không spam khi kéo slider.
        if getattr(self, "_auto_run_cb", None) and self._auto_run_cb.isChecked():
            self._auto_run_timer.start()

    def _on_auto_run_toggled(self, checked: bool):
        self._node.params["_auto_run"] = bool(checked)
        if checked:
            self._auto_run_timer.start()

    def _on_label_dragged(self, dx: int, dy: int):
        """User kéo text label trên ảnh → sync vào params label_dx/dy
        + spinbox/slider ở Params panel (block signal tránh feedback loop).
        Luôn debounce qua timer — drag liên tục có thể spam rất nhiều."""
        node = self._node
        node.params["label_dx"] = int(dx)
        node.params["label_dy"] = int(dy)
        for name, val in (("label_dx", dx), ("label_dy", dy)):
            pr = getattr(self, "_param_rows", {}).get(name)
            if pr:
                ed = pr._editor
                if hasattr(ed, "setValue"):
                    ed.blockSignals(True)
                    ed.setValue(int(val))
                    ed.blockSignals(False)
        self._auto_run_timer.start()

    def _rebuild_params_tab(self):
        """Rebuild Params tab — dùng khi visible_if của param khác đổi.
        An toàn khi dialog đã đóng / scroll widget đã bị Qt xoá."""
        try:
            scroll = getattr(self, "_params_scroll", None)
            if scroll is None:
                return
            scroll.setWidget(self._build_params_widget())
        except RuntimeError:
            # Underlying C++ widget đã bị xoá — ignore.
            pass

    # ════════════════════════════════════════════════════════════════
    #  Run
    # ════════════════════════════════════════════════════════════════
    def _on_run(self):
        node = self._node
        # Build inputs: defaults + upstream outputs
        inputs = {p.name: p.default for p in node.tool.inputs}
        for conn in self._graph.connections:
            if conn.dst_id == node.node_id:
                src = self._graph.nodes.get(conn.src_id)
                if src and conn.src_port in src.outputs:
                    inputs[conn.dst_port] = src.outputs[conn.src_port]

        try:
            out = node.tool.process_fn(inputs, node.params)
            node.outputs  = out or {}
            node.status   = "pass"
            if "pass" in node.outputs:
                node.status = "pass" if node.outputs["pass"] else "fail"
            node.error_msg = ""
        except Exception as e:
            node.outputs  = {}
            node.status   = "error"
            node.error_msg = str(e)

        self.refresh_outputs()
        self.run_requested.emit(node.node_id)

    # ════════════════════════════════════════════════════════════════
    #  Refresh
    # ════════════════════════════════════════════════════════════════
    def refresh_title(self):
        """Cập nhật tiêu đề cửa sổ + header khi node bị đổi tên."""
        node = self._node
        self.setWindowTitle(f"{node.tool.icon}  {node.name}  —  Detail")
        if hasattr(self, "_title_lbl"):
            self._title_lbl.setText(node.name)

    def refresh_outputs(self):
        node = self._node
        sc = {"pass":"#39ff14","fail":"#ff3860","error":"#ff3860",
              "idle":"#64748b","running":"#ffd700"}.get(node.status, "#64748b")
        self._status_lbl.setText(f"Status: {node.status.upper()}")
        self._status_lbl.setStyleSheet(
            f"color:{sc}; font-size:12px; font-weight:700; padding:2px;")

        if node.error_msg:
            self._img_info.setText(f"Error: {node.error_msg}")
            self._img_info.setStyleSheet(
                "color:#ff3860; font-size:10px; font-family:'Courier New'; padding:2px;")

        # Scalar outputs
        for name, lbl in self._out_labels.items():
            val = node.outputs.get(name)
            if val is None:
                lbl.setText("—"); lbl.setStyleSheet("color:#1e2d45; font-size:11px;")
            elif isinstance(val, bool):
                lbl.setText("✔ TRUE" if val else "✖ FALSE")
                lbl.setStyleSheet(
                    f"color:{'#39ff14' if val else '#ff3860'}; font-size:11px; font-weight:700;")
            elif isinstance(val, float):
                lbl.setText(f"{val:.5f}")
                lbl.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            elif isinstance(val, int):
                lbl.setText(str(val))
                lbl.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700;")
            else:
                sval = str(val)
                # Lỗi/text dài (vd OCR install hints) wrap qua nhiều dòng;
                # giữ tooltip = full text để hover xem được khi vẫn cắt.
                shown = sval if len(sval) <= 240 else sval[:240] + "…"
                lbl.setText(shown)
                lbl.setToolTip(sval)
                # Đỏ-cam cho lỗi (text bắt đầu bằng '['), trắng cho text thường.
                color = "#ff9966" if sval.startswith("[") else "#e2e8f0"
                lbl.setStyleSheet(f"color:{color}; font-size:11px;")

        # Color picker
        if node.tool.tool_id == "color_picker" and node.outputs:
            r2 = node.outputs.get("r", 0); g2 = node.outputs.get("g", 0)
            b2 = node.outputs.get("b", 0); H2 = node.outputs.get("h", 0)
            S2 = node.outputs.get("s", 0); V2 = node.outputs.get("v", 0)
            self._pixel_bar.setText(
                f"  ({node.params.get('pick_x',0)}, {node.params.get('pick_y',0)})  "
                f"  RGB {r2},{g2},{b2}  #{r2:02X}{g2:02X}{b2:02X}  "
                f"  HSV {H2},{S2},{V2}")
            self._pixel_bar.setStyleSheet(
                f"color:rgb({r2},{g2},{b2}); font-size:11px; font-family:'Courier New';"
                f"background:#111827; border-radius:4px; padding:3px 10px;"
                f"border:1px solid rgb({r2},{g2},{b2});")
            self._pixel_bar.show()

        # Image — ưu tiên `_display_image` (overlay; key private, không phải
        # port → không lộ ra UI / không truyền cho downstream). Fall back
        # `image` clean nếu tool không xuất visualization.
        img = node.outputs.get("_display_image")
        if img is None or not isinstance(img, np.ndarray):
            img = node.outputs.get("image")
        if img is not None and isinstance(img, np.ndarray):
            h2, w2 = img.shape[:2]
            if not node.error_msg:
                self._img_info.setText(
                    f"{w2}×{h2} px  |  {img.dtype}  |  {node.status.upper()}")
                self._img_info.setStyleSheet(
                    f"color:{sc}; font-size:10px; font-family:'Courier New'; padding:2px;")
            if self._img_label is not None:
                self._img_label.set_image(img)

            # Cache ảnh nguồn (BGR, không overlay) cho hover color picker.
            if node.tool.tool_id in ("color_segment", "color_picker", "color_match"):
                src = self._get_input_image()
                if src is None or not isinstance(src, np.ndarray):
                    # Tool chưa kết nối input → dùng tạm ảnh đang hiển thị.
                    src = img
                if isinstance(src, np.ndarray) and src.shape[:2] == img.shape[:2]:
                    self._source_bgr = src
                else:
                    self._source_bgr = img

            # color_segment: nếu port x/y/w/h kết nối → đồng bộ shape
            # overlay trên label với effective shape (đã apply port overrides
            # trong proc_color_segment). Tránh hiển thị shape vẽ tay cũ kỹ.
            if (node.tool.tool_id == "color_segment"
                    and getattr(self, "_shape_port_connected", False)):
                eff_t = node.outputs.get("_effective_roi_shape_type")
                eff_d = node.outputs.get("_effective_roi_shape_data")
                if eff_t and eff_d:
                    self._img_label.set_shape_data(eff_t, dict(eff_d))
                    self._img_label.set_roi_locked(True)

            # Label drag (Blob…): nạp rect + anchor để hit-test sau mỗi run.
            label_rects = node.outputs.get("_label_rects") or []
            anchors = node.outputs.get("_label_centroids") or []
            cur_off = (int(node.params.get("label_dx", 0)),
                       int(node.params.get("label_dy", 0)))
            self._img_label.set_label_rects(label_rects, anchors, cur_off)

            # crop_roi: hiển thị rect + sync spinbox từ output thực
            if node.tool.tool_id == "crop_roi":
                ox = int(node.outputs.get("x", node.params.get("x", 0)))
                oy = int(node.outputs.get("y", node.params.get("y", 0)))
                ow = int(node.outputs.get("w", node.params.get("crop_w", 0)))
                oh = int(node.outputs.get("h", node.params.get("crop_h", 0)))
                # Cập nhật rect đang hiển thị (nguồn truth = output thực,
                # bất kể port có connect hay không — fix "spinbox hiện 0").
                if ow > 0 and oh > 0:
                    self._img_label.set_rect_from_params(ox, oy, ow, oh)
                # Sync spinbox X/Y/Width/Height — block signals chống loop.
                spinbox_map = (("x", ox), ("y", oy),
                                ("crop_w", ow), ("crop_h", oh))
                for name, val in spinbox_map:
                    pr = getattr(self, "_param_rows", {}).get(name)
                    if pr:
                        ed = pr._editor
                        if hasattr(ed, "setValue"):
                            ed.blockSignals(True)
                            ed.setValue(int(val))
                            ed.blockSignals(False)

            # dist_point: sync spinbox X1/Y1/X2/Y2 từ port-fed values
            # (UI hiển thị giá trị thật đã dùng, không phải param mặc định 0).
            elif node.tool.tool_id == "dist_point":
                live = {"x1": node.outputs.get("x1"),
                        "y1": node.outputs.get("y1"),
                        "x2": node.outputs.get("x2"),
                        "y2": node.outputs.get("y2")}
                for name, val in live.items():
                    if val is None: continue
                    pr = getattr(self, "_param_rows", {}).get(name)
                    if pr:
                        ed = pr._editor
                        if hasattr(ed, "setValue"):
                            ed.blockSignals(True)
                            ed.setValue(int(val))
                            ed.blockSignals(False)

            # patmax/patfind: hiển thị output image từ engine
            elif node.tool.tool_id in ("patmax", "patmax_align", "patfind"):
                pass   # PatMaxDialog tự quản lý display

        elif not node.error_msg:
            if self._img_label is not None:
                self._img_label.set_image(None)
