"""
ui/node_item.py — T VisionPro style
Hiển thị T tool name, tooltip params, port colors.
"""
from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING

from PySide6.QtWidgets import (QGraphicsItem, QGraphicsEllipseItem, QMenu,
                                QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                                QLabel, QComboBox, QLineEdit,
                                QPushButton, QListWidget, QListWidgetItem,
                                QInputDialog, QMessageBox, QCheckBox,
                                QScrollArea, QWidget)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QFont,
                            QLinearGradient, QPainterPath, QCursor)

from core.flow_graph import NodeInstance
from core.tool_registry import ToolDef, auto_terminal_name

PORT_R        = 7
PORT_D        = PORT_R * 2
NODE_MIN_W    = 190
NODE_HEADER_H = 42
NODE_PORT_ROW = 22
NODE_PADDING  = 8

C_BG       = QColor(13, 18, 30)
C_BORDER   = QColor(30, 45, 69)
C_SEL      = QColor(0, 212, 255)
C_PASS     = QColor(57, 255, 20)
C_FAIL     = QColor(255, 56, 96)
C_WARN     = QColor(255, 215, 0)
C_DIM      = QColor(100, 116, 139)
C_PORT_IN  = QColor(0, 180, 220)
C_PORT_OUT = QColor(255, 140, 50)


class PortItem(QGraphicsEllipseItem):
    """Port hitbox — scene xử lý drag connection."""
    def __init__(self, node_item: "NodeItem", port_name: str,
                 is_output: bool, index: int, parent=None):
        super().__init__(-PORT_R, -PORT_R, PORT_D, PORT_D, parent)
        self.node_item  = node_item
        self.port_name  = port_name
        self.is_output  = is_output
        self.port_index = index
        self._hovered   = False

        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setZValue(20)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self._update_brush()

    def _update_brush(self):
        base = C_PORT_OUT if self.is_output else C_PORT_IN
        if self._hovered:
            self.setBrush(QBrush(base))
            self.setPen(QPen(Qt.white, 2))
        else:
            self.setBrush(QBrush(base.darker(200)))
            self.setPen(QPen(base, 1.5))

    def hoverEnterEvent(self, event):
        self._hovered = True
        self._update_brush()
        self.setScale(1.4)
        if self.is_output:
            val = self.node_item.node.outputs.get(self.port_name)
            self.setToolTip(
                f"<b>OUT • {self.port_name}</b><br>"
                f"<span style='color:#00d4ff'>{self._fmt_value(val)}</span>")
        else:
            self.setToolTip(f"<b>IN • {self.port_name}</b>")
        super().hoverEnterEvent(event)

    @staticmethod
    def _fmt_value(val) -> str:
        """Format giá trị output cho tooltip."""
        if val is None:
            return "(no output yet)"
        if isinstance(val, bool):
            return "✔ TRUE" if val else "✖ FALSE"
        if isinstance(val, float):
            return f"{val:.5g}"
        if isinstance(val, int):
            return str(val)
        if isinstance(val, str):
            return val if len(val) < 80 else val[:77] + "..."
        # numpy array
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                shp = "×".join(str(s) for s in val.shape)
                return f"ndarray {shp} {val.dtype}"
        except Exception:
            pass
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                # List of object-dicts (e.g. per-object PatMax results)
                head = ", ".join(f"{k}={PortItem._fmt_value(v)}"
                                 for k, v in list(val[0].items())[:3])
                return f"[{len(val)} items]<br>#0: {head}"
            return f"[list: {len(val)} items]"
        if isinstance(val, dict):
            return "{" + ", ".join(
                f"{k}={PortItem._fmt_value(v)}" for k, v in list(val.items())[:3]
            ) + "}"
        return str(val)[:60]

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._update_brush()
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class NodeSignals(QObject):
    selected   = Signal(str)
    moved      = Signal(str, float, float)
    delete_req = Signal(str)
    open_props = Signal(str)
    ports_changed = Signal(str)   # node_id — phát khi thay đổi extra terminals


# Mapping field từ "objects" list — dùng cho PatMax/PatFind
PATMAX_FIELDS = ["x", "y", "score", "angle", "scale",
                 "center_x", "center_y", "origin_x", "origin_y"]
# Field cơ bản dành cho mỗi ref point (origin chính + extra refs)
PATMAX_REF_FIELDS = ["x", "y", "angle"]


def _patmax_ref_options(node) -> list:
    """Build list (label, ref_idx, name) cho mỗi ref point của node PatMax.
    ref_idx = 0 → origin chính, ≥1 → extras[ref_idx-1].
    """
    out = [("Origin (main)", 0, None)]
    model = node.params.get("_patmax_model")
    refs = list(getattr(model, "extra_refs", []) or []) if model else []
    for j, ref in enumerate(refs, start=1):
        nm = str(ref.get("name", f"Ref {j}"))
        out.append((nm, j, nm))
    return out


class AddTerminalDialog(QDialog):
    """Dialog thêm terminal output — chọn object index + reference + field.
    Remove được apply NGAY khi click (không chờ Accept) — user expectation
    sau khi xóa thì port biến mất ngay, dù có Cancel."""

    def __init__(self, node, parent=None, on_remove=None):
        super().__init__(parent)
        self._node = node
        self._on_remove = on_remove
        self.setWindowTitle("➕  Add Output Terminal")
        self.setMinimumWidth(360)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)
        self.setStyleSheet("""
            QDialog { background:#0d1220; color:#e2e8f0; }
            QLabel  { color:#94a3b8; font-size:11px; }
            QComboBox, QLineEdit {
                background:#0a0e1a; color:#e2e8f0;
                border:1px solid #1e2d45; border-radius:3px; padding:3px 5px;
            }
            QPushButton {
                background:#1e2d45; color:#e2e8f0; border:none;
                border-radius:4px; padding:6px 14px; font-weight:600;
            }
            QPushButton:hover { background:#00d4ff; color:#000; }
            QListWidget { background:#0a0e1a; color:#e2e8f0;
                          border:1px solid #1e2d45; }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(10)

        self._ref_options = _patmax_ref_options(node)
        # Số object đã detect ở lần chạy gần nhất. Nếu chưa chạy → ít nhất 1
        # để user vẫn add được terminal trước khi search.
        n_obj_detected = len(node.outputs.get("objects") or [])
        self._n_obj = max(1, n_obj_detected)
        info = QLabel(
            f"Tool detect <b>{n_obj_detected}</b> object · "
            f"<b>{len(self._ref_options)}</b> reference point(s).<br>"
            f"Mỗi terminal map 1 (object, reference, field) → 1 output port."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        row_obj = QHBoxLayout()
        row_obj.addWidget(QLabel("Object:"))
        self._cb_obj = QComboBox()
        for i in range(self._n_obj):
            self._cb_obj.addItem(f"Object {i+1}")
        row_obj.addWidget(self._cb_obj, 1)
        lay.addLayout(row_obj)

        row_ref = QHBoxLayout()
        row_ref.addWidget(QLabel("Reference:"))
        self._cb_ref = QComboBox()
        for label, _idx, _nm in self._ref_options:
            self._cb_ref.addItem(label)
        self._cb_ref.currentIndexChanged.connect(self._on_ref_changed)
        row_ref.addWidget(self._cb_ref, 1)
        lay.addLayout(row_ref)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Field:"))
        self._cb_field = QComboBox()
        row2.addWidget(self._cb_field, 1)
        lay.addLayout(row2)
        self._on_ref_changed(0)  # populate field combo lần đầu

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Port name (optional):"))
        self._le_name = QLineEdit()
        self._le_name.setPlaceholderText("auto: <field>  (vd: ref1_x)")
        row3.addWidget(self._le_name, 1)
        lay.addLayout(row3)

        # Hiện list terminals đang có
        existing = node.params.get("_extra_terminals") or []
        if existing:
            lbl_ex = QLabel("Đang có:")
            lay.addWidget(lbl_ex)
            self._list = QListWidget()
            self._list.setFixedHeight(min(120, 22 * len(existing) + 8))
            for t in existing:
                name = auto_terminal_name(t)
                obj_idx = int(t.get("object", 0) or 0)
                self._list.addItem(
                    f"  • {name}  ←  Object {obj_idx + 1} · {t.get('field','x')}")
            lay.addWidget(self._list)
            btn_remove = QPushButton("🗑  Remove selected")
            btn_remove.clicked.connect(self._remove_selected)
            lay.addWidget(btn_remove)
        else:
            self._list = None

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Add Terminal"); btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet(btn_ok.styleSheet() +
                              "QPushButton{background:#0f3460;color:#00d4ff;}"
                              "QPushButton:hover{background:#00d4ff;color:#000;}")
        btn_row.addWidget(btn_cancel); btn_row.addWidget(btn_ok)
        lay.addLayout(btn_row)

    def _remove_selected(self):
        if not self._list:
            return
        row = self._list.currentRow()
        if row < 0:
            return
        # Apply ngay vào node.params — không chờ Accept để Cancel cũng giữ
        # được xóa (user feedback: "remove không mất node" trước khi sửa).
        terminals = list(self._node.params.get("_extra_terminals") or [])
        if 0 <= row < len(terminals):
            terminals.pop(row)
            self._node.params["_extra_terminals"] = terminals
            self._list.takeItem(row)
            if callable(self._on_remove):
                self._on_remove()

    def _on_ref_changed(self, idx: int):
        """Populate field combo theo reference đang chọn.
        Origin → toàn bộ PATMAX_FIELDS; extra ref → chỉ x/y/angle.
        """
        self._cb_field.clear()
        if 0 <= idx < len(self._ref_options):
            ref_idx = self._ref_options[idx][1]
            if ref_idx == 0:
                self._cb_field.addItems(PATMAX_FIELDS)
            else:
                self._cb_field.addItems(PATMAX_REF_FIELDS)
        else:
            self._cb_field.addItems(PATMAX_FIELDS)

    def get_new_terminal(self) -> dict:
        # Map (reference, field) → key trong objects dict:
        #   Origin (ref_idx=0) → field nguyên gốc (x, y, ...)
        #   Ref j (ref_idx=j>0) → "ref{j}_{field}" (vd: ref1_x)
        ref_combo_idx = self._cb_ref.currentIndex()
        field_basic = self._cb_field.currentText()
        ref_idx = 0
        if 0 <= ref_combo_idx < len(self._ref_options):
            ref_idx = self._ref_options[ref_combo_idx][1]
        if ref_idx == 0:
            field_key = field_basic
        else:
            field_key = f"ref{ref_idx}_{field_basic}"
        return {
            "object": int(self._cb_obj.currentIndex()),
            "field":  field_key,
            "name":   self._le_name.text().strip(),
        }



class ManageOutputsDialog(QDialog):
    """Dialog show/hide output ports. Áp dụng ngay khi tick → port xuất hiện/
    biến mất tức thời (callback refresh). Dùng được cho mọi tool — tool có
    nhiều scalar output (blob, find_circle, …) thường cần ẩn bớt cho gọn."""

    def __init__(self, node, parent=None, on_change=None):
        super().__init__(parent)
        self._node = node
        self._on_change = on_change
        self.setWindowTitle("👁  Manage Output Ports")
        self.setMinimumWidth(340)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)
        self.setStyleSheet("""
            QDialog{background:#0d1220;color:#e2e8f0;}
            QLabel{color:#94a3b8;font-size:11px;}
            QPushButton{background:#1e2d45;color:#e2e8f0;border:none;
                        border-radius:4px;padding:6px 14px;font-weight:600;}
            QPushButton:hover{background:#00d4ff;color:#000;}
            QCheckBox{color:#e2e8f0;font-size:12px;padding:3px 6px;}
            QCheckBox::indicator{width:14px;height:14px;}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14); lay.setSpacing(8)

        info = QLabel(
            f"Tick để hiện / bỏ tick để ẩn port. <i>image</i> luôn hiện. "
            f"Áp dụng ngay — đóng dialog khi xong.")
        info.setWordWrap(True)
        lay.addWidget(info)

        hidden = set(self._node.params.get("_hidden_outputs") or [])
        # Build list = tool.outputs + extra terminals (PatMax-style)
        port_names: List[str] = [p.name for p in self._node.tool.outputs]
        for term in (self._node.params.get("_extra_terminals") or []):
            n = auto_terminal_name(term)
            if n not in port_names:
                port_names.append(n)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{border:1px solid #1e2d45;background:#0a0e1a;}")
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(6, 6, 6, 6); inner_lay.setSpacing(2)
        self._checks: List[tuple] = []  # (name, QCheckBox)
        for name in port_names:
            cb = QCheckBox(name)
            is_image = (name == "image")
            cb.setChecked(name not in hidden)
            if is_image:
                cb.setEnabled(False)
                cb.setToolTip("Port 'image' luôn visible — không ẩn được "
                               "(chain port).")
            cb.toggled.connect(
                lambda on, _nm=name: self._toggle(_nm, on))
            inner_lay.addWidget(cb)
            self._checks.append((name, cb))
        inner_lay.addStretch()
        scroll.setWidget(inner)
        scroll.setMinimumHeight(min(360, 30 * max(1, len(port_names)) + 20))
        lay.addWidget(scroll, 1)

        btn_row = QHBoxLayout(); btn_row.addStretch()
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.accept)
        btn_close.setStyleSheet(btn_close.styleSheet() +
                                 "QPushButton{background:#0f3460;color:#00d4ff;}")
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

    def _toggle(self, port_name: str, visible: bool):
        if port_name == "image":
            return  # image không thể ẩn
        hidden = list(self._node.params.get("_hidden_outputs") or [])
        if visible and port_name in hidden:
            hidden.remove(port_name)
        elif not visible and port_name not in hidden:
            hidden.append(port_name)
        self._node.params["_hidden_outputs"] = hidden
        if callable(self._on_change):
            self._on_change()


class NodeItem(QGraphicsItem):
    def __init__(self, node: NodeInstance, signals: NodeSignals):
        super().__init__()
        self.node    = node
        self.signals = signals

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setPos(node.pos_x, node.pos_y)
        self.setZValue(10)

        tool: ToolDef = node.tool
        self._color       = QColor(tool.color)
        self._icon        = tool.icon
        self._name        = tool.name
        self._T_name = tool.T_equiv

        self._in_ports:  List[PortItem] = []
        self._out_ports: List[PortItem] = []
        self._compute_size()
        self._build_ports()

        # Tooltip
        tip = f"<b>{tool.name}</b>"
        if tool.T_equiv:
            tip += f"<br><span style='color:#00d4ff'>{tool.T_equiv}</span>"
        tip += f"<br>{tool.description}"
        self.setToolTip(tip)

    def _output_port_names(self) -> List[str]:
        """Tên các output ports = tool.outputs + extra terminals + extra outputs
        (Script Tool), sau khi lọc bỏ những port nằm trong `_hidden_outputs`
        (user ẩn qua dialog Manage Output Ports). `image` luôn visible."""
        hidden = set(self.node.params.get("_hidden_outputs") or [])
        hidden.discard("image")
        names = [p.name for p in self.node.tool.outputs
                 if p.name not in hidden]
        # PatMax-style structured terminals (object/field tuples)
        for term in (self.node.params.get("_extra_terminals") or []):
            n = auto_terminal_name(term)
            if n not in names and n not in hidden:
                names.append(n)
        # Script-style simple extra port names (added via right-click)
        for n in (self.node.params.get("_extra_outputs") or []):
            if n not in names and n not in hidden:
                names.append(n)
        return names

    def _visible_input_ports(self):
        """Inputs = tool.inputs (static) + `_extra_inputs` (user thêm qua
        right-click). Tool nào có `extra_input_type` mới cho phép thêm.
        Port mới mặc định cùng data_type với extra_input_type của tool."""
        from core.tool_registry import PortDef
        inputs = list(self.node.tool.inputs)
        extra_type = self.node.tool.extra_input_type or ""
        if extra_type:
            for name in (self.node.params.get("_extra_inputs") or []):
                inputs.append(PortDef(name, extra_type, required=False))
        return inputs

    def _compute_size(self):
        tool = self.node.tool
        n_in  = len(self._visible_input_ports())
        n_out = len(self._output_port_names())
        n_ports = max(n_in, n_out, 1)
        self._w = max(NODE_MIN_W, len(tool.name) * 7 + 60)
        self._h = NODE_HEADER_H + NODE_PADDING + n_ports * NODE_PORT_ROW + NODE_PADDING

    def _build_ports(self):
        for i, port in enumerate(self._visible_input_ports()):
            p = PortItem(self, port.name, False, i, self)
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            p.setPos(0, y)
            self._in_ports.append(p)
        for i, name in enumerate(self._output_port_names()):
            p = PortItem(self, name, True, i, self)
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            p.setPos(self._w, y)
            self._out_ports.append(p)

    def refresh_ports(self):
        """Rebuild ports (gọi sau khi đổi extra_terminals)."""
        for p in self._in_ports + self._out_ports:
            try:
                if self.scene():
                    self.scene().removeItem(p)
                else:
                    p.setParentItem(None)
            except RuntimeError:
                pass
        self._in_ports = []
        self._out_ports = []
        self._compute_size()
        self._build_ports()
        self.prepareGeometryChange()
        self.update()
        self.signals.ports_changed.emit(self.node.node_id)

    def boundingRect(self) -> QRectF:
        m = PORT_R + 4
        return QRectF(-m, -m, self._w + m * 2, self._h + m * 2)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        status = self.node.status

        if self.isSelected():
            border_col, border_w = C_SEL, 2.5
        elif status == "pass":
            border_col, border_w = C_PASS, 2.0
        elif status == "fail":
            border_col, border_w = C_FAIL, 2.0
        elif status == "running":
            border_col, border_w = C_WARN, 2.0
        elif status == "error":
            border_col, border_w = C_FAIL, 2.0
        else:
            border_col, border_w = C_BORDER, 1.5

        # Shadow
        shadow = QPainterPath()
        shadow.addRoundedRect(3, 3, self._w, self._h, 8, 8)
        painter.fillPath(shadow, QBrush(QColor(0, 0, 0, 80)))

        # Body
        body = QPainterPath()
        body.addRoundedRect(0, 0, self._w, self._h, 8, 8)
        painter.fillPath(body, QBrush(C_BG))
        painter.setPen(QPen(border_col, border_w))
        painter.drawPath(body)

        # Header gradient
        hdr = QPainterPath()
        hdr.addRoundedRect(0, 0, self._w, NODE_HEADER_H, 8, 8)
        cut = QPainterPath()
        cut.addRect(0, NODE_HEADER_H // 2, self._w, NODE_HEADER_H)
        hdr = hdr.united(cut)
        grad = QLinearGradient(0, 0, self._w, NODE_HEADER_H)
        grad.setColorAt(0, self._color.lighter(140))
        grad.setColorAt(1, self._color.darker(110))
        painter.fillPath(hdr, QBrush(grad))

        # Icon
        painter.setFont(QFont("Segoe UI Emoji", 14))
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(6, 0, 30, NODE_HEADER_H), Qt.AlignCenter, self._icon)

        # Tool name
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(Qt.white))
        painter.drawText(QRectF(36, 2, self._w - 42, NODE_HEADER_H // 2 + 2),
                         Qt.AlignVCenter | Qt.AlignLeft, self._name)

        # T equiv name (small, cyan)
        if self._T_name:
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QPen(QColor(0, 212, 255, 180)))
            painter.drawText(QRectF(36, NODE_HEADER_H // 2, self._w - 42, NODE_HEADER_H // 2),
                             Qt.AlignVCenter | Qt.AlignLeft, self._T_name)

        # Status badge
        if status in ("pass", "fail", "error", "running"):
            colors = {"pass": C_PASS, "fail": C_FAIL, "error": C_FAIL, "running": C_WARN}
            texts  = {"pass": "✔ PASS", "fail": "✖ FAIL", "error": "ERR", "running": "…"}
            badge_col = colors[status]
            badge_txt = texts[status]
            painter.setPen(QPen(badge_col))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(QRectF(0, self._h - 18, self._w - 6, 14),
                             Qt.AlignRight | Qt.AlignVCenter, badge_txt)

        # Port labels
        tool = self.node.tool
        painter.setFont(QFont("Segoe UI", 7))
        in_ports = self._visible_input_ports()
        n_static_in = len(tool.inputs)
        for i, port in enumerate(in_ports):
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            # Extra inputs (user thêm) → vàng để phân biệt với static
            col = C_PORT_IN.lighter(120) if i < n_static_in else QColor(255, 215, 0)
            painter.setPen(QPen(col))
            painter.drawText(QRectF(10, y - 8, self._w // 2 - 14, 16),
                             Qt.AlignLeft | Qt.AlignVCenter, port.name)

        out_names = self._output_port_names()
        n_static = len(tool.outputs)
        for i, name in enumerate(out_names):
            y = NODE_HEADER_H + NODE_PADDING + i * NODE_PORT_ROW + NODE_PORT_ROW // 2
            # Extra terminals → màu khác để phân biệt
            col = C_PORT_OUT.lighter(120) if i < n_static else QColor(255, 215, 0)
            painter.setPen(QPen(col))
            painter.drawText(QRectF(self._w // 2, y - 8, self._w // 2 - 12, 16),
                             Qt.AlignRight | Qt.AlignVCenter, name)

        # Output value previews
        if self.node.outputs:
            painter.setFont(QFont("Courier New", 7))
            painter.setPen(QPen(C_DIM))
            y_off = NODE_HEADER_H + NODE_PADDING + 2
            for key, val in list(self.node.outputs.items())[:3]:
                if isinstance(val, bool):
                    txt = f"{key}:{'✔' if val else '✖'}"
                elif isinstance(val, float):
                    txt = f"{key}:{val:.3g}"
                elif isinstance(val, int):
                    txt = f"{key}:{val}"
                elif isinstance(val, str) and len(val) < 20:
                    txt = f"{key}:{val[:12]}"
                else:
                    continue
                painter.drawText(QRectF(8, y_off + 2, self._w - 16, 12),
                                 Qt.AlignLeft | Qt.AlignVCenter, txt)
                y_off += 12
                if y_off > self._h - 20:
                    break

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.pos_x = self.pos().x()
            self.node.pos_y = self.pos().y()
            self.signals.moved.emit(self.node.node_id, self.pos().x(), self.pos().y())
            if self.scene():
                self.scene().update()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.signals.selected.emit(self.node.node_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.signals.open_props.emit(self.node.node_id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.setStyleSheet(
            "QMenu{background:#0d1220;color:#e2e8f0;border:1px solid #1e2d45;font-size:12px;}"
            "QMenu::item:selected{background:#1a2236;color:#00d4ff;}"
            "QMenu::separator{height:1px;background:#1e2d45;}")
        act_props  = menu.addAction(f"{self.node.tool.icon}  Properties / Detail")
        act_run    = menu.addAction("▶  Run this node")
        menu.addSeparator()
        act_viewer = menu.addAction("👁  View output in Image Viewer")
        # ➕ Add Input — tool có extra_input_type (vd Pass/Fail Judge).
        # Auto-name = letter kế tiếp chưa dùng.
        act_add_input = None
        act_remove_input = None
        if self.node.tool.extra_input_type:
            menu.addSeparator()
            act_add_input = menu.addAction("➕  Add Input Port")
            extras = self.node.params.get("_extra_inputs") or []
            act_remove_input = menu.addAction(
                f"➖  Remove Last Input Port ({extras[-1]})"
                if extras else "➖  Remove Last Input Port")
            act_remove_input.setEnabled(bool(extras))
        # ➕ Add Output — tool có extra_output_type (vd Script Tool). User nhập
        # tên port qua QInputDialog (phải là Python identifier hợp lệ).
        act_add_output = None
        act_remove_output = None
        if self.node.tool.extra_output_type:
            menu.addSeparator()
            act_add_output = menu.addAction("➕  Add Output Port…")
            outs = self.node.params.get("_extra_outputs") or []
            act_remove_output = menu.addAction(
                f"➖  Remove Last Output Port ({outs[-1]})"
                if outs else "➖  Remove Last Output Port")
            act_remove_output.setEnabled(bool(outs))
        # Add Terminal — chỉ tools hỗ trợ "objects" output mới có ý nghĩa
        supports_objects = any(p.name == "objects"
                                for p in self.node.tool.outputs)
        act_add_term = None
        if supports_objects:
            menu.addSeparator()
            act_add_term = menu.addAction("➕  Add / Manage Output Terminals…")
        # Manage Output Ports — show/hide từng output. Hiện cho mọi tool có
        # ≥2 outputs (1 output thì không ai ẩn). image port không bị ẩn được.
        act_manage = None
        if len(self.node.tool.outputs) >= 2:
            if not supports_objects:
                menu.addSeparator()
            act_manage = menu.addAction("👁  Show / Hide Output Ports…")
        menu.addSeparator()
        act_del    = menu.addAction("🗑  Delete")

        chosen = menu.exec(event.screenPos())
        if chosen == act_props:
            self.signals.open_props.emit(self.node.node_id)
        elif chosen == act_del:
            self.signals.delete_req.emit(self.node.node_id)
        elif chosen == act_run:
            if self.scene() and hasattr(self.scene(), "run_single_node"):
                self.scene().run_single_node(self.node.node_id)
        elif chosen == act_viewer:
            if self.scene() and hasattr(self.scene(), "view_in_viewer"):
                self.scene().view_in_viewer(self.node.node_id)
        elif act_add_input is not None and chosen == act_add_input:
            self._add_extra_input()
        elif act_remove_input is not None and chosen == act_remove_input:
            self._remove_last_extra_input()
        elif act_add_output is not None and chosen == act_add_output:
            self._add_extra_output()
        elif act_remove_output is not None and chosen == act_remove_output:
            self._remove_last_extra_output()
        elif act_add_term is not None and chosen == act_add_term:
            self._open_add_terminal_dialog()
        elif act_manage is not None and chosen == act_manage:
            self._open_manage_outputs_dialog()

    def _add_extra_input(self):
        """Append 1 port mới với tên = letter chưa dùng (A-Z, rồi AA-ZZ)."""
        used = {p.name for p in self.node.tool.inputs}
        extras = list(self.node.params.get("_extra_inputs") or [])
        used.update(extras)
        from string import ascii_uppercase
        new_name = None
        for c in ascii_uppercase:
            if c not in used:
                new_name = c; break
        if new_name is None:
            for a in ascii_uppercase:
                for b in ascii_uppercase:
                    if (a + b) not in used:
                        new_name = a + b; break
                if new_name: break
        if new_name is None:
            new_name = f"Extra{len(extras) + 1}"
        extras.append(new_name)
        self.node.params["_extra_inputs"] = extras
        self.refresh_ports()

    def _remove_last_extra_input(self):
        """Xóa extra port cuối + connection nối tới nó."""
        extras = list(self.node.params.get("_extra_inputs") or [])
        if not extras:
            return
        removed = extras.pop()
        self.node.params["_extra_inputs"] = extras
        self._prune_port_connections(removed, is_output=False)
        self.refresh_ports()

    def _add_extra_output(self):
        """Prompt user nhập tên port output. Validate identifier hợp lệ +
        không trùng với output đã có."""
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        existing = {p.name for p in self.node.tool.outputs}
        existing.update(self.node.params.get("_extra_outputs") or [])
        # Suggest tên mặc định: out1, out2, ...
        suggest = "out1"
        n = 1
        while suggest in existing:
            n += 1
            suggest = f"out{n}"
        text, ok = QInputDialog.getText(
            None, "Add Output Port",
            "Tên port (Python identifier hợp lệ):", text=suggest)
        if not ok:
            return
        name = text.strip()
        if not name.isidentifier():
            QMessageBox.warning(
                None, "Invalid Name",
                f"'{name}' không phải Python identifier hợp lệ.\n"
                "Chỉ dùng a-z, A-Z, 0-9, _ và không bắt đầu bằng số.")
            return
        if name in existing:
            QMessageBox.warning(
                None, "Duplicate Name",
                f"Port '{name}' đã tồn tại.")
            return
        outs = list(self.node.params.get("_extra_outputs") or [])
        outs.append(name)
        self.node.params["_extra_outputs"] = outs
        self.refresh_ports()

    def _remove_last_extra_output(self):
        """Xóa output port cuối + connection bắt đầu từ port đó."""
        outs = list(self.node.params.get("_extra_outputs") or [])
        if not outs:
            return
        removed = outs.pop()
        self.node.params["_extra_outputs"] = outs
        self._prune_port_connections(removed, is_output=True)
        self.refresh_ports()

    def _prune_port_connections(self, port_name: str, is_output: bool):
        """Xóa connection liên quan đến port vừa bị remove + ConnectionItem
        tương ứng trong scene."""
        scene = self.scene()
        graph = getattr(scene, "graph", None)
        if graph is None:
            return
        nid = self.node.node_id
        if is_output:
            graph.connections = [
                c for c in graph.connections
                if not (c.src_id == nid and c.src_port == port_name)
            ]
        else:
            graph.connections = [
                c for c in graph.connections
                if not (c.dst_id == nid and c.dst_port == port_name)
            ]
        stale = [cid for cid, _ in list(getattr(scene, "_conn_items", {}).items())
                 if not any(c.conn_id == cid for c in graph.connections)]
        for cid in stale:
            ci = scene._conn_items.pop(cid, None)
            if ci is not None and ci.scene() is scene:
                scene.removeItem(ci)

    def _open_manage_outputs_dialog(self):
        dlg = ManageOutputsDialog(self.node, on_change=self.refresh_ports)
        dlg.exec()

    def _open_add_terminal_dialog(self):
        # on_remove callback → refresh port ngay sau khi click "Remove
        # selected" trong dialog, không phải chờ Accept.
        dlg = AddTerminalDialog(self.node, on_remove=self.refresh_ports)
        if dlg.exec() != QDialog.Accepted:
            return
        terminals = list(self.node.params.get("_extra_terminals") or [])
        # Thêm mới (nếu user nhập)
        new = dlg.get_new_terminal()
        if new and (new.get("field") or new.get("name")):
            # Đảm bảo tên duy nhất. Chỉ ghi đè name khi clash; giữ name rỗng
            # nếu auto-name không trùng để display dùng auto_terminal_name().
            explicit = (new.get("name") or "").strip()
            base = explicit or auto_terminal_name(new)
            existing_names = ([p.name for p in self.node.tool.outputs] +
                                [auto_terminal_name(t) for t in terminals])
            if base in existing_names:
                name = base; n = 1
                while name in existing_names:
                    n += 1; name = f"{base}_{n}"
                new["name"] = name
            else:
                new["name"] = explicit  # giữ rỗng → fallback auto-name
            terminals.append(new)
        self.node.params["_extra_terminals"] = terminals
        self.refresh_ports()

    def get_port_scene_pos(self, port_name: str, is_output: bool) -> Optional[QPointF]:
        ports = self._out_ports if is_output else self._in_ports
        for p in ports:
            if p.port_name == port_name:
                return p.scene_center()
        return None

    def node_width(self) -> float:
        return self._w

    def node_height(self) -> float:
        return self._h
