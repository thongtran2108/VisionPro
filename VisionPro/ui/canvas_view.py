"""
ui/canvas_view.py — v5 T style
Thêm view_in_viewer signal, giữ nguyên connection logic đã fix.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import math

from PySide6.QtWidgets import (QGraphicsScene, QGraphicsView, QGraphicsItem,
                                QGraphicsPathItem, QMenu, QStyle)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QPainterPath,
                            QPainterPathStroker, QTransform, QKeyEvent,
                            QWheelEvent, QDragEnterEvent, QDropEvent,
                            QDragMoveEvent)

from core.flow_graph import FlowGraph, Connection
from core.tool_registry import TOOL_BY_ID
from ui.node_item import NodeItem, NodeSignals, PortItem


class ConnectionItem(QGraphicsPathItem):
    Z_NORMAL = 5
    Z_HOVER  = 9     # trên các dây khác (vẫn dưới node z=10) để dễ truy vết

    def __init__(self, conn: Connection, src_pos: QPointF, dst_pos: QPointF):
        super().__init__()
        self.conn    = conn
        self.src_pos = src_pos
        self.dst_pos = dst_pos
        self._hovered = False
        self.setZValue(self.Z_NORMAL)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._redraw()

    def _redraw(self):
        s, d = self.src_pos, self.dst_pos
        dx   = abs(d.x() - s.x())
        ctrl = max(60.0, dx * 0.5)
        path = QPainterPath(s)
        path.cubicTo(QPointF(s.x() + ctrl, s.y()),
                     QPointF(d.x() - ctrl, d.y()), d)
        self.setPath(path)
        if self._hovered:
            pen = QPen(QColor(255, 235, 80), 3.5)   # vàng sáng, dày — nổi bật
        elif self.isSelected():
            pen = QPen(QColor(255, 100, 50), 2.5)
        else:
            pen = QPen(QColor(0, 212, 255), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)

    def paint(self, painter, option, widget=None):
        # Bỏ khung chọn nét đứt mặc định của Qt — đã có pen cam (selected) +
        # node sáng viền làm tín hiệu, khung đứt chỉ gây rối.
        option.state &= ~QStyle.State_Selected
        super().paint(painter, option, widget)

    def update_positions(self, src: QPointF, dst: QPointF):
        self.src_pos, self.dst_pos = src, dst
        self._redraw()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._redraw()
            self._notify_highlight()
        return super().itemChange(change, value)

    def _notify_highlight(self):
        """Báo scene tính lại node nào cần sáng (endpoint của dây đang
        chọn/hover)."""
        scene = self.scene()
        if scene is not None and hasattr(scene, "_refresh_node_highlights"):
            scene._refresh_node_highlights()

    def endpoint_label(self) -> str:
        """'NodeNguồn · port  →  NodeĐích · port' — để tooltip/nhận diện dây."""
        scene = self.scene()
        graph = getattr(scene, "graph", None)
        if graph is None:
            return f"{self.conn.src_port} → {self.conn.dst_port}"
        sn = graph.nodes.get(self.conn.src_id)
        dn = graph.nodes.get(self.conn.dst_id)
        sname = sn.tool.name if sn else "?"
        dname = dn.tool.name if dn else "?"
        return (f"{sname} · {self.conn.src_port}  →  "
                f"{dname} · {self.conn.dst_port}")

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.setZValue(self.Z_HOVER)
        self.setToolTip(self.endpoint_label())
        self._redraw()
        self._notify_highlight()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.setZValue(self.Z_NORMAL)
        self._redraw()
        self._notify_highlight()
        super().hoverLeaveEvent(event)

    def shape(self) -> QPainterPath:
        """Hitbox rộng hơn nét vẽ (~14px) để dễ trúng dây khi right-click /
        click chọn — path gốc rất mảnh nên itemAt() hay trượt."""
        stroker = QPainterPathStroker()
        stroker.setWidth(14.0)
        return stroker.createStroke(self.path())

    def contextMenuEvent(self, event):
        """Right-click trên dây → chọn dây + menu Unlink / Re-link 2 đầu tới
        một node đang có sẵn trên canvas."""
        scene = self.scene()
        if scene is None:
            return
        # Right-click = chọn dây (highlight). Honor "chuột phải để chọn".
        scene.clearSelection()
        self.setSelected(True)

        menu = QMenu()
        menu.setStyleSheet(
            "QMenu{background:#0d1220;color:#e2e8f0;border:1px solid #1e2d45;font-size:12px;}"
            "QMenu::item:selected{background:#1a2236;color:#00d4ff;}"
            "QMenu::separator{height:1px;background:#1e2d45;}")

        graph = scene.graph
        src_node = graph.nodes.get(self.conn.src_id)
        dst_node = graph.nodes.get(self.conn.dst_id)
        src_name = src_node.tool.name if src_node else "?"
        dst_name = dst_node.tool.name if dst_node else "?"
        hdr = menu.addAction(
            f"🔗  {src_name}.{self.conn.src_port} → "
            f"{dst_name}.{self.conn.dst_port}")
        hdr.setEnabled(False)
        menu.addSeparator()
        act_unlink = menu.addAction("✂  Unlink (xoá kết nối)")
        menu.addSeparator()

        # Re-link: chọn node + port khác để nối lại đầu nguồn / đầu đích.
        src_menu = menu.addMenu("⬅  Link nguồn tới node…")
        dst_menu = menu.addMenu("➡  Link đích tới node…")
        relink: dict = {}   # action -> ("src"|"dst", node_id, port_name)
        for nid, ni in scene._node_items.items():
            node = ni.node
            label = f"{node.tool.icon}  {node.tool.name}"
            # Nguồn mới = output port của node khác (không phải node đích).
            if nid != self.conn.dst_id:
                out_names = ni._output_port_names()
                if out_names:
                    sub = src_menu.addMenu(label)
                    for pn in out_names:
                        relink[sub.addAction(pn)] = ("src", nid, pn)
            # Đích mới = input port của node khác (không phải node nguồn).
            if nid != self.conn.src_id:
                in_ports = ni._visible_input_ports()
                if in_ports:
                    sub = dst_menu.addMenu(label)
                    for p in in_ports:
                        relink[sub.addAction(p.name)] = ("dst", nid, p.name)
        src_menu.setEnabled(not src_menu.isEmpty())
        dst_menu.setEnabled(not dst_menu.isEmpty())

        chosen = menu.exec(event.screenPos())
        event.accept()
        if chosen is None:
            return
        if chosen == act_unlink:
            scene.unlink_connection(self.conn.conn_id)
        elif chosen in relink:
            kind, nid, pn = relink[chosen]
            if kind == "src":
                scene.relink_connection(self.conn.conn_id,
                                        new_src_id=nid, new_src_port=pn)
            else:
                scene.relink_connection(self.conn.conn_id,
                                        new_dst_id=nid, new_dst_port=pn)


class TempCurve(QGraphicsPathItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        pen = QPen(QColor(255, 220, 50), 2, Qt.DashLine)
        pen.setCapStyle(Qt.RoundCap)
        self.setPen(pen)

    def update(self, src: QPointF, dst: QPointF):
        dx   = abs(dst.x() - src.x())
        ctrl = max(60.0, dx * 0.5)
        path = QPainterPath(src)
        path.cubicTo(QPointF(src.x() + ctrl, src.y()),
                     QPointF(dst.x() - ctrl, dst.y()), dst)
        self.setPath(path)


class AOIScene(QGraphicsScene):
    node_selected    = Signal(str)
    node_deselected  = Signal()
    connection_added = Signal()
    graph_changed    = Signal()
    run_single       = Signal(str)
    view_in_viewer   = Signal(str)   # NEW

    def __init__(self, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.setBackgroundBrush(QBrush(QColor(8, 12, 22)))

        self._node_items: Dict[str, NodeItem] = {}
        self._conn_items: Dict[str, ConnectionItem] = {}

        self._signals = NodeSignals()
        self._signals.selected.connect(self.node_selected)
        self._signals.delete_req.connect(self._delete_node)
        self._signals.open_props.connect(self.node_selected)
        self._signals.moved.connect(self._on_node_moved)
        self._signals.ports_changed.connect(self._on_ports_changed)

        self._drag_port: Optional[PortItem] = None
        self._temp_curve: Optional[TempCurve] = None
        self._dragging_conn = False

        self._load_graph()

    def _load_graph(self):
        for node in self.graph.nodes.values():
            self._add_node_item(node)
        for conn in self.graph.connections:
            self._add_conn_item(conn)

    def _add_node_item(self, node) -> NodeItem:
        item = NodeItem(node, self._signals)
        self.addItem(item)
        self._node_items[node.node_id] = item
        return item

    def _add_conn_item(self, conn: Connection) -> Optional[ConnectionItem]:
        si = self._node_items.get(conn.src_id)
        di = self._node_items.get(conn.dst_id)
        if not si or not di:
            return None
        sp = si.get_port_scene_pos(conn.src_port, True)
        dp = di.get_port_scene_pos(conn.dst_port, False)
        if sp is None or dp is None:
            return None
        ci = ConnectionItem(conn, sp, dp)
        self.addItem(ci)
        self._conn_items[conn.conn_id] = ci
        return ci

    def add_node(self, tool_id: str, pos: QPointF) -> NodeItem:
        node = self.graph.add_node(tool_id, pos.x(), pos.y())
        item = self._add_node_item(node)
        self.graph_changed.emit()
        return item

    def _delete_node(self, node_id: str):
        for conn in self.graph.connections_for_node(node_id):
            ci = self._conn_items.pop(conn.conn_id, None)
            if ci:
                self.removeItem(ci)
        ni = self._node_items.pop(node_id, None)
        if ni:
            self.removeItem(ni)
        self.graph.remove_node(node_id)
        self.node_deselected.emit()
        self.graph_changed.emit()

    def _on_ports_changed(self, node_id: str):
        """Khi node thêm/xoá/ẩn output terminal — vẽ lại các connection liên
        quan. Port bị ẩn (_hidden_outputs) → sp=None → hide edge tạm thời để
        không có line dangling ở vị trí cũ; show lại khi port unhide."""
        for conn in self.graph.connections_for_node(node_id):
            ci = self._conn_items.get(conn.conn_id)
            if not ci:
                continue
            si = self._node_items.get(conn.src_id)
            di = self._node_items.get(conn.dst_id)
            if not si or not di:
                continue
            sp = si.get_port_scene_pos(conn.src_port, True)
            dp = di.get_port_scene_pos(conn.dst_port, False)
            if sp and dp:
                ci.update_positions(sp, dp)
                ci.setVisible(True)
            else:
                ci.setVisible(False)

    def _on_node_moved(self, node_id: str, x: float, y: float):
        for conn in self.graph.connections_for_node(node_id):
            ci = self._conn_items.get(conn.conn_id)
            if not ci:
                continue
            si = self._node_items.get(conn.src_id)
            di = self._node_items.get(conn.dst_id)
            if si and di:
                sp = si.get_port_scene_pos(conn.src_port, True)
                dp = di.get_port_scene_pos(conn.dst_port, False)
                if sp and dp:
                    ci.update_positions(sp, dp)

    def _port_at(self, scene_pos: QPointF) -> Optional[PortItem]:
        """Tìm PortItem GẦN NHẤT trong bán kính SNAP (px scene).
        QGraphicsScene.items() trả về theo z-order, không theo khoảng cách —
        nên với các port xếp dọc khít nhau (Acquire Image: image/width/height/...)
        có thể trả về nhầm port khác → fix: chọn theo Euclidean distance.
        """
        SNAP = 16.0
        best: Optional[PortItem] = None
        best_d2 = SNAP * SNAP + 1
        for item in self.items(QRectF(scene_pos.x()-SNAP, scene_pos.y()-SNAP,
                                       SNAP*2, SNAP*2)):
            if not isinstance(item, PortItem):
                continue
            c = item.scene_center()
            dx = c.x() - scene_pos.x(); dy = c.y() - scene_pos.y()
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = item
        return best

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            port = self._port_at(event.scenePos())
            if port is not None:
                self._drag_port    = port
                self._dragging_conn = True
                self._temp_curve   = TempCurve()
                self.addItem(self._temp_curve)
                self._temp_curve.update(port.scene_center(), event.scenePos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_conn and self._temp_curve and self._drag_port:
            self._temp_curve.update(self._drag_port.scene_center(), event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_conn and event.button() == Qt.LeftButton:
            self._finish_connection(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _finish_connection(self, release_pos: QPointF):
        if self._temp_curve:
            self.removeItem(self._temp_curve)
            self._temp_curve = None
        src_port = self._drag_port
        self._drag_port = None
        self._dragging_conn = False
        if src_port is None:
            return
        dst_port = self._port_at(release_pos)
        if dst_port is None or dst_port is src_port:
            return
        if src_port.is_output and not dst_port.is_output:
            out_port, in_port = src_port, dst_port
        elif not src_port.is_output and dst_port.is_output:
            out_port, in_port = dst_port, src_port
        else:
            return
        if out_port.node_item.node.node_id == in_port.node_item.node.node_id:
            return
        conn = self.graph.add_connection(
            out_port.node_item.node.node_id, out_port.port_name,
            in_port.node_item.node.node_id, in_port.port_name)
        if conn:
            self._add_conn_item(conn)
            self.connection_added.emit()
            self.graph_changed.emit()

    def _refresh_node_highlights(self):
        """Node + nub port là endpoint của 1 dây đang được chọn hoặc hover →
        sáng lên. Tính tập trung để node/port nối nhiều dây vẫn đúng (sáng nếu
        có ÍT NHẤT 1 dây active)."""
        active_nodes = set()
        active_ports = set()   # (node_id, port_name, is_output)
        for ci in self._conn_items.values():
            if ci.isSelected() or getattr(ci, "_hovered", False):
                active_nodes.add(ci.conn.src_id)
                active_nodes.add(ci.conn.dst_id)
                active_ports.add((ci.conn.src_id, ci.conn.src_port, True))
                active_ports.add((ci.conn.dst_id, ci.conn.dst_port, False))
        for nid, ni in self._node_items.items():
            ni.set_highlight(nid in active_nodes)
            ni.apply_port_highlights(active_ports)

    def unlink_connection(self, conn_id: str):
        """Xoá 1 connection (qua right-click dây → Unlink)."""
        ci = self._conn_items.pop(conn_id, None)
        if ci is not None and ci.scene() is self:
            self.removeItem(ci)
        self.graph.remove_connection(conn_id)
        self._refresh_node_highlights()
        self.graph_changed.emit()

    def relink_connection(self, conn_id: str,
                          new_src_id: Optional[str] = None,
                          new_src_port: Optional[str] = None,
                          new_dst_id: Optional[str] = None,
                          new_dst_port: Optional[str] = None):
        """Nối lại 1 đầu của connection sang node/port khác. Giữ nguyên đầu
        không đổi. Trả về Connection mới (hoặc None nếu invalid)."""
        conn = next((c for c in self.graph.connections
                     if c.conn_id == conn_id), None)
        if conn is None:
            return None
        src_id   = new_src_id   if new_src_id   is not None else conn.src_id
        src_port = new_src_port if new_src_port is not None else conn.src_port
        dst_id   = new_dst_id   if new_dst_id   is not None else conn.dst_id
        dst_port = new_dst_port if new_dst_port is not None else conn.dst_port
        if src_id == dst_id:
            return None   # không cho self-loop

        # Xoá dây cũ trước khi tạo dây mới.
        old = self._conn_items.pop(conn_id, None)
        if old is not None and old.scene() is self:
            self.removeItem(old)
        self.graph.remove_connection(conn_id)

        new_conn = self.graph.add_connection(src_id, src_port, dst_id, dst_port)
        # add_connection dedup theo dst port → connection cũ chiếm port đó (nếu
        # có) bị xoá khỏi graph; dọn ConnectionItem mồ côi tương ứng.
        live = {c.conn_id for c in self.graph.connections}
        for cid in [c for c in self._conn_items if c not in live]:
            it = self._conn_items.pop(cid, None)
            if it is not None and it.scene() is self:
                self.removeItem(it)
        if new_conn:
            self._add_conn_item(new_conn)
            self.connection_added.emit()
            self.graph_changed.emit()
        self._refresh_node_highlights()
        return new_conn

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.selectedItems():
                if isinstance(item, NodeItem):
                    self._delete_node(item.node.node_id)
                elif isinstance(item, ConnectionItem):
                    self._conn_items.pop(item.conn.conn_id, None)
                    self.graph.remove_connection(item.conn.conn_id)
                    self.removeItem(item)
                    self.graph_changed.emit()
            self._refresh_node_highlights()
        super().keyPressEvent(event)

    def run_single_node(self, node_id: str):
        self.run_single.emit(node_id)

    def view_in_viewer(self, node_id: str):
        self.view_in_viewer.emit(node_id)

    def refresh_all_nodes(self):
        for ni in self._node_items.values():
            ni.update()

    def refresh_node(self, node_id: str):
        ni = self._node_items.get(node_id)
        if ni:
            ni.update()

    def refresh_connections(self):
        for conn in self.graph.connections:
            ci = self._conn_items.get(conn.conn_id)
            if not ci:
                continue
            si = self._node_items.get(conn.src_id)
            di = self._node_items.get(conn.dst_id)
            if si and di:
                sp = si.get_port_scene_pos(conn.src_port, True)
                dp = di.get_port_scene_pos(conn.dst_port, False)
                if sp and dp:
                    ci.update_positions(sp, dp)


class AOICanvas(QGraphicsView):
    def __init__(self, graph: FlowGraph, parent=None):
        super().__init__(parent)
        self._scene = AOIScene(graph)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setAcceptDrops(True)
        self._zoom = 1.0
        self._panning = False
        self._pan_start = None

    @property
    def aoi_scene(self) -> AOIScene:
        return self._scene

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self._zoom = max(0.1, min(4.0, self._zoom * factor))
        self.setTransform(QTransform.fromScale(self._zoom, self._zoom))

    def zoom_fit(self):
        items = [i for i in self._scene.items() if isinstance(i, NodeItem)]
        if not items:
            return
        rect = items[0].mapRectToScene(items[0].boundingRect())
        for it in items[1:]:
            rect = rect.united(it.mapRectToScene(it.boundingRect()))
        self.fitInView(rect.adjusted(-60, -60, 60, 60), Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()

    def zoom_reset(self):
        self._zoom = 1.0
        self.setTransform(QTransform.fromScale(1.0, 1.0))

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning   = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        # Chuột phải = CHỌN (node tự show context menu, dây show Unlink/Re-link
        # qua contextMenuEvent). Không còn pan bằng chuột phải — pan dùng chuột
        # giữa, hoặc chuột trái trên vùng trống.
        # Left-click trên vùng trống (không trúng node nào) → pan canvas
        # thay vì rubber-band select, giống Figma/Photoshop.
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item is None:
                self._panning   = True
                self._pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            d = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - d.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (
                Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        """Vùng trống: chặn menu rỗng (không có gì để chọn). Trên node/dây:
        forward để item tự show context menu (Properties / Unlink / Re-link…)."""
        if self.itemAt(event.pos()) is None:
            event.accept()
            return
        super().contextMenuEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        tool_id = event.mimeData().text()
        if tool_id in TOOL_BY_ID:
            pos = self.mapToScene(event.position().toPoint()) - QPointF(95, 45)
            self._scene.add_node(tool_id, pos)
            event.acceptProposedAction()
