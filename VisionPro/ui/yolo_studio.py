"""
ui/yolo_studio.py
YOLO Studio — Tích hợp vào AOI Vision Pro
Gồm 3 tab chính:
  1. LABEL  — vẽ polygon/bbox label trên ảnh, lưu YOLO format
  2. DATASET — quản lý dataset, split train/val, xem thống kê
  3. TRAIN   — cấu hình & chạy YOLO training, xem log realtime
"""
from __future__ import annotations
import os, sys, shutil, yaml, csv, math, time, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QFrame, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QScrollArea, QGroupBox, QInputDialog,
    QMessageBox, QSizePolicy, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem,
    QApplication, QMenu, QSlider, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtCore import (Qt, Signal, QTimer, QThread, QObject,
                             QPointF, QRectF, QSize)
from PySide6.QtGui import (QPainter, QPen, QBrush, QColor, QPixmap,
                            QImage, QFont, QPolygonF, QMouseEvent,
                            QKeyEvent, QCursor, QIcon)


# ═══════════════════════════════════════════════════════════════
#  COLORS cho các class
# ═══════════════════════════════════════════════════════════════
CLASS_COLORS = [
    QColor(255, 80,  80),   # red
    QColor(80,  200, 80),   # green
    QColor(80,  120, 255),  # blue
    QColor(255, 200, 0),    # yellow
    QColor(0,   220, 220),  # cyan
    QColor(220, 0,   220),  # magenta
    QColor(255, 140, 0),    # orange
    QColor(140, 80,  255),  # purple
    QColor(0,   255, 140),  # mint
    QColor(255, 60,  160),  # pink
]

def get_class_color(idx: int) -> QColor:
    return CLASS_COLORS[idx % len(CLASS_COLORS)]


# ═══════════════════════════════════════════════════════════════
#  Label Canvas — vẽ polygon / bbox
# ═══════════════════════════════════════════════════════════════
class LabelCanvas(QGraphicsView):
    annotation_added   = Signal(dict)   # {type, class_id, points/bbox}
    annotation_removed = Signal(int)    # index
    status_changed     = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(QBrush(QColor(15, 20, 35)))

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._img_w = 1; self._img_h = 1
        self._disp_w = 1; self._disp_h = 1

        # Drawing state
        self.mode         = "bbox"    # "bbox" | "polygon"
        self.current_class_id = 0
        self.num_classes  = 1
        self._drawing     = False
        self._poly_pts: List[QPointF] = []
        self._temp_items  = []
        self._bbox_start: Optional[QPointF] = None
        self._temp_rect_item = None

        # Pan
        self._panning     = False
        self._pan_start   = None

        # All annotations [{type, class_id, pts_norm}]
        self.annotations: List[dict] = []
        self._ann_items:  List = []

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ── Image ──────────────────────────────────────────────────
    def load_image(self, path_or_array):
        self._scene.clear()
        self._temp_items.clear()
        self._ann_items.clear()
        self.annotations.clear()
        self._poly_pts.clear()
        self._drawing = False

        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array)
        else:
            img = path_or_array

        if img is None:
            return
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self._img_h, self._img_w = img.shape[:2]
        h, w, ch = img.shape
        qimg = QImage(img.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self._pixmap_item = QGraphicsPixmapItem(pix)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(0, 0, w, h)
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._disp_w = w; self._disp_h = h

    def load_labels(self, label_path: str):
        """Load YOLO label file và vẽ lên canvas."""
        if not os.path.exists(label_path):
            return
        with open(label_path) as f:
            lines = f.read().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if len(coords) == 4:
                # bbox: cx cy w h (normalized)
                cx,cy,bw,bh = coords
                x1 = (cx - bw/2) * self._img_w
                y1 = (cy - bh/2) * self._img_h
                x2 = (cx + bw/2) * self._img_w
                y2 = (cy + bh/2) * self._img_h
                pts_norm = [(cx-bw/2,cy-bh/2),(cx+bw/2,cy-bh/2),
                            (cx+bw/2,cy+bh/2),(cx-bw/2,cy+bh/2)]
                ann = {"type":"bbox","class_id":cls_id,"pts_norm":pts_norm}
            else:
                # polygon
                pts_norm = [(coords[i],coords[i+1]) for i in range(0,len(coords),2)]
                ann = {"type":"polygon","class_id":cls_id,"pts_norm":pts_norm}
            self.annotations.append(ann)
            self._draw_annotation(ann)

    def _scene_to_norm(self, pt: QPointF) -> Tuple[float,float]:
        return (pt.x()/self._img_w, pt.y()/self._img_h)

    def _norm_to_scene(self, nx,ny) -> QPointF:
        return QPointF(nx * self._img_w, ny * self._img_h)

    def _draw_annotation(self, ann: dict):
        col = get_class_color(ann["class_id"])
        pen = QPen(col, 2)
        pts_scene = [self._norm_to_scene(nx,ny) for nx,ny in ann["pts_norm"]]
        poly = QPolygonF(pts_scene)
        item = QGraphicsPolygonItem(poly)
        item.setPen(pen)
        item.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(), 40)))
        self._scene.addItem(item)
        self._ann_items.append(item)
        # Class label
        if pts_scene:
            lx = pts_scene[0].x(); ly = pts_scene[0].y()
            txt = self._scene.addText(f"cls{ann['class_id']}")
            txt.setDefaultTextColor(col)
            txt.setPos(lx, ly - 18)
            self._ann_items.append(txt)

    def delete_last_annotation(self):
        if not self.annotations:
            return
        self.annotations.pop()
        # Remove last 2 items (polygon + label)
        for _ in range(min(2, len(self._ann_items))):
            item = self._ann_items.pop()
            self._scene.removeItem(item)

    def clear_all_annotations(self):
        self.annotations.clear()
        for item in self._ann_items:
            self._scene.removeItem(item)
        self._ann_items.clear()
        for item in self._temp_items:
            self._scene.removeItem(item)
        self._temp_items.clear()
        self._poly_pts.clear()
        self._drawing = False

    # ── YOLO export ────────────────────────────────────────────
    def to_yolo_lines(self) -> List[str]:
        lines = []
        for ann in self.annotations:
            cls_id = ann["class_id"]
            pts    = ann["pts_norm"]
            if ann["type"] == "bbox":
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
                bw = max(xs)-min(xs);     bh = max(ys)-min(ys)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
                lines.append(f"{cls_id} {coords}")
        return lines

    # ── Mouse ───────────────────────────────────────────────────
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton or (
                event.button() == Qt.LeftButton and
                event.modifiers() == Qt.ControlModifier):
            self._panning   = True
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            return

        scene_pt = self.mapToScene(event.position().toPoint())

        if event.button() == Qt.LeftButton and self._pixmap_item:
            if self.mode == "bbox":
                self._drawing    = True
                self._bbox_start = scene_pt
            elif self.mode == "polygon":
                self._drawing = True
                # Close polygon if near first point
                if (len(self._poly_pts) >= 3 and
                        (scene_pt - self._poly_pts[0]).manhattanLength() < 15):
                    self._finish_polygon()
                    return
                self._poly_pts.append(scene_pt)
                # Draw dot
                dot = self._scene.addEllipse(
                    scene_pt.x()-3, scene_pt.y()-3, 6, 6,
                    QPen(Qt.yellow,1), QBrush(Qt.yellow))
                self._temp_items.append(dot)
                if len(self._poly_pts) > 1:
                    p1 = self._poly_pts[-2]; p2 = self._poly_pts[-1]
                    line = self._scene.addLine(p1.x(),p1.y(),p2.x(),p2.y(),
                                               QPen(Qt.yellow,1,Qt.DashLine))
                    self._temp_items.append(line)
                self.status_changed.emit(
                    f"Polygon: {len(self._poly_pts)} points | Right-click or close to finish")

        elif event.button() == Qt.RightButton:
            if self.mode == "polygon" and len(self._poly_pts) >= 3:
                self._finish_polygon()
            elif self.mode == "bbox" and self._drawing:
                self._drawing = False
                if self._temp_rect_item:
                    self._scene.removeItem(self._temp_rect_item)
                    self._temp_rect_item = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_start:
            d = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - d.y())
            return

        scene_pt = self.mapToScene(event.position().toPoint())
        nx = scene_pt.x()/self._img_w; ny = scene_pt.y()/self._img_h
        self.status_changed.emit(
            f"Mode: {self.mode.upper()}  |  "
            f"Class: {self.current_class_id}  |  "
            f"Pos: ({scene_pt.x():.0f}, {scene_pt.y():.0f})  "
            f"Norm: ({nx:.3f}, {ny:.3f})")

        if self._drawing and self.mode == "bbox" and self._bbox_start:
            if self._temp_rect_item:
                self._scene.removeItem(self._temp_rect_item)
            col = get_class_color(self.current_class_id)
            rect = QRectF(self._bbox_start, scene_pt).normalized()
            self._temp_rect_item = self._scene.addRect(
                rect, QPen(col, 2, Qt.DashLine),
                QBrush(QColor(col.red(),col.green(),col.blue(),30)))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            return

        if event.button() == Qt.LeftButton and self.mode == "bbox" and self._drawing:
            self._drawing = False
            if self._temp_rect_item:
                self._scene.removeItem(self._temp_rect_item)
                self._temp_rect_item = None
            scene_pt = self.mapToScene(event.position().toPoint())
            if self._bbox_start:
                self._finish_bbox(self._bbox_start, scene_pt)
                self._bbox_start = None

    def _finish_bbox(self, p1: QPointF, p2: QPointF):
        rect = QRectF(p1, p2).normalized()
        if rect.width() < 5 or rect.height() < 5:
            return
        x1n = rect.left()/self._img_w;   y1n = rect.top()/self._img_h
        x2n = rect.right()/self._img_w;  y2n = rect.bottom()/self._img_h
        x1n = max(0,min(1,x1n)); y1n = max(0,min(1,y1n))
        x2n = max(0,min(1,x2n)); y2n = max(0,min(1,y2n))
        pts_norm = [(x1n,y1n),(x2n,y1n),(x2n,y2n),(x1n,y2n)]
        ann = {"type":"bbox","class_id":self.current_class_id,"pts_norm":pts_norm}
        self.annotations.append(ann)
        self._draw_annotation(ann)
        self.annotation_added.emit(ann)
        self.status_changed.emit(
            f"BBox added — class {self.current_class_id}  "
            f"({x1n:.3f},{y1n:.3f}) → ({x2n:.3f},{y2n:.3f})")

    def _finish_polygon(self):
        if len(self._poly_pts) < 3:
            return
        for item in self._temp_items:
            self._scene.removeItem(item)
        self._temp_items.clear()
        pts_norm = [self._scene_to_norm(p) for p in self._poly_pts]
        # Clamp
        pts_norm = [(max(0,min(1,x)),max(0,min(1,y))) for x,y in pts_norm]
        ann = {"type":"polygon","class_id":self.current_class_id,"pts_norm":pts_norm}
        self.annotations.append(ann)
        self._draw_annotation(ann)
        self.annotation_added.emit(ann)
        self._poly_pts.clear()
        self._drawing = False
        self.status_changed.emit(
            f"Polygon added — class {self.current_class_id}  "
            f"{len(pts_norm)} points")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.delete_last_annotation()
        elif event.key() == Qt.Key_Escape:
            # Cancel current drawing
            for item in self._temp_items:
                self._scene.removeItem(item)
            self._temp_items.clear()
            self._poly_pts.clear()
            self._drawing = False
            if self._temp_rect_item:
                self._scene.removeItem(self._temp_rect_item)
                self._temp_rect_item = None
        super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════
#  YOLO Training Worker
# ═══════════════════════════════════════════════════════════════
class YoloTrainWorker(QObject):
    log_line   = Signal(str)
    progress   = Signal(int)          # epoch progress %
    finished   = Signal(str)          # save_dir
    error      = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config  = config
        self._stop   = False

    def run(self):
        try:
            from ultralytics import YOLO
            cfg = self.config
            self.log_line.emit(f"Loading model: {cfg['model']}")
            model = YOLO(cfg["model"])

            self.log_line.emit("Starting training...")
            self.log_line.emit(
                f"  epochs={cfg['epochs']}  imgsz={cfg['imgsz']}  "
                f"batch={cfg['batch']}  lr={cfg['lr0']}")

            result = model.train(
                data      = cfg["data_yaml"],
                epochs    = cfg["epochs"],
                imgsz     = cfg["imgsz"],
                batch     = cfg["batch"],
                lr0       = cfg["lr0"],
                workers   = cfg.get("workers", 4),
                name      = cfg.get("name","yolo_run"),
                plots     = True,
                augment   = cfg.get("augment", True),
                seed      = 42,
                verbose   = True,
            )
            save_dir = str(result.save_dir)
            self.log_line.emit(f"✅ Training done! Saved: {save_dir}")
            self._parse_csv_log(save_dir)
            self.finished.emit(save_dir)
        except Exception as e:
            self.error.emit(str(e))

    def _parse_csv_log(self, save_dir: str):
        csv_path = os.path.join(save_dir, "results.csv")
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                return
            headers = [h.strip() for h in rows[0]]
            self.log_line.emit("── Training Results ──")
            self.log_line.emit(" | ".join(headers))
            for row in rows[1:]:
                self.log_line.emit(" | ".join(v.strip() for v in row))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  MAIN YOLO STUDIO DIALOG
# ═══════════════════════════════════════════════════════════════
class YoloStudioDialog(QDialog):
    """
    Cửa sổ YOLO Studio — Label + Dataset + Train
    Standalone, không phụ thuộc ui_form hay các file ngoài.
    """
    model_trained = Signal(str)   # path to best.pt

    def __init__(self, parent=None, initial_image=None):
        super().__init__(parent)
        self.setWindowTitle("🤖  YOLO Studio  —  Label · Dataset · Train")
        self.setMinimumSize(1200, 780)
        self.resize(1400, 860)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)
        self.setModal(False)
        self.setStyleSheet("""
            QDialog { background:#0a0e1a; color:#e2e8f0; }
            QGroupBox { border:1px solid #1e2d45; border-radius:6px;
                        margin-top:8px; padding-top:8px;
                        color:#64748b; font-size:11px; font-weight:700; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QLabel { color:#e2e8f0; }
            QListWidget { background:#0d1220; color:#e2e8f0;
                          border:1px solid #1e2d45; border-radius:4px;
                          font-size:11px; }
            QListWidget::item:selected { background:#1a2236; color:#00d4ff; }
            QTextEdit { background:#050810; color:#94a3b8;
                        border:1px solid #1e2d45; border-radius:4px;
                        font-family:'Courier New'; font-size:11px; }
        """)

        # State
        self._project_dir:   Optional[str] = None
        self._classes:       List[str]     = []
        self._image_list:    List[str]     = []
        self._current_idx:   int           = -1
        self._train_thread:  Optional[QThread] = None
        self._train_worker:  Optional[YoloTrainWorker] = None
        self._initial_image  = initial_image

        self._build_ui()
        self._connect_signals()

        if initial_image is not None:
            QTimer.singleShot(200, lambda: self._load_single_image(initial_image))

    # ════════════════════════════════════════════════════════════
    #  Build UI
    # ════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        # Header
        hdr = QWidget(); hdr.setFixedHeight(50)
        hdr.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #0d1a2e,stop:1 #0a0e1a);"
            "border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(16,0,16,0)
        logo = QLabel("🤖  YOLO Studio")
        logo.setStyleSheet(
            "color:#00d4ff; font-size:15px; font-weight:700; "
            "letter-spacing:1px; background:transparent;")
        hl.addWidget(logo)

        # Project info
        self._proj_lbl = QLabel("No project open")
        self._proj_lbl.setStyleSheet(
            "color:#64748b; font-size:11px; font-family:'Courier New'; background:transparent;")
        hl.addWidget(self._proj_lbl)
        hl.addStretch()

        # Open/New project buttons
        for txt, tip, slot in [
            ("📁 New Project", "Tạo project mới", self._new_project),
            ("📂 Open Project","Mở project có sẵn", self._open_project),
        ]:
            b = QPushButton(txt); b.setToolTip(tip)
            b.setFixedHeight(30)
            b.setStyleSheet(
                "QPushButton{background:#111827;border:1px solid #1e2d45;"
                "border-radius:4px;color:#94a3b8;font-size:11px;padding:0 10px;}"
                "QPushButton:hover{background:#00d4ff22;color:#00d4ff;"
                "border-color:#00d4ff;}")
            b.clicked.connect(slot)
            hl.addWidget(b)
        root.addWidget(hdr)

        # Pre-create labels used inside tab builders (must exist before addTab)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(
            "color:#64748b; font-size:10px; font-family:'Courier New';")
        self._ann_count_lbl = QLabel("Annotations: 0")
        self._ann_count_lbl.setStyleSheet(
            "color:#1e2d45; font-size:10px; font-family:'Courier New';")

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0d1220;}
            QTabBar::tab{background:#060a14;color:#64748b;padding:9px 18px;
                         border:none;font-size:12px;font-weight:700;
                         border-right:1px solid #1e2d45;}
            QTabBar::tab:selected{color:#00d4ff;background:#0d1220;
                                  border-bottom:2px solid #00d4ff;}
            QTabBar::tab:hover{color:#e2e8f0;}
        """)
        self._tabs.addTab(self._build_label_tab(),   "✏  Label")
        self._tabs.addTab(self._build_dataset_tab(), "📦  Dataset")
        self._tabs.addTab(self._build_train_tab(),   "🚀  Train")
        root.addWidget(self._tabs, 1)

        # Status bar
        sb = QWidget(); sb.setFixedHeight(26)
        sb.setStyleSheet("background:#060a14; border-top:1px solid #1e2d45;")
        sl = QHBoxLayout(sb); sl.setContentsMargins(10,0,10,0)
        # Re-use pre-created labels
        sl.addWidget(self._status_lbl, 1)
        sl.addWidget(self._ann_count_lbl)
        root.addWidget(sb)

    # ── LABEL TAB ──────────────────────────────────────────────
    def _build_label_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Left toolbar
        left = QWidget(); left.setFixedWidth(220)
        left.setStyleSheet("background:#0d1220; border-right:1px solid #1e2d45;")
        ll = QVBoxLayout(left); ll.setContentsMargins(8,8,8,8); ll.setSpacing(8)

        # Mode
        mode_grp = QGroupBox("Draw Mode")
        mg = QVBoxLayout(mode_grp); mg.setContentsMargins(6,10,6,6); mg.setSpacing(4)
        self._btn_bbox = self._mode_btn("⬜  Bounding Box", True)
        self._btn_poly = self._mode_btn("🔷  Polygon",     False)
        self._btn_bbox.clicked.connect(lambda: self._set_mode("bbox"))
        self._btn_poly.clicked.connect(lambda: self._set_mode("polygon"))
        mg.addWidget(self._btn_bbox); mg.addWidget(self._btn_poly)

        hint = QLabel("BBox: drag  |  Poly: click pts\nRight-click / close = finish\nCtrl+Z = undo  |  Esc = cancel")
        hint.setStyleSheet("color:#1e2d45; font-size:9px; padding:2px;")
        hint.setWordWrap(True)
        mg.addWidget(hint)
        ll.addWidget(mode_grp)

        # Classes
        cls_grp = QGroupBox("Classes")
        cg = QVBoxLayout(cls_grp); cg.setContentsMargins(6,10,6,6); cg.setSpacing(4)
        self._class_list = QListWidget(); self._class_list.setFixedHeight(140)
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        cg.addWidget(self._class_list)

        btn_row = QHBoxLayout()
        btn_add_cls = self._small_btn("+ Add")
        btn_del_cls = self._small_btn("− Del")
        btn_add_cls.clicked.connect(self._add_class)
        btn_del_cls.clicked.connect(self._del_class)
        btn_row.addWidget(btn_add_cls); btn_row.addWidget(btn_del_cls)
        cg.addLayout(btn_row)
        ll.addWidget(cls_grp)

        # Image list
        img_grp = QGroupBox("Images")
        ig = QVBoxLayout(img_grp); ig.setContentsMargins(6,10,6,6); ig.setSpacing(4)
        self._img_list = QListWidget()
        self._img_list.currentRowChanged.connect(self._on_image_selected)
        ig.addWidget(self._img_list)

        btn_row2 = QHBoxLayout()
        btn_add_img  = self._small_btn("+ Add")
        btn_add_dir  = self._small_btn("+ Folder")
        btn_del_img  = self._small_btn("− Del")
        btn_add_img.clicked.connect(self._add_images)
        btn_add_dir.clicked.connect(self._add_image_folder)
        btn_del_img.clicked.connect(self._remove_image)
        btn_row2.addWidget(btn_add_img)
        btn_row2.addWidget(btn_add_dir)
        btn_row2.addWidget(btn_del_img)
        ig.addLayout(btn_row2)

        # Nav
        nav = QHBoxLayout()
        self._btn_prev = self._small_btn("◀ Prev")
        self._btn_next = self._small_btn("Next ▶")
        self._img_idx_lbl = QLabel("0/0")
        self._img_idx_lbl.setStyleSheet("color:#64748b; font-size:10px;")
        self._img_idx_lbl.setAlignment(Qt.AlignCenter)
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next.clicked.connect(self._next_image)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._img_idx_lbl,1)
        nav.addWidget(self._btn_next)
        ig.addLayout(nav)
        ll.addWidget(img_grp, 1)

        # Save button
        self._btn_save_label = QPushButton("💾  Save Labels")
        self._btn_save_label.setFixedHeight(34)
        self._btn_save_label.setStyleSheet(
            "QPushButton{background:#0f3460;border:1px solid #00d4ff;"
            "border-radius:5px;color:#00d4ff;font-size:12px;font-weight:700;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        self._btn_save_label.clicked.connect(self._save_current_labels)
        ll.addWidget(self._btn_save_label)

        lay.addWidget(left)

        # Center — canvas
        canvas_wrap = QWidget()
        cw = QVBoxLayout(canvas_wrap); cw.setContentsMargins(0,0,0,0); cw.setSpacing(0)

        # Canvas toolbar
        ctb = QWidget(); ctb.setFixedHeight(36)
        ctb.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        ctl = QHBoxLayout(ctb); ctl.setContentsMargins(8,4,8,4); ctl.setSpacing(8)

        btn_clear_ann = self._small_btn("🗑 Clear All")
        btn_undo      = self._small_btn("↩ Undo (Ctrl+Z)")
        btn_zoom_out  = self._small_btn("−")
        btn_zoom_in   = self._small_btn("+")
        btn_zoom_1    = self._small_btn("1:1")
        btn_fit       = self._small_btn("⊡ Fit")
        btn_clear_ann.clicked.connect(lambda: self._canvas.clear_all_annotations())
        btn_undo.clicked.connect(lambda: self._canvas.delete_last_annotation())
        # QGraphicsView zoom: scale(f,f) tích lũy transform; 1:1 = resetTransform.
        btn_zoom_out.clicked.connect(lambda: self._canvas.scale(1/1.25, 1/1.25))
        btn_zoom_in.clicked.connect(lambda: self._canvas.scale(1.25, 1.25))
        btn_zoom_1.clicked.connect(lambda: self._canvas.resetTransform())
        btn_fit.clicked.connect(lambda: self._canvas.fitInView(
            self._canvas._pixmap_item, Qt.KeepAspectRatio)
            if self._canvas._pixmap_item else None)
        for b in (btn_clear_ann, btn_undo,
                  btn_zoom_out, btn_zoom_in, btn_zoom_1, btn_fit):
            ctl.addWidget(b)
        ctl.addStretch()

        # Capture from AOI image
        self._btn_capture = self._small_btn("📸 Use AOI Image")
        self._btn_capture.setStyleSheet(
            "QPushButton{background:#1b4332;border:1px solid #39ff14;"
            "border-radius:3px;color:#39ff14;font-size:11px;padding:0 8px;}"
            "QPushButton:hover{background:#39ff14;color:#000;}")
        self._btn_capture.clicked.connect(self._use_aoi_image)
        ctl.addWidget(self._btn_capture)
        cw.addWidget(ctb)

        self._canvas = LabelCanvas()
        self._canvas.status_changed.connect(self._status_lbl.setText)
        self._canvas.annotation_added.connect(self._on_annotation_added)
        cw.addWidget(self._canvas, 1)
        lay.addWidget(canvas_wrap, 1)

        return w

    # ── DATASET TAB ────────────────────────────────────────────
    def _build_dataset_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)

        # Top: project info
        info_grp = QGroupBox("Project Info")
        ig = QVBoxLayout(info_grp); ig.setContentsMargins(10,12,10,10); ig.setSpacing(6)
        self._ds_info = QLabel("Open a project to see dataset info.")
        self._ds_info.setStyleSheet(
            "color:#94a3b8; font-size:12px; font-family:'Courier New';")
        self._ds_info.setWordWrap(True)
        ig.addWidget(self._ds_info)
        lay.addWidget(info_grp)

        # Split controls
        split_grp = QGroupBox("Train/Val Split")
        sg = QHBoxLayout(split_grp); sg.setContentsMargins(10,12,10,10); sg.setSpacing(12)

        sg.addWidget(QLabel("Train %:"))
        self._split_train = QSpinBox()
        self._split_train.setRange(50,95); self._split_train.setValue(80)
        self._split_train.setStyleSheet(
            "QSpinBox{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:3px;border-radius:3px;}")
        sg.addWidget(self._split_train)

        sg.addWidget(QLabel("Val %:"))
        self._split_val = QLabel("20")
        self._split_val.setStyleSheet("color:#00d4ff; font-size:13px; font-weight:700;")
        sg.addWidget(self._split_val)
        self._split_train.valueChanged.connect(
            lambda v: self._split_val.setText(str(100-v)))

        btn_split = QPushButton("⚡ Auto Split")
        btn_split.setFixedHeight(30)
        btn_split.setStyleSheet(
            "QPushButton{background:#0f3460;border:1px solid #00d4ff;"
            "border-radius:4px;color:#00d4ff;font-size:11px;font-weight:600;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        btn_split.clicked.connect(self._auto_split)
        sg.addWidget(btn_split)
        sg.addStretch()
        lay.addWidget(split_grp)

        # YAML generator
        yaml_grp = QGroupBox("data.yaml Generator")
        yg = QVBoxLayout(yaml_grp); yg.setContentsMargins(10,12,10,10); yg.setSpacing(6)
        self._yaml_preview = QTextEdit()
        self._yaml_preview.setFixedHeight(160)
        self._yaml_preview.setReadOnly(True)
        yg.addWidget(self._yaml_preview)
        btn_gen_yaml = self._action_btn("📄 Generate data.yaml", self._generate_yaml)
        yg.addWidget(btn_gen_yaml)
        lay.addWidget(yaml_grp)

        # Dataset stats table
        stat_grp = QGroupBox("Label Statistics")
        sg2 = QVBoxLayout(stat_grp); sg2.setContentsMargins(10,12,10,10)
        self._stats_table = QTableWidget(0, 3)
        self._stats_table.setHorizontalHeaderLabels(["Class", "Train", "Val"])
        self._stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._stats_table.setStyleSheet("""
            QTableWidget{background:#0a0e1a;color:#e2e8f0;
                         gridline-color:#1e2d45;border:1px solid #1e2d45;font-size:11px;}
            QTableWidget::item:selected{background:#1a2236;color:#00d4ff;}
            QHeaderView::section{background:#0d1220;color:#64748b;
                                  border:1px solid #1e2d45;padding:3px;}
        """)
        self._stats_table.setFixedHeight(180)
        sg2.addWidget(self._stats_table)
        btn_refresh_stats = self._action_btn("🔄 Refresh Stats", self._refresh_stats)
        sg2.addWidget(btn_refresh_stats)
        lay.addWidget(stat_grp)
        lay.addStretch()
        return w

    # ── TRAIN TAB ──────────────────────────────────────────────
    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # Left: config
        left = QWidget(); left.setFixedWidth(300)
        left.setStyleSheet("background:#0d1220; border-right:1px solid #1e2d45;")
        ll = QVBoxLayout(left); ll.setContentsMargins(10,10,10,10); ll.setSpacing(8)

        # Model selection
        model_grp = QGroupBox("Base Model")
        mg = QVBoxLayout(model_grp); mg.setContentsMargins(8,12,8,8); mg.setSpacing(4)
        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt",
            "yolov8n-seg.pt","yolov8s-seg.pt","yolov8m-seg.pt",
            "yolo11n.pt","yolo11s.pt","yolo11m.pt",
        ])
        self._model_combo.setStyleSheet(
            "QComboBox{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:4px 6px;border-radius:3px;}"
            "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
            "selection-background-color:#1a2236;}")
        mg.addWidget(self._model_combo)

        btn_custom_model = self._small_btn("📂 Custom .pt file")
        btn_custom_model.clicked.connect(self._browse_model)
        mg.addWidget(btn_custom_model)
        ll.addWidget(model_grp)

        # Hyperparams
        hyp_grp = QGroupBox("Hyperparameters")
        hg = QVBoxLayout(hyp_grp); hg.setContentsMargins(8,12,8,8); hg.setSpacing(5)
        self._hp_epochs  = self._param_row("Epochs",    hg, "int",   100, 1, 5000)
        self._hp_imgsz   = self._param_row("Image Size",hg, "int",   640, 32, 4096, step=32)
        self._hp_batch   = self._param_row("Batch Size",hg, "int",   16,  1, 512)
        self._hp_lr0     = self._param_row("LR0",       hg, "float", 0.01,1e-5,1.0,4)
        self._hp_workers = self._param_row("Workers",   hg, "int",   4, 0, 32)
        self._hp_augment = QCheckBox("Data Augmentation")
        self._hp_augment.setChecked(True)
        self._hp_augment.setStyleSheet("color:#94a3b8;")
        hg.addWidget(self._hp_augment)
        ll.addWidget(hyp_grp)

        # data.yaml
        data_grp = QGroupBox("data.yaml")
        dg = QVBoxLayout(data_grp); dg.setContentsMargins(8,12,8,8); dg.setSpacing(4)
        self._yaml_path_edit = QLineEdit()
        self._yaml_path_edit.setPlaceholderText("Path to data.yaml...")
        self._yaml_path_edit.setStyleSheet(
            "QLineEdit{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:4px;border-radius:3px;font-size:11px;}")
        dg.addWidget(self._yaml_path_edit)
        btn_browse_yaml = self._small_btn("📂 Browse")
        btn_browse_yaml.clicked.connect(self._browse_yaml)
        dg.addWidget(btn_browse_yaml)
        ll.addWidget(data_grp)

        # Run name
        run_grp = QGroupBox("Run Name")
        rg = QVBoxLayout(run_grp); rg.setContentsMargins(8,12,8,8)
        self._run_name = QLineEdit("yolo_aoi_run")
        self._run_name.setStyleSheet(
            "QLineEdit{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:4px;border-radius:3px;}")
        rg.addWidget(self._run_name)
        ll.addWidget(run_grp)
        ll.addStretch()

        # Train / Stop buttons
        self._btn_start_train = QPushButton("🚀  Start Training")
        self._btn_start_train.setFixedHeight(38)
        self._btn_start_train.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1b4332,stop:1 #2d6a4f);"
            "border:1px solid #39ff14;border-radius:5px;"
            "color:#39ff14;font-size:13px;font-weight:700;}"
            "QPushButton:hover{background:#2d6a4f;}"
            "QPushButton:disabled{background:#1e2d45;color:#1e2d45;border-color:#1e2d45;}")
        self._btn_start_train.clicked.connect(self._start_training)
        ll.addWidget(self._btn_start_train)

        self._btn_stop_train = QPushButton("■  Stop")
        self._btn_stop_train.setFixedHeight(30)
        self._btn_stop_train.setEnabled(False)
        self._btn_stop_train.setStyleSheet(
            "QPushButton{background:#4a0404;border:1px solid #ff3860;"
            "border-radius:4px;color:#ff3860;font-size:12px;font-weight:700;}"
            "QPushButton:hover{background:#ff3860;color:#fff;}"
            "QPushButton:disabled{background:#1e2d45;color:#1e2d45;border-color:#1e2d45;}")
        self._btn_stop_train.clicked.connect(self._stop_training)
        ll.addWidget(self._btn_stop_train)

        lay.addWidget(left)

        # Right: log + progress
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(10,10,10,10); rl.setSpacing(8)

        # Progress
        prog_row = QHBoxLayout()
        self._train_progress = QProgressBar()
        self._train_progress.setRange(0,0)
        self._train_progress.setValue(0)
        self._train_progress.setFixedHeight(10)
        self._train_progress.setStyleSheet(
            "QProgressBar{background:#0a0e1a;border:none;border-radius:5px;}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #39ff14,stop:1 #00d4ff);border-radius:5px;}")
        self._train_progress.hide()
        prog_row.addWidget(self._train_progress,1)

        self._train_status = QLabel("Idle")
        self._train_status.setStyleSheet("color:#64748b; font-size:11px;")
        prog_row.addWidget(self._train_status)
        rl.addLayout(prog_row)

        # Log
        log_lbl = QLabel("TRAINING LOG")
        log_lbl.setStyleSheet(
            "color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:2px;")
        rl.addWidget(log_lbl)

        self._train_log = QTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setStyleSheet(
            "QTextEdit{background:#050810;color:#94a3b8;"
            "border:1px solid #1e2d45;border-radius:4px;"
            "font-family:'Courier New';font-size:11px;}")
        rl.addWidget(self._train_log, 1)

        # Result info
        self._result_grp = QGroupBox("Last Training Result")
        rg2 = QVBoxLayout(self._result_grp); rg2.setContentsMargins(8,12,8,8)
        self._result_info = QLabel("—")
        self._result_info.setStyleSheet(
            "color:#94a3b8; font-size:11px; font-family:'Courier New';")
        self._result_info.setWordWrap(True)
        rg2.addWidget(self._result_info)

        btn_open_result = self._small_btn("📂 Open Result Folder")
        btn_open_result.clicked.connect(self._open_result_folder)
        rg2.addWidget(btn_open_result)

        btn_use_model = self._small_btn("✅ Use best.pt in AOI Pipeline")
        btn_use_model.setStyleSheet(
            "QPushButton{background:#1b4332;border:1px solid #39ff14;"
            "border-radius:3px;color:#39ff14;font-size:11px;padding:2px 8px;}"
            "QPushButton:hover{background:#39ff14;color:#000;}")
        btn_use_model.clicked.connect(self._use_trained_model)
        rg2.addWidget(btn_use_model)
        rl.addWidget(self._result_grp)
        self._result_grp.hide()

        lay.addWidget(right, 1)
        return w

    # ════════════════════════════════════════════════════════════
    #  Signal connections
    # ════════════════════════════════════════════════════════════
    def _connect_signals(self):
        pass   # already connected inline

    # ════════════════════════════════════════════════════════════
    #  PROJECT
    # ════════════════════════════════════════════════════════════
    def _new_project(self):
        name, ok = QInputDialog.getText(self,"New Project","Project name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        base = Path(name + "_dataset")
        if base.exists():
            QMessageBox.warning(self,"Exists",f"'{base}' already exists.")
            return
        try:
            for sub in ["images/train","images/val","labels/train","labels/val"]:
                (base/sub).mkdir(parents=True, exist_ok=True)
            classes_file = Path(name + "_classes.txt")
            classes_file.write_text("")
            self._project_dir = str(base)
            self._load_project(str(base), str(classes_file))
            QMessageBox.information(self,"Created",f"Project created:\n{base}")
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

    def _open_project(self):
        d = QFileDialog.getExistingDirectory(self,"Select dataset folder")
        if not d:
            return
        # Find classes file
        base_name = Path(d).name.replace("_dataset","")
        cls_file = str(Path(d).parent / (base_name + "_classes.txt"))
        if not os.path.exists(cls_file):
            cls_file2, _ = QFileDialog.getOpenFileName(
                self,"Select classes.txt","","Text (*.txt)")
            if cls_file2:
                cls_file = cls_file2
            else:
                cls_file = ""
        self._project_dir = d
        self._load_project(d, cls_file)

    def _load_project(self, dataset_dir: str, classes_file: str):
        self._project_dir = dataset_dir
        self._classes = []
        if classes_file and os.path.exists(classes_file):
            with open(classes_file) as f:
                self._classes = [l.strip() for l in f if l.strip()]

        self._proj_lbl.setText(f"Project: {Path(dataset_dir).name}")
        self._class_list.clear()
        for i, cls in enumerate(self._classes):
            item = QListWidgetItem(f"  {i}: {cls}")
            item.setForeground(get_class_color(i))
            self._class_list.addItem(item)

        # Load images from images/train + images/val
        self._image_list = []
        for split in ("train","val"):
            img_dir = os.path.join(dataset_dir,"images",split)
            if os.path.exists(img_dir):
                for fn in sorted(os.listdir(img_dir)):
                    if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
                        self._image_list.append(
                            os.path.join(img_dir, fn))

        self._img_list.clear()
        for p in self._image_list:
            rel = os.path.relpath(p, dataset_dir)
            self._img_list.addItem(rel)

        if self._image_list:
            self._img_list.setCurrentRow(0)
        self._update_img_nav()
        self._refresh_ds_info()

        # Auto-fill yaml path
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        if os.path.exists(yaml_path):
            self._yaml_path_edit.setText(yaml_path)

    # ════════════════════════════════════════════════════════════
    #  LABELING
    # ════════════════════════════════════════════════════════════
    def _set_mode(self, mode: str):
        self._canvas.mode = mode
        active = "QPushButton{background:#0f3460;border:1px solid #00d4ff;border-radius:4px;color:#00d4ff;font-size:11px;padding:3px 8px;}"
        idle   = "QPushButton{background:#111827;border:1px solid #1e2d45;border-radius:4px;color:#64748b;font-size:11px;padding:3px 8px;}"
        self._btn_bbox.setStyleSheet(active if mode=="bbox" else idle)
        self._btn_poly.setStyleSheet(active if mode=="polygon" else idle)

    def _add_class(self):
        name, ok = QInputDialog.getText(self,"Add Class","Class name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        self._classes.append(name)
        idx = len(self._classes)-1
        item = QListWidgetItem(f"  {idx}: {name}")
        item.setForeground(get_class_color(idx))
        self._class_list.addItem(item)
        self._class_list.setCurrentRow(idx)
        self._canvas.num_classes = len(self._classes)
        self._save_classes()

    def _del_class(self):
        row = self._class_list.currentRow()
        if row < 0: return
        self._classes.pop(row)
        self._class_list.takeItem(row)
        self._save_classes()

    def _on_class_selected(self, row: int):
        if row >= 0:
            self._canvas.current_class_id = row

    def _save_classes(self):
        if not self._project_dir:
            return
        base_name = Path(self._project_dir).name.replace("_dataset","")
        cls_file = os.path.join(Path(self._project_dir).parent,
                                base_name + "_classes.txt")
        with open(cls_file,"w") as f:
            f.write("\n".join(self._classes))

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,"Select Images","",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not paths: return
        if self._project_dir:
            dest_dir = os.path.join(self._project_dir,"images","train")
            os.makedirs(dest_dir, exist_ok=True)
            for p in paths:
                dst = os.path.join(dest_dir, os.path.basename(p))
                if not os.path.exists(dst):
                    shutil.copy2(p, dst)
                if dst not in self._image_list:
                    self._image_list.append(dst)
                    rel = os.path.relpath(dst, self._project_dir)
                    self._img_list.addItem(rel)
        else:
            for p in paths:
                if p not in self._image_list:
                    self._image_list.append(p)
                    self._img_list.addItem(os.path.basename(p))
        self._update_img_nav()

    def _add_image_folder(self):
        d = QFileDialog.getExistingDirectory(self,"Select image folder")
        if not d: return
        added = 0
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff")):
                p = os.path.join(d, fn)
                if self._project_dir:
                    dest_dir = os.path.join(self._project_dir,"images","train")
                    os.makedirs(dest_dir, exist_ok=True)
                    dst = os.path.join(dest_dir, fn)
                    if not os.path.exists(dst):
                        shutil.copy2(p, dst)
                    p = dst
                if p not in self._image_list:
                    self._image_list.append(p)
                    rel = os.path.relpath(p, self._project_dir) if self._project_dir else fn
                    self._img_list.addItem(rel)
                    added += 1
        self._status_lbl.setText(f"Added {added} images.")
        self._update_img_nav()

    def _remove_image(self):
        row = self._img_list.currentRow()
        if row < 0: return
        self._image_list.pop(row)
        self._img_list.takeItem(row)
        self._update_img_nav()

    def _on_image_selected(self, row: int):
        if row < 0 or row >= len(self._image_list):
            return
        self._current_idx = row
        path = self._image_list[row]
        self._canvas.load_image(path)
        # Load existing labels
        label_path = self._get_label_path(path)
        if label_path and os.path.exists(label_path):
            self._canvas.load_labels(label_path)
        self._img_idx_lbl.setText(f"{row+1}/{len(self._image_list)}")

    def _update_img_nav(self):
        n = len(self._image_list)
        self._img_idx_lbl.setText(f"{self._current_idx+1}/{n}" if n else "0/0")

    def _prev_image(self):
        self._save_current_labels()
        if self._current_idx > 0:
            self._img_list.setCurrentRow(self._current_idx - 1)

    def _next_image(self):
        self._save_current_labels()
        if self._current_idx < len(self._image_list) - 1:
            self._img_list.setCurrentRow(self._current_idx + 1)

    def _get_label_path(self, image_path: str) -> Optional[str]:
        """Lấy đường dẫn label .txt tương ứng."""
        p = Path(image_path)
        if self._project_dir:
            # Thay images/ → labels/
            rel = p.relative_to(self._project_dir)
            parts = list(rel.parts)
            if parts[0] == "images":
                parts[0] = "labels"
            label_p = Path(self._project_dir).joinpath(*parts).with_suffix(".txt")
            return str(label_p)
        return str(p.with_suffix(".txt"))

    def _save_current_labels(self):
        if self._current_idx < 0:
            return
        lines = self._canvas.to_yolo_lines()
        path  = self._image_list[self._current_idx]
        label_path = self._get_label_path(path)
        if not label_path:
            return
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path,"w") as f:
            f.write("\n".join(lines))
        n = len(lines)
        self._status_lbl.setText(
            f"Saved {n} annotations → {os.path.basename(label_path)}")
        self._ann_count_lbl.setText(f"Annotations: {n}")

    def _on_annotation_added(self, ann: dict):
        n = len(self._canvas.annotations)
        self._ann_count_lbl.setText(f"Annotations: {n}")

    def _use_aoi_image(self):
        """Lấy ảnh từ node AOI đang chọn (truyền vào qua parent)."""
        if self._initial_image is not None and isinstance(self._initial_image, np.ndarray):
            self._canvas.load_image(self._initial_image)
            self._status_lbl.setText("Loaded image from AOI pipeline.")
        else:
            QMessageBox.information(self,"Info",
                "Mở YOLO Studio từ node AOI có output image để dùng tính năng này.")

    def _load_single_image(self, img):
        self._canvas.load_image(img)

    # ════════════════════════════════════════════════════════════
    #  DATASET TAB ACTIONS
    # ════════════════════════════════════════════════════════════
    def _auto_split(self):
        if not self._project_dir:
            QMessageBox.warning(self,"No Project","Open a project first.")
            return
        train_ratio = self._split_train.value() / 100.0
        img_train = os.path.join(self._project_dir,"images","train")
        img_val   = os.path.join(self._project_dir,"images","val")
        lbl_train = os.path.join(self._project_dir,"labels","train")
        lbl_val   = os.path.join(self._project_dir,"labels","val")
        for d in (img_train,img_val,lbl_train,lbl_val):
            os.makedirs(d, exist_ok=True)

        # Collect all images from train (to re-split)
        all_imgs = [f for f in os.listdir(img_train)
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
        # Also pick up from val
        val_imgs = [f for f in os.listdir(img_val)
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
        for fn in val_imgs:
            src = os.path.join(img_val, fn)
            dst = os.path.join(img_train, fn)
            if not os.path.exists(dst):
                shutil.move(src, dst)
            # Move label too
            lbl_src = os.path.join(lbl_val, Path(fn).stem+".txt")
            lbl_dst = os.path.join(lbl_train, Path(fn).stem+".txt")
            if os.path.exists(lbl_src) and not os.path.exists(lbl_dst):
                shutil.move(lbl_src, lbl_dst)
        all_imgs = [f for f in os.listdir(img_train)
                    if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]

        import random; random.shuffle(all_imgs)
        n_train = int(len(all_imgs) * train_ratio)
        val_set = all_imgs[n_train:]

        for fn in val_set:
            # Move image
            src = os.path.join(img_train, fn)
            dst = os.path.join(img_val,   fn)
            if os.path.exists(src):
                shutil.move(src, dst)
            # Move label
            lbl_src = os.path.join(lbl_train, Path(fn).stem+".txt")
            lbl_dst = os.path.join(lbl_val,   Path(fn).stem+".txt")
            if os.path.exists(lbl_src):
                shutil.move(lbl_src, lbl_dst)

        n_val = len(val_set)
        self._status_lbl.setText(
            f"Split done: {n_train} train / {n_val} val")
        self._refresh_ds_info()
        self._refresh_stats()

    def _generate_yaml(self):
        if not self._project_dir:
            QMessageBox.warning(self,"No Project","Open a project first.")
            return
        abs_path = os.path.abspath(self._project_dir)
        content = {
            "path":  abs_path,
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(self._classes),
            "names": self._classes if self._classes else ["object"],
        }
        yaml_str = yaml.dump(content, default_flow_style=False, allow_unicode=True)
        self._yaml_preview.setText(yaml_str)

        yaml_path = os.path.join(self._project_dir,"data.yaml")
        with open(yaml_path,"w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
        self._yaml_path_edit.setText(yaml_path)
        self._status_lbl.setText(f"data.yaml saved → {yaml_path}")

    def _refresh_ds_info(self):
        if not self._project_dir:
            return
        def count_dir(d):
            if not os.path.exists(d): return 0
            return len([f for f in os.listdir(d)
                        if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
        n_train = count_dir(os.path.join(self._project_dir,"images","train"))
        n_val   = count_dir(os.path.join(self._project_dir,"images","val"))
        self._ds_info.setText(
            f"Project: {Path(self._project_dir).name}\n"
            f"Train images: {n_train}  |  Val images: {n_val}  "
            f"|  Total: {n_train+n_val}\n"
            f"Classes ({len(self._classes)}): {', '.join(self._classes) or '(none)'}")

    def _refresh_stats(self):
        if not self._project_dir: return
        stats: Dict[int,Dict[str,int]] = {}
        for split,lbl_dir_name in [("train","labels/train"),("val","labels/val")]:
            lbl_dir = os.path.join(self._project_dir, lbl_dir_name)
            if not os.path.exists(lbl_dir): continue
            for fn in os.listdir(lbl_dir):
                if not fn.endswith(".txt"): continue
                with open(os.path.join(lbl_dir,fn)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id not in stats: stats[cls_id]={"train":0,"val":0}
                            stats[cls_id][split] += 1
        self._stats_table.setRowCount(0)
        for cls_id in sorted(stats.keys()):
            row = self._stats_table.rowCount()
            self._stats_table.insertRow(row)
            cls_name = self._classes[cls_id] if cls_id < len(self._classes) else str(cls_id)
            for col,val in enumerate([cls_name,
                                       str(stats[cls_id]["train"]),
                                       str(stats[cls_id]["val"])]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    item.setForeground(get_class_color(cls_id))
                self._stats_table.setItem(row, col, item)

    # ════════════════════════════════════════════════════════════
    #  TRAINING
    # ════════════════════════════════════════════════════════════
    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self,"Select .pt model","","YOLO (*.pt)")
        if p:
            self._model_combo.addItem(p)
            self._model_combo.setCurrentText(p)

    def _browse_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self,"Select data.yaml","","YAML (*.yaml *.yml)")
        if p: self._yaml_path_edit.setText(p)

    def _start_training(self):
        yaml_path = self._yaml_path_edit.text().strip()
        if not yaml_path or not os.path.exists(yaml_path):
            QMessageBox.warning(self,"Missing YAML",
                "Chọn data.yaml trước. Vào tab Dataset → Generate data.yaml.")
            return

        config = {
            "model":     self._model_combo.currentText(),
            "data_yaml": yaml_path,
            "epochs":    self._hp_epochs.value(),
            "imgsz":     self._hp_imgsz.value(),
            "batch":     self._hp_batch.value(),
            "lr0":       self._hp_lr0.value(),
            "workers":   self._hp_workers.value(),
            "augment":   self._hp_augment.isChecked(),
            "name":      self._run_name.text().strip() or "yolo_aoi_run",
        }

        self._train_log.clear()
        self._train_log.append(f"Config: {json.dumps(config, indent=2)}")
        self._train_progress.show()
        self._train_progress.setRange(0,0)   # indeterminate
        self._btn_start_train.setEnabled(False)
        self._btn_stop_train.setEnabled(True)
        self._train_status.setText("Training...")
        self._train_status.setStyleSheet("color:#ffd700; font-size:11px;")
        self._result_grp.hide()

        self._train_worker = YoloTrainWorker(config)
        self._train_thread = QThread()
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.log_line.connect(self._on_train_log)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.finished.connect(self._train_thread.quit)
        self._train_worker.error.connect(self._train_thread.quit)
        self._train_thread.start()
        self._last_save_dir = None

    def _stop_training(self):
        if self._train_thread and self._train_thread.isRunning():
            self._train_thread.requestInterruption()
            self._train_thread.quit()
            self._train_thread.wait(3000)
        self._finalize_training()
        self._train_status.setText("Stopped")
        self._train_status.setStyleSheet("color:#ff3860; font-size:11px;")

    def _on_train_log(self, line: str):
        self._train_log.append(line)
        self._train_log.verticalScrollBar().setValue(
            self._train_log.verticalScrollBar().maximum())

    def _on_train_done(self, save_dir: str):
        self._last_save_dir = save_dir
        self._finalize_training()
        self._train_status.setText("✅ Done!")
        self._train_status.setStyleSheet("color:#39ff14; font-size:11px;")
        best_pt = os.path.join(save_dir,"weights","best.pt")
        self._result_info.setText(
            f"Save dir: {save_dir}\n"
            f"best.pt: {'✔ exists' if os.path.exists(best_pt) else '✖ not found'}")
        self._result_grp.show()
        QMessageBox.information(self,"Training Complete",
            f"Training finished!\nSaved to:\n{save_dir}")

    def _on_train_error(self, msg: str):
        self._finalize_training()
        self._train_status.setText("Error!")
        self._train_status.setStyleSheet("color:#ff3860; font-size:11px;")
        self._train_log.append(f"\n❌ ERROR: {msg}")
        QMessageBox.critical(self,"Training Error", msg)

    def _finalize_training(self):
        self._btn_start_train.setEnabled(True)
        self._btn_stop_train.setEnabled(False)
        self._train_progress.hide()

    def _open_result_folder(self):
        d = getattr(self,"_last_save_dir",None)
        if d and os.path.exists(d):
            import subprocess, platform
            if platform.system()=="Windows":
                os.startfile(d)
            elif platform.system()=="Darwin":
                subprocess.Popen(["open",d])
            else:
                subprocess.Popen(["xdg-open",d])

    def _use_trained_model(self):
        d = getattr(self,"_last_save_dir",None)
        if not d:
            return
        best_pt = os.path.join(d,"weights","best.pt")
        if os.path.exists(best_pt):
            self.model_trained.emit(best_pt)
            QMessageBox.information(self,"Model Ready",
                f"Model đã được đăng ký:\n{best_pt}\n\n"
                "Dùng trong YOLO Detect node trong AOI pipeline.")
        else:
            QMessageBox.warning(self,"Not Found","best.pt không tồn tại.")

    # ════════════════════════════════════════════════════════════
    #  Helper builders
    # ════════════════════════════════════════════════════════════
    def _mode_btn(self, text: str, active: bool) -> QPushButton:
        b = QPushButton(text); b.setFixedHeight(30)
        a = ("QPushButton{background:#0f3460;border:1px solid #00d4ff;"
             "border-radius:4px;color:#00d4ff;font-size:11px;padding:3px 8px;}")
        i = ("QPushButton{background:#111827;border:1px solid #1e2d45;"
             "border-radius:4px;color:#64748b;font-size:11px;padding:3px 8px;}"
             "QPushButton:hover{color:#e2e8f0;}")
        b.setStyleSheet(a if active else i)
        return b

    def _small_btn(self, text: str) -> QPushButton:
        b = QPushButton(text); b.setFixedHeight(26)
        b.setStyleSheet(
            "QPushButton{background:#111827;border:1px solid #1e2d45;"
            "border-radius:3px;color:#94a3b8;font-size:11px;padding:0 6px;}"
            "QPushButton:hover{background:#1a2236;color:#e2e8f0;}")
        return b

    def _action_btn(self, text: str, slot) -> QPushButton:
        b = QPushButton(text); b.setFixedHeight(30)
        b.setStyleSheet(
            "QPushButton{background:#0f3460;border:1px solid #00d4ff;"
            "border-radius:4px;color:#00d4ff;font-size:11px;font-weight:600;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}")
        b.clicked.connect(slot)
        return b

    def _param_row(self, label, layout, ptype, default, mn, mx, decimals=2, step=1):
        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        lbl.setMinimumWidth(90)
        if ptype == "int":
            sp = QSpinBox(); sp.setRange(mn,mx); sp.setValue(default)
            sp.setSingleStep(step)
        else:
            sp = QDoubleSpinBox()
            sp.setRange(mn,mx); sp.setValue(default)
            sp.setDecimals(decimals); sp.setSingleStep(0.001)
        sp.setStyleSheet(
            f"{'QSpinBox' if ptype=='int' else 'QDoubleSpinBox'}"
            "{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:2px 4px;border-radius:3px;}")
        rl.addWidget(lbl); rl.addWidget(sp)
        layout.addWidget(row)
        return sp
