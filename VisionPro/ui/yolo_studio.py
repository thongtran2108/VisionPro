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
#  Training error → human-friendly message
# ═══════════════════════════════════════════════════════════════
def _explain_train_error(msg: str) -> str:
    """Map raw ultralytics/torch exception string → actionable message.
    Giữ nguyên msg gốc + thêm hướng dẫn fix khi nhận ra pattern quen thuộc."""
    low = msg.lower()

    # numpy 2.x vs packages compiled against numpy 1.x (rất hay gặp sau
    # khi cài/upgrade ultralytics hoặc torch). Triệu chứng điển hình:
    #   "numpy.core.multiarray failed to import"
    #   "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"
    if ("numpy.core.multiarray failed to import" in low
            or "numpy.core.umath failed to import" in low
            or "compiled using numpy 1" in low
            or "_array_api not found" in low):
        return (
            f"{msg}\n\n"
            "── Nguyên nhân ──\n"
            "NumPy không tương thích nhị phân (binary) với torch / opencv / "
            "ultralytics. Thường do NumPy 2.x được cài đè trong khi các "
            "package kia vẫn build với NumPy 1.x (hoặc ngược lại).\n\n"
            "── Cách fix ──\n"
            "Cách 1 — pin NumPy 1.x (nhanh nhất, an toàn):\n"
            "    pip install \"numpy<2\" --force-reinstall\n\n"
            "Cách 2 — upgrade toàn bộ stack lên bản mới (hỗ trợ NumPy 2.x):\n"
            "    pip install -U numpy ultralytics torch torchvision opencv-python\n\n"
            "Sau khi cài, khởi động lại app rồi train lại.")

    # CUDA / GPU không khả dụng (ultralytics tự fall back về CPU, nhưng đôi
    # khi user pin device=0 hoặc torch không build CUDA).
    if ("torch not compiled with cuda" in low
            or "cuda is not available" in low
            or "no cuda gpus are available" in low):
        return (
            f"{msg}\n\n"
            "── Nguyên nhân ──\n"
            "Torch hiện tại không có CUDA, hoặc máy không có GPU NVIDIA.\n\n"
            "── Cách fix ──\n"
            "• Train trên CPU: bỏ qua, ultralytics tự dùng CPU.\n"
            "• Muốn dùng GPU: cài torch bản CUDA tại\n"
            "    https://pytorch.org/get-started/locally/")

    # ultralytics chưa cài.
    if isinstance(msg, str) and (
            "no module named 'ultralytics'" in low
            or "no module named ultralytics" in low):
        return (
            f"{msg}\n\n"
            "── Cách fix ──\n"
            "    pip install ultralytics")

    # Dataset thiếu / sai path trong data.yaml.
    if ("dataset" in low and "not found" in low) or "no labels found" in low:
        return (
            f"{msg}\n\n"
            "── Cách fix ──\n"
            "• Kiểm tra lại data.yaml: các path 'train' / 'val' phải trỏ\n"
            "  đúng thư mục chứa ảnh (.jpg/.png) và label (.txt YOLO format).\n"
            "• Vào tab Dataset → Generate data.yaml để regenerate.")

    return msg


# ═══════════════════════════════════════════════════════════════
#  YOLO Training Worker
# ═══════════════════════════════════════════════════════════════
class YoloTrainWorker(QObject):
    log_line     = Signal(str)
    progress     = Signal(int)          # epoch progress %
    finished     = Signal(str)          # save_dir
    error        = Signal(str)
    metrics      = Signal(dict)         # final {mAP50, mAP50-95, ...}
    save_dir     = Signal(str)          # emitted khi training start (cho live chart)
    epoch_end    = Signal(int, int, dict)  # (epoch, total, metrics_dict)

    def __init__(self, config: dict):
        super().__init__()
        self.config  = config
        self._stop   = False

    def run(self):
        try:
            # Bắt buộc matplotlib dùng backend Agg TRƯỚC khi ultralytics
            # import nó. Trên Windows + PySide6, nếu mpl đã chọn Qt5Agg /
            # TkAgg, ultralytics plot ở cuối train có thể segfault và đóng
            # nguyên app. Agg là pure-buffer, an toàn cho worker thread.
            os.environ.setdefault("MPLBACKEND", "Agg")
            try:
                import matplotlib
                matplotlib.use("Agg", force=True)
            except Exception:
                pass

            from ultralytics import YOLO
            cfg = self.config
            self.log_line.emit(f"Loading model: {cfg['model']}")
            model = YOLO(cfg["model"])

            # Resolve device cho ultralytics.
            #   "" / None → auto (CUDA if available, else CPU)
            #   "cpu"     → CPU
            #   "0"       → GPU 0
            device = cfg.get("device", "")
            dev_label = device if device else "auto"
            self.log_line.emit("Starting training...")
            self.log_line.emit(
                f"  epochs={cfg['epochs']}  imgsz={cfg['imgsz']}  "
                f"batch={cfg['batch']}  lr={cfg['lr0']}  "
                f"patience={cfg.get('patience', 50)}  "
                f"workers={cfg.get('workers', 4)}  device={dev_label}")

            # ── Callbacks: emit save_dir sớm + per-epoch metrics ─────
            #   on_train_start  → biết save_dir để UI polling results.png
            #   on_fit_epoch_end → log + emit metrics live cho UI chart
            total_epochs = cfg["epochs"]
            worker_self = self

            def _cb_train_start(trainer):
                try:
                    worker_self.save_dir.emit(str(trainer.save_dir))
                except Exception:
                    pass

            def _cb_fit_epoch_end(trainer):
                try:
                    epoch = int(getattr(trainer, "epoch", 0)) + 1
                    m = dict(getattr(trainer, "metrics", {}) or {})
                    # Bổ sung training losses (trainer.tloss = tensor 3-elem
                    # cho detection: box/cls/dfl) — convert sang float.
                    tloss = getattr(trainer, "tloss", None)
                    if tloss is not None:
                        try:
                            arr = tloss.detach().cpu().tolist() if hasattr(
                                tloss, "detach") else list(tloss)
                            names = getattr(trainer, "loss_names",
                                            ["box_loss","cls_loss","dfl_loss"])
                            for n, v in zip(names, arr):
                                m[f"train/{n}"] = float(v)
                        except Exception:
                            pass
                    worker_self.epoch_end.emit(epoch, total_epochs, m)
                    # Compact log line cho user theo dõi
                    def _f(k, fmt="{:.4f}"):
                        return fmt.format(m[k]) if k in m else "—"
                    worker_self.log_line.emit(
                        f"Epoch {epoch:>3}/{total_epochs}   "
                        f"mAP50={_f('metrics/mAP50(B)')}   "
                        f"mAP50-95={_f('metrics/mAP50-95(B)')}   "
                        f"P={_f('metrics/precision(B)')}   "
                        f"R={_f('metrics/recall(B)')}   "
                        f"box_loss={_f('train/box_loss')}")
                except Exception as cb_e:
                    worker_self.log_line.emit(
                        f"⚠ epoch callback error: {type(cb_e).__name__}: {cb_e}")

            try:
                model.add_callback("on_train_start", _cb_train_start)
                model.add_callback("on_fit_epoch_end", _cb_fit_epoch_end)
            except Exception:
                # Phiên bản ultralytics quá cũ → bỏ qua callback, train vẫn chạy
                pass

            # Build kwargs — chỉ pass device khi user chỉ định để giữ
            # default behaviour (auto-detect) cho "Auto".
            train_kwargs = dict(
                data      = cfg["data_yaml"],
                epochs    = cfg["epochs"],
                imgsz     = cfg["imgsz"],
                batch     = cfg["batch"],
                lr0       = cfg["lr0"],
                workers   = cfg.get("workers", 4),
                patience  = cfg.get("patience", 50),
                name      = cfg.get("name","yolo_run"),
                plots     = True,
                # `augment` arg trong train() = TTA cho validation phase,
                # KHÔNG phải training augmentation (mosaic/hsv/flip luôn
                # bật theo default ultralytics).
                augment   = cfg.get("val_tta", False),
                seed      = 42,
                verbose   = True,
            )
            if device:
                train_kwargs["device"] = device

            result = model.train(**train_kwargs)
            save_dir = str(result.save_dir)
            self.log_line.emit(f"✅ Training done! Saved: {save_dir}")
            final_metrics = self._parse_csv_log(save_dir)
            if final_metrics:
                self.metrics.emit(final_metrics)
            # ONNX export KHÔNG còn auto — user bấm nút "Export ONNX" sau
            # khi train xong. Lý do: torch.onnx + simplifier hay crash hẳn
            # process trên 1 số torch build (vd torch 2.12+cpu), gây "train
            # xong tự out app". Export riêng giảm risk.
            self.finished.emit(save_dir)
        except Exception as e:
            self.error.emit(str(e))

    def _parse_csv_log(self, save_dir: str) -> dict:
        """In log + return metrics row cuối: mAP50, mAP50-95, P, R."""
        csv_path = os.path.join(save_dir, "results.csv")
        if not os.path.exists(csv_path):
            return {}
        try:
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                return {}
            headers = [h.strip() for h in rows[0]]
            self.log_line.emit("── Training Results ──")
            self.log_line.emit(" | ".join(headers))
            for row in rows[1:]:
                self.log_line.emit(" | ".join(v.strip() for v in row))
            # Extract final-epoch metrics
            last = [v.strip() for v in rows[-1]]
            out = {}
            for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                       "metrics/precision(B)", "metrics/recall(B)"):
                if k in headers:
                    try:
                        out[k] = float(last[headers.index(k)])
                    except (ValueError, IndexError):
                        pass
            return out
        except Exception:
            return {}


# ═══════════════════════════════════════════════════════════════
#  MAIN YOLO STUDIO DIALOG
# ═══════════════════════════════════════════════════════════════
class YoloStudioDialog(QDialog):
    """
    Cửa sổ YOLO Studio — Label + Dataset + Train, độc lập với pipeline.
    Sau train: hiện charts độ chính xác + nút Save Model To… cho user
    chọn nơi lưu. Pipeline dùng model qua YOLO Detect node (file picker).
    """

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
        self._epoch_history: List[Tuple[int, dict]] = []
        # Chỉ chấp nhận ndarray; mọi giá trị khác (vd bool từ slot
        # clicked/triggered) coi như không có ảnh để tránh crash trong
        # _load_single_image → canvas.load_image (đòi img.shape).
        if not isinstance(initial_image, np.ndarray):
            initial_image = None
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

        # Device selection (Auto / CPU / GPU 0)
        # Auto → để ultralytics tự chọn (CUDA nếu có, fallback CPU)
        # CPU  → ép device='cpu'
        # GPU 0 → ép device='0' (chỉ nên dùng khi torch build có CUDA và máy có GPU NVIDIA)
        dev_grp = QGroupBox("Device")
        dvg = QVBoxLayout(dev_grp); dvg.setContentsMargins(8,12,8,8); dvg.setSpacing(4)
        self._device_combo = QComboBox()
        self._device_combo.addItem("Auto (CUDA nếu có, fallback CPU)", "")
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.addItem("GPU 0 (CUDA)", "0")
        self._device_combo.setStyleSheet(
            "QComboBox{background:#0a0e1a;border:1px solid #1e2d45;"
            "color:#e2e8f0;padding:4px 6px;border-radius:3px;}"
            "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
            "selection-background-color:#1a2236;}")
        self._device_combo.setToolTip(
            "Auto: ultralytics tự chọn.\n"
            "CPU: ép train trên CPU (chậm nhưng luôn chạy được).\n"
            "GPU 0: yêu cầu torch build CUDA + máy có GPU NVIDIA, không có sẽ lỗi.")
        dvg.addWidget(self._device_combo)
        self._device_hint = QLabel(self._detect_device_hint())
        self._device_hint.setStyleSheet("color:#64748b; font-size:10px;")
        self._device_hint.setWordWrap(True)
        dvg.addWidget(self._device_hint)
        ll.addWidget(dev_grp)

        # Hyperparams
        # workers=0 trên Windows để tránh hang do fork() semantics khác Unix;
        # các OS khác giữ 4 cho throughput tốt.
        _default_workers = 0 if sys.platform.startswith("win") else 4
        hyp_grp = QGroupBox("Hyperparameters")
        hg = QVBoxLayout(hyp_grp); hg.setContentsMargins(8,12,8,8); hg.setSpacing(5)
        self._hp_epochs   = self._param_row("Epochs",    hg, "int",   100, 1, 5000)
        self._hp_patience = self._param_row("Patience",  hg, "int",   50,  0, 5000)
        self._hp_imgsz    = self._param_row("Image Size",hg, "int",   640, 32, 4096, step=32)
        self._hp_batch    = self._param_row("Batch Size",hg, "int",   16,  1, 512)
        self._hp_lr0      = self._param_row("LR0",       hg, "float", 0.01,1e-5,1.0,4)
        self._hp_workers  = self._param_row("Workers",   hg, "int",   _default_workers, 0, 32)
        self._hp_val_tta = QCheckBox("Validation TTA  (chậm hơn, mAP nhỉnh hơn)")
        self._hp_val_tta.setChecked(False)
        self._hp_val_tta.setToolTip(
            "Test-Time Augmentation cho validation phase. Training "
            "augmentation (mosaic/hsv/flip) luôn bật theo default "
            "ultralytics, không control được ở đây.")
        self._hp_val_tta.setStyleSheet("color:#94a3b8;")
        hg.addWidget(self._hp_val_tta)
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

        # Live metrics — update theo từng epoch trong khi train chạy.
        # 4 ô: Epoch, mAP50, mAP50-95, P, R + 1 dòng losses bên dưới.
        self._live_grp = QGroupBox("Live Metrics")
        lvg = QVBoxLayout(self._live_grp); lvg.setContentsMargins(8,12,8,8); lvg.setSpacing(4)
        lv_row = QHBoxLayout(); lv_row.setSpacing(10)
        self._live_epoch  = self._live_metric_label("Epoch", "—")
        self._live_map50  = self._live_metric_label("mAP50", "—")
        self._live_map95  = self._live_metric_label("mAP50-95", "—")
        self._live_p      = self._live_metric_label("Precision", "—")
        self._live_r      = self._live_metric_label("Recall", "—")
        for lbl in (self._live_epoch, self._live_map50, self._live_map95,
                    self._live_p, self._live_r):
            lv_row.addWidget(lbl)
        lv_row.addStretch()
        lvg.addLayout(lv_row)
        self._live_losses = QLabel("box_loss: —   cls_loss: —   dfl_loss: —")
        self._live_losses.setStyleSheet(
            "color:#94a3b8; font-size:11px; font-family:'Courier New';")
        lvg.addWidget(self._live_losses)
        rl.addWidget(self._live_grp)
        self._live_grp.hide()

        # Live chart — reload results.png từ save_dir mỗi vài giây trong khi
        # train. ultralytics tự sinh results.png sau mỗi epoch (plots=True).
        self._live_chart_lbl = QLabel("(chart sẽ hiện sau epoch đầu tiên)")
        self._live_chart_lbl.setAlignment(Qt.AlignCenter)
        self._live_chart_lbl.setMinimumHeight(180)
        self._live_chart_lbl.setStyleSheet(
            "background:#050810; border:1px solid #1e2d45; border-radius:4px;"
            "color:#64748b; font-size:11px;")
        self._live_chart_lbl.hide()
        rl.addWidget(self._live_chart_lbl)

        self._live_save_dir: Optional[str] = None
        self._live_chart_timer = QTimer(self)
        self._live_chart_timer.setInterval(3000)  # 3s
        self._live_chart_timer.timeout.connect(self._refresh_live_chart)

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

        # Result info + accuracy charts + save button
        self._result_grp = QGroupBox("Last Training Result")
        rg2 = QVBoxLayout(self._result_grp); rg2.setContentsMargins(8,12,8,8)
        self._result_info = QLabel("—")
        self._result_info.setStyleSheet(
            "color:#94a3b8; font-size:11px; font-family:'Courier New';")
        self._result_info.setWordWrap(True)
        rg2.addWidget(self._result_info)

        # ── Model Evaluation panel ──────────────────────────────
        # Sau khi train xong: chấm điểm chất lượng + detect overfit từ
        # epoch_history, kèm khuyến nghị cụ thể (train thêm, giảm epochs,
        # thêm augmentation...). Hiện ngay phía trên charts để user nhìn
        # trước rồi mới xem chi tiết.
        self._eval_grp = QGroupBox("📊 Model Evaluation")
        self._eval_grp.setStyleSheet(
            "QGroupBox{border:1px solid #1e2d45; border-radius:6px;"
            "margin-top:8px; padding-top:10px;"
            "color:#00d4ff; font-size:11px; font-weight:700;}"
            "QGroupBox::title{subcontrol-origin:margin; left:10px; padding:0 6px;}")
        eg = QVBoxLayout(self._eval_grp); eg.setContentsMargins(10, 14, 10, 10); eg.setSpacing(8)

        # Verdict row: Quality | Overfit | Convergence
        verdict_row = QHBoxLayout(); verdict_row.setSpacing(10)
        self._eval_quality = self._verdict_box("Quality", "—", "#64748b")
        self._eval_overfit = self._verdict_box("Overfitting", "—", "#64748b")
        self._eval_converge = self._verdict_box("Convergence", "—", "#64748b")
        verdict_row.addWidget(self._eval_quality, 1)
        verdict_row.addWidget(self._eval_overfit, 1)
        verdict_row.addWidget(self._eval_converge, 1)
        eg.addLayout(verdict_row)

        # Diagnosis + recommendation
        self._eval_text = QTextEdit()
        self._eval_text.setReadOnly(True)
        self._eval_text.setMaximumHeight(160)
        self._eval_text.setStyleSheet(
            "QTextEdit{background:#050810; color:#cbd5e1;"
            "border:1px solid #1e2d45; border-radius:4px;"
            "font-family:'Courier New'; font-size:11px; padding:6px;}")
        eg.addWidget(self._eval_text)
        rg2.addWidget(self._eval_grp)

        # Charts: results.png + confusion_matrix.png (cuộn ngang nếu nhiều)
        self._charts_scroll = QScrollArea()
        self._charts_scroll.setWidgetResizable(True)
        self._charts_scroll.setFixedHeight(280)
        self._charts_scroll.setStyleSheet(
            "QScrollArea{background:#050810;border:1px solid #1e2d45;"
            "border-radius:4px;}")
        self._charts_container = QWidget()
        self._charts_layout = QHBoxLayout(self._charts_container)
        self._charts_layout.setContentsMargins(6, 6, 6, 6)
        self._charts_layout.setSpacing(6)
        self._charts_scroll.setWidget(self._charts_container)
        rg2.addWidget(self._charts_scroll)

        btns_row = QHBoxLayout(); btns_row.setSpacing(6)
        btn_open_result = self._small_btn("📂 Open Result Folder")
        btn_open_result.clicked.connect(self._open_result_folder)
        btns_row.addWidget(btn_open_result)

        # ONNX export là step riêng (không auto chạy ở cuối train) — tránh
        # crash hẳn process khi torch.onnx có vấn đề với build hiện tại.
        self._btn_export_onnx = self._small_btn("⚡ Export ONNX")
        self._btn_export_onnx.setStyleSheet(
            "QPushButton{background:#1a365d;border:1px solid #00d4ff;"
            "border-radius:3px;color:#00d4ff;font-size:11px;padding:2px 8px;}"
            "QPushButton:hover{background:#00d4ff;color:#000;}"
            "QPushButton:disabled{background:#1e2d45;color:#1e2d45;border-color:#1e2d45;}")
        self._btn_export_onnx.setToolTip(
            "Export best.pt → best.onnx (FP32) cho CPU inference nhanh hơn 3-4x.\n"
            "Tách khỏi pipeline train để nếu lỗi cũng không crash app.")
        self._btn_export_onnx.clicked.connect(self._export_onnx_clicked)
        btns_row.addWidget(self._btn_export_onnx)

        btn_save_model = self._small_btn("💾 Save Model To…")
        btn_save_model.setStyleSheet(
            "QPushButton{background:#1b4332;border:1px solid #39ff14;"
            "border-radius:3px;color:#39ff14;font-size:11px;padding:2px 8px;}"
            "QPushButton:hover{background:#39ff14;color:#000;}")
        btn_save_model.clicked.connect(self._save_model_to)
        btns_row.addWidget(btn_save_model)
        rg2.addLayout(btns_row)

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
            "patience":  self._hp_patience.value(),
            "imgsz":     self._hp_imgsz.value(),
            "batch":     self._hp_batch.value(),
            "lr0":       self._hp_lr0.value(),
            "workers":   self._hp_workers.value(),
            "val_tta":   self._hp_val_tta.isChecked(),
            "name":      self._run_name.text().strip() or "yolo_aoi_run",
            "device":    self._device_combo.currentData() or "",
        }

        self._train_log.clear()
        self._train_log.append(f"Config: {json.dumps(config, indent=2)}")
        self._train_progress.show()
        self._train_progress.setRange(0, max(1, config["epochs"]))
        self._train_progress.setValue(0)
        self._btn_start_train.setEnabled(False)
        self._btn_stop_train.setEnabled(True)
        self._train_status.setText("Training...")
        self._train_status.setStyleSheet("color:#ffd700; font-size:11px;")
        self._result_grp.hide()
        # Reset live metrics
        self._live_grp.show()
        self._live_chart_lbl.show()
        self._live_chart_lbl.clear()
        self._live_chart_lbl.setText("(chart sẽ hiện sau epoch đầu tiên)")
        self._set_live_metric(self._live_epoch, "Epoch", f"0 / {config['epochs']}")
        for w, lbl in ((self._live_map50, "mAP50"),
                       (self._live_map95, "mAP50-95"),
                       (self._live_p, "Precision"),
                       (self._live_r, "Recall")):
            self._set_live_metric(w, lbl, "—")
        self._live_losses.setText(
            "box_loss: —   cls_loss: —   dfl_loss: —")
        self._live_save_dir = None
        # Reset history dùng cho live chart — mỗi run mới = chart từ đầu
        self._epoch_history: List[Tuple[int, dict]] = []

        self._train_worker = YoloTrainWorker(config)
        self._train_thread = QThread()
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.log_line.connect(self._on_train_log)
        self._train_worker.metrics.connect(self._on_train_metrics)
        self._train_worker.save_dir.connect(self._on_save_dir)
        self._train_worker.epoch_end.connect(self._on_epoch_end)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.finished.connect(self._train_thread.quit)
        self._train_worker.error.connect(self._train_thread.quit)
        # Cleanup an toàn — worker + thread tự xóa khi thread kết thúc, tránh
        # leak / dangling references nếu user train nhiều lần liên tiếp.
        self._train_thread.finished.connect(self._train_worker.deleteLater)
        self._train_thread.finished.connect(self._train_thread.deleteLater)
        self._train_thread.start()
        self._last_save_dir = None
        self._last_metrics: dict = {}

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

    def _on_train_metrics(self, metrics: dict):
        self._last_metrics = metrics or {}

    def _on_save_dir(self, save_dir: str):
        """Worker thông báo save_dir ngay khi training start → bật timer
        reload results.png cho live chart."""
        self._live_save_dir = save_dir
        self._train_log.append(f"📁 Save dir: {save_dir}")
        if not self._live_chart_timer.isActive():
            self._live_chart_timer.start()

    def _on_epoch_end(self, epoch: int, total: int, metrics: dict):
        """Update live metrics panel + progress bar mỗi khi 1 epoch xong."""
        if total > 0:
            self._train_progress.setRange(0, total)
            self._train_progress.setValue(epoch)
        self._set_live_metric(self._live_epoch, "Epoch", f"{epoch} / {total}")
        def _f(k):
            return f"{metrics[k]:.4f}" if k in metrics else "—"
        self._set_live_metric(self._live_map50,  "mAP50",     _f("metrics/mAP50(B)"))
        self._set_live_metric(self._live_map95,  "mAP50-95",  _f("metrics/mAP50-95(B)"))
        self._set_live_metric(self._live_p,      "Precision", _f("metrics/precision(B)"))
        self._set_live_metric(self._live_r,      "Recall",    _f("metrics/recall(B)"))
        self._live_losses.setText(
            f"box_loss: {_f('train/box_loss')}   "
            f"cls_loss: {_f('train/cls_loss')}   "
            f"dfl_loss: {_f('train/dfl_loss')}")
        # Lưu epoch này vào history rồi vẽ lại chart từ data có sẵn (không
        # phụ thuộc results.png — ultralytics 8.x chỉ tự sinh file đó ở
        # cuối train, không phải sau mỗi epoch).
        self._epoch_history.append((epoch, dict(metrics or {})))
        self._refresh_live_chart()

    def _refresh_live_chart(self):
        """Build line chart từ epoch_history (mAP + losses theo epoch).
        Ưu tiên tự vẽ với matplotlib vì ultralytics chỉ sinh results.png
        cuối train. Fallback dùng results.png nếu đã có (sau khi train xong)."""
        # Ưu tiên 1: vẽ từ history (luôn available trong khi train)
        pix = self._build_history_chart()
        if pix is None or pix.isNull():
            # Fallback: results.png nếu ultralytics đã sinh xong
            d = self._live_save_dir
            if d:
                path = os.path.join(d, "results.png")
                if os.path.exists(path):
                    pix2 = QPixmap(path)
                    if not pix2.isNull():
                        pix = pix2
        if pix is None or pix.isNull():
            return
        target_w = max(360, self._live_chart_lbl.width() - 8)
        self._live_chart_lbl.setPixmap(
            pix.scaledToWidth(target_w, Qt.SmoothTransformation))
        self._live_chart_lbl.setToolTip(
            f"{len(self._epoch_history)} epoch(s) plotted")

    def _build_history_chart(self) -> Optional[QPixmap]:
        """Render 2-panel line chart (mAP + losses) từ self._epoch_history
        bằng matplotlib → QPixmap. Trả về None nếu chưa có data hoặc
        matplotlib không khả dụng."""
        if not self._epoch_history:
            return None
        try:
            os.environ.setdefault("MPLBACKEND", "Agg")
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from io import BytesIO
        except Exception:
            return None

        try:
            epochs = [e for e, _ in self._epoch_history]
            def _series(key):
                return [m.get(key, float("nan")) for _, m in self._epoch_history]
            map50    = _series("metrics/mAP50(B)")
            map95    = _series("metrics/mAP50-95(B)")
            prec     = _series("metrics/precision(B)")
            recall   = _series("metrics/recall(B)")
            box_loss = _series("train/box_loss")
            cls_loss = _series("train/cls_loss")
            dfl_loss = _series("train/dfl_loss")

            fig, axes = plt.subplots(1, 2, figsize=(11, 3.0), dpi=100)
            ax1, ax2 = axes

            # Panel 1: validation metrics (mAP/P/R) — scale 0..1
            ax1.plot(epochs, map50,  label="mAP50",    color="#00d4ff",
                     marker="o", lw=2, markersize=4)
            ax1.plot(epochs, map95,  label="mAP50-95", color="#39ff14",
                     marker="s", lw=2, markersize=4)
            ax1.plot(epochs, prec,   label="P",        color="#ffd700",
                     marker="^", lw=1.5, markersize=3, alpha=0.85)
            ax1.plot(epochs, recall, label="R",        color="#ff8c00",
                     marker="v", lw=1.5, markersize=3, alpha=0.85)
            ax1.set_title("Validation Metrics", color="#e2e8f0", fontsize=10)
            ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
            ax1.legend(facecolor="#0d1220", edgecolor="#1e2d45",
                       labelcolor="#e2e8f0", fontsize=8, loc="best")
            ax1.set_ylim(-0.02, 1.02)

            # Panel 2: training losses
            ax2.plot(epochs, box_loss, label="box_loss", color="#ff3860",
                     marker="o", lw=2, markersize=4)
            ax2.plot(epochs, cls_loss, label="cls_loss", color="#ff8c00",
                     marker="s", lw=2, markersize=4)
            ax2.plot(epochs, dfl_loss, label="dfl_loss", color="#ffd700",
                     marker="^", lw=2, markersize=4)
            ax2.set_title("Training Losses", color="#e2e8f0", fontsize=10)
            ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
            ax2.legend(facecolor="#0d1220", edgecolor="#1e2d45",
                       labelcolor="#e2e8f0", fontsize=8, loc="best")

            for ax in (ax1, ax2):
                ax.set_facecolor("#050810")
                ax.tick_params(colors="#94a3b8", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#1e2d45")
                ax.grid(True, alpha=0.15, color="#1e2d45", linestyle="--")
                # X-axis integer ticks vì epoch là integer
                if len(epochs) <= 30:
                    ax.set_xticks(epochs)

            fig.patch.set_facecolor("#050810")
            fig.tight_layout(pad=1.2)

            buf = BytesIO()
            fig.savefig(buf, format="png",
                        facecolor="#050810", edgecolor="none")
            plt.close(fig)

            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            return pix if not pix.isNull() else None
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
            return None

    # ── Model Evaluation ────────────────────────────────────────
    def _verdict_box(self, title: str, value: str, color: str) -> QLabel:
        """Tạo ô verdict: tiêu đề nhỏ + giá trị to màu sắc."""
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumHeight(56)
        lbl.setStyleSheet(
            "QLabel{background:#050810; border:1px solid #1e2d45;"
            "border-radius:5px; padding:8px;}")
        self._set_verdict(lbl, title, value, color)
        return lbl

    def _set_verdict(self, lbl: QLabel, title: str, value: str, color: str):
        lbl.setText(
            f"<div style='color:#64748b;font-size:10px;letter-spacing:1.5px;"
            f"margin-bottom:3px;'>{title.upper()}</div>"
            f"<div style='color:{color};font-size:15px;font-weight:700;'>"
            f"{value}</div>")

    def _update_evaluation(self):
        """Phân tích self._epoch_history → render lên _eval_grp."""
        res = self._evaluate_model(self._epoch_history)
        if not res:
            self._set_verdict(self._eval_quality, "Quality", "N/A", "#64748b")
            self._set_verdict(self._eval_overfit, "Overfitting", "N/A", "#64748b")
            self._set_verdict(self._eval_converge, "Convergence", "N/A", "#64748b")
            self._eval_text.setPlainText(
                "Không đủ data để đánh giá (history rỗng).")
            return
        self._set_verdict(self._eval_quality, "Quality",
                          res["quality_label"], res["quality_color"])
        self._set_verdict(self._eval_overfit, "Overfitting",
                          res["overfit_label"], res["overfit_color"])
        self._set_verdict(self._eval_converge, "Convergence",
                          res["converge_label"], res["converge_color"])

        # Build rich-text report
        diag = res["diagnosis"]
        recs = res["recommendations"]
        parts = []
        parts.append("<b style='color:#00d4ff;'>━━ Chẩn đoán ━━</b><br/>")
        for d in diag:
            parts.append(f"• {d}<br/>")
        parts.append("<br/><b style='color:#39ff14;'>━━ Khuyến nghị ━━</b><br/>")
        for r in recs:
            parts.append(f"→ {r}<br/>")
        self._eval_text.setHtml("".join(parts))

    def _evaluate_model(self, history) -> dict:
        """Phân tích training history → verdict + diagnosis + recommendations.
        history: list[(epoch, metrics_dict)] từ on_fit_epoch_end callback.

        Heuristics:
        • Quality từ mAP50 cuối cùng (hoặc best mAP50 nếu được lưu).
        • Overfitting: so train_loss vs val_loss + xem mAP50 có peak rồi tụt.
        • Convergence: kiểm tra mAP50 ở các epoch cuối có còn tăng hay đã
          plateau (không cần train thêm) hay vẫn improving (nên train thêm).
        """
        if not history:
            return {}

        rows = [m for _, m in history]
        epochs = [e for e, _ in history]
        n = len(rows)
        last = rows[-1]

        def _g(d, k, default=float("nan")):
            v = d.get(k, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        map50  = _g(last, "metrics/mAP50(B)", 0.0)
        map95  = _g(last, "metrics/mAP50-95(B)", 0.0)
        prec   = _g(last, "metrics/precision(B)", 0.0)
        recall = _g(last, "metrics/recall(B)", 0.0)

        map50_curve = [_g(r, "metrics/mAP50(B)", 0.0) for r in rows]
        best_map50 = max(map50_curve) if map50_curve else 0.0
        best_idx = map50_curve.index(best_map50) if map50_curve else 0

        # ── Quality (dựa trên best mAP50) ──
        if best_map50 >= 0.85:
            q_label, q_color = "Excellent", "#39ff14"
        elif best_map50 >= 0.70:
            q_label, q_color = "Good", "#00d4ff"
        elif best_map50 >= 0.50:
            q_label, q_color = "Acceptable", "#ffd700"
        elif best_map50 >= 0.30:
            q_label, q_color = "Poor", "#ff8c00"
        else:
            q_label, q_color = "Very Poor", "#ff3860"

        # ── Overfitting detection ──
        # Tín hiệu 1: mAP50 đạt peak rồi tụt > 0.05 (5%) ở các epoch sau.
        # Tín hiệu 2: val_loss / train_loss ratio rất cao ở các epoch cuối.
        overfit_score = 0
        overfit_signals = []

        if n >= 5:
            # mAP50 peak-and-drop
            tail_drop = best_map50 - map50_curve[-1]
            # Chỉ tính là overfit nếu peak ở epoch trước cuối VÀ tụt > 0.05
            if best_idx < n - 2 and tail_drop > 0.05:
                overfit_score += 2
                overfit_signals.append(
                    f"mAP50 đỉnh epoch {epochs[best_idx]} ({best_map50:.3f}) "
                    f"rồi tụt {tail_drop:.3f} ở epoch cuối ({map50_curve[-1]:.3f})")
            elif best_idx < n - 2 and tail_drop > 0.02:
                overfit_score += 1
                overfit_signals.append(
                    f"mAP50 đỉnh epoch {epochs[best_idx]} rồi tụt nhẹ "
                    f"{tail_drop:.3f}")

            # val_loss vs train_loss ratio
            tail = rows[-max(3, n // 5):]  # ~20% cuối hoặc 3 epoch
            t_box = [_g(r, "train/box_loss") for r in tail
                      if not math.isnan(_g(r, "train/box_loss"))]
            v_box = [_g(r, "val/box_loss") for r in tail
                      if not math.isnan(_g(r, "val/box_loss"))]
            if t_box and v_box:
                t_avg = sum(t_box) / len(t_box)
                v_avg = sum(v_box) / len(v_box)
                if t_avg > 1e-6:
                    ratio = v_avg / t_avg
                    if ratio > 2.0:
                        overfit_score += 2
                        overfit_signals.append(
                            f"val_loss/train_loss = {ratio:.2f} "
                            f"(>2.0 → gap quá lớn)")
                    elif ratio > 1.5:
                        overfit_score += 1
                        overfit_signals.append(
                            f"val_loss/train_loss = {ratio:.2f} "
                            f"(>1.5 → có gap)")

        if overfit_score >= 3:
            o_label, o_color = "Strong", "#ff3860"
        elif overfit_score >= 1:
            o_label, o_color = "Mild", "#ffd700"
        else:
            o_label, o_color = "No", "#39ff14"

        # ── Convergence: mAP50 ở các epoch cuối có còn tăng đáng kể? ──
        # Lấy delta giữa avg 25% cuối và avg 25% áp cuối; nếu vẫn dương
        # → chưa hội tụ (nên train thêm); nếu ~0 → đã plateau (đủ rồi).
        if n >= 6:
            q = max(1, n // 4)
            avg_last = sum(map50_curve[-q:]) / q
            avg_prev = sum(map50_curve[-2*q:-q]) / q
            delta = avg_last - avg_prev
            if delta > 0.03:
                c_label, c_color = "Improving", "#ffd700"
                converge_msg = (f"mAP50 vẫn còn tăng (+{delta:.3f} trong "
                                f"~{q} epoch cuối) — có thể train thêm")
            elif delta < -0.02:
                c_label, c_color = "Diverging", "#ff3860"
                converge_msg = (f"mAP50 đang giảm ({delta:+.3f}) — "
                                f"dấu hiệu overfit, nên stop sớm")
            else:
                c_label, c_color = "Plateau", "#39ff14"
                converge_msg = (f"mAP50 đã plateau (Δ={delta:+.3f}) — "
                                f"hội tụ, không cần train thêm")
        else:
            c_label, c_color = "Too Short", "#64748b"
            converge_msg = f"Chỉ có {n} epoch — không đủ data để đánh giá convergence"

        # ── Build diagnosis text ──
        diagnosis = []
        diagnosis.append(
            f"Final epoch: mAP50={map50:.4f}, mAP50-95={map95:.4f}, "
            f"P={prec:.4f}, R={recall:.4f}")
        diagnosis.append(
            f"Best mAP50: {best_map50:.4f} (epoch {epochs[best_idx]}/{epochs[-1]})")
        if overfit_signals:
            for s in overfit_signals:
                diagnosis.append(s)
        else:
            diagnosis.append("Không phát hiện dấu hiệu overfit rõ rệt.")
        diagnosis.append(converge_msg)

        # P/R imbalance
        pr_gap = abs(prec - recall)
        if pr_gap > 0.2 and (prec > 0 or recall > 0):
            if prec > recall:
                diagnosis.append(
                    f"P >> R ({prec:.3f} vs {recall:.3f}): model bỏ sót nhiều "
                    f"object (false negatives nhiều)")
            else:
                diagnosis.append(
                    f"R >> P ({recall:.3f} vs {prec:.3f}): model detect nhiều "
                    f"object sai (false positives nhiều)")

        # ── Recommendations ──
        recs = []
        if best_map50 < 0.3:
            recs.append("Thêm data labeled (mỗi class ≥ 100-200 ảnh là tối thiểu).")
            recs.append("Tăng epochs (50-100+) và imgsz lên 640 hoặc 1280.")
            recs.append("Thử model lớn hơn: yolov8s/m thay vì yolov8n.")
        elif best_map50 < 0.5:
            recs.append("Train thêm epochs (50-100) — model chưa đủ ngấm data.")
            recs.append("Cân nhắc thử yolov8s.pt nếu đang dùng nano.")
        elif overfit_score >= 3:
            recs.append("GIẢM epochs (best ở epoch "
                        f"{epochs[best_idx]}, không cần thêm).")
            recs.append("Patience giảm xuống ~10-20 để stop sớm.")
            recs.append("Thêm augmentation (mosaic mặc định bật, có thể "
                        "tăng degrees/translate/scale trong cfg ultralytics).")
            recs.append("Thêm data validation — overfit thường do val set quá nhỏ.")
        elif overfit_score >= 1:
            recs.append(f"Cân nhắc dùng best.pt (epoch {epochs[best_idx]}) "
                        f"thay vì last.pt.")
            recs.append("Theo dõi sát các epoch sau — nếu mAP tụt tiếp thì stop.")
        elif c_label == "Improving":
            recs.append("Train thêm epochs — mAP50 vẫn còn tăng.")
        else:
            recs.append("Mô hình OK để dùng. Bấm 💾 Save Model To… để export.")
            if best_map50 >= 0.7:
                recs.append("Test trên ảnh thực tế ngoài val set để verify generalization.")

        if pr_gap > 0.2:
            if prec > recall:
                recs.append("Giảm confidence threshold (lúc inference) để "
                            "detect được nhiều object hơn (tăng recall).")
            else:
                recs.append("Tăng confidence threshold (lúc inference) để "
                            "loại bớt false positives (tăng precision).")

        return {
            "quality_label": q_label, "quality_color": q_color,
            "overfit_label": o_label, "overfit_color": o_color,
            "converge_label": c_label, "converge_color": c_color,
            "diagnosis": diagnosis,
            "recommendations": recs,
        }

    def _on_train_done(self, save_dir: str):
        self._last_save_dir = save_dir
        self._finalize_training()
        self._train_status.setText("✅ Done!")
        self._train_status.setStyleSheet("color:#39ff14; font-size:11px;")
        best_pt   = os.path.join(save_dir, "weights", "best.pt")
        best_onnx = os.path.join(save_dir, "weights", "best.onnx")
        m = self._last_metrics
        metrics_line = ""
        if m:
            def _g(k): return f"{m[k]:.4f}" if k in m else "—"
            metrics_line = (
                f"\nmAP50: {_g('metrics/mAP50(B)')}  "
                f"mAP50-95: {_g('metrics/mAP50-95(B)')}  "
                f"P: {_g('metrics/precision(B)')}  "
                f"R: {_g('metrics/recall(B)')}")
        self._result_info.setText(
            f"Save dir: {save_dir}\n"
            f"best.pt:   {'✔ exists' if os.path.exists(best_pt) else '✖ missing'}\n"
            f"best.onnx: {'✔ exists (fast CPU)' if os.path.exists(best_onnx) else '✖ missing'}"
            + metrics_line)
        # Evaluate model trước khi show — verdict + diagnosis + recommendation
        self._update_evaluation()
        self._populate_charts(save_dir)
        self._result_grp.show()
        QMessageBox.information(self,"Training Complete",
            f"Training finished!\nSaved to:\n{save_dir}" + metrics_line)

    def _populate_charts(self, save_dir: str):
        """Show accuracy/training charts ultralytics tự sinh trong save_dir."""
        # Clear cũ
        while self._charts_layout.count():
            it = self._charts_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Charts ưu tiên hiện theo thứ tự — bỏ qua file không tồn tại.
        chart_files = [
            ("results.png",            "Loss + mAP curves"),
            ("confusion_matrix.png",   "Confusion Matrix"),
            ("confusion_matrix_normalized.png", "Confusion Matrix (norm)"),
            ("PR_curve.png",           "Precision-Recall"),
            ("F1_curve.png",           "F1 Curve"),
            ("P_curve.png",            "Precision Curve"),
            ("R_curve.png",            "Recall Curve"),
        ]
        added = 0
        for fn, title in chart_files:
            path = os.path.join(save_dir, fn)
            if not os.path.exists(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            # Chart wrapper: title + image (click → preview lớn)
            wrap = QWidget()
            v = QVBoxLayout(wrap); v.setContentsMargins(2, 2, 2, 2); v.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                "color:#00d4ff; font-size:10px; font-weight:700; padding:2px;")
            t.setAlignment(Qt.AlignCenter)
            v.addWidget(t)
            img_lbl = QLabel()
            img_lbl.setPixmap(pix.scaledToHeight(220, Qt.SmoothTransformation))
            img_lbl.setCursor(Qt.PointingHandCursor)
            img_lbl.setToolTip(f"Click để xem full size\n{path}")
            img_lbl.mousePressEvent = (
                lambda _ev, p=path, ttl=title: self._preview_chart(p, ttl))
            img_lbl.setStyleSheet(
                "background:#0a0e1a; border:1px solid #1e2d45; border-radius:3px;")
            v.addWidget(img_lbl)
            self._charts_layout.addWidget(wrap)
            added += 1
        if added == 0:
            empty = QLabel("(Chưa có chart — train chưa sinh results.png)")
            empty.setStyleSheet("color:#64748b; font-size:11px;")
            empty.setAlignment(Qt.AlignCenter)
            self._charts_layout.addWidget(empty)
        self._charts_layout.addStretch()

    def _preview_chart(self, path: str, title: str):
        """Mở dialog xem chart full-size."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"📊  {title}")
        dlg.resize(900, 700)
        v = QVBoxLayout(dlg); v.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        lbl = QLabel()
        lbl.setPixmap(QPixmap(path))
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl)
        v.addWidget(scroll)
        dlg.exec()

    def _on_train_error(self, msg: str):
        self._finalize_training()
        self._train_status.setText("Error!")
        self._train_status.setStyleSheet("color:#ff3860; font-size:11px;")
        self._train_log.append(f"\n❌ ERROR: {msg}")
        QMessageBox.critical(self, "Training Error", _explain_train_error(msg))

    def _finalize_training(self):
        self._btn_start_train.setEnabled(True)
        self._btn_stop_train.setEnabled(False)
        self._train_progress.hide()
        if self._live_chart_timer.isActive():
            self._live_chart_timer.stop()
        # Lần chart refresh cuối để chắc chắn có results.png mới nhất
        self._refresh_live_chart()

    # ── Live metrics helpers ────────────────────────────────────
    def _live_metric_label(self, title: str, value: str) -> QLabel:
        """Tạo 1 ô metric kiểu 2 dòng: tiêu đề nhỏ + giá trị to."""
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumWidth(96)
        lbl.setStyleSheet(
            "QLabel{background:#050810; border:1px solid #1e2d45;"
            "border-radius:4px; padding:6px;}")
        self._set_live_metric(lbl, title, value)
        return lbl

    def _set_live_metric(self, lbl: QLabel, title: str, value: str):
        lbl.setText(
            f"<div style='color:#64748b;font-size:9px;letter-spacing:1px;'>"
            f"{title.upper()}</div>"
            f"<div style='color:#00d4ff;font-size:15px;font-weight:700;"
            f"font-family:Courier New;'>{value}</div>")

    def _detect_device_hint(self) -> str:
        """Hint string hiển thị dưới combo Device — báo có CUDA hay không.
        Tránh import torch ở top-level (chỉ import khi cần) vì torch nặng."""
        try:
            import torch
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                return f"✓ Phát hiện {n} GPU CUDA — {name}"
            return "ℹ Không phát hiện CUDA — chọn GPU 0 sẽ lỗi, dùng Auto/CPU."
        except Exception:
            return "ℹ torch chưa load — Auto sẽ tự thử CUDA rồi fallback CPU."

    def _export_onnx_clicked(self):
        """Manual ONNX export — chạy trong thread riêng để không block UI
        và để nếu torch.onnx crash thì chỉ thread đó chết, không kéo theo
        cả app như khi auto-export ở cuối train."""
        d = getattr(self, "_last_save_dir", None)
        if not d:
            QMessageBox.warning(self, "No Model", "Chưa có training nào hoàn tất.")
            return
        best_pt = os.path.join(d, "weights", "best.pt")
        if not os.path.exists(best_pt):
            QMessageBox.warning(self, "Missing best.pt",
                f"Không tìm thấy:\n{best_pt}")
            return

        # Pre-flight check: kiểm tra các package ONNX cần. Nếu thiếu,
        # cho user chọn: để ultralytics AutoUpdate (chậm, cần internet)
        # hoặc cancel và tự cài bằng pip.
        missing = self._check_onnx_packages()
        if missing:
            msg = (
                "Để export ONNX, ultralytics yêu cầu các package sau (chưa cài):\n"
                f"   {', '.join(missing)}\n\n"
                "Có 2 cách:\n"
                "  1. Để ultralytics tự AutoUpdate (download tự động — cần\n"
                "     internet, có thể mất 1-2 phút, đôi khi fail do pip\n"
                "     permission).\n"
                "  2. Cancel rồi tự cài trước:\n"
                "       pip install onnx \"onnx<2.0.0\" onnxslim onnxruntime\n\n"
                "Tiếp tục với AutoUpdate?")
            r = QMessageBox.question(self, "ONNX packages missing", msg,
                                      QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)
            if r != QMessageBox.Yes:
                return
            self._train_log.append(
                f"⏳ Auto-installing {', '.join(missing)} — đợi tối đa 1-2 phút...")
        else:
            self._train_log.append("✓ ONNX packages OK — bắt đầu export.")

        self._btn_export_onnx.setEnabled(False)
        self._train_log.append("── Exporting ONNX (manual) ──")

        imgsz = self._hp_imgsz.value()
        save_dir = d
        cfg_yaml = self._yaml_path_edit.text().strip()

        def _do_export():
            try:
                os.environ.setdefault("MPLBACKEND", "Agg")
                from ultralytics import YOLO
                m = YOLO(best_pt)
                onnx_path = m.export(
                    format="onnx",
                    imgsz=imgsz,
                    simplify=True,
                    opset=12,
                    dynamic=False,
                    half=False,
                )
                # Sidecar classes.txt
                try:
                    import yaml as _yaml
                    with open(cfg_yaml) as f:
                        data = _yaml.safe_load(f) or {}
                    names = data.get("names") or []
                    if isinstance(names, dict):
                        names = [names[k] for k in sorted(
                            names.keys(), key=lambda x: int(x))]
                    if names:
                        side = os.path.splitext(str(onnx_path))[0] + ".classes.txt"
                        with open(side, "w", encoding="utf-8") as f:
                            f.write("\n".join(str(n) for n in names))
                except Exception:
                    pass
                return True, str(onnx_path)
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"

        # Chạy trong QThread đơn giản
        class _ExportRunner(QObject):
            done = Signal(bool, str)
            def run(self):
                ok, msg = _do_export()
                self.done.emit(ok, msg)

        self._export_runner = _ExportRunner()
        self._export_thread = QThread()
        self._export_runner.moveToThread(self._export_thread)
        self._export_thread.started.connect(self._export_runner.run)
        self._export_runner.done.connect(self._on_export_done)
        self._export_runner.done.connect(self._export_thread.quit)
        self._export_thread.finished.connect(self._export_runner.deleteLater)
        self._export_thread.finished.connect(self._export_thread.deleteLater)
        self._export_thread.start()

    def _on_export_done(self, ok: bool, msg: str):
        self._btn_export_onnx.setEnabled(True)
        if ok:
            self._train_log.append(f"✅ ONNX exported: {msg}")
            # Refresh result info để hiện onnx ✔
            if self._last_save_dir:
                best_onnx = os.path.join(
                    self._last_save_dir, "weights", "best.onnx")
                if os.path.exists(best_onnx):
                    txt = self._result_info.text().replace(
                        "best.onnx: ✖ missing",
                        "best.onnx: ✔ exists (fast CPU)")
                    self._result_info.setText(txt)
            QMessageBox.information(self, "Export OK",
                f"ONNX exported:\n{msg}")
        else:
            self._train_log.append(f"⚠ ONNX export failed: {msg}")
            # Nếu fail vì ultralytics AutoUpdate không cài được package,
            # hướng dẫn user cài thủ công.
            hint = ""
            low = msg.lower()
            if ("onnx" in low and ("not found" in low or "no module" in low
                    or "autoupdate" in low or "requirements" in low)):
                hint = (
                    "\n\nCó vẻ ultralytics AutoUpdate không cài được các "
                    "package ONNX. Mở terminal/cmd và chạy:\n"
                    "    pip install onnx \"onnx<2.0.0\" onnxslim onnxruntime\n"
                    "rồi quay lại bấm Export ONNX.")
            QMessageBox.warning(self, "Export Failed",
                f"Không export được ONNX:\n{msg}{hint}\n\n"
                "Train vẫn dùng được file best.pt (chậm hơn ONNX khoảng 3-4x).")

    def _check_onnx_packages(self) -> list:
        """Trả về list các package ONNX còn thiếu (cho export).
        ultralytics yêu cầu: onnx 1.12+/<2.0, onnxslim 0.1.71+, onnxruntime."""
        missing = []
        # onnx
        try:
            import onnx  # noqa: F401
        except ImportError:
            missing.append("onnx")
        # onnxslim
        try:
            import onnxslim  # noqa: F401
        except ImportError:
            missing.append("onnxslim")
        # onnxruntime
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            missing.append("onnxruntime")
        return missing

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

    def _save_model_to(self):
        """Copy best.pt + best.onnx + classes.txt sang folder user chọn.
        Sau đó user trỏ vào file đó từ YOLO Detect node trong pipeline."""
        d = getattr(self, "_last_save_dir", None)
        if not d:
            return
        weights_dir = os.path.join(d, "weights")
        candidates = []
        for fn in ("best.pt", "best.onnx", "best.classes.txt"):
            p = os.path.join(weights_dir, fn)
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            QMessageBox.warning(self, "Not Found",
                "Không tìm thấy file model nào trong weights/.")
            return

        target = QFileDialog.getExistingDirectory(
            self, "Chọn nơi lưu model", os.path.expanduser("~"))
        if not target:
            return

        copied = []
        try:
            for src in candidates:
                dst = os.path.join(target, os.path.basename(src))
                shutil.copy2(src, dst)
                copied.append(dst)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            return

        listing = "\n".join("  • " + os.path.basename(p) for p in copied)
        QMessageBox.information(self, "Model Saved",
            f"Đã copy {len(copied)} file vào:\n{target}\n\n{listing}\n\n"
            "Mở pipeline → YOLO Detect node → Browse vào file vừa lưu.")

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
