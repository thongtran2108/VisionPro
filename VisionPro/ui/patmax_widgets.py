"""
ui/patmax_widgets.py
Helper widgets cho PatMaxDialog — tách ra để file dialog gọn, dễ debug.
  - ModelPreviewWidget : khung hiển thị model trained (thumbnail + status)
  - ResultTable        : bảng top-K kết quả search
"""
from __future__ import annotations
from typing import List, Optional

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QColor

from core.patmax_engine import PatMaxModel, PatMaxResult


# ════════════════════════════════════════════════════════════════════
#  Model preview widget
# ════════════════════════════════════════════════════════════════════
class ModelPreviewWidget(QWidget):
    """Hiển thị thông tin model đã train."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setStyleSheet(
            "background:#0a0e1a; border:1px solid #1e2d45; border-radius:6px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(12)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(80, 80)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet(
            "background:#050810; border:1px solid #1e2d45; border-radius:4px;")
        self._thumb.setText("No Model")
        lay.addWidget(self._thumb)

        # Info
        info = QWidget()
        il = QVBoxLayout(info); il.setContentsMargins(0,0,0,0); il.setSpacing(3)
        self._name_lbl = QLabel("No model trained")
        self._name_lbl.setStyleSheet(
            "color:#64748b; font-size:13px; font-weight:700;")
        self._info1 = QLabel("")
        self._info1.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._info2 = QLabel("")
        self._info2.setStyleSheet("color:#94a3b8; font-size:11px;")
        self._hash_lbl = QLabel("")
        self._hash_lbl.setStyleSheet(
            "color:#1e2d45; font-size:10px; font-family:'Courier New';")
        il.addWidget(self._name_lbl)
        il.addWidget(self._info1)
        il.addWidget(self._info2)
        il.addWidget(self._hash_lbl)
        il.addStretch()
        lay.addWidget(info, 1)

        # Status badge
        self._badge = QLabel("UNTRAINED")
        self._badge.setFixedSize(90, 28)
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            "background:#1e2d45; color:#64748b; border-radius:4px;"
            "font-size:11px; font-weight:700; letter-spacing:1px;")
        lay.addWidget(self._badge)

    def update_model(self, model: Optional[PatMaxModel]):
        if model is None or not model.trained:
            self._name_lbl.setText("No model trained")
            self._name_lbl.setStyleSheet("color:#64748b; font-size:13px; font-weight:700;")
            self._info1.setText("")
            self._info2.setText("")
            self._hash_lbl.setText("")
            self._badge.setText("UNTRAINED")
            self._badge.setStyleSheet(
                "background:#1e2d45; color:#64748b; border-radius:4px;"
                "font-size:11px; font-weight:700;")
            self._thumb.setText("No Model")
            self._thumb.setPixmap(QPixmap())
            return

        self._name_lbl.setText("Model Trained ✔")
        self._name_lbl.setStyleSheet(
            "color:#39ff14; font-size:13px; font-weight:700;")
        x,y,w,h = model.train_roi or (0,0,0,0)
        self._info1.setText(
            f"ROI: ({x},{y})  {w}×{h} px   "
            f"Edges: {model.edge_count}")
        self._info2.setText(
            f"Origin: ({model.origin_x:.1f}, {model.origin_y:.1f})")
        self._hash_lbl.setText(f"Hash: {model.model_hash}")
        self._badge.setText("TRAINED")
        self._badge.setStyleSheet(
            "background:#1b4332; color:#39ff14; border-radius:4px;"
            "border:1px solid #39ff14; font-size:11px; font-weight:700;")

        # Thumbnail
        if model.thumbnail is not None:
            import cv2
            arr = model.thumbnail.copy()
            if len(arr.shape)==2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2]==3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h2,w2,ch = arr.shape
            qimg = QImage(arr.data.tobytes(), w2, h2, ch*w2, QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qimg).scaled(
                78, 78, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._thumb.setPixmap(pix)
        else:
            self._thumb.setText("No thumb")


# ════════════════════════════════════════════════════════════════════
#  Result table
# ════════════════════════════════════════════════════════════════════
class ResultTable(QTableWidget):
    """Bảng kết quả: mỗi row = 1 (match × point).
    Point = origin chính của pattern (j=0) + từng extra ref (j>=1).
    Vd: 1 match + 1 extra ref → 2 rows.
    """
    result_selected = Signal(int, int)   # (result_idx, ref_idx) -1=origin

    def __init__(self, parent=None):
        super().__init__(0, 7, parent)
        self.setHorizontalHeaderLabels(
            ["#", "Score", "Point", "X", "Y", "Angle (°)", "Scale"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget { background:#0a0e1a; color:#e2e8f0;
                           gridline-color:#1e2d45; border:1px solid #1e2d45;
                           font-size:11px; font-family:'Courier New'; }
            QTableWidget::item:selected { background:#1a2236; color:#00d4ff; }
            QTableWidget::item:alternate { background:#0d1220; }
            QHeaderView::section { background:#0d1220; color:#64748b;
                                   border:1px solid #1e2d45; padding:3px; }
        """)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        # Map row → (result_idx, ref_idx). ref_idx = -1: origin chính
        self._row_map: List[Tuple[int, int]] = []

    def _on_selection_changed(self):
        rows = self.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._row_map):
            res_idx, ref_idx = self._row_map[r]
            self.result_selected.emit(res_idx, ref_idx)

    def populate(self, results: List[PatMaxResult], model=None):
        from core.patmax_engine import transform_ref_to_image
        self.setRowCount(0)
        self._row_map = []
        extras = list(getattr(model, "extra_refs", []) or []) if model else []

        row = 0
        for i, r in enumerate(results):
            score_col = ("#39ff14" if r.score >= 0.7
                         else "#ffd700" if r.score >= 0.5 else "#ff3860")

            # Hàng 0 cho mỗi match: Origin chính
            self.insertRow(row)
            self._row_map.append((i, -1))
            self._fill_row(row, str(i+1), r.score, "Origin",
                            r.origin_x, r.origin_y, r.angle, r.scale, score_col)
            row += 1

            # Mỗi extra ref → 1 row
            for j, ref in enumerate(extras, start=1):
                try:
                    ex, ey, eang = transform_ref_to_image(model, ref, r)
                except Exception:
                    continue
                nm = str(ref.get("name", f"Ref {j}"))
                self.insertRow(row)
                self._row_map.append((i, j - 1))
                # Score cell trống cho ref rows (chỉ origin show score)
                self._fill_row(row, "", None, nm,
                                ex, ey, eang, r.scale, score_col,
                                muted=True)
                row += 1

    def _fill_row(self, row, idx_txt, score, point_name,
                   x, y, angle, scale, score_col, muted: bool = False):
        cells = [
            (idx_txt,                    "{}"),
            ("" if score is None else score, "{:.4f}"),
            (point_name,                 "{}"),
            (x,                          "{:.2f}"),
            (y,                          "{:.2f}"),
            (angle,                      "{:+.2f}"),
            (scale,                      "{:.3f}"),
        ]
        for j, (val, fmt) in enumerate(cells):
            if isinstance(val, str):
                txt = val
            elif val == "":
                txt = ""
            else:
                txt = fmt.format(val)
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignCenter)
            if j == 1 and not muted:
                item.setForeground(QColor(score_col))
            elif muted:
                item.setForeground(QColor("#94a3b8"))
            self.setItem(row, j, item)
