"""
ui/patmax_align_panel.py
Phần UI riêng của PatMax Align Tool — tách khỏi PatMaxDialog cho dễ debug.

Cung cấp:
  - build_align_panel(node, parent_layout)
        Build group "PatMax Align Tool" với 2 dropdown Algorithm + Train Mode,
        gắn vào parent_layout. Trả về (algorithm_combo, train_mode_combo).
        Cả hai sync 2-chiều với node.params["algorithm"] / ["train_mode"].

  - precompute_for_align(model, algorithm, train_mode_align,
                         angle_low, angle_high, angle_step,
                         scale_low, scale_high, scale_step)
        Auto-precompute oriented templates ngay sau Train (chỉ cho
        train_mode == "Shape Models with Transform"). Trả số templates.
"""
from __future__ import annotations
from typing import Tuple

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QComboBox, QGroupBox)


ALGORITHM_CHOICES = [
    "PatMax",
    "PatQuick",
    "PatMax & PatQuick",
    "PatFlex",
    "PatMax - High Sensitivity",
    "Perspective PatMax",
]
TRAIN_MODE_CHOICES = [
    "Image",
    "Shape Models with Image",
    "Shape Models with Transform",
]
ALGORITHM_DEFAULT  = "PatQuick"
TRAIN_MODE_DEFAULT = "Image"

_COMBO_QSS = (
    "QComboBox{background:#0a0e1a;border:1px solid #1e2d45;"
    "color:#e2e8f0;padding:3px 6px;border-radius:4px;}"
    "QComboBox QAbstractItemView{background:#0d1220;color:#e2e8f0;"
    "selection-background-color:#1a2236;}"
)


def _row(label_text: str, combo: QComboBox) -> QWidget:
    w = QWidget(); h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
    lbl = QLabel(label_text)
    lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
    lbl.setMinimumWidth(80)
    h.addWidget(lbl); h.addWidget(combo, 1)
    return w


def build_align_panel(node, parent_layout) -> Tuple[QComboBox, QComboBox]:
    """
    Add group "PatMax Align Tool" (Algorithm + Train Mode) vào parent_layout.

    Đồng bộ 2-chiều với node.params:
      - khôi phục giá trị đã lưu khi mở lại dialog
      - update node.params khi user thay đổi
    """
    grp = QGroupBox("PatMax Align Tool")
    g = QVBoxLayout(grp)
    g.setContentsMargins(10, 22, 10, 10); g.setSpacing(6)

    # Algorithm
    algo_combo = QComboBox()
    algo_combo.addItems(ALGORITHM_CHOICES)
    cur_algo = node.params.get("algorithm", ALGORITHM_DEFAULT)
    if cur_algo in ALGORITHM_CHOICES:
        algo_combo.setCurrentText(cur_algo)
    else:
        algo_combo.setCurrentText(ALGORITHM_DEFAULT)
    algo_combo.setStyleSheet(_COMBO_QSS)
    algo_combo.currentTextChanged.connect(
        lambda t: node.params.__setitem__("algorithm", t))
    g.addWidget(_row("Algorithm:", algo_combo))

    # Train Mode
    tm_combo = QComboBox()
    tm_combo.addItems(TRAIN_MODE_CHOICES)
    cur_tm = node.params.get("train_mode", TRAIN_MODE_DEFAULT)
    if cur_tm in TRAIN_MODE_CHOICES:
        tm_combo.setCurrentText(cur_tm)
    else:
        tm_combo.setCurrentText(TRAIN_MODE_DEFAULT)
    tm_combo.setStyleSheet(_COMBO_QSS)
    tm_combo.currentTextChanged.connect(
        lambda t: node.params.__setitem__("train_mode", t))
    g.addWidget(_row("Train Mode:", tm_combo))

    parent_layout.addWidget(grp)
    return algo_combo, tm_combo


def precompute_for_align(model,
                         algorithm: str,
                         train_mode_align: str,
                         angle_low: float, angle_high: float, angle_step: float,
                         scale_low: float, scale_high: float, scale_step: float
                         ) -> int:
    """
    Auto-precompute oriented templates cho path 'Shape Models with Transform'.
    Trả về số templates đã build; 0 nếu không cần / không build được.
    """
    if train_mode_align != "Shape Models with Transform":
        return 0
    try:
        from core.patmax_engine import (_ensure_precomputed,
                                         _shape_model_pair,
                                         ALGO_WEIGHTS)
    except ImportError as e:
        print(f"[PatMax Align] precompute import error: {e}")
        return 0
    weights = ALGO_WEIGHTS.get(algorithm, ALGO_WEIGHTS["PatMax"])
    pg, pe = _shape_model_pair(model, train_mode_align)
    if pg is None or pe is None:
        return 0
    tmpls = _ensure_precomputed(
        model, weights, pg, pe,
        angle_low, angle_high, angle_step,
        scale_low, scale_high, scale_step)
    return len(tmpls)
