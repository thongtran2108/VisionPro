"""
ui/results_panel.py
Panel kết quả tổng hợp — hiển thị PASS/FAIL, thống kê, log.
"""
from __future__ import annotations
from typing import Dict, Any, List
import time

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QTableWidget, QTableWidgetItem,
                                QHeaderView, QScrollArea, QFrame, QSizePolicy,
                                QProgressBar, QTextEdit)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QBrush

from core.flow_graph import FlowGraph


class StatCard(QWidget):
    def __init__(self, label: str, value: str, color: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)
        self.setStyleSheet(f"""
            QWidget {{
                background: #111827;
                border: 1px solid {color}44;
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        self._val_lbl = QLabel(value)
        self._val_lbl.setStyleSheet(f"color:{color}; font-size:22px; font-weight:700; "
                                    f"font-family:'Courier New'; background:transparent; border:none;")
        self._lbl_lbl = QLabel(label)
        self._lbl_lbl.setStyleSheet("color:#64748b; font-size:10px; font-weight:600; "
                                    "letter-spacing:1px; background:transparent; border:none;")
        lay.addWidget(self._val_lbl)
        lay.addWidget(self._lbl_lbl)

    def set_value(self, v: str):
        self._val_lbl.setText(v)


class ResultsPanel(QWidget):
    """Bottom panel — kết quả pipeline, log, thống kê."""
    clear_log_req = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMaximumHeight(320)

        self._pass_count  = 0
        self._fail_count  = 0
        self._total_count = 0
        self._log_entries: List[dict] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Header row ──────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        hdr.setFixedHeight(36)
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 12, 0)
        hl.setSpacing(12)

        title = QLabel("▣  INSPECTION RESULTS")
        title.setStyleSheet("color:#00d4ff; font-size:11px; font-weight:700; letter-spacing:2px;")
        hl.addWidget(title)
        hl.addStretch()

        self._clear_btn = QPushButton("Clear Log")
        self._clear_btn.setFixedHeight(24)
        self._clear_btn.setStyleSheet("""
            QPushButton{background:#1e2d45;border:none;border-radius:3px;
                        color:#94a3b8;font-size:11px;padding:0 10px;}
            QPushButton:hover{background:#00d4ff;color:#000;}
        """)
        self._clear_btn.clicked.connect(self._clear_log)
        hl.addWidget(self._clear_btn)
        lay.addWidget(hdr)

        # ── Content ─────────────────────────────────────────────────
        content = QWidget()
        cl = QHBoxLayout(content)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(12)
        lay.addWidget(content)

        # Left — stat cards
        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        cards_row = QHBoxLayout()
        self._card_pass  = StatCard("PASS",  "0", "#39ff14")
        self._card_fail  = StatCard("FAIL",  "0", "#ff3860")
        self._card_total = StatCard("TOTAL", "0", "#00d4ff")
        cards_row.addWidget(self._card_pass)
        cards_row.addWidget(self._card_fail)
        cards_row.addWidget(self._card_total)
        ll.addLayout(cards_row)

        # Yield bar
        yield_lbl = QLabel("Yield Rate")
        yield_lbl.setStyleSheet("color:#64748b; font-size:10px;")
        ll.addWidget(yield_lbl)
        self._yield_bar = QProgressBar()
        self._yield_bar.setRange(0, 100)
        self._yield_bar.setValue(0)
        self._yield_bar.setFormat("%p%")
        self._yield_bar.setFixedHeight(18)
        self._yield_bar.setStyleSheet("""
            QProgressBar{background:#0a0e1a;border:1px solid #1e2d45;
                         border-radius:3px;color:#e2e8f0;font-size:11px;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                 stop:0 #ff3860,stop:0.5 #ffd700,stop:1 #39ff14);}
        """)
        ll.addWidget(self._yield_bar)

        # Duration
        self._dur_lbl = QLabel("Last run: —")
        self._dur_lbl.setStyleSheet("color:#1e2d45; font-size:10px; font-family:'Courier New';")
        ll.addWidget(self._dur_lbl)
        ll.addStretch()
        cl.addWidget(left)

        # Sep
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#1e2d45;")
        cl.addWidget(sep)

        # Right — log table
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        log_title = QLabel("Execution Log")
        log_title.setStyleSheet("color:#64748b; font-size:10px; font-weight:700; letter-spacing:1px;")
        rl.addWidget(log_title)

        self._log_table = QTableWidget(0, 4)
        self._log_table.setHorizontalHeaderLabels(["Node", "Status", "Duration", "Output"])
        self._log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._log_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._log_table.verticalHeader().setVisible(False)
        self._log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._log_table.setAlternatingRowColors(True)
        self._log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._log_table.setStyleSheet("""
            QTableWidget{background:#0a0e1a;color:#e2e8f0;
                         gridline-color:#1e2d45;border:1px solid #1e2d45;
                         font-size:11px; font-family:'Courier New';}
            QTableWidget::item:selected{background:#1a2236;color:#00d4ff;}
            QTableWidget::item:alternate{background:#0d1220;}
        """)
        rl.addWidget(self._log_table)
        cl.addWidget(right, 1)

    # ── Public API ────────────────────────────────────────────────
    def report_run(self, graph: FlowGraph, results: Dict[str, Any], duration_ms: float):
        """Called after pipeline execution."""
        self._pass_count  = 0
        self._fail_count  = 0
        self._total_count = len(results)
        self._log_table.setRowCount(0)

        for nid, res in results.items():
            node   = graph.nodes.get(nid)
            status = res.get("status", "idle")
            if status == "pass":
                self._pass_count += 1
            elif status in ("fail", "error"):
                self._fail_count += 1

            # Table row
            row = self._log_table.rowCount()
            self._log_table.insertRow(row)

            name_item = QTableWidgetItem(node.tool.name if node else nid)
            name_item.setForeground(QBrush(QColor("#e2e8f0")))

            status_colors = {
                "pass": "#39ff14", "fail": "#ff3860",
                "error": "#ff3860", "idle": "#64748b", "running": "#ffd700"
            }
            sc = status_colors.get(status, "#64748b")
            st_item = QTableWidgetItem(status.upper())
            st_item.setForeground(QBrush(QColor(sc)))
            st_item.setFont(QFont("Courier New", 10, QFont.Bold))

            # Per-node elapsed_ms từ FlowGraph.execute; fallback về total
            # nếu node cũ chưa có field.
            node_ms = res.get("elapsed_ms")
            if node_ms is None and node is not None:
                node_ms = getattr(node, "last_run_ms", 0.0)
            if node_ms is None:
                node_ms = duration_ms
            dur_item = QTableWidgetItem(f"{node_ms:.0f}ms")
            # Tô đỏ nếu > 200ms để dễ thấy bottleneck
            dur_color = "#ff8a4d" if node_ms > 200 else "#64748b"
            dur_item.setForeground(QBrush(QColor(dur_color)))

            outputs = res.get("outputs", {})
            out_parts = []
            for k, v in outputs.items():
                if k == "image":
                    continue
                if isinstance(v, bool):
                    out_parts.append(f"{k}={'T' if v else 'F'}")
                elif isinstance(v, float):
                    out_parts.append(f"{k}={v:.4g}")
                elif isinstance(v, (int, str)):
                    out_parts.append(f"{k}={v}")
            out_item = QTableWidgetItem("  ".join(out_parts))
            out_item.setForeground(QBrush(QColor("#94a3b8")))

            self._log_table.setItem(row, 0, name_item)
            self._log_table.setItem(row, 1, st_item)
            self._log_table.setItem(row, 2, dur_item)
            self._log_table.setItem(row, 3, out_item)

        self._update_stats(duration_ms)

    def _update_stats(self, duration_ms: float):
        self._card_pass.set_value(str(self._pass_count))
        self._card_fail.set_value(str(self._fail_count))
        self._card_total.set_value(str(self._total_count))

        if self._total_count > 0:
            yield_pct = int(self._pass_count / self._total_count * 100)
        else:
            yield_pct = 0
        self._yield_bar.setValue(yield_pct)

        self._dur_lbl.setText(f"Last run: {duration_ms:.1f} ms  |  "
                              f"{time.strftime('%H:%M:%S')}")

    def _clear_log(self):
        self._log_table.setRowCount(0)
        self._pass_count = self._fail_count = self._total_count = 0
        self._card_pass.set_value("0")
        self._card_fail.set_value("0")
        self._card_total.set_value("0")
        self._yield_bar.setValue(0)
        self._dur_lbl.setText("Last run: —")
