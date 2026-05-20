"""
ui/tool_library.py — Cognex VisionPro style
Panel thư viện tool bên trái — categories theo Cognex.
"""
from __future__ import annotations
from typing import Dict, List

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QScrollArea,
                                QLabel, QPushButton, QFrame, QApplication)
from PySide6.QtCore import Qt, QMimeData, QPoint
from PySide6.QtGui import (QDrag, QPixmap, QPainter, QColor, QFont, QMouseEvent, QCursor)

from core.tool_registry import TOOL_REGISTRY, CATEGORIES, ToolDef

# Category accent colors (Cognex-inspired)
CAT_COLORS = {
    "Acquire Image":     "#0f3460",
    "Pattern Find":      "#16213e",
    "Fixture":           "#1a1a2e",
    "Caliper":           "#1b4332",
    "Blob Analysis":     "#2d6a4f",
    "Edge & Geometry":   "#134074",
    "Color Analysis":    "#6b2737",
    "ID & Read":         "#3d0c02",
    "Measurement":       "#134074",
    "Surface Inspection":"#4a0404",
    "Image Processing":  "#2c3e50",
    "Calibration":       "#1a472a",
    "Logic & Flow":      "#1c1c2e",
    "Output & Display":  "#0d1117",
}

CAT_ICONS = {
    "Acquire Image":     "📷",
    "Pattern Find":      "🎯",
    "Fixture":           "📌",
    "Caliper":           "📐",
    "Blob Analysis":     "🔵",
    "Edge & Geometry":   "〰",
    "Color Analysis":    "🎨",
    "ID & Read":         "📦",
    "Measurement":       "↔",
    "Surface Inspection":"🔴",
    "Image Processing":  "🔧",
    "Calibration":       "📋",
    "Logic & Flow":      "⚙",
    "Output & Display":  "🖥",
}


class ToolButton(QFrame):
    def __init__(self, tool: ToolDef, parent=None):
        super().__init__(parent)
        self.tool = tool
        self._drag_start = None
        self.setFixedHeight(52)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setObjectName("toolBtn")
        border_color = tool.color
        self.setStyleSheet(f"""
            #toolBtn {{
                background: #0d1220;
                border: 1px solid #1e2d45;
                border-left: 3px solid {border_color};
                border-radius: 5px;
            }}
            #toolBtn:hover {{
                background: #1a2236;
                border-color: #00d4ff;
                border-left: 3px solid {border_color};
            }}
        """)

        # Tooltip: name + cognex equiv + description
        tip = f"<b>{tool.name}</b>"
        if tool.cognex_equiv:
            tip += f"<br><i style='color:#00d4ff'>{tool.cognex_equiv}</i>"
        tip += f"<br>{tool.description}"
        self.setToolTip(tip)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(1)

        top_row = QLabel(f"{tool.icon}  {tool.name}")
        top_row.setStyleSheet(
            "color:#e2e8f0; font-size:12px; font-weight:600; "
            "background:transparent; border:none;")

        # Show Cognex equivalent in small text
        if tool.cognex_equiv:
            bot_row = QLabel(tool.cognex_equiv)
            bot_row.setStyleSheet(
                "color:#00d4ff; font-size:9px; font-style:italic; "
                "background:transparent; border:none;")
        else:
            bot_row = QLabel(tool.description[:42] + ("…" if len(tool.description) > 42 else ""))
            bot_row.setStyleSheet(
                "color:#64748b; font-size:10px; background:transparent; border:none;")

        lay.addWidget(top_row)
        lay.addWidget(bot_row)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if (self._drag_start and
                event.buttons() & Qt.LeftButton and
                (event.pos() - self._drag_start).manhattanLength() > 8):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.tool.tool_id)
            drag.setMimeData(mime)

            pix = QPixmap(180, 40)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(QColor(17, 24, 39, 220))
            p.setPen(QColor(0, 212, 255))
            p.drawRoundedRect(0, 0, 180, 40, 6, 6)
            p.setPen(Qt.white)
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            p.drawText(pix.rect(), Qt.AlignCenter, f"{self.tool.icon} {self.tool.name}")
            p.end()

            drag.setPixmap(pix)
            drag.setHotSpot(QPoint(90, 20))
            drag.exec(Qt.CopyAction)
            self._drag_start = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start = None
        super().mouseReleaseEvent(event)


class CategorySection(QWidget):
    def __init__(self, category: str, tools: List[ToolDef], parent=None):
        super().__init__(parent)
        self._collapsed = False
        self._tools = tools
        self._category = category
        cat_color = CAT_COLORS.get(category, "#1e2d45")
        cat_icon  = CAT_ICONS.get(category, "●")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(2)

        self._header = QPushButton(f"▾  {cat_icon}  {category}  ({len(tools)})")
        self._header.setCheckable(True)
        self._header.setStyleSheet(f"""
            QPushButton {{
                background: {cat_color}88;
                color: #94a3b8;
                border: none;
                border-bottom: 1px solid #1e2d45;
                border-left: 2px solid {cat_color};
                border-radius: 0;
                padding: 7px 12px;
                text-align: left;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ color: #00d4ff; background: {cat_color}cc; }}
            QPushButton:checked {{ color: #64748b; }}
        """)
        self._header.clicked.connect(self._toggle)
        layout.addWidget(self._header)

        self._content = QWidget()
        cl = QVBoxLayout(self._content)
        cl.setContentsMargins(4, 2, 4, 4)
        cl.setSpacing(3)
        self._buttons: List[ToolButton] = []
        for t in tools:
            btn = ToolButton(t)
            cl.addWidget(btn)
            self._buttons.append(btn)
        layout.addWidget(self._content)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        cat_icon = CAT_ICONS.get(self._category, "●")
        sym = "▸" if self._collapsed else "▾"
        self._header.setText(
            f"{sym}  {cat_icon}  {self._category}  ({len(self._tools)})")

    def filter(self, text: str):
        any_vis = False
        for btn in self._buttons:
            vis = (text.lower() in btn.tool.name.lower() or
                   text.lower() in btn.tool.description.lower() or
                   text.lower() in btn.tool.cognex_equiv.lower())
            btn.setVisible(vis)
            any_vis = any_vis or vis
        self.setVisible(any_vis or not text)


class ToolLibraryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(230)
        self.setMaximumWidth(270)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Title
        title = QWidget()
        title.setFixedHeight(44)
        title.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        tl = QVBoxLayout(title)
        tl.setContentsMargins(12, 6, 12, 6)
        t1 = QLabel("⬡  COGNEX TOOL LIBRARY")
        t1.setStyleSheet("color:#00d4ff; font-size:10px; font-weight:700; letter-spacing:2px;")
        t2 = QLabel(f"{len(TOOL_REGISTRY)} tools")
        t2.setStyleSheet("color:#1e2d45; font-size:9px;")
        tl.addWidget(t1); tl.addWidget(t2)
        lay.addWidget(title)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Search tools or Cognex name...")
        self._search.setStyleSheet("""
            QLineEdit {
                background:#0a0e1a; border:none;
                border-bottom:1px solid #1e2d45;
                color:#e2e8f0; padding:8px 12px; font-size:12px;
            }
            QLineEdit:focus { border-bottom-color:#00d4ff; }
        """)
        self._search.textChanged.connect(self._on_search)
        lay.addWidget(self._search)

        # Scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(scroll)

        container = QWidget()
        self._cl = QVBoxLayout(container)
        self._cl.setContentsMargins(0, 0, 0, 0)
        self._cl.setSpacing(0)
        scroll.setWidget(container)

        # Build by category
        by_cat: Dict[str, List[ToolDef]] = {}
        for tool in TOOL_REGISTRY:
            by_cat.setdefault(tool.category, []).append(tool)

        self._sections: List[CategorySection] = []
        for cat in CATEGORIES:
            tools = by_cat.get(cat, [])
            if tools:
                sec = CategorySection(cat, tools)
                self._cl.addWidget(sec)
                self._sections.append(sec)

        self._cl.addStretch()

    def _on_search(self, text: str):
        for sec in self._sections:
            sec.filter(text)
