"""
ui/code_editor.py
Python code editor — QPlainTextEdit subclass với:
  • Line number gutter
  • Python syntax highlighting (QSyntaxHighlighter)
  • Autocomplete (QCompleter) — keywords + builtins + custom names
  • Dark theme, monospace font
  • Auto-indent, tab → 4 spaces
"""
from __future__ import annotations
from typing import List
import builtins
import keyword

from PySide6.QtCore import Qt, QRect, QSize, QRegularExpression, QStringListModel
from PySide6.QtGui import (QColor, QFont, QPainter, QPalette, QTextCharFormat,
                            QSyntaxHighlighter, QTextCursor, QFontMetrics,
                            QKeyEvent)
from PySide6.QtWidgets import (QPlainTextEdit, QTextEdit, QWidget, QCompleter)


# ── Color palette (VS Dark+ inspired) ──────────────────────────────
C_BG          = QColor(13, 18, 32)         # editor bg
C_GUTTER_BG   = QColor(10, 14, 26)         # line-number bg
C_GUTTER_FG   = QColor(80, 90, 110)        # line-number fg
C_GUTTER_CUR  = QColor(0, 212, 255)        # current line number
C_CUR_LINE    = QColor(26, 34, 54, 90)     # current line bg highlight
C_FG          = QColor(220, 230, 240)      # default text
C_KEYWORD     = QColor(86, 156, 214)       # if/for/def/...
C_BUILTIN     = QColor(78, 201, 176)       # print/len/range/...
C_STRING      = QColor(206, 145, 120)      # "..."
C_COMMENT     = QColor(106, 153, 85)       # # comment
C_NUMBER      = QColor(181, 206, 168)      # 123, 4.5
C_DECORATOR   = QColor(220, 220, 170)      # @decorator
C_DEF         = QColor(220, 220, 170)      # function/class name after def/class
C_SELF        = QColor(86, 156, 214)       # self / cls
C_OPERATOR    = QColor(180, 180, 180)


PY_BUILTINS = sorted(set(dir(builtins)) - {"_", "__", "help"})


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter cho Python — keyword, string, number, comment, def."""

    def __init__(self, document):
        super().__init__(document)
        self._rules: List[tuple] = []

        # Keywords
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(C_KEYWORD)
        kw_fmt.setFontWeight(QFont.Bold)
        for kw in keyword.kwlist:
            self._rules.append((QRegularExpression(rf"\b{kw}\b"), kw_fmt))

        # Builtins
        bi_fmt = QTextCharFormat()
        bi_fmt.setForeground(C_BUILTIN)
        for bi in PY_BUILTINS:
            self._rules.append((QRegularExpression(rf"\b{bi}\b"), bi_fmt))

        # self / cls
        self_fmt = QTextCharFormat()
        self_fmt.setForeground(C_SELF)
        self_fmt.setFontItalic(True)
        self._rules.append((QRegularExpression(r"\bself\b"), self_fmt))
        self._rules.append((QRegularExpression(r"\bcls\b"), self_fmt))

        # Numbers (int, float, hex)
        num_fmt = QTextCharFormat()
        num_fmt.setForeground(C_NUMBER)
        self._rules.append((QRegularExpression(
            r"\b[+-]?(0[xX][0-9a-fA-F]+|\d+(\.\d+)?([eE][+-]?\d+)?)\b"), num_fmt))

        # Decorator
        dec_fmt = QTextCharFormat()
        dec_fmt.setForeground(C_DECORATOR)
        self._rules.append((QRegularExpression(r"@\w+"), dec_fmt))

        # Function / class name (after def/class)
        def_fmt = QTextCharFormat()
        def_fmt.setForeground(C_DEF)
        def_fmt.setFontWeight(QFont.Bold)
        self._rules.append((QRegularExpression(r"(?<=\bdef\s)\w+"), def_fmt))
        self._rules.append((QRegularExpression(r"(?<=\bclass\s)\w+"), def_fmt))

        # Strings — single & double quote (single-line; triple-quote not handled)
        str_fmt = QTextCharFormat()
        str_fmt.setForeground(C_STRING)
        self._rules.append((QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), str_fmt))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt))

        # Comment (# to end of line) — applied LAST to win over above rules
        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(C_COMMENT)
        self._comment_fmt.setFontItalic(True)
        self._comment_rule = QRegularExpression(r"#[^\n]*")

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)
        # Comment last to override (avoid string containing # being colored as code)
        it = self._comment_rule.globalMatch(text)
        while it.hasNext():
            m = it.next()
            self.setFormat(m.capturedStart(), m.capturedLength(),
                            self._comment_fmt)


class _LineNumberArea(QWidget):
    """Gutter widget — vẽ số dòng bên trái editor."""
    def __init__(self, editor: "CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_area_paint(event)


class CodeEditor(QPlainTextEdit):
    """Python code editor — line numbers, dark theme, autocomplete."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_font()
        self._setup_theme()
        self._gutter = _LineNumberArea(self)
        self._highlighter = PythonHighlighter(self.document())

        # Tab = 4 spaces
        fm = QFontMetrics(self.font())
        self.setTabStopDistance(4 * fm.horizontalAdvance(' '))

        # Wire gutter updates
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_gutter_width(0)
        self._highlight_current_line()

        # Completer
        self._completer = QCompleter(self)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(Qt.CaseSensitive)
        self._completer.setWrapAround(False)
        self._completer.activated.connect(self._insert_completion)
        self._completer_model = QStringListModel(self)
        self._completer.setModel(self._completer_model)
        self.set_completions([])  # initial = keywords + builtins only

    # ── Setup ────────────────────────────────────────────────────────
    def _setup_font(self):
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(11)
        self.setFont(font)

    def _setup_theme(self):
        p = self.palette()
        p.setColor(QPalette.Base, C_BG)
        p.setColor(QPalette.Text, C_FG)
        p.setColor(QPalette.Highlight, QColor(0, 120, 180))
        p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(p)
        self.setStyleSheet(
            "QPlainTextEdit {"
            f" background:{C_BG.name()};"
            f" color:{C_FG.name()};"
            " border:1px solid #1e2d45;"
            " selection-background-color:#264f78;"
            " selection-color:#ffffff;"
            "}")
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    # ── Public: autocomplete words ──────────────────────────────────
    def set_completions(self, extra_words: List[str]):
        """Set completion list = keywords + builtins + extra (input/output names)."""
        words = sorted(set(keyword.kwlist) | set(PY_BUILTINS) | set(extra_words or []))
        self._completer_model.setStringList(words)

    # ── Line number area ────────────────────────────────────────────
    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_gutter_width(self, _new_block_count: int):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int):
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(),
                                self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(),
                                       self.line_number_area_width(), cr.height()))

    def line_number_area_paint(self, event):
        painter = QPainter(self._gutter)
        painter.fillRect(event.rect(), C_GUTTER_BG)
        block = self.firstVisibleBlock()
        block_num = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        cur_block = self.textCursor().blockNumber()
        font = self.font()
        painter.setFont(font)
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                num = str(block_num + 1)
                painter.setPen(C_GUTTER_CUR if block_num == cur_block else C_GUTTER_FG)
                painter.drawText(0, int(top),
                                 self._gutter.width() - 6,
                                 self.fontMetrics().height(),
                                 Qt.AlignRight | Qt.AlignVCenter, num)
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_num += 1

    # ── Current-line highlight ──────────────────────────────────────
    def _highlight_current_line(self):
        sel = QTextEdit.ExtraSelection()
        sel.format.setBackground(C_CUR_LINE)
        sel.format.setProperty(QTextCharFormat.FullWidthSelection, True)
        sel.cursor = self.textCursor()
        sel.cursor.clearSelection()
        self.setExtraSelections([sel])

    # ── Auto-indent + tab handling ──────────────────────────────────
    def keyPressEvent(self, event: QKeyEvent):
        # Completer popup visible: forward navigation keys
        if self._completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return,
                                Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return

        # Ctrl+Space → force show completer
        if (event.key() == Qt.Key_Space
                and event.modifiers() & Qt.ControlModifier):
            self._show_completer(force=True)
            return

        # Tab key → 4 spaces (when no selection)
        if event.key() == Qt.Key_Tab and not self.textCursor().hasSelection():
            self.insertPlainText("    ")
            return

        # Enter → giữ indent của dòng trước + thêm 4 spaces nếu dòng kết bằng ':'
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            cursor = self.textCursor()
            line = cursor.block().text()
            indent = len(line) - len(line.lstrip(" "))
            extra = "    " if line.rstrip().endswith(":") else ""
            super().keyPressEvent(event)
            self.insertPlainText(" " * indent + extra)
            return

        super().keyPressEvent(event)

        # Auto popup completer khi gõ ký tự alpha-num
        if event.text() and (event.text().isalnum() or event.text() == "_"):
            self._show_completer(force=False)
        else:
            self._completer.popup().hide()

    def _current_word(self) -> str:
        cursor = self.textCursor()
        cursor.select(QTextCursor.WordUnderCursor)
        return cursor.selectedText()

    def _show_completer(self, force: bool):
        prefix = self._current_word()
        if not force and len(prefix) < 2:
            self._completer.popup().hide()
            return
        if prefix != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(prefix)
            self._completer.popup().setCurrentIndex(
                self._completer.completionModel().index(0, 0))
        cr = self.cursorRect()
        cr.setWidth(self._completer.popup().sizeHintForColumn(0)
                    + self._completer.popup().verticalScrollBar().sizeHint().width())
        self._completer.complete(cr)

    def _insert_completion(self, completion: str):
        cursor = self.textCursor()
        prefix = self._completer.completionPrefix()
        extra = len(completion) - len(prefix)
        cursor.movePosition(QTextCursor.EndOfWord)
        cursor.insertText(completion[-extra:] if extra > 0 else "")
        self.setTextCursor(cursor)
