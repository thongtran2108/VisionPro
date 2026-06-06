"""
ui/loading_screen.py — Splash / màn hình loading khởi động.

Hiện trong lúc dựng MainWindow + nạp project mặc định để app không bị
"đứng hình trắng". Cập nhật message + % qua show_message() — tự pump
QApplication.processEvents() để splash repaint ngay cả khi main thread
đang bận build UI (chưa vào event loop).
"""
from __future__ import annotations
import os
from typing import Optional

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QProgressBar,
                               QFrame, QApplication)
from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtGui import QPixmap


class LoadingScreen(QWidget):
    """Splash frameless, dark theme. Dùng:

        splash = LoadingScreen(); splash.show()
        splash.show_message("Đang khởi tạo…", 10)
        ... build / load ...
        splash.finish(main_window)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint
                            | Qt.WindowStaysOnTopHint)
        self.setFixedSize(460, 280)
        self.setObjectName("LoadingScreen")
        self.setStyleSheet("""
            #LoadingScreen{background:#0a0e1a;border:1px solid #1e2d45;}
            #ls_title{color:#00d4ff;font-size:22px;font-weight:700;
                      letter-spacing:3px;background:transparent;}
            #ls_tag{color:#475569;font-size:11px;letter-spacing:1px;
                    background:transparent;}
            #ls_status{color:#94a3b8;font-size:12px;
                       font-family:'Courier New';background:transparent;}
            QProgressBar{background:#0a0e1a;border:1px solid #1e2d45;
                         border-radius:3px;height:6px;text-align:center;}
            QProgressBar::chunk{background:#00d4ff;border-radius:3px;}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(36, 30, 36, 26)
        lay.setSpacing(10)

        # Logo (nếu có assets/logo.png — fallback bỏ qua nếu thiếu).
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(root, "assets", "logo.png")
        if os.path.isfile(logo_path):
            logo = QLabel()
            logo.setAlignment(Qt.AlignCenter)
            logo.setPixmap(QPixmap(logo_path).scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setStyleSheet("background:transparent;")
            lay.addWidget(logo)

        title = QLabel("VISION ULTIMATE")
        title.setObjectName("ls_title")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        tag = QLabel("Automated Optical Inspection")
        tag.setObjectName("ls_tag")
        tag.setAlignment(Qt.AlignCenter)
        lay.addWidget(tag)

        lay.addStretch(1)

        self._status = QLabel("Đang khởi động…")
        self._status.setObjectName("ls_status")
        self._status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        lay.addWidget(self._bar)

        self._center_on_screen()

    def _center_on_screen(self):
        scr = QApplication.primaryScreen()
        if scr:
            c = scr.availableGeometry().center()
            self.move(c.x() - self.width() // 2, c.y() - self.height() // 2)

    def show_message(self, text: str, pct: Optional[int] = None):
        """Cập nhật status + % rồi pump event để splash repaint ngay.
        ExcludeUserInputEvents → tránh re-entrancy do click khi UI đang dựng."""
        if text:
            self._status.setText(text)
        if pct is not None:
            self._bar.setValue(max(0, min(100, int(pct))))
        if not self.isVisible():
            self.show()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    def finish(self, window=None):
        """Đóng splash khi cửa sổ chính đã sẵn sàng."""
        if window is not None:
            try:
                window.raise_()
                window.activateWindow()
            except Exception:
                pass
        self.close()
        self.deleteLater()


class BusyOverlay(QWidget):
    """Lớp phủ 'đang tải' phủ lên parent — hiện trong lúc dựng dialog nặng
    (đồng bộ trên main thread, vd NodeDetailDialog / PatMaxDialog khi bấm node).

    LƯU Ý quan trọng: KHÔNG dùng spinner tự quay trong card. Widget vẽ trên
    main thread → khi main thread bận dựng dialog, event loop kẹt nên mọi
    QTimer/QMovie/QProgressBar đều đứng yên (không animate được). Chuyển động
    trực quan để báo "đang bận" do con trỏ Qt.WaitCursor của OS đảm nhiệm —
    trên Windows con trỏ vẫn quay kể cả khi app đang block. Card chỉ hiện glyph
    chờ tĩnh (⏳) + text để không trông như spinner bị đơ.

    Cách dùng:
        self._busy = BusyOverlay(main_window)   # tạo 1 lần
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._busy.show_busy("Đang mở Blob Analysis…")
        QTimer.singleShot(0, build_dialog)      # dựng ở tick kế → overlay kịp vẽ
        ... build ...
        self._busy.hide_busy(); QApplication.restoreOverrideCursor()
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("BusyOverlay")
        self.setStyleSheet("""
            #BusyOverlay{background:rgba(6,10,20,160);}
            #busy_card{background:#0d1220;border:1px solid #00d4ff;
                       border-radius:8px;}
            #busy_spin{color:#00d4ff;font-size:30px;background:transparent;}
            #busy_text{color:#e2e8f0;font-size:13px;font-weight:600;
                       background:transparent;}
        """)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("busy_card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(32, 22, 32, 22)
        cl.setSpacing(10)
        self._spin = QLabel("⏳")
        self._spin.setObjectName("busy_spin")
        self._spin.setAlignment(Qt.AlignCenter)
        self._text = QLabel("Đang tải…")
        self._text.setObjectName("busy_text")
        self._text.setAlignment(Qt.AlignCenter)
        cl.addWidget(self._spin)
        cl.addWidget(self._text)
        lay.addWidget(card)
        self.hide()

    def show_busy(self, text: str = "Đang tải…"):
        self._text.setText(text)
        p = self.parentWidget()
        if p is not None:
            self.setGeometry(p.rect())
        self.show()
        self.raise_()
        # Pump 1 nhịp để overlay + con trỏ busy vẽ ngay trước khi main thread
        # bận dựng dialog.
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    def hide_busy(self):
        self.hide()
