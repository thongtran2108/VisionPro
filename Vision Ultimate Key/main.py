"""
Vision Ultimate - Automated Optical Inspection System
Entry point
"""
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QIcon, QFont, QPalette, QColor

try:
    from ui.main_window import MainWindow
except ImportError as _imp_err:
    # cv2 / paddle được build cho numpy 1.x không load được dưới numpy 2.x
    # (ABI đổi ở numpy 2.0) → ImportError 'numpy.core.multiarray failed to
    # import'. Biến crash khó hiểu này thành hướng dẫn sửa cụ thể.
    _m = str(_imp_err).lower()
    if any(k in _m for k in ("multiarray", "numpy._core", "numpy.core",
                             "abi version", "_arraymodule")) or (
            "numpy" in _m and "import" in _m):
        sys.stderr.write(
            "\n[Vision Ultimate] Không nạp được thư viện native (OpenCV/PaddlePaddle).\n"
            "Nguyên nhân: numpy 2.x không tương thích với các gói build cho "
            "numpy 1.x (cv2, paddlepaddle...).\n"
            "Cách sửa — kích hoạt đúng venv rồi hạ numpy:\n"
            "    pip install \"numpy<2\"\n"
            f"Chi tiết gốc: {_imp_err}\n")
        sys.exit(1)
    raise


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(15, 20, 35))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(220, 230, 240))
    palette.setColor(QPalette.ColorRole.Base,            QColor(10, 14, 26))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(20, 28, 48))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(15, 20, 35))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 230, 240))
    palette.setColor(QPalette.ColorRole.Text,            QColor(220, 230, 240))
    palette.setColor(QPalette.ColorRole.Button,          QColor(20, 28, 48))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 230, 240))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(0, 212, 255))
    palette.setColor(QPalette.ColorRole.Link,            QColor(0, 212, 255))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(0, 150, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(80, 90, 110))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(80, 90, 110))
    app.setPalette(palette)

    app.setStyleSheet("""
        QToolTip {
            background: #0d1220;
            color: #e2e8f0;
            border: 1px solid #00d4ff;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }
        QScrollBar:vertical {
            background: #0a0e1a; width: 8px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #1e2d45; border-radius: 4px; min-height: 30px;
        }
        QScrollBar::handle:vertical:hover { background: #00d4ff; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: #0a0e1a; height: 8px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #1e2d45; border-radius: 4px; min-width: 30px;
        }
        QScrollBar::handle:horizontal:hover { background: #00d4ff; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        QSplitter::handle { background: #1e2d45; width: 1px; height: 1px; }
        QSplitter::handle:hover { background: #00d4ff; }
        QMenuBar {
            background: #060a14;
            color: #94a3b8;
            border-bottom: 1px solid #1e2d45;
            font-size: 13px;
        }
        QMenuBar::item:selected { background: #1a2236; color: #e2e8f0; }
        QMenu {
            background: #0d1220;
            color: #e2e8f0;
            border: 1px solid #1e2d45;
        }
        QMenu::item:selected { background: #1a2236; color: #00d4ff; }
        QMenu::separator { height: 1px; background: #1e2d45; }
        QStatusBar {
            background: #060a14;
            color: #64748b;
            border-top: 1px solid #1e2d45;
            font-family: 'Courier New';
            font-size: 11px;
        }
        QTabWidget::pane {
            border: 1px solid #1e2d45;
            background: #0d1220;
        }
        QTabBar::tab {
            background: #0a0e1a;
            color: #64748b;
            padding: 6px 14px;
            border: 1px solid #1e2d45;
            border-bottom: none;
            font-size: 12px; font-weight: 600;
        }
        QTabBar::tab:selected { background: #0d1220; color: #00d4ff; border-top: 2px solid #00d4ff; }
        QTabBar::tab:hover { color: #e2e8f0; }
        QGroupBox {
            border: 1px solid #1e2d45;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
            color: #64748b;
            font-size: 11px; font-weight: 600;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QLineEdit {
            background: #0a0e1a; border: 1px solid #1e2d45;
            color: #e2e8f0; padding: 4px 8px;
            border-radius: 4px; font-family: 'Courier New'; font-size: 12px;
        }
        QLineEdit:focus { border-color: #00d4ff; }
        QSpinBox, QDoubleSpinBox {
            background: #0a0e1a; border: 1px solid #1e2d45;
            color: #e2e8f0; padding: 3px 6px;
            border-radius: 4px; font-family: 'Courier New';
        }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #00d4ff; }
        QComboBox {
            background: #0a0e1a; border: 1px solid #1e2d45;
            color: #e2e8f0; padding: 4px 8px; border-radius: 4px;
        }
        QComboBox:focus { border-color: #00d4ff; }
        QComboBox QAbstractItemView {
            background: #0d1220; color: #e2e8f0;
            border: 1px solid #1e2d45; selection-background-color: #1a2236;
        }
        QCheckBox { color: #94a3b8; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #1e2d45; border-radius: 3px;
            background: #0a0e1a;
        }
        QCheckBox::indicator:checked {
            background: #00d4ff;
            image: none;
        }
        QSlider::groove:horizontal {
            height: 4px; background: #1e2d45; border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #00d4ff; width: 12px; height: 12px;
            border-radius: 6px; margin: -4px 0;
        }
        QSlider::sub-page:horizontal { background: #00d4ff; border-radius: 2px; }
        QProgressBar {
            background: #0a0e1a; border: 1px solid #1e2d45;
            border-radius: 3px; height: 6px; text-align: center;
        }
        QProgressBar::chunk { background: #00d4ff; border-radius: 3px; }
        QHeaderView::section {
            background: #0d1220; color: #64748b;
            border: 1px solid #1e2d45; padding: 4px 8px;
            font-size: 11px; font-weight: 600;
        }
        QTableWidget {
            background: #0a0e1a; color: #e2e8f0;
            gridline-color: #1e2d45; border: 1px solid #1e2d45;
        }
        QTableWidget::item:selected { background: #1a2236; color: #00d4ff; }
    """)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Vision Ultimate")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("VisionPro")

    # Logo T xanh → window icon (taskbar + title bar fallback).
    _logo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "logo.png")
    if os.path.isfile(_logo):
        app.setWindowIcon(QIcon(_logo))

    font = QFont("Segoe UI", 10)
    app.setFont(font)
    apply_dark_theme(app)

    # ── i18n: nạp ngôn ngữ đã lưu (vi/en/zh) trước khi dựng UI/dialog ──
    # Phải sau setOrganizationName/AppName (QSettings dựa vào đó). Đổi ngôn
    # ngữ trong app cần khởi động lại để áp dụng → đọc 1 lần ở đây là đủ.
    from core.i18n import init_language, localize_tool_registry, tr
    init_language()
    # Dịch tại chỗ name/description/label/tooltip của toàn bộ tool registry
    # (phủ mọi nơi tiêu thụ mà không phải wrap từng chỗ).
    localize_tool_registry()

    # Disable mouse wheel cho QComboBox / QSpinBox / QDoubleSpinBox.
    # Block KỂ CẢ KHI focused — user phải click vào button/arrow hoặc dùng
    # keyboard để đổi giá trị, không vô tình scroll đổi.
    from PySide6.QtWidgets import (QComboBox, QAbstractSpinBox)
    from PySide6.QtCore import QObject, QEvent

    class _NoWheelFilter(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Wheel and isinstance(
                    obj, (QComboBox, QAbstractSpinBox)):
                event.ignore()
                return True
            return False

    app._no_wheel_filter = _NoWheelFilter()
    app.installEventFilter(app._no_wheel_filter)

    # ── License gate ──────────────────────────────────────────────
    # Chưa kích hoạt (thiếu/sai/sai-máy/hết-hạn file license.key) → hiện cổng
    # Kích hoạt (Machine ID + import key). Không active → thoát hẳn.
    from core.licensing import LicenseManager
    from ui.login_dialog import ActivationDialog
    _lm = LicenseManager()
    if not _lm.is_activated():
        if ActivationDialog(_lm).exec() != QDialog.Accepted or not _lm.is_activated():
            sys.exit(0)

    # Startup flow:
    #  1. Nếu có file default (QSettings["default_aoi"]) + file còn tồn tại
    #     → bỏ qua picker, MainWindow + auto-load default.
    #  2. Nếu không có default (hoặc file đã bị xóa) → show picker để user
    #     chọn (recent / browse / new blank).
    #  3. Cancel picker → exit app.
    from ui.startup_picker import StartupAOIPicker
    from ui.loading_screen import LoadingScreen

    def _build_window(load_path: str):
        """Splash → dựng MainWindow (kèm progress) → nạp project (đồng bộ,
        splash đã lo feedback "không treo") → show → đóng splash."""
        splash = LoadingScreen()
        splash.show_message(tr("Đang khởi tạo giao diện…"), 5)
        win = MainWindow(
            on_progress=lambda pct, msg: splash.show_message(msg, pct))
        if load_path:
            splash.show_message(tr("Đang nạp dự án…"), 75)
            win.load_pipeline_from_path(
                load_path, background=False,
                progress_cb=lambda p: splash.show_message(
                    tr("Đang nạp dự án…"), 75 + int(p * 0.2)))
        splash.show_message(tr("Hoàn tất"), 100)
        win.show()
        splash.finish(win)
        return win

    default = StartupAOIPicker.get_default_path()
    if default:
        window = _build_window(default)
        sys.exit(app.exec())

    picker = StartupAOIPicker()
    if picker.exec() != QDialog.Accepted:
        sys.exit(0)

    window = _build_window(picker.chosen_path() or "")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
