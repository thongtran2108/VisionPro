"""
ui/main_window.py — v3
- Fix: double-click node → NodeDetailDialog (cửa sổ mới)
- Fix: ImageViewerPanel là tab xem ảnh chính
- Fix: nối port working
"""
from __future__ import annotations
import os, time
from typing import Optional, Callable

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QSplitter, QLabel, QPushButton, QStatusBar,
                                QFileDialog, QMessageBox, QProgressBar,
                                QProgressDialog,
                                QFrame, QTabWidget, QApplication, QDialog)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QSettings, QEvent
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QFont

from core.i18n import (tr, LANGUAGES, current_language, set_language)
from core.flow_graph import FlowGraph
from core.paths import projects_dir
from ui.canvas_view import AOICanvas
from ui.tool_library import ToolLibraryPanel
from ui.properties_panel import PropertiesPanel
from ui.results_panel import ResultsPanel
from ui.image_viewer import ImageViewerPanel
from ui.loading_screen import BusyOverlay
# NodeDetailDialog (3027 dòng) lazy-import trong _open_node_detail — không
# cần lúc khởi động, giảm thời gian import startup.
from core.plc import PLCManager, get_plc_manager


# ── Worker ────────────────────────────────────────────────────────
class PipelineWorker(QObject):
    progress = Signal(int)
    finished = Signal(dict, float)
    error    = Signal(str)

    def __init__(self, graph: FlowGraph, acquire_node_id: str = ""):
        super().__init__()
        self.graph = graph
        self.acquire_node_id = acquire_node_id or ""

    def run(self):
        try:
            self.graph.reset_status()
            t0 = time.perf_counter()
            results = self.graph.execute(
                progress_cb=self.progress.emit,
                acquire_node_id=self.acquire_node_id or None,
            )
            self.finished.emit(results, (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self.error.emit(str(e))


# ── Pipeline load worker ──────────────────────────────────────────
class PipelineLoadWorker(QObject):
    """Nạp file .aoi/.json ở thread nền — đọc file + parse JSON + dựng
    FlowGraph (decode PatMaxModel base64 nặng với file lớn) → không treo UI.
    Chỉ tạo data model; widget canvas dựng trên main thread ở finished."""
    progress = Signal(int, str)
    finished = Signal(object)   # FlowGraph
    error    = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            import json
            self.progress.emit(8, tr("Đang đọc file…"))
            with open(self.path, "r", encoding="utf-8") as f:
                self.progress.emit(28, tr("Đang phân tích JSON…"))
                data = json.load(f)
            self.progress.emit(45, tr("Đang dựng pipeline…"))
            graph = FlowGraph.from_dict(
                data,
                progress_cb=lambda p: self.progress.emit(
                    45 + int(p * 0.5), tr("Đang dựng node…")))
            self.progress.emit(100, tr("Hoàn tất"))
            self.finished.emit(graph)
        except Exception as e:
            self.error.emit(str(e))


# ── Toolbar button styles ─────────────────────────────────────────
_TB_STYLE = """
    QPushButton{background:#111827;border:1px solid #1e2d45;border-radius:5px;
                color:#94a3b8;font-size:12px;font-weight:600;padding:0 12px;}
    QPushButton:hover{background:#1a2236;border-color:#00d4ff;color:#e2e8f0;}
    QPushButton:pressed{background:#0d1a2a;}
"""
_RUN_STYLE = """
    QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #006688,stop:1 #009bbb);
                border:none;border-radius:5px;color:#000;
                font-size:13px;font-weight:700;letter-spacing:1px;}
    QPushButton:hover{background:#00d4ff;}
"""
_STOP_STYLE = """
    QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #8b1a2a,stop:1 #cc1a3a);
                border:none;border-radius:5px;color:#fff;
                font-size:13px;font-weight:700;letter-spacing:1px;}
    QPushButton:hover{background:#ff3860;}
"""
# Login button — đỏ-cam khi đang KHOÁ (chưa login), xanh khi đã mở khoá.
_LOGIN_OFF_STYLE = (
    "QPushButton{background:#2a1015;border:1px solid #ff3860;border-radius:5px;"
    "color:#ff6b81;font-size:12px;font-weight:700;padding:0 12px;}"
    "QPushButton:hover{background:#ff3860;color:#fff;}")
_LOGIN_ON_STYLE = (
    "QPushButton{background:#0a2a18;border:1px solid #39ff14;border-radius:5px;"
    "color:#39ff14;font-size:12px;font-weight:700;padding:0 12px;}"
    "QPushButton:hover{background:#143a22;color:#7dff5a;}")


def _tb_btn(icon: str, label: str, tip: str = "") -> QPushButton:
    b = QPushButton(f"{icon}  {label}")
    b.setToolTip(tip)
    b.setFixedHeight(34)
    b.setStyleSheet(_TB_STYLE)
    return b


def _sep() -> QFrame:
    s = QFrame()
    s.setFrameShape(QFrame.VLine)
    s.setStyleSheet("color:#1e2d45;")
    return s


class LockOverlay(QWidget):
    """Lớp phủ mờ đặt trên 1 panel khi CHƯA đăng nhập. Mục đích: panel khoá
    mà im lặng → user không hiểu sao bấm không được. Overlay vừa cho thấy rõ
    panel đang khoá (mờ + dòng nhắc), vừa CHẶN tương tác và phát `clicked` để
    MainWindow hiện thông báo/mời đăng nhập. Tự bám kích thước panel cha."""
    clicked = Signal()

    def __init__(self, target: QWidget, hint: str = ""):
        super().__init__(target)
        self._target = target
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: rgba(6,10,20,0.55);")
        self._lbl = QLabel(hint, self)
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setWordWrap(True)
        self._lbl.setStyleSheet(
            "color:#e2e8f0; background:rgba(13,18,32,0.95);"
            "border:1px solid #00d4ff; border-radius:8px; padding:10px 14px;"
            "font-size:12px; font-weight:600;")
        target.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, ev):
        # Panel cha resize/show → overlay bám theo để luôn phủ kín.
        if obj is self._target and ev.type() in (QEvent.Resize, QEvent.Show):
            self._relayout()
        return False

    def _relayout(self):
        self.setGeometry(self._target.rect())
        self._lbl.setFixedWidth(max(140, min(self.width() - 24, 220)))
        self._lbl.adjustSize()
        self._lbl.move(max(0, (self.width() - self._lbl.width()) // 2),
                       max(0, (self.height() - self._lbl.height()) // 2))

    def showEvent(self, ev):
        self._relayout()
        self.raise_()
        super().showEvent(ev)

    def mousePressEvent(self, ev):
        self.clicked.emit()
        ev.accept()


# ── Main Window ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, on_progress: Optional[Callable[[int, str], None]] = None):
        super().__init__()
        # on_progress(pct, msg) — splash khởi động hook vào đây để hiện tiến
        # độ dựng UI. None = chạy bình thường (vd khi test, không splash).
        self._on_progress = on_progress
        self.setWindowTitle("NEO Vision Pro  v1.0")
        self.resize(1560, 940)
        self.setMinimumSize(1100, 700)
        # Window icon = logo T xanh (assets/logo.png). Title bar + taskbar.
        _logo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "assets", "logo.png")
        if os.path.isfile(_logo):
            from PySide6.QtGui import QIcon as _QIcon
            self.setWindowIcon(_QIcon(_logo))

        self._graph         = FlowGraph()
        # Expose graph cho proc cross-node (Save Image dựng ảnh composite).
        from core.tool_registry import EXEC_CTX
        EXEC_CTX.graph = self._graph
        self._current_file: Optional[str] = None
        # Dirty tracking — flip True khi graph hoặc graph.ui_state thay đổi.
        # Reset False sau save/load/new. Khi close mà dirty → prompt save.
        self._dirty: bool = False
        self._worker_thread: Optional[QThread] = None
        self._is_running    = False
        self._detail_dialogs: dict = {}   # node_id → NodeDetailDialog

        # Đăng nhập admin: chưa login → khoá Sửa (canvas/params/tool library)
        # và Chạy tool. Login (mặc định VisionUltimate/8888) mở khoá. Trạng thái login
        # chỉ trong phiên chạy, không lưu đĩa.
        from core.auth import AuthManager
        self._auth = AuthManager()

        # PLC integration — persistent manager, shared with PLCDialog
        self._plc_manager = get_plc_manager()   # shared — node PLC Write dùng chung
        self._plc_dialog = None
        # Trigger route acquire_node_id cho lần run kế tiếp (set bởi
        # _on_plc_trigger, clear bởi _finalize_run). Rỗng = chạy toàn pipeline.
        self._plc_route_acquire: str = ""

        # SFC/MES integration — scanner + API GET + API POST
        from core.sfc import SfcManager
        self._sfc_manager = SfcManager()
        # Save Image (naming_mode = "Mã scan") đọc EXEC_CTX.sfc.last_sn.
        EXEC_CTX.sfc = self._sfc_manager
        self._sfc_dialog = None
        self._sfc_pending_post = False  # set khi PLC trigger có sequence POST

        # Single-instance cache cho dialog nặng (lazy-build): mở lần đầu mới
        # dựng, đang mở thì bring-to-front thay vì tạo bản sao.
        self._yolo_dialog = None
        # Background loader (.aoi) — thread + worker + progress dialog.
        self._load_thread: Optional[QThread] = None
        self._load_worker = None
        self._load_dialog = None
        self._loading_path: str = ""

        self._emit_progress(15, tr("Đang dựng giao diện…"))
        self._build_ui()
        self._emit_progress(50, tr("Đang tạo menu…"))
        self._build_menu()
        self._emit_progress(62, tr("Đang kết nối tín hiệu…"))
        self._connect_signals()
        self._restore_state()
        # Khởi động ở trạng thái CHƯA đăng nhập → khoá Sửa & Chạy tool.
        self._apply_auth_state()
        self._emit_progress(70, tr("Giao diện sẵn sàng"))

        # Overlay 'đang mở…' hiện khi bấm node để dựng dialog nặng (lazy-build).
        self._busy = BusyOverlay(self)
        self._detail_pending: set = set()  # node_id đang dựng dialog (chống trùng)
        self._heavy_pending: set = set()   # label dialog toolbar đang dựng

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._tick)
        self._status_timer.start(800)

        # Default = Full Image View khi mở app. Ẩn tool library, toolbar,
        # tab bar, properties → user thấy ảnh chiếm toàn màn hình ngay.
        # F11 hoặc Esc hoặc View → Full Image View để thoát.
        QTimer.singleShot(0, lambda: self._act_full_view.setChecked(True))
        # Prefetch import các dialog nặng ở thread nền (sau khi UI đã hiện) →
        # lần đầu bấm node / mở YOLO Studio dựng nhanh hơn, bớt kẹt main thread.
        QTimer.singleShot(1200, self._prewarm_heavy_dialogs)

    def _prewarm_heavy_dialogs(self):
        """Prefetch import module dialog nặng ở thread nền. Import chỉ parse
        module + define class (KHÔNG tạo widget) → an toàn chạy ngoài main
        thread; lần đầu mở dialog đỡ phải trả phí import → bớt kẹt."""
        import threading

        def _work():
            for mod in ("ui.node_detail_dialog", "ui.patmax_dialog",
                        "ui.yolo_studio", "ui.plc_dialog", "ui.sfc_dialog",
                        "ui.camera_dialog"):
                try:
                    __import__(mod)
                except Exception:
                    pass
        threading.Thread(target=_work, daemon=True, name="prewarm").start()

    def _emit_progress(self, pct: int, msg: str):
        """Báo tiến độ cho splash khởi động (no-op nếu không có on_progress)."""
        if self._on_progress:
            try:
                self._on_progress(pct, msg)
            except Exception:
                pass

    # ── UI BUILD ─────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toolbar = self._build_toolbar()
        root.addWidget(self._toolbar)

        # ── Outer horizontal split: [library | center | props] ───
        outer = QSplitter(Qt.Horizontal)
        outer.setHandleWidth(1)
        outer.setStyleSheet("QSplitter::handle{background:#1e2d45;}")
        root.addWidget(outer, 1)

        # Left — tool library
        self._tool_lib = ToolLibraryPanel()
        outer.addWidget(self._tool_lib)

        # Center — vertical split: [canvas+viewer | results]
        center_split = QSplitter(Qt.Vertical)
        center_split.setHandleWidth(1)
        center_split.setStyleSheet("QSplitter::handle{background:#1e2d45;}")

        # Top center — tabs: Canvas | Image Viewer
        self._center_tabs = QTabWidget()
        self._center_tabs.setStyleSheet("""
            QTabWidget::pane{border:none;background:#0a0e1a;}
            QTabBar::tab{background:#060a14;color:#64748b;
                         padding:7px 16px;border:none;
                         font-size:12px;font-weight:600;border-right:1px solid #1e2d45;}
            QTabBar::tab:selected{color:#00d4ff;background:#0a0e1a;
                                  border-bottom:2px solid #00d4ff;}
            QTabBar::tab:hover{color:#e2e8f0;}
        """)

        self._canvas = AOICanvas(self._graph)
        self._center_tabs.addTab(self._canvas, tr("🔧  Pipeline Canvas"))

        self._img_viewer = ImageViewerPanel()
        self._img_viewer.set_graph(self._graph)
        # Image-viewer ui state (overlay ticks, multi-view mode/custom views)
        # → mark pipeline dirty để prompt save khi close.
        self._img_viewer.state_changed.connect(self._mark_dirty)
        self._center_tabs.addTab(self._img_viewer, tr("👁  Image Viewer"))
        # Default tab = Image Viewer (user mở app lên là thấy ảnh, không
        # phải pipeline canvas — workflow inspection ưu tiên view kết quả).
        self._center_tabs.setCurrentWidget(self._img_viewer)

        # Properties panel chứa tabs Info/Params/Output/Preview liên quan tới
        # node — không cần thiết khi user đang xem ảnh. Ẩn panel khi sang tab
        # Image Viewer, show lại khi quay về Pipeline Canvas.
        self._center_tabs.currentChanged.connect(self._on_center_tab_changed)

        center_split.addWidget(self._center_tabs)

        # Bottom — results
        self._results = ResultsPanel()
        center_split.addWidget(self._results)
        center_split.setSizes([680, 200])
        outer.addWidget(center_split)

        # Right — properties
        self._props = PropertiesPanel()
        self._props.set_graph(self._graph)
        outer.addWidget(self._props)

        outer.setSizes([240, 1020, 280])
        outer.setCollapsible(0, False)
        outer.setCollapsible(2, False)

        # Lock overlays — phủ Thư viện Tool & Properties khi chưa đăng nhập:
        # cho thấy rõ panel đang khoá + bắt click để báo cần đăng nhập (thay
        # vì disable im lặng khiến user không hiểu sao bấm không được).
        self._lock_overlays = []
        for _panel, _hint in (
                (self._tool_lib, tr("🔒 Đăng nhập để dùng Thư viện Tool")),
                (self._props,    tr("🔒 Đăng nhập để sửa tham số"))):
            _ov = LockOverlay(_panel, _hint)
            _ov.clicked.connect(lambda: self._notify_edit_locked(prompt=True))
            self._lock_overlays.append(_ov)

    def _build_toolbar(self) -> QWidget:
        tb = QWidget()
        tb.setFixedHeight(50)
        tb.setStyleSheet("background:#060a14; border-bottom:1px solid #1e2d45;")
        hl = QHBoxLayout(tb)
        hl.setContentsMargins(12, 8, 12, 8)
        hl.setSpacing(6)

        # Logo T xanh + text. Pixmap thay cho emoji ⬡ — fallback giữ text
        # nếu file logo thiếu để app vẫn chạy.
        from PySide6.QtGui import QPixmap
        _logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets", "logo.png")
        if os.path.isfile(_logo_path):
            _pm = QPixmap(_logo_path).scaled(
                28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_icon = QLabel()
            logo_icon.setPixmap(_pm)
            logo_icon.setStyleSheet("background:transparent;")
            hl.addWidget(logo_icon)
        logo = QLabel("Vision Ultimate")
        logo.setStyleSheet("color:#00d4ff;font-size:15px;font-weight:700;letter-spacing:2px;")
        hl.addWidget(logo)
        hl.addWidget(_sep())

        self._btn_new   = _tb_btn("📄", tr("New"),  "Ctrl+N")
        self._btn_open  = _tb_btn("📂", tr("Open"), "Ctrl+O")
        self._btn_save  = _tb_btn("💾", tr("Save"), "Ctrl+S")
        for b in (self._btn_new, self._btn_open, self._btn_save):
            hl.addWidget(b)

        hl.addWidget(_sep())
        self._btn_fit   = _tb_btn("⊡", tr("Fit"),   tr("Fit canvas"))
        self._btn_zoom1 = _tb_btn("1:1", tr("Zoom"), tr("Reset zoom"))
        self._btn_clear = _tb_btn("🗑", tr("Clear"), tr("Clear all"))
        for b in (self._btn_fit, self._btn_zoom1, self._btn_clear):
            hl.addWidget(b)

        hl.addStretch()

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedSize(160, 20)
        self._progress.hide()
        self._progress.setStyleSheet("""
            QProgressBar{background:#0a0e1a;border:1px solid #1e2d45;
                         border-radius:3px;color:#00d4ff;font-size:10px;text-align:center;}
            QProgressBar::chunk{background:#00d4ff;border-radius:3px;}
        """)
        hl.addWidget(self._progress)

        self._node_count = QLabel(f"0 {tr('nodes')}")
        self._node_count.setStyleSheet("color:#1e2d45;font-size:11px;font-family:'Courier New';")
        hl.addWidget(self._node_count)
        hl.addWidget(_sep())

        self._run_btn = QPushButton(tr("▶   RUN"))
        self._run_btn.setFixedSize(110, 34)
        self._run_btn.setStyleSheet(_RUN_STYLE)
        hl.addWidget(self._run_btn)

        sep_y = QFrame(); sep_y.setFrameShape(QFrame.VLine)
        sep_y.setStyleSheet("color:#1e2d45;")
        hl.addWidget(sep_y)

        self._btn_yolo = QPushButton("🤖  YOLO")
        self._btn_yolo.setFixedHeight(34)
        self._btn_yolo.setStyleSheet(
            "QPushButton{background:#1a0a3a;border:1px solid #9b59b6;"
            "border-radius:5px;color:#9b59b6;"
            "font-size:12px;font-weight:700;padding:0 12px;}"
            "QPushButton:hover{background:#9b59b6;color:#fff;}")
        self._btn_yolo.clicked.connect(self._open_yolo_studio)
        hl.addWidget(self._btn_yolo)

        sep_l = QFrame(); sep_l.setFrameShape(QFrame.VLine)
        sep_l.setStyleSheet("color:#1e2d45;")
        hl.addWidget(sep_l)

        # Ngôn ngữ đổi qua menu “🌐 Language” trên menu bar (không đặt combo
        # cạnh nút đăng nhập nữa cho gọn toolbar).

        # Login/Logout — khoá/mở quyền Sửa & Chạy tool. Bấm khi đang khoá →
        # mở dialog đăng nhập; đang mở khoá → đăng xuất.
        self._btn_login = QPushButton(tr("🔒  Đăng nhập"))
        self._btn_login.setFixedHeight(34)
        self._btn_login.setCursor(Qt.PointingHandCursor)
        self._btn_login.clicked.connect(self._on_login_button)
        hl.addWidget(self._btn_login)
        return tb

    def _change_language(self, code: str):
        """Đổi ngôn ngữ: lưu lựa chọn vào QSettings + nhắc khởi động lại app.
        Không retranslate runtime — đơn giản & chắc chắn. Đồng bộ menu Language
        để phản ánh lựa chọn (dù app chưa restart)."""
        if not code or code == current_language():
            return
        set_language(code, persist=True)
        for a in getattr(self, "_lang_actions", []):
            a.setChecked(a.data() == code)
        # tr() đã dùng ngôn ngữ mới → hộp thoại hiện đúng ngôn ngữ vừa chọn.
        QMessageBox.information(
            self, tr("Đổi ngôn ngữ"),
            tr("Ngôn ngữ sẽ áp dụng sau khi khởi động lại ứng dụng."))

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu(tr("File"))
        for label, shortcut, slot in [
            ("New Pipeline",   "Ctrl+N", self._new_pipeline),
            ("Open Pipeline…", "Ctrl+O", self._open_pipeline),
            ("Save Pipeline",  "Ctrl+S", self._save_pipeline),
            ("Save As…", "Ctrl+Shift+S", self._save_as),
        ]:
            a = file_m.addAction(tr(label))
            a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
        file_m.addSeparator()
        a_switch = file_m.addAction(tr("Switch Project…"))
        a_switch.setShortcut(QKeySequence("Ctrl+Shift+P"))
        a_switch.setToolTip(tr("Mở dialog Recent files → pick project khác."))
        a_switch.triggered.connect(self._switch_project)
        file_m.addSeparator()
        q = file_m.addAction(tr("Quit"))
        q.setShortcut(QKeySequence("Ctrl+Q"))
        q.triggered.connect(self.close)

        run_m = mb.addMenu(tr("Run"))
        a5 = run_m.addAction(tr("Run Pipeline")); a5.setShortcut("F5")
        a5.triggered.connect(self._toggle_run)
        a6 = run_m.addAction(tr("Stop"));         a6.setShortcut("F6")
        a6.triggered.connect(self._stop_run)

        view_m = mb.addMenu(tr("View"))
        af = view_m.addAction(tr("Fit Canvas")); af.setShortcut("F")
        af.triggered.connect(lambda: self._canvas.zoom_fit())
        ar = view_m.addAction(tr("Reset Zoom")); ar.setShortcut("R")
        ar.triggered.connect(lambda: self._canvas.zoom_reset())
        view_m.addSeparator()
        av = view_m.addAction(tr("Switch to Image Viewer")); av.setShortcut("Tab")
        av.triggered.connect(lambda: self._center_tabs.setCurrentIndex(
            1 if self._center_tabs.currentIndex() == 0 else 0))
        view_m.addSeparator()
        self._act_full_view = view_m.addAction(tr("Full Image View"))
        self._act_full_view.setShortcut("F11")
        self._act_full_view.setCheckable(True)
        self._act_full_view.toggled.connect(self._toggle_full_image_view)

        yolo_m = mb.addMenu("YOLO")
        act_yolo = yolo_m.addAction(tr("🤖 Open YOLO Studio"))
        act_yolo.setShortcut("Ctrl+Y")
        act_yolo.triggered.connect(self._open_yolo_studio)
        yolo_m.addSeparator()
        act_yolo_label = yolo_m.addAction(tr("✏ Label Images..."))
        act_yolo_label.triggered.connect(self._open_yolo_studio)

        tools_m = mb.addMenu(tr("Tools"))
        act_plc = tools_m.addAction(tr("🔌  PLC Connection…"))
        act_plc.setShortcut("Ctrl+P")
        act_plc.triggered.connect(self._open_plc_dialog)
        act_sfc = tools_m.addAction(tr("🌐  SFC / MES Integration…"))
        act_sfc.setShortcut("Ctrl+Shift+S")
        act_sfc.triggered.connect(self._open_sfc_dialog)
        act_cam = tools_m.addAction(tr("📷  Camera Setup…"))
        act_cam.setShortcut("Ctrl+Shift+C")
        act_cam.triggered.connect(self._open_camera_dialog)

        acct_m = mb.addMenu(tr("Account"))
        self._act_login = acct_m.addAction(tr("🔓  Đăng nhập…"))
        self._act_login.triggered.connect(self._do_login)
        self._act_logout = acct_m.addAction(tr("🔒  Đăng xuất"))
        self._act_logout.triggered.connect(self._do_logout)
        acct_m.addSeparator()
        self._act_chpw = acct_m.addAction(tr("🔑  Đổi mật khẩu…"))
        self._act_chpw.triggered.connect(self._change_password)

        help_m = mb.addMenu(tr("Help"))
        help_m.addAction(tr("About")).triggered.connect(self._about)
        help_m.addAction(tr("Shortcuts")).triggered.connect(self._shortcuts)

        # Language menu — luôn truy cập được kể cả khi toolbar bị ẩn (Full
        # Image View mặc định lúc khởi động ẩn toolbar nhưng giữ menu bar).
        lang_m = mb.addMenu(tr("🌐 Language"))
        self._lang_actions = []
        _grp = QActionGroup(self); _grp.setExclusive(True)
        _cur = current_language()
        for code, label in LANGUAGES:
            a = lang_m.addAction(label)
            a.setCheckable(True)
            a.setChecked(code == _cur)
            a.setData(code)
            _grp.addAction(a)
            a.triggered.connect(lambda _checked=False, c=code: self._change_language(c))
            self._lang_actions.append(a)

    # ── Connect signals ───────────────────────────────────────────
    def _connect_signals(self):
        self._btn_new.clicked.connect(self._new_pipeline)
        self._btn_open.clicked.connect(self._open_pipeline)
        self._btn_save.clicked.connect(self._save_pipeline)
        self._btn_fit.clicked.connect(self._canvas.zoom_fit)
        self._btn_zoom1.clicked.connect(self._canvas.zoom_reset)
        self._btn_clear.clicked.connect(self._clear_canvas)
        self._run_btn.clicked.connect(self._toggle_run)

        scene = self._canvas.aoi_scene
        scene.node_selected.connect(self._on_node_selected)
        scene.node_deselected.connect(self._on_node_deselected)
        scene.graph_changed.connect(self._on_graph_changed)
        scene.run_single.connect(self._run_single_node)
        # Double-click → open detail dialog (open_props signal = node_selected when double-clicked)
        # node_item emits open_props via signals.open_props → connected to node_selected
        # We intercept: scene.node_selected when triggered by double-click vs single-click
        # Solution: AOIScene.node_selected is emitted for BOTH; use separate signal
        # node_item.signals.open_props → open detail dialog
        scene._signals.open_props.connect(self._open_node_detail)
        # single select only updates props panel
        scene._signals.selected.connect(self._props.show_node)
        scene._signals.renamed.connect(self._on_node_renamed)

        self._props.params_changed.connect(self._on_graph_changed)

    # ── Auth / Login ──────────────────────────────────────────────
    def _apply_auth_state(self):
        """Đồng bộ UI theo trạng thái đăng nhập: mở/khoá Sửa & Chạy tool."""
        authed = self._auth.is_authenticated
        if hasattr(self, "_btn_login"):
            if authed:
                self._btn_login.setText(f"👤  {self._auth.username}  ·  {tr('Đăng xuất')}")
                self._btn_login.setStyleSheet(_LOGIN_ON_STYLE)
                self._btn_login.setToolTip(tr("Đang đăng nhập — bấm để đăng xuất."))
            else:
                self._btn_login.setText(tr("🔒  Đăng nhập"))
                self._btn_login.setStyleSheet(_LOGIN_OFF_STYLE)
                self._btn_login.setToolTip(
                    tr("Đăng nhập để Sửa tool / YOLO / pipeline. RUN không cần đăng nhập."))
        if hasattr(self, "_act_login"):
            self._act_login.setEnabled(not authed)
            self._act_logout.setEnabled(authed)
            self._act_chpw.setEnabled(authed)
        # Tool library + Properties chỉ thao tác được khi đã login. Dùng lock
        # overlay (mờ + bắt click → báo cần đăng nhập) thay cho setEnabled(False)
        # vốn xám và im lặng khiến user không hiểu sao bấm không được.
        for _ov in getattr(self, "_lock_overlays", []):
            _ov.setVisible(not authed)
            if not authed:
                _ov.raise_()
        # Canvas: khoá di chuyển/nối/xoá node khi chưa login (vẫn xem được).
        if hasattr(self, "_canvas"):
            self._canvas.aoi_scene.set_editable(authed)
        # YOLO Studio (train/label) = sửa → bấm sẽ hỏi đăng nhập (gate trong
        # _open_yolo_studio). RUN thì KHÔNG khoá (operator chạy inspection
        # không cần đăng nhập). Để nút enabled để click còn hiện hộp login.

    def _require_login(self, action: str = "") -> bool:
        """Đảm bảo đã đăng nhập. Chưa → mở dialog login. Trả True nếu đã login."""
        if self._auth.is_authenticated:
            return True
        from ui.login_dialog import LoginDialog
        dlg = LoginDialog(self._auth, self)
        if action:
            dlg.setWindowTitle(f"Đăng nhập — {action}")
        dlg.exec()
        self._apply_auth_state()
        return self._auth.is_authenticated

    def _notify_edit_locked(self, prompt: bool = False):
        """Báo cho user khi 1 thao tác CHỈNH SỬA bị chặn vì chưa đăng nhập.
        Luôn hiện thông báo ở status bar; hộp thoại mời đăng nhập được throttle
        (tối đa ~8s/lần) để không spam khi user thao tác liên tiếp."""
        if self._auth.is_authenticated:
            return
        self.statusBar().showMessage(
            tr("🔒 Chưa đăng nhập — chế độ chỉ xem. Bấm nút 🔒 Đăng nhập để chỉnh sửa."),
            5000)
        if not prompt:
            return
        import time
        now = time.monotonic()
        if now - getattr(self, "_last_lock_prompt", 0.0) < 8.0:
            return
        self._last_lock_prompt = now
        # Defer modal ra ngoài event handler đang chạy (mousePress/keyPress/drop
        # của canvas) để tránh mở nested event loop ngay trong handler.
        QTimer.singleShot(0, self._prompt_login_modal)

    def _prompt_login_modal(self):
        if self._auth.is_authenticated:
            return
        r = QMessageBox.question(
            self, tr("Chưa đăng nhập"),
            tr("Bạn cần đăng nhập để chỉnh sửa pipeline (thêm / di chuyển / nối "
               "/ xoá tool, sửa tham số).\n\nĐăng nhập ngay?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r == QMessageBox.Yes:
            self._do_login()

    def _on_login_button(self):
        self._do_logout() if self._auth.is_authenticated else self._do_login()

    def _do_login(self):
        if self._auth.is_authenticated:
            return
        if self._require_login():
            self.statusBar().showMessage(
                f"Đã đăng nhập: {self._auth.username} — đã mở khoá Sửa & Chạy.", 4000)

    def _do_logout(self):
        self._auth.logout()
        self._apply_auth_state()
        self.statusBar().showMessage("Đã đăng xuất — Sửa/Chạy tool đã bị khoá.", 4000)

    def _change_password(self):
        if not self._require_login("Đổi mật khẩu"):
            return
        from ui.login_dialog import ChangePasswordDialog
        if ChangePasswordDialog(self._auth, self).exec():
            QMessageBox.information(self, "Đổi mật khẩu",
                                    "Đã đổi mật khẩu thành công.")

    # ── Node selection ────────────────────────────────────────────
    def _on_node_selected(self, node_id: str):
        self._props.show_node(node_id)

    def _on_node_deselected(self):
        self._props.clear()

    def _on_graph_changed(self, *_):
        n = len(self._graph.nodes)
        c = len(self._graph.connections)
        self._node_count.setText(f"{n} {tr('nodes')}")
        self.statusBar().clearMessage()
        # Structure-only change → rebuild combo nhưng KHÔNG re-render ảnh (tránh
        # giật khi kéo node vào pipeline). Re-render chỉ chạy sau Run.
        self._img_viewer.refresh_node_list(redisplay=False)
        self._mark_dirty()

    def _on_node_renamed(self, node_id: str):
        """Đổi tên node → cập nhật props panel, viewer dropdown, detail dialog
        đang mở; mark dirty để nhắc lưu."""
        self._on_graph_changed()
        if self._graph and node_id in self._graph.nodes:
            self._props.show_node(node_id)
        dlg = self._detail_dialogs.get(node_id)
        if dlg and dlg.isVisible() and hasattr(dlg, "refresh_title"):
            dlg.refresh_title()

    def _mark_dirty(self):
        """Đánh dấu pipeline có thay đổi chưa lưu. Title hiện "•" prefix
        để user thấy dirty state, closeEvent sẽ prompt save."""
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _maybe_save_changes(self) -> bool:
        """Nếu dirty → prompt Yes/No/Cancel. Trả True nếu được phép tiếp
        tục (đã save hoặc user chọn Discard), False nếu Cancel."""
        if not self._dirty:
            return True
        r = QMessageBox.question(
            self, "Unsaved changes",
            "Pipeline có thay đổi chưa lưu. Lưu thay đổi không?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            self._save_pipeline()
            # Nếu user huỷ Save-As → _current_file vẫn None → vẫn dirty
            if self._dirty:
                return False
        return True

    # ── Open node detail dialog ───────────────────────────────────
    def _open_node_detail(self, node_id: str):
        if not self._graph or node_id not in self._graph.nodes:
            return
        # Cửa sổ chi tiết là nơi Sửa params + Chạy node → cần đăng nhập.
        if not self._require_login("Sửa tool"):
            return

        # Dialog đã mở → bring to front (tức thì, không cần overlay).
        dlg = self._detail_dialogs.get(node_id)
        if dlg and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return

        # Đang dựng dialog cho node này rồi (double double-click) → bỏ qua.
        if node_id in self._detail_pending:
            return
        self._detail_pending.add(node_id)

        # Con trỏ busy (OS vẽ → VẪN QUAY khi main thread block, vd Windows) +
        # overlay 'đang mở…'. Dựng dialog nặng ở tick event-loop kế (singleShot)
        # để overlay + con trỏ kịp đổi trước khi main thread bận dựng.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._busy.show_busy(f"Đang mở {self._graph.nodes[node_id].name}…")
        QTimer.singleShot(0, lambda nid=node_id: self._build_node_detail(nid))

    def _build_node_detail(self, node_id: str):
        """Dựng + show dialog chi tiết (chạy ở tick sau khi overlay đã hiện)."""
        try:
            if not self._graph or node_id not in self._graph.nodes:
                return
            node = self._graph.nodes[node_id]

            # PatMax / PatFind → PatMaxDialog chuyên dụng. YOLO Detect đi qua
            # NodeDetailDialog mặc định; YOLO Studio mở riêng từ toolbar.
            if node.tool.tool_id in ("patmax", "patmax_align", "patfind"):
                from ui.patmax_dialog import PatMaxDialog
                dlg = PatMaxDialog(node, self._graph, self)
                dlg.run_requested.connect(self._on_detail_run)
                dlg.model_trained.connect(
                    lambda: self._canvas.aoi_scene.refresh_node(node_id))
            else:
                from ui.node_detail_dialog import NodeDetailDialog
                dlg = NodeDetailDialog(node, self._graph, self)
                dlg.run_requested.connect(self._on_detail_run)
            dlg.finished.connect(
                lambda _, nid=node_id: self._detail_dialogs.pop(nid, None))
            self._detail_dialogs[node_id] = dlg
            dlg.show()
        finally:
            self._detail_pending.discard(node_id)
            self._busy.hide_busy()
            QApplication.restoreOverrideCursor()

    def _on_detail_run(self, node_id: str):
        """Node chạy từ detail dialog → refresh canvas + viewer."""
        self._canvas.aoi_scene.refresh_node(node_id)
        self._props.refresh_outputs()
        self._img_viewer.refresh_node_list()
        self._img_viewer.show_node(node_id)
        self._center_tabs.setCurrentIndex(1)   # Switch to image viewer

    # ── Run pipeline ──────────────────────────────────────────────
    def _toggle_run(self):
        if self._is_running:
            self._stop_run()
        else:
            self._start_run()

    def _start_run(self):
        # RUN KHÔNG cần đăng nhập — operator chạy inspection thoải mái. Chỉ
        # Sửa tool/pipeline/YOLO mới cần đăng nhập admin.
        if not self._graph.nodes:
            self.statusBar().showMessage("No nodes to run.", 3000)
            return
        self._is_running = True
        self._run_btn.setText("■   STOP")
        self._run_btn.setStyleSheet(_STOP_STYLE)
        self._progress.show()
        self._progress.setValue(0)
        self._set_status("RUNNING", "#ffd700")

        # PLC TriggerRoute → giới hạn chạy nhánh subgraph từ acquire_node_id.
        # Rỗng (= manual Run hoặc legacy single trigger) → chạy toàn pipeline.
        route_acq = getattr(self, "_plc_route_acquire", "") or ""

        self._worker_thread = QThread()
        self._worker = PipelineWorker(self._graph, acquire_node_id=route_acq)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_run_done)
        self._worker.error.connect(self._on_run_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.start()

    def _stop_run(self):
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self._finalize_run()

    def _on_run_done(self, results: dict, dur_ms: float):
        scene = self._canvas.aoi_scene
        scene.refresh_all_nodes()
        scene.refresh_connections()
        self._results.report_run(self._graph, results, dur_ms)
        self._props.refresh_outputs()
        self._img_viewer.refresh_node_list()
        self._img_viewer.refresh_current()

        # Refresh any open detail dialogs (PatMaxDialog không có refresh_outputs)
        for nid, dlg in self._detail_dialogs.items():
            if dlg.isVisible() and hasattr(dlg, "refresh_outputs"):
                dlg.refresh_outputs()

        self._finalize_run()
        self._set_status("PASS", "#39ff14")
        # Per-node breakdown — giúp xác định node nào chậm
        breakdown = sorted(
            [(self._graph.nodes[nid].name if nid in self._graph.nodes else nid,
              r.get("elapsed_ms", 0.0))
             for nid, r in results.items()],
            key=lambda x: -x[1])[:3]
        slow = "  |  " + ", ".join(f"{n}: {ms:.0f}ms" for n, ms in breakdown) if breakdown else ""
        self.statusBar().showMessage(
            f"Pipeline done in {dur_ms:.1f} ms  —  {len(results)} nodes{slow}", 8000)
        QTimer.singleShot(5000, lambda: self._set_status("IDLE", "#64748b"))

        # Đẩy kết quả về PLC (nếu đang connected)
        self._send_result_to_plc(results)

        # ── SFC Step 4: API POST (nếu sequence yêu cầu) ──────────
        if getattr(self, "_sfc_pending_post", False):
            self._sfc_pending_post = False
            self._sfc_manager.set_graph(self._graph)
            p = self._sfc_manager.api_post_cfg
            # Resolve body để log
            body_resolved, missing = self._sfc_manager.resolve_placeholders(
                p.body_template)
            self._sfc_log(
                f"[SFC] Step 4: POST {p.url}"
                + (f"\n        ⚠ Placeholder không resolve: {missing}" if missing else "")
                + f"\n        body = {body_resolved[:300]}")
            text, ok, err = self._sfc_manager.api_post()
            if err:
                self._sfc_log(f"[SFC] Step 4 ERROR: {err}")
            else:
                preview = text[:200].replace("\n", " ")
                self._sfc_log(
                    f"[SFC] Step 4 {'OK' if ok else 'FAIL'}: {preview!r}")
            self._sfc_log("──────── SFC SEQUENCE END ────────")

    def _on_run_error(self, msg: str):
        self._finalize_run()
        self._set_status("ERROR", "#ff3860")
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _finalize_run(self):
        self._is_running = False
        self._run_btn.setText("▶   RUN")
        self._run_btn.setStyleSheet(_RUN_STYLE)
        self._progress.hide()
        # Clear PLC route hint sau khi run xong → lần Run thủ công kế tiếp
        # luôn chạy toàn pipeline, không vô tình giới hạn subgraph.
        self._plc_route_acquire = ""

    def _run_single_node(self, node_id: str):
        # Chạy 1 node = "chạy" (không sửa) → KHÔNG cần đăng nhập, như nút RUN.
        node = self._graph.nodes.get(node_id)
        if not node:
            return
        # Save Image proc dựng ảnh composite từ graph → cần EXEC_CTX.graph.
        from core.tool_registry import EXEC_CTX
        EXEC_CTX.graph = self._graph
        inputs = {p.name: p.default for p in node.tool.inputs}
        for conn in self._graph.connections:
            if conn.dst_id == node_id:
                src = self._graph.nodes.get(conn.src_id)
                if src and conn.src_port in src.outputs:
                    inputs[conn.dst_port] = src.outputs[conn.src_port]
        try:
            out = node.tool.process_fn(inputs, node.params)
            node.outputs = out or {}
            node.status = "pass"
            if "pass" in node.outputs:
                node.status = "pass" if node.outputs["pass"] else "fail"
            node.error_msg = ""
        except Exception as e:
            node.outputs = {}
            node.status = "error"
            node.error_msg = str(e)

        self._canvas.aoi_scene.refresh_node(node_id)
        self._props.refresh_outputs()
        self._img_viewer.refresh_node_list()
        self._img_viewer.show_node(node_id)
        self._center_tabs.setCurrentIndex(1)

    # ── File ops ──────────────────────────────────────────────────
    def _new_pipeline(self):
        if not self._maybe_save_changes():
            return
        self._graph = FlowGraph()
        self._current_file = None
        self._rebuild_canvas()
        self._dirty = False
        self._update_title()

    def _rebuild_canvas(self):
        old = self._canvas
        idx = self._center_tabs.indexOf(old)
        self._canvas = AOICanvas(self._graph)
        scene = self._canvas.aoi_scene
        scene.node_selected.connect(self._on_node_selected)
        scene.node_deselected.connect(self._on_node_deselected)
        scene.graph_changed.connect(self._on_graph_changed)
        scene.run_single.connect(self._run_single_node)
        # Thao tác chỉnh sửa canvas bị chặn (chưa đăng nhập) → báo cho user.
        scene.edit_locked.connect(lambda: self._notify_edit_locked(prompt=True))
        scene._signals.open_props.connect(self._open_node_detail)
        scene._signals.selected.connect(self._props.show_node)
        scene._signals.renamed.connect(self._on_node_renamed)
        self._center_tabs.removeTab(idx)
        self._center_tabs.insertTab(idx, self._canvas, "🔧  Pipeline Canvas")
        # Sau khi load file/new blank → land trên Image Viewer (user thấy
        # ảnh ngay, không phải canvas trống). Trước đây ép setCurrentIndex
        # về canvas, không khớp workflow.
        self._center_tabs.setCurrentWidget(self._img_viewer)
        old.deleteLater()
        # Đồng bộ graph hiện tại cho proc cross-node (Save Image composite).
        from core.tool_registry import EXEC_CTX
        EXEC_CTX.graph = self._graph
        self._props.set_graph(self._graph)
        self._props.clear()
        self._img_viewer.set_graph(self._graph)
        self._img_viewer.refresh_node_list()
        self._on_graph_changed()
        self._update_title()
        # Scene mới mặc định editable=True → áp lại trạng thái khoá theo login.
        self._apply_auth_state()

    def _open_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Pipeline", projects_dir(),
            "AOI Pipeline (*.aoi *.json);;All Files (*)")
        if not path:
            return
        self.load_pipeline_from_path(path)

    def _switch_project(self):
        """File → Switch Project: pop StartupAOIPicker để chọn project khác.
        Có node → confirm discard trước. New blank → clear; file → load."""
        if self._graph.nodes:
            r = QMessageBox.question(
                self, "Switch Project",
                "Discard pipeline hiện tại và switch sang project khác?",
                QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        from ui.startup_picker import StartupAOIPicker
        dlg = StartupAOIPicker(self)
        if dlg.exec() != QDialog.Accepted:
            return
        chosen = dlg.chosen_path()
        if chosen:
            self.load_pipeline_from_path(chosen)
        else:
            # New blank
            self._graph = FlowGraph()
            self._current_file = None
            self._rebuild_canvas()
            self._update_title()

    def load_pipeline_from_path(self, path: str, background: bool = True,
                                progress_cb=None) -> bool:
        """Load .aoi/.json file vào canvas.

        background=True (mặc định, dùng cho menu Open / Switch Project khi app
            đã chạy): nạp file ở thread nền + progress dialog → không treo UI.
            Trả True nếu đã *bắt đầu* nạp (kết quả xử lý trong callback).
        background=False (dùng lúc khởi động — splash đã lo feedback): nạp đồng
            bộ, progress_cb(pct: float) báo tiến độ cho splash. Trả True nếu OK.
        """
        if not self._maybe_save_changes():
            return False

        if not background:
            try:
                graph = FlowGraph.load(path, progress_cb=progress_cb)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot load:\n{e}")
                return False
            self._apply_loaded_graph(graph, path)
            return True

        # ── Background: thread nền + progress dialog ──────────────
        if self._load_thread and self._load_thread.isRunning():
            return False  # đang nạp file khác, bỏ qua

        dlg = QProgressDialog("Đang nạp pipeline…", "", 0, 100, self)
        dlg.setWindowTitle("Loading")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)          # không cho hủy giữa chừng
        dlg.setMinimumDuration(150)        # file nhỏ nạp nhanh → không flash
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        self._load_dialog = dlg
        self._loading_path = path

        self._load_thread = QThread()
        self._load_worker = PipelineLoadWorker(path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        # QUAN TRỌNG: connect vào bound-method (QObject slot), KHÔNG dùng
        # lambda. Lambda/hàm thường → Qt mặc định DirectConnection → callback
        # chạy trong worker thread → tạo widget sai thread → crash. Bound-
        # method của MainWindow → AutoConnection = QueuedConnection (main thread).
        self._load_worker.progress.connect(self._on_bg_load_progress)
        self._load_worker.finished.connect(self._on_bg_load_done)
        self._load_worker.error.connect(self._on_bg_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.error.connect(self._load_thread.quit)
        self._load_thread.start()
        return True

    def _apply_loaded_graph(self, graph: FlowGraph, path: str):
        """Gắn graph đã nạp vào UI (chạy trên main thread)."""
        self._graph = graph
        self._current_file = path
        self._rebuild_canvas()
        self._add_recent_file(path)
        # _rebuild_canvas → _on_graph_changed → _mark_dirty (false positive
        # vì đó là load chứ không phải edit). Reset lại.
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Loaded: {path}", 3000)

    def _on_bg_load_progress(self, pct: int, msg: str):
        if self._load_dialog:
            self._load_dialog.setLabelText(msg)
            self._load_dialog.setValue(pct)

    def _on_bg_load_done(self, graph: FlowGraph):
        if self._load_dialog:
            self._load_dialog.close()
            self._load_dialog = None
        self._apply_loaded_graph(graph, self._loading_path)

    def _on_bg_load_error(self, msg: str):
        if self._load_dialog:
            self._load_dialog.close()
            self._load_dialog = None
        QMessageBox.critical(self, "Error", f"Cannot load:\n{msg}")

    @staticmethod
    def _get_recent_files() -> list:
        """List path đã mở/save gần đây (mới nhất đầu). Lưu trong QSettings."""
        s = QSettings()
        raw = s.value("recent_files", []) or []
        if isinstance(raw, str):
            raw = [raw]
        # Lọc bỏ file đã bị xóa khỏi disk
        return [p for p in raw if isinstance(p, str) and os.path.isfile(p)]

    def _add_recent_file(self, path: str, cap: int = 10):
        """Add path lên đầu recent list, dedupe, cap số lượng."""
        path = os.path.abspath(path)
        recents = self._get_recent_files()
        if path in recents:
            recents.remove(path)
        recents.insert(0, path)
        QSettings().setValue("recent_files", recents[:cap])

    def _save_pipeline(self):
        if not self._current_file:
            self._save_as()
            return
        try:
            self._graph.save(self._current_file)
            self._dirty = False
            self._update_title()
            self.statusBar().showMessage(f"Saved: {self._current_file}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save:\n{e}")

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", os.path.join(projects_dir(), "pipeline.aoi"),
            "AOI Pipeline (*.aoi);;JSON (*.json)")
        if path:
            self._current_file = path
            self._save_pipeline()
            self._add_recent_file(path)
            self._update_title()

    def _clear_canvas(self):
        if not self._require_login("Xoá pipeline"):
            return
        r = QMessageBox.question(self, "Clear", "Remove all nodes?",
                                 QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            for nid in list(self._graph.nodes.keys()):
                self._canvas.aoi_scene._delete_node(nid)
            self._on_graph_changed()

    # ── Status helpers ────────────────────────────────────────────
    def _set_status(self, text: str, color: str):
        sb = self.statusBar()
        sb.showMessage(f"  ● {text}", 0)

    def _on_center_tab_changed(self, idx: int):
        """Ẩn Properties panel (Info/Params/Output/Preview) khi user sang tab
        Image Viewer — các tab đó liên quan tới node properties, không cần
        thiết lúc xem ảnh. Show lại khi quay về Pipeline Canvas.
        Full Image View vẫn override (giữ panel ẩn bất kể tab nào)."""
        # Trong Full Image View, panel đã bị ẩn bởi _toggle_full_image_view;
        # đừng đè lên.
        if getattr(self, "_act_full_view", None) is not None \
                and self._act_full_view.isChecked():
            return
        widget = self._center_tabs.widget(idx)
        is_image_viewer = widget is getattr(self, "_img_viewer", None)
        self._props.setVisible(not is_image_viewer)
        if is_image_viewer:
            # Refit ảnh sau khi layout settle (Qt cần 1 tick để resize)
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._img_viewer._fit)

    def _toggle_full_image_view(self, checked: bool):
        """Full Image View: chỉ hiện Image Viewer canvas + Inspection Results.
        Ẩn tool library, properties panel, toolbar trên cùng và tab bar.
        Menu bar giữ lại để user toggle về (F11 / Esc / View → Full Image View).
        """
        self._tool_lib.setVisible(not checked)
        self._toolbar.setVisible(not checked)
        # Ẩn tab bar khi full view (chỉ còn Image Viewer hiển thị)
        self._center_tabs.tabBar().setVisible(not checked)
        if checked:
            self._props.setVisible(False)
            for i in range(self._center_tabs.count()):
                if "Image" in self._center_tabs.tabText(i):
                    self._center_tabs.setCurrentIndex(i)
                    break
            self.statusBar().showMessage(
                tr("  ● Full Image View — nhấn F11 hoặc Esc để thoát"), 0)
        else:
            # Khi thoát Full View, để _on_center_tab_changed quyết định
            # Properties panel hiện/ẩn dựa vào tab hiện tại (giữ Properties
            # ẩn nếu user vẫn đang ở Image Viewer tab).
            self.statusBar().clearMessage()
            self._on_center_tab_changed(self._center_tabs.currentIndex())
        # Refit image sau khi layout settle (Qt cần 1 tick để resize)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._img_viewer._fit)

    def keyPressEvent(self, event):
        """Esc thoát Full Image View khi đang bật."""
        from PySide6.QtCore import Qt
        if event.key() == Qt.Key_Escape and self._act_full_view.isChecked():
            self._act_full_view.setChecked(False)
            return
        super().keyPressEvent(event)

    def _tick(self):
        zoom = int(getattr(self._canvas, '_zoom', 1.0) * 100)
        n = len(self._graph.nodes)
        c = len(self._graph.connections)
        self.statusBar().showMessage(
            f"  Nodes: {n}   Connections: {c}   Zoom: {zoom}%", 0)

    def _update_title(self):
        fname = os.path.basename(self._current_file) if self._current_file else "Untitled"
        marker = "• " if self._dirty else ""
        self.setWindowTitle(f"{marker}Vision Ultimate — {fname}")

    # ── Heavy dialog open (overlay + busy cursor) ─────────────────
    def _open_heavy_dialog(self, label: str, build_fn):
        """Mở 1 dialog nặng (toolbar) với overlay + con trỏ busy giống node-
        detail: hiện overlay 'đang mở…' + WaitCursor NGAY, rồi chạy build_fn()
        (import + dựng + show dialog) ở tick event-loop kế để overlay kịp vẽ.
        Bring-to-front (nếu dialog đang mở) phải check TRƯỚC khi gọi hàm này."""
        if label in self._heavy_pending:
            return
        self._heavy_pending.add(label)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._busy.show_busy(f"Đang mở {label}…")

        def _run():
            try:
                build_fn()
            finally:
                self._heavy_pending.discard(label)
                self._busy.hide_busy()
                QApplication.restoreOverrideCursor()
        QTimer.singleShot(0, _run)

    # ── About ─────────────────────────────────────────────────────
    def _open_yolo_studio(self, initial_image=None):
        """Mở YOLO Studio dialog — độc lập với pipeline. Studio chỉ
        label + train + save file; user trỏ file đó vào YOLO Detect
        node bằng tay (Browse) trong pipeline."""
        # YOLO = train/label = sửa → cần đăng nhập admin.
        if not self._require_login("YOLO Studio"):
            return
        # QPushButton.clicked / QAction.triggered emit `checked: bool` làm
        # arg đầu — nếu connect trực tiếp slot này, initial_image sẽ là
        # bool thay vì ndarray. Coerce về None.
        if isinstance(initial_image, bool) or initial_image is None:
            initial_image = None
            if self._current_node_id_for_yolo():
                node = self._graph.nodes.get(self._current_node_id_for_yolo())
                if node:
                    initial_image = node.outputs.get("image")

        # Single-instance: đang mở → bring-to-front (tức thì, không overlay).
        if self._yolo_dialog is not None:
            try:
                if self._yolo_dialog.isVisible():
                    self._yolo_dialog.raise_()
                    self._yolo_dialog.activateWindow()
                    return
            except RuntimeError:
                self._yolo_dialog = None   # C++ object đã bị xóa

        def _build():
            # Import trong builder (dưới overlay) — yolo_studio 2461 dòng +
            # ultralytics nặng, đã được pre-warm ở thread nền lúc khởi động.
            from ui.yolo_studio import YoloStudioDialog
            dlg = YoloStudioDialog(self, initial_image)
            dlg.finished.connect(lambda *_: setattr(self, "_yolo_dialog", None))
            self._yolo_dialog = dlg
            dlg.show()
        self._open_heavy_dialog("YOLO Studio", _build)

    def _current_node_id_for_yolo(self):
        """Trả về node_id đang được chọn (nếu có)."""
        for nid, node in self._graph.nodes.items():
            if node.status in ("pass","fail") and "image" in node.outputs:
                return nid
        return None

    # ── PLC ───────────────────────────────────────────────────────
    def _open_plc_dialog(self):
        if self._plc_dialog and self._plc_dialog.isVisible():
            self._plc_dialog.set_graph(self._graph)
            self._plc_dialog.raise_()
            self._plc_dialog.activateWindow()
            return

        def _build():
            from ui.plc_dialog import PLCDialog
            self._plc_dialog = PLCDialog(self._plc_manager, self._graph, self)
            self._plc_dialog.trigger_fired.connect(self._on_plc_trigger)
            self._plc_dialog.showMaximized()
        self._open_heavy_dialog("PLC", _build)

    def _open_sfc_dialog(self):
        if self._sfc_dialog and self._sfc_dialog.isVisible():
            self._sfc_dialog.set_graph(self._graph)
            self._sfc_dialog.raise_()
            self._sfc_dialog.activateWindow()
            return

        def _build():
            from ui.sfc_dialog import SfcDialog
            self._sfc_dialog = SfcDialog(self._sfc_manager, self._graph, self)
            self._sfc_dialog.showMaximized()
        self._open_heavy_dialog("SFC / MES", _build)

    def _open_camera_dialog(self):
        if getattr(self, "_cam_dialog", None) and self._cam_dialog.isVisible():
            self._cam_dialog.raise_(); self._cam_dialog.activateWindow(); return

        def _build():
            from ui.camera_dialog import CameraSetupDialog
            self._cam_dialog = CameraSetupDialog(self)
            self._cam_dialog.show()
        self._open_heavy_dialog("Camera", _build)

    def _on_plc_trigger(self, acquire_node_id: str = ""):
        """PLC kích hoạt → tuỳ cấu hình SFC sequence:

        - Sequence OFF: chạy pipeline ngay (behaviour cũ).
        - Sequence ON:  Scan barcode → API GET → Pipeline → API POST
                        (POST chạy ở `_on_run_done` sau khi pipeline xong).

        `acquire_node_id` (từ TriggerRoute): nếu set, chỉ chạy nhánh
        pipeline xuất phát từ acquire đó. Rỗng = chạy toàn pipeline.
        """
        if self._is_running:
            self.statusBar().showMessage("PLC trigger ignored — pipeline already running", 3000)
            return

        # Lưu acquire_node_id cho lần run sắp tới — _start_run / SFC sequence
        # đều dùng chung. Clear sau khi _on_run_done finalize.
        self._plc_route_acquire = acquire_node_id or ""

        seq = self._sfc_sequence_settings()
        if not seq.get("enabled"):
            self._sfc_pending_post = False
            self._sfc_log("[SFC] Sequence disabled — running pipeline only")
            self._start_run()
            return

        abort_on_fail = bool(seq.get("abort_on_fail", True))
        self._sfc_log("──────── SFC SEQUENCE START ────────")

        # ── Step 1: Scan ────────────────────────────────────────
        if seq.get("step_scan", True):
            self._sfc_manager.set_graph(self._graph)
            cfg = self._sfc_manager.scanner
            self._sfc_log(
                f"[SFC] Step 1: Scan {cfg.port}@{cfg.baudrate} "
                f"trigger={cfg.trigger_hex!r} timeout={cfg.timeout_ms}ms")
            sn, err = self._sfc_manager.scan_once()
            if err or not sn:
                self._sfc_log(f"[SFC] Step 1 FAIL: {err or 'no data (timeout)'}")
                if abort_on_fail:
                    self._sfc_log("[SFC] ABORT — abort_on_fail=True")
                    return
            else:
                self._sfc_log(f"[SFC] Step 1 OK: SN={sn!r}")
        else:
            self._sfc_log("[SFC] Step 1 skipped (disabled)")

        # ── Step 2: API GET ─────────────────────────────────────
        if seq.get("step_get", True) and self._sfc_manager.api_get_cfg.enabled:
            g = self._sfc_manager.api_get_cfg
            resolved_url, _miss = self._sfc_manager.resolve_placeholders(
                g.url_template, extra={"SN": self._sfc_manager.last_sn})
            self._sfc_log(
                f"[SFC] Step 2: GET {resolved_url}\n"
                f"        expected_status={g.expected_status}"
                + (f", must_contain={g.expected_text!r}" if g.expected_text else ""))
            text, ok, err = self._sfc_manager.api_get()
            if err:
                self._sfc_log(f"[SFC] Step 2 ERROR: {err}")
                if abort_on_fail:
                    self._sfc_log("[SFC] ABORT — abort_on_fail=True")
                    return
            elif not ok:
                self._sfc_log(
                    f"[SFC] Step 2 FAIL: response không khớp\n"
                    f"        response.text = {text[:200]!r}")
                if abort_on_fail:
                    self._sfc_log("[SFC] ABORT — abort_on_fail=True")
                    return
            else:
                preview = text[:200].replace("\n", " ")
                self._sfc_log(f"[SFC] Step 2 OK: {preview!r}")
        else:
            self._sfc_log("[SFC] Step 2 skipped (disabled hoặc Enable API GET=off)")

        # ── Step 3: Pipeline run (Step 4 POST chạy ở _on_run_done) ──
        self._sfc_pending_post = bool(
            seq.get("step_post", True) and self._sfc_manager.api_post_cfg.enabled)
        self._sfc_log(
            f"[SFC] Step 3: Pipeline run "
            f"(POST sau khi xong = {self._sfc_pending_post})")
        self._start_run()

    def _sfc_log(self, msg: str) -> None:
        """Log SFC sequence event → console + status bar + PLC dialog log
        (nếu đang mở). Centralize ở 1 chỗ để dễ trace."""
        print(msg)
        # Status bar — chỉ dòng tóm tắt (cắt nếu dài)
        self.statusBar().showMessage(msg.split("\n", 1)[0][:200], 5000)
        # PLC dialog log — append đầy đủ
        if self._plc_dialog is not None and hasattr(self._plc_dialog, "_log"):
            try:
                self._plc_dialog._log(msg)
            except Exception:
                pass

    def _sfc_sequence_settings(self) -> dict:
        """Đọc settings sequence từ PLC dialog (nếu mở) hoặc QSettings."""
        if self._plc_dialog is not None and hasattr(
                self._plc_dialog, "get_sfc_sequence_settings"):
            return self._plc_dialog.get_sfc_sequence_settings()
        # Fallback: đọc trực tiếp QSettings
        from PySide6.QtCore import QSettings
        s = QSettings(); s.beginGroup("plc")
        out = {
            "enabled":       s.value("sfc_seq_enabled", False, type=bool),
            "step_scan":     s.value("sfc_seq_step_scan", True, type=bool),
            "step_get":      s.value("sfc_seq_step_get", True, type=bool),
            "step_post":     s.value("sfc_seq_step_post", True, type=bool),
            "abort_on_fail": s.value("sfc_seq_abort_fail", True, type=bool),
        }
        s.endGroup()
        return out

    def _send_result_to_plc(self, results: dict):
        """Gửi PASS/FAIL + dữ liệu mapping (length, area...) về PLC.
        Judge node ưu tiên (port 'pass' của node được chỉ định trong PLC
        config). Rỗng = legacy: pass khi mọi node trong graph pass."""
        if not self._plc_manager.is_connected:
            return
        judge_id = getattr(self._plc_manager.config, "result_judge_node_id", "") or ""
        if judge_id and judge_id in self._graph.nodes:
            jn = self._graph.nodes[judge_id]
            # Ưu tiên giá trị thực của port 'pass'; fallback theo node.status.
            jpass = jn.outputs.get("pass") if jn.outputs else None
            if jpass is not None:
                passed = bool(jpass)
            else:
                passed = jn.status not in ("fail", "error")
        else:
            passed = all(n.status not in ("fail", "error")
                         for n in self._graph.nodes.values())
        try:
            self._plc_manager.write_result(passed=passed)
        except Exception as e:
            self.statusBar().showMessage(f"PLC write PASS/FAIL error: {e}", 5000)
            return

        mappings = self._plc_manager.config.data_mappings
        if not mappings:
            self.statusBar().showMessage(
                f"→ PLC: {'PASS' if passed else 'FAIL'}", 3000)
            return

        report = self._plc_manager.write_data_mappings(results)
        ok = sum(1 for r in report if "error" not in r)
        err = len(report) - ok
        self.statusBar().showMessage(
            f"→ PLC: {'PASS' if passed else 'FAIL'}  "
            f"({ok} values{', '+str(err)+' errors' if err else ''})", 4000)

    def _about(self):
        # Thời hạn kích hoạt: _days_left do LicenseManager tính (theo expires
        # hoặc duration_days). Không có hạn → license vĩnh viễn; <0 → đã hết hạn.
        lic_html = ""
        try:
            from core.licensing import LicenseManager
            payload = LicenseManager().status().payload or {}
            days = payload.get("_days_left")
            if days is None:
                lic_html = f"<p>{tr('🔑 Bản quyền: vĩnh viễn')}</p>"
            elif int(days) >= 0:
                line = tr("🔑 Bản quyền: còn lại {days} ngày").format(
                    days=f"<b style='color:#00d4ff;'>{int(days)}</b>")
                expiry = payload.get("_effective_expiry", "")
                if expiry:
                    line += " " + tr("(hết hạn {date})").format(date=expiry)
                lic_html = f"<p>{line}</p>"
            else:
                lic_html = f"<p>{tr('🔑 Bản quyền đã hết hạn')}</p>"
        except Exception:
            pass
        QMessageBox.about(self, "Vision Ultimate",
            tr("<h2 style='color:#00d4ff;'>Vision Ultimate v1.0</h2>"
            "<p>Automated Optical Inspection<br>PySide6 + OpenCV</p>"
            "<ul><li>Drag-drop pipeline</li>"
            "<li>30+ inspection tools</li>"
            "<li>Real-time image viewer</li>"
            "<li>Pass/Fail judgment</li></ul>") + lic_html)

    def _shortcuts(self):
        QMessageBox.information(self, tr("Shortcuts"),
            tr("F5 — Run pipeline\n"
            "F6 — Stop\n"
            "Delete — Delete selected\n"
            "Tab — Toggle Canvas/Viewer\n"
            "F — Fit canvas\n"
            "R — Reset zoom\n"
            "Scroll — Zoom\n"
            "Middle-drag — Pan canvas\n"
            "Double-click node — Open detail window\n"
            "Drag port→port — Connect nodes\n"
            "Ctrl+S/O/N — Save/Open/New"))

    # ── State ─────────────────────────────────────────────────────
    def _restore_state(self):
        s = QSettings()
        g = s.value("geometry")
        if g:
            self.restoreGeometry(g)

    def closeEvent(self, event):
        if not self._maybe_save_changes():
            event.ignore()
            return
        QSettings().setValue("geometry", self.saveGeometry())
        if self._is_running:
            self._stop_run()
        # Chờ thread nạp file (nếu đang chạy) để không leak QThread.
        if self._load_thread and self._load_thread.isRunning():
            self._load_thread.quit()
            self._load_thread.wait(3000)
        for dlg in list(self._detail_dialogs.values()):
            dlg.close()
        if self._yolo_dialog is not None:
            try:
                self._yolo_dialog.close()
            except RuntimeError:
                pass
        try:
            self._plc_manager.disconnect()
        except Exception:
            pass
        try:
            from core.camera import CameraRegistry
            CameraRegistry.instance().close_all()
        except Exception:
            pass
        super().closeEvent(event)
