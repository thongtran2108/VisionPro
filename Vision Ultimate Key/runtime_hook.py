"""
Runtime hook — chạy TRƯỚC main.py khi app khởi động.

Mục đích:
1. Set Qt plugin path đúng (PySide6 đôi khi load nhầm khi frozen).
2. Hi-DPI scaling cho Windows 4K/2K monitor.
3. Set cwd về folder chứa .exe để các đường dẫn relative
   (assets/, vendor/, model files) hoạt động đúng.
"""
import os
import sys

# ── 1. Set cwd về folder chứa .exe ────────────────────────────────────
# Khi user double-click .exe từ Desktop, cwd có thể là C:\Users\<name>\
# khiến code đọc "assets/logo.png", "nutfixed.aoi" v.v. bị fail.
if getattr(sys, 'frozen', False):
    # Đang chạy từ PyInstaller .exe
    exe_dir = os.path.dirname(sys.executable)
    os.chdir(exe_dir)
    # Thêm exe_dir vào sys.path để `import vendor.mvs...` chạy được
    sys.path.insert(0, exe_dir)
    sys.path.insert(0, os.path.join(exe_dir, '_internal'))

# ── 1b. Native DLL search path cho PaddleOCR + PyTorch (OCR Max & YOLO) ─
# App gói CẢ paddle (OCR) LẪN torch (YOLO) — 2 stack native nặng. Trong app
# freeze phát sinh 2 vấn đề:
#   1) Windows không tự dò DLL ở thư mục con → libpaddle.pyd / torchvision._C
#      không tìm được DLL phụ thuộc → "Can not import paddle core" hoặc
#      "[WinError 127] procedure could not be found".
#   2) Cả paddle và torch đều ship Intel OpenMP (libiomp5md.dll) + MKL. Nạp
#      2 bản OpenMP trong 1 process → Intel OMP abort ("OMP: Error #15") hoặc
#      lệch symbol (WinError 127). KMP_DUPLICATE_LIB_OK=TRUE cho phép trùng.
# Xử lý: set KMP_DUPLICATE_LIB_OK, rồi với MỖI thư mục lib vừa
# add_dll_directory (dò DLL trực tiếp) VỪA prepend vào PATH (dò DLL bắc cầu
# — dependency-của-dependency mà add_dll_directory không phủ).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
if getattr(sys, 'frozen', False):
    _nbase = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    _lib_subdirs = (
        # paddle (OCR Max)
        ('paddle', 'libs'), ('paddle', 'base'), ('paddle', 'fluid'),
        # torch / torchvision (YOLO)
        ('torch', 'lib'), ('torchvision',),
    )
    _seen_dll_dirs = set()
    for _root in (_nbase, os.path.join(_nbase, '_internal')):
        for _parts in _lib_subdirs:
            _cand = os.path.join(_root, *_parts)
            if os.path.isdir(_cand) and _cand not in _seen_dll_dirs:
                _seen_dll_dirs.add(_cand)
                if hasattr(os, 'add_dll_directory'):
                    try:
                        os.add_dll_directory(_cand)
                    except OSError:
                        pass
                os.environ['PATH'] = _cand + os.pathsep + os.environ.get('PATH', '')

# ── 2. Qt plugin path ─────────────────────────────────────────────────
# PySide6 lookup plugins (platforms/, imageformats/, styles/...) qua
# QT_PLUGIN_PATH. Khi frozen, set thẳng để tránh load nhầm DLL hệ thống.
if getattr(sys, 'frozen', False):
    base = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    qt_plugins = os.path.join(base, 'PySide6', 'plugins')
    if not os.path.isdir(qt_plugins):
        # Layout fallback (onedir)
        qt_plugins = os.path.join(base, '_internal', 'PySide6', 'plugins')
    if os.path.isdir(qt_plugins):
        os.environ['QT_PLUGIN_PATH'] = qt_plugins
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
            qt_plugins, 'platforms')

# ── 3. Hi-DPI awareness (Windows) ─────────────────────────────────────
# Phải set TRƯỚC khi QApplication được tạo. PySide6 6.5+ enable mặc định
# nhưng set tường minh để chắc chắn trên monitor scaling 125%/150%/200%.
os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')
os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')

# ── 4. Tắt verbose log của OpenCV khi load video codec ────────────────
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

# ── 5. Log file cho lỗi không bắt được (chỉ khi build --windowed) ────
# Không có console → exception trong main thread khiến app exit silently.
# Redirect stderr ra file để debug dễ hơn. GHI vào %APPDATA%\VisionPro\logs
# (ghi được không cần admin) thay vì cạnh .exe (Program Files = read-only).
if getattr(sys, 'frozen', False) and sys.stderr is None:
    import platform
    if platform.system() == 'Windows':
        _base = os.environ.get('APPDATA') or os.path.expanduser('~')
    elif platform.system() == 'Darwin':
        _base = os.path.expanduser('~/Library/Application Support')
    else:
        _base = os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
    _log_dir = os.path.join(_base, 'VisionPro', 'logs')
    try:
        os.makedirs(_log_dir, exist_ok=True)
        log_path = os.path.join(_log_dir, 'error.log')
        sys.stderr = open(log_path, 'a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr  # gộp luôn print() vào log
    except Exception:
        pass  # không write được thì thôi, không crash
