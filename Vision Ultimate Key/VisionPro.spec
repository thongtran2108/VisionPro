# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file cho AOI Vision Pro
─────────────────────────────────────────
Build:
    pyinstaller VisionPro.spec --clean --noconfirm

Output: dist/VisionUltimate/VisionUltimate.exe (+ folder _internal)

Lưu ý:
- Dùng --onedir (mặc định của spec này), KHÔNG --onefile
  Lý do: PySide6 + OpenCV + ctypes DLL của MVS rất hay lỗi khi extract
  từ archive onefile, và khởi động nhanh hơn nhiều.
- DLL của MVS SDK (MvCameraControl.dll + các DLL phụ thuộc): nếu có
  folder `dll/` cạnh main.py, spec TỰ bundle toàn bộ vào _internal/dll/.
  Khi chạy, driver tự dò DLL theo thứ tự: <exe>/dll → _internal/dll →
  C:/Program Files (x86)/MVS/... (xem vendor/mvs/MvCameraControl_class.py).
  Không có folder dll/ → không bundle, user phải cài MVS SDK trên máy đích.
"""
import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

# ────────────────────────────────────────────────────────────────────
# 1. PROJECT ROOT
# ────────────────────────────────────────────────────────────────────
# Khi build, spec file nằm ngay trong folder VisionPro/.
PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC) if '__file__' not in dir()
                               else os.path.dirname(__file__))

# ────────────────────────────────────────────────────────────────────
# 2. DATA FILES — assets, models, config, vendor SDK Python wrapper
# ────────────────────────────────────────────────────────────────────
datas = [
    # Logo + window icon
    ('assets/logo.png', 'assets'),
    # File splash đầu tiên (nếu có)
    ('1.png', '.'),
    # Bảng dịch i18n (vi/en/zh) — bắt buộc kèm để đổi ngôn ngữ hoạt động
    ('core/i18n_data.json', 'core'),
    # Default model files (YOLO nuts demo)
    ('model.json', '.'),
    ('model.npz', '.'),
    ('nutfixed.aoi', '.'),
    # Vendor MVS Python wrapper — KHÔNG kèm DLL (xem lưu ý ở header)
    ('vendor/mvs/*.py', 'vendor/mvs'),
]

# Lọc file không tồn tại để build không fail khi user chưa có demo files
datas = [(src, dst) for src, dst in datas
         if os.path.exists(os.path.join(PROJECT_ROOT, src.split('*')[0]))
         or '*' in src]

# ────────────────────────────────────────────────────────────────────
# 3. HIDDEN IMPORTS — modules PyInstaller không tự detect
# ────────────────────────────────────────────────────────────────────
hiddenimports = [
    # PySide6 — Qt sub-modules đôi khi không detect đủ
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtSvg',
    'PySide6.QtPrintSupport',  # cần khi save image/PDF

    # OpenCV
    'cv2',

    # YAML (dùng trong yolo_studio.py)
    'yaml',

    # Math/sci stack
    'numpy',
    'numpy.core._multiarray_umath',
    'numpy.core._multiarray_tests',

    # Vendor MVS — tất cả sub-module
    'vendor',
    'vendor.mvs',
    'vendor.mvs.MvCameraControl_class',
    'vendor.mvs.CameraParams_const',
    'vendor.mvs.CameraParams_header',
    'vendor.mvs.MvErrorDefine_const',
    'vendor.mvs.PixelType_header',

    # License (ký số Ed25519) — verify file license.key khi khởi động
    'cryptography',
    'cryptography.hazmat.primitives.asymmetric.ed25519',
    'cryptography.hazmat.primitives.serialization',
    'cryptography.hazmat.bindings._rust',   # backend Rust của cryptography 41+
    'cffi',
    '_cffi_backend',
]

# ── Optional dependencies ──
# Code dùng lazy-import (try/except) cho: ultralytics, onnxruntime,
# easyocr, pytesseract, pyzbar. Nếu user CÀI sẵn trong venv build,
# tự động bundle để app chạy được offline. Nếu chưa cài, skip silently.
_optional = [
    ('cryptography', 'cryptography (license Ed25519)'),
    ('PIL', 'Pillow / PIL'),
    ('scipy', 'SciPy'),
    ('onnxruntime', 'ONNX Runtime'),
    ('onnx', 'ONNX'),
    ('ultralytics', 'Ultralytics YOLO'),
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('easyocr', 'EasyOCR'),
    ('pytesseract', 'Tesseract wrapper'),
    ('pyzbar', 'pyzbar (barcode/QR)'),
]
# collect_submodules() import từng submodule trong 1 isolated subprocess để
# liệt kê. Vài submodule làm subprocess CRASH CỨNG (access violation
# 0xC0000005) — KHÔNG phải ImportError → except ImportError không bắt được,
# cả build chết. Điển hình: onnx.reference (pure-Python evaluator) kéo native
# ext xung đột numpy/protobuf. Xử lý 2 lớp:
#   • filter: bỏ submodule độc hại khỏi việc duyệt (không recurse vào nó).
#   • except Exception rộng: 1 dep collect lỗi thì SKIP dep đó, build vẫn xong.
_submodule_filters = {
    # onnx.reference không cần lúc runtime (inference qua onnxruntime).
    'onnx': lambda name: not name.startswith('onnx.reference'),
}

binaries = []
for mod_name, _label in _optional:
    try:
        __import__(mod_name)
    except Exception:
        print(f"[spec]  – Skip (chưa cài): {_label}")
        continue
    try:
        _flt = _submodule_filters.get(mod_name)
        if _flt is not None:
            hiddenimports += collect_submodules(mod_name, filter=_flt)
        else:
            hiddenimports += collect_submodules(mod_name)
        datas += collect_data_files(mod_name)
        binaries += collect_dynamic_libs(mod_name)
        print(f"[spec]  ✓ Include optional dep: {_label}")
    except Exception as _e:
        print(f"[spec]  ⚠ Bỏ qua '{_label}' — collect lỗi "
              f"({type(_e).__name__}): {_e}")

# ── PaddlePaddle / PaddleOCR (OCR Max) ───────────────────────────────
# Gói RIÊNG (không nằm trong _optional) vì paddle cực khó đóng gói:
# libpaddle.pyd phụ thuộc 1 loạt DLL trong paddle/libs (mkl, iomp5,
# openblas, common...). Thiếu → app .exe báo:
#   "DLL load failed while importing libpaddle: The specified module
#    could not be found."
# Gom từng bước ĐỘC LẬP: 1 bước lỗi vẫn giữ các bước còn lại (nhất là DLL).
def _bundle_paddle(pkg, label):
    try:
        __import__(pkg)
    except Exception:
        print(f"[spec]  – Skip (chưa cài): {label}")
        return
    import importlib, glob as _glob
    _root = os.path.dirname(importlib.import_module(pkg).__file__)
    _parent = os.path.dirname(_root)
    try:
        datas.extend(collect_data_files(pkg))          # dict/font/config/version
    except Exception as _e:
        print(f"[spec]    ! data {pkg}: {_e}")
    try:
        binaries.extend(collect_dynamic_libs(pkg))
    except Exception as _e:
        print(f"[spec]    ! dylibs {pkg}: {_e}")
    # Glob MỌI .dll/.pyd trong package (giữ nguyên cây thư mục) → fix DLL load
    _n = 0
    for _p in (_glob.glob(os.path.join(_root, '**', '*.dll'), recursive=True) +
               _glob.glob(os.path.join(_root, '**', '*.pyd'), recursive=True)):
        binaries.append((_p, os.path.relpath(os.path.dirname(_p), _parent)))
        _n += 1
    try:
        hiddenimports.extend(collect_submodules(pkg))  # dynamic import lúc runtime
    except Exception as _e:
        print(f"[spec]    ! submodules {pkg} ({type(_e).__name__}): {_e}")
    print(f"[spec]  ✓ Include {label} (+{_n} dll/pyd)")

_bundle_paddle('paddle', 'PaddlePaddle')
_bundle_paddle('paddleocr', 'PaddleOCR')

# PaddleOCR 2.x chèn thư mục GÓI CỦA CHÍNH NÓ vào sys.path rồi import các package
# con như TOP-LEVEL: `import tools.infer...`, `import ppocr...` (KHÔNG phải
# `paddleocr.tools`). Trong app .exe, sys.path có _internal (sys._MEIPASS) nên
# resolver đi tìm `_internal\tools\__init__.py`, `_internal\ppocr\...` ở GỐC.
# collect_submodules('paddleocr') chỉ đặt tên 'paddleocr.tools...' → KHÔNG khớp
# → app báo:
#   "PaddleOCR init lỗi: [Errno 2] No such file or directory:
#    ...\_internal\tools\__init__.py".
# Fix: copy NGUYÊN CÂY tools/ ppocr/ ppstructure/ (kèm mọi file: .py, .yml, dict,
# font…) của paddleocr ra GỐC _internal để `import tools`/`import ppocr` chạy.
def _paddleocr_toplevel_pkgs():
    try:
        import paddleocr
    except Exception:
        return
    import glob as _g
    _root = os.path.dirname(paddleocr.__file__)
    _n = 0
    for _top in ('tools', 'ppocr', 'ppstructure'):
        _src = os.path.join(_root, _top)
        if not os.path.isdir(_src):
            continue
        for _f in _g.glob(os.path.join(_src, '**', '*'), recursive=True):
            if os.path.isfile(_f):
                # dst = đường dẫn tương đối so với gói paddleocr → bắt đầu bằng
                # <_top> ⇒ file rơi vào _internal\<_top>\... (gốc _internal).
                datas.append((_f, os.path.relpath(os.path.dirname(_f), _root)))
                _n += 1
    print(f"[spec]  ✓ PaddleOCR top-level (tools/ppocr/ppstructure) +{_n} files")

_paddleocr_toplevel_pkgs()

# Cython: vài thành phần trong stack paddle/paddleocr đọc file mẫu trong
# Cython/Utility (*.pyx, *.pxd, *.c, *.cpp, *.h, cả .py) lúc chạy. PyInstaller
# mặc định không gom → app .exe báo:
#   "PaddleOCR init lỗi: ...\_internal\Cython\Utility\CppSupport.cpp".
# Gom data (kèm .py) + submodules; tách bước để lỗi submodule không mất data.
try:
    import Cython  # noqa: F401
    try:
        datas += collect_data_files('Cython', include_py_files=True)
    except Exception as _e:
        print(f"[spec]    ! Cython data: {_e}")
    try:
        hiddenimports += collect_submodules('Cython')
    except Exception as _e:
        print(f"[spec]    ! Cython submodules ({type(_e).__name__}): {_e}")
    # Glob thẳng thư mục Utility cho chắc (đây là thứ báo thiếu).
    import glob as _cglob
    _cy_root = os.path.dirname(Cython.__file__)
    _cy_parent = os.path.dirname(_cy_root)
    _cn = 0
    for _p in _cglob.glob(os.path.join(_cy_root, 'Utility', '*')):
        if os.path.isfile(_p):
            datas.append((_p, os.path.relpath(os.path.dirname(_p), _cy_parent)))
            _cn += 1
    print(f"[spec]  ✓ Include Cython (+{_cn} Utility files)")
except Exception:
    print("[spec]  – Skip Cython (chưa cài)")

# ── Camera SDK DLL (MVS HikRobot) ────────────────────────────────────
# Nếu có folder dll/ cạnh main.py, copy TẤT CẢ .dll vào _internal/dll/.
# Đặt nguyên cụm DLL cùng 1 thư mục để chúng tìm thấy nhau khi load
# (driver gọi os.add_dll_directory tới thư mục này lúc chạy).
_dll_dir = os.path.join(PROJECT_ROOT, 'dll')
if os.path.isdir(_dll_dir):
    _n_dll = 0
    for _fn in sorted(os.listdir(_dll_dir)):
        if _fn.lower().endswith('.dll'):
            datas.append((os.path.join(_dll_dir, _fn), 'dll'))
            _n_dll += 1
    print(f"[spec]  ✓ Bundle {_n_dll} MVS DLL từ dll/ → _internal/dll/")
else:
    print("[spec]  – Khong co folder dll/ (camera MVS se can SDK tren may dich)")

# ────────────────────────────────────────────────────────────────────
# 4. EXCLUDES — bỏ module nặng không dùng để giảm size .exe
# ────────────────────────────────────────────────────────────────────
excludes = [
    'tkinter',              # dùng PySide6, không cần Tk
    'onnx.reference',       # pure-Python evaluator — crash lúc collect, vô dụng runtime
    'matplotlib',           # project không dùng
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'sphinx',
    'pandas',               # nếu KHÔNG dùng pandas, exclude tiết kiệm ~50MB
    'PyQt5',                # tránh conflict với PySide6
    'PyQt6',
    # Qt modules không dùng — tiết kiệm thêm vài chục MB
    'PySide6.Qt3DCore',
    'PySide6.Qt3DRender',
    'PySide6.Qt3DInput',
    'PySide6.Qt3DLogic',
    'PySide6.Qt3DAnimation',
    'PySide6.Qt3DExtras',
    'PySide6.QtCharts',
    'PySide6.QtDataVisualization',
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    'PySide6.QtBluetooth',
    'PySide6.QtNfc',
    'PySide6.QtPositioning',
    'PySide6.QtLocation',
    'PySide6.QtQuick',
    'PySide6.QtQuick3D',
    'PySide6.QtQuickWidgets',
    'PySide6.QtQml',
]

# ────────────────────────────────────────────────────────────────────
# 5. ANALYSIS
# ────────────────────────────────────────────────────────────────────
a = Analysis(
    ['main.py'],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# ────────────────────────────────────────────────────────────────────
# 6. EXECUTABLE
# ────────────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionUltimate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                  # KHÔNG dùng UPX — hay bị antivirus flag
    console=False,              # GUI app: ẩn console
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/logo.png',     # PyInstaller convert .png → .ico tự động
    # Metadata cho file .exe (chuột phải → Properties)
    version='version_info.txt' if os.path.exists(
        os.path.join(PROJECT_ROOT, 'version_info.txt')) else None,
)

# ────────────────────────────────────────────────────────────────────
# 7. COLLECT — gom tất cả vào dist/VisionPro/
# ────────────────────────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VisionUltimate',
)
