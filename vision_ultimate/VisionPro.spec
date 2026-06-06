# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file cho AOI Vision Pro
─────────────────────────────────────────
Build:
    pyinstaller VisionPro.spec --clean --noconfirm

Output: dist/VisionPro/VisionPro.exe (+ folder _internal)

Lưu ý:
- Dùng --onedir (mặc định của spec này), KHÔNG --onefile
  Lý do: PySide6 + OpenCV + ctypes DLL của MVS rất hay lỗi khi extract
  từ archive onefile, và khởi động nhanh hơn nhiều.
- DLL của MVS SDK (MvCameraControl.dll) KHÔNG được bundle — driver yêu
  cầu user cài MVS SDK của HikRobot trên máy đích (giấy phép + Windows
  service). Code đã có logic tự dò DLL ở C:/Program Files (x86)/MVS/...
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
]

# ── Optional dependencies ──
# Code dùng lazy-import (try/except) cho: ultralytics, onnxruntime,
# easyocr, pytesseract, pyzbar. Nếu user CÀI sẵn trong venv build,
# tự động bundle để app chạy được offline. Nếu chưa cài, skip silently.
_optional = [
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
binaries = []
for mod_name, _label in _optional:
    try:
        __import__(mod_name)
        hiddenimports += collect_submodules(mod_name)
        datas += collect_data_files(mod_name)
        binaries += collect_dynamic_libs(mod_name)
        print(f"[spec]  ✓ Include optional dep: {_label}")
    except ImportError:
        print(f"[spec]  – Skip (chưa cài): {_label}")

# ────────────────────────────────────────────────────────────────────
# 4. EXCLUDES — bỏ module nặng không dùng để giảm size .exe
# ────────────────────────────────────────────────────────────────────
excludes = [
    'tkinter',              # dùng PySide6, không cần Tk
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
