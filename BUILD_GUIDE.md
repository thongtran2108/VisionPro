# BUILD GUIDE — AOI Vision Pro → .exe

Hướng dẫn đóng gói project VisionPro thành file `.exe` chạy độc lập trên Windows (không cần cài Python trên máy đích).

---

## 1. Chuẩn bị môi trường build

### 1.1. Yêu cầu

- **Windows 10/11 64-bit** (build trên máy nào → chạy được trên máy đó + version Windows tương đương)
- **Python 3.10 hoặc 3.11** (project đang dùng 3.10 dựa theo `__pycache__/*.cpython-310.pyc`)
- ~15 GB ổ cứng trống (build PySide6 + optional deps cần nhiều)

> ⚠️ **Quan trọng**: build trên Windows 10 → chạy được trên cả Win 10 và Win 11. Nhưng build trên Win 11 → có thể fail trên Win 10 vì khác CRT version. **Khuyến nghị build trên Windows 10**.

### 1.2. Tạo venv build SẠCH

Đây là bước quan trọng nhất. PyInstaller bundle TẤT CẢ packages trong venv → nếu venv lẫn packages thừa, file `.exe` phình to vô lý.

```cmd
cd C:\path\to\VisionPro
python -m venv venv_build
venv_build\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
REM Bat buoc (cryptography = cap/verify license):
pip install PySide6>=6.5.0 opencv-python>=4.8.0 numpy>=1.24.0 Pillow>=10.0.0 PyYAML cryptography>=41.0.0
pip install pyinstaller lap

REM Optional - chi cai neu can feature tuong ung:
REM YOLO Studio (train/detect):
pip install ultralytics onnxruntime "onnx<2.0.0" onnxslim

REM OCR Max tool:
pip install easyocr            REM hoac: pytesseract

REM ID Reader (barcode/QR):
pip install pyzbar
```

> 💡 **Tip**: cài optional package nào thì `VisionPro.spec` tự động bundle gói đó (xem log `[spec] ✓ Include optional dep:` khi build). Không cài → app vẫn build được, chỉ là feature đó không chạy trên máy đích.

### 1.3. Copy 5 file build vào folder VisionPro/

Đặt 5 file này ngay cạnh `main.py`:

```
VisionPro/
├── main.py
├── VisionPro.spec       ← FILE BUILD
├── runtime_hook.py      ← FILE BUILD
├── version_info.txt     ← FILE BUILD
├── build.bat            ← FILE BUILD
├── BUILD_GUIDE.md       ← FILE NÀY
├── assets/
├── core/
├── ui/
└── vendor/
```

---

## 2. Build

### Cách 1 — Dùng script tự động

```cmd
venv_build\Scripts\activate
.\build.bat
```

### Cách 2 — Build thủ công

```cmd
venv_build\Scripts\activate
pyinstaller VisionPro.spec --clean --noconfirm
```

Thời gian build: **5-15 phút** tùy số optional dependencies đã cài.

Output: `dist\VisionUltimate\VisionUltimate.exe`

Test: double-click `VisionUltimate.exe`. Nếu app khởi động và hiện startup picker → build thành công 🎉

---

## 3. Xử lý các lỗi thường gặp

### 3.1. App "flash" rồi tắt ngay, không thấy gì

Bình thường khi `console=False` (mode `--windowed`) → không có console nên error message không hiện.

**Cách debug:**

1. Mở `VisionPro.spec`, đổi:
   ```python
   console=False,    # đổi thành True
   ```
2. Build lại: `pyinstaller VisionPro.spec --clean --noconfirm`
3. Chạy `VisionUltimate.exe` từ terminal (`cmd`) — sẽ thấy error message rõ ràng.

Hoặc xem `dist\VisionUltimate\error.log` (runtime_hook đã setup auto-log stderr).

### 3.2. `ModuleNotFoundError: No module named 'xxx'`

PyInstaller miss module. Mở `VisionPro.spec`, thêm vào `hiddenimports`:

```python
hiddenimports = [
    ...
    'tên_module_bị_miss',
]
```

Module hay miss với project có ML stack: `skimage.filters.rank.core_cy_3d`, `scipy.special.cython_special`, `sklearn.utils._cython_blas`.

### 3.3. `MvCameraControl.dll không tìm thấy`

Đây không phải lỗi build — driver MVS SDK cần cài riêng trên **máy đích**.

**2 cách xử lý:**

**Cách A — User tự cài MVS SDK** (khuyến nghị, vì HikRobot yêu cầu license):
- Tải MVS SDK từ trang HikRobot/Hikvision.
- Cài vào `C:\Program Files (x86)\MVS\`.
- Code đã có logic auto-detect đường dẫn này.

**Cách B — Bundle DLL kèm `.exe`** (chỉ làm nếu có quyền redistribute) — **đã tự động**:
- Tạo folder `dll/` trong project (cùng cấp `main.py`).
- Copy **toàn bộ `.dll`** trong `C:\Program Files (x86)\MVS\Development\Win64_x64\`
  vào `dll/` — tức `MvCameraControl.dll` **kèm tất cả DLL phụ thuộc**
  (`FormatConversion.dll`, `MediaProcess.dll`, `MVGigEVisionSDK.dll`,
  `MvUsb3vTL.dll`, các DLL GenICam/log…). Xem `dll/README.md`.
- Build lại — `VisionPro.spec` tự copy chúng vào `_internal/dll/` (log
  `[spec] ✓ Bundle N MVS DLL…`). Không cần sửa spec thủ công nữa.

> ⚠️ **Sai lầm hay gặp**: chỉ copy mỗi `MvCameraControl.dll` → tìm thấy DLL
> nhưng vẫn lỗi `DLL load failed` vì thiếu DLL phụ thuộc. Phải copy **cả cụm**.

> ⚠️ **DLL để nhầm chỗ**: nếu copy DLL thủ công vào bản đã build, phải đặt ở
> `dist\VisionUltimate\dll\` (cạnh `.exe`) **hoặc** `dist\VisionUltimate\_internal\dll\`.
> Cả hai vị trí này driver đều tự dò (xem thứ tự ưu tiên trong
> `vendor/mvs/MvCameraControl_class.py`). Cách bền nhất: bỏ DLL vào `dll/`
> của project rồi **build lại**.

### 3.4. File `.exe` quá to (>1 GB)

Bình thường nếu có PyTorch/Ultralytics (Torch ~1 GB). Giảm bằng:

1. **Không cài optional deps không dùng** trong venv build. Nếu khách không xài YOLO Studio → uninstall ultralytics + torch trước khi build:
   ```cmd
   pip uninstall ultralytics torch torchvision
   ```
   Tiết kiệm ngay ~1.2 GB.

2. **Loại Qt modules không dùng** — `VisionPro.spec` đã exclude rồi (Qt3D, WebEngine, Multimedia...). Có thể thêm vào `excludes` nếu chắc chắn không dùng.

3. **Loại numpy test modules** (thêm vào `excludes`):
   ```python
   'numpy.tests', 'numpy.f2py', 'numpy.distutils',
   ```

### 3.5. Antivirus / Windows Defender báo virus

Vấn đề kinh điển của PyInstaller `.exe`. Nguyên nhân: bootloader của PyInstaller giống pattern của một số malware.

**Giảm false positive:**

- Đã dùng `--onedir` thay `--onefile` ✓ (trong spec)
- Đã không dùng UPX ✓ (`upx=False`)
- Đã có `version_info.txt` để file `.exe` không "naked" ✓

Nếu vẫn bị flag → cần **code signing certificate** (~200-400 USD/năm từ Sectigo/DigiCert) ký file `.exe`. Đây là cách duy nhất triệt để.

Tạm thời: user cần add exception trong Windows Defender cho folder `dist\VisionUltimate\`.

### 3.6. App chạy nhưng OpenCV/numpy báo lỗi DLL

```
ImportError: DLL load failed while importing _multiarray_umath
```

Thường do mix numpy 2.x với opencv build cho numpy 1.x.

**Fix**: trong venv build, ép numpy < 2.0:
```cmd
pip install "numpy<2.0" --force-reinstall
```

(Bạn đã gặp vấn đề này ở project trước rồi — vẫn giải pháp đó.)

### 3.7. Hi-DPI / app bị mờ trên màn hình 4K

`runtime_hook.py` đã set `QT_ENABLE_HIGHDPI_SCALING=1` rồi. Nếu vẫn mờ:

1. Chuột phải `VisionUltimate.exe` → Properties → Compatibility → Change high DPI settings.
2. Tick **"Override high DPI scaling behavior"** → chọn **"Application"**.

---

## 4. Đóng gói lên thành 1 installer chuyên nghiệp (optional)

Sau khi có `dist\VisionUltimate\`, dùng **Inno Setup** (free) để tạo file installer:

1. Tải Inno Setup: https://jrsoftware.org/isinfo.php
2. Tạo file `installer.iss`:

```pascal
[Setup]
AppName=AOI Vision Pro
AppVersion=1.0.0
AppPublisher=YourCompany
DefaultDirName={autopf}\VisionPro
DefaultGroupName=AOI Vision Pro
UninstallDisplayIcon={app}\VisionUltimate.exe
OutputBaseFilename=VisionPro_Setup_v1.0.0
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
SetupIconFile=assets\logo.ico

[Files]
Source: "dist\VisionUltimate\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\AOI Vision Pro"; Filename: "{app}\VisionUltimate.exe"
Name: "{commondesktop}\AOI Vision Pro"; Filename: "{app}\VisionUltimate.exe"

[Run]
Filename: "{app}\VisionUltimate.exe"; Description: "Run AOI Vision Pro now"; Flags: nowait postinstall skipifsilent
```

3. Right-click `installer.iss` → Compile → ra file `VisionPro_Setup_v1.0.0.exe` (~200-500 MB tùy deps).

User chạy installer → app cài vào `C:\Program Files\VisionPro\` + có shortcut Desktop + uninstaller chuẩn.

---

## 5. Checklist cuối trước khi giao cho khách hàng

- [ ] Build trên Windows 10 (tương thích ngược)
- [ ] Test trên 1 máy Windows **không cài Python** xem chạy được không
- [ ] Test với màn hình 1920×1080 và 4K
- [ ] Nếu dùng camera HikRobot: hướng dẫn khách cài MVS SDK trước
- [ ] Nếu dùng PLC: test kết nối Modbus/socket trên máy đích
- [ ] Tạo file `README.txt` đặt cạnh `VisionUltimate.exe` ghi cách dùng cơ bản

---

## 6. Cấu trúc output

```
dist/
└── VisionPro/
    ├── VisionUltimate.exe              ← Main executable
    ├── _internal/                 ← Python runtime + all libs (PyInstaller layout)
    │   ├── python311.dll
    │   ├── PySide6/
    │   ├── cv2/
    │   ├── numpy/
    │   └── ... (rất nhiều file)
    ├── assets/
    │   └── logo.png
    ├── vendor/
    │   └── mvs/
    ├── 1.png, model.json, model.npz, nutfixed.aoi
    ├── projects/                  ← User pipelines (auto-tạo bởi build.bat)
    ├── logs/                      ← User logs
    └── error.log                  ← Auto-generated khi có exception
```

Folder `_internal/` là layout chuẩn của PyInstaller 6+. **Không đổi tên, không xóa**.

---

Có vấn đề gì khi build, copy paste error message vào chat — sẽ debug cùng. 🛠️
