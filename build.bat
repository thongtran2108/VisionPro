@echo off
REM ══════════════════════════════════════════════════════════════════════
REM   VisionPro — Build script
REM ══════════════════════════════════════════════════════════════════════
REM   Chạy: build.bat
REM   Yêu cầu: đã activate venv build, đã `pip install -r requirements.txt`
REM            đã `pip install pyinstaller`
REM ══════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo   AOI Vision Pro - PyInstaller Build
echo ====================================================================
echo.

REM ── Check pyinstaller ────────────────────────────────────────────────
where pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [X] pyinstaller chua duoc cai. Chay:
    echo     pip install pyinstaller
    pause
    exit /b 1
)

REM ── Hien thi version ─────────────────────────────────────────────────
echo [i] Python:
python --version
echo [i] PyInstaller:
pyinstaller --version
echo.

REM ── Bao dam dependency BAT BUOC (cryptography cho license) da cai ─────
echo [i] Kiem tra dependencies (cryptography cho license)...
python -c "import cryptography" 2>nul
if errorlevel 1 (
    echo [!] Thieu 'cryptography' - dang cai theo requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [X] Cai requirements that bai. Chay thu cong:
        echo     pip install -r requirements.txt
        pause
        exit /b 1
    )
)
echo     OK.
echo.

REM ── Xoa output cu ────────────────────────────────────────────────────
echo [1/4] Xoa build output cu...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo     done.
echo.

REM ── Build ────────────────────────────────────────────────────────────
echo [2/4] Build (5-15 phut, tuy may + dependencies)...
pyinstaller VisionPro.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo [X] BUILD FAILED.
    pause
    exit /b 1
)
echo.

REM ── Du lieu user (projects/logs/error.log) nam o %APPDATA%\VisionPro ──
REM    (ghi duoc khong can admin) — app tu tao luc chay, khong tao o day.
echo [3/4] Du lieu user -^> %%APPDATA%%\VisionPro (app tu tao luc chay).
echo.

REM ── Tinh size ────────────────────────────────────────────────────────
echo [4/4] Kiem tra output...
for /f "tokens=3" %%a in ('dir /s /a-d dist\VisionUltimate ^| findstr /C:"File(s)"') do set SIZE=%%a
echo     Total size: !SIZE! bytes
echo.

echo ====================================================================
echo   BUILD THANH CONG
echo ====================================================================
echo.
echo   Output: dist\VisionUltimate\VisionUltimate.exe
echo.
echo   Cac buoc tiep theo:
echo   - Double-click dist\VisionUltimate\VisionUltimate.exe de test
echo   - Neu app khong khoi dong: doi `console=False` thanh `console=True`
echo     trong VisionPro.spec, build lai de xem error message
echo   - De tao installer .exe chuyen nghiep: dung Inno Setup
echo     (xem BUILD_GUIDE.md)
echo.

pause
endlocal
