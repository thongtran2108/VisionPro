@echo off
REM ══════════════════════════════════════════════════════════════════════
REM   Build PHAN MEM ACTIVATE rieng (GUI) -> dist\VisionProActivator.exe
REM   Chay trong venv da co: PySide6, cryptography, pyinstaller.
REM   (Cong cu nha phat hanh — KHONG ship private key cho khach.)
REM ══════════════════════════════════════════════════════════════════════
setlocal

where pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [X] Chua co pyinstaller. Chay: pip install pyinstaller
    pause & exit /b 1
)

REM Build tu thu muc goc project de PyInstaller thay package 'core'.
pushd "%~dp0.."

pyinstaller --noconfirm --clean --onefile --windowed ^
    --name VisionProActivator ^
    --paths "." ^
    --hidden-import core.licensing ^
    --icon "assets\logo.png" ^
    "tools\activator.py"

popd

echo.
echo ====================================================================
echo   Output: dist\VisionProActivator.exe
echo.
echo   * KHONG can file .pem. Activator va app dung chung bi mat HMAC trong
echo     core\licensing.py (build tu cung source nen tu khop).
echo   * Mang VisionProActivator.exe sang may dich -> chon thoi han -> Kich
echo     hoat. Doi bi mat? Sua _LICENSE_SECRET roi build lai CA HAI.
echo ====================================================================
pause
endlocal
