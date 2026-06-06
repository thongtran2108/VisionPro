; ════════════════════════════════════════════════════════════════════
;   AOI Vision Pro — Inno Setup installer script
; ════════════════════════════════════════════════════════════════════
;
; CÁCH DÙNG:
;   1. Build app bằng PyInstaller trước:    .\build.bat
;      → tạo ra folder dist\VisionPro\
;
;   2. Tải Inno Setup:
;      https://jrsoftware.org/isinfo.php  (free, ~3 MB)
;
;   3. Mở file này bằng Inno Setup Compiler
;      → Build → Compile  (Ctrl+F9)
;
;   4. Output: installer\VisionPro_Setup_v1.0.0.exe
;      ↑ Đây là FILE DUY NHẤT giao cho khách hàng
;
; ════════════════════════════════════════════════════════════════════

#define MyAppName "AOI Vision Ultimate"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "VisionPro"
#define MyAppExeName "VisionUltimate.exe"
#define MyAppId "{{A8F3E2B1-9C4D-4E7A-B5F2-1D8E9C3A7B4F}"

[Setup]
; ── App identification ──────────────────────────────────────────────
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppCopyright=Copyright (C) 2026 {#MyAppPublisher}

; ── Cài đặt vào Program Files (cần admin) ──────────────────────────
DefaultDirName={autopf}\VisionPro
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin

; ── Output ──────────────────────────────────────────────────────────
OutputDir=installer
OutputBaseFilename=VisionUltimate_Setup_v{#MyAppVersion}

; ── Nén tối đa (giảm size installer ~30-50%) ───────────────────────
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes

; ── 64-bit only ─────────────────────────────────────────────────────
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; ── UI ──────────────────────────────────────────────────────────────
WizardStyle=modern
SetupIconFile=assets\logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
ShowLanguageDialog=auto

; ── Yêu cầu Windows 10 trở lên ──────────────────────────────────────
MinVersion=10.0

; ── License (tuỳ chọn — uncomment nếu có file LICENSE.txt) ─────────
; LicenseFile=LICENSE.txt

; ── Thông tin hiển thị trong Add/Remove Programs ────────────────────
AppPublisherURL=
AppSupportURL=
AppUpdatesURL=
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Setup
VersionInfoCopyright=Copyright (C) 2026 {#MyAppPublisher}


[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"


[Tasks]
Name: "desktopicon"; Description: "Tạo shortcut trên Desktop"; \
    GroupDescription: "Additional icons:"; Flags: checkedonce
Name: "quicklaunchicon"; Description: "Tạo shortcut Quick Launch"; \
    GroupDescription: "Additional icons:"; Flags: unchecked


[Files]
; ── Toàn bộ folder dist\VisionPro\ ──────────────────────────────────
; recursesubdirs: bao gồm sub-folder (assets, vendor, _internal/lib...)
; createallsubdirs: tạo cả folder rỗng
Source: "dist\VisionUltimate\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; ── Optional: kèm MVS SDK installer nếu có (HikRobot camera) ───────
; Source: "MVS_SDK_V2.4.1.msi"; DestDir: "{tmp}"; Flags: deleteafterinstall


[Icons]
; ── Start Menu (luôn tạo) ──────────────────────────────────────────
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"

Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

; ── Desktop (theo lựa chọn user) ───────────────────────────────────
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon

; ── Quick Launch (theo lựa chọn user) ──────────────────────────────
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; \
    Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; \
    Tasks: quicklaunchicon


[Run]
; ── Optional: chạy MVS SDK installer trước khi cài (nếu có) ────────
; Filename: "msiexec.exe"; Parameters: "/i ""{tmp}\MVS_SDK_V2.4.1.msi"" /qb"; \
;     StatusMsg: "Đang cài MVS Camera SDK..."; Check: NeedsMVSSDK

; ── Chạy app sau khi cài xong (theo lựa chọn user) ─────────────────
Filename: "{app}\{#MyAppExeName}"; \
    Description: "Khởi động {#MyAppName} ngay"; \
    Flags: nowait postinstall skipifsilent


[UninstallDelete]
; ── Xoá folder logs/, projects/ khi uninstall ──────────────────────
; (User data — Inno Setup mặc định không xoá file user tạo)
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\error.log"
; KHÔNG xoá {app}\projects vì có thể chứa file user quan trọng


[Code]
// ── Custom logic Pascal ─────────────────────────────────────────────

// Check xem MVS Camera SDK đã cài chưa
function NeedsMVSSDK: Boolean;
begin
  Result := not FileExists(
    ExpandConstant('{commonpf32}\MVS\Development\Win64_x64\MvCameraControl.dll'));
end;

// Cảnh báo nếu user chạy installer trên máy không 64-bit
function InitializeSetup(): Boolean;
begin
  if not IsWin64 then
  begin
    MsgBox('AOI Vision Pro yêu cầu Windows 64-bit.' + #13#10 +
           'Hệ điều hành hiện tại không được hỗ trợ.',
           mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
