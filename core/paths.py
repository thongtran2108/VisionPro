"""
core/paths.py — Thư mục dữ liệu user GHI ĐƯỢC (không phụ thuộc Qt).

Khi app cài vào Program Files, KHÔNG ghi cạnh .exe (user thường không có
quyền). Mọi dữ liệu do app tạo ra (projects, logs, error.log) → ghi vào
thư mục per-user, GHI ĐƯỢC mà không cần admin:
    Windows : %APPDATA%\\VisionPro
    macOS   : ~/Library/Application Support/VisionPro
    Linux   : $XDG_CONFIG_HOME/VisionPro  (mặc định ~/.config/VisionPro)

Dùng chung 1 chỗ với license (xem core/licensing) → mọi state của app nằm
gọn trong %APPDATA%\\VisionPro.
"""
from __future__ import annotations

import os
import platform

APP_DIR_NAME = "VisionPro"


def user_data_dir() -> str:
    """Thư mục gốc chứa dữ liệu user (ghi được, per-user)."""
    sysname = platform.system()
    if sysname == "Windows":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sysname == "Darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, APP_DIR_NAME)


def _ensure(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def projects_dir() -> str:
    """Nơi mặc định lưu/mở pipeline (.aoi). Tạo nếu chưa có."""
    return _ensure(os.path.join(user_data_dir(), "projects"))


def logs_dir() -> str:
    """Nơi ghi log (error.log…). Tạo nếu chưa có."""
    return _ensure(os.path.join(user_data_dir(), "logs"))
