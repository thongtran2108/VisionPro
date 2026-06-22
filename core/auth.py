"""
core/auth.py — Đăng nhập admin + đổi mật khẩu.

• Tài khoản mặc định lần đầu: VisionUltimate / 8888 (đổi được mật khẩu).
• Mật khẩu KHÔNG lưu plaintext — lưu PBKDF2-HMAC-SHA256 (salt ngẫu nhiên,
  200k vòng) trong QSettings (HKCU trên Windows). Có thể đổi mật khẩu.
• "Đăng nhập" mở khoá quyền Sửa/Chạy tool; "Đăng xuất" khoá lại. Trạng thái
  đăng nhập chỉ tồn tại trong phiên chạy (in-memory), không lưu đĩa.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Tuple

from PySide6.QtCore import QSettings

DEFAULT_USERNAME = "VisionUltimate"
DEFAULT_PASSWORD = "8888"
# Tên đăng nhập mặc định CŨ → tự đổi sang DEFAULT_USERNAME khi mở app
# (giữ nguyên mật khẩu đã có). Cho máy đã cài bản trước với tên cũ.
_LEGACY_USERNAMES = ("VUTM",)
_ITERATIONS = 200_000
_GROUP = "auth"


def _hash(password: str, salt: bytes, iterations: int = _ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                               salt, iterations)


class AuthManager:
    """Quản lý credential admin + trạng thái đăng nhập của phiên."""

    def __init__(self, settings: QSettings | None = None):
        self._s = settings or QSettings()
        self._authed = False
        self._ensure_seeded()

    # ── Persistence ────────────────────────────────────────────────
    def _ensure_seeded(self):
        """Lần đầu chạy (chưa có hash) → seed VisionUltimate / 8888.
        Máy đã cài bản cũ (tên VUTM) → tự đổi tên sang VisionUltimate,
        GIỮ NGUYÊN mật khẩu đang dùng."""
        self._s.beginGroup(_GROUP)
        has = self._s.value("hash", "")
        cur_user = str(self._s.value("username", "") or "")
        self._s.endGroup()
        if not has:
            self._set_credentials(DEFAULT_USERNAME, DEFAULT_PASSWORD)
        elif cur_user in _LEGACY_USERNAMES:
            self._s.beginGroup(_GROUP)
            self._s.setValue("username", DEFAULT_USERNAME)
            self._s.endGroup()
            self._s.sync()

    def _set_credentials(self, username: str, password: str):
        salt = os.urandom(16)
        h = _hash(password, salt)
        self._s.beginGroup(_GROUP)
        self._s.setValue("username", username)
        self._s.setValue("salt", salt.hex())
        self._s.setValue("hash", h.hex())
        self._s.setValue("iterations", _ITERATIONS)
        self._s.endGroup()
        self._s.sync()

    # ── Queries ────────────────────────────────────────────────────
    @property
    def username(self) -> str:
        self._s.beginGroup(_GROUP)
        u = self._s.value("username", DEFAULT_USERNAME) or DEFAULT_USERNAME
        self._s.endGroup()
        return str(u)

    def verify(self, username: str, password: str) -> bool:
        self._s.beginGroup(_GROUP)
        u = str(self._s.value("username", DEFAULT_USERNAME) or DEFAULT_USERNAME)
        salt_hex = self._s.value("salt", "")
        hash_hex = self._s.value("hash", "")
        iters = int(self._s.value("iterations", _ITERATIONS) or _ITERATIONS)
        self._s.endGroup()
        if not salt_hex or not hash_hex:
            return False
        if (username or "").strip() != u:
            return False
        try:
            salt = bytes.fromhex(str(salt_hex))
            expected = bytes.fromhex(str(hash_hex))
        except ValueError:
            return False
        # So sánh hằng-thời-gian chống timing attack.
        return hmac.compare_digest(_hash(password, salt, iters), expected)

    # ── Session ────────────────────────────────────────────────────
    @property
    def is_authenticated(self) -> bool:
        return self._authed

    def login(self, username: str, password: str) -> bool:
        if self.verify(username, password):
            self._authed = True
        return self._authed

    def logout(self):
        self._authed = False

    # ── Mutations ──────────────────────────────────────────────────
    def change_password(self, old_password: str, new_password: str
                        ) -> Tuple[bool, str]:
        """Đổi mật khẩu. Trả (ok, message)."""
        if not self.verify(self.username, old_password):
            return False, "Mật khẩu hiện tại không đúng."
        new_password = new_password or ""
        if len(new_password) < 4:
            return False, "Mật khẩu mới phải có ít nhất 4 ký tự."
        self._set_credentials(self.username, new_password)
        return True, "Đổi mật khẩu thành công."

    def reset_to_default(self):
        """Khôi phục VisionUltimate / 8888 (khi quên mật khẩu — gọi từ tool quản trị)."""
        self._set_credentials(DEFAULT_USERNAME, DEFAULT_PASSWORD)
