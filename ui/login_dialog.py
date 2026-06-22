"""
ui/login_dialog.py — Đăng nhập admin, đổi mật khẩu, và kích hoạt license.

  • LoginDialog          : nhập user/mật khẩu để mở khoá Sửa/Chạy tool.
  • ChangePasswordDialog : đổi mật khẩu admin.
  • ActivationDialog     : cổng kích hoạt khi mở app — hiện Machine ID +
                           import file license.key. Chưa active → khoá hẳn.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFrame, QApplication,
                               QFileDialog, QWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication

from core.i18n import tr
from core.auth import AuthManager
from core.licensing import LicenseManager

# Palette dùng chung (khớp dark theme app).
_BG      = "#0a0e1a"
_PANEL   = "#0d1220"
_BORDER  = "#1e2d45"
_ACCENT  = "#00d4ff"
_TEXT    = "#e2e8f0"
_MUTED   = "#94a3b8"
_DANGER  = "#ff3860"
_OK      = "#39ff14"

_INPUT_CSS = (
    f"QLineEdit{{background:{_BG};border:1px solid {_BORDER};color:{_TEXT};"
    f"padding:8px 10px;border-radius:5px;font-size:13px;}}"
    f"QLineEdit:focus{{border-color:{_ACCENT};}}")
_PRIMARY_CSS = (
    f"QPushButton{{background:{_ACCENT};border:none;border-radius:5px;"
    f"color:#000;font-weight:700;font-size:13px;padding:8px 16px;}}"
    f"QPushButton:hover{{background:#33ddff;}}"
    f"QPushButton:disabled{{background:#1e2d45;color:#64748b;}}")
_GHOST_CSS = (
    f"QPushButton{{background:{_BORDER};border:none;border-radius:5px;"
    f"color:{_MUTED};font-size:13px;padding:8px 16px;}}"
    f"QPushButton:hover{{background:#2c3e60;color:{_TEXT};}}")


def _eye_button(line: QLineEdit) -> QPushButton:
    """Nút hiện/ẩn mật khẩu cho 1 QLineEdit."""
    b = QPushButton("👁")
    b.setCheckable(True)
    b.setFixedWidth(38)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet(
        f"QPushButton{{background:{_BG};border:1px solid {_BORDER};"
        f"border-radius:5px;color:{_MUTED};font-size:13px;padding:8px 0;}}"
        f"QPushButton:checked{{color:{_ACCENT};border-color:{_ACCENT};}}")

    def _toggle(on):
        line.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password)
    b.toggled.connect(_toggle)
    return b


class _BaseDialog(QDialog):
    """Khung dialog tối, không viền hệ thống rườm rà."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setStyleSheet(f"QDialog{{background:{_PANEL};color:{_TEXT};}}"
                           f"QLabel{{color:{_TEXT};background:transparent;}}")


class LoginDialog(_BaseDialog):
    """Đăng nhập admin để mở khoá Sửa/Chạy tool."""

    def __init__(self, auth: AuthManager, parent=None):
        super().__init__(tr("Đăng nhập"), parent)
        self._auth = auth
        self.setFixedWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(12)

        title = QLabel(tr("🔒  Đăng nhập Admin"))
        title.setStyleSheet(f"font-size:16px;font-weight:700;color:{_ACCENT};")
        root.addWidget(title)
        sub = QLabel(tr("Đăng nhập để được Sửa tham số và Chạy tool."))
        sub.setStyleSheet(f"font-size:12px;color:{_MUTED};")
        sub.setWordWrap(True)
        root.addWidget(sub)

        root.addWidget(self._sep())

        root.addWidget(self._lbl(tr("Tài khoản")))
        self._user = QLineEdit(self._auth.username)
        self._user.setStyleSheet(_INPUT_CSS)
        root.addWidget(self._user)

        root.addWidget(self._lbl(tr("Mật khẩu")))
        pw_row = QHBoxLayout(); pw_row.setSpacing(6)
        self._pw = QLineEdit()
        self._pw.setEchoMode(QLineEdit.Password)
        self._pw.setStyleSheet(_INPUT_CSS)
        self._pw.setPlaceholderText(tr("Nhập mật khẩu…"))
        pw_row.addWidget(self._pw, 1)
        pw_row.addWidget(_eye_button(self._pw))
        root.addLayout(pw_row)

        self._err = QLabel("")
        self._err.setStyleSheet(f"color:{_DANGER};font-size:12px;")
        self._err.setWordWrap(True)
        root.addWidget(self._err)

        btns = QHBoxLayout(); btns.addStretch()
        cancel = QPushButton(tr("Huỷ")); cancel.setStyleSheet(_GHOST_CSS)
        cancel.clicked.connect(self.reject)
        login = QPushButton(tr("Đăng nhập")); login.setStyleSheet(_PRIMARY_CSS)
        login.setDefault(True)
        login.clicked.connect(self._try_login)
        btns.addWidget(cancel); btns.addWidget(login)
        root.addLayout(btns)

        self._pw.returnPressed.connect(self._try_login)
        self._user.returnPressed.connect(lambda: self._pw.setFocus())
        self._pw.setFocus()

    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet(f"font-size:11px;color:{_MUTED};")
        return l

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"color:{_BORDER};background:{_BORDER};max-height:1px;")
        return f

    def _try_login(self):
        if self._auth.login(self._user.text(), self._pw.text()):
            self.accept()
        else:
            self._err.setText(tr("✖  Sai tài khoản hoặc mật khẩu."))
            self._pw.selectAll(); self._pw.setFocus()


class ChangePasswordDialog(_BaseDialog):
    """Đổi mật khẩu admin."""

    def __init__(self, auth: AuthManager, parent=None):
        super().__init__(tr("Đổi mật khẩu"), parent)
        self._auth = auth
        self.setFixedWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 20); root.setSpacing(10)

        title = QLabel(tr("🔑  Đổi mật khẩu Admin"))
        title.setStyleSheet(f"font-size:16px;font-weight:700;color:{_ACCENT};")
        root.addWidget(title)

        for cap, attr, ph in (
            ("Mật khẩu hiện tại", "_old", "Mật khẩu đang dùng…"),
            ("Mật khẩu mới", "_new", "Tối thiểu 4 ký tự…"),
            ("Nhập lại mật khẩu mới", "_cfm", "Gõ lại mật khẩu mới…"),
        ):
            lbl = QLabel(tr(cap)); lbl.setStyleSheet(f"font-size:11px;color:{_MUTED};")
            root.addWidget(lbl)
            row = QHBoxLayout(); row.setSpacing(6)
            le = QLineEdit(); le.setEchoMode(QLineEdit.Password)
            le.setStyleSheet(_INPUT_CSS); le.setPlaceholderText(tr(ph))
            setattr(self, attr, le)
            row.addWidget(le, 1); row.addWidget(_eye_button(le))
            root.addLayout(row)

        self._msg = QLabel("")
        self._msg.setStyleSheet(f"color:{_DANGER};font-size:12px;")
        self._msg.setWordWrap(True)
        root.addWidget(self._msg)

        btns = QHBoxLayout(); btns.addStretch()
        cancel = QPushButton(tr("Huỷ")); cancel.setStyleSheet(_GHOST_CSS)
        cancel.clicked.connect(self.reject)
        save = QPushButton(tr("Lưu")); save.setStyleSheet(_PRIMARY_CSS)
        save.setDefault(True); save.clicked.connect(self._save)
        btns.addWidget(cancel); btns.addWidget(save)
        root.addLayout(btns)
        self._cfm.returnPressed.connect(self._save)

    def _save(self):
        if self._new.text() != self._cfm.text():
            self._msg.setStyleSheet(f"color:{_DANGER};font-size:12px;")
            self._msg.setText(tr("✖  Mật khẩu nhập lại không khớp."))
            return
        ok, msg = self._auth.change_password(self._old.text(), self._new.text())
        if ok:
            self.accept()
        else:
            self._msg.setStyleSheet(f"color:{_DANGER};font-size:12px;")
            self._msg.setText("✖  " + msg)


class ActivationDialog(_BaseDialog):
    """Cổng kích hoạt khi mở app — hiện Machine ID + import file license.key.
    accept() chỉ khi license đã hợp lệ; reject() = thoát app."""

    def __init__(self, lm: LicenseManager, parent=None):
        super().__init__(tr("Kích hoạt VisionPro"), parent)
        self._lm = lm
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 22); root.setSpacing(12)

        title = QLabel(tr("🔑  Kích hoạt chương trình"))
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{_ACCENT};")
        root.addWidget(title)

        desc = QLabel(tr(
            "Chương trình cần file <b>license.key</b> hợp lệ để chạy. Gửi "
            "<b>Machine ID</b> dưới đây cho nhà cung cấp để nhận file license "
            "dành riêng cho máy này, sau đó bấm <b>“Chọn file license…”</b>."))
        desc.setStyleSheet(f"font-size:12px;color:{_MUTED};")
        desc.setWordWrap(True)
        root.addWidget(desc)

        # Machine ID + copy
        root.addWidget(self._lbl(tr("MACHINE ID (máy này)")))
        mid_row = QHBoxLayout(); mid_row.setSpacing(6)
        self._mid = QLineEdit(self._lm.machine_id())
        self._mid.setReadOnly(True)
        self._mid.setStyleSheet(
            f"QLineEdit{{background:{_BG};border:1px solid {_BORDER};"
            f"color:{_ACCENT};padding:8px 10px;border-radius:5px;"
            f"font-family:'Courier New';font-size:14px;font-weight:700;"
            f"letter-spacing:2px;}}")
        mid_row.addWidget(self._mid, 1)
        copy = QPushButton(tr("Copy")); copy.setStyleSheet(_GHOST_CSS)
        copy.setCursor(Qt.PointingHandCursor)
        copy.clicked.connect(self._copy_mid)
        mid_row.addWidget(copy)
        root.addLayout(mid_row)

        # Trạng thái
        self._status = QLabel("")
        self._status.setStyleSheet(f"font-size:13px;color:{_MUTED};")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        root.addWidget(self._sep())

        btns = QHBoxLayout()
        quit_b = QPushButton(tr("Thoát")); quit_b.setStyleSheet(_GHOST_CSS)
        quit_b.clicked.connect(self.reject)
        btns.addWidget(quit_b); btns.addStretch()
        imp = QPushButton(tr("📂  Chọn file license…"))
        imp.setStyleSheet(_GHOST_CSS); imp.clicked.connect(self._import)
        btns.addWidget(imp)
        self._continue = QPushButton(tr("Tiếp tục"))
        self._continue.setStyleSheet(_PRIMARY_CSS)
        self._continue.clicked.connect(self.accept)
        btns.addWidget(self._continue)
        root.addLayout(btns)

        self._refresh_status()

    def _lbl(self, t):
        l = QLabel(t); l.setStyleSheet(f"font-size:11px;color:{_MUTED};")
        return l

    def _sep(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"color:{_BORDER};background:{_BORDER};max-height:1px;")
        return f

    def _copy_mid(self):
        QGuiApplication.clipboard().setText(self._mid.text())
        self._status.setText(tr("✓  Đã copy Machine ID vào clipboard."))
        self._status.setStyleSheet(f"font-size:13px;color:{_OK};")

    def _refresh_status(self):
        st = self._lm.status()
        if st.is_valid:
            p = st.payload or {}
            who = p.get("customer", "")
            eff = p.get("_effective_expiry")
            if eff:
                dleft = p.get("_days_left")
                exp = eff + (f" (còn {dleft} ngày)" if dleft is not None else "")
            else:
                exp = tr("Vĩnh viễn")
            self._status.setText(f"✓  Đã kích hoạt — {who}  •  Hết hạn: {exp}")
            self._status.setStyleSheet(f"font-size:13px;color:{_OK};font-weight:700;")
            self._continue.setEnabled(True)
        else:
            self._status.setText("✖  " + st.message)
            self._status.setStyleSheet(f"font-size:13px;color:{_DANGER};")
            self._continue.setEnabled(False)

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Chọn file license"), "",
            tr("License (*.key *.lic *.txt);;Tất cả (*)"))
        if not path:
            return
        st = self._lm.import_license(path)
        if st.is_valid:
            self._refresh_status()
        else:
            self._status.setText("✖  " + st.message)
            self._status.setStyleSheet(f"font-size:13px;color:{_DANGER};")
            self._continue.setEnabled(False)
