#!/usr/bin/env python3
"""
tools/activator.py — PHẦN MỀM KÍCH HOẠT (GUI).

Một app độc lập: mở lên → chọn thời hạn → bấm Kích hoạt → App Vision tự nhận
(không cần keypair, không file .pem, không build thêm). Tính năng:

  • Tự đọc Machine ID máy hiện tại.
  • TỰ NHẬN DIỆN máy đã có key hay chưa (khách + hạn + số ngày còn lại).
  • CHỌN THỜI HẠN: 30/90/180/365 ngày · vĩnh viễn · tuỳ chỉnh số ngày · theo
    ngày hết hạn.
  • Cấp cho máy này / không khoá máy (ANY) / máy khác (nhập Machine ID).
  • Ký bằng HMAC bí mật dùng chung (core.licensing) — app build từ cùng source
    nên tự xác thực được.

Chạy:   python tools/activator.py
Build:  tools/build_activator.bat   (ra dist/VisionProActivator.exe)
"""
from __future__ import annotations

import os
import sys
import uuid
import datetime as dt


# ── Cho phép import core.licensing dù chạy từ source hay đã đóng gói ──
def _app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


_HERE = _app_dir()
_ROOT = _HERE if getattr(sys, "frozen", False) else os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from core import licensing                                       # noqa: E402
from core.licensing import LicenseManager, sign_payload, machine_id  # noqa: E402

from PySide6.QtCore import Qt, QDate                             # noqa: E402
from PySide6.QtGui import QGuiApplication, QIcon                 # noqa: E402
from PySide6.QtWidgets import (                                  # noqa: E402
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDateEdit, QCheckBox,
    QPlainTextEdit, QFileDialog, QMessageBox, QFrame,
)

# ── Palette (khớp dark theme của app) ───────────────────────────────
_BG, _PANEL, _BORDER = "#0a0e1a", "#0d1220", "#1e2d45"
_ACCENT, _TEXT, _MUTED = "#00d4ff", "#e2e8f0", "#94a3b8"
_DANGER, _OK = "#ff3860", "#39ff14"

_INPUT_CSS = (
    f"QLineEdit,QComboBox,QSpinBox,QDateEdit{{background:{_BG};"
    f"border:1px solid {_BORDER};color:{_TEXT};padding:6px 8px;"
    f"border-radius:5px;font-size:13px;}}"
    f"QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDateEdit:focus{{"
    f"border-color:{_ACCENT};}}"
    f"QComboBox QAbstractItemView{{background:{_PANEL};color:{_TEXT};"
    f"selection-background-color:#1a2236;border:1px solid {_BORDER};}}")
_PRIMARY_CSS = (
    f"QPushButton{{background:{_ACCENT};border:none;border-radius:5px;"
    f"color:#000;font-weight:700;font-size:13px;padding:9px 18px;}}"
    f"QPushButton:hover{{background:#33ddff;}}"
    f"QPushButton:disabled{{background:#1e2d45;color:#64748b;}}")
_GHOST_CSS = (
    f"QPushButton{{background:{_BORDER};border:none;border-radius:5px;"
    f"color:{_MUTED};font-size:13px;padding:9px 16px;}}"
    f"QPushButton:hover{{background:#2c3e60;color:{_TEXT};}}")


class ActivatorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._lm = LicenseManager()
        self.setWindowTitle("VisionPro Activator")
        self.setMinimumWidth(560)
        self.setStyleSheet(
            f"QWidget{{background:{_PANEL};color:{_TEXT};font-size:13px;}}"
            f"QGroupBox{{border:1px solid {_BORDER};border-radius:6px;"
            f"margin-top:10px;padding-top:10px;color:{_MUTED};font-weight:600;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}"
            f"QLabel{{background:transparent;}}" + _INPUT_CSS)
        self._build_ui()
        self._refresh_status()

    # ── UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("🔑  VisionPro Activator")
        title.setStyleSheet(f"font-size:19px;font-weight:700;color:{_ACCENT};")
        root.addWidget(title)
        sub = QLabel("Chọn thời hạn rồi bấm <b>Kích hoạt</b> — App Vision tự nhận.")
        sub.setStyleSheet(f"color:{_MUTED};")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # — Máy & trạng thái —
        gb_mid = QGroupBox("Máy & trạng thái")
        g = QGridLayout(gb_mid)
        g.addWidget(QLabel("Machine ID:"), 0, 0)
        self.le_mid = QLineEdit(self._lm.machine_id())
        self.le_mid.setReadOnly(True)
        self.le_mid.setStyleSheet(
            f"QLineEdit{{background:{_BG};border:1px solid {_BORDER};"
            f"color:{_ACCENT};padding:6px 8px;border-radius:5px;"
            f"font-family:'Courier New';font-weight:700;letter-spacing:2px;}}")
        g.addWidget(self.le_mid, 0, 1)
        btn_copy = QPushButton("Copy"); btn_copy.setStyleSheet(_GHOST_CSS)
        btn_copy.clicked.connect(self._copy_mid)
        g.addWidget(btn_copy, 0, 2)
        self.lbl_status = QLabel("…")
        self.lbl_status.setWordWrap(True)
        g.addWidget(self.lbl_status, 1, 0, 1, 3)
        root.addWidget(gb_mid)

        # — Tuỳ chọn license —
        gb_opt = QGroupBox("Tuỳ chọn license")
        go = QGridLayout(gb_opt)

        go.addWidget(QLabel("Cấp cho máy:"), 0, 0)
        self.cb_target = QComboBox()
        self.cb_target.addItem("Máy này (khoá theo máy)", "this")
        self.cb_target.addItem("Không khoá máy (ANY)", "any")
        self.cb_target.addItem("Máy khác (nhập Machine ID)…", "other")
        self.cb_target.currentIndexChanged.connect(self._on_target_changed)
        go.addWidget(self.cb_target, 0, 1, 1, 2)
        self.le_other = QLineEdit()
        self.le_other.setPlaceholderText("XXXX-XXXX-XXXX-XXXX")
        self.le_other.setEnabled(False)
        go.addWidget(self.le_other, 1, 1, 1, 2)

        go.addWidget(QLabel("Thời hạn:"), 2, 0)
        self.cb_dur = QComboBox()
        self.cb_dur.addItem("30 ngày (kể từ active)", ("days", 30))
        self.cb_dur.addItem("90 ngày (kể từ active)", ("days", 90))
        self.cb_dur.addItem("180 ngày (kể từ active)", ("days", 180))
        self.cb_dur.addItem("365 ngày (kể từ active)", ("days", 365))
        self.cb_dur.addItem("Vĩnh viễn", ("perm", None))
        self.cb_dur.addItem("Tuỳ chỉnh số ngày…", ("custom", None))
        self.cb_dur.addItem("Theo ngày hết hạn…", ("expiry", None))
        self.cb_dur.setCurrentIndex(3)            # mặc định 365 ngày
        self.cb_dur.currentIndexChanged.connect(self._on_dur_changed)
        go.addWidget(self.cb_dur, 2, 1, 1, 2)

        self.sp_days = QSpinBox(); self.sp_days.setRange(1, 36500)
        self.sp_days.setValue(365); self.sp_days.setSuffix(" ngày")
        self.sp_days.setVisible(False)
        go.addWidget(self.sp_days, 3, 1, 1, 2)
        self.de_exp = QDateEdit(); self.de_exp.setCalendarPopup(True)
        self.de_exp.setDisplayFormat("yyyy-MM-dd")
        self.de_exp.setDate(QDate.currentDate().addYears(1))
        self.de_exp.setVisible(False)
        go.addWidget(self.de_exp, 4, 1, 1, 2)

        go.addWidget(QLabel("Khách hàng:"), 5, 0)
        self.le_cust = QLineEdit(); self.le_cust.setPlaceholderText("Self-Activated")
        go.addWidget(self.le_cust, 5, 1, 1, 2)

        go.addWidget(QLabel("Bản (edition):"), 6, 0)
        self.cb_edition = QComboBox(); self.cb_edition.setEditable(True)
        self.cb_edition.addItems(["Standard", "Pro", "Enterprise"])
        go.addWidget(self.cb_edition, 6, 1, 1, 2)
        root.addWidget(gb_opt)

        # — Lưu file / tuỳ chọn —
        gb_out = QGroupBox("Lưu file license & tuỳ chọn")
        gx = QGridLayout(gb_out)
        gx.addWidget(QLabel("Lưu license.key vào:"), 0, 0)
        self.le_out = QLineEdit(_HERE)
        gx.addWidget(self.le_out, 0, 1)
        b_out = QPushButton("Chọn…"); b_out.setStyleSheet(_GHOST_CSS)
        b_out.clicked.connect(self._browse_out)
        gx.addWidget(b_out, 0, 2)
        self.chk_reset = QCheckBox("Đặt lại mốc ngày trước khi active "
                                   "(test / bán lại máy)")
        self.chk_reset.setToolTip(
            "Xoá mốc chống-chỉnh-giờ + ngày active đã lưu để 'số ngày' đếm "
            "lại từ đầu. Bình thường KHÔNG cần.")
        gx.addWidget(self.chk_reset, 1, 0, 1, 3)
        root.addWidget(gb_out)

        # — Hành động —
        row = QHBoxLayout()
        b_quit = QPushButton("Thoát"); b_quit.setStyleSheet(_GHOST_CSS)
        b_quit.clicked.connect(self.close)
        row.addWidget(b_quit)
        b_deact = QPushButton("Gỡ kích hoạt"); b_deact.setStyleSheet(_GHOST_CSS)
        b_deact.clicked.connect(self._deactivate)
        row.addWidget(b_deact)
        b_refresh = QPushButton("Làm mới"); b_refresh.setStyleSheet(_GHOST_CSS)
        b_refresh.clicked.connect(self._refresh_status)
        row.addWidget(b_refresh)
        row.addStretch()
        self.b_act = QPushButton("🔑  Kích hoạt")
        self.b_act.setStyleSheet(_PRIMARY_CSS)
        self.b_act.clicked.connect(self._activate)
        row.addWidget(self.b_act)
        root.addLayout(row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{_BORDER};max-height:1px;")
        root.addWidget(sep)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        self.log.setStyleSheet(
            f"QPlainTextEdit{{background:{_BG};border:1px solid {_BORDER};"
            f"color:{_TEXT};font-family:'Courier New';font-size:12px;}}")
        root.addWidget(self.log)

    # ── Helpers ─────────────────────────────────────────────────────
    def _log(self, msg: str):
        from datetime import datetime
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {msg}")

    def _copy_mid(self):
        QGuiApplication.clipboard().setText(self.le_mid.text())
        self._log("Đã copy Machine ID.")

    def _on_target_changed(self):
        self.le_other.setEnabled(self.cb_target.currentData() == "other")

    def _on_dur_changed(self):
        mode = self.cb_dur.currentData()[0]
        self.sp_days.setVisible(mode == "custom")
        self.de_exp.setVisible(mode == "expiry")

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu", _HERE)
        if d:
            self.le_out.setText(d)

    def _refresh_status(self):
        st = self._lm.status()
        if st.is_valid:
            p = st.payload or {}
            eff = p.get("_effective_expiry")
            if eff:
                dleft = p.get("_days_left")
                han = f"{eff}" + (f" (còn {dleft} ngày)"
                                  if dleft is not None else "")
            else:
                han = "Vĩnh viễn"
            self.lbl_status.setText(
                f"✓ ĐÃ KÍCH HOẠT — {p.get('customer', '')}  •  Hết hạn: {han}")
            self.lbl_status.setStyleSheet(f"color:{_OK};font-weight:700;")
            self.b_act.setText("🔄  Cấp lại / Gia hạn")
        else:
            self.lbl_status.setText(f"✖ CHƯA kích hoạt — {st.message}")
            self.lbl_status.setStyleSheet(f"color:{_DANGER};font-weight:700;")
            self.b_act.setText("🔑  Kích hoạt")

    # ── Lựa chọn → payload ──
    def _selected_term(self):
        """Trả (expires, duration_days) theo combo thời hạn."""
        mode, val = self.cb_dur.currentData()
        if mode == "days":
            return None, int(val)
        if mode == "custom":
            return None, int(self.sp_days.value())
        if mode == "expiry":
            return self.de_exp.date().toString("yyyy-MM-dd"), None
        return None, None              # perm

    def _selected_target(self) -> str:
        t = self.cb_target.currentData()
        if t == "any":
            return "ANY"
        if t == "other":
            return self.le_other.text().strip().upper()
        return machine_id()

    # ── Actions ─────────────────────────────────────────────────────
    def _activate(self):
        target = self._selected_target()
        if self.cb_target.currentData() == "other" and not target:
            QMessageBox.warning(self, "Thiếu Machine ID",
                                "Nhập Machine ID của máy cần cấp.")
            return
        expires, duration = self._selected_term()

        if self.chk_reset.isChecked():
            self._lm.reset_clock_state()
            self._log("Đã đặt lại mốc ngày (state).")

        payload = {
            "v": 1,
            "customer": self.le_cust.text().strip() or "Self-Activated",
            "license_id": "VP-" + uuid.uuid4().hex[:8].upper(),
            "machine_id": target,
            "issued": dt.date.today().isoformat(),
            "expires": expires,
            "duration_days": duration,
            "edition": self.cb_edition.currentText().strip() or "Standard",
            "features": ["all"],
        }
        text = sign_payload(payload)

        # Ghi 1 bản file license.key (record / để mang sang máy khác).
        out_dir = self.le_out.text().strip() or _HERE
        out_file = os.path.join(out_dir, licensing.LICENSE_FILENAME)
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            self._log(f"(Không ghi được file: {e})")

        # Cài vào store per-user (đúng nơi app đọc) → active máy này ngay.
        st = self._lm.install_text(text)
        han = (f"{duration} ngày kể từ hôm nay" if duration
               else (expires or "Vĩnh viễn"))
        if st.is_valid:
            self._log(f"✓ ĐÃ KÍCH HOẠT — {payload['customer']} • {han}")
            self._log(f"  License: {self._lm.store_path}")
            QMessageBox.information(
                self, "Kích hoạt thành công",
                f"Đã kích hoạt máy này.\nThời hạn: {han}\n\n"
                f"Mở VisionPro là chạy được ngay (cả exe lẫn code).")
        elif st.code == "machine_mismatch":
            self._log(f"✓ Đã tạo license cho máy {target} (KHÔNG phải máy này).")
            self._log(f"  File: {out_file}")
            QMessageBox.information(
                self, "Đã tạo file license",
                f"License cấp cho máy:\n{target}\n\nFile đã lưu:\n{out_file}\n\n"
                f"Mang file này sang máy đó: bỏ cạnh VisionUltimate.exe hoặc "
                f"bấm 'Chọn file license…' trong app.")
        else:
            self._log(f"✖ Lỗi: {st.message}")
            QMessageBox.critical(self, "Kích hoạt thất bại", st.message)
        self._refresh_status()

    def _deactivate(self):
        if QMessageBox.question(
                self, "Gỡ kích hoạt",
                "Xoá license khỏi máy này (về trạng thái chưa kích hoạt)?\n"
                "Nhớ TẮT VisionPro trước.") != QMessageBox.Yes:
            return
        removed = self._lm.deactivate()
        if removed:
            for p in removed:
                self._log(f"Đã gỡ: {p}")
        else:
            self._log("Không tìm thấy license nào để gỡ.")
        self._refresh_status()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VisionPro Activator")
    logo = os.path.join(_ROOT, "assets", "logo.png")
    if os.path.isfile(logo):
        app.setWindowIcon(QIcon(logo))
    w = ActivatorWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
