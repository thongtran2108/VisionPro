#!/usr/bin/env python3
"""
tools/activate.py — KÍCH HOẠT MỘT-BƯỚC (CLI).

Chạy trực tiếp trên máy cần kích hoạt → tự đọc Machine ID của máy đó → ký
(HMAC bí mật dùng chung) + cài license.key vào đúng chỗ app đọc → mở app là
chạy. KHÔNG cần keypair / file .pem (app build từ cùng source nên tự khớp).

Cách dùng
─────────
  python tools/activate.py                      # active máy hiện tại (vĩnh viễn)
  python tools/activate.py --customer "Cong ty ABC"
  python tools/activate.py --expires 2027-12-31 # hết hạn theo ngày
  python tools/activate.py --duration-days 365  # hết hạn theo số ngày từ active
  python tools/activate.py --any                # license KHÔNG khoá máy
  python tools/activate.py --machine 1A2B-3C4D-5E6F-7890   # tạo cho ID khác
  python tools/activate.py --status | --deactivate
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import uuid

# Cho phép import core.licensing dù chạy từ đâu / khi đã đóng gói.
def _here() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

_ROOT = os.path.dirname(_here()) if not getattr(sys, "frozen", False) \
    else _here()
sys.path.insert(0, _ROOT)

from core import licensing                      # noqa: E402
from core.licensing import LicenseManager, sign_payload, machine_id  # noqa: E402


def _print_status(lm: LicenseManager):
    """Hiện trạng thái kích hoạt + CHÍNH XÁC các nơi app tìm license."""
    st = lm.status()
    print(f"Machine ID : {lm.machine_id()}")
    print(f"Trạng thái : {st.code} — {st.message}")
    if st.is_valid and st.payload:
        p = st.payload
        print(f"Khách      : {p.get('customer')}")
        eff = p.get("_effective_expiry")
        if eff:
            dleft = p.get("_days_left")
            extra = f"  (còn {dleft} ngày)" if dleft is not None else ""
            print(f"Hết hạn    : {eff}{extra}")
            if p.get("duration_days"):
                print(f"             ({p.get('duration_days')} ngày kể từ {p.get('_activated_on')})")
        else:
            print("Hết hạn    : Vĩnh viễn")
    print("App tìm license ở các nơi sau (theo thứ tự):")
    for p in lm.search_paths():
        if st.is_valid and st.path == p:
            mark = "✓ ĐANG DÙNG"
        elif os.path.isfile(p):
            mark = "có file"
        else:
            mark = "trống"
        print(f"   - {p}   [{mark}]")


def _pause_if_double_clicked():
    """Khi double-click trên Windows, giữ cửa sổ để đọc kết quả."""
    if os.name == "nt" and sys.stdin and sys.stdin.isatty():
        try:
            input("\nNhấn Enter để đóng…")
        except EOFError:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kích hoạt VisionPro một-bước.")
    ap.add_argument("--customer", default="", help="Tên khách hàng (ghi vào license).")
    ap.add_argument("--expires", default=None, help="Hết hạn NGÀY TUYỆT ĐỐI YYYY-MM-DD.")
    ap.add_argument("--duration-days", default=None, type=int,
                    help="Hết hạn theo SỐ NGÀY kể từ ngày active (chống chỉnh giờ).")
    ap.add_argument("--any", action="store_true", help="License không khoá máy.")
    ap.add_argument("--machine", default="", help="Active cho Machine ID khác (mặc định: máy này).")
    ap.add_argument("--edition", default="Standard", help="Bản: Standard/Pro/...")
    ap.add_argument("--license-id", default="", help="Mã license (tự sinh nếu trống).")
    ap.add_argument("--out", default="", help="Ghi thêm 1 bản license.key vào thư mục này.")
    ap.add_argument("--status", action="store_true",
                    help="Xem trạng thái kích hoạt + nơi app đọc license, rồi thoát.")
    ap.add_argument("--deactivate", action="store_true",
                    help="Gỡ license khỏi máy (về trạng thái chưa kích hoạt).")
    ap.add_argument("--reset-state", action="store_true",
                    help="(TEST) Xoá mốc chống-chỉnh-giờ + ngày active đã lưu.")
    args = ap.parse_args(argv)

    print("──────── KÍCH HOẠT VISIONPRO ────────")

    lm0 = LicenseManager()
    if args.status:
        _print_status(lm0)
        _pause_if_double_clicked()
        return 0
    if args.reset_state:
        lm0.reset_clock_state()
        print("✓ Đã xoá state chống-chỉnh-giờ (mốc ngày + ngày active).")
        _pause_if_double_clicked()
        return 0
    if args.deactivate:
        removed = lm0.deactivate()
        if removed:
            print("✓ Đã gỡ license tại:")
            for p in removed:
                print("   -", p)
        else:
            print("• Không tìm thấy license nào để gỡ.")
        print("→ Mở app sẽ hiện lại cổng Kích hoạt (nhớ TẮT app trước khi test).")
        _pause_if_double_clicked()
        return 0

    if args.expires:
        try:
            dt.date.fromisoformat(args.expires)
        except ValueError:
            print("✗ --expires phải dạng YYYY-MM-DD.")
            _pause_if_double_clicked()
            return 1

    target = "ANY" if args.any else (args.machine.strip() or machine_id())
    print(f"Máy này (Machine ID): {machine_id()}")
    print(f"Cấp license cho     : {target}")

    dur = int(args.duration_days) if args.duration_days else None
    payload = {
        "v": 1,
        "customer": args.customer or "Self-Activated",
        "license_id": args.license_id or ("VP-" + uuid.uuid4().hex[:8].upper()),
        "machine_id": target,
        "issued": dt.date.today().isoformat(),
        "expires": args.expires,
        "duration_days": dur,
        "edition": args.edition,
        "features": ["all"],
    }
    text = sign_payload(payload)

    # Cài vào store per-user (đúng nơi app đọc) → active ngay.
    lm = LicenseManager()
    st = lm.install_text(text)

    # Ghi thêm 1 bản license.key (record / để bỏ cạnh .exe nếu muốn).
    out_dir = args.out or os.getcwd()
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, licensing.LICENSE_FILENAME), "w",
                  encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass

    if dur:
        han = f"{dur} ngày kể từ hôm nay (chống chỉnh giờ)"
    else:
        han = payload['expires'] or 'Vĩnh viễn'
    if st.is_valid:
        print(f"\n✓ ĐÃ KÍCH HOẠT — {payload['customer']}")
        print(f"   Hết hạn : {han}   Bản: {payload['edition']}")
        print(f"   License : {lm.store_path}")
        print("   → Mở app là chạy được ngay.")
        _pause_if_double_clicked()
        return 0
    # install_text thất bại — thường do active cho máy KHÁC (target != máy này)
    # nên verify khoá-máy fail; file vẫn được ghi ở out_dir để mang sang máy đó.
    if st.code == "machine_mismatch":
        print(f"\n✓ Đã tạo license cho máy {target} (KHÔNG phải máy này).")
        print(f"   File: {os.path.join(out_dir, licensing.LICENSE_FILENAME)}")
        print("   → Mang file này sang máy đó & import (hoặc bỏ cạnh app).")
        _pause_if_double_clicked()
        return 0
    print(f"\n✗ Lỗi: {st.message}")
    _pause_if_double_clicked()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
