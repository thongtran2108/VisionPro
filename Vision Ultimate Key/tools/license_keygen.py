#!/usr/bin/env python3
"""
tools/license_keygen.py — Phát hành file license.key (nhà phát hành).

Dùng khi KHÔNG tiện chạy activator trên máy khách: khách gửi Machine ID →
bạn tạo file license.key rồi gửi lại (khách bỏ cạnh app hoặc import).

License ký bằng HMAC bí mật dùng chung trong core/licensing.py — KHÔNG cần
keypair. App build từ cùng source sẽ tự xác thực được.

Cách dùng
─────────
1) Lấy Machine ID của máy khách (khách đọc trong hộp Kích hoạt, hoặc chạy
   ngay trên máy đó):
       python tools/license_keygen.py --machine-id

2) Phát hành license cho 1 máy:
       python tools/license_keygen.py --issue \
           --machine 1A2B-3C4D-5E6F-7890 \
           --customer "Cong ty ABC" \
           --out ABC_license.key
   Tuỳ chọn: --expires 2027-12-31     (ngày tuyệt đối)
             --duration-days 365      (số ngày kể từ ngày active)
             --edition Pro  --license-id VP-0001
             --machine ANY            (license KHÔNG khoá máy)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

# Cho phép import core.licensing khi chạy từ bất kỳ đâu.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)

from core import licensing  # noqa: E402


def cmd_machine_id(args):
    print(licensing.machine_id())
    return 0


def _auto_license_id() -> str:
    import uuid
    return "VP-" + uuid.uuid4().hex[:8].upper()


def cmd_issue(args):
    if args.expires:
        try:
            dt.date.fromisoformat(args.expires)
        except ValueError:
            print("✗ --expires phải dạng YYYY-MM-DD.")
            return 1

    dur = int(args.duration_days) if args.duration_days else None
    payload = {
        "v": 1,
        "customer": args.customer,
        "license_id": args.license_id or _auto_license_id(),
        "machine_id": args.machine,
        "issued": dt.date.today().isoformat(),
        "expires": args.expires,            # None = vĩnh viễn (ngày tuyệt đối)
        "duration_days": dur,               # None = không giới hạn theo ngày active
        "edition": args.edition,
        "features": ["all"],
    }
    text = licensing.sign_payload(payload)

    out = args.out or f"{licensing.LICENSE_FILENAME}"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    # Tự kiểm tra lại (verify) — máy phát hành thường khác máy đích nên bỏ qua
    # phần khoá máy, chỉ xác nhận chữ ký (HMAC) + cấu trúc.
    st = licensing.verify_text(text, expect_machine=args.machine
                               if args.machine != "ANY" else None)
    if dur:
        han = f"{dur} ngày kể từ ngày active"
    else:
        han = payload['expires'] or 'Vĩnh viễn'
    print(f"✓ Đã phát hành license → {out}")
    print(f"   Khách: {payload['customer']}  |  Máy: {payload['machine_id']}")
    print(f"   Hết hạn: {han}  |  Bản: {payload['edition']}")
    sig_ok = st.code in ("valid", "machine_mismatch")  # mismatch = ký OK, khác máy
    print(f"   Chữ ký: {'OK' if sig_ok else 'LỖI (' + st.code + ')'}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        description="VisionPro license keygen (nhà phát hành).")
    p.add_argument("--machine-id", action="store_true",
                   help="In Machine ID của máy hiện tại rồi thoát.")
    p.add_argument("--issue", action="store_true", help="Phát hành 1 license.")
    p.add_argument("--machine", default="ANY",
                   help="Machine ID đích (mặc định ANY = không khoá máy).")
    p.add_argument("--customer", default="Unknown", help="Tên khách hàng.")
    p.add_argument("--license-id", default="", help="Mã license (tự sinh nếu trống).")
    p.add_argument("--expires", default=None, help="Hết hạn NGÀY TUYỆT ĐỐI YYYY-MM-DD.")
    p.add_argument("--duration-days", default=None, type=int,
                   help="Hết hạn theo SỐ NGÀY kể từ ngày active (chống chỉnh giờ).")
    p.add_argument("--edition", default="Standard", help="Bản: Standard/Pro/...")
    p.add_argument("--out", default="", help="File license xuất ra.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.machine_id:
        return cmd_machine_id(args)
    if args.issue:
        return cmd_issue(args)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
