"""
core/licensing.py — Kích hoạt chương trình (offline, KHÔNG cần keypair).

Mô hình đơn giản — HMAC bí mật dùng chung:
  • App và Activator build từ CÙNG source này → cùng _LICENSE_SECRET, nên
    license do Activator ký, App tự xác thực được. KHÔNG cần public/private
    key, KHÔNG cần "build rồi nhúng key", KHÔNG cần file .pem.
  • File `license.key` = base64(JSON envelope) gồm:
        { "data": {payload}, "mac": hmac_sha256(secret, canonical(data)) }
  • License KHOÁ THEO MÁY: payload.machine_id phải khớp machine_id() của máy
    đang chạy (đặt "ANY" để cấp license không khoá máy).
  • Hết hạn (tuỳ chọn): payload.expires="YYYY-MM-DD" / duration_days / null.

Bảo mật: bí mật nằm trong app → mức "chống sao chép thông thường", không
chống reverse-engineer chuyên sâu. Muốn mạnh hơn: đổi _LICENSE_SECRET (rồi
build lại app + activator). Module KHÔNG phụ thuộc Qt (tool CLI tái dùng được).
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import hmac
import json
import os
import platform
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

# ════════════════════════════════════════════════════════════════════
#  BÍ MẬT DÙNG CHUNG (HMAC) — App + Activator build từ source này nên tự
#  khớp. KHÔNG cần keypair. Đổi chuỗi này = vô hiệu MỌI license cũ (phải
#  build lại CẢ app lẫn activator cho khớp). Giữ kín source là đủ cho mức
#  chống sao chép thông thường.
# ════════════════════════════════════════════════════════════════════
_LICENSE_SECRET = b"VisionPro-License-HMAC-v1-change-me-7Qm2kZx9Lp3Rt"


def _license_key() -> bytes:
    return hashlib.sha256(_LICENSE_SECRET).digest()


LICENSE_FILENAME = "license.key"
_MID_NAMESPACE = b"VisionPro-MachineID-v1"

# Mã trạng thái → message tiếng Việt cho UI.
_STATUS_MSG = {
    "valid":            "License hợp lệ — chương trình đã được kích hoạt.",
    "missing":          "Chưa có file license — chương trình chưa được kích hoạt.",
    "malformed":        "File license sai định dạng hoặc đã hỏng.",
    "bad_signature":    "Chữ ký license không hợp lệ (file bị sửa hoặc khác phiên bản app).",
    "machine_mismatch": "License không dành cho máy này (Machine ID không khớp).",
    "expired":          "License đã hết hạn.",
    "not_yet_valid":    "License chưa tới ngày hiệu lực.",
}


@dataclass
class LicenseStatus:
    ok: bool
    code: str                       # valid|missing|malformed|bad_signature|...
    message: str
    payload: Optional[dict] = None
    path: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.ok and self.code == "valid"


# ── Machine fingerprint ────────────────────────────────────────────
def _raw_machine_token() -> str:
    """Chuỗi định danh phần cứng/OS ổn định nhất có thể, theo nền tảng."""
    parts = []
    sysname = platform.system()
    if sysname == "Windows":
        # MachineGuid: ổn định theo lần cài Windows, không đổi khi reboot.
        try:
            import winreg
            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                    0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
                guid, _ = winreg.QueryValueEx(k, "MachineGuid")
                parts.append(str(guid))
        except Exception:
            pass
    elif sysname == "Linux":
        for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                with open(p, "r") as f:
                    parts.append(f.read().strip())
                    break
            except OSError:
                continue
    elif sysname == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                text=True, timeout=5)
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    parts.append(line.split('"')[-2])
                    break
        except Exception:
            pass
    # Fallback chung: MAC address (getnode). Đủ ổn định cho dev/test.
    if not parts:
        parts.append(f"{uuid.getnode():012x}")
    parts.append(platform.machine())   # arch để giảm trùng giữa các máy
    return "|".join(parts)


def machine_id() -> str:
    """Machine ID hiển thị cho user (16 hex, nhóm 4: XXXX-XXXX-XXXX-XXXX)."""
    digest = hashlib.sha256(_MID_NAMESPACE + _raw_machine_token().encode()).hexdigest()
    h = digest[:16].upper()
    return "-".join(h[i:i + 4] for i in range(0, 16, 4))


# ── Ký / verify ────────────────────────────────────────────────────
def _canonical(data: dict) -> bytes:
    """Bytes chuẩn hoá của payload để ký + verify nhất quán 2 phía."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sign_payload(payload: dict) -> str:
    """Tạo nội dung file license.key (base64) từ payload, ký bằng HMAC-SHA256
    với bí mật dùng chung. App build từ cùng source sẽ tự xác thực được."""
    mac = hmac.new(_license_key(), _canonical(payload), hashlib.sha256).hexdigest()
    envelope = {"data": payload, "mac": mac}
    blob = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(blob).decode("ascii")


def _today() -> _dt.date:
    return _dt.date.today()


def _parse_date(s) -> Optional[_dt.date]:
    if not s:
        return None
    try:
        return _dt.date.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


def verify_text(text: str, *, expect_machine: Optional[str] = None,
                check_time: bool = True) -> LicenseStatus:
    """Verify nội dung 1 license (chuỗi base64). expect_machine=None → dùng
    machine_id() của máy hiện tại; truyền chuỗi khác để test.
    check_time=False → CHỈ kiểm tra chữ ký + khoá máy, KHÔNG kiểm tra hạn
    (để LicenseManager tự áp cơ chế chống chỉnh giờ)."""
    text = (text or "").strip()
    if not text:
        return LicenseStatus(False, "missing", _STATUS_MSG["missing"])
    # Parse envelope
    try:
        blob = base64.b64decode(text, validate=True)
        envelope = json.loads(blob.decode("utf-8"))
        data = envelope["data"]
        mac = str(envelope["mac"])
    except Exception:
        return LicenseStatus(False, "malformed", _STATUS_MSG["malformed"])
    # Verify HMAC (bí mật dùng chung app + activator)
    expected = hmac.new(_license_key(), _canonical(data),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        return LicenseStatus(False, "bad_signature",
                             _STATUS_MSG["bad_signature"], payload=data)
    # Khoá máy
    lic_machine = str(data.get("machine_id", "ANY"))
    if lic_machine != "ANY":
        cur = expect_machine if expect_machine is not None else machine_id()
        if lic_machine.upper() != cur.upper():
            return LicenseStatus(False, "machine_mismatch",
                                 _STATUS_MSG["machine_mismatch"], payload=data)
    # Hiệu lực thời gian (bỏ qua khi check_time=False → LicenseManager tự lo)
    if check_time:
        today = _today()
        iso = data.get("issued")
        if iso:
            try:
                if today < _dt.date.fromisoformat(iso):
                    return LicenseStatus(False, "not_yet_valid",
                                         _STATUS_MSG["not_yet_valid"], payload=data)
            except ValueError:
                pass
        exp = data.get("expires")
        if exp:
            try:
                if today > _dt.date.fromisoformat(exp):
                    return LicenseStatus(False, "expired", _STATUS_MSG["expired"],
                                         payload=data)
            except ValueError:
                return LicenseStatus(False, "malformed", _STATUS_MSG["malformed"],
                                     payload=data)
    return LicenseStatus(True, "valid", _STATUS_MSG["valid"], payload=data)


def verify_file(path: str, *, expect_machine: Optional[str] = None,
                check_time: bool = True) -> LicenseStatus:
    if not path or not os.path.isfile(path):
        return LicenseStatus(False, "missing", _STATUS_MSG["missing"], path=path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return LicenseStatus(False, "malformed", _STATUS_MSG["malformed"], path=path)
    st = verify_text(text, expect_machine=expect_machine, check_time=check_time)
    st.path = path
    return st


# ── Nơi lưu license của app ────────────────────────────────────────
def _app_dir() -> str:
    """Thư mục chứa app (cạnh .exe khi đóng gói, hoặc gốc project khi dev)."""
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _user_data_dir() -> str:
    """Thư mục ghi được cho license đã import (per-user)."""
    sysname = platform.system()
    if sysname == "Windows":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sysname == "Darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "VisionPro")


# ════════════════════════════════════════════════════════════════════
#  CHỐNG CHỈNH ĐỒNG HỒ (anti clock-rollback)
#  Lưu MỐC NGÀY CAO NHẤT từng thấy (last_seen) + NGÀY ACTIVE của từng
#  license. Hết hạn tính theo effective_now = max(hôm_nay, last_seen) nên
#  chỉnh giờ LÙI không kéo dài được license. Lưu ở file (có HMAC chống sửa
#  tay) + registry HKCU (Windows) cho khó xoá.
# ════════════════════════════════════════════════════════════════════
_STATE_FILE = ".license_state"
_STATE_SECRET = b"VisionPro-State-v1-x7Qm2k"
_REG_STATE_PATH = r"Software\VisionPro\State"


def _state_file_path() -> str:
    return os.path.join(_user_data_dir(), _STATE_FILE)


def _state_hmac(data_b64: str) -> str:
    key = hashlib.sha256(_STATE_SECRET + _raw_machine_token().encode()).digest()
    return hmac.new(key, data_b64.encode(), hashlib.sha256).hexdigest()


def _merge_state(dst: dict, src: dict):
    ls = src.get("last_seen")
    if ls and (dst["last_seen"] is None or ls > dst["last_seen"]):
        dst["last_seen"] = ls
    for k, v in (src.get("acts") or {}).items():
        # Ngày active SỚM NHẤT thắng (không cho reset bằng cách kích hoạt lại).
        if v and (k not in dst["acts"] or v < dst["acts"][k]):
            dst["acts"][k] = v


def _read_state_registry() -> Optional[dict]:
    if platform.system() != "Windows":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_STATE_PATH) as k:
            ls, _ = winreg.QueryValueEx(k, "ls")
            acts_raw, _ = winreg.QueryValueEx(k, "acts")
        return {"last_seen": ls or None, "acts": json.loads(acts_raw or "{}")}
    except Exception:
        return None


def _write_state_registry(state: dict):
    if platform.system() != "Windows":
        return
    try:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _REG_STATE_PATH) as k:
            winreg.SetValueEx(k, "ls", 0, winreg.REG_SZ,
                              state.get("last_seen") or "")
            winreg.SetValueEx(k, "acts", 0, winreg.REG_SZ,
                              json.dumps(state.get("acts", {})))
    except Exception:
        pass


def _clear_state_registry():
    if platform.system() != "Windows":
        return
    try:
        import winreg
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _REG_STATE_PATH)
    except Exception:
        pass


def _read_state() -> dict:
    state = {"last_seen": None, "acts": {}}
    try:
        with open(_state_file_path(), "r", encoding="utf-8") as f:
            env = json.load(f)
        d_b64 = env.get("d", "")
        if d_b64 and env.get("m") == _state_hmac(d_b64):
            _merge_state(state, json.loads(base64.b64decode(d_b64).decode()))
        # HMAC sai = file bị sửa tay → bỏ qua (coi như chưa có), không tin.
    except Exception:
        pass
    reg = _read_state_registry()
    if reg:
        _merge_state(state, reg)
    return state


def _write_state(state: dict):
    payload = json.dumps({"last_seen": state.get("last_seen"),
                          "acts": state.get("acts", {})},
                         separators=(",", ":"))
    d_b64 = base64.b64encode(payload.encode()).decode()
    env = {"d": d_b64, "m": _state_hmac(d_b64)}
    try:
        os.makedirs(os.path.dirname(_state_file_path()), exist_ok=True)
        with open(_state_file_path(), "w", encoding="utf-8") as f:
            json.dump(env, f)
    except OSError:
        pass
    _write_state_registry(state)


class LicenseManager:
    """Quản lý trạng thái kích hoạt: tìm license ở nhiều nơi, verify, import."""

    def __init__(self):
        self._store = os.path.join(_user_data_dir(), LICENSE_FILENAME)

    @property
    def store_path(self) -> str:
        return self._store

    def search_paths(self) -> list:
        """Thứ tự ưu tiên: cạnh app trước, rồi tới store per-user."""
        return [os.path.join(_app_dir(), LICENSE_FILENAME), self._store]

    def machine_id(self) -> str:
        return machine_id()

    def status(self) -> LicenseStatus:
        """Trạng thái license tốt nhất tìm được. Nếu không có file nào →
        'missing'; nếu có file nhưng sai → trả lỗi cụ thể của file đầu tiên."""
        first_err: Optional[LicenseStatus] = None
        for p in self.search_paths():
            if not os.path.isfile(p):
                continue
            # check_time=False: chữ ký + khoá máy OK trước, hạn để
            # _apply_time_guard xử (mốc cao nhất luôn được cập nhật).
            st = verify_file(p, check_time=False)
            if st.is_valid:
                return self._apply_time_guard(st)
            if first_err is None:
                first_err = st
        if first_err is not None:
            return first_err
        return LicenseStatus(False, "missing", _STATUS_MSG["missing"])

    def _touch_state(self, license_id, *, record_activation: bool = True):
        """Đọc state, tính effective_now = max(hôm nay, mốc cao nhất), ghi nhận
        ngày active cho license_id (nếu chưa), cập nhật mốc cao nhất + persist.
        Trả (effective, activated_on, rolled_back)."""
        today = _today()
        state = _read_state()
        last_seen = _parse_date(state.get("last_seen"))
        effective = today if last_seen is None else max(today, last_seen)
        acts = dict(state.get("acts", {}))
        lic_id = str(license_id or "default")
        if record_activation and lic_id not in acts:
            acts[lic_id] = effective.isoformat()
        activated_on = _parse_date(acts.get(lic_id)) or effective
        new_last = effective if last_seen is None else max(effective, last_seen)
        if (state.get("last_seen") != new_last.isoformat()
                or state.get("acts") != acts):
            _write_state({"last_seen": new_last.isoformat(), "acts": acts})
        rolled_back = last_seen is not None and today < last_seen
        return effective, activated_on, rolled_back

    def _apply_time_guard(self, st: LicenseStatus) -> LicenseStatus:
        """Chống chỉnh đồng hồ. effective_now = max(hôm nay, mốc cao nhất từng
        thấy). Hết hạn tính theo expires (tuyệt đối) HOẶC duration_days (số ngày
        KỂ TỪ ngày active) — lấy mốc sớm hơn. Chỉnh giờ LÙI không kéo dài hạn."""
        data = dict(st.payload or {})
        effective, activated_on, rolled_back = self._touch_state(
            data.get("license_id"))

        # Chưa tới ngày hiệu lực?
        issued = _parse_date(data.get("issued"))
        if issued is not None and effective < issued:
            return LicenseStatus(False, "not_yet_valid",
                                 _STATUS_MSG["not_yet_valid"], payload=data,
                                 path=st.path)

        # Hạn hiệu lực = sớm nhất giữa expires và (activated_on + duration_days).
        limit: Optional[_dt.date] = _parse_date(data.get("expires"))
        dur = data.get("duration_days")
        if dur:
            try:
                d_end = activated_on + _dt.timedelta(days=int(dur))
                limit = d_end if limit is None else min(limit, d_end)
            except (ValueError, TypeError):
                pass

        if limit is not None:
            data["_activated_on"] = activated_on.isoformat()
            data["_effective_expiry"] = limit.isoformat()
            data["_days_left"] = (limit - effective).days
            st.payload = data
            if effective > limit:
                msg = _STATUS_MSG["expired"]
                if rolled_back:
                    msg = "License đã hết hạn (phát hiện chỉnh ngược đồng hồ máy)."
                return LicenseStatus(False, "expired", msg, payload=data,
                                     path=st.path)
        else:
            st.payload = data
        return st

    def reset_clock_state(self) -> bool:
        """Xoá state chống-chỉnh-giờ (mốc ngày + ngày active). CHỈ dùng để
        TEST hoặc bán lại máy — production không nên gọi (cho phép reset
        duration). Trả True nếu có xoá file."""
        ok = False
        try:
            if os.path.isfile(_state_file_path()):
                os.remove(_state_file_path())
                ok = True
        except OSError:
            pass
        _clear_state_registry()
        return ok

    def is_activated(self) -> bool:
        return self.status().is_valid

    def install_text(self, text: str) -> LicenseStatus:
        """Verify chuỗi license rồi ghi vào store per-user → kích hoạt.
        Dùng bởi activator (kích hoạt một-bước) và import dialog."""
        st = verify_text(text)
        if not st.is_valid:
            return st
        try:
            os.makedirs(os.path.dirname(self._store), exist_ok=True)
            with open(self._store, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            return LicenseStatus(False, "malformed",
                                 f"Không ghi được license vào máy: {e}")
        # Ghi nhận NGÀY ACTIVE ngay lúc cài → duration đếm từ đây (không phải
        # lần chạy status đầu tiên).
        self._touch_state((st.payload or {}).get("license_id"))
        return self.status()

    def import_license(self, src_path: str) -> LicenseStatus:
        """Verify file nguồn; nếu hợp lệ copy vào store per-user → kích hoạt."""
        st = verify_file(src_path)
        if not st.is_valid:
            return st
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            return LicenseStatus(False, "malformed",
                                 f"Không đọc được file license: {e}")
        return self.install_text(text)

    def deactivate(self) -> list:
        """Xoá license khỏi MỌI nơi app đọc (cạnh app + store per-user) →
        quay về trạng thái chưa kích hoạt. Trả về danh sách path đã xoá."""
        removed = []
        for p in self.search_paths():
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    removed.append(p)
            except OSError:
                pass
        return removed

