"""
core/sfc.py — SFC/MES Integration (Scanner + API GET + API POST)

Gộp 3 mảng "thông tin sản phẩm" thường gặp trong dây chuyền inspection:
1. Scanner serial (đọc SN/QR/DataMatrix qua cổng COM)
2. API GET (lookup SN trên SFC/MES để check eligibility, lấy nhân viên, token…)
3. API POST (gửi kết quả lên SFC sau khi pipeline chạy)

Setup thông qua ``ui/sfc_dialog.py``, chạy độc lập với pipeline; lúc cần
giá trị từ pipeline (vd ``patmax.found``) thì SfcManager resolve qua
``FlowGraph`` được inject.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Helpers ───────────────────────────────────────────────────────
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_.-]+)\}")


def list_serial_ports() -> List[Tuple[str, str]]:
    """Liệt kê các cổng COM/serial trên máy. Trả list (device, description).

    Yêu cầu pyserial; nếu chưa cài trả [].
    """
    try:
        from serial.tools import list_ports
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for p in list_ports.comports():
        desc = p.description or ""
        if p.manufacturer and p.manufacturer not in desc:
            desc = f"{desc} ({p.manufacturer})"
        out.append((p.device, desc.strip() or p.device))
    return out


def parse_hex_bytes(s: str) -> bytes:
    """Parse '02 F4 03' / '0x02,0xF4,0x03' / '02f403' → bytes. Empty → b''."""
    if not s:
        return b""
    cleaned = s.replace(",", " ").replace("0x", "").replace("0X", "")
    parts = cleaned.split()
    if len(parts) == 1:
        h = parts[0]
        if len(h) % 2 != 0:
            raise ValueError(f"Hex string length phải chẵn: {s!r}")
        return bytes.fromhex(h)
    return bytes(int(p, 16) for p in parts)


def parse_json_dict(s: str) -> Dict[str, Any]:
    """Parse JSON dict (vd headers). Empty/invalid → {}."""
    if not s or not s.strip():
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


# ── Config dataclasses ────────────────────────────────────────────
@dataclass
class ScannerConfig:
    port: str = ""
    baudrate: int = 9600
    bytesize: int = 8
    parity: str = "N"           # N/E/O/M/S
    stopbits: float = 1
    timeout_ms: int = 1000
    trigger_hex: str = "02 F4 03"
    read_size: int = 128
    encoding: str = "utf-8"
    strip_chars: str = "\r\n\x02\x03"
    expected_length: int = 0    # 0 = any
    contains: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApiGetConfig:
    """URL có thể chứa placeholder {SN} + {node.port}."""
    enabled: bool = False
    url_template: str = ""
    headers_json: str = "{}"
    timeout_ms: int = 5000
    expected_status: int = 200
    expected_text: str = ""
    parse_json: bool = False
    json_path: str = ""
    bypass_proxy: bool = True     # default True — đa số API nội bộ LAN

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApiPostConfig:
    """Body là chuỗi JSON; field value có thể chứa placeholder
    {SN}, {node.port}, {api_get.value}, …"""
    enabled: bool = False
    url: str = ""
    method: str = "POST"        # POST | PUT
    headers_json: str = '{"Content-Type":"application/json"}'
    body_template: str = (
        '{\n'
        '  "sn": "{SN}",\n'
        '  "result": "{patmax.found}",\n'
        '  "stationName": "OP1"\n'
        '}'
    )
    timeout_ms: int = 5000
    expected_status: int = 200
    expected_text: str = ""
    bypass_proxy: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _requests_kwargs(bypass_proxy: bool) -> dict:
    """Build kwargs cho requests. Khi bypass_proxy=True → tắt HTTP_PROXY env
    (tránh request đi qua Squid/corporate proxy → 403 / HTML error page)."""
    if bypass_proxy:
        return {"proxies": {"http": None, "https": None}, "verify": False}
    return {}


# ── Manager ───────────────────────────────────────────────────────
class SfcManager:
    """Stateful manager — giữ kết nối serial, config, snapshot SN gần nhất.

    Cách dùng:
        mgr = SfcManager()
        mgr.set_graph(graph)
        mgr.scanner = ScannerConfig(port="COM3", ...)
        sn, err = mgr.scan_once()             # ghi snapshot self.last_sn
        text, ok, err = mgr.api_get()         # dùng last_sn cho {SN}
        text, ok, err = mgr.api_post()        # resolve {node.port} từ graph
    """

    def __init__(self) -> None:
        self.scanner = ScannerConfig()
        self.api_get_cfg = ApiGetConfig()
        self.api_post_cfg = ApiPostConfig()

        self._graph = None
        self._lock = threading.Lock()
        self._serial = None    # pyserial.Serial khi đang giữ open (tuỳ chọn)
        # State được cập nhật sau mỗi lần scan/get
        self.last_sn: str = ""
        self.last_get_text: str = ""
        self.last_get_value: str = ""   # extracted theo json_path
        self.last_post_text: str = ""

        # Callbacks tuỳ chọn — UI có thể subscribe để cập nhật log/label.
        self.on_log: Optional[Callable[[str], None]] = None

    # ── Wiring ────────────────────────────────────────────────
    def set_graph(self, graph) -> None:
        self._graph = graph

    def _log(self, msg: str) -> None:
        print(msg)
        if self.on_log:
            try:
                self.on_log(msg)
            except Exception:
                pass

    # ── Placeholder resolution ────────────────────────────────
    def resolve_placeholders(self, template: str,
                             extra: Optional[Dict[str, Any]] = None
                             ) -> Tuple[str, List[str]]:
        """Replace {key} trong template.

        Keys hỗ trợ:
            {SN}              → self.last_sn
            {api_get.value}   → giá trị extract từ API GET gần nhất
            {api_get.text}    → response.text gần nhất
            {<node_id>.<port>}→ outputs port của node trong graph
            {<tool_id>.<port>}→ outputs port của node đầu tiên có tool_id đó
            extra dict        → giá trị custom thêm vào (override)

        Trả (resolved_text, missing_keys).
        """
        extra = extra or {}
        missing: List[str] = []

        def _lookup(key: str) -> Optional[str]:
            if key in extra:
                return str(extra[key])
            if key == "SN":
                return self.last_sn
            if key == "api_get.value":
                return self.last_get_value
            if key == "api_get.text":
                return self.last_get_text
            if "." in key and self._graph is not None:
                node_key, port = key.split(".", 1)
                # 1) Match by exact node_id
                node = self._graph.nodes.get(node_key)
                if node is None:
                    # 2) Match by tool_id (first node having that tool)
                    for n in self._graph.nodes.values():
                        if n.tool_id == node_key:
                            node = n
                            break
                if node is not None and port in node.outputs:
                    v = node.outputs.get(port)
                    if v is None:
                        return ""
                    return str(v)
            return None

        def _sub(m: re.Match) -> str:
            key = m.group(1)
            val = _lookup(key)
            if val is None:
                missing.append(key)
                return m.group(0)
            return val

        return _PLACEHOLDER_RE.sub(_sub, template), missing

    # ── Scanner ───────────────────────────────────────────────
    def scan_once(self) -> Tuple[str, str]:
        """Mở port, gửi trigger, đọc 1 lần rồi đóng. Trả (data, error)."""
        cfg = self.scanner
        try:
            import serial
        except ImportError:
            err = "pyserial chưa cài (pip install pyserial)"
            self._log(f"[Scanner] {err}")
            return "", err

        if not cfg.port:
            return "", "Chưa chọn cổng COM"

        parity_map = {"N": serial.PARITY_NONE, "E": serial.PARITY_EVEN,
                      "O": serial.PARITY_ODD, "M": serial.PARITY_MARK,
                      "S": serial.PARITY_SPACE}
        stopbits_map = {1: serial.STOPBITS_ONE,
                        1.5: serial.STOPBITS_ONE_POINT_FIVE,
                        2: serial.STOPBITS_TWO}

        ser = None
        try:
            trigger = parse_hex_bytes(cfg.trigger_hex)
            ser = serial.Serial(
                port=cfg.port, baudrate=cfg.baudrate,
                bytesize=cfg.bytesize,
                parity=parity_map.get(cfg.parity.upper(), serial.PARITY_NONE),
                stopbits=stopbits_map.get(cfg.stopbits, serial.STOPBITS_ONE),
                timeout=cfg.timeout_ms / 1000.0,
            )
            if trigger:
                ser.write(trigger)
            raw = ser.read(cfg.read_size)
            if not raw:
                raw = ser.readline()
            text = raw.decode(cfg.encoding, errors="replace").strip(cfg.strip_chars)
            with self._lock:
                self.last_sn = text
            self._log(f"[Scanner] {cfg.port}@{cfg.baudrate} → {text!r}")
            return text, ""
        except Exception as e:
            self._log(f"[Scanner] Error: {e}")
            return "", str(e)
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass

    # ── API GET ───────────────────────────────────────────────
    def api_get(self, override_sn: Optional[str] = None
                ) -> Tuple[str, bool, str]:
        """Gọi GET. Returns (response_text, pass, error).

        URL resolve từ ``api_get_cfg.url_template`` với {SN} = override_sn
        (nếu truyền) hoặc self.last_sn.
        """
        cfg = self.api_get_cfg
        if not cfg.url_template:
            return "", False, "URL template trống"

        try:
            import requests
        except ImportError:
            err = "requests chưa cài (pip install requests)"
            return "", False, err

        sn = override_sn if override_sn is not None else self.last_sn
        url, missing = self.resolve_placeholders(
            cfg.url_template, extra={"SN": sn})
        if missing:
            self._log(f"[ApiGet] Placeholder missing: {missing}")

        headers = parse_json_dict(cfg.headers_json)
        kwargs = _requests_kwargs(cfg.bypass_proxy)
        try:
            # Im lặng warning InsecureRequestWarning khi verify=False
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass
            r = requests.get(url, headers=headers,
                             timeout=cfg.timeout_ms / 1000.0, **kwargs)
            text = r.text
            status = r.status_code
            value = ""
            if cfg.parse_json:
                try:
                    js = r.json()
                    if cfg.json_path:
                        cur: Any = js
                        for k in cfg.json_path.split("."):
                            if isinstance(cur, dict) and k in cur:
                                cur = cur[k]
                            else:
                                cur = None
                                break
                        value = "" if cur is None else str(cur)
                except Exception:
                    pass
            with self._lock:
                self.last_get_text = text
                self.last_get_value = value
            ok = (status == cfg.expected_status) and (
                (cfg.expected_text in text) if cfg.expected_text else True)
            self._log(f"[ApiGet] {url} → {status} "
                      f"({'PASS' if ok else 'FAIL'})")
            return text, ok, ""
        except Exception as e:
            self._log(f"[ApiGet] Error: {e}")
            return "", False, str(e)

    # ── API POST ──────────────────────────────────────────────
    def api_post(self) -> Tuple[str, bool, str]:
        """Resolve placeholder trong body_template (JSON) rồi POST."""
        cfg = self.api_post_cfg
        if not cfg.url:
            return "", False, "URL trống"

        try:
            import requests
        except ImportError:
            return "", False, "requests chưa cài (pip install requests)"

        body_str, missing = self.resolve_placeholders(cfg.body_template)
        if missing:
            self._log(f"[ApiPost] Placeholder missing: {missing}")
        try:
            body = json.loads(body_str)
        except Exception as e:
            msg = f"Body JSON parse fail sau substitute: {e}"
            self._log(f"[ApiPost] {msg}\n  body={body_str!r}")
            return "", False, msg

        headers = parse_json_dict(cfg.headers_json)
        kwargs = _requests_kwargs(cfg.bypass_proxy)
        try:
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass
            if cfg.method.upper() == "PUT":
                r = requests.put(cfg.url, json=body, headers=headers,
                                 timeout=cfg.timeout_ms / 1000.0, **kwargs)
            else:
                r = requests.post(cfg.url, json=body, headers=headers,
                                  timeout=cfg.timeout_ms / 1000.0, **kwargs)
            text = r.text
            status = r.status_code
            with self._lock:
                self.last_post_text = text
            ok = (status == cfg.expected_status) and (
                (cfg.expected_text in text) if cfg.expected_text else True)
            self._log(f"[ApiPost] {cfg.method} {cfg.url} → {status} "
                      f"({'PASS' if ok else 'FAIL'}) body={body!r}")
            return text, ok, ""
        except Exception as e:
            self._log(f"[ApiPost] Error: {e}")
            return "", False, str(e)
