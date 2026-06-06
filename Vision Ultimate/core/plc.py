"""
core/plc.py — PLC connectivity for Vision Ultimate

Hỗ trợ 5 dòng PLC:
  - Omron CP2E       (FINS over TCP, port 9600)
  - Omron NX1P2      (FINS over UDP, port 9600)
  - Inovance H3U/H5U (Modbus TCP, port 502)
  - Mitsubishi FX5U  (MC Protocol 3E / SLMP binary over TCP, port 5000)
  - Panasonic FP     (MEWTOCOL-COM over serial — COM port + baud, cần pyserial)

API thống nhất qua lớp ``PLCDriver``:
    connect(), disconnect()
    read_word(area, address)        -> int      (16-bit unsigned)
    write_word(area, address, val)  -> None
    read_bit(area, address, bit)    -> bool
    write_bit(area, address, bit, val) -> None

``PLCManager`` cung cấp polling trigger ở thread riêng và gửi kết quả về PLC.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional


# ── Memory area codes ─────────────────────────────────────────────
class MemoryArea(str, Enum):
    DM_WORD  = "DM_WORD"     # Data Memory word    (D0, D100…)
    CIO_WORD = "CIO_WORD"    # CIO area word
    CIO_BIT  = "CIO_BIT"     # CIO area bit
    DM_BIT   = "DM_BIT"      # DM bit
    W_WORD   = "W_WORD"      # Work word
    H_WORD   = "H_WORD"      # Holding word


# ── FINS hex codes for Omron area access ──────────────────────────
_FINS_AREA_CODES = {
    MemoryArea.DM_WORD:  b'\x82',
    MemoryArea.DM_BIT:   b'\x02',
    MemoryArea.CIO_WORD: b'\xB0',
    MemoryArea.CIO_BIT:  b'\x30',
    MemoryArea.W_WORD:   b'\xB1',
    MemoryArea.H_WORD:   b'\xB2',
}

_FINS_MEMORY_AREA_READ  = b'\x01\x01'
_FINS_MEMORY_AREA_WRITE = b'\x01\x02'


class FinsError(IOError):
    """FINS end_code khác 00 00 → PLC trả lỗi."""

    # MRES/SRES codes phổ biến nhất; xem Omron W342 spec để đầy đủ.
    _MESSAGES = {
        0x0000: "Normal completion",
        0x0101: "Local node not in network",
        0x0102: "Token timeout",
        0x0103: "Retries failed",
        0x0104: "Too many send frames",
        0x0105: "Node address range error",
        0x0106: "Node address duplication",
        0x0201: "Destination node not in network",
        0x0202: "No node",
        0x0203: "Communications controller error",
        0x0204: "Controller error",
        0x0205: "Response timeout",
        0x1001: "Command too long",
        0x1002: "Command too short",
        0x1003: "Elements/data don't match",
        0x1004: "Command format error",
        0x1101: "Area class error",
        0x1103: "Start address out of range",
        0x1104: "End address out of range",
        0x110B: "Response too long",
        0x110C: "Parameter error",
        0x2002: "Read-only area",
        0x2003: "FROM/PLC setup not finished",
        0x2102: "Cannot write in run mode",
        0x2502: "Setting error",
    }

    def __init__(self, end_code: int):
        self.end_code = end_code
        msg = self._MESSAGES.get(end_code, f"FINS end_code 0x{end_code:04X}")
        super().__init__(f"FINS error 0x{end_code:04X}: {msg}")


def _check_fins_end_code(frame: bytes) -> None:
    """Frame ở đây là FINS response frame (không có FINS/TCP header)."""
    if len(frame) < 14:
        raise IOError(f"FINS response too short: {len(frame)} bytes")
    end_code = int.from_bytes(frame[12:14], 'big')
    if end_code != 0:
        raise FinsError(end_code)


# ── Base driver ───────────────────────────────────────────────────
class PLCDriver(ABC):
    """Giao diện chung cho mọi driver PLC."""

    def __init__(self, ip: str, port: int, timeout: float = 2.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.connected = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_word(self, area: MemoryArea, address: int) -> int: ...

    @abstractmethod
    def write_word(self, area: MemoryArea, address: int, value: int) -> None: ...

    def read_bit(self, area: MemoryArea, address: int, bit: int = 0) -> bool:
        word = self.read_word(self._word_area_for(area), address)
        return bool((word >> bit) & 1)

    def write_bit(self, area: MemoryArea, address: int, bit: int, value: bool) -> None:
        word_area = self._word_area_for(area)
        cur = self.read_word(word_area, address)
        mask = 1 << bit
        new = (cur | mask) if value else (cur & ~mask & 0xFFFF)
        self.write_word(word_area, address, new)

    @staticmethod
    def _word_area_for(area: MemoryArea) -> MemoryArea:
        if area == MemoryArea.CIO_BIT:
            return MemoryArea.CIO_WORD
        if area == MemoryArea.DM_BIT:
            return MemoryArea.DM_WORD
        return area


# ── Omron CP2E — FINS over TCP ────────────────────────────────────
class OmronCP2E(PLCDriver):
    """FINS/TCP driver cho Omron CP2E."""

    _FINS_MAGIC = b'\x46\x49\x4e\x53'   # b'FINS'

    def __init__(self, ip: str, port: int = 9600, timeout: float = 2.0):
        super().__init__(ip, port, timeout)
        self.sock: Optional[socket.socket] = None
        self.da1 = b'\x00'  # destination node (PLC)
        self.sa1 = b'\x00'  # source node (PC)
        self._sid = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))

        # FINS/TCP node-address handshake (cmd=0)
        handshake = (self._FINS_MAGIC
                     + (12).to_bytes(4, 'big')      # length = 8 + 4
                     + (0).to_bytes(4, 'big')       # cmd 0 = client→server
                     + (0).to_bytes(4, 'big')       # error code
                     + (0).to_bytes(4, 'big'))      # client node addr (auto)
        self.sock.sendall(handshake)
        resp = self._recv_exact(24)
        # Response: magic(4) + len(4) + cmd(4) + err(4) + clientNode(4) + serverNode(4)
        self.sa1 = resp[19:20]
        self.da1 = resp[23:24]
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _next_sid(self) -> bytes:
        self._sid = (self._sid + 1) & 0xFF
        return self._sid.to_bytes(1, 'big')

    def _recv_exact(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PLC closed the connection")
            buf += chunk
        return buf

    def _send_fins(self, command_code: bytes, body: bytes) -> bytes:
        # FINS command header (10 bytes) + command_code (2) + body
        fins_cmd = (
            b'\x80\x00\x02'          # ICF, RSV, GCT
            + b'\x00' + self.da1 + b'\x00'   # DNA, DA1, DA2
            + b'\x00' + self.sa1 + b'\x00'   # SNA, SA1, SA2
            + self._next_sid()
            + command_code
            + body
        )
        # FINS/TCP frame header (cmd=2 = FINS frame send)
        length = 8 + len(fins_cmd)
        frame_hdr = (self._FINS_MAGIC
                     + length.to_bytes(4, 'big')
                     + (2).to_bytes(4, 'big')
                     + (0).to_bytes(4, 'big'))
        with self._lock:
            self.sock.sendall(frame_hdr + fins_cmd)
            # Response: TCP header (16) + FINS header (10) + cmd code (2) + endcode (2) + data
            hdr = self._recv_exact(16)
            resp_len = int.from_bytes(hdr[4:8], 'big')
            body = self._recv_exact(resp_len - 8)
        return body

    def read_word(self, area: MemoryArea, address: int) -> int:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = code + address.to_bytes(2, 'big') + b'\x00' + (1).to_bytes(2, 'big')
        resp = self._send_fins(_FINS_MEMORY_AREA_READ, body)
        _check_fins_end_code(resp)
        return int.from_bytes(resp[14:16], 'big')

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = (code + address.to_bytes(2, 'big') + b'\x00'
                + (1).to_bytes(2, 'big')
                + (value & 0xFFFF).to_bytes(2, 'big'))
        resp = self._send_fins(_FINS_MEMORY_AREA_WRITE, body)
        _check_fins_end_code(resp)


# ── Omron NX1P2 — FINS over UDP ───────────────────────────────────
class OmronNX1P2(PLCDriver):
    """FINS/UDP driver cho Omron NX1P2."""

    def __init__(self, ip: str, port: int = 9600, timeout: float = 2.0,
                 dest_node: int = 1, src_node: int = 25):
        super().__init__(ip, port, timeout)
        self.dest_node = dest_node
        self.src_node = src_node
        self.sock: Optional[socket.socket] = None
        self._sid = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        # Bind ephemeral local port; OS picks one (avoid clash with PLC)
        self.sock.bind(('', 0))
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _next_sid(self) -> bytes:
        self._sid = (self._sid + 1) & 0xFF
        return self._sid.to_bytes(1, 'big')

    def _send_fins(self, command_code: bytes, body: bytes) -> bytes:
        frame = (
            b'\x80\x00\x07'
            + b'\x00' + self.dest_node.to_bytes(1, 'big') + b'\x00'
            + b'\x00' + self.src_node.to_bytes(1, 'big') + b'\x00'
            + self._next_sid()
            + command_code
            + body
        )
        with self._lock:
            self.sock.sendto(frame, (self.ip, self.port))
            data, _ = self.sock.recvfrom(4096)
        return data

    def read_word(self, area: MemoryArea, address: int) -> int:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = code + address.to_bytes(2, 'big') + b'\x00' + (1).to_bytes(2, 'big')
        resp = self._send_fins(_FINS_MEMORY_AREA_READ, body)
        _check_fins_end_code(resp)
        return int.from_bytes(resp[14:16], 'big')

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        if not self.connected:
            self.connect()
        code = _FINS_AREA_CODES[self._word_area_for(area)]
        body = (code + address.to_bytes(2, 'big') + b'\x00'
                + (1).to_bytes(2, 'big')
                + (value & 0xFFFF).to_bytes(2, 'big'))
        resp = self._send_fins(_FINS_MEMORY_AREA_WRITE, body)
        _check_fins_end_code(resp)


# ── Inovance H3U / H5U — Modbus TCP ───────────────────────────────
class InovanceH3UH5U(PLCDriver):
    """Modbus/TCP driver cho Inovance H3U / H5U.

    Address mapping (Inovance):
      DM_WORD  → D register  (FC 03 / 06)
      CIO_BIT  → M coil      (FC 01 / 05)
    """

    def __init__(self, ip: str, port: int = 502, timeout: float = 2.0,
                 unit_id: int = 1):
        super().__init__(ip, port, timeout)
        self.unit_id = unit_id
        self.sock: Optional[socket.socket] = None
        self._tid = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _next_tid(self) -> int:
        self._tid = (self._tid + 1) & 0xFFFF
        return self._tid

    def _request(self, function_code: int, payload: bytes) -> bytes:
        if not self.connected:
            self.connect()
        tid = self._next_tid()
        pdu = bytes([function_code]) + payload
        adu = (tid.to_bytes(2, 'big')
               + b'\x00\x00'                           # protocol id
               + (len(pdu) + 1).to_bytes(2, 'big')     # length = pdu + unit
               + bytes([self.unit_id])
               + pdu)
        with self._lock:
            self.sock.sendall(adu)
            hdr = self._recv_exact(7)
            length = int.from_bytes(hdr[4:6], 'big') - 1
            body = self._recv_exact(length)
        if body[0] & 0x80:
            raise IOError(f"Modbus exception {body[1]} for FC {function_code}")
        return body[1:]

    def _recv_exact(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PLC closed the connection")
            buf += chunk
        return buf

    def read_word(self, area: MemoryArea, address: int) -> int:
        # FC 03 — Read holding registers
        payload = address.to_bytes(2, 'big') + (1).to_bytes(2, 'big')
        data = self._request(0x03, payload)
        # data: byte_count (1) + register_values (2)
        return int.from_bytes(data[1:3], 'big')

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        # FC 06 — Write single register
        payload = address.to_bytes(2, 'big') + (value & 0xFFFF).to_bytes(2, 'big')
        self._request(0x06, payload)

    def read_bit(self, area: MemoryArea, address: int, bit: int = 0) -> bool:
        if area == MemoryArea.CIO_BIT:
            # FC 01 — Read M coil
            payload = address.to_bytes(2, 'big') + (1).to_bytes(2, 'big')
            data = self._request(0x01, payload)
            return bool(data[1] & 0x01)
        # DM_BIT và mọi area khác: đọc word rồi mask (H3U không có Modbus
        # mapping cho từng bit của D register).
        return super().read_bit(area, address, bit)

    def write_bit(self, area: MemoryArea, address: int, bit: int, value: bool) -> None:
        if area == MemoryArea.CIO_BIT:
            # FC 05 — Write single M coil
            payload = address.to_bytes(2, 'big') + (b'\xFF\x00' if value else b'\x00\x00')
            self._request(0x05, payload)
            return
        # DM_BIT: read-modify-write trên D word
        super().write_bit(area, address, bit, value)


# ── Mitsubishi FX5U / iQ-F — MC Protocol 3E (SLMP) binary over TCP ──
class MitsubishiFX5U(PLCDriver):
    """MC Protocol 3E (binary) driver cho Mitsubishi FX5U / iQ-F qua Ethernet.

    FX5U có cổng LAN tích hợp, hỗ trợ SLMP / MC Protocol khung 3E binary.
    Cấu hình phía PLC (GX Works3): Module Parameter > Ethernet Port >
    External Device Configuration > thêm "SLMP Connection", đặt Port (vd
    5000), Protocol = TCP, rồi Write parameter + reset nguồn.

    Map vùng nhớ (MemoryArea → device Mitsubishi):
        DM_WORD  → D   (Data register, word)   ← hay dùng nhất
        CIO_WORD → W   (Link register, word)
        W_WORD   → W   (Link register, word)
        H_WORD   → R   (File register, word)
        CIO_BIT  → M   (Internal relay — đọc/ghi bit native)
        DM_BIT   → D   (bit trong word D, read-modify-write)
    """

    # Mã device cho khung 3E binary: (mã 1 byte, is_bit)
    _DEVICE_CODES = {
        'D': (0xA8, False), 'W': (0xB4, False), 'R': (0xAF, False),
        'M': (0x90, True),  'X': (0x9C, True),  'Y': (0x9D, True),
        'L': (0x92, True),  'B': (0xA0, True),
    }
    # Ánh xạ MemoryArea (Omron-centric) → device letter của Mitsubishi
    _AREA_DEVICE = {
        MemoryArea.DM_WORD:  'D',
        MemoryArea.DM_BIT:   'D',
        MemoryArea.CIO_WORD: 'W',
        MemoryArea.W_WORD:   'W',
        MemoryArea.H_WORD:   'R',
        MemoryArea.CIO_BIT:  'M',
    }

    def __init__(self, ip: str, port: int = 5000, timeout: float = 2.0):
        super().__init__(ip, port, timeout)
        self.sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self.connected:
            self.disconnect()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        self.connected = True

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.connected = False

    def _recv_exact(self, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PLC closed the connection")
            buf += chunk
        return buf

    def _send_recv(self, request_data: bytes) -> bytes:
        """Bọc command vào khung 3E binary, gửi, trả phần data của response."""
        subheader      = b'\x50\x00'      # 5000: 3E frame request
        network_no     = b'\x00'          # mạng cục bộ
        pc_no          = b'\xFF'           # CPU cục bộ
        dst_module_io  = b'\xFF\x03'       # 03FF: CPU đích
        dst_module_sta = b'\x00'
        cpu_timer      = b'\x10\x00'       # thời gian chờ CPU (16 × 250ms)
        data_len = struct.pack('<H', len(cpu_timer) + len(request_data))
        frame = (subheader + network_no + pc_no + dst_module_io +
                 dst_module_sta + data_len + cpu_timer + request_data)
        with self._lock:
            self.sock.sendall(frame)
            # Response 3E: 9 byte header, [7:9] = độ dài (end_code + data)
            head = self._recv_exact(9)
            length = struct.unpack('<H', head[7:9])[0]
            body = self._recv_exact(length)
        end_code = struct.unpack('<H', body[0:2])[0]
        if end_code != 0:
            raise IOError("Mitsubishi MC end_code 0x%04X" % end_code)
        return body[2:]

    @classmethod
    def _device_bytes(cls, dev: str, addr: int) -> bytes:
        """3 byte địa chỉ (little endian) + 1 byte mã device."""
        code, _ = cls._DEVICE_CODES[dev]
        return struct.pack('<I', addr)[:3] + struct.pack('B', code)

    def _read_words_dev(self, dev: str, addr: int, count: int) -> list:
        req = (b'\x01\x04' + b'\x00\x00' +              # batch read, word
               self._device_bytes(dev, addr) + struct.pack('<H', count))
        data = self._send_recv(req)
        return [struct.unpack('<H', data[i:i + 2])[0]
                for i in range(0, count * 2, 2)]

    def _write_words_dev(self, dev: str, addr: int, values) -> None:
        payload = b''.join(struct.pack('<H', v & 0xFFFF) for v in values)
        req = (b'\x01\x14' + b'\x00\x00' +              # batch write, word
               self._device_bytes(dev, addr) +
               struct.pack('<H', len(values)) + payload)
        self._send_recv(req)

    def _read_bits_dev(self, dev: str, addr: int, count: int) -> list:
        req = (b'\x01\x04' + b'\x01\x00' +              # batch read, bit
               self._device_bytes(dev, addr) + struct.pack('<H', count))
        data = self._send_recv(req)
        bits = []
        for b in data:                                  # 2 bit / byte
            bits.append((b >> 4) & 1)
            bits.append(b & 1)
        return bits[:count]

    def _write_bits_dev(self, dev: str, addr: int, values) -> None:
        payload = bytearray()
        for i in range(0, len(values), 2):
            hi = 0x10 if values[i] else 0x00
            lo = 0x01 if (i + 1 < len(values) and values[i + 1]) else 0x00
            payload.append(hi | lo)
        req = (b'\x01\x14' + b'\x01\x00' +              # batch write, bit
               self._device_bytes(dev, addr) +
               struct.pack('<H', len(values)) + bytes(payload))
        self._send_recv(req)

    def read_word(self, area: MemoryArea, address: int) -> int:
        if not self.connected:
            self.connect()
        dev = self._AREA_DEVICE[self._word_area_for(area)]
        return self._read_words_dev(dev, address, 1)[0]

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        if not self.connected:
            self.connect()
        dev = self._AREA_DEVICE[self._word_area_for(area)]
        self._write_words_dev(dev, address, [value & 0xFFFF])

    def read_bit(self, area: MemoryArea, address: int, bit: int = 0) -> bool:
        dev = self._AREA_DEVICE.get(area)
        if dev and self._DEVICE_CODES[dev][1]:          # device bit native (M…)
            if not self.connected:
                self.connect()
            return bool(self._read_bits_dev(dev, address, 1)[0])
        # DM_BIT và word-device khác: đọc word rồi mask
        return super().read_bit(area, address, bit)

    def write_bit(self, area: MemoryArea, address: int, bit: int, value: bool) -> None:
        dev = self._AREA_DEVICE.get(area)
        if dev and self._DEVICE_CODES[dev][1]:          # device bit native (M…)
            if not self.connected:
                self.connect()
            self._write_bits_dev(dev, address, [1 if value else 0])
            return
        super().write_bit(area, address, bit, value)


# ── Panasonic FP-X / FP-XH / FP7 — MEWTOCOL-COM over serial ────────
class PanasonicFP(PLCDriver):
    """MEWTOCOL-COM (serial) driver cho Panasonic FP-X / FP-XH / FP7.

    Khác các PLC Ethernet ở trên: dòng FP kết nối qua cổng COM (RS-232 /
    RS-485). Vì vậy 2 field ``ip`` / ``port`` được TÁI SỬ DỤNG:
        ip   → tên cổng COM   (vd 'COM6' Windows, '/dev/ttyUSB0' Linux)
        port → baud rate       (vd 9600, 19200, 115200)

    Chỉ hỗ trợ thanh ghi dữ liệu DT (Data register). Mọi MemoryArea word
    đều ánh xạ về DT[address]; bit = đọc DT word rồi mask (read-modify-write).
    Cần pyserial (pip install pyserial).

    MEWTOCOL-COM frame:
        Command :  %<stn(2)>#<cmd><type><start(5)><end(5)>[data]<BCC><CR>
        Resp OK :  %<stn(2)>$<cmd>[data]<BCC><CR>
        Resp err:  %<stn(2)>!<err(2)><BCC><CR>
        BCC = XOR mọi byte từ '%' đến hết data → 2 ký tự ASCII hex hoa.
        Word trong response = "LLHH" (low byte trước).
    """

    _STATION = 1

    def __init__(self, ip: str, port: int = 9600, timeout: float = 1.0):
        super().__init__(ip, port, timeout)
        self.com = str(ip)            # alias cho rõ nghĩa: ip giữ tên COM
        self.baud = int(port)         # alias: port giữ baud rate
        self.ser = None
        self._parity = None           # cache parity sau khi probe
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import serial
        except ImportError as e:
            raise RuntimeError(
                "pyserial chưa cài — Panasonic FP cần serial "
                "(pip install pyserial)") from e
        if self.connected:
            self.disconnect()
        # Probe parity: Panasonic FP factory hay 8O1, nhiều máy đã đổi 8N1
        # → thử NONE rồi ODD. Sau khi tìm được thì cache vào self._parity.
        candidates = ([self._parity] if self._parity is not None
                      else [serial.PARITY_NONE, serial.PARITY_ODD])
        last_err = None
        for par in candidates:
            try:
                ser = serial.Serial(
                    port=self.com, baudrate=self.baud,
                    bytesize=serial.EIGHTBITS, parity=par,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout, write_timeout=self.timeout)
            except Exception as e:
                last_err = e
                continue
            try:
                resp = self._txn(ser, self._build_read_cmd(0, 0))
                s = resp.decode('ascii', 'ignore')
                if (s.startswith("%%%02d$" % self._STATION)
                        or s.startswith("%%%02d!" % self._STATION)):
                    self._parity = par
                    self.ser = ser
                    self.connected = True
                    return
                ser.close()                  # prefix lạ → sai parity
            except Exception as e:
                last_err = e
                try:
                    ser.close()
                except Exception:
                    pass
        raise IOError("Panasonic MEWTOCOL connect failed on %s@%s: %s"
                      % (self.com, self.baud, last_err))

    def disconnect(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.connected = False

    @staticmethod
    def _bcc(body: bytes) -> bytes:
        b = 0
        for c in body:
            b ^= c
        return ("%02X" % b).encode('ascii')

    def _build_read_cmd(self, start: int, end: int) -> bytes:
        body = ("%%%02d#RDD%05d%05d" % (self._STATION, start, end)).encode('ascii')
        return body + self._bcc(body) + b"\r"

    def _build_write_cmd(self, start: int, values) -> bytes:
        end = start + len(values) - 1
        parts = []
        for v in values:
            u = int(v) & 0xFFFF
            parts.append("%02X%02X" % (u & 0xFF, (u >> 8) & 0xFF))  # LLHH
        body = (("%%%02d#WDD%05d%05d" % (self._STATION, start, end))
                + "".join(parts)).encode('ascii')
        return body + self._bcc(body) + b"\r"

    def _txn(self, ser, cmd: bytes) -> bytes:
        """Gửi cmd, đọc response đến CR. Trả bytes (không gồm CR)."""
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        ser.write(cmd)
        out = bytearray()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            b = ser.read(1)
            if not b:
                continue
            if b == b"\r":
                return bytes(out)
            out += b
            if len(out) > 4096:               # safety: response không bao giờ lớn vậy
                break
        raise TimeoutError("MEWTOCOL response timeout")

    def _parse_read(self, resp: bytes) -> list:
        s = resp.decode('ascii', 'replace')
        if s.startswith("%%%02d!" % self._STATION):
            raise IOError("Panasonic error response: %s" % s)
        pref = "%%%02d$RD" % self._STATION
        if not s.startswith(pref):
            raise IOError("Bad MEWTOCOL RD response: %r" % s)
        data = s[len(pref):-2]                # bỏ BCC 2 ký tự cuối
        out = []
        n = (len(data) // 4) * 4
        for i in range(0, n, 4):              # mỗi 4 char = 1 word "LLHH"
            lo = int(data[i:i + 2], 16)
            hi = int(data[i + 2:i + 4], 16)
            out.append((hi << 8) | lo)
        return out

    def _check_write(self, resp: bytes) -> None:
        s = resp.decode('ascii', 'replace')
        if s.startswith("%%%02d!" % self._STATION):
            raise IOError("Panasonic error response: %s" % s)
        if not s.startswith("%%%02d$WD" % self._STATION):
            raise IOError("Bad MEWTOCOL WD response: %r" % s)

    def read_word(self, area: MemoryArea, address: int) -> int:
        if not self.connected:
            self.connect()
        with self._lock:
            resp = self._txn(self.ser, self._build_read_cmd(address, address))
        vals = self._parse_read(resp)
        return vals[0] if vals else 0

    def write_word(self, area: MemoryArea, address: int, value: int) -> None:
        if not self.connected:
            self.connect()
        with self._lock:
            resp = self._txn(self.ser,
                             self._build_write_cmd(address, [value & 0xFFFF]))
        self._check_write(resp)


# ── Registry ──────────────────────────────────────────────────────
DRIVER_BY_MODEL = {
    "Omron CP2E":         OmronCP2E,
    "Omron NX1P2":        OmronNX1P2,
    "Inovance H3U/H5U":   InovanceH3UH5U,
    "Mitsubishi FX5U":    MitsubishiFX5U,
    "Panasonic FP":       PanasonicFP,
}


# ── Manager / polling thread ──────────────────────────────────────
@dataclass
class TriggerRoute:
    """1 PLC trigger độc lập → 1 Acquire branch.
    Mỗi trigger có Area/Address/Bit/Value/Auto-clear riêng — không phải
    nhiều giá trị trên cùng 1 word. Khi đọc PLC khớp Value (hoặc bit lên 1),
    monitor fire callback với acquire_node_id để chạy nhánh tương ứng.
    `acquire_node_id` rỗng = chạy toàn pipeline khi trigger này fire.
    """
    area: MemoryArea = MemoryArea.DM_WORD
    address: int = 100
    bit: int = 0
    value: int = 1
    auto_clear: bool = True
    acquire_node_id: str = ""

    def to_dict(self) -> dict:
        return {
            "area": self.area.value,
            "address": self.address,
            "bit": self.bit,
            "value": self.value,
            "auto_clear": self.auto_clear,
            "acquire_node_id": self.acquire_node_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerRoute":
        return cls(
            area=MemoryArea(d.get("area", MemoryArea.DM_WORD.value)),
            address=int(d.get("address", 100)),
            bit=int(d.get("bit", 0)),
            value=int(d.get("value",
                             d.get("trigger_value", 1))),  # back-compat
            auto_clear=bool(d.get("auto_clear", True)),
            acquire_node_id=str(d.get("acquire_node_id", "")),
        )


@dataclass
class PLCConfig:
    model: str = "Omron CP2E"
    ip: str = "192.168.250.1"
    port: int = 9600
    poll_interval_ms: int = 100

    # FINS node addresses — chỉ áp dụng cho Omron NX1P2 (UDP).
    # Mặc định dest_node = octet cuối của IP, src_node có thể bất kỳ
    # nhưng phải khớp cấu hình FINS settings của PLC.
    fins_dest_node: int = 1
    fins_src_node: int = 25

    # Triggers (PLC → AOI): list[TriggerRoute], mỗi entry là 1 trigger
    # độc lập (area + address + bit + value + auto_clear + acquire). Monitor
    # poll từng entry, rising-edge khớp giá trị thì fire callback với
    # acquire_node_id của entry đó. Rỗng = không có trigger → monitor idle.
    trigger_routes: list = field(default_factory=list)

    # Result (gửi PASS/FAIL về PLC)
    result_area: MemoryArea = MemoryArea.DM_WORD
    result_address: int = 101
    result_pass_value: int = 1
    result_fail_value: int = 2
    # Nếu set, OK/NG dựa vào node này (port "pass" output). Rỗng = fallback
    # legacy: pass khi MỌI node trong pipeline đều pass/idle.
    result_judge_node_id: str = ""

    # Numeric data (gửi số liệu, optional)
    data_area: MemoryArea = MemoryArea.DM_WORD
    data_start_address: int = 110

    # Float / int32 word order: 'ABCD' = high word first, 'CDAB' = low word first
    # Omron NJ/NX với REAL kiểu IEEE thường dùng CDAB; Modbus/PLC khác đa số ABCD.
    float_word_order: str = "ABCD"

    # Mapping cụ thể từ output của node → địa chỉ PLC
    data_mappings: list = field(default_factory=list)


@dataclass
class DataMapping:
    """Một mapping: lấy output ``output_key`` của node ``node_id`` → ghi vào PLC.

    data_type:
        - 'int16'         : ghi 1 word, giá trị bị trunc về int và mask 0xFFFF (có dấu hỗ trợ qua 2's complement)
        - 'int32'         : ghi 2 word
        - 'float32'       : ghi 2 word IEEE-754
        - 'scaled_int16'  : value × scale → round → ghi 1 word int16 (signed)
        - 'scaled_int32'  : value × scale → round → ghi 2 word int32 (signed)
    """
    node_id: str = ""
    output_key: str = ""
    area: MemoryArea = MemoryArea.DM_WORD
    address: int = 0
    data_type: str = "float32"
    scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "output_key": self.output_key,
            "area": self.area.value,
            "address": self.address,
            "data_type": self.data_type,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DataMapping":
        return cls(
            node_id=d.get("node_id", ""),
            output_key=d.get("output_key", ""),
            area=MemoryArea(d.get("area", MemoryArea.DM_WORD.value)),
            address=int(d.get("address", 0)),
            data_type=d.get("data_type", "float32"),
            scale=float(d.get("scale", 1.0)),
        )


class PLCManager:
    """Quản lý driver + thread polling trigger.

    Cách dùng (từ GUI):
        mgr = PLCManager()
        mgr.config = PLCConfig(...)
        mgr.connect()
        mgr.start_monitor(on_trigger=lambda: window.run_pipeline())
        # khi pipeline xong:
        mgr.write_result(passed=True, values=[12, 34])
    """

    def __init__(self):
        self.config = PLCConfig()
        self.driver: Optional[PLCDriver] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Callback nhận `acquire_node_id` (rỗng = chạy toàn pipeline).
        self._on_trigger: Optional[Callable[[str], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None
        # Last-seen value per trigger index — rising-edge detection độc lập
        # cho mỗi trigger trong cfg.trigger_routes.
        self._last_per_trigger: Dict[int, int] = {}

    @property
    def is_connected(self) -> bool:
        return self.driver is not None and self.driver.connected

    @property
    def is_monitoring(self) -> bool:
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def connect(self) -> None:
        cls = DRIVER_BY_MODEL.get(self.config.model)
        if cls is None:
            raise ValueError(f"Unknown PLC model: {self.config.model}")
        self.disconnect()
        if cls is OmronNX1P2:
            self.driver = cls(self.config.ip, self.config.port,
                              dest_node=self.config.fins_dest_node,
                              src_node=self.config.fins_src_node)
        else:
            self.driver = cls(self.config.ip, self.config.port)
        self.driver.connect()

    def disconnect(self) -> None:
        self.stop_monitor()
        if self.driver is not None:
            try:
                self.driver.disconnect()
            except Exception:
                pass
        self.driver = None

    # ── Trigger monitor ──
    def start_monitor(self,
                      on_trigger: Callable[[str], None],
                      on_error: Optional[Callable[[str], None]] = None) -> None:
        if not self.is_connected:
            raise RuntimeError("PLC not connected")
        if self.is_monitoring:
            return
        self._on_trigger = on_trigger
        self._on_error = on_error
        self._stop.clear()
        self._last_per_trigger = {}
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="PLCMonitor")
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop.set()
        t = self._monitor_thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._monitor_thread = None

    def _monitor_loop(self) -> None:
        cfg = self.config
        while not self._stop.is_set():
            # Snapshot triggers mỗi vòng để pick up UI thay đổi (apply
            # qua _gather_config rồi self.config = …).
            triggers = list(cfg.trigger_routes)
            if not triggers:
                # No triggers configured → idle wait, không spam log
                self._stop.wait(max(0.05, cfg.poll_interval_ms / 1000.0))
                continue
            for idx, route in enumerate(triggers):
                if self._stop.is_set():
                    break
                try:
                    is_bit = route.area in (MemoryArea.CIO_BIT, MemoryArea.DM_BIT)
                    if is_bit:
                        val = 1 if self.driver.read_bit(
                            route.area, route.address, route.bit) else 0
                        target = 1
                    else:
                        val = self.driver.read_word(route.area, route.address)
                        target = int(route.value)

                    # Rising-edge detect per trigger index
                    last = self._last_per_trigger.get(idx)
                    if val == target and last != target:
                        if route.auto_clear:
                            try:
                                if is_bit:
                                    self.driver.write_bit(
                                        route.area, route.address,
                                        route.bit, False)
                                else:
                                    self.driver.write_word(
                                        route.area, route.address, 0)
                            except Exception:
                                pass
                        cb = self._on_trigger
                        if cb is not None:
                            try:
                                cb(str(route.acquire_node_id or ""))
                            except Exception as e:
                                if self._on_error:
                                    self._on_error(
                                        f"Trigger #{idx+1} callback error: {e}")
                    self._last_per_trigger[idx] = val
                except Exception as e:
                    if self._on_error:
                        self._on_error(
                            f"PLC read error (trigger #{idx+1}): {e}")
                    self._stop.wait(0.5)
                    break
            self._stop.wait(max(0.01, cfg.poll_interval_ms / 1000.0))

    # ── Write back ──
    def write_result(self, passed: bool) -> None:
        """Ghi mã PASS/FAIL về địa chỉ result."""
        if not self.is_connected:
            raise RuntimeError("PLC not connected")
        cfg = self.config
        code = cfg.result_pass_value if passed else cfg.result_fail_value
        self.driver.write_word(cfg.result_area, cfg.result_address, code)

    @staticmethod
    def _to_signed_word(v: int) -> int:
        v = int(v)
        if v < 0:
            v = (1 << 16) + v
        return v & 0xFFFF

    def write_value(self, area: MemoryArea, address: int, value: float,
                    data_type: str = "float32", scale: float = 1.0) -> None:
        """Ghi 1 giá trị numeric theo data_type được chỉ định."""
        if not self.is_connected:
            raise RuntimeError("PLC not connected")
        v = float(value)
        word_order = self.config.float_word_order

        if data_type == "int16":
            self.driver.write_word(area, address, self._to_signed_word(v))

        elif data_type == "scaled_int16":
            iv = int(round(v * scale))
            self.driver.write_word(area, address, self._to_signed_word(iv))

        elif data_type in ("int32", "scaled_int32"):
            iv = int(round(v * scale)) if data_type == "scaled_int32" else int(v)
            if iv < 0:
                iv = (1 << 32) + iv
            iv &= 0xFFFFFFFF
            hi = (iv >> 16) & 0xFFFF
            lo = iv & 0xFFFF
            w0, w1 = (hi, lo) if word_order == "ABCD" else (lo, hi)
            self.driver.write_word(area, address,     w0)
            self.driver.write_word(area, address + 1, w1)

        elif data_type == "float32":
            raw = struct.pack('>f', v)               # ABCD big-endian
            w_high = int.from_bytes(raw[0:2], 'big')
            w_low  = int.from_bytes(raw[2:4], 'big')
            w0, w1 = (w_high, w_low) if word_order == "ABCD" else (w_low, w_high)
            self.driver.write_word(area, address,     w0)
            self.driver.write_word(area, address + 1, w1)

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def write_data_mappings(self, results: dict) -> list:
        """Ghi tất cả mappings dựa trên dict ``results`` = {node_id: outputs_dict}.

        Trả về list mô tả từng mapping đã ghi (cho UI log). Mapping nào lỗi sẽ
        gắn ``error`` thay vì raise — để không cản các mapping kế tiếp.
        """
        report = []
        for m in self.config.data_mappings:
            entry = {"node_id": m.node_id, "output_key": m.output_key,
                     "address": m.address, "data_type": m.data_type}
            out = results.get(m.node_id)
            if not isinstance(out, dict) or m.output_key not in out:
                entry["error"] = "output not found"
                report.append(entry); continue
            v = out[m.output_key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                entry["error"] = f"value not numeric ({type(v).__name__})"
                report.append(entry); continue
            try:
                self.write_value(m.area, m.address, v, m.data_type, m.scale)
                entry["value"] = v
            except Exception as e:
                entry["error"] = str(e)
            report.append(entry)
        return report


# ── Shared singleton ──────────────────────────────────────────────
# Cho node trong pipeline (vd PLC Write) dùng CHUNG instance đã được cấu hình /
# kết nối ở PLC dialog, thay vì mở kết nối mới.
_SHARED_MANAGER: Optional["PLCManager"] = None

def get_plc_manager() -> "PLCManager":
    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = PLCManager()
    return _SHARED_MANAGER