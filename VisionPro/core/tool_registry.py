"""
core/tool_registry.py — T VisionPro Style
Tools mô phỏng T VisionPro: PatMax, Caliper, Blob, Edge, Color,
Barcode, OCR, Fixture, Calibration, Display, Logic, Communication.
Giữ nguyên kiến trúc kéo-thả pipeline.
"""
from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import os

@dataclass
class PortDef:
    name: str
    data_type: str
    required: bool = True
    default: Any = None


@dataclass
class ParamDef:
    name: str
    label: str
    ptype: str   # int|float|bool|enum|str|color
    default: Any = 0
    min_val: Any = None
    max_val: Any = None
    choices: List[str] = field(default_factory=list)
    step: Any = 1
    tooltip: str = ""
    # Conditional visibility: dict {param_name: required_value}
    visible_if: Dict[str, Any] = field(default_factory=dict)
    # Hiển thị slider kèm spinbox cho int/float param (cần min_val/max_val)
    use_slider: bool = False
    # Filter cho QFileDialog khi name chứa 'path' (str type). Vd:
    # "Models (*.pt *.onnx);;All Files (*)". None = ảnh (default).
    file_filter: Optional[str] = None


@dataclass
class ToolDef:
    tool_id: str
    name: str
    category: str
    description: str
    color: str
    icon: str
    inputs: List[PortDef]
    outputs: List[PortDef]
    params: List[ParamDef]
    process_fn: Callable
    T_equiv: str = ""   # Tên tool tương đương trong T
    # Khi != "" → tool hỗ trợ user thêm input port động qua right-click "Add
    # Input". Giá trị = data_type của port mới (vd "bool" cho Pass/Fail Judge).
    extra_input_type: str = ""
    # Tương tự cho output: != "" → right-click "Add Output Port" (prompt tên).
    # Giá trị = data_type của port mới (vd "any" cho Script Tool).
    extra_output_type: str = ""
    # Khi non-empty: tool hỗ trợ "Add Output Terminal…" để pick (object_idx,
    # field) → tạo 1 port output. Mỗi field 1 lựa chọn trong combo. Engine
    # trong proc_* gọi `_apply_extra_terminals(out, objects_list, params)`
    # để map giá trị vào port.
    terminal_fields: List[str] = field(default_factory=list)
    # Output key chứa list[dict] objects để dialog đếm số object detect được.
    # Default "objects" (PatMax). YOLO dùng "detections".
    terminal_source_key: str = "objects"


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def _gray(img):
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

def _bgr(img):
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img

# Kích thước tham chiếu (cạnh ngắn). Ảnh nhỏ hơn → scale = 1.0;
# ảnh lớn hơn → scale tăng theo tỷ lệ để chữ & nét không bị tí hon.
_DRAW_BASE_DIM = 720.0

# Target pixel count cho auto downscale của các detection tool. Ảnh > target
# sẽ được resize xuống còn ~target trước khi chạy thuật toán nặng (Hough,
# Canny, findContours, absdiff…). Kết quả (coord, radius, contour…) scale
# ngược về full-res cho overlay + output.
_DETECT_TARGET_PX = 1_500_000

def _auto_downscale(gray, ds_param: int = 0,
                     target_px: int = _DETECT_TARGET_PX):
    """Trả (small, ds). ds_param>0 → force; else auto target ~target_px.
    Dùng cho proc_find_circle / proc_blob / proc_surface_defect /
    proc_scratch_detect / proc_find_line — pattern coarse-then-refine
    không cần thiết, kết quả chỉ scale ngược ×ds là đủ chính xác cho
    UI overlay (ds=4 cho 20MP → sai ±4px, < 0.1% width).
    """
    try:
        ds_param = int(ds_param or 0)
    except (TypeError, ValueError):
        ds_param = 0
    if ds_param > 0:
        ds = max(1, ds_param)
    else:
        h, w = gray.shape[:2]
        ds = max(1, int(round((h * w / target_px) ** 0.5)))
    if ds > 1:
        small = cv2.resize(gray, None, fx=1.0/ds, fy=1.0/ds,
                            interpolation=cv2.INTER_AREA)
    else:
        small = gray
    return small, ds


def _draw_scale(img):
    if img is None:
        return 1.0
    h, w = img.shape[:2]
    short = min(h, w)
    return max(1.0, short / _DRAW_BASE_DIM)

def _t(base, s):
    """Scaled line/box thickness (>=1)."""
    return max(1, int(round(base * s)))

def _fs(base, s):
    """Scaled font scale for cv2.putText."""
    return float(base) * s

def _draw_pass_fail(img, is_pass, text=""):
    vis = _bgr(img.copy())
    # Summary log only — không vẽ banner PASS/FAIL lên ảnh.
    label = f"{'PASS' if is_pass else 'FAIL'}{' '+text if text else ''}"
    print(f"[Pipeline] {label}")
    return vis


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: ACQUIRE IMAGE
# ═══════════════════════════════════════════════════════════════════

# LRU cache (path → (mtime_ns, size, img)) cho proc_acquire_image. Tránh
# decode PNG/JPG lặp lại khi cùng 1 file load nhiều lần (worst-case 20MP
# PNG tốn ~200ms decode). Cap 16 entries — đủ cho folder mode cycle.
_ACQUIRE_CACHE: "dict[str, tuple]" = {}
_ACQUIRE_CACHE_MAX = 16

# Background decode pool — prefetch next folder frames để hide decode
# latency cho 20MP PNG (~150-240ms/frame) khi cycle qua folder.
import threading as _threading
_ACQUIRE_LOCK = _threading.Lock()
_PREFETCH_THREAD: "Optional[_threading.Thread]" = None
_PREFETCH_QUEUE: "list[str]" = []
_PREFETCH_DEPTH = 2  # số frame nhìn trước


def _decode_into_cache(path: str) -> bool:
    """Decode 1 file và đẩy vào cache. Return True nếu thực sự decode
    (cache miss), False nếu đã có sẵn / lỗi đọc."""
    try:
        st = os.stat(path)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return False
    with _ACQUIRE_LOCK:
        cached = _ACQUIRE_CACHE.get(path)
        if cached is not None and cached[0] == key:
            return False
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(path)
        if img is None:
            return False
    with _ACQUIRE_LOCK:
        if len(_ACQUIRE_CACHE) >= _ACQUIRE_CACHE_MAX:
            _ACQUIRE_CACHE.pop(next(iter(_ACQUIRE_CACHE)))
        _ACQUIRE_CACHE[path] = (key, img)
    return True


def _prefetch_worker():
    """Decode tuần tự các file trong queue. Khi queue rỗng → thread exit."""
    while True:
        with _ACQUIRE_LOCK:
            if not _PREFETCH_QUEUE:
                return
            path = _PREFETCH_QUEUE.pop(0)
        try:
            _decode_into_cache(path)
        except Exception:
            pass


def _kick_prefetch(paths: "list[str]"):
    """Append paths vào prefetch queue, spawn worker thread nếu chưa chạy.
    Worker chạy tuần tự (1 thread) → tránh cv2 contention + đỡ I/O thrashing.
    """
    global _PREFETCH_THREAD
    if not paths:
        return
    with _ACQUIRE_LOCK:
        for p in paths:
            if p and p not in _PREFETCH_QUEUE:
                _PREFETCH_QUEUE.append(p)
        if _PREFETCH_THREAD is not None and _PREFETCH_THREAD.is_alive():
            return
        _PREFETCH_THREAD = _threading.Thread(
            target=_prefetch_worker, daemon=True)
    _PREFETCH_THREAD.start()


def acquire_prefetch(paths):
    """Public API — yêu cầu prefetch 1 hoặc nhiều file vào cache nền.
    Properties panel / file picker có thể gọi ngay khi user chọn file để
    decode bắt đầu trong lúc user còn nhìn UI, → khi Run thì cache hit.
    """
    if isinstance(paths, str):
        paths = [paths]
    _kick_prefetch([p for p in (paths or []) if p])


def _load_image_cached(path: str):
    """Return (img, w, h) hoặc (None, 0, 0). Cache theo (path, mtime, size)
    để re-load file không đổi là instant; file bị overwrite → mtime đổi →
    decode lại tự động. Dùng IMREAD_UNCHANGED để giữ gray nếu source gray
    (skip expand 1→3 channels, decode nhanh hơn ~30%).
    """
    try:
        st = os.stat(path)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None, 0, 0
    with _ACQUIRE_LOCK:
        cached = _ACQUIRE_CACHE.get(path)
        if cached is not None and cached[0] == key:
            img = cached[1]
            h, w = img.shape[:2]
            return img, w, h
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Fallback default (handle exotic formats)
        img = cv2.imread(path)
        if img is None:
            return None, 0, 0
    with _ACQUIRE_LOCK:
        # Cap cache size — FIFO drop để giữ memory bounded với ảnh 20MP×16=~1GB
        if len(_ACQUIRE_CACHE) >= _ACQUIRE_CACHE_MAX:
            _ACQUIRE_CACHE.pop(next(iter(_ACQUIRE_CACHE)))
        _ACQUIRE_CACHE[path] = (key, img)
    h, w = img.shape[:2]
    return img, w, h


def proc_acquire_image(inputs, params):
    """TAcqFifoTool — Acquire image từ file đơn hoặc folder (frame index).

    Ưu tiên: folder_path > file_path.
    Khi folder_path là thư mục có ảnh, sẽ lấy ảnh tại index `frame_index`
    (modulo số file). Hỗ trợ .png .jpg .jpeg .bmp .tif .tiff.
    """
    mode = params.get("source_mode", "Folder")
    folder = params.get("folder_path", "") or ""
    files = []
    if mode == "Folder" and folder and os.path.isdir(folder):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        try:
            files = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))
            )
        except OSError:
            files = []

    if files:
        idx = int(params.get("frame_index", 0)) % len(files)
        path = os.path.join(folder, files[idx])
        img, w, h = _load_image_cached(path)
        if img is not None:
            # Auto-advance: lần Run kế tiếp sẽ sang ảnh kế tiếp (cycle)
            if params.get("auto_advance", True):
                params["frame_index"] = (idx + 1) % len(files)
            # Prefetch _PREFETCH_DEPTH frame kế tiếp trong background — khi
            # user Run lần sau, frame đã sẵn trong cache (0ms decode).
            prefetch_paths = []
            for k in range(1, _PREFETCH_DEPTH + 1):
                j = (idx + k) % len(files)
                p = os.path.join(folder, files[j])
                if p != path:
                    prefetch_paths.append(p)
            if prefetch_paths:
                _kick_prefetch(prefetch_paths)
            return {"image": img, "width": w, "height": h,
                    "acquired": True, "frame_number": idx,
                    "file_name": files[idx], "frame_count": len(files)}

    path = params.get("file_path", "")
    if path and os.path.exists(path):
        img, w, h = _load_image_cached(path)
        if img is not None:
            return {"image": img, "width": w, "height": h,
                    "acquired": True, "frame_number": 0,
                    "file_name": os.path.basename(path), "frame_count": 1}

    w = max(1, params.get("width", 640))
    h = max(1, params.get("height", 480))
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "No Image Acquired", (w//2 - 120, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 80), 2)
    return {"image": img, "width": w, "height": h, "acquired": False,
            "frame_number": 0, "file_name": "", "frame_count": 0}

def proc_camera_acquire(inputs, params):
    """TAcqFifoTool (Camera) — Capture từ OpenCV/USB hoặc HikRobot/Do3think MVS."""
    backend = (params.get("backend") or "OpenCV").strip()
    try:
        from core.camera import CameraRegistry, CameraError
        reg = CameraRegistry.instance()
        if backend in ("HikRobot/Do3think", "HikRobot", "Do3think", "MVS"):
            kwargs = {"device_index": int(params.get("device_index", 0))}
            sn = (params.get("serial") or "").strip()
            if sn:
                kwargs["serial"] = sn
            kwargs["access_mode"] = (params.get("access_mode") or "exclusive").lower()
            kwargs["heartbeat_ms"] = int(params.get("heartbeat_ms", 5000))
            cam = reg.get_or_open("mvs", **kwargs)
            # Continuous grab: pipeline lấy frame buffered <1ms thay vì
            # chờ ~33ms cho khung kế tiếp. Tắt khi cần single-frame chính
            # xác lúc trigger (vd. PLC trigger camera đồng bộ).
            if bool(params.get("continuous_grab", True)):
                cam.start_continuous()
            else:
                cam.stop_continuous()
        else:
            cam = reg.get_or_open(
                "opencv",
                index=int(params.get("camera_id", 0)),
                width=int(params.get("width", 0)),
                height=int(params.get("height", 0)))
        frame = cam.grab(timeout_ms=int(params.get("timeout_ms", 1000)))
    except Exception as e:
        msg = str(e)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Cam err: {msg[:40]}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        return {"image": frame, "width": 640, "height": 480,
                "acquired": False, "frame_number": 0}
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    return {"image": frame, "width": w, "height": h,
            "acquired": True, "frame_number": 0}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: PATMAX / PATTERN FIND
# ═══════════════════════════════════════════════════════════════════

def _build_patmax_objects(results, model, obj_origin_overrides=None):
    """Build list `objects` từ results.
    Mỗi object có:
      - x, y       : origin marker (đã apply per-object rule + override)
      - origin_x/y : raw pattern-origin transformed (giữ nguyên)
      - center_x/y : bbox center
      - ref{N}_x/y/angle: từng extra ref (đã transform theo pose)
      - refs       : list dict cho UI duyệt
    Per-object rule (xem patmax_engine.resolve_obj_origin_xy):
      obj 0 → pattern-origin transformed; obj 1+ → tâm vật thể.
    """
    from core.patmax_engine import (transform_ref_to_image,
                                     resolve_obj_origin_xy)
    objs = []
    extras = list(getattr(model, "extra_refs", []) or []) if model else []
    for i, r in enumerate(results):
        ox, oy = resolve_obj_origin_xy(r, i, obj_origin_overrides)
        obj = {"x": ox, "y": oy, "score": r.score,
               "angle": r.angle, "scale": r.scale,
               "center_x": r.x, "center_y": r.y,
               "origin_x": r.origin_x, "origin_y": r.origin_y}
        refs_data = []
        for j, ref in enumerate(extras, start=1):
            try:
                ex, ey, eang = transform_ref_to_image(model, ref, r)
            except Exception:
                continue
            nm = str(ref.get("name", f"Ref {j}"))
            refs_data.append({"name": nm, "x": ex, "y": ey, "angle": eang})
            obj[f"ref{j}_x"]     = ex
            obj[f"ref{j}_y"]     = ey
            obj[f"ref{j}_angle"] = eang
        obj["refs"] = refs_data
        objs.append(obj)
    return objs


def auto_terminal_name(term: dict) -> str:
    """Auto-generate output port name từ terminal spec.
    Nếu user đặt name → dùng name. Ngược lại: obj 0 (default) → "<field>";
    obj > 0 → "<field>_<obj_idx>" để phân biệt nhiều object.
    Migration: nếu explicit name khớp đúng pattern legacy "<field>_<obj>"
    (auto-name của bản cũ) → coi như auto, trả về tên ngắn mới.
    """
    field = str(term.get("field", "x"))
    try:
        obj_idx = int(term.get("object", 0) or 0)
    except (TypeError, ValueError):
        obj_idx = 0
    explicit = (term.get("name") or "").strip()
    if explicit and explicit == f"{field}_{obj_idx}":
        explicit = ""  # legacy auto-saved name → drop, use new short form
    if explicit:
        return explicit
    return field if obj_idx == 0 else f"{field}_{obj_idx}"


def _apply_extra_terminals(out: dict, objects: list, params: dict):
    """Apply extra output terminals từ params['_extra_terminals'].
    term = {"object": int, "field": str, "name": str}
    field có thể là field cơ bản (x, y, angle, score, scale, ...) hoặc
    ref-aware (ref1_x, ref2_y, ...).
    """
    for term in (params.get("_extra_terminals") or []):
        try:
            obj_idx = int(term.get("object", 0))
            field = str(term.get("field", "x"))
            name = auto_terminal_name(term)
            if 0 <= obj_idx < len(objects):
                out[name] = objects[obj_idx].get(field, 0.0)
            else:
                out[name] = 0.0
        except Exception:
            continue


def proc_patmax(inputs, params):
    """
    TPatMaxPatternAlignTool — dùng PatMaxEngine.
    Model được train trong PatMaxDialog (double-click node).
    Hỗ trợ multi-pattern: nếu params có "_patmax_models" (list) sẽ search
    qua tất cả models và gộp kết quả qua run_patmax_multi.
    """
    from core.patmax_engine import (PatMaxModel, run_patmax, run_patmax_multi,
                                     draw_patmax_results, _empty_vis)
    img = inputs.get("image")
    if img is None:
        return {"image": None, "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0, "num_found": 0}

    models_list = params.get("_patmax_models") or []
    model: PatMaxModel = params.get("_patmax_model") or PatMaxModel()
    use_multi = (params.get("_patmax_roi_mode") == "multi_pattern"
                 and isinstance(models_list, list)
                 and any(m.is_valid() for m in models_list))

    show_ref  = bool(params.get("show_reference", True))
    show_xy   = bool(params.get("show_xy",   show_ref))
    show_bbox = bool(params.get("show_bbox", show_ref))

    if not use_multi and not model.is_valid():
        clean = _bgr(img.copy())
        vis = _empty_vis(_bgr(img))
        return {"image": clean, "_display_image": vis,
                "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0, "num_found": 0}

    ref = models_list[0] if use_multi else model
    ang_low  = ref.angle_low
    ang_high = ref.angle_high
    ang_step = max(0.5, ref.angle_step)
    sc_low   = ref.scale_low
    sc_high  = ref.scale_high
    sc_step  = max(0.01, getattr(ref, "scale_step", 0.1) or 0.1)

    # Ưu tiên giá trị đã save trong model (PatMaxDialog auto-save). Fall back
    # node.params chỉ khi model chưa có (PatMaxModel default = 1, ổn).
    nr = max(1, int(ref.num_results or 0)) or int(params.get("num_results", 1))
    ot = float(getattr(ref, "overlap_threshold", 0.5) or 0.5)
    at = float(ref.accept_threshold or 0.5)

    # Speed knobs
    try:
        ds = max(1, int(params.get("coarse_downscale", 1)))
    except (TypeError, ValueError):
        ds = 1
    chans = (True, bool(params.get("use_edge", True)),
                    bool(params.get("use_sqdiff", True)))

    if use_multi:
        results, score_map = run_patmax_multi(
            _bgr(img), [m for m in models_list if m.is_valid()],
            accept_threshold=at,
            angle_low=ang_low, angle_high=ang_high, angle_step=ang_step,
            scale_low=sc_low,  scale_high=sc_high,  scale_step=sc_step,
            num_results_per_model=nr,
            overlap_threshold=ot,
            coarse_downscale=ds, channels=chans,
        )
        # Vẽ với model đầu tiên (origin reference) — multi-pattern share style
        model = ref
    else:
        results, score_map = run_patmax(
            _bgr(img), model,
            accept_threshold=at,
            angle_low=ang_low, angle_high=ang_high, angle_step=ang_step,
            scale_low=sc_low,  scale_high=sc_high,  scale_step=sc_step,
            num_results=nr,
            overlap_threshold=ot,
            coarse_downscale=ds, channels=chans,
        )

    overrides = _patmax_obj_origin_overrides(params)
    clean = _bgr(img.copy())
    vis = draw_patmax_results(clean, results, model,
                                show_xy=show_xy, show_bbox=show_bbox,
                                obj_origin_overrides=overrides)

    objects = _build_patmax_objects(results, model, overrides)
    if results:
        r = results[0]
        # Top-level x/y dùng obj 0 đã resolve (override hoặc pattern-origin).
        out = {"image": clean, "_display_image": vis,
               "found": True, "score": r.score,
               "x": objects[0]["x"], "y": objects[0]["y"],
               "angle": r.angle, "scale": r.scale,
               "num_found": len(results), "objects": objects}
    else:
        out = {"image": clean, "_display_image": vis,
               "found": False, "score": 0.0,
               "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0,
               "num_found": 0, "objects": []}
    _apply_extra_terminals(out, objects, params)
    return out


def _patmax_obj_origin_overrides(params: dict):
    """Parse per-object origin overrides từ node params.
    Format chấp nhận: dict {int: (x, y)} hoặc {str: [x, y]} (do JSON
    serialization). Trả dict {int: (float, float)}.
    """
    raw = params.get("_per_obj_origin_overrides") if params else None
    if not isinstance(raw, dict):
        return None
    out = {}
    for k, v in raw.items():
        try:
            idx = int(k)
            x, y = float(v[0]), float(v[1])
            out[idx] = (x, y)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return out or None


def _is_gray_image(img) -> bool:
    """True nếu ảnh là grayscale (1 kênh, hoặc 3 kênh nhưng B==G==R)."""
    if img is None:
        return False
    if len(img.shape) == 2:
        return True
    if len(img.shape) == 3 and img.shape[2] == 1:
        return True
    if len(img.shape) == 3 and img.shape[2] >= 3:
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        return bool(np.array_equal(b, g) and np.array_equal(g, r))
    return False


def proc_patmax_align(inputs, params):
    """
    PatMax Align Tool — dispatch theo Algorithm + Train Mode (T-style
    behavioral approximation). Validate input gray; nếu không gray trả ảnh
    gốc + found=False (UI dialog popup cảnh báo khi user ấn Train).
    """
    from core.patmax_engine import (PatMaxModel, run_patmax_align,
                                     draw_patmax_results, _empty_vis)
    img = inputs.get("image")
    if img is None:
        return {"image": None, "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0,
                "num_found": 0, "objects": []}
    if not _is_gray_image(img):
        clean = _bgr(img)
        return {"image": clean, "_display_image": clean,
                "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0,
                "num_found": 0, "objects": []}

    model: PatMaxModel = params.get("_patmax_model") or PatMaxModel()
    show_ref  = bool(params.get("show_reference", True))
    show_xy   = bool(params.get("show_xy",   show_ref))
    show_bbox = bool(params.get("show_bbox", show_ref))
    if not model.is_valid():
        clean = _bgr(img.copy())
        vis = _empty_vis(_bgr(img))
        return {"image": clean, "_display_image": vis,
                "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0,
                "num_found": 0, "objects": []}

    algorithm        = str(params.get("algorithm", "PatQuick"))
    train_mode_align = str(params.get("train_mode", "Image"))

    ang_low  = model.angle_low
    ang_high = model.angle_high
    ang_step = max(0.5, model.angle_step)
    sc_low   = model.scale_low
    sc_high  = model.scale_high
    sc_step  = max(0.01, getattr(model, "scale_step", 0.1) or 0.1)
    nr = max(1, int(model.num_results or 0)) or int(params.get("num_results", 1))
    ot = float(getattr(model, "overlap_threshold", 0.5) or 0.5)
    at = float(model.accept_threshold or params.get("accept_threshold", 0.5))

    try:
        ds = max(1, int(params.get("coarse_downscale", 1)))
    except (TypeError, ValueError):
        ds = 1

    results, _ = run_patmax_align(
        _bgr(img), model,
        algorithm=algorithm,
        train_mode_align=train_mode_align,
        accept_threshold=at,
        angle_low=ang_low, angle_high=ang_high, angle_step=ang_step,
        scale_low=sc_low,  scale_high=sc_high,  scale_step=sc_step,
        num_results=nr,
        overlap_threshold=ot,
        coarse_downscale=ds,
        build_score_map=False,   # production: skip heatmap (caller discards)
    )
    overrides = _patmax_obj_origin_overrides(params)
    clean = _bgr(img.copy())
    vis = draw_patmax_results(clean, results, model,
                                show_xy=show_xy, show_bbox=show_bbox,
                                obj_origin_overrides=overrides)
    objects = _build_patmax_objects(results, model, overrides)
    if results:
        r = results[0]
        out = {"image": clean, "_display_image": vis,
               "found": True, "score": r.score,
               "x": objects[0]["x"], "y": objects[0]["y"],
               "angle": r.angle, "scale": r.scale,
               "num_found": len(results), "objects": objects}
    else:
        out = {"image": clean, "_display_image": vis,
               "found": False, "score": 0.0,
               "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0,
               "num_found": 0, "objects": []}
    _apply_extra_terminals(out, objects, params)
    return out


def proc_patfind(inputs, params):
    """
    TPMAlignTool — PatFind nhanh (NCC), dùng chung PatMaxEngine nhưng không xoay.
    """
    from core.patmax_engine import (PatMaxModel, run_patmax,
                                     draw_patmax_results, _empty_vis)
    img = inputs.get("image")
    if img is None:
        return {"image": None, "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "num_found": 0}

    model: PatMaxModel = params.get("_patmax_model") or PatMaxModel()
    show_ref  = bool(params.get("show_reference", True))
    show_xy   = bool(params.get("show_xy",   show_ref))
    show_bbox = bool(params.get("show_bbox", show_ref))
    if not model.is_valid():
        clean = _bgr(img.copy())
        vis = _empty_vis(_bgr(img))
        print("[PatFind] No model — double-click node to train")
        return {"image": clean, "_display_image": vis,
                "found": False, "score": 0.0,
                "x": 0.0, "y": 0.0, "num_found": 0}

    # PatFind: no rotation, no scale change
    results, _ = run_patmax(
        _bgr(img), model,
        accept_threshold=params.get("accept_threshold", model.accept_threshold),
        angle_low=0.0, angle_high=0.0, angle_step=1.0,
        scale_low=1.0,  scale_high=1.0,  scale_step=0.1,
        num_results=params.get("num_results", model.num_results),
    )
    clean = _bgr(img.copy())
    vis = draw_patmax_results(clean, results, model,
                                show_xy=show_xy, show_bbox=show_bbox)
    if results:
        r = results[0]
        return {"image": clean, "_display_image": vis,
                "found": True, "score": r.score,
                "x": r.x, "y": r.y, "num_found": len(results)}
    return {"image": clean, "_display_image": vis,
            "found": False, "score": 0.0,
            "x": 0.0, "y": 0.0, "num_found": 0}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: FIXTURE (Coordinate Transform)
# ═══════════════════════════════════════════════════════════════════

def proc_fixture(inputs, params):
    """
    TFixtureTool — Thiết lập hệ tọa độ dựa trên PatMax result.
    Giúp các tool sau bất biến với vị trí/góc của part.
    """
    img   = inputs.get("image")
    ref_x = float(inputs.get("ref_x", params.get("origin_x", 0)))
    ref_y = float(inputs.get("ref_y", params.get("origin_y", 0)))
    angle = float(inputs.get("ref_angle", inputs.get("angle", 0)))

    if img is None:
        return {"image":None,"transform_matrix":None,"offset_x":0.0,"offset_y":0.0,"angle":0.0}

    h, w = img.shape[:2]
    cx = params.get("origin_x", w/2); cy = params.get("origin_y", h/2)
    dx = ref_x - cx; dy = ref_y - cy

    # Build transform
    M = cv2.getRotationMatrix2D((ref_x, ref_y), -angle, 1.0)
    warped = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(30, 30, 30))

    # warpAffine returns a fresh array → vẽ overlay trực tiếp, không copy
    # (bản cũ làm `vis = warped.copy()` tốn ~30ms cho ảnh 20MP).
    vis = warped
    s = _draw_scale(vis)
    # Draw coordinate axes
    ax = int(w/2); ay = int(h/2)
    axis_len = int(60 * s)
    cv2.arrowedLine(vis, (ax, ay), (ax + axis_len, ay),
                     (0, 80, 255), _t(2, s), tipLength=0.2)
    cv2.arrowedLine(vis, (ax, ay), (ax, ay - axis_len),
                     (0, 220, 80), _t(2, s), tipLength=0.2)
    print(f"[Fixture] dx={dx:.1f} dy={dy:.1f} angle={angle:.1f}deg")

    return {"image": vis, "transform_matrix": M.tolist(),
            "offset_x": float(dx), "offset_y": float(dy),
            "angle": float(angle)}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: CALIPER
# ═══════════════════════════════════════════════════════════════════

def proc_caliper(inputs, params):
    """
    TCaliperTool — Đo cạnh (edge) chính xác sub-pixel,
    đo khoảng cách giữa 2 cạnh (Width measurement).
    """
    img = inputs.get("image")
    if img is None:
        return {"image":None,"edge1_pos":0.0,"edge2_pos":0.0,
                "width":0.0,"pass":False,"edges_found":0}

    gray = _gray(img); vis = _bgr(img.copy())
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))

    # ROI line — input port nếu connect, else params
    def _roi(k, default):
        v = inputs.get(k)
        return int(v) if v is not None else int(params.get(k, default))
    x1 = _roi("x1", img.shape[1]//4)
    y1 = _roi("y1", img.shape[0]//2)
    x2 = _roi("x2", img.shape[1]*3//4)
    y2 = _roi("y2", img.shape[0]//2)
    width_px = params.get("caliper_width", 20)
    polarity = params.get("polarity","Either")  # Dark→Light | Light→Dark | Either
    filter_half = max(1, params.get("filter_half_size", 2))
    num_edges   = params.get("num_edges", 2)
    threshold   = params.get("edge_threshold", 10.0)

    # Sample profile along caliper line
    length = int(math.hypot(x2-x1, y2-y1))
    if length < 4:
        return {"image":vis,"edge1_pos":0.0,"edge2_pos":0.0,
                "width":0.0,"pass":False,"edges_found":0}

    xs = np.linspace(x1,x2,length).astype(int)
    ys = np.linspace(y1,y2,length).astype(int)
    xs = np.clip(xs,0,gray.shape[1]-1)
    ys = np.clip(ys,0,gray.shape[0]-1)
    profile = gray[ys, xs].astype(float)

    # Gaussian derivative (T-style edge filter)
    sigma = filter_half
    kernel_size = 2*filter_half*3+1
    t = np.arange(-filter_half*3, filter_half*3+1)
    g_deriv = -t * np.exp(-t**2/(2*sigma**2))
    if len(g_deriv) < len(profile):
        deriv = np.convolve(profile, g_deriv, mode='same')
    else:
        deriv = np.gradient(profile)

    # Find edges based on polarity
    edges = []
    for i in range(filter_half, length-filter_half):
        d = deriv[i]
        if polarity == "Dark→Light" and d > threshold:
            edges.append((i, d))
        elif polarity == "Light→Dark" and d < -threshold:
            edges.append((i, abs(d)))
        elif polarity == "Either" and abs(d) > threshold:
            edges.append((i, abs(d)))

    # Sub-pixel refinement & keep strongest
    edges.sort(key=lambda e: -e[1])
    edges = edges[:num_edges]
    edges.sort(key=lambda e: e[0])

    # Draw caliper
    cv2.line(vis,(x1,y1),(x2,y2),(0,200,255),_t(1,s))
    # Width band
    angle_rad = math.atan2(y2-y1, x2-x1)
    perp_x = int(-math.sin(angle_rad)*width_px//2)
    perp_y = int( math.cos(angle_rad)*width_px//2)
    band_pts = np.array([(x1+perp_x,y1+perp_y),(x2+perp_x,y2+perp_y),
                          (x2-perp_x,y2-perp_y),(x1-perp_x,y1-perp_y)], np.int32)
    overlay = vis.copy()
    cv2.fillPoly(overlay,[band_pts],(0,200,255))
    cv2.addWeighted(vis,0.8,overlay,0.2,0,vis)

    # Draw found edges
    edge_positions = []
    for idx, (pos, strength) in enumerate(edges):
        ex = int(xs[min(pos, len(xs)-1)])
        ey = int(ys[min(pos, len(ys)-1)])
        col = (0,255,80) if idx==0 else (255,180,0)
        cv2.circle(vis,(ex,ey),_t(6,s),col,_t(2,s))
        cv2.line(vis,(ex+perp_x,ey+perp_y),(ex-perp_x,ey-perp_y),col,_t(2,s))
        if show_labels:
            cv2.putText(vis,f"E{idx+1}:{pos:.1f}",(ex+int(8*s),ey-int(8*s)),
                        cv2.FONT_HERSHEY_SIMPLEX,_fs(0.45,s),col,_t(1,s))
        edge_positions.append(float(pos))

    e1 = edge_positions[0] if len(edge_positions)>0 else 0.0
    e2 = edge_positions[1] if len(edge_positions)>1 else 0.0
    width_pix = abs(e2-e1)
    scale = params.get("pixel_to_mm", 1.0)
    width_mm = width_pix * scale

    min_w = params.get("min_width", 0.0)
    max_w = params.get("max_width", 9999.0)
    is_pass = (len(edge_positions) >= max(1,num_edges)) and (min_w <= width_mm <= max_w)
    print(f"[Caliper] width={width_mm:.3f}mm edges={len(edges)} {'PASS' if is_pass else 'FAIL'}")

    return {"image":vis,"edge1_pos":e1,"edge2_pos":e2,
            "width":width_mm,"pass":is_pass,"edges_found":len(edges)}

def proc_caliper_multi(inputs, params):
    """TCaliperTool (Multi-edge) — Tìm tất cả cạnh trong vùng."""
    img = inputs.get("image")
    if img is None:
        return {"image":None,"edges":[],"count":0,"pass":False}
    gray = _gray(img); vis = _bgr(img.copy())
    s = _draw_scale(vis)
    # ROI line — input port nếu connect, else params
    def _roi(k, default):
        v = inputs.get(k)
        return int(v) if v is not None else int(params.get(k, default))
    x1 = _roi("x1", 0)
    y1 = _roi("y1", img.shape[0]//2)
    x2 = _roi("x2", img.shape[1])
    y2 = _roi("y2", img.shape[0]//2)
    length = int(math.hypot(x2-x1, y2-y1))
    if length<4:
        return {"image":vis,"edges":[],"count":0,"pass":False}
    xs=np.clip(np.linspace(x1,x2,length).astype(int),0,gray.shape[1]-1)
    ys=np.clip(np.linspace(y1,y2,length).astype(int),0,gray.shape[0]-1)
    profile=gray[ys,xs].astype(float)
    deriv=np.gradient(profile)
    thresh=params.get("edge_threshold",10.0)
    edges=[]
    for i in range(1,length-1):
        if abs(deriv[i])>thresh and abs(deriv[i])>abs(deriv[i-1]) and abs(deriv[i])>abs(deriv[i+1]):
            edges.append({"pos":float(i),"strength":float(abs(deriv[i])),
                          "polarity":"D→L" if deriv[i]>0 else "L→D",
                          "x":int(xs[i]),"y":int(ys[i])})
    cv2.line(vis,(x1,y1),(x2,y2),(0,200,255),_t(1,s))
    for e in edges:
        cv2.circle(vis,(e["x"],e["y"]),_t(5,s),(0,255,200),_t(2,s))
    min_c=params.get("min_count",1); max_c=params.get("max_count",100)
    is_pass=min_c<=len(edges)<=max_c
    print(f"[CaliperMulti] edges={len(edges)} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"edges":edges,"count":len(edges),"pass":is_pass}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: BLOB ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def proc_blob(inputs, params):
    """
    TBlobTool — Phân tích vùng (blob) toàn diện:
    diện tích, chu vi, circularity, bounding box, centroid, orientation.

    Auto downscale: threshold + findContours chạy trên ảnh ~1.5MP
    (param `downscale`: 0=auto, ≥1=force). Tọa độ/area scale ngược về
    full-res. Cho ảnh 20MP: ~35ms → ~5ms.

    Offset (offset_x/offset_y): khi mask đến từ một crop ROI nhỏ hơn
    `image`, port `offset_x/offset_y` chỉ vị trí gốc của mask trong
    `image`. Mọi toạ độ (cx/cy, contour, bbox, label) đều được cộng
    offset → vẽ đúng vị trí trên ảnh full và `cx/cy` ra ở hệ full image.
    """
    img  = inputs.get("image")
    mask = inputs.get("mask")
    if img is None:
        return {"image":None,"count":0,"pass":False,"total_area":0.0,
                "blobs":[],"centroids":[]}

    # Offset of mask relative to image (for cropped ROI mask use case).
    # Nếu không cấp port → 0. Nếu mask cùng size với image (workflow cũ)
    # thì offset 0 không ảnh hưởng.
    ox = inputs.get("offset_x")
    oy = inputs.get("offset_y")
    off_x = int(float(ox)) if ox is not None else 0
    off_y = int(float(oy)) if oy is not None else 0

    gray = _gray(img)
    ds_param = int(params.get("downscale", 0) or 0)
    if mask is None:
        # Downscale gray trước threshold → findContours chạy trên ảnh nhỏ
        small_gray, ds = _auto_downscale(gray, ds_param)
        thresh_val = params.get("threshold", 128)
        inv = params.get("invert", False)
        t   = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        if params.get("auto_threshold", True):
            t |= cv2.THRESH_OTSU; thresh_val = 0
        _, mask_small = cv2.threshold(small_gray, thresh_val, 255, t)
    else:
        # User cung cấp mask full-res → downscale luôn để findContours nhanh
        small_gray, ds = _auto_downscale(_gray(mask), ds_param)
        # threshold lại để nhị phân hoá (mask có thể là grayscale 0-255)
        _, mask_small = cv2.threshold(small_gray, 127, 255, cv2.THRESH_BINARY)

    contours, hier = cv2.findContours(mask_small, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    scale    = params.get("pixel_to_mm2", 1.0)
    min_a    = params.get("min_area", 50.0)
    max_a    = params.get("max_area", 1e8)
    min_circ = params.get("min_circularity", 0.0)
    max_circ = params.get("max_circularity", 1.1)
    min_elo  = params.get("min_elongation", 0.0)
    max_elo  = params.get("max_elongation", 1000.0)

    vis = _bgr(img.copy())
    s = _draw_scale(vis)
    show_contours = bool(params.get("show_contours", True))
    show_bbox     = bool(params.get("show_bbox", True))
    show_centroid = bool(params.get("show_centroid", True))
    contour_thick = int(params.get("contour_thickness", 2))
    bbox_thick    = int(params.get("bbox_thickness", 1))
    _color_map    = {"Yellow":(0,200,255),"Cyan":(255,255,0),
                     "Green":(0,255,80),"Red":(0,0,255),
                     "White":(255,255,255),"Magenta":(255,0,255),
                     "Orange":(0,140,255),"Blue":(255,80,0)}
    contour_color = _color_map.get(params.get("contour_color","Yellow"),(0,200,255))
    bbox_color    = _color_map.get(params.get("bbox_color","Orange"),(0,140,255))
    centroid_color= _color_map.get(params.get("centroid_color","Green"),(0,255,80))
    show_labels   = bool(params.get("show_labels", False))
    label_dx      = int(params.get("label_dx", 6))
    label_dy      = int(params.get("label_dy", -6))
    label_size    = float(params.get("label_size", 0.45))
    label_thick   = int(params.get("label_thickness", 1))
    label_color   = {"Yellow":(0,200,255),"Cyan":(255,255,0),
                     "Green":(0,255,80),"Red":(0,0,255),
                     "White":(255,255,255),"Magenta":(255,0,255)
                    }.get(params.get("label_color","Yellow"),(0,200,255))
    label_font    = {"Simplex":cv2.FONT_HERSHEY_SIMPLEX,
                     "Plain":cv2.FONT_HERSHEY_PLAIN,
                     "Duplex":cv2.FONT_HERSHEY_DUPLEX,
                     "Complex":cv2.FONT_HERSHEY_COMPLEX,
                     "Triplex":cv2.FONT_HERSHEY_TRIPLEX,
                     "Script Simplex":cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                     "Script Complex":cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    }.get(params.get("label_font","Simplex"),cv2.FONT_HERSHEY_SIMPLEX)
    blobs = []; centroids = []; total_area = 0.0
    # Pixel extremes per blob (parallel với `blobs`) cho selection_mode —
    # mỗi blob: first/last/top/bot point trên contour (full-res image coords).
    extreme_pts: List[Dict[str, Tuple[float, float]]] = []
    label_rects = []   # mỗi entry: (x, y, w, h) trong toạ độ ảnh — hit test drag
    ds_sq = ds * ds  # tỉ lệ area downscale → full-res

    for cnt in contours:
        area_ds = cv2.contourArea(cnt)
        area = area_ds * ds_sq                  # full-res pixels²
        if area < min_a or area > max_a: continue

        perimeter_ds = cv2.arcLength(cnt, True)
        perimeter = perimeter_ds * ds
        circularity = (4*math.pi*area/(perimeter**2)) if perimeter>0 else 0
        if not (min_circ <= circularity <= max_circ): continue

        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = (M["m10"]/M["m00"]) * ds + off_x
        cy = (M["m01"]/M["m00"]) * ds + off_y

        # Rotated bounding box (minAreaRect) & orientation cho overlay/elongation
        rect = cv2.minAreaRect(cnt)
        (bx, by), (bw_ds, bh_ds), angle_deg = rect
        bw = bw_ds * ds
        bh = bh_ds * ds
        elongation = max(bw, bh) / max(min(bw, bh), 0.001)
        if not (min_elo <= elongation <= max_elo): continue

        # Axis-aligned bounding rect (cv2.boundingRect) — emit x,y,w,h
        # cho downstream tool nối thẳng (crop_roi, region_score, …) không
        # cần qua rotation. Toạ độ image full-res (đã scale + offset).
        aabb_x_ds, aabb_y_ds, aabb_w_ds, aabb_h_ds = cv2.boundingRect(cnt)
        aabb_x = aabb_x_ds * ds + off_x
        aabb_y = aabb_y_ds * ds + off_y
        aabb_w = aabb_w_ds * ds
        aabb_h = aabb_h_ds * ds

        # Convex hull & convexity (ratio nên không cần scale)
        hull = cv2.convexHull(cnt)
        hull_area_ds = cv2.contourArea(hull)
        convexity = area_ds / hull_area_ds if hull_area_ds > 0 else 0

        area_mm = area * scale
        total_area += area_mm

        blob_info = {
            "area":area_mm,"perimeter":perimeter*math.sqrt(scale),
            "circularity":circularity,"elongation":elongation,
            "convexity":convexity,"cx":float(cx),"cy":float(cy),
            "angle":float(angle_deg),"bbox_w":float(bw),"bbox_h":float(bh),
            # AABB primary outputs
            "x":float(aabb_x),"y":float(aabb_y),
            "w":float(aabb_w),"h":float(aabb_h),
        }
        blobs.append(blob_info)
        centroids.append((float(cx),float(cy)))

        # Pixel extremes trên contour (cho selection_mode). cnt là dạng
        # CHAIN_APPROX_SIMPLE → ngắn nhưng vẫn chứa các góc/extreme thực
        # của blob. Scale + offset về full-res image coords.
        _pts = cnt.reshape(-1, 2).astype(np.float32)
        if ds != 1:
            _pts = _pts * ds
        _pts[:, 0] += off_x
        _pts[:, 1] += off_y
        _lft_i = int(np.argmin(_pts[:, 0]))   # x nhỏ nhất = trái nhất
        _rgt_i = int(np.argmax(_pts[:, 0]))   # x lớn nhất = phải nhất
        _top_i = int(np.argmin(_pts[:, 1]))   # y nhỏ nhất = trên cùng
        _bot_i = int(np.argmax(_pts[:, 1]))   # y lớn nhất = dưới cùng
        extreme_pts.append({
            "first": (float(_pts[_lft_i, 0]), float(_pts[_lft_i, 1])),
            "last":  (float(_pts[_rgt_i, 0]), float(_pts[_rgt_i, 1])),
            "top":   (float(_pts[_top_i, 0]), float(_pts[_top_i, 1])),
            "bot":   (float(_pts[_bot_i, 0]), float(_pts[_bot_i, 1])),
        })

        # Draw — scale contour + bbox lên full-res, cộng offset để vẽ
        # đúng vị trí mask trong image.
        cnt_full = (cnt * ds).astype(np.int32) if ds > 1 else cnt.astype(np.int32)
        if off_x or off_y:
            cnt_full = cnt_full + np.array([[off_x, off_y]], dtype=np.int32)
        box_full = cv2.boxPoints(((bx*ds + off_x, by*ds + off_y), (bw, bh),
                                    angle_deg)).astype(np.int32)
        if show_contours:
            cv2.drawContours(vis,[cnt_full],-1,contour_color,_t(contour_thick,s))
        if show_bbox:
            cv2.drawContours(vis,[box_full],-1,bbox_color,_t(bbox_thick,s))
        # Centroid dot KHÔNG vẽ ở đây — sẽ vẽ 1 chấm duy nhất tại pixel
        # được chọn (theo selection_mode) sau loop, để chấm + vòng magenta
        # cùng di chuyển khi user đổi mode.
        if show_labels:
            # mm² ký tự Unicode → Hershey font không render được (ra "??").
            # Dùng "mm2" ASCII.
            text = f"{area_mm:.1f} mm2"
            tx = int(cx) + int(label_dx * s)
            ty = int(cy) + int(label_dy * s)
            fs_val = _fs(label_size, s)
            th_val = _t(label_thick, s)
            cv2.putText(vis, text, (tx, ty),
                        label_font, fs_val, label_color, th_val,
                        cv2.LINE_AA)
            # Bounding rect của text → dialog hit-test khi drag
            (tw, th), bl = cv2.getTextSize(text, label_font, fs_val, th_val)
            label_rects.append((tx, ty - th - bl, tw, th + bl))

    min_cnt = params.get("min_count", 1)
    max_cnt = params.get("max_count", 1000)
    is_pass = min_cnt <= len(blobs) <= max_cnt
    print(f"[Blob] count={len(blobs)} total_area={total_area:.2f}mm² {'PASS' if is_pass else 'FAIL'}")

    # Scalar shortcuts theo `selection_mode` — chọn 1 PIXEL trên contour
    # blob → x,y,cx,cy phản ánh đúng tọa độ pixel đó:
    #   First   → pixel trái nhất (x nhỏ nhất) trên mọi contour
    #   Last    → pixel phải nhất (x lớn nhất) trên mọi contour
    #   Highest → pixel cao nhất (y nhỏ nhất, trên cùng ảnh)
    #   Lowest  → pixel thấp nhất (y lớn nhất, dưới cùng ảnh)
    #   Average → trung tâm trung bình (mean của centroids)
    # w,h,area,bbox_w,bbox_h,angle giữ semantic cũ = property của blob
    # "owner" (chứa pixel được chọn) hoặc trung bình cho Average mode.
    _scalar_keys = ["x","y","w","h","cx","cy","area","bbox_w","bbox_h","angle"]
    sel_mode = params.get("selection_mode", "First")
    sel_pt = None     # (x, y) pixel được chọn → dùng cho output + marker
    if not blobs:
        _sel = {k: 0.0 for k in _scalar_keys}
    else:
        owner_idx = None  # blob "sở hữu" pixel được chọn
        if sel_mode == "Last":
            owner_idx = max(range(len(blobs)),
                            key=lambda i: extreme_pts[i]["last"][0])
            sel_pt = extreme_pts[owner_idx]["last"]
        elif sel_mode == "Highest":
            owner_idx = min(range(len(blobs)),
                            key=lambda i: extreme_pts[i]["top"][1])
            sel_pt = extreme_pts[owner_idx]["top"]
        elif sel_mode == "Lowest":
            owner_idx = max(range(len(blobs)),
                            key=lambda i: extreme_pts[i]["bot"][1])
            sel_pt = extreme_pts[owner_idx]["bot"]
        elif sel_mode == "Average":
            _n = len(blobs)
            sel_pt = (sum(float(b["cx"]) for b in blobs) / _n,
                      sum(float(b["cy"]) for b in blobs) / _n)
        else:  # "First" — default = pixel trái nhất
            owner_idx = min(range(len(blobs)),
                            key=lambda i: extreme_pts[i]["first"][0])
            sel_pt = extreme_pts[owner_idx]["first"]

        if owner_idx is None:  # Average — trung bình mọi field
            _n = len(blobs)
            _sel = {k: sum(float(b.get(k, 0.0)) for b in blobs) / _n
                    for k in _scalar_keys}
        else:
            _sel = {k: float(blobs[owner_idx].get(k, 0.0))
                    for k in _scalar_keys}
        # Override x,y,cx,cy = pixel được chọn (user thấy đúng tọa độ)
        _sel["x"]  = sel_pt[0]; _sel["y"]  = sel_pt[1]
        _sel["cx"] = sel_pt[0]; _sel["cy"] = sel_pt[1]

    # Centroid dot + highlight ring tại pixel được chọn → user thấy rõ
    # vị trí ứng với x,y output. Cả chấm centroid_color và vòng magenta
    # đều nhảy theo selection_mode để khớp với (x,y) emit ra port.
    if sel_pt is not None and show_centroid:
        _px, _py = int(sel_pt[0]), int(sel_pt[1])
        cv2.circle(vis, (_px, _py), _t(4, s), centroid_color, -1)
        cv2.circle(vis, (_px, _py), _t(10, s), (255, 0, 255), _t(2, s))

    return {"image":vis,"count":len(blobs),"pass":is_pass,
            "total_area":total_area,"blobs":blobs,"centroids":centroids,
            "x": _sel["x"], "y": _sel["y"], "w": _sel["w"], "h": _sel["h"],
            "cx": _sel["cx"], "cy": _sel["cy"], "area": _sel["area"],
            "bbox_w": _sel["bbox_w"], "bbox_h": _sel["bbox_h"],
            "angle": _sel["angle"],
            # _label_rects: list (x,y,w,h) image coords — UI dùng để hit-test
            # khi user kéo label trên canvas. Không expose qua port.
            "_label_rects": label_rects,
            "_label_centroids": [(float(cx2), float(cy2)) for (cx2, cy2) in centroids]}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: EDGE / LINE FIND
# ═══════════════════════════════════════════════════════════════════

def proc_find_line(inputs, params):
    """TFindLineTool — Tìm đường thẳng từ các điểm edge (least-squares).

    Tối ưu: chỉ Canny trong band ROI (không phải full ảnh) → giảm hẳn
    cost cho ảnh lớn. Auto downscale band rộng nếu ảnh > 1.5MP.
    """
    img = inputs.get("image")
    if img is None:
        return {"image":None,"found":False,"angle":0.0,"distance":0.0,
                "point_x":0.0,"point_y":0.0,"pass":False}
    gray = _gray(img); vis = _bgr(img.copy())
    s = _draw_scale(vis)
    h, w = gray.shape
    t1 = params.get("canny_low", 50); t2 = params.get("canny_high", 150)

    # ROI band — input port nếu connect, else params (full-res coord)
    def _roi(k, default):
        v = inputs.get(k)
        return int(v) if v is not None else int(params.get(k, default))
    rx1 = max(0, _roi("x1", 0))
    ry1 = max(0, _roi("y1", h//2 - 30))
    rx2 = min(w, _roi("x2", w))
    ry2 = min(h, _roi("y2", h//2 + 30))
    if rx2 <= rx1 or ry2 <= ry1:
        return {"image": vis, "found": False, "angle": 0.0, "distance": 0.0,
                "point_x": float(w/2), "point_y": float(h/2), "pass": False}

    # Crop ROI trước, sau đó Canny CHỈ trên band → tiết kiệm O(W*H/(roi_w*roi_h))
    band = gray[ry1:ry2, rx1:rx2]
    # Auto downscale nếu band vẫn lớn
    band_small, ds = _auto_downscale(band, params.get("downscale", 0))
    edges_small = cv2.Canny(band_small, t1, t2)

    # Lấy điểm edge ở band space, scale + offset về full-res
    pts_yx = np.column_stack(np.where(edges_small > 0))
    found = False; angle = 0.0; dist = 0.0
    px = float(w/2); py = float(h/2)
    if len(pts_yx) > 5:
        xs = pts_yx[:, 1].astype(np.float32) * ds + rx1
        ys = pts_yx[:, 0].astype(np.float32) * ds + ry1
        fit = cv2.fitLine(
            np.column_stack([xs, ys]), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        vx = float(fit[0]); vy = float(fit[1])
        x0 = float(fit[2]); y0 = float(fit[3])
        angle = float(math.degrees(math.atan2(vy, vx)))
        px = x0; py = y0
        t_range = max(w, h) * 2
        pt1 = (int(px - vx*t_range), int(py - vy*t_range))
        pt2 = (int(px + vx*t_range), int(py + vy*t_range))
        cv2.line(vis, pt1, pt2, (0, 220, 80), _t(2, s))
        cv2.circle(vis, (int(px), int(py)), _t(6, s), (0, 220, 80), -1)
        dist = float(math.hypot(px - w/2, py - h/2)) \
               * params.get("pixel_to_mm", 1.0)
        found = True

    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 150, 200), _t(1, s))
    ang_min = params.get("min_angle", -180.0)
    ang_max = params.get("max_angle", 180.0)
    is_pass = found and (ang_min <= angle <= ang_max)
    print(f"[FindLine] angle={angle:.2f}deg "
          f"{'PASS' if is_pass else ('FAIL' if found else 'NOT FOUND')} ds={ds}")
    return {"image": vis, "found": found, "angle": angle, "distance": dist,
            "point_x": px, "point_y": py, "pass": is_pass}

def proc_find_circle(inputs, params):
    """TFindCircleTool — Tìm & fit đường tròn chính xác."""
    img = inputs.get("image")
    if img is None:
        return {"image":None,"found":False,"cx":0.0,"cy":0.0,
                "radius":0.0,"pass":False}
    gray=_gray(img); vis=_bgr(img.copy())
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))

    # Coarse downscale cho HoughCircles — auto ~1.5MP.
    small, ds = _auto_downscale(gray, params.get("downscale", 0))
    blurred = cv2.GaussianBlur(small, (9, 9), 2)
    # Scale radius/dist params về coord space của ảnh downscaled
    raw_min_r = float(params.get("min_radius", 5))
    raw_max_r = float(params.get("max_radius", 300))
    raw_min_d = float(params.get("min_dist", 30))
    min_r_ds = max(3, int(round(raw_min_r / ds)))
    max_r_ds = max(min_r_ds + 1, int(round(raw_max_r / ds)))
    min_d_ds = max(3, int(round(raw_min_d / ds)))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
        params.get("dp", 1.2), min_d_ds,
        param1=params.get("param1", 100), param2=params.get("param2", 30),
        minRadius=min_r_ds, maxRadius=max_r_ds)
    found=False; cx=0.0; cy=0.0; radius=0.0
    if circles is not None:
        c = circles[0][0]
        # Scale lại về full-res coords
        cx = float(c[0]) * ds
        cy = float(c[1]) * ds
        radius = float(c[2]) * ds
        found=True
        cv2.circle(vis,(int(cx),int(cy)),int(radius),(0,220,80),_t(2,s))
        cv2.circle(vis,(int(cx),int(cy)),_t(3,s),(0,220,80),-1)
        cv2.line(vis,(int(cx),int(cy)),(int(cx+radius),int(cy)),(100,200,255),_t(1,s))
        px2mm=params.get("pixel_to_mm",1.0)
        if show_labels:
            cv2.putText(vis,f"R={radius*px2mm:.3f}mm cx={cx:.1f} cy={cy:.1f}",
                        (int(cx-radius),max(0,int(cy-radius)-int(8*s))),
                        cv2.FONT_HERSHEY_SIMPLEX,_fs(0.55,s),(0,220,80),_t(2,s))
    min_r=params.get("min_r_check",0.0); max_r=params.get("max_r_check",9999.0)
    is_pass=found and (min_r<=radius*params.get("pixel_to_mm",1.0)<=max_r)
    print(f"[FindCircle] {'PASS' if is_pass else ('FAIL' if found else 'NOT FOUND')} r={radius:.2f}px ds={ds}")
    return {"image":vis,"found":found,"cx":cx,"cy":cy,
            "radius":radius*params.get("pixel_to_mm",1.0),"pass":is_pass}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: COLOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def proc_color_picker(inputs, params):
    """TColorTool (Picker) — Click chuột lấy màu → xuất HSV/RGB range."""
    img=inputs.get("image")
    if img is None:
        return {"image":None,"color_hsv":None,"h":0,"s":0,"v":0,"r":0,"g":0,"b":0}
    bgr=_bgr(img)
    # Pick point — input port nếu connect (vd track theo PatMax), else params
    vx = inputs.get("pick_x")
    vy = inputs.get("pick_y")
    x = int(vx) if vx is not None else int(params.get("pick_x", 0))
    y = int(vy) if vy is not None else int(params.get("pick_y", 0))
    H2,W2=bgr.shape[:2]
    x=max(0,min(x,W2-1)); y=max(0,min(y,H2-1))
    B,G,R=int(bgr[y,x,0]),int(bgr[y,x,1]),int(bgr[y,x,2])
    hsv_img=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    H,S,V=int(hsv_img[y,x,0]),int(hsv_img[y,x,1]),int(hsv_img[y,x,2])
    tol=params.get("tolerance",20)
    color_hsv=[max(0,H-tol),max(0,S-tol),max(0,V-tol),
               min(180,H+tol),min(255,S+tol),min(255,V+tol)]
    vis=bgr.copy()
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))
    cv2.circle(vis,(x,y),_t(10,s),(0,255,255),_t(2,s))
    sw_x1=x+int(12*s); sw_y1=y-int(20*s); sw_x2=x+int(42*s); sw_y2=y+int(10*s)
    cv2.rectangle(vis,(sw_x1,sw_y1),(sw_x2,sw_y2),(B,G,R),-1)
    cv2.rectangle(vis,(sw_x1,sw_y1),(sw_x2,sw_y2),(255,255,255),_t(1,s))
    if show_labels:
        cv2.putText(vis,f"H{H} S{S} V{V}",(x+int(46*s),y),cv2.FONT_HERSHEY_SIMPLEX,_fs(0.5,s),(0,255,255),_t(1,s))
    return {"image":vis,"color_hsv":color_hsv,"h":H,"s":S,"v":V,"r":R,"g":G,"b":B}

def _shape_to_mask(shape_type: str, shape_data: dict,
                   h: int, w: int) -> Optional[np.ndarray]:
    """Build a binary mask (uint8 0/255) of an ROI shape on an h×w canvas.
    Returns None nếu shape không hợp lệ."""
    if not shape_type or not shape_data:
        return None
    m = np.zeros((h, w), dtype=np.uint8)
    try:
        if shape_type == "rect":
            x = int(shape_data.get("x", 0)); y = int(shape_data.get("y", 0))
            rw = int(shape_data.get("w", 0)); rh = int(shape_data.get("h", 0))
            if rw <= 0 or rh <= 0: return None
            cv2.rectangle(m, (x, y), (x + rw, y + rh), 255, -1)
        elif shape_type == "ellipse":
            x = int(shape_data.get("x", 0)); y = int(shape_data.get("y", 0))
            rw = int(shape_data.get("w", 0)); rh = int(shape_data.get("h", 0))
            if rw <= 0 or rh <= 0: return None
            cv2.ellipse(m, (x + rw // 2, y + rh // 2),
                        (max(1, rw // 2), max(1, rh // 2)), 0, 0, 360, 255, -1)
        elif shape_type == "circle":
            cx = int(shape_data.get("cx", 0)); cy = int(shape_data.get("cy", 0))
            r = int(shape_data.get("r", 0))
            if r <= 0: return None
            cv2.circle(m, (cx, cy), r, 255, -1)
        elif shape_type == "polygon":
            pts = shape_data.get("pts") or []
            if len(pts) < 3: return None
            arr = np.array([[int(px), int(py)] for px, py in pts], dtype=np.int32)
            cv2.fillPoly(m, [arr], 255)
        else:
            return None
    except Exception:
        return None
    return m


def _apply_shape_port_overrides(shape_type: str, shape_data: dict,
                                  px, py, pw, ph) -> dict:
    """Override vị trí/size của shape từ input ports x/y/w/h.
    None = giữ giá trị cũ. Trả về dict mới (không mutate input).

    • rect / ellipse: x, y, w, h ghi đè trực tiếp lên bbox.
    • circle: x, y → top-left bbox; w (hoặc h) → đường kính.
    • polygon: x, y → dịch toàn bộ pts; w, h hiện chưa scale (giữ size).
    """
    if not shape_type or not shape_data:
        return shape_data
    if px is None and py is None and pw is None and ph is None:
        return shape_data
    out = dict(shape_data)
    try:
        if shape_type in ("rect", "ellipse"):
            if px is not None: out["x"] = int(float(px))
            if py is not None: out["y"] = int(float(py))
            if pw is not None: out["w"] = max(1, int(float(pw)))
            if ph is not None: out["h"] = max(1, int(float(ph)))
        elif shape_type == "circle":
            old_r = int(out.get("r", 0))
            new_r = old_r
            if pw is not None:
                new_r = max(1, int(float(pw)) // 2)
            elif ph is not None:
                new_r = max(1, int(float(ph)) // 2)
            new_cx = int(out.get("cx", 0))
            new_cy = int(out.get("cy", 0))
            if px is not None: new_cx = int(float(px)) + new_r
            if py is not None: new_cy = int(float(py)) + new_r
            out["cx"] = new_cx; out["cy"] = new_cy; out["r"] = new_r
            out["x"] = new_cx - new_r; out["y"] = new_cy - new_r
            out["w"] = out["h"] = 2 * new_r
        elif shape_type == "polygon":
            pts = out.get("pts") or []
            if not pts:
                return out
            old_x = int(out.get("x", min(int(p[0]) for p in pts)))
            old_y = int(out.get("y", min(int(p[1]) for p in pts)))
            new_x = int(float(px)) if px is not None else old_x
            new_y = int(float(py)) if py is not None else old_y
            dx = new_x - old_x; dy = new_y - old_y
            if dx != 0 or dy != 0:
                out["pts"] = [(p[0] + dx, p[1] + dy) for p in pts]
                out["x"] = new_x; out["y"] = new_y
    except Exception:
        return shape_data
    return out


def _draw_shape_outline(vis: np.ndarray, shape_type: str,
                        shape_data: dict, color=(0, 200, 255), thick=2):
    """Vẽ outline shape lên ảnh visualization (in-place)."""
    if not shape_type or not shape_data:
        return
    try:
        if shape_type == "rect":
            x = int(shape_data.get("x", 0)); y = int(shape_data.get("y", 0))
            w = int(shape_data.get("w", 0)); h = int(shape_data.get("h", 0))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thick)
        elif shape_type == "ellipse":
            x = int(shape_data.get("x", 0)); y = int(shape_data.get("y", 0))
            w = int(shape_data.get("w", 0)); h = int(shape_data.get("h", 0))
            cv2.ellipse(vis, (x + w // 2, y + h // 2),
                        (max(1, w // 2), max(1, h // 2)), 0, 0, 360, color, thick)
        elif shape_type == "circle":
            cx = int(shape_data.get("cx", 0)); cy = int(shape_data.get("cy", 0))
            r = int(shape_data.get("r", 0))
            cv2.circle(vis, (cx, cy), r, color, thick)
        elif shape_type == "polygon":
            pts = shape_data.get("pts") or []
            if len(pts) >= 2:
                arr = np.array([[int(px), int(py)] for px, py in pts], dtype=np.int32)
                cv2.polylines(vis, [arr], True, color, thick)
    except Exception:
        pass


def proc_color_segment(inputs, params):
    """TColorSegmenterTool — Phân đoạn theo không gian màu, xuất mask + ratio.

    Hỗ trợ `color_space`: HSV (default) / RGB / HSL / Lab / Gray. Mỗi mode
    có bộ low/high riêng (visible_if ẩn các slider không thuộc mode hiện
    hành). Auto downscale BGR→space + inRange + morph chạy trên ảnh
    ~1.5MP (`downscale`: 0=auto, ≥1=force); ratio scale-invariant, mask
    output resize lại full-res. Pixel count báo theo full-res space.
    """
    img=inputs.get("image")
    if img is None:
        return {"image":None,"mask":None,"pass":False,"pixel_ratio":0.0,"pixel_count":0}
    bgr = _bgr(img)
    H, W = bgr.shape[:2]
    color_space = params.get("color_space", "HSV")
    tol = int(params.get("tolerance", 0) or 0)

    # ── Build low/high vector cho mode hiện hành ──────────────────────
    # Picked color override (HSV only — port `color_hsv` đến từ color_picker)
    picked = inputs.get("color_hsv", None)
    wrap_hue = False           # True → cần handle hue wrap-around
    hue_max = 180              # OpenCV HSV/HLS dùng H 0-180
    if color_space == "HSV" and picked \
            and isinstance(picked, (list, tuple)) and len(picked) == 6:
        c1_lo, c2_lo, c3_lo, c1_hi, c2_hi, c3_hi = [int(x) for x in picked]
        wrap_hue = True
    elif color_space == "HSV":
        c1_lo = params.get("h_low", 0);   c1_hi = params.get("h_high", 180)
        c2_lo = params.get("s_low", 50);  c2_hi = params.get("s_high", 255)
        c3_lo = params.get("v_low", 50);  c3_hi = params.get("v_high", 255)
        wrap_hue = True
    elif color_space == "RGB":
        # inRange chạy trên BGR — đổi thứ tự thresholds tương ứng.
        r_lo = params.get("r_low", 0);  r_hi = params.get("r_high", 255)
        g_lo = params.get("g_low", 0);  g_hi = params.get("g_high", 255)
        b_lo = params.get("b_low", 0);  b_hi = params.get("b_high", 255)
        c1_lo, c2_lo, c3_lo = b_lo, g_lo, r_lo
        c1_hi, c2_hi, c3_hi = b_hi, g_hi, r_hi
    elif color_space == "HSL":
        # OpenCV gọi là HLS (thứ tự: H, L, S)
        c1_lo = params.get("hsl_h_low", 0);   c1_hi = params.get("hsl_h_high", 180)
        c2_lo = params.get("hsl_l_low", 0);   c2_hi = params.get("hsl_l_high", 255)
        c3_lo = params.get("hsl_s_low", 0);   c3_hi = params.get("hsl_s_high", 255)
        wrap_hue = True
    elif color_space == "Lab":
        c1_lo = params.get("lab_l_low", 0);   c1_hi = params.get("lab_l_high", 255)
        c2_lo = params.get("lab_a_low", 0);   c2_hi = params.get("lab_a_high", 255)
        c3_lo = params.get("lab_b_low", 0);   c3_hi = params.get("lab_b_high", 255)
    elif color_space == "Gray":
        c1_lo = params.get("gray_low", 0);    c1_hi = params.get("gray_high", 255)
        c2_lo = c3_lo = 0; c2_hi = c3_hi = 255
    else:
        # Fallback an toàn → HSV defaults
        c1_lo = 0; c1_hi = 180; c2_lo = 50; c2_hi = 255; c3_lo = 50; c3_hi = 255
        wrap_hue = True
        color_space = "HSV"

    # Tolerance áp dụng cho mọi kênh; clamp theo max range
    if tol > 0:
        ch_max = (hue_max if wrap_hue else 255, 255, 255)
        c1_lo = max(0, c1_lo - tol); c1_hi = min(ch_max[0], c1_hi + tol)
        c2_lo = max(0, c2_lo - tol); c2_hi = min(ch_max[1], c2_hi + tol)
        c3_lo = max(0, c3_lo - tol); c3_hi = min(ch_max[2], c3_hi + tol)

    # ── Downscale BGR trước cvtColor ─────────────────────────────────
    ds_param = int(params.get("downscale", 0) or 0)
    if ds_param > 0:
        ds = max(1, ds_param)
    else:
        ds = max(1, int(round((H * W / _DETECT_TARGET_PX) ** 0.5)))
    if ds > 1:
        bgr_small = cv2.resize(bgr, None, fx=1.0/ds, fy=1.0/ds,
                                interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr

    # ── Convert + threshold ──────────────────────────────────────────
    if color_space == "HSV":
        cs_img = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    elif color_space == "RGB":
        cs_img = bgr_small        # threshold dùng thứ tự B,G,R
    elif color_space == "HSL":
        cs_img = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HLS)
    elif color_space == "Lab":
        cs_img = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    else:  # Gray
        cs_img = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)

    if color_space == "Gray":
        mask_small = cv2.inRange(cs_img, c1_lo, c1_hi)
    elif wrap_hue and c1_lo > c1_hi:
        # Hue wrap-around (vd đỏ: 170-180 ∪ 0-10)
        m1 = cv2.inRange(cs_img,
                          np.array([0, c2_lo, c3_lo]),
                          np.array([c1_hi, c2_hi, c3_hi]))
        m2 = cv2.inRange(cs_img,
                          np.array([c1_lo, c2_lo, c3_lo]),
                          np.array([hue_max, c2_hi, c3_hi]))
        mask_small = cv2.bitwise_or(m1, m2)
    else:
        mask_small = cv2.inRange(cs_img,
                                  np.array([c1_lo, c2_lo, c3_lo]),
                                  np.array([c1_hi, c2_hi, c3_hi]))

    k=params.get("morph_open",0)
    if k>0:
        # Kernel size scale theo ds để giữ semantic (kernel ý nghĩa full-res)
        kk = max(1, int(round(k / max(1, ds))))
        mask_small=cv2.morphologyEx(mask_small,cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kk,kk)))

    # ROI shape: build mask ở full-res rồi resize xuống small để bitwise_and.
    # ratio = inside/area_of_shape (đều ở small space → tỉ lệ giữ nguyên).
    # Port x/y/w/h ưu tiên hơn drawn shape — auto-tracking theo upstream
    # (PatMax/Fixture). None = giữ giá trị drawn.
    roi_shape_type = params.get("_roi_shape_type")
    roi_shape_data = params.get("_roi_shape_data")
    port_x = inputs.get("x"); port_y = inputs.get("y")
    port_w = inputs.get("w"); port_h = inputs.get("h")
    any_port = (port_x is not None or port_y is not None
                or port_w is not None or port_h is not None)
    # Case "port truyền x/y mà chưa vẽ shape" → synth rect (hoặc circle/
    # ellipse theo dropdown) từ params roi_w/roi_h + port values.
    if (not roi_shape_data) and any_port:
        rs = (params.get("roi_shape", "Full Image") or "Full Image")
        if rs != "Full Image":
            fb_w = int(params.get("roi_w", 100) or 100)
            fb_h = int(params.get("roi_h", 100) or 100)
            rx = int(float(port_x)) if port_x is not None else 0
            ry = int(float(port_y)) if port_y is not None else 0
            rw = max(1, int(float(port_w))) if port_w is not None else fb_w
            rh = max(1, int(float(port_h))) if port_h is not None else fb_h
            if rs == "Circle":
                r = max(1, min(rw, rh) // 2)
                roi_shape_type = "circle"
                roi_shape_data = {"cx": rx + r, "cy": ry + r, "r": r,
                                   "x": rx, "y": ry, "w": 2 * r, "h": 2 * r}
            elif rs == "Ellipse":
                roi_shape_type = "ellipse"
                roi_shape_data = {"x": rx, "y": ry, "w": rw, "h": rh}
            else:  # Rectangle (default); Polygon cũng fallback rect
                roi_shape_type = "rect"
                roi_shape_data = {"x": rx, "y": ry, "w": rw, "h": rh}
    else:
        roi_shape_data = _apply_shape_port_overrides(
            roi_shape_type, roi_shape_data,
            port_x, port_y, port_w, port_h)
    roi_mask_full = _shape_to_mask(roi_shape_type, roi_shape_data, H, W)
    if roi_mask_full is not None:
        if ds > 1:
            roi_mask_small = cv2.resize(
                roi_mask_full, (mask_small.shape[1], mask_small.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        else:
            roi_mask_small = roi_mask_full
        mask_small = cv2.bitwise_and(mask_small, roi_mask_small)
        denom_small = int(np.count_nonzero(roi_mask_small)) or 1
    else:
        denom_small = mask_small.size

    cnt_small = int(np.count_nonzero(mask_small))
    ratio = cnt_small / denom_small               # scale-invariant
    cnt = cnt_small * (ds * ds)                   # full-res pixel count
    min_r=params.get("min_ratio",0.01); max_r=params.get("max_ratio",1.0)
    is_pass=min_r<=ratio<=max_r

    # Resize mask về full-res cho output port + visualization.
    if ds > 1:
        mask = cv2.resize(mask_small, (W, H),
                          interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small

    vis=bgr.copy(); overlay=vis.copy()
    overlay[mask>0]=[0,220,80]; cv2.addWeighted(vis,0.55,overlay,0.45,0,vis)
    if roi_mask_full is not None:
        # Làm tối phần ngoài ROI để dễ thấy vùng đang xét
        dim = (vis * 0.35).astype(np.uint8)
        vis = np.where(roi_mask_full[..., None] > 0, vis, dim)
        _draw_shape_outline(vis, roi_shape_type, roi_shape_data,
                            color=(0, 200, 255),
                            thick=max(1, int(round(_draw_scale(vis) * 2))))
    print(f"[ColorSeg/{color_space}] ratio={ratio:.3f} pixels={cnt}/{denom_small*ds*ds} "
          f"ds={ds} {'PASS' if is_pass else 'FAIL'}")
    out={"image":vis,"mask":mask,"pass":is_pass,"pixel_ratio":ratio,"pixel_count":cnt}
    # Expose effective ROI shape (đã apply port overrides) cho UI vẽ overlay.
    if roi_shape_type and roi_shape_data:
        out["_effective_roi_shape_type"] = roi_shape_type
        out["_effective_roi_shape_data"] = roi_shape_data
    if params.get("show_mask",False):
        out["_display_image"]=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return out

def proc_color_match(inputs, params):
    """TColorMatchTool — So khớp màu trung bình trong ROI với màu tham chiếu."""
    img=inputs.get("image")
    if img is None:
        return {"image":None,"pass":False,"delta_e":0.0,"mean_r":0,"mean_g":0,"mean_b":0}
    bgr=_bgr(img)
    # ROI rect — input port nếu connect, else params
    def _roi(k, default):
        v = inputs.get(k)
        return int(v) if v is not None else int(params.get(k, default))
    x = _roi("x", 0); y = _roi("y", 0)
    w = _roi("w", 50); h = _roi("h", 50)
    H2,W2=bgr.shape[:2]
    x=max(0,min(x,W2-1)); y=max(0,min(y,H2-1))
    w=max(1,min(w,W2-x)); h=max(1,min(h,H2-y))
    roi=bgr[y:y+h, x:x+w]
    mean_b,mean_g,mean_r,_=cv2.mean(roi)
    # Reference color
    ref_r=params.get("ref_r",128); ref_g=params.get("ref_g",128); ref_b=params.get("ref_b",128)
    # Delta E approximation (Euclidean in RGB)
    delta_e=math.sqrt((mean_r-ref_r)**2+(mean_g-ref_g)**2+(mean_b-ref_b)**2)
    max_de=params.get("max_delta_e",30.0)
    is_pass=delta_e<=max_de
    vis=bgr.copy()
    s = _draw_scale(vis)
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,212,255),_t(2,s))
    # Color swatches
    sw_h=int(20*s); sw_w=int(20*s); pad=int(2*s); gap=int(4*s)
    cv2.rectangle(vis,(x,y+h+pad),(x+sw_w,y+h+pad+sw_h),(int(mean_b),int(mean_g),int(mean_r)),-1)
    cv2.rectangle(vis,(x+sw_w+gap,y+h+pad),(x+2*sw_w+gap,y+h+pad+sw_h),(ref_b,ref_g,ref_r),-1)
    print(f"[ColorMatch] dE={delta_e:.2f} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"pass":is_pass,"delta_e":delta_e,
            "mean_r":int(mean_r),"mean_g":int(mean_g),"mean_b":int(mean_b)}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: ID / READ
# ═══════════════════════════════════════════════════════════════════

def proc_id_reader(inputs, params):
    """TIDReaderTool — Đọc Barcode 1D/2D, QR, DataMatrix."""
    img=inputs.get("image")
    if img is None: return {"image":None,"data":"","symbology":"","pass":False}
    gray=_gray(img); vis=_bgr(img.copy())
    s = _draw_scale(vis)
    data=""; symbology=""
    try:
        from pyzbar.pyzbar import decode
        decoded=decode(gray)
        if decoded:
            obj=decoded[0]; data=obj.data.decode("utf-8",errors="replace")
            symbology=obj.type
            pts=obj.polygon
            if pts:
                hull=cv2.convexHull(np.array([(p.x,p.y) for p in pts],dtype=np.int32))
                cv2.polylines(vis,[hull],True,(0,220,80),_t(2,s))
    except ImportError:
        try:
            det=cv2.QRCodeDetector()
            qr_data,pts,_=det.detectAndDecode(gray)
            if qr_data: data=qr_data; symbology="QR"
            if pts is not None:
                pts2=pts.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(vis,[pts2],True,(0,220,80),_t(2,s))
        except: pass

    expected=params.get("expected_data","")
    is_pass=bool(data) and (expected in data if expected else True)
    print(f"[IDReader] {symbology or 'NONE'}: {data or 'NOT FOUND'} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"data":data,"symbology":symbology,"pass":is_pass}

_EASYOCR_READERS: Dict[tuple, Any] = {}

def _ocr_preprocess(gray, mode: str, invert: bool):
    """Binarize/invert grayscale để OCR engine bám tốt hơn trên nền
    phức tạp. Adaptive xử lý sáng không đều, Otsu cho contrast cao đều,
    binary cho ngưỡng cố định 127."""
    if gray is None:
        return None
    out = gray
    m = (mode or "none").lower()
    if m == "otsu":
        _, out = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif m == "adaptive":
        out = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    elif m == "binary":
        _, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if invert:
        out = 255 - out
    return out

def _ocr_lang_to_easyocr(lang: str) -> List[str]:
    """Map Tesseract 3-char codes (eng, vie, jpn, …) → EasyOCR
    2-char codes (en, vi, ja, …). Hỗ trợ 'vie+eng' style multi-lang."""
    mapping = {"eng":"en", "vie":"vi", "jpn":"ja", "kor":"ko",
               "chi_sim":"ch_sim", "chi_tra":"ch_tra",
               "fra":"fr", "deu":"de", "spa":"es", "rus":"ru",
               "ita":"it", "por":"pt", "tha":"th"}
    codes = []
    for part in str(lang or "eng").replace("+", ",").split(","):
        p = part.strip().lower()
        if not p:
            continue
        codes.append(mapping.get(p, p))
    return codes or ["en"]

def _get_easyocr_reader(langs: List[str]):
    """Lazy-init + cache EasyOCR Reader (load model ~5-15s lần đầu)."""
    key = tuple(sorted(set(langs)))
    r = _EASYOCR_READERS.get(key)
    if r is not None:
        return r
    import easyocr
    r = easyocr.Reader(list(key), gpu=False, verbose=False)
    _EASYOCR_READERS[key] = r
    return r

def proc_ocr_max(inputs, params):
    """TOCRMaxTool — Đọc & xác nhận ký tự (OCR).

    Engines:
      • tesseract — pytesseract + Tesseract binary + langpack hệ thống
        (Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-vie`,
         pip:    `pip install pytesseract`).
      • easyocr   — `pip install easyocr` (không cần binary, tốt với
        diacritics + nền phức tạp; load model lần đầu ~5-15s).
      • auto      — thử tesseract trước, fallback easyocr nếu fail.

    Preprocess: 'otsu' / 'adaptive' / 'binary' giúp tách chữ khỏi nền
    phức tạp trước khi đẩy vào engine.
    """
    img = inputs.get("image")
    if img is None:
        return {"image": None, "text": "", "pass": False, "confidence": 0.0}
    gray = _gray(img)
    vis = _bgr(img.copy())
    s = _draw_scale(vis)

    engine = str(params.get("engine", "auto")).lower()
    preprocess = str(params.get("preprocess", "none")).lower()
    invert = bool(params.get("invert", False))
    lang = params.get("lang", "eng")
    psm = int(params.get("psm", 6))

    proc_gray = _ocr_preprocess(gray, preprocess, invert)

    text = ""
    conf = 0.0
    used = ""
    err = ""

    def _run_tesseract():
        import pytesseract
        data = pytesseract.image_to_data(
            proc_gray, lang=lang,
            config=f"--psm {psm} --oem 3",
            output_type=pytesseract.Output.DICT)
        t_acc, c_acc, words = "", 0.0, 0
        for i, t in enumerate(data["text"]):
            c = int(data["conf"][i])
            if c > 0 and t.strip():
                t_acc += t + " "
                c_acc += c
                words += 1
                x2, y2, w2, h2 = (data["left"][i], data["top"][i],
                                   data["width"][i], data["height"][i])
                cv2.rectangle(vis, (x2, y2), (x2 + w2, y2 + h2),
                              (0, 200, 255), _t(1, s))
        return t_acc.strip(), c_acc / max(words, 1)

    def _run_easyocr():
        langs = _ocr_lang_to_easyocr(lang)
        reader = _get_easyocr_reader(langs)
        results = reader.readtext(proc_gray)
        t_acc, c_acc, words = "", 0.0, 0
        for box, t, c in results:
            if not str(t).strip():
                continue
            t_acc += str(t) + " "
            c_acc += float(c) * 100.0
            words += 1
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 200, 255), _t(1, s))
        return t_acc.strip(), c_acc / max(words, 1)

    def _explain(e):
        # Map exception → (is_setup_issue, short human message).
        # Setup issues = missing module / missing binary → có lệnh cài cụ thể.
        if isinstance(e, ModuleNotFoundError):
            mod = str(e).split("'")[1] if "'" in str(e) else "module"
            return True, f"thiếu '{mod}' (pip install {mod})"
        name = type(e).__name__
        msg = str(e)
        if "TesseractNotFound" in name or "tesseract is not installed" in msg.lower():
            return True, "thiếu Tesseract binary trên PATH"
        return False, f"{name}: {msg}"

    try:
        if engine == "tesseract":
            text, conf = _run_tesseract()
            used = "tesseract"
        elif engine == "easyocr":
            text, conf = _run_easyocr()
            used = "easyocr"
        else:
            try:
                text, conf = _run_tesseract()
                used = "tesseract"
            except Exception as e1:
                try:
                    text, conf = _run_easyocr()
                    used = "easyocr"
                except Exception as e2:
                    t_setup, t_msg = _explain(e1)
                    e_setup, e_msg = _explain(e2)
                    if t_setup and e_setup:
                        # Cả 2 đều chưa cài → hướng dẫn cài nhanh nhất.
                        text = ("[OCR chưa cài engine nào] Chạy: "
                                "pip install easyocr  "
                                "(khuyên dùng — không cần binary, hỗ trợ tiếng Việt tốt). "
                                "Hoặc: pip install pytesseract + cài Tesseract binary "
                                "(Windows: UB-Mannheim installer; Ubuntu: apt install "
                                "tesseract-ocr tesseract-ocr-vie; macOS: brew install "
                                "tesseract tesseract-lang).")
                    else:
                        text = f"[OCR fail] tesseract: {t_msg} | easyocr: {e_msg}"
                    conf = 0.0
    except ModuleNotFoundError as e:
        mod = (str(e).split("'")[1] if "'" in str(e) else str(e))
        text = (f"[{mod} chưa cài] pip install {mod}"
                + ("  (+ Tesseract binary & langpack)" if mod == "pytesseract" else ""))
        conf = 0.0
    except Exception as e:
        _, msg = _explain(e)
        text = f"[OCR error] {msg}"
        conf = 0.0

    expected = params.get("expected_text", "")
    min_conf = params.get("min_confidence", 60.0)
    has_text = bool(text) and not text.startswith("[")
    is_pass = ((expected in text if expected else has_text)
               and conf >= min_conf)
    print(f"[OCR/{used or 'fail'}] text={text!r} conf={conf:.1f}% "
          f"{'PASS' if is_pass else 'FAIL'}")
    return {"image": vis, "text": text, "pass": is_pass, "confidence": conf}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: MEASUREMENT
# ═══════════════════════════════════════════════════════════════════

def proc_distance_point(inputs, params):
    """TDistancePointPointTool — Đo khoảng cách 2 điểm.

    Calibration modes:
      • "Scale": mm = px × pixel_to_mm (mặc định, back-compat).
      • "Two Points": calib tuyến tính 2 điểm —
        mm = (px - p1_px) × (p2_mm - p1_mm) / (p2_px - p1_px) + p1_mm.
        Dùng khi đã biết 2 cặp tham chiếu (vd: 150 px = 2.1 mm,
        210 px = 2.5 mm) → tự suy slope + offset.

    Label trên ảnh kéo được như Blob: anchor = trung điểm 2 đầu,
    `label_dx/label_dy` lưu offset từ anchor đến text (drag canvas
    sync ngược về params).
    """
    img=inputs.get("image")
    x1=float(inputs.get("x1",params.get("x1",0)))
    y1=float(inputs.get("y1",params.get("y1",0)))
    x2=float(inputs.get("x2",params.get("x2",100)))
    y2=float(inputs.get("y2",params.get("y2",0)))
    dist_px = math.hypot(x2-x1, y2-y1)
    calib_mode = str(params.get("calib_mode", "Scale"))
    if calib_mode == "Two Points":
        p1_px = float(params.get("calib_p1_px", 0.0))
        p1_mm = float(params.get("calib_p1_mm", 0.0))
        p2_px = float(params.get("calib_p2_px", 100.0))
        p2_mm = float(params.get("calib_p2_mm", 1.0))
        denom = p2_px - p1_px
        if abs(denom) < 1e-9:
            # Calib điểm trùng → fallback Scale để tránh chia 0
            dist = dist_px * params.get("pixel_to_mm", 1.0)
        else:
            slope = (p2_mm - p1_mm) / denom
            dist = (dist_px - p1_px) * slope + p1_mm
    else:
        dist = dist_px * params.get("pixel_to_mm", 1.0)
    is_pass = params.get("min_dist", 0.0) <= dist <= params.get("max_dist", 9999.0)

    vis=_bgr(img.copy()) if img is not None else np.zeros((200,400,3),dtype=np.uint8)
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", True))
    # Fail → đỏ cho cả line + 2 điểm + label, dễ thấy NG trên ảnh output.
    fail_color = (0, 0, 255)
    line_color = fail_color if not is_pass else (0, 220, 255)
    pt_color   = fail_color if not is_pass else (0, 200, 255)
    cv2.line(vis,(int(x1),int(y1)),(int(x2),int(y2)),line_color,_t(2,s))
    cv2.circle(vis,(int(x1),int(y1)),_t(5,s),pt_color,-1)
    cv2.circle(vis,(int(x2),int(y2)),_t(5,s),pt_color,-1)
    mx,my=int((x1+x2)/2),int((y1+y2)/2)
    label_rects = []
    if show_labels:
        _color_map    = {"Yellow":(0,200,255),"Cyan":(255,255,0),
                         "Green":(0,255,80),"Red":(0,0,255),
                         "White":(255,255,255),"Magenta":(255,0,255),
                         "Orange":(0,140,255),"Blue":(255,80,0)}
        label_color = (fail_color if not is_pass
                       else _color_map.get(params.get("label_color","Yellow"),
                                            (0,220,255)))
        label_font  = {"Simplex":cv2.FONT_HERSHEY_SIMPLEX,
                       "Plain":cv2.FONT_HERSHEY_PLAIN,
                       "Duplex":cv2.FONT_HERSHEY_DUPLEX,
                       "Complex":cv2.FONT_HERSHEY_COMPLEX,
                       "Triplex":cv2.FONT_HERSHEY_TRIPLEX,
                      }.get(params.get("label_font","Simplex"),cv2.FONT_HERSHEY_SIMPLEX)
        label_size  = float(params.get("label_size", 0.6))
        label_thick = int(params.get("label_thickness", 2))
        label_dx    = int(params.get("label_dx", 0))
        label_dy    = int(params.get("label_dy", -10))
        text = f"{dist:.3f}mm"
        tx = mx + int(label_dx * s)
        ty = my + int(label_dy * s)
        fs_val = _fs(label_size, s)
        th_val = _t(label_thick, s)
        cv2.putText(vis, text, (tx, ty), label_font, fs_val, label_color,
                    th_val, cv2.LINE_AA)
        (tw, th), bl = cv2.getTextSize(text, label_font, fs_val, th_val)
        label_rects.append((tx, ty - th - bl, tw, th + bl))
    print(f"[Distance] {dist_px:.2f}px → {dist:.3f}mm "
          f"({'PASS' if is_pass else 'FAIL'})")
    return {"image":vis,"distance":dist,"distance_px":dist_px,
            "x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
            "pass":is_pass,
            "_label_rects": label_rects,
            "_label_centroids": [(float(mx), float(my))] if label_rects else []}


def proc_distance_point_line(inputs, params):
    """TDistancePointLineTool — Khoảng cách (vuông góc) từ 1 điểm đến
    1 đường thẳng. Đường thẳng định nghĩa theo 1 trong 2 mode:
      - "Two Points":   (lx1, ly1) → (lx2, ly2)
      - "Point + Angle": qua (lx1, ly1) hợp với trục X góc `line_angle` (độ)
    Inputs port: px, py — điểm cần đo; lx1/ly1/lx2/ly2 hoặc line_angle.
    Output:  distance (mm theo pixel_to_mm), signed_distance (có dấu — âm
             nếu điểm bên trái đường nhìn từ p1→p2), foot_x/foot_y (toạ độ
             chân đường vuông góc), pass.
    """
    img = inputs.get("image")
    # Point
    px = float(inputs.get("px", params.get("px", 0)))
    py = float(inputs.get("py", params.get("py", 0)))
    # Line — ưu tiên port; fallback params
    mode = params.get("mode", "Two Points")
    lx1 = float(inputs.get("lx1", params.get("lx1", 0)))
    ly1 = float(inputs.get("ly1", params.get("ly1", 0)))
    if mode == "Point + Angle":
        ang = float(inputs.get("line_angle",
                                params.get("line_angle", 0.0)))
        rad = math.radians(ang)
        # Điểm thứ 2 ở khoảng cách lớn để vẽ "vô tận"
        lx2 = lx1 + math.cos(rad) * 1000.0
        ly2 = ly1 + math.sin(rad) * 1000.0
    else:
        lx2 = float(inputs.get("lx2", params.get("lx2", 100)))
        ly2 = float(inputs.get("ly2", params.get("ly2", 0)))

    # Vector along line
    vx = lx2 - lx1; vy = ly2 - ly1
    vlen = math.hypot(vx, vy)
    if vlen < 1e-6:
        # Degenerate line → fall back to point-to-point từ p1
        signed = math.hypot(px - lx1, py - ly1)
        dist_px = abs(signed)
        fx, fy = lx1, ly1
    else:
        # 2D cross product (vector từ p1 đến point) → signed distance
        wx = px - lx1; wy = py - ly1
        cross = vx * wy - vy * wx
        signed = cross / vlen
        dist_px = abs(signed)
        # Foot point: projection
        t = (wx * vx + wy * vy) / (vlen * vlen)
        fx = lx1 + t * vx
        fy = ly1 + t * vy

    px2mm = params.get("pixel_to_mm", 1.0)
    distance = dist_px * px2mm
    signed_mm = signed * px2mm

    min_d = params.get("min_dist", 0.0)
    max_d = params.get("max_dist", 9999.0)
    is_pass = min_d <= distance <= max_d

    if img is not None:
        vis = _bgr(img.copy())
    else:
        h_def = max(int(abs(py) + abs(ly2 - ly1) + 200), 200)
        w_def = max(int(abs(px) + abs(lx2 - lx1) + 200), 400)
        vis = np.zeros((h_def, w_def, 3), dtype=np.uint8)
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))

    # Vẽ đường line dài qua p1,p2 (extend cả 2 đầu để thấy rõ)
    if vlen >= 1e-6:
        ux = vx / vlen; uy = vy / vlen
        L = max(vis.shape[:2]) * 2
        e1 = (int(lx1 - ux * L), int(ly1 - uy * L))
        e2 = (int(lx2 + ux * L), int(ly2 + uy * L))
        cv2.line(vis, e1, e2, (255, 180, 0), _t(2, s), cv2.LINE_AA)
        # Endpoints của segment định nghĩa line
        cv2.circle(vis, (int(lx1), int(ly1)), _t(5, s), (255, 180, 0), -1)
        cv2.circle(vis, (int(lx2), int(ly2)), _t(5, s), (255, 180, 0), -1)
    # Foot of perpendicular
    cv2.circle(vis, (int(fx), int(fy)), _t(5, s), (0, 220, 255), -1)
    # Perpendicular segment from point to foot
    cv2.line(vis, (int(px), int(py)), (int(fx), int(fy)),
              (0, 220, 255), _t(2, s), cv2.LINE_AA)
    # Point marker
    cv2.circle(vis, (int(px), int(py)), _t(6, s), (0, 100, 255), -1)
    cv2.circle(vis, (int(px), int(py)), _t(6, s), (255, 255, 255), _t(1, s))

    if show_labels:
        midx = int((px + fx) / 2); midy = int((py + fy) / 2)
        cv2.putText(vis, f"{distance:.3f}mm",
                     (midx + int(8*s), midy - int(8*s)),
                     cv2.FONT_HERSHEY_SIMPLEX, _fs(0.6, s),
                     (0, 220, 255), _t(2, s))
    print(f"[DistPL] d={distance:.3f}mm (signed={signed_mm:+.3f}) "
          f"foot=({fx:.1f},{fy:.1f}) {'PASS' if is_pass else 'FAIL'}")
    return {"image": vis, "distance": distance,
            "signed_distance": signed_mm,
            "foot_x": float(fx), "foot_y": float(fy),
            "pass": is_pass}

def proc_angle_lines(inputs, params):
    """TAngleLineLineTool — Đo góc giữa 2 đường thẳng."""
    img=inputs.get("image")
    a1=float(inputs.get("angle1",params.get("line1_angle",0.0)))
    a2=float(inputs.get("angle2",params.get("line2_angle",45.0)))
    diff=abs(a1-a2)%180
    if diff>90: diff=180-diff
    vis=_bgr(img.copy()) if img is not None else np.zeros((200,400,3),dtype=np.uint8)
    s = _draw_scale(vis)
    h2,w2=vis.shape[:2]
    cx,cy=w2//2,h2//2
    for ang,col in [(a1,(0,220,255)),(a2,(255,140,0))]:
        rad=math.radians(ang); l=min(w2,h2)//3
        cv2.line(vis,(cx-int(math.cos(rad)*l),cy-int(math.sin(rad)*l)),
                      (cx+int(math.cos(rad)*l),cy+int(math.sin(rad)*l)),col,_t(2,s))
    is_pass=params.get("min_angle",0.0)<=diff<=params.get("max_angle",90.0)
    print(f"[AngleLines] angle={diff:.2f}deg {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"angle":diff,"pass":is_pass}

def proc_area(inputs, params):
    """Measure area from blob/contours."""
    img=inputs.get("image"); mask=inputs.get("mask"); contours=inputs.get("contours",[])
    if mask is None and img is not None: mask=_gray(img)
    vis=_bgr(img.copy() if img is not None else np.zeros((100,100,3),dtype=np.uint8))
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))
    scale=params.get("pixel_to_mm2",1.0); areas=[]
    if contours:
        for c in contours:
            a=cv2.contourArea(c); a_mm=a*scale; areas.append(a_mm)
            M2=cv2.moments(c)
            if M2["m00"]>0:
                ccx=int(M2["m10"]/M2["m00"]); ccy=int(M2["m01"]/M2["m00"])
                cv2.drawContours(vis,[c],-1,(0,200,255),_t(2,s))
                if show_labels:
                    cv2.putText(vis,f"{a_mm:.2f}",(ccx,ccy),cv2.FONT_HERSHEY_SIMPLEX,_fs(0.5,s),(0,200,255),_t(1,s))
    elif mask is not None:
        cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a=cv2.contourArea(c); a_mm=a*scale; areas.append(a_mm)
            M2=cv2.moments(c)
            if M2["m00"]>0:
                ccx=int(M2["m10"]/M2["m00"]); ccy=int(M2["m01"]/M2["m00"])
                if show_labels:
                    cv2.putText(vis,f"{a_mm:.1f}",(ccx,ccy),cv2.FONT_HERSHEY_SIMPLEX,_fs(0.5,s),(0,200,255),_t(1,s))
    total=sum(areas)
    is_pass=params.get("min_area",0.0)<=total<=params.get("max_area",1e9)
    print(f"[Area] total={total:.3f}mm² count={len(areas)} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"total_area":total,"count":len(areas),"pass":is_pass}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════

def proc_image_convert(inputs, params):
    """TImageConvertTool — Chuyển đổi định dạng ảnh.
    Grayscale mode trả ảnh 1-channel (downstream _gray/_bgr xử lý được);
    bỏ double-convert BGR→GRAY→BGR thừa của bản cũ.
    """
    img = inputs.get("image")
    if img is None:
        return {"image": None}
    mode = params.get("mode", "Grayscale")
    if mode == "Grayscale":
        # Single channel — Image Viewer dùng Format_Grayscale8 trực tiếp
        out = _gray(img) if len(img.shape) == 3 else img
    else:
        bgr = _bgr(img)
        if   mode == "BGR to RGB": out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif mode == "Invert":     out = cv2.bitwise_not(bgr)
        elif mode == "HSV":        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        elif mode == "LAB":        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        elif mode == "YCrCb":      out = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        else:                       out = bgr
    return {"image": out}


def proc_sharpen(inputs, params):
    """Sharpen 3×3 kernel. filter2D với ddepth=-1 đã clip uint8 → bỏ
    np.clip + astype thừa (save 1 pass qua ảnh)."""
    img = inputs.get("image")
    if img is None:
        return {"image": None}
    s = params.get("strength", 1.0)
    k = np.array([[0, -1, 0], [-1, 4+s, -1], [0, -1, 0]], dtype=np.float32)
    return {"image": cv2.filter2D(img, -1, k)}

def proc_morphology(inputs, params):
    img=inputs.get("image")
    if img is None: return {"image":None,"mask":None}
    g=_gray(img)

    # Tiền xử lý tuỳ chọn
    if params.get("invert_input", False):
        g = cv2.bitwise_not(g)
    if params.get("auto_binarize", False):
        _, g = cv2.threshold(g, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    op_name = params.get("operation", "Open")
    op_map = {"Erode":cv2.MORPH_ERODE,"Dilate":cv2.MORPH_DILATE,
              "Open":cv2.MORPH_OPEN,"Close":cv2.MORPH_CLOSE,
              "Gradient":cv2.MORPH_GRADIENT,
              "Top Hat":cv2.MORPH_TOPHAT,"Black Hat":cv2.MORPH_BLACKHAT}
    sh={"Rect":cv2.MORPH_RECT,"Ellipse":cv2.MORPH_ELLIPSE,
        "Cross":cv2.MORPH_CROSS}.get(params.get("shape","Ellipse"),
                                     cv2.MORPH_ELLIPSE)
    k=max(1,int(params.get("kernel_size",3)))
    iters=max(1,int(params.get("iterations",1)))
    kernel=cv2.getStructuringElement(sh,(k,k))

    # Combo ops: hai phép liên tiếp với cùng kernel/iter
    if op_name == "Open+Close":
        out = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=iters)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=iters)
    elif op_name == "Close+Open":
        out = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=iters)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=iters)
    else:
        op = op_map.get(op_name, cv2.MORPH_OPEN)
        out = cv2.morphologyEx(g, op, kernel, iterations=iters)

    result = {"image": _bgr(out), "mask": out}

    # Overlay: nguồn gốc + vùng kết quả tô xanh (so sánh trực quan)
    if params.get("show_overlay", False):
        base = _bgr(img if img.ndim == 3 else _bgr(_gray(img)))
        if base.shape[:2] != out.shape[:2]:
            base = cv2.resize(base, (out.shape[1], out.shape[0]))
        overlay = base.copy()
        overlay[out > 0] = [0, 220, 80]
        vis = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
        result["_display_image"] = vis

    return result

def proc_threshold(inputs, params):
    img=inputs.get("image")
    if img is None: return {"image":None,"mask":None}
    g=_gray(img)
    METHOD={"Binary":cv2.THRESH_BINARY,"Binary INV":cv2.THRESH_BINARY_INV,
            "Trunc":cv2.THRESH_TRUNC,"To Zero":cv2.THRESH_TOZERO,
            "To Zero INV":cv2.THRESH_TOZERO_INV,
            "Otsu":cv2.THRESH_BINARY|cv2.THRESH_OTSU,
            "Otsu INV":cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU,
            "Triangle":cv2.THRESH_BINARY|cv2.THRESH_TRIANGLE,
            "Triangle INV":cv2.THRESH_BINARY_INV|cv2.THRESH_TRIANGLE}
    t=METHOD.get(params.get("method","Otsu"),cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    _,mask=cv2.threshold(g,params.get("threshold",127),params.get("max_value",255),t)
    return {"image":_bgr(mask),"mask":mask}

def proc_gaussian_blur(inputs, params):
    img=inputs.get("image")
    if img is None: return {"image":None}
    k=max(1,params.get("kernel_size",5)); k=k+1 if k%2==0 else k
    return {"image":cv2.GaussianBlur(img,(k,k),params.get("sigma",0.0))}

def proc_crop(inputs, params):
    """
    Crop ROI:
    - Mỗi port x/y/w/h: nếu CÓ giá trị từ upstream → ưu tiên dùng;
      port nào KHÔNG kết nối thì fall back về params / _drawn_roi.
    - Cho phép half-connect: ví dụ PatMax chỉ cung cấp x/y (không w/h) →
      x/y track theo PatMax, w/h lấy từ Width/Height params.

    Output:
      - `image`     : ẢNH GỐC + bounding box overlay vẽ ở vùng ROI
                       (để node panel & downstream display thấy được vị trí
                        ROI trong ảnh gốc).
      - `roi_image` : ảnh đã cắt (cho downstream tool xử lý vùng ROI).
    """
    img = inputs.get("image")
    if img is None:
        return {"image": None, "_display_image": None, "roi_image": None,
                "x": 0, "y": 0, "w": 0, "h": 0}

    ih, iw = img.shape[:2]

    port_x = inputs.get("x")
    port_y = inputs.get("y")
    port_w = inputs.get("w")
    port_h = inputs.get("h")

    # Default từ _drawn_roi (vẽ tay) hoặc params spinbox.
    # Khi port kết nối → port luôn thắng (auto-tracking ưu tiên). _drawn_roi
    # chỉ dùng khi port port tương ứng không có giá trị.
    drawn = params.get("_drawn_roi")
    if drawn and isinstance(drawn, (list, tuple)) and len(drawn) == 4:
        dx, dy, dw, dh = [int(v) for v in drawn]
        manual_src = "MANUAL ROI"
    else:
        if not params.get("_crop_initialized"):
            params["x"] = 0
            params["y"] = 0
            params["crop_w"] = iw
            params["crop_h"] = ih
            params["_crop_initialized"] = True
        dx = int(params.get("x", 0))
        dy = int(params.get("y", 0))
        dw = int(params.get("crop_w", iw))
        dh = int(params.get("crop_h", ih))
        manual_src = "PARAMS"

    # Per-port override: port > drawn > params.
    x  = int(float(port_x)) if port_x is not None else dx
    y  = int(float(port_y)) if port_y is not None else dy
    cw = int(float(port_w)) if port_w is not None else dw
    ch = int(float(port_h)) if port_h is not None else dh

    tracked_ports = [n for n, v in [("x", port_x), ("y", port_y),
                                     ("w", port_w), ("h", port_h)]
                     if v is not None]
    if not tracked_ports:
        mode_label = manual_src
    elif len(tracked_ports) == 4:
        mode_label = "TRACKED"
    else:
        mode_label = "TRACKED " + "".join(tracked_ports)

    # Clamp
    x  = max(0, min(x,  iw - 1))
    y  = max(0, min(y,  ih - 1))
    cw = max(1, min(cw, iw - x))
    ch = max(1, min(ch, ih - y))

    # Persist giá trị thực tế (đã clamp) ngược vào params khi không TRACKED.
    if not tracked_ports:
        params["x"] = x
        params["y"] = y
        params["crop_w"] = cw
        params["crop_h"] = ch

    # Crop ảnh thực sự (clean — không có overlay) — cho downstream qua `roi_image`
    roi = img[y:y + ch, x:x + cw].copy()

    # `image` output = ảnh GỐC pass-through (không copy thừa, chỉ ensure BGR
    # nếu cần). `_display_image` = COPY + bounding box overlay cho node panel.
    # Trước đây làm 2 copy (clean.copy() + disp.copy()) cho ảnh 20MP =
    # ~60MB thừa; giờ chỉ 1 copy cho display.
    clean = _bgr(img)              # no-copy nếu img đã BGR
    disp = clean.copy()
    s = _draw_scale(disp)
    col = (0, 255, 180) if tracked_ports else (0, 212, 255)
    cv2.rectangle(disp, (x, y), (x + cw, y + ch), col, _t(2, s))

    print(f"[Crop] {mode_label} ({x},{y}) {cw}x{ch}")

    return {"image": clean, "_display_image": disp, "roi_image": roi,
            "x": x, "y": y, "w": cw, "h": ch}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: SURFACE INSPECTION
# ═══════════════════════════════════════════════════════════════════

def proc_surface_defect(inputs, params):
    """Phát hiện khuyết tật bề mặt — so sánh với reference hoặc model thống kê.

    Auto downscale: blur + diff + threshold + morph + findContours chạy
    trên ảnh ~1.5MP. Diện tích/coord scale ngược về full-res.
    20MP: ~45ms → ~6ms.
    """
    img = inputs.get("image"); ref = inputs.get("reference")
    if img is None:
        return {"image": None, "pass": False,
                "defect_area": 0, "defect_count": 0}
    bgr = _bgr(img)
    gray_full = _gray(bgr)
    gray, ds = _auto_downscale(gray_full, params.get("downscale", 0))

    if ref is not None:
        ref_full = _gray(_bgr(ref))
        # Align ref dimensions với gray (downscaled). Nếu shape khớp full-res,
        # downscale tương ứng; ngược lại fallback self-diff.
        if ref_full.shape == gray_full.shape:
            ref_small, _ = _auto_downscale(ref_full, ds)
            diff = cv2.absdiff(gray, ref_small)
        else:
            diff = cv2.absdiff(gray, cv2.GaussianBlur(gray, (21, 21), 0))
    else:
        diff = cv2.absdiff(gray, cv2.GaussianBlur(gray, (21, 21), 0))

    _, mask = cv2.threshold(diff, params.get("threshold", 30),
                              255, cv2.THRESH_BINARY)
    k = params.get("morph_k", 3)
    # Kernel scale theo ds để giữ semantics (k pixel ở full-res ~ k/ds ở ds)
    kk = max(1, int(round(k / max(1, ds))))
    if kk > 0:
        ker = np.ones((kk, kk), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
        mask = cv2.dilate(mask, ker)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    min_a = params.get("min_defect_px", 10)
    ds_sq = ds * ds
    cnts = [c for c in cnts if cv2.contourArea(c) * ds_sq >= min_a]
    defect_area = int(sum(cv2.contourArea(c) * ds_sq for c in cnts))
    max_a = params.get("max_defect_area", 1000)
    max_c = params.get("max_defect_count", 0)
    is_pass = (defect_area <= max_a
                 and len(cnts) <= (max_c if max_c > 0 else len(cnts) + 1))
    vis = bgr.copy()
    s = _draw_scale(vis)
    for c in cnts:
        c_full = (c * ds).astype(np.int32) if ds > 1 else c
        cv2.drawContours(vis, [c_full], -1, (0, 60, 255), _t(2, s))
        x2, y2, w2, h2 = cv2.boundingRect(c_full)
        cv2.rectangle(vis, (x2, y2), (x2 + w2, y2 + h2),
                       (0, 60, 255), _t(1, s))
    print(f"[SurfaceDefect] area={defect_area}px² count={len(cnts)} "
          f"{'PASS' if is_pass else 'FAIL'} ds={ds}")
    return {"image": vis, "pass": is_pass,
            "defect_area": defect_area, "defect_count": len(cnts)}

def proc_scratch_detect(inputs, params):
    """Phát hiện vết xước dạng đường thẳng dài."""
    img = inputs.get("image")
    if img is None:
        return {"image": None, "pass": False,
                "scratch_count": 0, "total_length": 0.0}
    gray_full = _gray(img)
    vis = _bgr(img.copy())
    s = _draw_scale(vis)
    # Auto downscale: blur + Canny + HoughLinesP trên ảnh nhỏ → length, coord
    # scale ngược ×ds. 20MP: ~30ms → ~5ms.
    gray, ds = _auto_downscale(gray_full, params.get("downscale", 0))
    k = params.get("blur_k", 3)
    if k > 0:
        kk = max(1, k | 1)  # odd kernel size
        gray = cv2.GaussianBlur(gray, (kk, kk), 0)
    edges = cv2.Canny(gray, params.get("canny_low", 30),
                       params.get("canny_high", 100))
    min_len = params.get("min_scratch_length", 50)
    max_gap = params.get("max_gap", 5)
    # min_len/max_gap user-set ở full-res → scale xuống ds space
    min_len_ds = max(2, int(round(min_len / max(1, ds))))
    max_gap_ds = max(1, int(round(max_gap / max(1, ds))))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, params.get("hough_thresh", 30),
        minLineLength=min_len_ds, maxLineGap=max_gap_ds)
    scratches = []; total_len = 0.0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Scale back to full-res
            x1f = x1 * ds; y1f = y1 * ds
            x2f = x2 * ds; y2f = y2 * ds
            length = math.hypot(x2f - x1f, y2f - y1f)
            if length >= min_len:
                scratches.append((x1f, y1f, x2f, y2f, length))
                total_len += length
                cv2.line(vis, (int(x1f), int(y1f)), (int(x2f), int(y2f)),
                          (0, 60, 255), _t(2, s))
    max_s = params.get("max_scratches", 0)
    is_pass = len(scratches) <= max_s
    print(f"[Scratch] count={len(scratches)} total_len={total_len:.0f}px "
          f"{'PASS' if is_pass else 'FAIL'} ds={ds}")
    return {"image": vis, "pass": is_pass,
            "scratch_count": len(scratches), "total_length": total_len}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: GEOMETRY / CONTOUR
# ═══════════════════════════════════════════════════════════════════

def proc_find_contours(inputs, params):
    img=inputs.get("image"); mask=inputs.get("mask")
    if mask is None and img is not None: mask=_gray(img)
    if mask is None: return {"image":None,"contours":[],"count":0}
    RETR={"External":cv2.RETR_EXTERNAL,"List":cv2.RETR_LIST,
          "Tree":cv2.RETR_TREE,"CCOMP":cv2.RETR_CCOMP}
    contours,_=cv2.findContours(mask,RETR.get(params.get("retrieval","External"),cv2.RETR_EXTERNAL),
                                  cv2.CHAIN_APPROX_SIMPLE)
    mn=params.get("min_area",10.0); mx=params.get("max_area",1e6)
    contours=[c for c in contours if mn<=cv2.contourArea(c)<=mx]
    vis=_bgr(img.copy() if img is not None else mask)
    s = _draw_scale(vis)
    cv2.drawContours(vis,contours,-1,(0,255,100),_t(2,s))
    min_c=params.get("min_count",1); max_c=params.get("max_count",1000)
    is_pass=min_c<=len(contours)<=max_c
    print(f"[FindContours] count={len(contours)} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"contours":contours,"count":len(contours),"pass":is_pass}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: CALIBRATION
# ═══════════════════════════════════════════════════════════════════

def proc_calibrate_grid(inputs, params):
    """TCalibCheckerboardTool — Hiệu chỉnh camera từ checkerboard.

    Coarse-then-refine: findChessboardCorners chạy trên ảnh downscaled
    (~1.5MP) với FAST_CHECK flag, scale corners ngược về full-res rồi
    cornerSubPix refine tại độ chính xác sub-pixel → vẫn precise nhưng
    nhanh hơn 5-10×. 20MP: ~50ms → ~10ms.
    """
    img = inputs.get("image")
    if img is None:
        return {"image": None, "calibrated": False,
                "pixel_to_mm": 1.0, "rms_error": 0.0}
    gray = _gray(img)
    vis = _bgr(img.copy())
    cols = params.get("grid_cols", 9)
    rows = params.get("grid_rows", 6)

    small, ds = _auto_downscale(gray, params.get("downscale", 0))
    # FAST_CHECK quick-reject khi không có pattern; NORMALIZE_IMAGE +
    # ADAPTIVE_THRESH tăng robustness với lighting không đều.
    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FAST_CHECK)
    ret, corners_ds = cv2.findChessboardCorners(small, (cols, rows), cb_flags)
    if not ret:
        print(f"[Calibrate] Checkerboard NOT found (ds={ds})")
        return {"image": vis, "calibrated": False,
                "pixel_to_mm": 1.0, "rms_error": 0.0}

    # Scale corners về full-res rồi cornerSubPix refine (giữ độ chính xác)
    corners = corners_ds * float(ds)
    win = max(5, int(11 * ds))   # window size scale theo ds
    win = win | 1                 # ensure odd
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
    cv2.drawChessboardCorners(vis, (cols, rows), corners, ret)
    # Estimate pixel/mm from square size
    if len(corners) >= 2:
        p1 = corners[0][0]; p2 = corners[1][0]
        px_per_square = float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        mm_per_square = params.get("square_size_mm", 25.4)
        px_to_mm = mm_per_square / max(px_per_square, 0.001)
    else:
        px_to_mm = 1.0
    print(f"[Calibrate] {px_to_mm:.5f} mm/px (ds={ds})")
    return {"image": vis, "calibrated": True,
            "pixel_to_mm": px_to_mm, "rms_error": 0.0}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: LOGIC & FLOW
# ═══════════════════════════════════════════════════════════════════

def proc_logic_and(inputs,params):
    return {"result":bool(inputs.get("A",False)) and bool(inputs.get("B",False))}
def proc_logic_or(inputs,params):
    return {"result":bool(inputs.get("A",False)) or bool(inputs.get("B",False))}
def proc_logic_not(inputs,params):
    return {"result":not bool(inputs.get("A",False))}
def proc_compare(inputs,params):
    a=float(inputs.get("A",0)); b=float(inputs.get("B",params.get("value",0)))
    op=params.get("operator","==")
    result={"==":a==b,"!=":a!=b,">":a>b,">=":a>=b,"<":a<b,"<=":a<=b}.get(op,False)
    return {"result":result,"pass":result}

def proc_judge(inputs,params):
    # Iterate mọi input bool — port mặc định A, B + extra ports user thêm
    # qua right-click "Add Input". Port chưa nối (None) bỏ qua.
    conds=[bool(v) for v in inputs.values() if v is not None]
    result=(all(conds) if params.get("mode","ALL")=="ALL" else any(conds)) if conds else False
    return {"result":result,"pass":result}

def proc_script(inputs, params):
    """Script Tool — chạy Python code tùy biến.

    Context cho code:
      • a, b, c, ...      = inputs['A'], inputs['B'], ... (lowercase)
      • inputs            = dict đầy đủ tất cả input port (uppercase keys)
      • params            = dict tham số node
      • outputs           = dict — gán outputs['name'] = value để set port output
      • pass_value        = bool — alias cho output 'pass' (vì `pass` là Python keyword)
      • Hoặc gán biến top-level cùng tên port (vd `result = 100` → port `result`)

    Output ports: declared trong tool.outputs + `_extra_outputs` param.
    Pipeline status (pass/fail) dùng port `pass`.
    """
    code = params.get("expression", "")
    ctx = {"inputs": inputs, "params": params, "outputs": {}}
    # Bind inputs vừa uppercase (giữ nguyên key) vừa lowercase (tiện viết code)
    for k, v in inputs.items():
        ctx[k] = v
        if isinstance(k, str) and k != k.lower():
            ctx[k.lower()] = v
    try:
        exec(code, ctx)
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}"
        print(f"[Script] ERROR: {err}\n{traceback.format_exc(limit=3)}")
        ctx["__error"] = err

    # Build output dict — đọc theo tên port đã declared
    out: Dict[str, Any] = {}
    extras = list(params.get("_extra_outputs") or [])
    user_outputs = ctx.get("outputs") if isinstance(ctx.get("outputs"), dict) else {}

    def _read(name):
        # Ưu tiên outputs[name] → ctx[name] → None
        if name in user_outputs:
            return user_outputs[name]
        return ctx.get(name)

    # Port 'pass' đặc biệt: ưu tiên outputs['pass'] → pass_value → result
    pass_val = user_outputs.get("pass", ctx.get("pass_value"))
    if pass_val is None:
        pass_val = ctx.get("result")
    out["pass"] = bool(pass_val) if pass_val is not None else False

    # Port 'result' (static)
    out["result"] = _read("result")

    # Extra output ports
    for name in extras:
        if name in ("pass", "result"):
            continue
        out[name] = _read(name)

    if ctx.get("__error"):
        out["__error"] = ctx["__error"]
    return out


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: OUTPUT / DISPLAY
# ═══════════════════════════════════════════════════════════════════

def proc_display(inputs,params):
    """TRecordDisplayTool — Annotate & display image."""
    img=inputs.get("image")
    if img is None: return {"image":None}
    vis=_bgr(img.copy())
    s = _draw_scale(vis)
    text=params.get("label",""); show_pass=params.get("show_result",True)
    passed=inputs.get("pass",None)
    if text:
        cv2.putText(vis,text,(params.get("tx",10),params.get("ty",int(30*s))),
                    cv2.FONT_HERSHEY_SIMPLEX,_fs(params.get("font_scale",0.8),s),(0,212,255),_t(2,s))
    if show_pass and passed is not None:
        col=(0,220,80) if passed else (0,60,255)
        label="PASS" if passed else "FAIL"
        box_w=int(200*s); box_h=int(45*s); pad=int(5*s)
        cv2.rectangle(vis,(pad,pad),(pad+box_w,pad+box_h),(0,0,0),-1)
        cv2.putText(vis,label,(int(10*s),int(35*s)),cv2.FONT_HERSHEY_SIMPLEX,_fs(1.0,s),col,_t(2,s))
    return {"image":vis}

def proc_message(inputs, params):
    """Message — hiển thị text khác nhau dựa trên port pass (bool).
    Không cần image input: tự tạo canvas, hoặc vẽ overlay lên image
    nếu được nối port image."""
    passed = inputs.get("pass", None)
    img = inputs.get("image")

    msg_pass = params.get("msg_pass", "PASS")
    msg_fail = params.get("msg_fail", "FAIL")
    msg_none = params.get("msg_none", "NO INPUT")

    if passed is True:
        text = msg_pass; col_name = params.get("color_pass", "Green")
    elif passed is False:
        text = msg_fail; col_name = params.get("color_fail", "Red")
    else:
        text = msg_none; col_name = params.get("color_none", "Yellow")

    color = {"Yellow":(0,200,255),"Cyan":(255,255,0),
             "Green":(0,220,80),"Red":(0,60,255),
             "White":(255,255,255),"Magenta":(255,0,255),
             "Orange":(0,140,255),"Blue":(255,80,0)
            }.get(col_name, (0,200,255))
    font_map = {"Simplex":cv2.FONT_HERSHEY_SIMPLEX,
                "Plain":cv2.FONT_HERSHEY_PLAIN,
                "Duplex":cv2.FONT_HERSHEY_DUPLEX,
                "Complex":cv2.FONT_HERSHEY_COMPLEX,
                "Triplex":cv2.FONT_HERSHEY_TRIPLEX,
                "Script Simplex":cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                "Script Complex":cv2.FONT_HERSHEY_SCRIPT_COMPLEX}
    font = font_map.get(params.get("font", "Duplex"), cv2.FONT_HERSHEY_DUPLEX)

    # Canvas: dùng image nếu có, không thì tạo canvas đen 640x180
    if img is not None and isinstance(img, np.ndarray):
        vis = _bgr(img.copy())
    else:
        vis = np.zeros((180, 640, 3), dtype=np.uint8)
    s = _draw_scale(vis)
    H, W = vis.shape[:2]

    font_size = float(params.get("font_size", 1.2))
    thickness = int(params.get("thickness", 3))
    fs_val = _fs(font_size, s); th_val = _t(thickness, s)
    (tw, th), bl = cv2.getTextSize(text, font, fs_val, th_val)

    position = params.get("position", "Top-Left")
    pad = int(16 * s)
    if position == "Top-Left":
        ax, ay = pad, pad + th
    elif position == "Top-Right":
        ax, ay = W - tw - pad, pad + th
    elif position == "Top-Center":
        ax, ay = (W - tw) // 2, pad + th
    elif position == "Bottom-Left":
        ax, ay = pad, H - pad - bl
    elif position == "Bottom-Right":
        ax, ay = W - tw - pad, H - pad - bl
    elif position == "Bottom-Center":
        ax, ay = (W - tw) // 2, H - pad - bl
    else:  # Center
        ax, ay = (W - tw) // 2, (H + th) // 2

    # Offset có thể chỉnh bằng slider HOẶC kéo text trên canvas
    # (sync về label_dx/label_dy qua _on_label_dragged).
    dx = int(params.get("label_dx", 0))
    dy = int(params.get("label_dy", 0))
    tx = ax + int(dx * s); ty = ay + int(dy * s)

    if params.get("show_background", True):
        bg_pad = int(8 * s)
        cv2.rectangle(vis,
                      (tx - bg_pad, ty - th - bg_pad),
                      (tx + tw + bg_pad, ty + bl + bg_pad),
                      (0, 0, 0), -1)
    cv2.putText(vis, text, (tx, ty), font, fs_val, color, th_val, cv2.LINE_AA)
    print(f"[Message] pass={passed} text='{text}'")

    # Expose label rect + anchor để dialog hit-test drag — đồng nhất với Blob.
    label_rects = [(tx, ty - th - bl, tw, th + bl)]
    label_anchors = [(float(ax), float(ay))]
    return {"image": vis, "text": text, "pass": passed,
            "_label_rects": label_rects,
            "_label_centroids": label_anchors}


def proc_save_image(inputs,params):
    """TSaveImageTool — Lưu ảnh ra file."""
    img=inputs.get("image")
    if img is None: return {"saved":False,"path":""}
    import os,time
    path=params.get("save_path","output/result.png")
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    if params.get("timestamp",True):
        b,e=os.path.splitext(path); path=f"{b}_{int(time.time())}{e or '.png'}"
    ok=cv2.imwrite(path,img)
    return {"saved":ok,"path":path}

def proc_csv_log(inputs,params):
    """Ghi kết quả vào CSV log."""
    import os,csv,time
    values={k:v for k,v in inputs.items() if isinstance(v,(int,float,bool,str))}
    values["timestamp"]=time.strftime("%Y-%m-%d %H:%M:%S")
    path=params.get("csv_path","log/results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    write_header=not os.path.exists(path)
    try:
        with open(path,"a",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(values.keys()))
            if write_header: w.writeheader()
            w.writerow(values)
        return {"logged":True,"path":path}
    except Exception as e:
        return {"logged":False,"path":path}


# ═══════════════════════════════════════════════════════════════════
#  YIELD STATISTICS — production accumulator (pass/fail/total/yield%)
# ═══════════════════════════════════════════════════════════════════
# State sống ngoài node để counter cộng dồn qua nhiều run pipeline.
# UI panel (ResultsPanel) đọc trực tiếp output node để hiển thị.
_YIELD_STATE: Dict[str, Any] = {
    "pass": 0, "fail": 0, "total": 0, "last_pass": None,
}


def yield_stats_reset() -> None:
    """Reset counter về 0 — UI gọi khi nhấn Clear Log."""
    _YIELD_STATE["pass"] = 0
    _YIELD_STATE["fail"] = 0
    _YIELD_STATE["total"] = 0
    _YIELD_STATE["last_pass"] = None


def yield_stats_snapshot() -> Dict[str, Any]:
    total = _YIELD_STATE["total"]
    yld = (_YIELD_STATE["pass"] / total * 100.0) if total > 0 else 0.0
    return {
        "pass_count":  _YIELD_STATE["pass"],
        "fail_count":  _YIELD_STATE["fail"],
        "total_count": total,
        "yield_rate":  float(yld),
        "last_pass":   _YIELD_STATE["last_pass"],
    }


def proc_yield_stats(inputs, params):
    """Yield Statistics — accumulator pass/fail/total/yield% qua nhiều run.

    Mỗi lần pipeline chạy, port `pass` (bool) từ upstream (vd Pass/Fail Judge)
    được đọc → counter cộng dồn:
      pass=True  → pass_count += 1, total_count += 1
      pass=False → fail_count += 1, total_count += 1
      pass=None  → giữ nguyên (port chưa nối)
    Tick `reset` rồi Run để xóa counter ngay lần chạy đó.
    """
    if params.get("reset", False):
        yield_stats_reset()

    passed = inputs.get("pass", None)
    if passed is not None:
        _YIELD_STATE["total"] += 1
        if passed:
            _YIELD_STATE["pass"] += 1
        else:
            _YIELD_STATE["fail"] += 1
        _YIELD_STATE["last_pass"] = bool(passed)

    snap = yield_stats_snapshot()
    print(f"[YieldStats] pass={snap['pass_count']} fail={snap['fail_count']} "
          f"total={snap['total_count']} yield={snap['yield_rate']:.1f}%")
    return {
        "pass":        bool(snap["last_pass"]) if snap["last_pass"] is not None else False,
        "pass_count":  snap["pass_count"],
        "fail_count":  snap["fail_count"],
        "total_count": snap["total_count"],
        "yield_rate":  snap["yield_rate"],
    }


# ═══════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════
P = ParamDef  # shorthand


# ═══════════════════════════════════════════════════════════════
#  YOLO DETECTION
# ═══════════════════════════════════════════════════════════════

# Cache: {onnx_path: (session_or_net, names, backend, mtime)}
# backend: "onnxruntime" | "cv2dnn"
_YOLO_ONNX_CACHE: Dict[str, Tuple[Any, List[str], str, float]] = {}


def yolo_inspect_model(path: str) -> Dict[str, Any]:
    """Đọc metadata 1 file .pt hoặc .onnx YOLO mà KHÔNG chạy inference.

    Trả về dict: {task, names, imgsz, format, size_mb, ok, error?}. UI
    panel YOLO Detect dùng để hiện thông số khi user add file.
    """
    info: Dict[str, Any] = {"ok": False, "path": path}
    if not path or not os.path.exists(path):
        info["error"] = "file not found"
        return info
    try:
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)
        ext = os.path.splitext(path)[1].lower()
        info["format"] = ext.lstrip(".")
        if ext == ".onnx":
            names = _yolo_read_classes(path)
            info["names"] = names
            info["num_classes"] = len(names)
            # Đọc input shape từ ONNX
            try:
                import onnx
                m = onnx.load(path, load_external_data=False)
                shape = m.graph.input[0].type.tensor_type.shape
                dims = [d.dim_value or d.dim_param or "?"
                        for d in shape.dim]
                info["input_shape"] = dims
                if len(dims) >= 3 and isinstance(dims[-1], int):
                    info["imgsz"] = dims[-1]
                # Task type từ metadata (ultralytics)
                meta = {p.key: p.value for p in m.metadata_props}
                if "task" in meta:
                    info["task"] = meta["task"]
                if "stride" in meta:
                    info["stride"] = meta["stride"]
            except Exception:
                pass
            info["ok"] = True
        elif ext == ".pt":
            from ultralytics import YOLO as _YOLO
            model = _YOLO(path)
            raw_names = getattr(model, "names", None) or {}
            if isinstance(raw_names, dict):
                names = [raw_names[k] for k in sorted(
                    raw_names.keys(), key=lambda x: int(x))]
            else:
                names = list(raw_names)
            info["names"] = names
            info["num_classes"] = len(names)
            info["task"] = getattr(model, "task", "detect")
            # Imgsz từ checkpoint args (ultralytics key đổi qua versions:
            # newer dùng 'train_args', older dùng 'args').
            try:
                ck = getattr(model, "ckpt", None) or {}
                for k in ("train_args", "args"):
                    a = ck.get(k) if isinstance(ck, dict) else None
                    if isinstance(a, dict) and "imgsz" in a:
                        info["imgsz"] = int(a["imgsz"]); break
            except Exception:
                pass
            info["ok"] = True
        else:
            info["error"] = f"unsupported extension '{ext}'"
        # Onnx sibling cho .pt → báo cho user biết sẽ chạy fast path
        if ext == ".pt":
            sib = os.path.splitext(path)[0] + ".onnx"
            if os.path.exists(sib):
                info["onnx_sibling"] = sib
    except ModuleNotFoundError as e:
        info["error"] = f"module thiếu: {e}"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _yolo_letterbox(img: np.ndarray, new_size: int = 640,
                    color: Tuple[int, int, int] = (114, 114, 114)
                    ) -> Tuple[np.ndarray, float, int, int]:
    """Resize giữ aspect-ratio + pad sang square new_size×new_size.
    Trả về (canvas, scale, pad_x, pad_y) — scale = ratio để un-letterbox
    box về ảnh gốc; pad_x/pad_y là số pixel pad mỗi bên (top-left)."""
    h, w = img.shape[:2]
    r = min(new_size / w, new_size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def _yolo_read_classes(onnx_path: str) -> List[str]:
    """Đọc class names từ ONNX metadata (ultralytics embed sẵn) →
    fallback sidecar `<model>.classes.txt` → fallback `class_<i>`."""
    # 1) ONNX metadata (ultralytics ≥8.x lưu key 'names' = dict-as-str)
    try:
        import onnx
        m = onnx.load(onnx_path, load_external_data=False)
        meta = {p.key: p.value for p in m.metadata_props}
        raw = meta.get("names")
        if raw:
            import ast
            obj = ast.literal_eval(raw)
            if isinstance(obj, dict):
                return [obj[k] for k in sorted(obj.keys(), key=lambda x: int(x))]
            if isinstance(obj, list):
                return list(obj)
    except Exception:
        pass
    # 2) Sidecar txt
    side = os.path.splitext(onnx_path)[0] + ".classes.txt"
    if os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                return [l.strip() for l in f if l.strip()]
        except Exception:
            pass
    return []


def _yolo_load_onnx(onnx_path: str) -> Tuple[Any, List[str], str]:
    """Lazy-load + cache ONNX session. Ưu tiên onnxruntime (nhanh hơn,
    multi-thread tối ưu); fallback cv2.dnn nếu chưa cài."""
    mtime = os.path.getmtime(onnx_path)
    cached = _YOLO_ONNX_CACHE.get(onnx_path)
    if cached and cached[3] == mtime:
        return cached[0], cached[1], cached[2]

    names = _yolo_read_classes(onnx_path)
    sess = None
    backend = ""
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path,
                                     sess_options=opts,
                                     providers=["CPUExecutionProvider"])
        backend = "onnxruntime"
    except ModuleNotFoundError:
        # Fallback: OpenCV DNN
        net = cv2.dnn.readNetFromONNX(onnx_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        sess = net
        backend = "cv2dnn"

    _YOLO_ONNX_CACHE[onnx_path] = (sess, names, backend, mtime)
    return sess, names, backend


def _yolo_onnx_predict(sess, backend: str, blob: np.ndarray) -> np.ndarray:
    """Run inference. Trả về output[0] dạng (1, 4+nc, N) hoặc (1, N, 4+nc).
    YOLOv8 export default = (1, 4+nc, N)."""
    if backend == "onnxruntime":
        in_name = sess.get_inputs()[0].name
        out = sess.run(None, {in_name: blob})[0]
    else:  # cv2dnn
        sess.setInput(blob)
        out = sess.forward()
    return out


def _yolo_postprocess(pred: np.ndarray, conf_thres: float, iou_thres: float,
                      max_det: int, scale: float, pad_x: int, pad_y: int,
                      classes_filter: Optional[List[int]] = None
                      ) -> List[Tuple[int, int, int, int, float, int]]:
    """Decode YOLOv8 raw output → list of (x1,y1,x2,y2,conf,cls_id)
    in ảnh-gốc coords. NMS class-aware qua cv2.dnn.NMSBoxes (C++)."""
    # Normalize shape về (N, 4+nc)
    p = pred[0] if pred.ndim == 3 else pred
    if p.shape[0] < p.shape[1]:           # (4+nc, N) → (N, 4+nc)
        p = p.T
    if p.shape[1] < 5:                    # output rỗng
        return []

    boxes_cxcywh = p[:, :4]
    scores_all = p[:, 4:]                  # (N, nc) — YOLOv8 không có obj
    cls_ids = scores_all.argmax(axis=1)
    confs = scores_all[np.arange(len(scores_all)), cls_ids]

    mask = confs >= conf_thres
    if classes_filter:
        mask &= np.isin(cls_ids, classes_filter)
    if not mask.any():
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    # cxcywh → xywh (top-left + w + h) cho cv2.dnn.NMSBoxes
    xy = boxes_cxcywh[:, :2] - boxes_cxcywh[:, 2:] / 2
    wh = boxes_cxcywh[:, 2:]
    boxes_xywh = np.concatenate([xy, wh], axis=1).tolist()
    confs_list = confs.astype(float).tolist()

    # Class-aware NMS: offset boxes theo class để không suppress chéo class
    max_coord = float(max(boxes_cxcywh[:, 0].max(),
                          boxes_cxcywh[:, 1].max()) + 1.0) * 1e3
    offsets = cls_ids.astype(float) * max_coord
    boxes_offset = [(b[0] + o, b[1] + o, b[2], b[3])
                    for b, o in zip(boxes_xywh, offsets)]
    keep = cv2.dnn.NMSBoxes(boxes_offset, confs_list,
                             conf_thres, iou_thres)
    if len(keep) == 0:
        return []
    keep = np.array(keep).flatten()[:max_det]

    out = []
    for i in keep:
        bx, by, bw, bh = boxes_xywh[i]
        # Un-letterbox: subtract pad → divide scale → ảnh gốc
        x1 = int(round((bx - pad_x) / max(scale, 1e-9)))
        y1 = int(round((by - pad_y) / max(scale, 1e-9)))
        x2 = int(round((bx + bw - pad_x) / max(scale, 1e-9)))
        y2 = int(round((by + bh - pad_y) / max(scale, 1e-9)))
        out.append((x1, y1, x2, y2, float(confs[i]), int(cls_ids[i])))
    return out


def _yolo_detect_onnx(img: np.ndarray, onnx_path: str, params: dict,
                       vis: np.ndarray, s: float) -> Tuple[List[dict], str]:
    """ONNX/cv2dnn detect path — vẽ box lên vis, trả về (detections, backend)."""
    sess, names, backend = _yolo_load_onnx(onnx_path)
    imgsz = int(params.get("imgsz", 640))
    canvas, scale, pad_x, pad_y = _yolo_letterbox(_bgr(img), imgsz)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0,
                                  size=(imgsz, imgsz), swapRB=True,
                                  crop=False)
    pred = _yolo_onnx_predict(sess, backend, blob)
    raw = _yolo_postprocess(
        pred,
        conf_thres=float(params.get("confidence", 0.5)),
        iou_thres=float(params.get("iou", 0.45)),
        max_det=int(params.get("max_det", 300)),
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        classes_filter=list(params.get("classes_filter") or []) or None,
    )

    show_labels = bool(params.get("show_labels", False))
    colors = [(0, 220, 80), (0, 150, 255), (255, 180, 0),
              (220, 80, 220), (0, 220, 220), (255, 80, 80)]
    detections = []
    for x1, y1, x2, y2, conf_val, cls_id in raw:
        cls_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
        col = colors[cls_id % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, _t(2, s))
        if show_labels:
            cv2.putText(vis, f"{cls_name} {conf_val:.2f}",
                        (x1, y1 - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        _fs(0.6, s), col, _t(2, s))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w_ = max(0, x2 - x1); h_ = max(0, y2 - y1)
        detections.append({"class": cls_name, "cls_id": cls_id,
                            "conf": conf_val, "cx": cx, "cy": cy,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "w": w_, "h": h_,
                            "x": x1, "y": y1})
    return detections, backend


def _yolo_empty_outputs(vis=None) -> dict:
    """Default outputs khi không detect được — đảm bảo mọi port luôn có
    value (downstream Crop ROI / Fixture không bị None)."""
    return {"image": vis, "detections": [], "count": 0, "pass": False,
            "x": 0, "y": 0, "w": 0, "h": 0, "cx": 0, "cy": 0,
            "confidence": 0.0, "class": "", "cls_id": -1}


def _yolo_best_outputs(detections: list) -> dict:
    """Pick detection có conf cao nhất → scalar outputs x/y/w/h/cx/cy/
    confidence/class/cls_id. Polygon (seg) detection không có x1..x2 thì
    fallback bbox quanh polygon (đã có cx/cy)."""
    if not detections:
        return {"x": 0, "y": 0, "w": 0, "h": 0, "cx": 0, "cy": 0,
                "confidence": 0.0, "class": "", "cls_id": -1}
    best = max(detections, key=lambda d: float(d.get("conf", 0.0)))
    x1 = int(best.get("x1", best.get("cx", 0)))
    y1 = int(best.get("y1", best.get("cy", 0)))
    x2 = int(best.get("x2", x1))
    y2 = int(best.get("y2", y1))
    return {
        "x": x1, "y": y1,
        "w": max(0, x2 - x1), "h": max(0, y2 - y1),
        "cx": int(best.get("cx", (x1 + x2) // 2)),
        "cy": int(best.get("cy", (y1 + y2) // 2)),
        "confidence": float(best.get("conf", 0.0)),
        "class": str(best.get("class", "")),
        "cls_id": int(best.get("cls_id", -1)),
    }


def proc_yolo_detect(inputs, params):
    """
    YOLOv8/v11 Object Detection & Segmentation.

    Model có thể là:
      • `.onnx` — chạy ONNX Runtime (ưu tiên) hoặc cv2.dnn fallback.
        Inference CPU nhanh hơn PyTorch ~3-4x ở FP32 (không mất accuracy).
      • `.pt`   — chạy ultralytics PyTorch. Nếu có file `.onnx` cùng tên
        cạnh đó thì auto-switch sang ONNX cho tốc độ. Segmentation
        models luôn dùng ultralytics (cần proto-mask postprocess).

    Outputs scalar x/y/w/h/cx/cy là của BEST detection (max conf) —
    chain trực tiếp vào Crop ROI / Fixture. List đầy đủ ở port
    `detections`.
    """
    img = inputs.get("image")
    if img is None:
        return _yolo_empty_outputs(None)

    model_path = params.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        vis = _bgr(img.copy())
        print("[YOLO] No model — open YOLO Studio to train (right-click node)")
        return _yolo_empty_outputs(vis)

    vis = _bgr(img.copy())
    s = _draw_scale(vis)

    # ── Fast path: ONNX (Runtime hoặc cv2.dnn) ─────────────────────
    # .pt model có file .onnx cùng tên → ưu tiên ONNX cho tốc độ.
    ext = os.path.splitext(model_path)[1].lower()
    onnx_candidate = (model_path if ext == ".onnx"
                      else os.path.splitext(model_path)[0] + ".onnx")
    is_seg = "-seg" in os.path.basename(model_path).lower()

    if (not is_seg) and os.path.exists(onnx_candidate):
        try:
            detections, backend = _yolo_detect_onnx(
                img, onnx_candidate, params, vis, s)
            n = len(detections)
            min_det = params.get("min_count", 1)
            max_det_check = params.get("max_count", 9999)
            is_pass = min_det <= n <= max_det_check
            print(f"[YOLO/{backend}] {n} detected "
                  f"{'PASS' if is_pass else 'FAIL'}")
            out = {"image": vis, "detections": detections,
                   "count": n, "pass": is_pass}
            out.update(_yolo_best_outputs(detections))
            # Per-object extra terminals: x_1, cx_2, conf_3, ... do user pick
            _apply_extra_terminals(out, detections, params)
            return out
        except Exception as e:
            print(f"[YOLO/onnx] {type(e).__name__}: {str(e)[:120]} — "
                  f"fallback ultralytics")
            # rơi xuống nhánh ultralytics

    # ── Fallback path: ultralytics PyTorch (.pt hoặc seg) ──────────
    try:
        from ultralytics import YOLO as _YOLO
        model = _YOLO(model_path)
        conf  = params.get("confidence", 0.5)
        iou   = params.get("iou", 0.45)
        imgsz = params.get("imgsz", 640)
        max_det = params.get("max_det", 300)
        classes_filter = params.get("classes_filter", [])

        kw = dict(conf=conf, iou=iou, imgsz=imgsz,
                  max_det=max_det, verbose=False)
        if classes_filter:
            kw["classes"] = classes_filter

        results = model.predict(_bgr(img), **kw)
        show_labels = bool(params.get("show_labels", False))
        detections = []

        # Seg-only: fill mask bằng overlay 30% (mặc định OFF — chỉ vẽ outline).
        fill_seg = bool(params.get("fill_segments", False))
        # Vẽ bbox bao polygon cho seg model (mặc định ON — user thường muốn
        # thấy bbox cleaner thay vì polygon zigzag).
        seg_show_bbox = bool(params.get("seg_show_bbox", True))

        for result in results:
            names = result.names
            # Segmentation masks
            if result.masks is not None:
                for seg, cls_id, conf_val in zip(
                        result.masks.xy, result.boxes.cls, result.boxes.conf):
                    pts = np.array(seg, dtype=np.int32).reshape((-1,1,2))
                    cx  = int(np.mean(pts[:,0,0]))
                    cy  = int(np.mean(pts[:,0,1]))
                    # Bbox bao polygon → cho phép x/y/w/h ports dùng cho
                    # seg detection (chain Crop ROI / Fixture).
                    xs = pts[:, 0, 0]; ys = pts[:, 0, 1]
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()), int(ys.max())
                    cls_name = names[int(cls_id)]
                    colors = [(0,220,80),(0,150,255),(255,180,0),
                              (220,80,220),(0,220,220),(255,80,80)]
                    col = colors[int(cls_id) % len(colors)]
                    if seg_show_bbox:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), col, _t(2, s))
                    else:
                        cv2.polylines(vis, [pts], True, col, _t(2, s))
                    if fill_seg:
                        overlay = vis.copy()
                        cv2.fillPoly(overlay, [pts], col)
                        cv2.addWeighted(vis, 0.7, overlay, 0.3, 0, vis)
                    if show_labels:
                        cv2.putText(vis, f"{cls_name} {float(conf_val):.2f}",
                                    (cx, cy-int(10*s)), cv2.FONT_HERSHEY_SIMPLEX,
                                    _fs(0.6, s), col, _t(2, s))
                    w_ = max(0, x2 - x1); h_ = max(0, y2 - y1)
                    detections.append({"class": cls_name, "cls_id": int(cls_id),
                                       "conf": float(conf_val), "cx": cx, "cy": cy,
                                       "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                       "w": w_, "h": h_,
                                       "x": x1, "y": y1})
            # Bounding boxes
            elif result.boxes is not None:
                for box, cls_id, conf_val in zip(
                        result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1,y1,x2,y2 = [int(v) for v in box]
                    cls_name = names[int(cls_id)]
                    colors = [(0,220,80),(0,150,255),(255,180,0),
                              (220,80,220),(0,220,220),(255,80,80)]
                    col = colors[int(cls_id) % len(colors)]
                    cv2.rectangle(vis, (x1,y1), (x2,y2), col, _t(2, s))
                    if show_labels:
                        cv2.putText(vis, f"{cls_name} {float(conf_val):.2f}",
                                    (x1, y1-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX,
                                    _fs(0.6, s), col, _t(2, s))
                    cx = (x1+x2)//2; cy = (y1+y2)//2
                    w_ = max(0, x2 - x1); h_ = max(0, y2 - y1)
                    detections.append({"class": cls_name, "cls_id": int(cls_id),
                                       "conf": float(conf_val), "cx": cx, "cy": cy,
                                       "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                                       "w": w_, "h": h_,
                                       "x": x1, "y": y1})

        n = len(detections)
        min_det = params.get("min_count", 1)
        max_det_check = params.get("max_count", 9999)
        is_pass = min_det <= n <= max_det_check

        print(f"[YOLO/torch] {n} detected {'PASS' if is_pass else 'FAIL'}")
        out = {"image": vis, "detections": detections,
               "count": n, "pass": is_pass}
        out.update(_yolo_best_outputs(detections))
        _apply_extra_terminals(out, detections, params)
        return out

    except Exception as e:
        print(f"[YOLO] Error: {str(e)[:120]}")
        return _yolo_empty_outputs(vis)


TOOL_REGISTRY: List[ToolDef] = [

  # ── ACQUIRE IMAGE ───────────────────────────────────────────────
  ToolDef("acquire_image","Acquire Image","Acquire Image",
    "Load ảnh từ file hoặc folder — TAcqFifoTool","#0f3460","🖼",
    [],[PortDef("image","image"),PortDef("width","number"),PortDef("height","number"),
        PortDef("acquired","bool"),PortDef("frame_number","number"),
        PortDef("frame_count","number"),PortDef("file_name","str")],
    [ParamDef("source_mode","Source Mode","enum","Folder",
              choices=["Folder","File"],
              tooltip="Chọn nguồn ảnh: 1 thư mục cycle qua các ảnh, hoặc 1 file"),
     ParamDef("folder_path","Image Folder","str","",
              tooltip="Thư mục chứa ảnh",
              visible_if={"source_mode":"Folder"}),
     ParamDef("frame_index","Frame Index","int",0,0,99999,
              tooltip="Index ảnh trong folder — sẽ tự modulo theo số file",
              visible_if={"source_mode":"Folder"}),
     ParamDef("auto_advance","Auto Advance","bool",True,
              tooltip="Mỗi lần Run sẽ tự sang ảnh kế tiếp (cycle qua folder)",
              visible_if={"source_mode":"Folder"}),
     ParamDef("file_path","Image File","str","",
              tooltip="Đường dẫn 1 file ảnh",
              visible_if={"source_mode":"File"}),
     P("width","Width","int",640,1,8192),P("height","Height","int",480,1,8192)],
    proc_acquire_image, "TAcqFifoTool"),

  ToolDef("camera_acquire","Camera Acquire","Acquire Image",
    "Capture từ camera (OpenCV / HikRobot / Do3think) — TAcqFifoTool","#0f3460","📷",
    [],[PortDef("image","image"),PortDef("width","number"),PortDef("height","number"),
        PortDef("acquired","bool")],
    [P("backend","Backend","enum","OpenCV",
        choices=["OpenCV","HikRobot/Do3think"],
        tooltip="OpenCV cho USB UVC; HikRobot/Do3think dùng MVS SDK (Windows only)"),
     P("camera_id","OpenCV index","int",0,0,16,
        visible_if={"backend":"OpenCV"}),
     P("width","Width","int",0,0,8192,tooltip="0=auto",
        visible_if={"backend":"OpenCV"}),
     P("height","Height","int",0,0,8192,tooltip="0=auto",
        visible_if={"backend":"OpenCV"}),
     P("device_index","MVS device index","int",0,0,16,
        tooltip="0-based theo enumeration order",
        visible_if={"backend":"HikRobot/Do3think"}),
     P("serial","MVS serial number","str","",
        tooltip="Để trống để dùng device_index. Khuyến nghị dùng serial khi có nhiều cam.",
        visible_if={"backend":"HikRobot/Do3think"}),
     P("access_mode","Access mode","enum","exclusive",
        choices=["exclusive","monitor","control"],
        visible_if={"backend":"HikRobot/Do3think"}),
     P("heartbeat_ms","Heartbeat (ms)","int",5000,500,60000,
        tooltip="GigE heartbeat timeout",
        visible_if={"backend":"HikRobot/Do3think"}),
     P("continuous_grab","Continuous grab","bool",True,
        tooltip="Thread chạy nền grab liên tục, pipeline lấy frame buffered <1ms. "
                "Tắt nếu cần single-frame chính xác lúc PLC trigger.",
        visible_if={"backend":"HikRobot/Do3think"}),
     P("timeout_ms","Grab timeout (ms)","int",1000,10,30000)],
    proc_camera_acquire, "TAcqFifoTool"),

  # ── PATTERN FIND ────────────────────────────────────────────────
  ToolDef("patmax","Search PatMax","Pattern Find",
    "Pattern matching nâng cao với xoay góc — TPatMaxPatternAlignTool",
    "#16213e","🎯",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("score","number"),
     PortDef("x","number"),PortDef("y","number"),PortDef("angle","number"),
     PortDef("scale","number"),PortDef("num_found","number"),
     PortDef("objects","list")],
    [P("accept_threshold","Accept Threshold","float",0.5,0,1,step=0.01,
       tooltip="Ngưỡng điểm số chấp nhận (0-1)"),
     P("angle_range","Angle Range (°)","float",0,0,180,step=5,
       tooltip="Tìm kiếm trong ±angle_range độ. 0=không xoay"),
     P("angle_step","Angle Step (°)","float",5,1,45,step=1),
     P("num_results","Max Results","int",1,1,20),
     P("coarse_downscale","Coarse downscale","enum","1",
       choices=["1","2","4"],
       tooltip="Speed-up: search ở 1/ds resolution. 2 ≈ 4× nhanh, 4 ≈ 16× nhanh. "
               "Độ chính xác ±ds pixel"),
     P("use_edge","Use edge channel","bool",True,
       tooltip="Tắt để giảm ~30% thời gian khi pattern không phụ thuộc edges"),
     P("use_sqdiff","Use SQDIFF channel","bool",True,
       tooltip="Tắt để giảm ~30% thời gian khi pattern có texture rõ"),
     P("show_xy","Show X,Y reference","bool",True,
       tooltip="Hiện origin marker + X/Y axes + label '(x,y)' trên ảnh output."),
     P("show_bbox","Show bounding box","bool",True,
       tooltip="Hiện rotated bounding box + score label trên ảnh output.")],
    proc_patmax, "TPatMaxPatternAlignTool"),

  ToolDef("patmax_align","PatMax Align Tool","Pattern Find",
    "PatMax Pattern Align — chọn Algorithm & Train Mode (TPMAlignTool)",
    "#16213e","🎯",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("score","number"),
     PortDef("x","number"),PortDef("y","number"),PortDef("angle","number"),
     PortDef("scale","number"),PortDef("num_found","number"),
     PortDef("objects","list")],
    [P("algorithm","Algorithm","enum","PatQuick",
       choices=["PatMax","PatQuick","PatMax & PatQuick","PatFlex",
                "PatMax - High Sensitivity","Perspective PatMax"],
       tooltip="Thuật toán matching pattern"),
     P("train_mode","Train Mode","enum","Image",
       choices=["Image","Shape Models with Image","Shape Models with Transform"],
       tooltip="Chế độ train pattern"),
     P("accept_threshold","Accept Threshold","float",0.5,0,1,step=0.01,
       tooltip="Ngưỡng điểm số chấp nhận (0-1)"),
     P("angle_range","Angle Range (°)","float",0,0,180,step=5,
       tooltip="Tìm kiếm trong ±angle_range độ. 0=không xoay"),
     P("angle_step","Angle Step (°)","float",5,1,45,step=1),
     P("num_results","Max Results","int",1,1,20),
     P("coarse_downscale","Coarse downscale","enum","1",
       choices=["1","2","4"],
       tooltip="Speed-up: search ở 1/ds resolution. 2 ≈ 4× nhanh, 4 ≈ 16× nhanh. "
               "Vị trí ±ds pixel; Perspective/PatFlex tự refine ở full-res"),
     P("show_xy","Show X,Y reference","bool",True,
       tooltip="Hiện origin marker + X/Y axes + label '(x,y)' trên ảnh output."),
     P("show_bbox","Show bounding box","bool",True,
       tooltip="Hiện rotated bounding box + score label trên ảnh output.")],
    proc_patmax_align, "TPMAlignTool"),

  ToolDef("patfind","PatFind","Pattern Find",
    "Pattern matching nhanh (NCC) — TPMAlignTool","#16213e","🔍",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("score","number"),
     PortDef("x","number"),PortDef("y","number"),PortDef("num_found","number")],
    [P("accept_threshold","Accept Threshold","float",0.5,0,1,step=0.01),
     P("show_xy","Show X,Y reference","bool",True,
       tooltip="Hiện origin marker + X/Y axes + label '(x,y)' trên ảnh output."),
     P("show_bbox","Show bounding box","bool",True,
       tooltip="Hiện rotated bounding box + score label trên ảnh output.")],
    proc_patfind, "TPMAlignTool"),

  # ── FIXTURE ─────────────────────────────────────────────────────
  ToolDef("fixture","Fixture","Fixture",
    "Thiết lập hệ tọa độ theo part — TFixtureTool","#1a1a2e","📌",
    [PortDef("image","image"),
     PortDef("ref_x","number",required=False),
     PortDef("ref_y","number",required=False),
     PortDef("ref_angle","number",required=False)],
    [PortDef("image","image"),PortDef("transform_matrix","any"),
     PortDef("offset_x","number"),PortDef("offset_y","number"),PortDef("angle","number")],
    [P("origin_x","Origin X","float",320,0,8192),
     P("origin_y","Origin Y","float",240,0,8192)],
    proc_fixture, "TFixtureTool"),

  # ── CALIPER ─────────────────────────────────────────────────────
  ToolDef("caliper","Caliper","Caliper",
    "Đo cạnh & khoảng cách 2 cạnh sub-pixel — TCaliperTool","#1b4332","📐",
    [PortDef("image","image"),
     PortDef("x1","number",required=False), PortDef("y1","number",required=False),
     PortDef("x2","number",required=False), PortDef("y2","number",required=False)],
    [PortDef("image","image"),PortDef("edge1_pos","number"),PortDef("edge2_pos","number"),
     PortDef("width","number"),PortDef("pass","bool"),PortDef("edges_found","number")],
    [P("x1","X1","int",100,0,8192,tooltip="Điểm đầu caliper"),
     P("y1","Y1","int",240,0,8192),
     P("x2","X2","int",540,0,8192,tooltip="Điểm cuối caliper"),
     P("y2","Y2","int",240,0,8192),
     P("caliper_width","Caliper Width (px)","int",20,1,200,tooltip="Bề rộng vùng tìm cạnh"),
     P("polarity","Polarity","enum","Either",
       choices=["Dark→Light","Light→Dark","Either"],tooltip="Cực tính cạnh tìm kiếm"),
     P("filter_half_size","Filter Half Size","int",2,1,20,tooltip="Gaussian derivative sigma"),
     P("num_edges","Num Edges","int",2,1,10),
     P("edge_threshold","Edge Threshold","float",10.0,0,500,
       tooltip="Ngưỡng cường độ gradient để nhận cạnh"),
     P("pixel_to_mm","Pixel → mm","float",1.0,0.0001,1000,step=0.0001),
     P("min_width","Min Width (mm)","float",0.0,0,10000),
     P("max_width","Max Width (mm)","float",9999.0,0,10000),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label edge (E1:42.5 …) lên ảnh output. Mặc định tắt — số đo vẫn được log ra console.")],
    proc_caliper, "TCaliperTool"),

  ToolDef("caliper_multi","Caliper Multi-Edge","Caliper",
    "Tìm tất cả cạnh trong vùng — TCaliperTool","#1b4332","📏",
    [PortDef("image","image"),
     PortDef("x1","number",required=False), PortDef("y1","number",required=False),
     PortDef("x2","number",required=False), PortDef("y2","number",required=False)],
    [PortDef("image","image"),PortDef("edges","any"),
     PortDef("count","number"),PortDef("pass","bool")],
    [P("x1","X1","int",0,0,8192),P("y1","Y1","int",240,0,8192),
     P("x2","X2","int",640,0,8192),P("y2","Y2","int",240,0,8192),
     P("edge_threshold","Edge Threshold","float",10.0,0,500),
     P("min_count","Min Edges","int",1,0,100),
     P("max_count","Max Edges","int",100,0,1000)],
    proc_caliper_multi, "TCaliperTool"),

  # ── BLOB ────────────────────────────────────────────────────────
  ToolDef("blob","Blob Analysis","Blob Analysis",
    "Phân tích vùng: area, circularity, elongation, bounding box — TBlobTool. "
    "Nối node Morphology phía trước nếu cần lọc nhiễu/vá lỗ.\n"
    "Khi mask đến từ ROI nhỏ hơn image (ví dụ crop_roi → color_segment → blob), "
    "nối crop_roi.x → offset_x, crop_roi.y → offset_y để contour/bbox/centroid "
    "vẽ đúng vị trí trên image và cx/cy ra ở hệ toạ độ full image.",
    "#2d6a4f","🔵",
    [PortDef("image","image"),PortDef("mask","image",required=False),
     PortDef("offset_x","number",required=False),
     PortDef("offset_y","number",required=False)],
    # Primary outputs: axis-aligned bbox (x,y,w,h) của blob đầu tiên — visible
    # mặc định cho new node. Centroid (cx,cy) + rotated bbox (bbox_w,bbox_h) +
    # các scalar phụ ẩn mặc định trong UI qua _hidden_outputs (user toggle qua
    # dialog "👁 Show / Hide Output Ports"). Tất cả vẫn được emit qua proc_blob
    # nên pipeline cũ không gãy.
    [PortDef("image","image"),
     PortDef("x","number"),PortDef("y","number"),
     PortDef("w","number"),PortDef("h","number"),
     PortDef("count","number"),PortDef("pass","bool"),
     PortDef("total_area","number"),PortDef("blobs","any"),PortDef("centroids","any"),
     PortDef("cx","number"),PortDef("cy","number"),PortDef("area","number"),
     PortDef("bbox_w","number"),PortDef("bbox_h","number"),
     PortDef("angle","number")],
    [P("auto_threshold","Auto Threshold (Otsu)","bool",True,
       tooltip="Chỉ áp dụng khi KHÔNG có port mask kết nối. Có mask → bỏ qua."),
     P("threshold","Manual Threshold","int",128,0,255),
     P("invert","Invert Mask","bool",False),
     P("min_area","Min Area (px²)","float",50,0,1e7),
     P("max_area","Max Area (px²)","float",1e7,0,1e9),
     P("min_circularity","Min Circularity","float",0.0,0,1,step=0.05,
       tooltip="0=mọi hình dạng, 1=hình tròn hoàn hảo"),
     P("max_circularity","Max Circularity","float",1.1,0,1.1,step=0.05),
     P("min_elongation","Min Elongation","float",0.0,0,1000,step=0.1,
       tooltip="Tỉ lệ dài/rộng của bounding box"),
     P("max_elongation","Max Elongation","float",1000.0,0,10000),
     P("pixel_to_mm2","px²→mm²","float",1.0,0.0001,1e6,step=0.0001),
     P("min_count","Min Count","int",1,0,10000),
     P("max_count","Max Count","int",1000,0,10000),
     P("selection_mode","Output Selection","enum","First",
       choices=["First","Last","Highest","Lowest","Average"],
       tooltip="Chọn 1 PIXEL trên contour blob để emit thành x,y output:\n"
               "• First: pixel trái nhất (x nhỏ nhất).\n"
               "• Last: pixel phải nhất (x lớn nhất).\n"
               "• Highest: pixel cao nhất (y nhỏ nhất, trên cùng ảnh).\n"
               "• Lowest: pixel thấp nhất (y lớn nhất, dưới cùng ảnh).\n"
               "• Average: trung tâm trung bình (centroid của centroids).\n"
               "Marker magenta vẽ tại pixel được chọn. w,h,area,... vẫn "
               "là property của blob chứa pixel đó."),
     P("downscale","Coarse Downscale","int",0,0,16,
       tooltip="0=auto target ~1.5MP. ≥1 force tỉ lệ. Giảm DS× → giảm "
               "DS²× thời gian threshold+findContours. Toạ độ/area scale "
               "ngược về full-res cho overlay & output."),
     P("show_contours","Show Contours","bool",True,
       tooltip="Vẽ contour quanh từng blob."),
     P("contour_color","Contour Color","enum","Yellow",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("contour_thickness","Contour Thickness","int",2,1,12,use_slider=True,
       tooltip="Độ dày nét contour (px, scale theo ảnh)."),
     P("show_bbox","Show BBox","bool",True,
       tooltip="Vẽ rotated bounding box."),
     P("bbox_color","BBox Color","enum","Orange",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("bbox_thickness","BBox Thickness","int",1,1,12,use_slider=True,
       tooltip="Độ dày nét bounding box (px, scale theo ảnh)."),
     P("show_centroid","Show Centroid","bool",True,
       tooltip="Chấm tại tâm blob."),
     P("centroid_color","Centroid Color","enum","Green",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("show_labels","Show Labels","bool",False,
       tooltip="Hiển thị label area (mm2) cạnh từng blob."),
     P("label_dx","Label Offset X","int",6,-300,300,use_slider=True,
       tooltip="Dịch label theo trục X (px, đã scale theo ảnh). "
               "Có thể kéo label trực tiếp trên ảnh — slider tự sync."),
     P("label_dy","Label Offset Y","int",-6,-300,300,use_slider=True,
       tooltip="Dịch label theo trục Y (px). Âm = lên trên, dương = xuống dưới. "
               "Có thể kéo label trực tiếp trên ảnh — slider tự sync."),
     P("label_color","Label Color","enum","Yellow",
       choices=["Yellow","Cyan","Green","Red","White","Magenta"]),
     P("label_font","Label Font","enum","Simplex",
       choices=["Simplex","Plain","Duplex","Complex","Triplex",
                "Script Simplex","Script Complex"]),
     P("label_size","Label Font Size","float",0.45,0.2,3.0,step=0.05,
       use_slider=True,
       tooltip="Cỡ chữ (font scale OpenCV)."),
     P("label_thickness","Label Thickness","int",1,1,8,use_slider=True,
       tooltip="Độ dày nét chữ.")],
    proc_blob, "TBlobTool"),

  # ── EDGE / LINE / CIRCLE ────────────────────────────────────────
  ToolDef("find_line","Find Line","Edge & Geometry",
    "Tìm đường thẳng từ edge — TFindLineTool","#134074","〰",
    [PortDef("image","image"),
     PortDef("x1","number",required=False), PortDef("y1","number",required=False),
     PortDef("x2","number",required=False), PortDef("y2","number",required=False)],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("angle","number"),
     PortDef("distance","number"),PortDef("point_x","number"),
     PortDef("point_y","number"),PortDef("pass","bool")],
    [P("x1","ROI X1","int",0,0,8192),P("y1","ROI Y1","int",200,0,8192),
     P("x2","ROI X2","int",640,0,8192),P("y2","ROI Y2","int",280,0,8192),
     P("canny_low","Canny Low","int",50,0,500),P("canny_high","Canny High","int",150,0,500),
     P("pixel_to_mm","Pixel→mm","float",1.0,0.0001,1000,step=0.0001),
     P("min_angle","Min Angle (°)","float",-180,-180,180),
     P("max_angle","Max Angle (°)","float",180,-180,180),
     P("downscale","Coarse Downscale","int",0,0,16,
       tooltip="0=auto (target ~1.5MP). Canny chỉ chạy trong ROI band; downscale thêm khi band lớn.")],
    proc_find_line, "TFindLineTool"),

  ToolDef("find_circle","Find Circle","Edge & Geometry",
    "Fit đường tròn chính xác — TFindCircleTool","#134074","⭕",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("cx","number"),
     PortDef("cy","number"),PortDef("radius","number"),PortDef("pass","bool")],
    [P("dp","DP","float",1.2,1,4,step=0.1),P("min_dist","Min Dist (px)","int",30,1,500),
     P("param1","Canny High","int",100,1,300),P("param2","Accum Thresh","int",30,1,300),
     P("min_radius","Min Radius (px)","int",5,0,1000),
     P("max_radius","Max Radius (px)","int",300,0,5000),
     P("pixel_to_mm","Pixel→mm","float",1.0,0.0001,1000,step=0.0001),
     P("min_r_check","Min Radius (mm)","float",0.0,0,10000),
     P("max_r_check","Max Radius (mm)","float",9999.0,0,10000),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label 'R=…mm cx=… cy=…' lên ảnh output. Mặc định tắt — vẫn được log.")],
    proc_find_circle, "TFindCircleTool"),

  # ── COLOR ───────────────────────────────────────────────────────
  ToolDef("color_picker","Color Picker","Color Analysis",
    "Click chuột lấy màu → xuất HSV range","#6b2737","🎨",
    [PortDef("image","image"),
     PortDef("pick_x","number",required=False),
     PortDef("pick_y","number",required=False)],
    [PortDef("image","image"),PortDef("color_hsv","any"),
     PortDef("h","number"),PortDef("s","number"),PortDef("v","number"),
     PortDef("r","number"),PortDef("g","number"),PortDef("b","number")],
    [P("pick_x","Pick X","int",0,0,8192,tooltip="Double-click node → click ảnh để lấy màu"),
     P("pick_y","Pick Y","int",0,0,8192),
     P("tolerance","HSV Tolerance","int",20,0,100),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label 'H S V' cạnh điểm picked lên ảnh output. Mặc định tắt.")],
    proc_color_picker, "TColorTool"),

  ToolDef("color_segment","Color Segmentation","Color Analysis",
    "Phân đoạn màu (HSV/RGB/HSL/Lab/Gray) — TColorSegmenterTool","#6b2737","🌈",
    [PortDef("image","image"),PortDef("color_hsv","any",required=False),
     PortDef("x","number",required=False),PortDef("y","number",required=False),
     PortDef("w","number",required=False),PortDef("h","number",required=False)],
    [PortDef("image","image"),PortDef("mask","image"),PortDef("pass","bool"),
     PortDef("pixel_ratio","number"),PortDef("pixel_count","number")],
    [P("color_space","Color Space","enum","HSV",
       choices=["HSV","RGB","HSL","Lab","Gray"],
       tooltip="Không gian màu dùng để lọc:\n"
               "• HSV — Hue/Saturation/Value (mặc định, tốt cho phân biệt màu).\n"
               "• RGB — Red/Green/Blue (lọc theo cường độ từng kênh).\n"
               "• HSL — Hue/Saturation/Lightness.\n"
               "• Lab — CIELAB (đồng nhất perceptually, tốt cho phân biệt tinh).\n"
               "• Gray — đơn kênh xám 0–255.\n"
               "Click 'Pick Color' trên header rồi click ảnh để tự set ngưỡng "
               "quanh màu pixel đã chọn."),
     # HSV
     P("h_low","H Low","int",0,0,180,use_slider=True,
       visible_if={"color_space":"HSV"}),
     P("h_high","H High","int",180,0,180,use_slider=True,
       visible_if={"color_space":"HSV"}),
     P("s_low","S Low","int",50,0,255,use_slider=True,
       visible_if={"color_space":"HSV"}),
     P("s_high","S High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"HSV"}),
     P("v_low","V Low","int",50,0,255,use_slider=True,
       visible_if={"color_space":"HSV"}),
     P("v_high","V High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"HSV"}),
     # RGB
     P("r_low","R Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"RGB"}),
     P("r_high","R High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"RGB"}),
     P("g_low","G Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"RGB"}),
     P("g_high","G High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"RGB"}),
     P("b_low","B Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"RGB"}),
     P("b_high","B High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"RGB"}),
     # HSL (OpenCV gọi là HLS — H 0-180, L 0-255, S 0-255)
     P("hsl_h_low","H Low","int",0,0,180,use_slider=True,
       visible_if={"color_space":"HSL"}),
     P("hsl_h_high","H High","int",180,0,180,use_slider=True,
       visible_if={"color_space":"HSL"}),
     P("hsl_l_low","L Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"HSL"}),
     P("hsl_l_high","L High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"HSL"}),
     P("hsl_s_low","S Low","int",50,0,255,use_slider=True,
       visible_if={"color_space":"HSL"}),
     P("hsl_s_high","S High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"HSL"}),
     # Lab (OpenCV 8-bit: L 0-255, a 0-255 ~127=neutral, b 0-255 ~127=neutral)
     P("lab_l_low","L Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"Lab"}),
     P("lab_l_high","L High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"Lab"}),
     P("lab_a_low","a Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"Lab"}),
     P("lab_a_high","a High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"Lab"}),
     P("lab_b_low","b Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"Lab"}),
     P("lab_b_high","b High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"Lab"}),
     # Gray
     P("gray_low","Gray Low","int",0,0,255,use_slider=True,
       visible_if={"color_space":"Gray"}),
     P("gray_high","Gray High","int",255,0,255,use_slider=True,
       visible_if={"color_space":"Gray"}),
     # Chung
     P("tolerance","Tolerance","int",0,0,100,use_slider=True,
       tooltip="Nới rộng mỗi kênh ± tolerance. Cũng dùng làm nửa-độ-rộng "
               "khi click Pick Color → set thresholds quanh pixel ± tol."),
     P("morph_open","Morph Open","int",0,0,50,use_slider=True),
     P("min_ratio","Min Ratio","float",0.01,0,1,step=0.001,use_slider=True),
     P("max_ratio","Max Ratio","float",1.0,0,1,step=0.001,use_slider=True),
     P("downscale","Coarse Downscale","int",0,0,16,
       tooltip="0=auto target ~1.5MP. cvtColor+inRange+morph chạy trên "
               "ảnh nhỏ; mask resize ngược về full-res. Cho ảnh 20MP: "
               "~80ms → ~6ms."),
     P("show_mask","Show Mask","bool",False,
       tooltip="Hiển thị mask nhị phân (trắng/đen) thay vì overlay xanh."),
     P("roi_shape","ROI Shape","enum","Full Image",
       choices=["Full Image","Rectangle","Circle","Ellipse","Polygon"],
       tooltip="Giới hạn vùng phân tích:\n"
               "• Full Image: toàn ảnh (mặc định).\n"
               "• Rectangle/Circle/Ellipse: kéo chuột để vẽ.\n"
               "• Polygon: click từng đỉnh, double-click đóng, right-click huỷ.\n"
               "Pixel ratio sẽ tính trên diện tích shape (không phải toàn ảnh)."),
     P("roi_w","ROI W","int",100,1,8192,use_slider=True,
       tooltip="Width khi shape được sinh từ port x/y (không vẽ tay) hoặc "
               "fallback khi port w không kết nối. Tự sync với shape khi "
               "vẽ tay để giữ ngữ cảnh khi sau này nối port."),
     P("roi_h","ROI H","int",100,1,8192,use_slider=True,
       tooltip="Height khi shape được sinh từ port x/y hoặc fallback khi "
               "port h không kết nối.")],
    proc_color_segment, "TColorSegmenterTool"),

  ToolDef("color_match","Color Match","Color Analysis",
    "So khớp màu trung bình ROI — TColorMatchTool","#6b2737","🎭",
    [PortDef("image","image"),
     PortDef("x","number",required=False), PortDef("y","number",required=False),
     PortDef("w","number",required=False), PortDef("h","number",required=False)],
    [PortDef("image","image"),PortDef("pass","bool"),PortDef("delta_e","number"),
     PortDef("mean_r","number"),PortDef("mean_g","number"),PortDef("mean_b","number")],
    [P("x","ROI X","int",0,0,8192),P("y","ROI Y","int",0,0,8192),
     P("w","ROI W","int",50,1,8192),P("h","ROI H","int",50,1,8192),
     P("ref_r","Ref R","int",128,0,255),P("ref_g","Ref G","int",128,0,255),
     P("ref_b","Ref B","int",128,0,255),
     P("max_delta_e","Max ΔE","float",30.0,0,441,tooltip="ΔE Euclidean RGB distance")],
    proc_color_match, "TColorMatchTool"),

  # ── ID / READ ───────────────────────────────────────────────────
  ToolDef("id_reader","ID Reader","ID & Read",
    "Đọc Barcode 1D/2D, QR, DataMatrix — TIDReaderTool","#3d0c02","📦",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("data","any"),
     PortDef("symbology","any"),PortDef("pass","bool")],
    [P("expected_data","Expected Data","str","",tooltip="Để trống = chấp nhận mọi code")],
    proc_id_reader, "TIDReaderTool"),

  ToolDef("ocr_max","OCR Max","ID & Read",
    "Nhận dạng & xác nhận ký tự — TOCRMaxTool. "
    "Engine 'auto' tự fallback tesseract→easyocr. "
    "Cài: pip install pytesseract (+ tesseract-ocr binary & langpack vie) "
    "hoặc pip install easyocr (không cần binary, tốt cho tiếng Việt + nền phức tạp).",
    "#3d0c02","🔤",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("text","any"),
     PortDef("pass","bool"),PortDef("confidence","number")],
    [P("engine","OCR Engine","enum","auto",
        choices=["auto","tesseract","easyocr"],
        tooltip="auto: thử tesseract → easyocr. easyocr không cần Tesseract binary; "
                "tốt cho tiếng Việt + nền phức tạp nhưng nặng (PyTorch, ~5-15s load lần đầu)."),
     P("lang","Language","str","eng",
        tooltip="Tesseract code: eng, vie, jpn, kor, chi_sim, 'vie+eng'… "
                "EasyOCR auto-map: vie→vi, jpn→ja. Tiếng Việt = 'vie'."),
     P("psm","PSM Mode","int",6,0,13,
        tooltip="Tesseract Page Segmentation Mode — 6=block, 7=single line, 8=single word, "
                "11=sparse text (chữ thưa, không layout cố định)."),
     P("preprocess","Preprocess","enum","none",
        choices=["none","otsu","adaptive","binary"],
        tooltip="Binarize trước OCR — giúp tách chữ khỏi nền phức tạp / sáng không đều. "
                "Otsu: ngưỡng tự động; adaptive: sáng cục bộ; binary: ngưỡng 127."),
     P("invert","Invert","bool",False,
        tooltip="Đảo trắng-đen (cho text sáng trên nền tối)."),
     P("expected_text","Expected Text","str",""),
     P("min_confidence","Min Confidence (%)","float",60.0,0,100)],
    proc_ocr_max, "TOCRMaxTool"),

  # ── MEASUREMENT ─────────────────────────────────────────────────
  ToolDef("dist_point","Distance Point-Point","Measurement",
    "Đo khoảng cách 2 điểm — TDistancePointPointTool. "
    "Calib 'Two Points' = nội suy tuyến tính từ 2 cặp (px, mm) đã đo.",
    "#134074","↔",
    [PortDef("image","image",required=False),
     PortDef("x1","number",required=False),PortDef("y1","number",required=False),
     PortDef("x2","number",required=False),PortDef("y2","number",required=False)],
    [PortDef("image","image"),PortDef("distance","number"),
     PortDef("distance_px","number"),
     PortDef("x1","number"),PortDef("y1","number"),
     PortDef("x2","number"),PortDef("y2","number"),
     PortDef("pass","bool")],
    [P("x1","X1","int",0,0,8192),P("y1","Y1","int",0,0,8192),
     P("x2","X2","int",100,0,8192),P("y2","Y2","int",0,0,8192),
     P("calib_mode","Calibration","enum","Scale",
       choices=["Scale","Two Points"],
       tooltip="Scale: mm = px × pixel_to_mm.\n"
               "Two Points: nội suy tuyến tính qua 2 mốc đã đo "
               "(vd 150px↔2.1mm, 210px↔2.5mm)."),
     P("pixel_to_mm","Pixel→mm","float",1.0,0.0001,1000,step=0.0001,
       visible_if={"calib_mode": "Scale"}),
     P("calib_p1_px","P1 pixels","float",0.0,0,1e6,step=0.1,
       tooltip="Khoảng cách pixel của mốc 1 (vd 150).",
       visible_if={"calib_mode": "Two Points"}),
     P("calib_p1_mm","P1 mm","float",0.0,-1e6,1e6,step=0.001,
       tooltip="Giá trị mm tương ứng mốc 1 (vd 2.1).",
       visible_if={"calib_mode": "Two Points"}),
     P("calib_p2_px","P2 pixels","float",100.0,0,1e6,step=0.1,
       tooltip="Khoảng cách pixel của mốc 2 (vd 210).",
       visible_if={"calib_mode": "Two Points"}),
     P("calib_p2_mm","P2 mm","float",1.0,-1e6,1e6,step=0.001,
       tooltip="Giá trị mm tương ứng mốc 2 (vd 2.5).",
       visible_if={"calib_mode": "Two Points"}),
     P("min_dist","Min (mm)","float",0.0,-1e6,1e6),
     P("max_dist","Max (mm)","float",9999.0,-1e6,1e6),
     P("show_labels","Show label on image","bool",True,
       tooltip="Vẽ label '…mm' lên ảnh và cho phép kéo trên canvas."),
     P("label_dx","Label Offset X","int",0,-500,500,use_slider=True,
       tooltip="Dịch label theo trục X từ trung điểm. "
               "Kéo label trên ảnh sẽ sync vào slider."),
     P("label_dy","Label Offset Y","int",-10,-500,500,use_slider=True,
       tooltip="Dịch label theo trục Y. Âm = lên trên."),
     P("label_color","Label Color","enum","Yellow",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("label_font","Label Font","enum","Simplex",
       choices=["Simplex","Plain","Duplex","Complex","Triplex"]),
     P("label_size","Label Size","float",0.6,0.2,3.0,step=0.05,use_slider=True),
     P("label_thickness","Label Thickness","int",2,1,8,use_slider=True)],
    proc_distance_point, "TDistancePointPointTool"),

  ToolDef("dist_point_line","Distance Point-Line","Measurement",
    "Khoảng cách vuông góc từ điểm đến đường thẳng — TDistancePointLineTool",
    "#134074","⊥",
    [PortDef("image","image",required=False),
     PortDef("px","number",required=False), PortDef("py","number",required=False),
     PortDef("lx1","number",required=False), PortDef("ly1","number",required=False),
     PortDef("lx2","number",required=False), PortDef("ly2","number",required=False),
     PortDef("line_angle","number",required=False)],
    [PortDef("image","image"), PortDef("distance","number"),
     PortDef("signed_distance","number"),
     PortDef("foot_x","number"), PortDef("foot_y","number"),
     PortDef("pass","bool")],
    [P("mode","Line Mode","enum","Two Points",
       choices=["Two Points","Point + Angle"],
       tooltip="Two Points: dùng lx1/ly1 + lx2/ly2. Point + Angle: lx1/ly1 + line_angle (nối từ Find Line)."),
     P("px","Point X","int",0,0,8192),
     P("py","Point Y","int",0,0,8192),
     P("lx1","Line P1 X","int",0,0,8192),
     P("ly1","Line P1 Y","int",100,0,8192),
     P("lx2","Line P2 X","int",100,0,8192),
     P("ly2","Line P2 Y","int",100,0,8192),
     P("line_angle","Line Angle (°)","float",0.0,-180,180,step=0.1,
       tooltip="Chỉ dùng khi mode = Point + Angle"),
     P("pixel_to_mm","Pixel→mm","float",1.0,0.0001,1000,step=0.0001),
     P("min_dist","Min (mm)","float",0.0,0,100000),
     P("max_dist","Max (mm)","float",9999.0,0,100000),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label '…mm' giữa điểm và chân đường vuông góc.")],
    proc_distance_point_line, "TDistancePointLineTool"),

  ToolDef("angle_lines","Angle Line-Line","Measurement",
    "Đo góc giữa 2 đường — TAngleLineLineTool","#134074","∠",
    [PortDef("image","image",required=False),
     PortDef("angle1","number",required=False),PortDef("angle2","number",required=False)],
    [PortDef("image","image"),PortDef("angle","number"),PortDef("pass","bool")],
    [P("line1_angle","Line 1 Angle (°)","float",0,-180,180,step=0.1),
     P("line2_angle","Line 2 Angle (°)","float",45,-180,180,step=0.1),
     P("min_angle","Min Angle (°)","float",0,0,180),
     P("max_angle","Max Angle (°)","float",90,0,180)],
    proc_angle_lines, "TAngleLineLineTool"),

  ToolDef("area_measure","Area Measure","Measurement",
    "Đo diện tích vùng từ mask/contours","#134074","⬛",
    [PortDef("image","image",required=False),PortDef("mask","image",required=False),
     PortDef("contours","contours",required=False)],
    [PortDef("image","image"),PortDef("total_area","number"),
     PortDef("count","number"),PortDef("pass","bool")],
    [P("pixel_to_mm2","px²→mm²","float",1.0,0.0001,1e6,step=0.0001),
     P("min_area","Min Area (mm²)","float",0,0,1e9),
     P("max_area","Max Area (mm²)","float",1e9,0,1e9),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label area cạnh từng contour lên ảnh output. Mặc định tắt.")],
    proc_area, "TMeasureRectangleTool"),

  # ── SURFACE INSPECTION ──────────────────────────────────────────
  ToolDef("surface_defect","Surface Defect","Surface Inspection",
    "Phát hiện khuyết tật bề mặt","#4a0404","🔴",
    [PortDef("image","image"),PortDef("reference","image",required=False)],
    [PortDef("image","image"),PortDef("pass","bool"),
     PortDef("defect_area","number"),PortDef("defect_count","number")],
    [P("threshold","Diff Threshold","int",30,0,255),
     P("morph_k","Morph Kernel","int",3,1,21,step=2),
     P("min_defect_px","Min Defect (px²)","int",10,0,10000),
     P("max_defect_area","Max Total Defect (px²)","int",1000,0,1000000),
     P("max_defect_count","Max Defect Count (0=any)","int",0,0,1000),
     P("downscale","Coarse Downscale","int",0,0,16,
       tooltip="0=auto (target ~1.5MP). 20MP → ds=4 → ~6ms vs ~45ms full-res.")],
    proc_surface_defect, ""),

  ToolDef("scratch_detect","Scratch Detection","Surface Inspection",
    "Phát hiện vết xước dạng đường thẳng","#4a0404","⚡",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("pass","bool"),
     PortDef("scratch_count","number"),PortDef("total_length","number")],
    [P("blur_k","Blur Kernel","int",3,0,21,step=2),
     P("canny_low","Canny Low","int",30,0,300),
     P("canny_high","Canny High","int",100,0,500),
     P("hough_thresh","Hough Threshold","int",30,1,300),
     P("min_scratch_length","Min Length (px)","int",50,1,2000),
     P("max_gap","Max Gap (px)","int",5,0,100),
     P("max_scratches","Max Scratches (0=none)","int",0,0,1000),
     P("downscale","Coarse Downscale","int",0,0,16,
       tooltip="0=auto (target ~1.5MP). Tọa độ/length scale ngược về full-res.")],
    proc_scratch_detect, ""),

  # ── IMAGE PROCESSING ────────────────────────────────────────────
  ToolDef("image_convert","Image Convert","Image Processing",
    "Chuyển đổi format ảnh — TImageConvertTool","#2c3e50","🔄",
    [PortDef("image","image")],[PortDef("image","image")],
    [P("mode","Mode","enum","Grayscale",
       choices=["Grayscale","BGR to RGB","Invert","HSV","LAB","YCrCb"])],
    proc_image_convert, "TImageConvertTool"),

  ToolDef("crop_roi","Crop ROI","Image Processing",
    "Cắt vùng ROI — nhận x/y/w/h từ PatMax để tracking.\n"
    "• image     = ảnh gốc clean (cho downstream xử lý, panel vẫn hiện bbox).\n"
    "• roi_image = vùng đã cắt từ ảnh gốc (clean, không vướng overlay upstream).",
    "#2c3e50","✂",
    [PortDef("image","image"),
     PortDef("x","number",required=False,default=None),
     PortDef("y","number",required=False,default=None),
     PortDef("w","number",required=False,default=None),
     PortDef("h","number",required=False,default=None)],
    [PortDef("image","image"),PortDef("roi_image","image"),
     PortDef("x","number"),PortDef("y","number"),
     PortDef("w","number"),PortDef("h","number")],
    [P("x","X","int",0,0,8192),P("y","Y","int",0,0,8192),
     P("crop_w","Width","int",320,1,8192),P("crop_h","Height","int",240,1,8192)],
    proc_crop, ""),

  ToolDef("threshold","Threshold","Image Processing",
    "Ngưỡng nhị phân đầy đủ — Binary/INV/Otsu/Triangle","#2c3e50","⚡",
    [PortDef("image","image")],[PortDef("image","image"),PortDef("mask","image")],
    [P("method","Method","enum","Otsu",
       choices=["Binary","Binary INV","Trunc","To Zero","To Zero INV",
                "Otsu","Otsu INV","Triangle","Triangle INV"]),
     P("threshold","Threshold","int",127,0,255),
     P("max_value","Max Value","int",255,0,255)],
    proc_threshold, ""),

  ToolDef("morphology","Morphology","Image Processing",
    "Biến đổi hình thái: Erode/Dilate/Open/Close — lọc nhiễu, vá lỗ, tách object trên mask nhị phân",
    "#2c3e50","🔲",
    [PortDef("image","image")],[PortDef("image","image"),PortDef("mask","image")],
    [P("operation","Operation","enum","Open",
       choices=["Erode","Dilate","Open","Close","Gradient",
                "Top Hat","Black Hat","Open+Close","Close+Open"],
       tooltip="• Erode: co vùng sáng — xoá đốm trắng nhỏ, làm mảnh nét.\n"
               "• Dilate: phình vùng sáng — vá lỗ đen, làm dày nét.\n"
               "• Open  = Erode → Dilate — xoá noise trắng nhưng giữ hình.\n"
               "• Close = Dilate → Erode — vá lỗ đen bên trong object.\n"
               "• Gradient: viền (Dilate − Erode) — trích biên.\n"
               "• Top Hat: ảnh − Open — nổi đốm sáng nhỏ.\n"
               "• Black Hat: Close − ảnh — nổi đốm tối nhỏ.\n"
               "• Open+Close: xoá noise rồi vá lỗ (mask sạch).\n"
               "• Close+Open: vá lỗ rồi xoá noise."),
     P("shape","Kernel Shape","enum","Ellipse",
       choices=["Rect","Ellipse","Cross"],
       tooltip="Hình kernel:\n• Ellipse: mượt, tự nhiên (mặc định).\n"
               "• Rect: góc cạnh, giữ cạnh thẳng.\n• Cross: chỉ 4 hướng, nhẹ."),
     P("kernel_size","Kernel Size","int",3,1,51,step=2,use_slider=True,
       tooltip="Kích thước kernel (pixel). Lớn = tác động mạnh hơn. "
               "Khuyến nghị số lẻ (3,5,7…)."),
     P("iterations","Iterations","int",1,1,20,use_slider=True,
       tooltip="Lặp phép morph N lần — tương đương kernel lớn hơn nhưng tinh chỉnh được."),
     P("auto_binarize","Auto Binarize (Otsu)","bool",False,
       tooltip="Tự động ngưỡng Otsu trước morph. Bật khi input là ảnh xám/màu "
               "(chưa phải mask nhị phân)."),
     P("invert_input","Invert Input","bool",False,
       tooltip="Đảo trắng ↔ đen trước khi morph. Dùng khi object là pixel đen "
               "trên nền trắng (vì morph thao tác trên vùng sáng)."),
     P("show_overlay","Show Overlay","bool",False,
       tooltip="Hiển thị overlay xanh của kết quả đè lên ảnh gốc — dễ so sánh "
               "vùng nào bị thay đổi.")],
    proc_morphology, ""),

  ToolDef("gaussian_blur","Gaussian Blur","Image Processing",
    "Làm mờ Gaussian","#2c3e50","🌀",
    [PortDef("image","image")],[PortDef("image","image")],
    [P("kernel_size","Kernel Size","int",5,1,99,step=2),
     P("sigma","Sigma","float",0.0,0,50,step=0.5)],
    proc_gaussian_blur, ""),

  ToolDef("sharpen","Sharpen","Image Processing",
    "Làm nét ảnh","#2c3e50","⭐",
    [PortDef("image","image")],[PortDef("image","image")],
    [P("strength","Strength","float",1.0,0.1,10,step=0.1)],
    proc_sharpen, ""),

  ToolDef("find_contours","Find Contours","Image Processing",
    "Tìm contour từ mask","#2c3e50","🔍",
    [PortDef("image","image"),PortDef("mask","image",required=False)],
    [PortDef("image","image"),PortDef("contours","contours"),
     PortDef("count","number"),PortDef("pass","bool")],
    [P("retrieval","Retrieval","enum","External",choices=["External","List","Tree","CCOMP"]),
     P("min_area","Min Area (px²)","float",10,0,1e6),
     P("max_area","Max Area (px²)","float",1e6,0,1e9),
     P("min_count","Min Count","int",1,0,10000),
     P("max_count","Max Count","int",1000,0,10000)],
    proc_find_contours, ""),

  # ── CALIBRATION ─────────────────────────────────────────────────
  ToolDef("calibrate_grid","Calibrate (Checkerboard)","Calibration",
    "Hiệu chỉnh camera từ checkerboard — TCalibCheckerboardTool",
    "#1a472a","📋",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("calibrated","bool"),
     PortDef("pixel_to_mm","number"),PortDef("rms_error","number")],
    [P("grid_cols","Grid Cols","int",9,2,30,tooltip="Số góc nội (cols-1)"),
     P("grid_rows","Grid Rows","int",6,2,30),
     P("square_size_mm","Square Size (mm)","float",25.4,0.1,1000,step=0.1),
     P("downscale","Coarse Downscale","int",0,0,16,
       tooltip="0=auto. findChessboardCorners coarse trên ảnh nhỏ, cornerSubPix refine full-res → cùng độ chính xác sub-pixel.")],
    proc_calibrate_grid, "TCalibCheckerboardTool"),

  # ── LOGIC & FLOW ────────────────────────────────────────────────
  ToolDef("logic_and","AND Gate","Logic & Flow","Logic AND","#1c1c2e","∧",
    [PortDef("A","bool"),PortDef("B","bool")],[PortDef("result","bool")],
    [],proc_logic_and,""),

  ToolDef("logic_or","OR Gate","Logic & Flow","Logic OR","#1c1c2e","∨",
    [PortDef("A","bool"),PortDef("B","bool")],[PortDef("result","bool")],
    [],proc_logic_or,""),

  ToolDef("logic_not","NOT Gate","Logic & Flow","Logic NOT","#1c1c2e","¬",
    [PortDef("A","bool")],[PortDef("result","bool")],
    [],proc_logic_not,""),

  ToolDef("compare","Compare","Logic & Flow",
    "So sánh 2 giá trị số","#1c1c2e","⚖",
    [PortDef("A","number"),PortDef("B","number",required=False)],
    [PortDef("result","bool"),PortDef("pass","bool")],
    [P("operator","Operator","enum",">=",choices=["==","!=",">",">=","<","<="]),
     P("value","Value B (if not connected)","float",0,-1e9,1e9)],
    proc_compare,""),

  ToolDef("judge","Pass/Fail Judge","Logic & Flow",
    "Kết hợp điều kiện → PASS/FAIL cuối cùng. Mặc định 2 input A, B; "
    "chuột phải node → ➕ Add Input để thêm port mới. ALL = tất cả input True; "
    "ANY = ít nhất 1 True. Input chưa nối được bỏ qua.",
    "#1c1c2e","🏁",
    [PortDef("A","bool",required=False),PortDef("B","bool",required=False)],
    [PortDef("result","bool"),PortDef("pass","bool")],
    [P("mode","Combine Mode","enum","ALL",choices=["ALL","ANY"])],
    proc_judge,"",
    extra_input_type="bool"),

  ToolDef("script","Script Tool","Logic & Flow",
    "Chạy Python code tùy biến với editor có syntax-highlight + gợi ý. "
    "Right-click node để Add Input / Add Output port. Trong code: input là "
    "biến lowercase (A→a, B→b...), gán `result = ...`, `pass_value = ...` "
    "hoặc `outputs['port_name'] = ...` để set output.",
    "#1c1c2e","🐍",
    [PortDef("A","any",required=False)],
    [PortDef("pass","bool"), PortDef("result","any")],
    [P("expression","Python Code","str",
       "# Inputs: a, b, c, ... (lowercase của port name)\n"
       "# Outputs: gán biến cùng tên port, hoặc outputs['name'] = ...\n"
       "# 'pass' là keyword Python → dùng pass_value\n"
       "result = a",
       tooltip="Mở Script Tool detail (double-click node) để dùng editor "
               "đầy đủ với syntax highlighting + autocomplete.")],
    proc_script,"TScriptTool",
    extra_input_type="any", extra_output_type="any"),

  # ── OUTPUT / DISPLAY ────────────────────────────────────────────
  ToolDef("display","Display","Output & Display",
    "Annotate & hiển thị ảnh kết quả — TRecordDisplayTool",
    "#0d1117","🖥",
    [PortDef("image","image"),PortDef("pass","bool",required=False)],
    [PortDef("image","image")],
    [P("label","Label Text","str",""),P("tx","Text X","int",10,0,8192),
     P("ty","Text Y","int",30,0,8192),P("font_scale","Font Scale","float",0.8,0.1,5,step=0.1),
     P("show_result","Show PASS/FAIL","bool",True)],
    proc_display,"TRecordDisplayTool"),

  ToolDef("message","Message","Output & Display",
    "Hiển thị message khác nhau theo port pass (PASS/FAIL/NONE). "
    "Có thể nối port image của tool khác để overlay lên ảnh.",
    "#0d1117","💬",
    [PortDef("pass","bool",required=False),
     PortDef("image","image",required=False)],
    [PortDef("image","image"),PortDef("text","any"),PortDef("pass","bool")],
    [P("msg_pass","Text khi PASS","str","PASS"),
     P("msg_fail","Text khi FAIL","str","FAIL"),
     P("msg_none","Text khi chưa có port","str","NO INPUT"),
     P("color_pass","Color PASS","enum","Green",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("color_fail","Color FAIL","enum","Red",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("color_none","Color NONE","enum","Yellow",
       choices=["Yellow","Cyan","Green","Red","White","Magenta","Orange","Blue"]),
     P("position","Anchor Position","enum","Top-Left",
       choices=["Top-Left","Top-Center","Top-Right","Center",
                "Bottom-Left","Bottom-Center","Bottom-Right"],
       tooltip="Vị trí gốc. Offset X/Y dưới đây dịch tiếp từ điểm này — "
               "hoặc kéo text trực tiếp trên ảnh để sync slider."),
     P("label_dx","Offset X","int",0,-2000,2000,use_slider=True,
       tooltip="Dịch text theo trục X từ anchor. Có thể kéo text trên ảnh."),
     P("label_dy","Offset Y","int",0,-2000,2000,use_slider=True,
       tooltip="Dịch text theo trục Y từ anchor. Có thể kéo text trên ảnh."),
     P("font","Font","enum","Duplex",
       choices=["Simplex","Plain","Duplex","Complex","Triplex",
                "Script Simplex","Script Complex"]),
     P("font_size","Font Size","float",1.2,0.2,5.0,step=0.05,use_slider=True),
     P("thickness","Thickness","int",3,1,12,use_slider=True),
     P("show_background","Show Background","bool",True,
       tooltip="Vẽ nền đen sau text để dễ đọc.")],
    proc_message, ""),

  ToolDef("save_image","Save Image","Output & Display",
    "Lưu ảnh ra file — TSaveImageTool","#0d1117","💾",
    [PortDef("image","image")],
    [PortDef("saved","bool"),PortDef("path","any")],
    [P("save_path","Save Path","str","output/result.png"),
     P("timestamp","Add Timestamp","bool",True)],
    proc_save_image,"TSaveImageTool"),

  ToolDef("csv_log","CSV Logger","Output & Display",
    "Ghi kết quả vào CSV log file","#0d1117","📊",
    [PortDef("pass","bool",required=False),
     PortDef("value_a","any",required=False),PortDef("value_b","any",required=False),
     PortDef("value_c","any",required=False)],
    [PortDef("logged","bool"),PortDef("path","any")],
    [P("csv_path","CSV Path","str","log/results.csv")],
    proc_csv_log,""),

  ToolDef("yield_stats","Yield Statistics","Output & Display",
    "Tích lũy PASS/FAIL/TOTAL/Yield Rate qua nhiều lần chạy pipeline. "
    "Nối port pass từ Pass/Fail Judge (hoặc bất kỳ tool nào có output pass) "
    "→ counter cộng dồn mỗi lần Run. Panel Inspection Results sẽ đọc trực tiếp "
    "output của tool này nếu có trong pipeline; bấm Clear Log để reset.",
    "#0d1117","📈",
    [PortDef("pass","bool",required=False)],
    [PortDef("pass","bool"),
     PortDef("pass_count","number"),
     PortDef("fail_count","number"),
     PortDef("total_count","number"),
     PortDef("yield_rate","number")],
    [P("reset","Reset on Next Run","bool",False,
       tooltip="Tick + Run pipeline → reset counter về 0 ngay lần chạy đó.")],
    proc_yield_stats,""),

  # ── YOLO DETECTION ─────────────────────────────────────────────
  ToolDef("yolo_detect","YOLO Detect","YOLO",
    "YOLOv8/v11 Object Detection & Segmentation. Add file model train sẵn "
    "(.pt hoặc .onnx) rồi chỉnh threshold predict bên dưới. "
    "Train mới: mở YOLO Studio từ toolbar. Outputs x/y/w/h là best "
    "detection (max conf) để chain vào Crop ROI / Fixture. "
    "Right-click node → ➕ Add / Manage Output Terminals để thêm port "
    "x_2, cy_3, conf_5... cho per-object access.",
    "#1a0a3a","🤖",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("detections","any"),
     PortDef("count","number"),PortDef("pass","bool"),
     PortDef("x","number"),PortDef("y","number"),
     PortDef("w","number"),PortDef("h","number"),
     PortDef("cx","number"),PortDef("cy","number"),
     PortDef("confidence","number"),PortDef("class","any"),
     PortDef("cls_id","number")],
    [P("model_path","Model File","str","",
       tooltip="File .pt (PyTorch) hoặc .onnx (fast CPU inference). "
               "Có .onnx cùng tên cạnh .pt thì auto-pick .onnx.",
       file_filter="YOLO Models (*.pt *.onnx);;PyTorch (*.pt);;"
                    "ONNX (*.onnx);;All Files (*)"),
     P("confidence","Confidence","float",0.5,0.01,1.0,step=0.01),
     P("iou","IoU Threshold","float",0.45,0.01,1.0,step=0.01),
     P("imgsz","Image Size","int",640,32,4096,step=32),
     P("max_det","Max Detections","int",300,1,1000),
     P("min_count","Min Objects (PASS)","int",1,0,1000),
     P("max_count","Max Objects (PASS)","int",9999,0,10000),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ '{class} {conf}' cạnh từng detection lên ảnh output. Mặc định tắt — kết quả vẫn có trong log & detections port."),
     P("seg_show_bbox","Seg: draw bbox instead of polygon","bool",True,
       tooltip="Seg model: vẽ bbox bao polygon (clean) thay vì polyline polygon (zigzag). Bbox detection không bị ảnh hưởng."),
     P("fill_segments","Seg: fill polygon (30% alpha)","bool",False,
       tooltip="Seg model: tô màu nửa-trong-suốt bên trong polygon. Mặc định OFF — chỉ vẽ outline cho ảnh sạch.")],
    proc_yolo_detect,"ultralytics YOLO",
    terminal_fields=["x","y","w","h","cx","cy","x1","y1","x2","y2",
                      "conf","class","cls_id"],
    terminal_source_key="detections"),

]

TOOL_BY_ID: Dict[str,ToolDef] = {t.tool_id: t for t in TOOL_REGISTRY}
CATEGORIES = [
    "Acquire Image","Pattern Find","Fixture","Caliper",
    "Blob Analysis","Edge & Geometry","Color Analysis","ID & Read",
    "Measurement","Surface Inspection","Image Processing",
    "Calibration","Logic & Flow","Output & Display","YOLO"
]
