"""
core/tool_registry.py — Cognex VisionPro Style
Tools mô phỏng Cognex VisionPro: PatMax, Caliper, Blob, Edge, Color,
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
    cognex_equiv: str = ""   # Tên tool tương đương trong Cognex


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

def proc_acquire_image(inputs, params):
    """CogAcqFifoTool — Acquire image từ file đơn hoặc folder (frame index).

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
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            # Auto-advance: lần Run kế tiếp sẽ sang ảnh kế tiếp (cycle)
            if params.get("auto_advance", True):
                params["frame_index"] = (idx + 1) % len(files)
            return {"image": img, "width": w, "height": h,
                    "acquired": True, "frame_number": idx,
                    "file_name": files[idx], "frame_count": len(files)}

    path = params.get("file_path", "")
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
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
    """CogAcqFifoTool (Camera) — Capture từ OpenCV/USB hoặc HikRobot/Do3think MVS."""
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

def _build_patmax_objects(results, model):
    """Build list `objects` từ results.
    Mỗi object: x/y = origin chính + ref{N}_x, ref{N}_y, ref{N}_angle cho
    từng extra ref (đã transform theo angle/scale/scale của result).
    Cũng có "refs": [{name, x, y, angle}, ...] để UI duyệt.
    """
    from core.patmax_engine import transform_ref_to_image
    objs = []
    extras = list(getattr(model, "extra_refs", []) or []) if model else []
    for r in results:
        obj = {"x": r.origin_x, "y": r.origin_y, "score": r.score,
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
            name = term.get("name") or f"{field}_{obj_idx}"
            if 0 <= obj_idx < len(objects):
                out[name] = objects[obj_idx].get(field, 0.0)
            else:
                out[name] = 0.0
        except Exception:
            continue


def proc_patmax(inputs, params):
    """
    CogPatMaxPatternAlignTool — dùng PatMaxEngine.
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

    clean = _bgr(img.copy())
    vis = draw_patmax_results(clean, results, model,
                                show_xy=show_xy, show_bbox=show_bbox)

    objects = _build_patmax_objects(results, model)
    if results:
        r = results[0]
        out = {"image": clean, "_display_image": vis,
               "found": True, "score": r.score,
               "x": r.origin_x, "y": r.origin_y,
               "angle": r.angle, "scale": r.scale,
               "num_found": len(results), "objects": objects}
    else:
        out = {"image": clean, "_display_image": vis,
               "found": False, "score": 0.0,
               "x": 0.0, "y": 0.0, "angle": 0.0, "scale": 1.0,
               "num_found": 0, "objects": []}
    _apply_extra_terminals(out, objects, params)
    return out


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
    PatMax Align Tool — dispatch theo Algorithm + Train Mode (Cognex-style
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
    clean = _bgr(img.copy())
    vis = draw_patmax_results(clean, results, model,
                                show_xy=show_xy, show_bbox=show_bbox)
    objects = _build_patmax_objects(results, model)
    if results:
        r = results[0]
        out = {"image": clean, "_display_image": vis,
               "found": True, "score": r.score,
               "x": r.origin_x, "y": r.origin_y,
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
    CogPMAlignTool — PatFind nhanh (NCC), dùng chung PatMaxEngine nhưng không xoay.
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
    CogFixtureTool — Thiết lập hệ tọa độ dựa trên PatMax result.
    Giúp các tool sau bất biến với vị trí/góc của part.
    """
    img   = inputs.get("image")
    ref_x = float(inputs.get("ref_x", params.get("origin_x", 0)))
    ref_y = float(inputs.get("ref_y", params.get("origin_y", 0)))
    angle = float(inputs.get("ref_angle", inputs.get("angle", 0)))

    if img is None:
        return {"image":None,"transform_matrix":None,"offset_x":0.0,"offset_y":0.0,"angle":0.0}

    h,w = img.shape[:2]
    cx = params.get("origin_x", w/2); cy = params.get("origin_y", h/2)
    dx = ref_x - cx; dy = ref_y - cy

    # Build transform
    M  = cv2.getRotationMatrix2D((ref_x, ref_y), -angle, 1.0)
    M[0,2] += 0; M[1,2] += 0
    warped = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(30,30,30))

    vis = warped.copy()
    s = _draw_scale(vis)
    # Draw coordinate axes
    ax = int(w/2); ay = int(h/2)
    axis_len = int(60 * s)
    cv2.arrowedLine(vis,(ax,ay),(ax+axis_len,ay),(0,80,255),_t(2,s),tipLength=0.2)
    cv2.arrowedLine(vis,(ax,ay),(ax,ay-axis_len),(0,220,80),_t(2,s),tipLength=0.2)
    print(f"[Fixture] dx={dx:.1f} dy={dy:.1f} angle={angle:.1f}deg")

    return {"image":vis,"transform_matrix":M.tolist(),
            "offset_x":float(dx),"offset_y":float(dy),"angle":float(angle)}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: CALIPER
# ═══════════════════════════════════════════════════════════════════

def proc_caliper(inputs, params):
    """
    CogCaliperTool — Đo cạnh (edge) chính xác sub-pixel,
    đo khoảng cách giữa 2 cạnh (Width measurement).
    """
    img = inputs.get("image")
    if img is None:
        return {"image":None,"edge1_pos":0.0,"edge2_pos":0.0,
                "width":0.0,"pass":False,"edges_found":0}

    gray = _gray(img); vis = _bgr(img.copy())
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))

    # ROI line từ params
    x1 = params.get("x1", img.shape[1]//4)
    y1 = params.get("y1", img.shape[0]//2)
    x2 = params.get("x2", img.shape[1]*3//4)
    y2 = params.get("y2", img.shape[0]//2)
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

    # Gaussian derivative (Cognex-style edge filter)
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
    """CogCaliperTool (Multi-edge) — Tìm tất cả cạnh trong vùng."""
    img = inputs.get("image")
    if img is None:
        return {"image":None,"edges":[],"count":0,"pass":False}
    gray = _gray(img); vis = _bgr(img.copy())
    s = _draw_scale(vis)
    x1=params.get("x1",0); y1=params.get("y1",img.shape[0]//2)
    x2=params.get("x2",img.shape[1]); y2=params.get("y2",img.shape[0]//2)
    length=int(math.hypot(x2-x1,y2-y1))
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
    CogBlobTool — Phân tích vùng (blob) toàn diện:
    diện tích, chu vi, circularity, bounding box, centroid, orientation.
    """
    img  = inputs.get("image")
    mask = inputs.get("mask")
    if img is None:
        return {"image":None,"count":0,"pass":False,"total_area":0.0,
                "blobs":[],"centroids":[]}

    gray = _gray(img)
    if mask is None:
        thresh_val = params.get("threshold", 128)
        inv = params.get("invert", False)
        t   = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        if params.get("auto_threshold", True):
            t |= cv2.THRESH_OTSU; thresh_val = 0
        _, mask = cv2.threshold(gray, thresh_val, 255, t)

    # Morphology cleanup
    k = params.get("morph_open", 0)
    if k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    k2 = params.get("morph_close", 0)
    if k2 > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k2,k2))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale    = params.get("pixel_to_mm2", 1.0)
    min_a    = params.get("min_area", 50.0)
    max_a    = params.get("max_area", 1e8)
    min_circ = params.get("min_circularity", 0.0)
    max_circ = params.get("max_circularity", 1.1)
    min_elo  = params.get("min_elongation", 0.0)
    max_elo  = params.get("max_elongation", 1000.0)

    vis = _bgr(img.copy())
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))
    blobs = []; centroids = []; total_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_a or area > max_a: continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4*math.pi*area/(perimeter**2)) if perimeter>0 else 0
        if not (min_circ <= circularity <= max_circ): continue

        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]

        # Bounding box & orientation
        rect = cv2.minAreaRect(cnt)
        (bx,by),(bw,bh),angle_deg = rect
        elongation = max(bw,bh)/max(min(bw,bh),0.001)
        if not (min_elo <= elongation <= max_elo): continue

        # Convex hull & convexity
        hull       = cv2.convexHull(cnt)
        hull_area  = cv2.contourArea(hull)
        convexity  = area/hull_area if hull_area>0 else 0

        area_mm = area * scale
        total_area += area_mm

        blob_info = {
            "area":area_mm,"perimeter":perimeter*math.sqrt(scale),
            "circularity":circularity,"elongation":elongation,
            "convexity":convexity,"cx":float(cx),"cy":float(cy),
            "angle":float(angle_deg),"bbox_w":float(bw),"bbox_h":float(bh)
        }
        blobs.append(blob_info)
        centroids.append((float(cx),float(cy)))

        # Draw
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(vis,[cnt],-1,(0,200,255),_t(2,s))
        cv2.drawContours(vis,[box],-1,(255,180,0),_t(1,s))
        cv2.circle(vis,(int(cx),int(cy)),_t(4,s),(0,255,80),-1)
        if show_labels:
            cv2.putText(vis,f"{area_mm:.1f}mm²",(int(cx)+int(6*s),int(cy)-int(6*s)),
                        cv2.FONT_HERSHEY_SIMPLEX,_fs(0.45,s),(0,200,255),_t(1,s))

    min_cnt = params.get("min_count", 1)
    max_cnt = params.get("max_count", 1000)
    is_pass = min_cnt <= len(blobs) <= max_cnt
    print(f"[Blob] count={len(blobs)} total_area={total_area:.2f}mm² {'PASS' if is_pass else 'FAIL'}")

    return {"image":vis,"count":len(blobs),"pass":is_pass,
            "total_area":total_area,"blobs":blobs,"centroids":centroids}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: EDGE / LINE FIND
# ═══════════════════════════════════════════════════════════════════

def proc_find_line(inputs, params):
    """CogFindLineTool — Tìm đường thẳng từ các điểm edge (least-squares)."""
    img = inputs.get("image")
    if img is None:
        return {"image":None,"found":False,"angle":0.0,"distance":0.0,
                "point_x":0.0,"point_y":0.0,"pass":False}
    gray = _gray(img); vis = _bgr(img.copy())
    s = _draw_scale(vis)
    h,w  = gray.shape
    t1   = params.get("canny_low",50); t2=params.get("canny_high",150)
    edges= cv2.Canny(gray,t1,t2)

    # ROI band
    rx1=params.get("x1",0); ry1=params.get("y1",h//2-30)
    rx2=params.get("x2",w); ry2=params.get("y2",h//2+30)
    roi_mask=np.zeros_like(edges)
    roi_mask[ry1:ry2,rx1:rx2]=255
    edges=cv2.bitwise_and(edges,roi_mask)

    pts=np.column_stack(np.where(edges>0))  # (y,x)
    found=False; angle=0.0; dist=0.0; px=float(w/2); py=float(h/2)
    if len(pts)>5:
        xs=pts[:,1].astype(float); ys=pts[:,0].astype(float)
        [vx,vy,x0,y0]=cv2.fitLine(np.column_stack([xs,ys]),cv2.DIST_L2,0,0.01,0.01)
        angle=float(math.degrees(math.atan2(float(vy),float(vx))))
        px=float(x0); py=float(y0)
        # Draw line
        t_range=max(w,h)*2
        pt1=(int(px-vx*t_range),int(py-vy*t_range))
        pt2=(int(px+vx*t_range),int(py+vy*t_range))
        cv2.line(vis,pt1,pt2,(0,220,80),_t(2,s))
        cv2.circle(vis,(int(px),int(py)),_t(6,s),(0,220,80),-1)
        # Distance from image center
        dist=float(math.hypot(px-w/2,py-h/2))*params.get("pixel_to_mm",1.0)
        found=True

    cv2.rectangle(vis,(rx1,ry1),(rx2,ry2),(0,150,200),_t(1,s))
    ang_min=params.get("min_angle",-180.0); ang_max=params.get("max_angle",180.0)
    is_pass=found and (ang_min<=angle<=ang_max)
    print(f"[FindLine] angle={angle:.2f}deg {'PASS' if is_pass else ('FAIL' if found else 'NOT FOUND')}")
    return {"image":vis,"found":found,"angle":angle,"distance":dist,
            "point_x":px,"point_y":py,"pass":is_pass}

def proc_find_circle(inputs, params):
    """CogFindCircleTool — Tìm & fit đường tròn chính xác."""
    img = inputs.get("image")
    if img is None:
        return {"image":None,"found":False,"cx":0.0,"cy":0.0,
                "radius":0.0,"pass":False}
    gray=_gray(img); vis=_bgr(img.copy())
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))
    blurred=cv2.GaussianBlur(gray,(9,9),2)
    circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,
        params.get("dp",1.2),params.get("min_dist",30),
        param1=params.get("param1",100),param2=params.get("param2",30),
        minRadius=params.get("min_radius",5),maxRadius=params.get("max_radius",300))
    found=False; cx=0.0; cy=0.0; radius=0.0
    if circles is not None:
        circles=np.uint16(np.around(circles))
        c=circles[0][0]
        cx=float(c[0]); cy=float(c[1]); radius=float(c[2])
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
    print(f"[FindCircle] {'PASS' if is_pass else ('FAIL' if found else 'NOT FOUND')} r={radius:.2f}px")
    return {"image":vis,"found":found,"cx":cx,"cy":cy,
            "radius":radius*params.get("pixel_to_mm",1.0),"pass":is_pass}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: COLOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def proc_color_picker(inputs, params):
    """CogColorTool (Picker) — Click chuột lấy màu → xuất HSV/RGB range."""
    img=inputs.get("image")
    if img is None:
        return {"image":None,"color_hsv":None,"h":0,"s":0,"v":0,"r":0,"g":0,"b":0}
    bgr=_bgr(img); x=params.get("pick_x",0); y=params.get("pick_y",0)
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

def proc_color_segment(inputs, params):
    """CogColorSegmenterTool — Phân đoạn màu HSV, xuất mask + ratio."""
    img=inputs.get("image")
    if img is None:
        return {"image":None,"mask":None,"pass":False,"pixel_ratio":0.0,"pixel_count":0}
    picked=inputs.get("color_hsv",None)
    if picked and isinstance(picked,(list,tuple)) and len(picked)==6:
        h_lo,s_lo,v_lo,h_hi,s_hi,v_hi=[int(x) for x in picked]
    else:
        h_lo=params.get("h_low",0);   h_hi=params.get("h_high",180)
        s_lo=params.get("s_low",50);  s_hi=params.get("s_high",255)
        v_lo=params.get("v_low",50);  v_hi=params.get("v_high",255)
    tol=params.get("tolerance",0)
    h_lo=max(0,h_lo-tol); h_hi=min(180,h_hi+tol)
    s_lo=max(0,s_lo-tol); s_hi=min(255,s_hi+tol)
    hsv=cv2.cvtColor(_bgr(img),cv2.COLOR_BGR2HSV)

    # Handle hue wrap-around (e.g. red: 0-10 & 170-180)
    if h_lo <= h_hi:
        mask=cv2.inRange(hsv,np.array([h_lo,s_lo,v_lo]),np.array([h_hi,s_hi,v_hi]))
    else:
        m1=cv2.inRange(hsv,np.array([0,s_lo,v_lo]),np.array([h_hi,s_hi,v_hi]))
        m2=cv2.inRange(hsv,np.array([h_lo,s_lo,v_lo]),np.array([180,s_hi,v_hi]))
        mask=cv2.bitwise_or(m1,m2)

    k=params.get("morph_open",0)
    if k>0:
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    cnt=int(np.count_nonzero(mask)); ratio=cnt/mask.size
    min_r=params.get("min_ratio",0.01); max_r=params.get("max_ratio",1.0)
    is_pass=min_r<=ratio<=max_r
    vis=_bgr(img.copy()); overlay=vis.copy()
    overlay[mask>0]=[0,220,80]; cv2.addWeighted(vis,0.55,overlay,0.45,0,vis)
    print(f"[ColorSeg] ratio={ratio:.3f} pixels={cnt} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"mask":mask,"pass":is_pass,"pixel_ratio":ratio,"pixel_count":cnt}

def proc_color_match(inputs, params):
    """CogColorMatchTool — So khớp màu trung bình trong ROI với màu tham chiếu."""
    img=inputs.get("image")
    if img is None:
        return {"image":None,"pass":False,"delta_e":0.0,"mean_r":0,"mean_g":0,"mean_b":0}
    bgr=_bgr(img)
    x=params.get("x",0); y=params.get("y",0)
    w=params.get("w",50); h=params.get("h",50)
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
    """CogIDReaderTool — Đọc Barcode 1D/2D, QR, DataMatrix."""
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

def proc_ocr_max(inputs, params):
    """CogOCRMaxTool — Đọc & xác nhận ký tự (OCR)."""
    img=inputs.get("image")
    if img is None: return {"image":None,"text":"","pass":False,"confidence":0.0}
    gray=_gray(img); vis=_bgr(img.copy())
    s = _draw_scale(vis)
    try:
        import pytesseract
        data=pytesseract.image_to_data(gray,lang=params.get("lang","eng"),
                                        config=f"--psm {params.get('psm',6)} --oem 3",
                                        output_type=pytesseract.Output.DICT)
        text=""; conf=0.0; words=0
        for i,t in enumerate(data["text"]):
            c=int(data["conf"][i])
            if c>0 and t.strip():
                text+=t+" "; conf+=c; words+=1
                x2,y2,w2,h2=data["left"][i],data["top"][i],data["width"][i],data["height"][i]
                cv2.rectangle(vis,(x2,y2),(x2+w2,y2+h2),(0,200,255),_t(1,s))
        text=text.strip(); conf=conf/max(words,1)
    except: text="[pytesseract N/A]"; conf=0.0
    expected=params.get("expected_text","")
    min_conf=params.get("min_confidence",60.0)
    is_pass=(expected in text if expected else bool(text)) and conf>=min_conf
    print(f"[OCR] text='{text}' conf={conf:.1f}% {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"text":text,"pass":is_pass,"confidence":conf}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: MEASUREMENT
# ═══════════════════════════════════════════════════════════════════

def proc_distance_point(inputs, params):
    """CogDistancePointPointTool — Đo khoảng cách 2 điểm."""
    img=inputs.get("image")
    x1=float(inputs.get("x1",params.get("x1",0)))
    y1=float(inputs.get("y1",params.get("y1",0)))
    x2=float(inputs.get("x2",params.get("x2",100)))
    y2=float(inputs.get("y2",params.get("y2",0)))
    dist=math.hypot(x2-x1,y2-y1)*params.get("pixel_to_mm",1.0)
    vis=_bgr(img.copy()) if img is not None else np.zeros((200,400,3),dtype=np.uint8)
    s = _draw_scale(vis)
    show_labels = bool(params.get("show_labels", False))
    cv2.line(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,220,255),_t(2,s))
    cv2.circle(vis,(int(x1),int(y1)),_t(5,s),(0,200,255),-1)
    cv2.circle(vis,(int(x2),int(y2)),_t(5,s),(0,200,255),-1)
    mx,my=int((x1+x2)/2),int((y1+y2)/2)
    if show_labels:
        cv2.putText(vis,f"{dist:.3f}mm",(mx,my-int(10*s)),cv2.FONT_HERSHEY_SIMPLEX,_fs(0.6,s),(0,220,255),_t(2,s))
    is_pass=params.get("min_dist",0.0)<=dist<=params.get("max_dist",9999.0)
    print(f"[Distance] {dist:.3f}mm {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"distance":dist,"pass":is_pass}

def proc_angle_lines(inputs, params):
    """CogAngleLineLineTool — Đo góc giữa 2 đường thẳng."""
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
    """CogImageConvertTool — Chuyển đổi định dạng ảnh."""
    img=inputs.get("image")
    if img is None: return {"image":None}
    mode=params.get("mode","Grayscale")
    if mode=="Grayscale":      out=_bgr(_gray(img))
    elif mode=="BGR to RGB":   out=cv2.cvtColor(_bgr(img),cv2.COLOR_BGR2RGB)
    elif mode=="Invert":       out=cv2.bitwise_not(_bgr(img))
    elif mode=="HSV":          out=cv2.cvtColor(_bgr(img),cv2.COLOR_BGR2HSV)
    elif mode=="LAB":          out=cv2.cvtColor(_bgr(img),cv2.COLOR_BGR2LAB)
    elif mode=="YCrCb":        out=cv2.cvtColor(_bgr(img),cv2.COLOR_BGR2YCrCb)
    else:                      out=_bgr(img)
    return {"image":out}

def proc_sharpen(inputs, params):
    img=inputs.get("image")
    if img is None: return {"image":None}
    s=params.get("strength",1.0)
    k=np.array([[0,-1,0],[-1,4+s,-1],[0,-1,0]],dtype=np.float32)
    return {"image":np.clip(cv2.filter2D(img,-1,k),0,255).astype(np.uint8)}

def proc_morphology(inputs, params):
    img=inputs.get("image")
    if img is None: return {"image":None,"mask":None}
    op={"Erode":cv2.MORPH_ERODE,"Dilate":cv2.MORPH_DILATE,"Open":cv2.MORPH_OPEN,
        "Close":cv2.MORPH_CLOSE,"Gradient":cv2.MORPH_GRADIENT,
        "Top Hat":cv2.MORPH_TOPHAT,"Black Hat":cv2.MORPH_BLACKHAT}.get(params.get("operation","Open"),cv2.MORPH_OPEN)
    sh={"Rect":cv2.MORPH_RECT,"Ellipse":cv2.MORPH_ELLIPSE,"Cross":cv2.MORPH_CROSS}.get(params.get("shape","Ellipse"),cv2.MORPH_ELLIPSE)
    k=max(1,params.get("kernel_size",3))
    kernel=cv2.getStructuringElement(sh,(k,k))
    g=_gray(img)
    result=cv2.morphologyEx(g,op,kernel,iterations=max(1,params.get("iterations",1)))
    return {"image":_bgr(result),"mask":result}

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

    # Default từ _drawn_roi (vẽ tay) hoặc params spinbox
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

    # Per-port override
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

    # `image` output = ảnh GỐC pass-through (clean, không vẽ gì) —
    #                  để downstream xử lý trên ảnh sạch.
    # `display_image` output = ảnh GỐC + bounding box overlay —
    #                          cho node panel hiển thị vị trí ROI.
    clean = _bgr(img.copy())
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
    """Phát hiện khuyết tật bề mặt — so sánh với reference hoặc model thống kê."""
    img=inputs.get("image"); ref=inputs.get("reference")
    if img is None: return {"image":None,"pass":False,"defect_area":0,"defect_count":0}
    bgr=_bgr(img)
    if ref is not None:
        r2=_bgr(ref)
        if r2.shape==bgr.shape: diff=cv2.absdiff(_gray(bgr),_gray(r2))
        else: g=_gray(bgr); diff=cv2.absdiff(g,cv2.GaussianBlur(g,(21,21),0))
    else:
        g=_gray(bgr); diff=cv2.absdiff(g,cv2.GaussianBlur(g,(21,21),0))
    _,mask=cv2.threshold(diff,params.get("threshold",30),255,cv2.THRESH_BINARY)
    k=params.get("morph_k",3)
    if k>0:
        ker=np.ones((k,k),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,ker)
        mask=cv2.dilate(mask,ker)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    min_a=params.get("min_defect_px",10)
    cnts=[c for c in cnts if cv2.contourArea(c)>=min_a]
    defect_area=int(sum(cv2.contourArea(c) for c in cnts))
    max_a=params.get("max_defect_area",1000); max_c=params.get("max_defect_count",0)
    is_pass=defect_area<=max_a and len(cnts)<=max(max_c,0 if max_c==0 else 999)
    vis=bgr.copy()
    s = _draw_scale(vis)
    cv2.drawContours(vis,cnts,-1,(0,60,255),_t(2,s))
    for c in cnts:
        x2,y2,w2,h2=cv2.boundingRect(c)
        cv2.rectangle(vis,(x2,y2),(x2+w2,y2+h2),(0,60,255),_t(1,s))
    print(f"[SurfaceDefect] area={defect_area}px² count={len(cnts)} {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"pass":is_pass,"defect_area":defect_area,"defect_count":len(cnts)}

def proc_scratch_detect(inputs, params):
    """Phát hiện vết xước dạng đường thẳng dài."""
    img=inputs.get("image")
    if img is None: return {"image":None,"pass":False,"scratch_count":0,"total_length":0.0}
    gray=_gray(img); vis=_bgr(img.copy())
    s = _draw_scale(vis)
    k=params.get("blur_k",3)
    if k>0: gray=cv2.GaussianBlur(gray,(k,k),0)
    edges=cv2.Canny(gray,params.get("canny_low",30),params.get("canny_high",100))
    min_len=params.get("min_scratch_length",50); max_gap=params.get("max_gap",5)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,params.get("hough_thresh",30),
                           minLineLength=min_len,maxLineGap=max_gap)
    scratches=[]; total_len=0.0
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2=l[0]
            length=math.hypot(x2-x1,y2-y1)
            if length>=min_len:
                scratches.append((x1,y1,x2,y2,length))
                total_len+=length
                cv2.line(vis,(x1,y1),(x2,y2),(0,60,255),_t(2,s))
    max_s=params.get("max_scratches",0)
    is_pass=len(scratches)<=max_s
    print(f"[Scratch] count={len(scratches)} total_len={total_len:.0f}px {'PASS' if is_pass else 'FAIL'}")
    return {"image":vis,"pass":is_pass,"scratch_count":len(scratches),"total_length":total_len}


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
    """CogCalibCheckerboardTool — Hiệu chỉnh camera từ checkerboard."""
    img=inputs.get("image")
    if img is None: return {"image":None,"calibrated":False,"pixel_to_mm":1.0,"rms_error":0.0}
    gray=_gray(img); vis=_bgr(img.copy())
    cols=params.get("grid_cols",9); rows=params.get("grid_rows",6)
    ret,corners=cv2.findChessboardCorners(gray,(cols,rows),None)
    if ret:
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
        corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        cv2.drawChessboardCorners(vis,(cols,rows),corners,ret)
        # Estimate pixel/mm from square size
        if len(corners)>=2:
            p1=corners[0][0]; p2=corners[1][0]
            px_per_square=float(math.hypot(p2[0]-p1[0],p2[1]-p1[1]))
            mm_per_square=params.get("square_size_mm",25.4)
            px_to_mm=mm_per_square/max(px_per_square,0.001)
        else: px_to_mm=1.0
        print(f"[Calibrate] {px_to_mm:.5f} mm/px")
        return {"image":vis,"calibrated":True,"pixel_to_mm":px_to_mm,"rms_error":0.0}
    print("[Calibrate] Checkerboard NOT found")
    return {"image":vis,"calibrated":False,"pixel_to_mm":1.0,"rms_error":0.0}


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
    conds=[bool(inputs.get(k)) for k in ("A","B","C","D") if inputs.get(k) is not None]
    result=(all(conds) if params.get("mode","ALL")=="ALL" else any(conds)) if conds else False
    return {"result":result,"pass":result}

def proc_script(inputs,params):
    """CogScriptTool — Chạy Python expression tùy chỉnh."""
    expr=params.get("expression","result = True")
    ctx={"inputs":inputs,"params":params,"result":False,"pass_value":False,**inputs}
    try: exec(expr,ctx)
    except Exception as e: ctx["result"]=False; ctx["error"]=str(e)
    return {"result":bool(ctx.get("result",False)),
            "pass":bool(ctx.get("pass_value",ctx.get("result",False))),
            "output":str(ctx.get("output",""))}


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY: OUTPUT / DISPLAY
# ═══════════════════════════════════════════════════════════════════

def proc_display(inputs,params):
    """CogRecordDisplayTool — Annotate & display image."""
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

def proc_save_image(inputs,params):
    """CogSaveImageTool — Lưu ảnh ra file."""
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
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════
P = ParamDef  # shorthand


# ═══════════════════════════════════════════════════════════════
#  YOLO DETECTION
# ═══════════════════════════════════════════════════════════════
def proc_yolo_detect(inputs, params):
    """
    YOLOv8/v11 Object Detection & Segmentation.
    Model .pt chọn từ YOLO Studio hoặc nhập đường dẫn.
    """
    img = inputs.get("image")
    if img is None:
        return {"image": None, "detections": [], "count": 0, "pass": False}

    model_path = params.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        vis = _bgr(img.copy())
        print("[YOLO] No model — open YOLO Studio to train (right-click node)")
        return {"image": vis, "detections": [], "count": 0, "pass": False}

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
        vis = _bgr(img.copy())
        s = _draw_scale(vis)
        show_labels = bool(params.get("show_labels", False))
        detections = []

        for result in results:
            names = result.names
            # Segmentation masks
            if result.masks is not None:
                for seg, cls_id, conf_val in zip(
                        result.masks.xy, result.boxes.cls, result.boxes.conf):
                    pts = np.array(seg, dtype=np.int32).reshape((-1,1,2))
                    cx  = int(np.mean(pts[:,0,0]))
                    cy  = int(np.mean(pts[:,0,1]))
                    cls_name = names[int(cls_id)]
                    colors = [(0,220,80),(0,150,255),(255,180,0),
                              (220,80,220),(0,220,220),(255,80,80)]
                    col = colors[int(cls_id) % len(colors)]
                    cv2.polylines(vis, [pts], True, col, _t(2, s))
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, [pts], col)
                    cv2.addWeighted(vis, 0.7, overlay, 0.3, 0, vis)
                    if show_labels:
                        cv2.putText(vis, f"{cls_name} {float(conf_val):.2f}",
                                    (cx, cy-int(10*s)), cv2.FONT_HERSHEY_SIMPLEX,
                                    _fs(0.6, s), col, _t(2, s))
                    detections.append({"class": cls_name, "cls_id": int(cls_id),
                                       "conf": float(conf_val), "cx": cx, "cy": cy})
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
                    detections.append({"class": cls_name, "cls_id": int(cls_id),
                                       "conf": float(conf_val), "cx": cx, "cy": cy,
                                       "x1":x1,"y1":y1,"x2":x2,"y2":y2})

        n = len(detections)
        min_det = params.get("min_count", 1)
        max_det_check = params.get("max_count", 9999)
        is_pass = min_det <= n <= max_det_check

        print(f"[YOLO] {n} detected {'PASS' if is_pass else 'FAIL'}")
        return {"image": vis, "detections": detections,
                "count": n, "pass": is_pass}

    except Exception as e:
        vis = _bgr(img.copy())
        print(f"[YOLO] Error: {str(e)[:120]}")
        return {"image": vis, "detections": [], "count": 0, "pass": False}


TOOL_REGISTRY: List[ToolDef] = [

  # ── ACQUIRE IMAGE ───────────────────────────────────────────────
  ToolDef("acquire_image","Acquire Image","Acquire Image",
    "Load ảnh từ file hoặc folder — CogAcqFifoTool","#0f3460","🖼",
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
    proc_acquire_image, "CogAcqFifoTool"),

  ToolDef("camera_acquire","Camera Acquire","Acquire Image",
    "Capture từ camera (OpenCV / HikRobot / Do3think) — CogAcqFifoTool","#0f3460","📷",
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
    proc_camera_acquire, "CogAcqFifoTool"),

  # ── PATTERN FIND ────────────────────────────────────────────────
  ToolDef("patmax","Search PatMax","Pattern Find",
    "Pattern matching nâng cao với xoay góc — CogPatMaxPatternAlignTool",
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
    proc_patmax, "CogPatMaxPatternAlignTool"),

  ToolDef("patmax_align","PatMax Align Tool","Pattern Find",
    "PatMax Pattern Align — chọn Algorithm & Train Mode (CogPMAlignTool)",
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
    proc_patmax_align, "CogPMAlignTool"),

  ToolDef("patfind","PatFind","Pattern Find",
    "Pattern matching nhanh (NCC) — CogPMAlignTool","#16213e","🔍",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("score","number"),
     PortDef("x","number"),PortDef("y","number"),PortDef("num_found","number")],
    [P("accept_threshold","Accept Threshold","float",0.5,0,1,step=0.01),
     P("show_xy","Show X,Y reference","bool",True,
       tooltip="Hiện origin marker + X/Y axes + label '(x,y)' trên ảnh output."),
     P("show_bbox","Show bounding box","bool",True,
       tooltip="Hiện rotated bounding box + score label trên ảnh output.")],
    proc_patfind, "CogPMAlignTool"),

  # ── FIXTURE ─────────────────────────────────────────────────────
  ToolDef("fixture","Fixture","Fixture",
    "Thiết lập hệ tọa độ theo part — CogFixtureTool","#1a1a2e","📌",
    [PortDef("image","image"),
     PortDef("ref_x","number",required=False),
     PortDef("ref_y","number",required=False),
     PortDef("ref_angle","number",required=False)],
    [PortDef("image","image"),PortDef("transform_matrix","any"),
     PortDef("offset_x","number"),PortDef("offset_y","number"),PortDef("angle","number")],
    [P("origin_x","Origin X","float",320,0,8192),
     P("origin_y","Origin Y","float",240,0,8192)],
    proc_fixture, "CogFixtureTool"),

  # ── CALIPER ─────────────────────────────────────────────────────
  ToolDef("caliper","Caliper","Caliper",
    "Đo cạnh & khoảng cách 2 cạnh sub-pixel — CogCaliperTool","#1b4332","📐",
    [PortDef("image","image")],
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
    proc_caliper, "CogCaliperTool"),

  ToolDef("caliper_multi","Caliper Multi-Edge","Caliper",
    "Tìm tất cả cạnh trong vùng — CogCaliperTool","#1b4332","📏",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("edges","any"),
     PortDef("count","number"),PortDef("pass","bool")],
    [P("x1","X1","int",0,0,8192),P("y1","Y1","int",240,0,8192),
     P("x2","X2","int",640,0,8192),P("y2","Y2","int",240,0,8192),
     P("edge_threshold","Edge Threshold","float",10.0,0,500),
     P("min_count","Min Edges","int",1,0,100),
     P("max_count","Max Edges","int",100,0,1000)],
    proc_caliper_multi, "CogCaliperTool"),

  # ── BLOB ────────────────────────────────────────────────────────
  ToolDef("blob","Blob Analysis","Blob Analysis",
    "Phân tích vùng toàn diện: area, circularity, elongation — CogBlobTool",
    "#2d6a4f","🔵",
    [PortDef("image","image"),PortDef("mask","image",required=False)],
    [PortDef("image","image"),PortDef("count","number"),PortDef("pass","bool"),
     PortDef("total_area","number"),PortDef("blobs","any"),PortDef("centroids","any")],
    [P("auto_threshold","Auto Threshold (Otsu)","bool",True),
     P("threshold","Manual Threshold","int",128,0,255),
     P("invert","Invert Mask","bool",False),
     P("morph_open","Morph Open (px)","int",0,0,50,tooltip="Xóa nhiễu nhỏ"),
     P("morph_close","Morph Close (px)","int",0,0,50,tooltip="Lấp lỗ nhỏ"),
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
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label area (mm²) cạnh từng blob lên ảnh output. Mặc định tắt — vẫn được log ra console.")],
    proc_blob, "CogBlobTool"),

  # ── EDGE / LINE / CIRCLE ────────────────────────────────────────
  ToolDef("find_line","Find Line","Edge & Geometry",
    "Tìm đường thẳng từ edge — CogFindLineTool","#134074","〰",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("found","bool"),PortDef("angle","number"),
     PortDef("distance","number"),PortDef("point_x","number"),
     PortDef("point_y","number"),PortDef("pass","bool")],
    [P("x1","ROI X1","int",0,0,8192),P("y1","ROI Y1","int",200,0,8192),
     P("x2","ROI X2","int",640,0,8192),P("y2","ROI Y2","int",280,0,8192),
     P("canny_low","Canny Low","int",50,0,500),P("canny_high","Canny High","int",150,0,500),
     P("pixel_to_mm","Pixel→mm","float",1.0,0.0001,1000,step=0.0001),
     P("min_angle","Min Angle (°)","float",-180,-180,180),
     P("max_angle","Max Angle (°)","float",180,-180,180)],
    proc_find_line, "CogFindLineTool"),

  ToolDef("find_circle","Find Circle","Edge & Geometry",
    "Fit đường tròn chính xác — CogFindCircleTool","#134074","⭕",
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
    proc_find_circle, "CogFindCircleTool"),

  # ── COLOR ───────────────────────────────────────────────────────
  ToolDef("color_picker","Color Picker","Color Analysis",
    "Click chuột lấy màu → xuất HSV range","#6b2737","🎨",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("color_hsv","any"),
     PortDef("h","number"),PortDef("s","number"),PortDef("v","number"),
     PortDef("r","number"),PortDef("g","number"),PortDef("b","number")],
    [P("pick_x","Pick X","int",0,0,8192,tooltip="Double-click node → click ảnh để lấy màu"),
     P("pick_y","Pick Y","int",0,0,8192),
     P("tolerance","HSV Tolerance","int",20,0,100),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label 'H S V' cạnh điểm picked lên ảnh output. Mặc định tắt.")],
    proc_color_picker, "CogColorTool"),

  ToolDef("color_segment","Color Segmentation","Color Analysis",
    "Phân đoạn màu HSV — CogColorSegmenterTool","#6b2737","🌈",
    [PortDef("image","image"),PortDef("color_hsv","any",required=False)],
    [PortDef("image","image"),PortDef("mask","image"),PortDef("pass","bool"),
     PortDef("pixel_ratio","number"),PortDef("pixel_count","number")],
    [P("h_low","H Low","int",0,0,180),P("h_high","H High","int",180,0,180),
     P("s_low","S Low","int",50,0,255),P("s_high","S High","int",255,0,255),
     P("v_low","V Low","int",50,0,255),P("v_high","V High","int",255,0,255),
     P("tolerance","Tolerance","int",0,0,100),
     P("morph_open","Morph Open","int",0,0,50),
     P("min_ratio","Min Ratio","float",0.01,0,1,step=0.001),
     P("max_ratio","Max Ratio","float",1.0,0,1,step=0.001)],
    proc_color_segment, "CogColorSegmenterTool"),

  ToolDef("color_match","Color Match","Color Analysis",
    "So khớp màu trung bình ROI — CogColorMatchTool","#6b2737","🎭",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("pass","bool"),PortDef("delta_e","number"),
     PortDef("mean_r","number"),PortDef("mean_g","number"),PortDef("mean_b","number")],
    [P("x","ROI X","int",0,0,8192),P("y","ROI Y","int",0,0,8192),
     P("w","ROI W","int",50,1,8192),P("h","ROI H","int",50,1,8192),
     P("ref_r","Ref R","int",128,0,255),P("ref_g","Ref G","int",128,0,255),
     P("ref_b","Ref B","int",128,0,255),
     P("max_delta_e","Max ΔE","float",30.0,0,441,tooltip="ΔE Euclidean RGB distance")],
    proc_color_match, "CogColorMatchTool"),

  # ── ID / READ ───────────────────────────────────────────────────
  ToolDef("id_reader","ID Reader","ID & Read",
    "Đọc Barcode 1D/2D, QR, DataMatrix — CogIDReaderTool","#3d0c02","📦",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("data","any"),
     PortDef("symbology","any"),PortDef("pass","bool")],
    [P("expected_data","Expected Data","str","",tooltip="Để trống = chấp nhận mọi code")],
    proc_id_reader, "CogIDReaderTool"),

  ToolDef("ocr_max","OCR Max","ID & Read",
    "Nhận dạng & xác nhận ký tự — CogOCRMaxTool","#3d0c02","🔤",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("text","any"),
     PortDef("pass","bool"),PortDef("confidence","number")],
    [P("lang","Language","str","eng"),P("psm","PSM Mode","int",6,0,13,
       tooltip="6=block, 7=single line, 8=single word"),
     P("expected_text","Expected Text","str",""),
     P("min_confidence","Min Confidence (%)","float",60.0,0,100)],
    proc_ocr_max, "CogOCRMaxTool"),

  # ── MEASUREMENT ─────────────────────────────────────────────────
  ToolDef("dist_point","Distance Point-Point","Measurement",
    "Đo khoảng cách 2 điểm — CogDistancePointPointTool","#134074","↔",
    [PortDef("image","image",required=False),
     PortDef("x1","number",required=False),PortDef("y1","number",required=False),
     PortDef("x2","number",required=False),PortDef("y2","number",required=False)],
    [PortDef("image","image"),PortDef("distance","number"),PortDef("pass","bool")],
    [P("x1","X1","int",0,0,8192),P("y1","Y1","int",0,0,8192),
     P("x2","X2","int",100,0,8192),P("y2","Y2","int",0,0,8192),
     P("pixel_to_mm","Pixel→mm","float",1.0,0.0001,1000,step=0.0001),
     P("min_dist","Min (mm)","float",0.0,0,100000),
     P("max_dist","Max (mm)","float",9999.0,0,100000),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ label '…mm' giữa 2 điểm lên ảnh output. Mặc định tắt.")],
    proc_distance_point, "CogDistancePointPointTool"),

  ToolDef("angle_lines","Angle Line-Line","Measurement",
    "Đo góc giữa 2 đường — CogAngleLineLineTool","#134074","∠",
    [PortDef("image","image",required=False),
     PortDef("angle1","number",required=False),PortDef("angle2","number",required=False)],
    [PortDef("image","image"),PortDef("angle","number"),PortDef("pass","bool")],
    [P("line1_angle","Line 1 Angle (°)","float",0,-180,180,step=0.1),
     P("line2_angle","Line 2 Angle (°)","float",45,-180,180,step=0.1),
     P("min_angle","Min Angle (°)","float",0,0,180),
     P("max_angle","Max Angle (°)","float",90,0,180)],
    proc_angle_lines, "CogAngleLineLineTool"),

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
    proc_area, "CogMeasureRectangleTool"),

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
     P("max_defect_count","Max Defect Count (0=any)","int",0,0,1000)],
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
     P("max_scratches","Max Scratches (0=none)","int",0,0,1000)],
    proc_scratch_detect, ""),

  # ── IMAGE PROCESSING ────────────────────────────────────────────
  ToolDef("image_convert","Image Convert","Image Processing",
    "Chuyển đổi format ảnh — CogImageConvertTool","#2c3e50","🔄",
    [PortDef("image","image")],[PortDef("image","image")],
    [P("mode","Mode","enum","Grayscale",
       choices=["Grayscale","BGR to RGB","Invert","HSV","LAB","YCrCb"])],
    proc_image_convert, "CogImageConvertTool"),

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
    "Biến đổi hình thái học","#2c3e50","🔲",
    [PortDef("image","image")],[PortDef("image","image"),PortDef("mask","image")],
    [P("operation","Operation","enum","Open",
       choices=["Erode","Dilate","Open","Close","Gradient","Top Hat","Black Hat"]),
     P("shape","Shape","enum","Ellipse",choices=["Rect","Ellipse","Cross"]),
     P("kernel_size","Kernel Size","int",3,1,51,step=2),
     P("iterations","Iterations","int",1,1,20)],
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
    "Hiệu chỉnh camera từ checkerboard — CogCalibCheckerboardTool",
    "#1a472a","📋",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("calibrated","bool"),
     PortDef("pixel_to_mm","number"),PortDef("rms_error","number")],
    [P("grid_cols","Grid Cols","int",9,2,30,tooltip="Số góc nội (cols-1)"),
     P("grid_rows","Grid Rows","int",6,2,30),
     P("square_size_mm","Square Size (mm)","float",25.4,0.1,1000,step=0.1)],
    proc_calibrate_grid, "CogCalibCheckerboardTool"),

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
    "Kết hợp điều kiện → PASS/FAIL cuối cùng","#1c1c2e","🏁",
    [PortDef("A","bool",required=False),PortDef("B","bool",required=False),
     PortDef("C","bool",required=False),PortDef("D","bool",required=False)],
    [PortDef("result","bool"),PortDef("pass","bool")],
    [P("mode","Combine Mode","enum","ALL",choices=["ALL","ANY"])],
    proc_judge,""),

  ToolDef("script","Script Tool","Logic & Flow",
    "Chạy Python expression tùy chỉnh — CogScriptTool","#1c1c2e","🐍",
    [PortDef("A","any",required=False),PortDef("B","any",required=False),
     PortDef("C","any",required=False)],
    [PortDef("result","bool"),PortDef("pass","bool"),PortDef("output","any")],
    [P("expression","Python Expression","str","result = True",
       tooltip="Dùng: inputs['A'], inputs['B'], result=True/False, output=value")],
    proc_script,"CogScriptTool"),

  # ── OUTPUT / DISPLAY ────────────────────────────────────────────
  ToolDef("display","Display","Output & Display",
    "Annotate & hiển thị ảnh kết quả — CogRecordDisplayTool",
    "#0d1117","🖥",
    [PortDef("image","image"),PortDef("pass","bool",required=False)],
    [PortDef("image","image")],
    [P("label","Label Text","str",""),P("tx","Text X","int",10,0,8192),
     P("ty","Text Y","int",30,0,8192),P("font_scale","Font Scale","float",0.8,0.1,5,step=0.1),
     P("show_result","Show PASS/FAIL","bool",True)],
    proc_display,"CogRecordDisplayTool"),

  ToolDef("save_image","Save Image","Output & Display",
    "Lưu ảnh ra file — CogSaveImageTool","#0d1117","💾",
    [PortDef("image","image")],
    [PortDef("saved","bool"),PortDef("path","any")],
    [P("save_path","Save Path","str","output/result.png"),
     P("timestamp","Add Timestamp","bool",True)],
    proc_save_image,"CogSaveImageTool"),

  ToolDef("csv_log","CSV Logger","Output & Display",
    "Ghi kết quả vào CSV log file","#0d1117","📊",
    [PortDef("pass","bool",required=False),
     PortDef("value_a","any",required=False),PortDef("value_b","any",required=False),
     PortDef("value_c","any",required=False)],
    [PortDef("logged","bool"),PortDef("path","any")],
    [P("csv_path","CSV Path","str","log/results.csv")],
    proc_csv_log,""),

  # ── YOLO DETECTION ─────────────────────────────────────────────
  ToolDef("yolo_detect","YOLO Detect","YOLO",
    "YOLOv8/v11 Object Detection & Segmentation — train từ YOLO Studio",
    "#1a0a3a","🤖",
    [PortDef("image","image")],
    [PortDef("image","image"),PortDef("detections","any"),
     PortDef("count","number"),PortDef("pass","bool")],
    [P("model_path","Model Path (.pt)","str","",
       tooltip="Đường dẫn file .pt — dùng YOLO Studio để train"),
     P("confidence","Confidence","float",0.5,0.01,1.0,step=0.01),
     P("iou","IoU Threshold","float",0.45,0.01,1.0,step=0.01),
     P("imgsz","Image Size","int",640,32,4096,step=32),
     P("max_det","Max Detections","int",300,1,1000),
     P("min_count","Min Objects (PASS)","int",1,0,1000),
     P("max_count","Max Objects (PASS)","int",9999,0,10000),
     P("show_labels","Display: show labels on image","bool",False,
       tooltip="Bật để vẽ '{class} {conf}' cạnh từng detection lên ảnh output. Mặc định tắt — kết quả vẫn có trong log & detections port.")],
    proc_yolo_detect,"ultralytics YOLO"),

]

TOOL_BY_ID: Dict[str,ToolDef] = {t.tool_id: t for t in TOOL_REGISTRY}
CATEGORIES = [
    "Acquire Image","Pattern Find","Fixture","Caliper",
    "Blob Analysis","Edge & Geometry","Color Analysis","ID & Read",
    "Measurement","Surface Inspection","Image Processing",
    "Calibration","Logic & Flow","Output & Display","YOLO"
]
