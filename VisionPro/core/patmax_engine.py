"""
core/patmax_engine.py — v2
Fix: thuật toán search đáng tin cậy hơn.
Dùng 3 phương pháp song song:
  1. Raw patch NCC (nhanh, chính xác với ánh sáng đồng đều)
  2. Edge-on-edge (bất biến với thay đổi màu/sáng)
  3. Gradient orientation (Cognex-style, robust nhất)
Lấy score = weighted max của 3 phương pháp.
Debug: in score tối đa ra console để người dùng điều chỉnh threshold.
"""
from __future__ import annotations
import cv2
import numpy as np
import copy
import json, os, math, hashlib
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class PatMaxResult:
    found: bool
    score: float
    x: float
    y: float
    angle: float
    scale: float
    width: float
    height: float
    corners: List[Tuple[float, float]] = field(default_factory=list)
    # Origin (điểm tham chiếu) đã transform theo angle/scale của result
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class PatMaxModel:
    trained: bool = False
    train_roi: Optional[Tuple[int,int,int,int]] = None
    origin_x: float = 0.0
    origin_y: float = 0.0
    pattern_w: int = 0
    pattern_h: int = 0
    # Extra reference points (additional XY origins, pattern-local coords).
    # Mỗi ref: {"name": str, "x": float, "y": float, "angle": float}
    # x, y tính theo gốc (0,0) là góc trên-trái ROI (cùng hệ với origin_x/y).
    extra_refs: List[Dict] = field(default_factory=list)
    # Raw patch (BGR) — dùng cho NCC
    patch_bgr: Optional[np.ndarray] = None
    # Gray patch
    patch_gray: Optional[np.ndarray] = None
    # Canny edges
    edge_image: Optional[np.ndarray] = None
    # Thumbnail hiển thị
    thumbnail: Optional[np.ndarray] = None
    edge_count: int = 0
    model_hash: str = ""
    # Shape info: "rect" | "circle" | "ellipse" | "polygon"
    shape_type: str = "rect"
    shape_data: Optional[dict] = None
    # Mask (h × w) — uint8 0/255, áp lên patch khi train với non-rect shape
    mask: Optional[np.ndarray] = None
    # Train mode: "evaluate" (DOFs at runtime) | "create" (precomputed templates)
    train_mode: str = "evaluate"
    # Precomputed templates khi train_mode == "create"
    precomputed_templates: Optional[list] = None
    # Search params
    accept_threshold: float = 0.5
    angle_low: float = 0.0
    angle_high: float = 0.0
    angle_step: float = 5.0
    scale_low: float = 1.0
    scale_high: float = 1.0
    scale_step: float = 0.1
    num_results: int = 1
    overlap_threshold: float = 0.5
    # Canny params dùng lúc train
    canny_low: int = 50
    canny_high: int = 150

    def is_valid(self) -> bool:
        return (self.trained
                and self.patch_gray is not None
                and self.pattern_w > 4
                and self.pattern_h > 4)


# ═══════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════
def _build_template(patch_gray: np.ndarray,
                     edge_image: np.ndarray,
                     angle: float,
                     scale: float,
                     weights: Optional[Tuple[float, float, float]] = None,
                     mask: Optional[np.ndarray] = None) -> dict:
    """Tạo template (rotated/scaled) từ patch base + edge base.
    mask (h × w uint8 0/255): rotated/scaled cùng → dùng cho cv2.matchTemplate."""
    pw = max(4, int(patch_gray.shape[1] * scale))
    ph = max(4, int(patch_gray.shape[0] * scale))
    patch = cv2.resize(patch_gray, (pw, ph))
    edge_p = cv2.resize(edge_image, (pw, ph))
    mask_s = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST) \
        if mask is not None else None
    if abs(angle) > 0.1:
        rad = math.radians(angle)
        cos_a = abs(math.cos(rad)); sin_a = abs(math.sin(rad))
        nW = int(ph * sin_a + pw * cos_a) + 2
        nH = int(ph * cos_a + pw * sin_a) + 2
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        M[0, 2] += (nW - pw) / 2; M[1, 2] += (nH - ph) / 2
        patch_rot = cv2.warpAffine(patch, M, (nW, nH),
                                    borderMode=cv2.BORDER_REPLICATE)
        edge_rot  = cv2.warpAffine(edge_p, M, (nW, nH))
        if mask_s is not None:
            mask_rot = cv2.warpAffine(mask_s, M, (nW, nH),
                                      flags=cv2.INTER_NEAREST)
        else:
            mask_rot = None
    else:
        nW, nH = pw, ph
        patch_rot = patch
        edge_rot  = edge_p
        mask_rot  = mask_s
    # Edge dilation thích ứng theo kích thước (cũ: cố định 5×5 → quá dày
    # với template nhỏ, edges blob ra → NCC edge giảm).
    k = max(2, min(5, min(nW, nH) // 18))
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edge_dil = cv2.dilate(edge_rot, kernel)
    out = {"angle": float(angle), "scale": float(scale),
            "patch": patch_rot, "edge_dil": edge_dil,
            "nW": nW, "nH": nH}
    if weights is not None:
        out["weights"] = weights
    if mask_rot is not None:
        out["mask"] = mask_rot
    return out


def _build_search_grid(angle_low, angle_high, angle_step,
                        scale_low, scale_high, scale_step):
    if abs(angle_high - angle_low) < 0.5:
        angles = [0.0]
    else:
        step = max(0.5, angle_step)
        angles = list(np.arange(angle_low, angle_high + step * 0.5, step))
        if 0.0 not in [round(a, 2) for a in angles]:
            angles.append(0.0)
    if abs(scale_high - scale_low) < 0.01:
        scales = [1.0]
    else:
        step_s = max(0.01, scale_step)
        scales = list(np.arange(scale_low, scale_high + step_s * 0.5, step_s))
    return angles, scales


def precompute_templates(model: PatMaxModel) -> int:
    """Build precomputed_templates từ ranges đã lưu trong model.
    Trả về số lượng template đã build."""
    if not model.is_valid():
        return 0
    angles, scales = _build_search_grid(
        model.angle_low, model.angle_high, model.angle_step,
        model.scale_low, model.scale_high, 0.1)
    tmpls = []
    for sc in scales:
        for ang in angles:
            tmpls.append(_build_template(model.patch_gray, model.edge_image,
                                          ang, sc, mask=model.mask))
    model.precomputed_templates = tmpls
    return len(tmpls)


def train_patmax(image: np.ndarray,
                 roi: Tuple[int,int,int,int],
                 origin_offset: Tuple[float,float] = (0.5, 0.5),
                 canny_low: int = 50,
                 canny_high: int = 150,
                 shape_type: str = "rect",
                 shape_data: Optional[dict] = None,
                 train_mode: str = "evaluate",
                 angle_low: float = 0.0,
                 angle_high: float = 0.0,
                 angle_step: float = 5.0,
                 scale_low: float = 1.0,
                 scale_high: float = 1.0,
                 scale_step: float = 0.1) -> PatMaxModel:

    x, y, w, h = roi
    H, W = image.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))

    bgr  = image[y:y+h, x:x+w].copy()
    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray_smooth, canny_low, canny_high)

    # Build mask trong toạ độ patch (h × w) cho non-rect shapes.
    # Không mean-fill nữa: cv2.matchTemplate có mask nguyên gốc → tính
    # NCC chỉ trên vùng-shape, không bóp méo bởi corner ngoài shape.
    mask = _build_shape_mask(shape_type, shape_data, x, y, w, h)
    if mask is not None:
        edges = np.where(mask > 0, edges, 0).astype(np.uint8)

    # Thumbnail 80×80
    th = cv2.resize(bgr, (80, 80))
    e_small = cv2.resize(edges, (80, 80))
    th[e_small > 0] = [0, 220, 80]

    ox = w * origin_offset[0]
    oy = h * origin_offset[1]
    model_hash = hashlib.md5(gray.tobytes()).hexdigest()[:8]

    print(f"[PatMax Train] ROI=({x},{y},{w},{h}) shape={shape_type} "
          f"mode={train_mode}  edges={int(np.count_nonzero(edges))}  hash={model_hash}")

    model = PatMaxModel(
        trained=True,
        train_roi=(x, y, w, h),
        origin_x=ox, origin_y=oy,
        pattern_w=w, pattern_h=h,
        patch_bgr=bgr,
        patch_gray=gray_smooth,
        edge_image=edges,
        thumbnail=th,
        edge_count=int(np.count_nonzero(edges)),
        model_hash=model_hash,
        canny_low=canny_low,
        canny_high=canny_high,
        shape_type=shape_type,
        shape_data=dict(shape_data) if shape_data else None,
        mask=mask,
        train_mode=train_mode if train_mode in ("evaluate", "create") else "evaluate",
        angle_low=angle_low, angle_high=angle_high, angle_step=angle_step,
        scale_low=scale_low, scale_high=scale_high,
    )
    if model.train_mode == "create":
        n = precompute_templates(model)
        print(f"[PatMax Train] precomputed {n} DOF templates")
    return model


def train_patmax_multi_region(image: np.ndarray,
                                regions: List[dict],
                                origin_offset: Tuple[float, float] = (0.5, 0.5),
                                canny_low: int = 50,
                                canny_high: int = 150,
                                train_mode: str = "evaluate",
                                angle_low: float = 0.0,
                                angle_high: float = 0.0,
                                angle_step: float = 5.0,
                                scale_low: float = 1.0,
                                scale_high: float = 1.0,
                                scale_step: float = 0.1) -> Optional[PatMaxModel]:
    """Train 1 PatMax model GỘP từ nhiều ROI (multi-region pattern).

    Bbox = union các ROI; mask = OR mask của từng region.
    Edges/patch ngoài union mask bị zero/mean để chỉ pattern union được match.
    `regions`: list of {"type": "rect|circle|ellipse|polygon", **shape_data}.
    """
    if not regions:
        return None
    H, W = image.shape[:2]

    # Tính bbox per-region (toạ độ ảnh)
    bboxes = []
    for r in regions:
        t = r.get("type", "rect")
        if t == "circle":
            cx = int(r.get("cx", 0)); cy = int(r.get("cy", 0))
            rd = int(r.get("r", 0))
            bboxes.append((cx - rd, cy - rd, 2 * rd, 2 * rd))
        else:
            bboxes.append((int(r.get("x", 0)), int(r.get("y", 0)),
                            int(r.get("w", 0)), int(r.get("h", 0))))
    # Union bbox
    x0 = max(0, min(b[0] for b in bboxes))
    y0 = max(0, min(b[1] for b in bboxes))
    x1 = min(W, max(b[0] + b[2] for b in bboxes))
    y1 = min(H, max(b[1] + b[3] for b in bboxes))
    uw = max(1, x1 - x0); uh = max(1, y1 - y0)

    # Build union mask trong toạ độ patch (uh × uw)
    union_mask = np.zeros((uh, uw), dtype=np.uint8)
    for r in regions:
        t = r.get("type", "rect")
        # Tạo dữ liệu shape relative tới (x0, y0) để dùng _build_shape_mask
        if t == "circle":
            shape_data = {"cx": int(r.get("cx", 0)), "cy": int(r.get("cy", 0)),
                          "r":  int(r.get("r", 0))}
        elif t == "polygon":
            shape_data = {"pts": [(int(px), int(py)) for px, py in r.get("pts", [])]}
        else:
            shape_data = {"x": int(r.get("x", 0)), "y": int(r.get("y", 0)),
                          "w": int(r.get("w", 0)), "h": int(r.get("h", 0))}
        m = _build_shape_mask(t, shape_data, x0, y0, uw, uh)
        if m is None and t == "rect":
            # rect có _build_shape_mask trả None — fill thủ công
            rx = max(0, int(r.get("x", 0)) - x0)
            ry = max(0, int(r.get("y", 0)) - y0)
            rw = max(1, min(int(r.get("w", 0)), uw - rx))
            rh = max(1, min(int(r.get("h", 0)), uh - ry))
            m = np.zeros((uh, uw), dtype=np.uint8)
            m[ry:ry + rh, rx:rx + rw] = 255
        if m is not None:
            union_mask = np.maximum(union_mask, m)

    bgr = image[y0:y1, x0:x1].copy()
    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_smooth, canny_low, canny_high)

    if union_mask.any():
        mean_val = int(gray_smooth[union_mask > 0].mean())
        gray_smooth = np.where(union_mask > 0, gray_smooth, mean_val).astype(np.uint8)
        edges = np.where(union_mask > 0, edges, 0).astype(np.uint8)

    th = cv2.resize(bgr, (80, 80))
    e_small = cv2.resize(edges, (80, 80))
    th[e_small > 0] = [0, 220, 80]

    ox = uw * origin_offset[0]; oy = uh * origin_offset[1]
    model_hash = hashlib.md5(gray.tobytes()).hexdigest()[:8]
    print(f"[PatMax Train multi-region] union ROI=({x0},{y0},{uw},{uh})  "
          f"regions={len(regions)}  edges={int(np.count_nonzero(edges))}  "
          f"hash={model_hash}")

    model = PatMaxModel(
        trained=True,
        train_roi=(x0, y0, uw, uh),
        origin_x=ox, origin_y=oy,
        pattern_w=uw, pattern_h=uh,
        patch_bgr=bgr, patch_gray=gray_smooth,
        edge_image=edges, thumbnail=th,
        edge_count=int(np.count_nonzero(edges)),
        model_hash=model_hash,
        canny_low=canny_low, canny_high=canny_high,
        shape_type="multi", shape_data={"regions": regions},
        mask=union_mask,
        train_mode=train_mode if train_mode in ("evaluate", "create") else "evaluate",
        angle_low=angle_low, angle_high=angle_high, angle_step=angle_step,
        scale_low=scale_low, scale_high=scale_high,
    )
    if model.train_mode == "create":
        n = precompute_templates(model)
        print(f"[PatMax Train multi-region] precomputed {n} DOF templates")
    return model


def train_patmax_multi_pattern(image: np.ndarray,
                                regions: List[dict],
                                origin_offset: Tuple[float, float] = (0.5, 0.5),
                                canny_low: int = 50,
                                canny_high: int = 150,
                                train_mode: str = "evaluate",
                                angle_low: float = 0.0,
                                angle_high: float = 0.0,
                                angle_step: float = 5.0,
                                scale_low: float = 1.0,
                                scale_high: float = 1.0,
                                scale_step: float = 0.1) -> List[PatMaxModel]:
    """Train list of PatMaxModels — mỗi region 1 model độc lập."""
    models: List[PatMaxModel] = []
    for r in regions:
        t = r.get("type", "rect")
        if t == "circle":
            cx = int(r.get("cx", 0)); cy = int(r.get("cy", 0))
            rd = int(r.get("r", 0))
            roi = (cx - rd, cy - rd, 2 * rd, 2 * rd)
            sd = {"cx": cx, "cy": cy, "r": rd}
        elif t == "polygon":
            pts = r.get("pts") or []
            xs = [px for px, _ in pts]; ys = [py for _, py in pts]
            x = int(min(xs)) if xs else 0; y = int(min(ys)) if ys else 0
            w = max(1, int(max(xs) - min(xs))) if xs else 1
            h = max(1, int(max(ys) - min(ys))) if ys else 1
            roi = (x, y, w, h)
            sd = {"pts": pts, "x": x, "y": y, "w": w, "h": h}
        else:
            x = int(r.get("x", 0)); y = int(r.get("y", 0))
            w = int(r.get("w", 0)); h = int(r.get("h", 0))
            roi = (x, y, w, h); sd = {"x": x, "y": y, "w": w, "h": h}
        m = train_patmax(image, roi, origin_offset=origin_offset,
                          canny_low=canny_low, canny_high=canny_high,
                          shape_type=t, shape_data=sd, train_mode=train_mode,
                          angle_low=angle_low, angle_high=angle_high,
                          angle_step=angle_step,
                          scale_low=scale_low, scale_high=scale_high,
                          scale_step=scale_step)
        if m is not None:
            models.append(m)
    return models


def run_patmax_multi(image: np.ndarray,
                      models: List[PatMaxModel],
                      accept_threshold: float = 0.5,
                      angle_low: float = 0.0,
                      angle_high: float = 0.0,
                      angle_step: float = 5.0,
                      scale_low: float = 1.0,
                      scale_high: float = 1.0,
                      scale_step: float = 0.1,
                      num_results_per_model: int = 1,
                      overlap_threshold: float = 0.5,
                      coarse_downscale: int = 1,
                      channels: Tuple[bool, bool, bool] = (True, True, True),
                      ) -> Tuple[List[PatMaxResult], np.ndarray]:
    """Search trên list models — gộp results, score map = max overlay."""
    all_results: List[PatMaxResult] = []
    score_acc: Optional[np.ndarray] = None
    for mi, model in enumerate(models):
        if not model.is_valid():
            continue
        results, sm = run_patmax(image, model,
                                  accept_threshold=accept_threshold,
                                  angle_low=angle_low, angle_high=angle_high,
                                  angle_step=angle_step,
                                  scale_low=scale_low, scale_high=scale_high,
                                  scale_step=scale_step,
                                  num_results=num_results_per_model,
                                  overlap_threshold=overlap_threshold,
                                  coarse_downscale=coarse_downscale,
                                  channels=channels)
        all_results.extend(results)
        if score_acc is None:
            score_acc = sm.astype(np.float32) if sm is not None else None
        elif sm is not None and sm.shape == score_acc.shape:
            score_acc = np.maximum(score_acc, sm.astype(np.float32))
    all_results.sort(key=lambda r: -r.score)
    if score_acc is None:
        score_acc = _empty_vis(image)
    else:
        score_acc = score_acc.astype(np.uint8)
    return all_results, score_acc


def _build_shape_mask(shape_type: str, shape_data: Optional[dict],
                       roi_x: int, roi_y: int, w: int, h: int
                       ) -> Optional[np.ndarray]:
    """Tạo mask (h × w) uint8 0/255 cho non-rect shape, toạ độ ảnh → patch."""
    if not shape_data or shape_type == "rect":
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    if shape_type == "ellipse":
        cx = int((shape_data.get("x", roi_x) - roi_x) + shape_data.get("w", w) / 2)
        cy = int((shape_data.get("y", roi_y) - roi_y) + shape_data.get("h", h) / 2)
        ax = max(1, int(shape_data.get("w", w) / 2))
        ay = max(1, int(shape_data.get("h", h) / 2))
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    elif shape_type == "circle":
        cx = int(shape_data.get("cx", roi_x + w / 2) - roi_x)
        cy = int(shape_data.get("cy", roi_y + h / 2) - roi_y)
        r = max(1, int(shape_data.get("r", min(w, h) / 2)))
        cv2.circle(mask, (cx, cy), r, 255, -1)
    elif shape_type == "polygon":
        pts = shape_data.get("pts") or []
        if len(pts) >= 3:
            arr = np.array([[int(px - roi_x), int(py - roi_y)] for px, py in pts],
                            dtype=np.int32)
            cv2.fillPoly(mask, [arr], 255)
        else:
            return None
    else:
        return None
    return mask


# ═══════════════════════════════════════════════════════════════
#  SEARCH — single angle/scale attempt
# ═══════════════════════════════════════════════════════════════
def _match_template(gray_img: np.ndarray,
                     img_edges: np.ndarray,
                     t: Dict,
                     max_locations: int = 1) -> List[Dict]:
    """
    Match một template đã build sẵn — trả TOP-K peaks trên score map.
    Cho phép multi-object detection (mỗi peak ≈ một object). Caller
    sau đó phân biệt overlap qua NMS.
    """
    patch_rot = t["patch"]
    edge_rot_d = t["edge_dil"]
    nW = t["nW"]; nH = t["nH"]
    angle = t["angle"]; scale = t["scale"]
    mask_rot = t.get("mask")        # uint8 0/255 hoặc None
    H, W = gray_img.shape[:2]

    if nW >= W - 2 or nH >= H - 2:
        return []

    # Channels có thể tắt qua _channels = (ncc, edge, sq)
    chan_ncc, chan_edge, chan_sq = t.get("_channels", (True, True, True))

    # cv2.matchTemplate hỗ trợ mask cho TM_CCOEFF_NORMED + TM_SQDIFF_NORMED:
    # mask phải cùng kích thước template, dtype uint8 hoặc float32 (8-bit OK).
    # Khi có mask, các pixel ngoài shape không tính vào correlation → loại bỏ
    # ảnh hưởng của 4 góc bbox (vốn không thuộc circle/ellipse/polygon).
    if not chan_ncc:
        return []   # NCC bắt buộc (làm cơ sở score)
    try:
        if mask_rot is not None:
            res_ncc = cv2.matchTemplate(gray_img, patch_rot,
                                        cv2.TM_CCOEFF_NORMED, mask=mask_rot)
        else:
            res_ncc = cv2.matchTemplate(gray_img, patch_rot, cv2.TM_CCOEFF_NORMED)
        res_ncc = np.nan_to_num(res_ncc, nan=0.0, posinf=0.0, neginf=0.0)
    except cv2.error:
        return []

    # Mask-mode matchTemplate đôi khi vượt [0,1] do numerical noise → clip.
    s_ncc_map = np.clip(res_ncc, 0.0, 1.0, out=res_ncc).astype(np.float32, copy=False)
    s_edge_map = None
    s_sq_map   = None

    if chan_edge:
        img_e_f   = img_edges.astype(np.float32) / 255.0
        templ_e_f = edge_rot_d.astype(np.float32) / 255.0
        try:
            res_edge = cv2.matchTemplate(img_e_f, templ_e_f, cv2.TM_CCOEFF_NORMED)
            res_edge = np.nan_to_num(res_edge, nan=0.0, posinf=0.0, neginf=0.0)
            s_edge_map = np.clip(res_edge, 0.0, 1.0, out=res_edge).astype(np.float32, copy=False)
        except cv2.error:
            chan_edge = False

    if chan_sq:
        try:
            if mask_rot is not None:
                res_sq = cv2.matchTemplate(gray_img, patch_rot,
                                           cv2.TM_SQDIFF_NORMED, mask=mask_rot)
            else:
                res_sq = cv2.matchTemplate(gray_img, patch_rot, cv2.TM_SQDIFF_NORMED)
            res_sq = np.nan_to_num(res_sq, nan=1.0, posinf=1.0, neginf=1.0)
            res_sq_inv = 1.0 - res_sq
            s_sq_map = np.clip(res_sq_inv, 0.0, 1.0, out=res_sq_inv).astype(np.float32, copy=False)
        except cv2.error:
            chan_sq = False

    # Weights configurable per template (algorithm-dependent).
    w_ncc, w_edge, w_sq = t.get("weights", (0.5, 0.3, 0.2))
    if not chan_edge: w_edge = 0.0
    if not chan_sq:   w_sq   = 0.0
    w_sum = w_ncc + w_edge + w_sq
    if w_sum > 0:
        w_ncc, w_edge, w_sq = w_ncc / w_sum, w_edge / w_sum, w_sq / w_sum
    # Tránh nhân với map zero không cần thiết — đặc biệt lợi khi PatQuick chỉ
    # dùng NCC (w_edge=w_sq=0): score_map = s_ncc_map nguyên gốc, 0 cộng/nhân.
    if w_edge == 0.0 and w_sq == 0.0:
        score_map = s_ncc_map
    else:
        score_map = w_ncc * s_ncc_map
        if w_edge > 0.0 and s_edge_map is not None:
            score_map = score_map + w_edge * s_edge_map
        if w_sq > 0.0 and s_sq_map is not None:
            score_map = score_map + w_sq * s_sq_map

    # Local NMS: suppress peaks gần nhau bằng cách lấy max trong cửa sổ
    win = max(3, min(nW, nH) // 2)
    if win % 2 == 0: win += 1
    kernel = np.ones((win, win), dtype=np.uint8)
    local_max = cv2.dilate(score_map, kernel)
    peaks_mask = (score_map == local_max) & (score_map > 0)

    ys, xs = np.where(peaks_mask)
    if len(ys) == 0:
        return []
    scores = score_map[ys, xs]

    # Lấy top-K peaks
    K = max(1, int(max_locations))
    if len(scores) > K:
        idx = np.argpartition(-scores, K)[:K]
        idx = idx[np.argsort(-scores[idx])]
    else:
        idx = np.argsort(-scores)

    out: List[Dict] = []
    for i in idx:
        ty = int(ys[i]); tx = int(xs[i])
        out.append({
            "score":  float(scores[i]),
            "s_ncc":  float(s_ncc_map[ty, tx]),
            "s_edge": float(s_edge_map[ty, tx]) if s_edge_map is not None else 0.0,
            "s_sq":   float(s_sq_map[ty, tx])   if s_sq_map   is not None else 0.0,
            "cx": float(tx + nW / 2),
            "cy": float(ty + nH / 2),
            "tx": tx, "ty": ty,
            "nW": nW, "nH": nH,
            "angle": angle, "scale": scale,
        })
    return out


# ═══════════════════════════════════════════════════════════════
#  RUN PATMAX — main search
# ═══════════════════════════════════════════════════════════════
def run_patmax(image: np.ndarray,
               model: PatMaxModel,
               accept_threshold: float = 0.5,
               angle_low: float = 0.0,
               angle_high: float = 0.0,
               angle_step: float = 5.0,
               scale_low: float = 1.0,
               scale_high: float = 1.0,
               scale_step: float = 0.1,
               num_results: int = 1,
               overlap_threshold: float = 0.5,
               coarse_downscale: int = 1,
               channels: Tuple[bool, bool, bool] = (True, True, True),
               ) -> Tuple[List[PatMaxResult], np.ndarray]:
    """
    Optimization knobs:
        coarse_downscale : 1 = full-res search (cũ);
                           2 = search ở 1/2 res rồi map về full-res (≈4x nhanh);
                           4 = search ở 1/4 (≈16x nhanh, độ chính xác giảm).
                           Sau khi tìm peak coarse, vị trí được trả về ở
                           full-res — KHÔNG refine. Phù hợp khi cần locate
                           nhanh; sai số ±downscale pixel.
        channels         : bật/tắt 3 channels matchTemplate (NCC, edge,
                           SQDIFF). Tắt SQDIFF/edge giảm ~33% mỗi cái.
    """
    if not model.is_valid():
        return [], _empty_vis(image)

    bgr      = image if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H_full, W_full = gray_full.shape

    # Pyramid downscale — tất cả match diễn ra ở gray_img/img_edges resolution
    ds = max(1, int(coarse_downscale))
    if ds > 1:
        gray_img = cv2.resize(gray_full, (W_full // ds, H_full // ds),
                               interpolation=cv2.INTER_AREA)
    else:
        gray_img = gray_full
    H, W = gray_img.shape

    canny_lo = model.canny_low
    canny_hi = model.canny_high
    img_edges = cv2.Canny(gray_img, canny_lo, canny_hi)

    # Chọn nguồn templates: precomputed (mode "create") hoặc build on-the-fly
    use_precomputed = (model.train_mode == "create"
                       and model.precomputed_templates)

    if use_precomputed:
        base_templates = model.precomputed_templates
        print(f"[PatMax Search] using {len(base_templates)} precomputed templates  "
              f"threshold={accept_threshold}  ds={ds}")
    else:
        angles, scales = _build_search_grid(angle_low, angle_high, angle_step,
                                              scale_low, scale_high, scale_step)
        base_templates = [_build_template(model.patch_gray, model.edge_image,
                                           ang, sc, mask=model.mask)
                          for sc in scales for ang in angles]
        print(f"[PatMax Search] evaluate-DOF: {len(base_templates)} templates "
              f"(angles={len(angles)} × scales={len(scales)})  "
              f"threshold={accept_threshold}  ds={ds}")

    # Nếu pyramid > 1, downscale patch+edge của từng template để khớp gray_img
    if ds > 1:
        templates = []
        for t in base_templates:
            nW2 = max(4, t["nW"] // ds)
            nH2 = max(4, t["nH"] // ds)
            t2 = dict(t)
            t2["patch"]    = cv2.resize(t["patch"],    (nW2, nH2), interpolation=cv2.INTER_AREA)
            t2["edge_dil"] = cv2.resize(t["edge_dil"], (nW2, nH2), interpolation=cv2.INTER_AREA)
            if "mask" in t and t["mask"] is not None:
                t2["mask"] = cv2.resize(t["mask"], (nW2, nH2), interpolation=cv2.INTER_NEAREST)
            t2["nW"] = nW2; t2["nH"] = nH2
            templates.append(t2)
    else:
        templates = base_templates

    # Truyền channels flag xuống _match_template qua key tạm
    chan_ncc, chan_edge, chan_sq = channels
    for t in templates:
        t["_channels"] = (chan_ncc, chan_edge, chan_sq)

    candidates: List[Dict] = []
    score_map = np.zeros((H, W), dtype=np.float32)
    best_any = 0.0

    # Oversample peaks per template — giảm xuống chỉ *2 khi num_results=1
    # để tránh phí cho NMS không cần thiết.
    if num_results <= 1:
        peaks_per_template = 4
    else:
        peaks_per_template = max(num_results * 4, 8)

    for t in templates:
        results_t = _match_template(gray_img, img_edges, t, peaks_per_template)
        for result in results_t:
            score = result["score"]
            best_any = max(best_any, score)

            tx, ty = result["tx"], result["ty"]
            nW, nH = result["nW"], result["nH"]
            s_for_map = result["s_ncc"]
            if (0 <= ty < H - nH) and (0 <= tx < W - nW):
                cy_i = min(int(result["cy"]), H - 1)
                cx_i = min(int(result["cx"]), W - 1)
                if score_map[cy_i, cx_i] < s_for_map:
                    score_map[cy_i, cx_i] = s_for_map

            # Map về full-res coordinates nếu pyramid > 1
            if ds > 1:
                result["cx"] *= ds; result["cy"] *= ds
                result["tx"] *= ds; result["ty"] *= ds
                result["nW"] *= ds; result["nH"] *= ds

            if score >= accept_threshold:
                candidates.append(result)

    print(f"[PatMax Search] best_score={best_any:.4f}  "
          f"candidates_above_threshold={len(candidates)}  "
          f"threshold={accept_threshold}")

    if best_any < accept_threshold:
        print(f"[PatMax Search] ⚠  Không tìm thấy. "
              f"Thử giảm threshold xuống {best_any * 0.85:.2f} "
              f"hoặc retrain với Canny phù hợp hơn.")

    # NMS
    candidates.sort(key=lambda d: -d["score"])
    kept: List[Dict] = []
    for cand in candidates:
        overlap = False
        for k in kept:
            dist = math.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"])
            min_dim = min(cand["nW"], cand["nH"], k["nW"], k["nH"])
            if dist < min_dim * overlap_threshold:
                overlap = True; break
        if not overlap:
            kept.append(cand)
        if len(kept) >= num_results:
            break

    # Origin offset từ tâm pattern (toạ độ pattern, không clamp)
    pdx = float(model.origin_x) - float(model.pattern_w) / 2.0
    pdy = float(model.origin_y) - float(model.pattern_h) / 2.0

    # Build results
    results: List[PatMaxResult] = []
    for d in kept:
        corners = _rotated_corners(d["cx"], d["cy"], d["nW"], d["nH"], d["angle"])
        rad_o = math.radians(-d["angle"])
        ca = math.cos(rad_o); sa = math.sin(rad_o)
        s = d["scale"] if d["scale"] else 1.0
        ox_t = float(d["cx"]) + s * (pdx * ca - pdy * sa)
        oy_t = float(d["cy"]) + s * (pdx * sa + pdy * ca)
        results.append(PatMaxResult(
            found=True, score=d["score"],
            x=d["cx"], y=d["cy"],
            angle=d["angle"], scale=d["scale"],
            width=float(d["nW"]), height=float(d["nH"]),
            corners=corners,
            origin_x=ox_t, origin_y=oy_t,
        ))

    # Score map visualization (blur để đẹp hơn)
    sm_blurred = cv2.GaussianBlur(score_map, (15, 15), 0)
    if ds > 1:
        sm_blurred = cv2.resize(sm_blurred, (W_full, H_full),
                                interpolation=cv2.INTER_LINEAR)
    sm_vis = _score_map_vis(bgr, sm_blurred)

    return results, sm_vis


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def _rotated_corners(cx, cy, w, h, angle_deg):
    hw, hh = w/2, h/2
    pts = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    rad = math.radians(-angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    return [(cx + p[0]*ca - p[1]*sa,
             cy + p[0]*sa + p[1]*ca) for p in pts]


# ═══════════════════════════════════════════════════════════════
#  DRAW SCALING — tỷ lệ vẽ theo độ phân giải ảnh
# ═══════════════════════════════════════════════════════════════
# Kích thước tham chiếu (cạnh ngắn). Ảnh nhỏ hơn → scale = 1.0;
# ảnh lớn hơn → scale tăng theo tỷ lệ để bộ rộng nét & chữ không bị tí hon.
_DRAW_BASE_DIM = 720.0

def _draw_scale(image: np.ndarray) -> float:
    if image is None:
        return 1.0
    h, w = image.shape[:2]
    short = min(h, w)
    return max(1.0, short / _DRAW_BASE_DIM)

def _t(base: int, s: float) -> int:
    """Scaled line/box thickness (>=1)."""
    return max(1, int(round(base * s)))

def _fs(base: float, s: float) -> float:
    """Scaled font scale for cv2.putText."""
    return float(base) * s


def _score_map_vis(image: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    vis = image.copy() if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if score_map.max() > 0.01:
        sm_u8 = (np.clip(score_map / score_map.max(), 0, 1) * 255).astype(np.uint8)
        heat  = cv2.applyColorMap(sm_u8, cv2.COLORMAP_JET)
        cv2.addWeighted(vis, 0.55, heat, 0.45, 0, vis)
    return vis


def _empty_vis(image: np.ndarray) -> np.ndarray:
    vis = image.copy() if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print("[PatMax] No model trained — double-click node, draw ROI then Train")
    return vis


def draw_patmax_results(image: np.ndarray,
                         results: List[PatMaxResult],
                         model: Optional[PatMaxModel] = None,
                         show_reference: bool = True,
                         show_xy: Optional[bool] = None,
                         show_bbox: Optional[bool] = None) -> np.ndarray:
    """Render k\u1ebft qu\u1ea3 PatMax l\u00ean \u1ea3nh.

    show_xy=False    \u2192 \u1ea9n origin marker, X/Y axes, label "O (x,y)".
    show_bbox=False  \u2192 \u1ea9n rotated bounding box v\u00e0 score label.
    show_reference   \u2192 master toggle (legacy). Khi show_xy/show_bbox
                       kh\u00f4ng truy\u1ec1n v\u00e0o, m\u1eb7c \u0111\u1ecbnh = show_reference.
    """
    vis = image.copy() if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Resolve toggles: show_xy / show_bbox ri\u00eang l\u1ebb override show_reference
    if show_xy is None:
        show_xy = show_reference
    if show_bbox is None:
        show_bbox = show_reference
    if not show_xy and not show_bbox:
        print(f"[PatMax] {len(results)} result(s) (overlay hidden)")
        return vis
    s = _draw_scale(vis)

    # Origin offset t\u1eeb t\u00e2m pattern (to\u1ea1 \u0111\u1ed9 pattern, c\u00f3 th\u1ec3 \u00e2m ho\u1eb7c >w/h)
    has_origin = False
    pdx = pdy = 0.0
    if model is not None and model.is_valid():
        pw = float(model.pattern_w); ph = float(model.pattern_h)
        pdx = float(model.origin_x) - pw / 2.0
        pdy = float(model.origin_y) - ph / 2.0
        has_origin = True

    # Style \u2014 match PatMaxDialog preview (BGR)
    COL_X     = (70, 70, 255)     # \u0111\u1ecf
    COL_Y     = (100, 220, 80)    # xanh l\u00e1
    COL_CYAN  = (255, 212, 0)     # cyan
    COL_OMARK = (0, 215, 255)     # v\u00e0ng (X marker t\u1ea1i origin)

    for i, r in enumerate(results):
        if not r.found:
            continue
        col = (0,220,80) if r.score >= 0.7 else (0,200,160) if r.score >= 0.5 else (0,150,220)

        # Rotated bounding box (toggle: show_bbox)
        if show_bbox and r.corners and len(r.corners) == 4:
            pts = np.array(r.corners, dtype=np.int32)
            cv2.polylines(vis, [pts], True, col, _t(2, s))

        # T\u00ednh origin (transformed)
        if has_origin:
            rad_o = math.radians(-r.angle)
            ca = math.cos(rad_o); sa = math.sin(rad_o)
            sc = r.scale if r.scale else 1.0
            ox = float(r.x) + sc * (pdx * ca - pdy * sa)
            oy = float(r.y) + sc * (pdx * sa + pdy * ca)
        else:
            ox = float(r.x); oy = float(r.y)
        ox_i = int(round(ox)); oy_i = int(round(oy))

        if show_xy:
            # Marker origin: v\u00f2ng (theo score) + X v\u00e0ng \u0111\u00e8 l\u00ean + cyan dot t\u00e2m
            r_o  = _t(11, s)
            r_d  = _t(3, s)
            arm  = _t(9, s)
            cv2.circle(vis, (ox_i, oy_i), r_o, col, _t(2, s), cv2.LINE_AA)
            cv2.line(vis, (ox_i - arm, oy_i - arm), (ox_i + arm, oy_i + arm),
                     COL_OMARK, _t(2, s), cv2.LINE_AA)
            cv2.line(vis, (ox_i - arm, oy_i + arm), (ox_i + arm, oy_i - arm),
                     COL_OMARK, _t(2, s), cv2.LINE_AA)
            cv2.circle(vis, (ox_i, oy_i), r_d, COL_CYAN, -1, cv2.LINE_AA)

            # \u2500\u2500 H\u1ec7 tr\u1ee5c XY t\u1ea1i origin, xoay theo r.angle \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
            axis_len = max(_t(40, s), int(min(r.width, r.height) * 0.40))
            rad = math.radians(-r.angle)
            cax = math.cos(rad); sax = math.sin(rad)
            # X axis (\u0111\u1ecf): d\u1ecdc theo angle
            x_end = (int(ox_i + cax * axis_len), int(oy_i + sax * axis_len))
            cv2.arrowedLine(vis, (ox_i, oy_i), x_end, COL_X, _t(2, s),
                            cv2.LINE_AA, tipLength=0.18)
            cv2.putText(vis, "X", (x_end[0] + _t(4, s), x_end[1] + _t(4, s)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5, s), COL_X, _t(2, s),
                        cv2.LINE_AA)
            # Y axis (xanh l\u00e1): 90\u00b0 clockwise so v\u1edbi X trong image-space (Y\u2193)
            y_end = (int(ox_i - sax * axis_len), int(oy_i + cax * axis_len))
            cv2.arrowedLine(vis, (ox_i, oy_i), y_end, COL_Y, _t(2, s),
                            cv2.LINE_AA, tipLength=0.18)
            cv2.putText(vis, "Y", (y_end[0] + _t(4, s), y_end[1] + _t(4, s)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5, s), COL_Y, _t(2, s),
                        cv2.LINE_AA)

            # Label "O (x.x, y.y)  +angle" \u2014 cyan, ph\u00eda tr\u00ean-ph\u1ea3i origin
            ol_txt = f"O ({ox:.1f},{oy:.1f})"
            if abs(r.angle) > 0.05:
                ol_txt += f"  {r.angle:+.1f}deg"
            cv2.putText(vis, ol_txt,
                        (ox_i + int(14 * s), oy_i - int(10 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5, s), COL_CYAN,
                        _t(2, s), cv2.LINE_AA)

            # \u2500\u2500 Extra reference points (\u0111i\u1ec3m tham chi\u1ebfu b\u1ed5 sung) \u2500\u2500\u2500\u2500\u2500\u2500
            extra_refs = list(getattr(model, "extra_refs", []) or []) if model else []
            for ref in extra_refs:
                try:
                    rx_local = float(ref.get("x", 0.0))
                    ry_local = float(ref.get("y", 0.0))
                    r_name   = str(ref.get("name", "")).strip() or "R"
                    r_ang_off = float(ref.get("angle", 0.0))
                except (TypeError, ValueError):
                    continue
                # Transform ref t\u1eeb pattern-local v\u1ec1 image coords (gi\u1ed1ng origin)
                edx = rx_local - pw / 2.0
                edy = ry_local - ph / 2.0
                exf = float(r.x) + sc * (edx * ca - edy * sa)
                eyf = float(r.y) + sc * (edx * sa + edy * ca)
                ex_i = int(round(exf)); ey_i = int(round(eyf))
                # Marker: v\u00f2ng + X v\u00e0ng + cyan dot t\u00e2m (gi\u1ed1ng origin)
                cv2.circle(vis, (ex_i, ey_i), r_o, col, _t(2, s), cv2.LINE_AA)
                cv2.line(vis, (ex_i - arm, ey_i - arm), (ex_i + arm, ey_i + arm),
                         COL_OMARK, _t(2, s), cv2.LINE_AA)
                cv2.line(vis, (ex_i - arm, ey_i + arm), (ex_i + arm, ey_i - arm),
                         COL_OMARK, _t(2, s), cv2.LINE_AA)
                cv2.circle(vis, (ex_i, ey_i), r_d, COL_CYAN, -1, cv2.LINE_AA)
                # H\u1ec7 tr\u1ee5c XY t\u1ea1i ref, xoay theo (r.angle + ref.angle)
                total_ang = float(r.angle) + r_ang_off
                rad_e = math.radians(-total_ang)
                cae = math.cos(rad_e); sae = math.sin(rad_e)
                xe_end = (int(ex_i + cae * axis_len), int(ey_i + sae * axis_len))
                cv2.arrowedLine(vis, (ex_i, ey_i), xe_end, COL_X, _t(2, s),
                                cv2.LINE_AA, tipLength=0.18)
                cv2.putText(vis, "X", (xe_end[0] + _t(4, s), xe_end[1] + _t(4, s)),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5, s), COL_X,
                            _t(2, s), cv2.LINE_AA)
                ye_end = (int(ex_i - sae * axis_len), int(ey_i + cae * axis_len))
                cv2.arrowedLine(vis, (ex_i, ey_i), ye_end, COL_Y, _t(2, s),
                                cv2.LINE_AA, tipLength=0.18)
                cv2.putText(vis, "Y", (ye_end[0] + _t(4, s), ye_end[1] + _t(4, s)),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5, s), COL_Y,
                            _t(2, s), cv2.LINE_AA)
                # Label "Name (x, y) +angle"
                el_txt = f"{r_name} ({exf:.1f},{eyf:.1f})"
                if abs(total_ang) > 0.05:
                    el_txt += f"  {total_ang:+.1f}deg"
                cv2.putText(vis, el_txt,
                            (ex_i + int(14 * s), ey_i - int(10 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5, s), COL_CYAN,
                            _t(2, s), cv2.LINE_AA)

        if show_bbox:
            # Label score \u1edf g\u00f3c bbox
            lx = max(0, int(r.x - r.width/2))
            ly = max(int(16 * s), int(r.y - r.height/2) - int(8 * s))
            label = f"#{i+1} {r.score:.3f}"
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        _fs(0.55, s), col, _t(2, s))

    # Summary text \u0111\u01b0\u1ee3c log ra console; kh\u00f4ng v\u1ebd l\u00ean \u1ea3nh.
    print(f"[PatMax] {len(results)} result(s)")
    return vis


# ═══════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ═══════════════════════════════════════════════════════════════

def transform_ref_to_image(model: PatMaxModel, ref: dict,
                            result: PatMaxResult) -> Tuple[float, float, float]:
    """Transform 1 extra reference point từ pattern-local sang image coords
    sau khi pattern đã match. Trả (x_image, y_image, angle_total).
    angle_total = result.angle + ref.angle (degrees).
    """
    pw = float(model.pattern_w); ph = float(model.pattern_h)
    rx_local = float(ref.get("x", 0.0))
    ry_local = float(ref.get("y", 0.0))
    r_ang_off = float(ref.get("angle", 0.0))
    edx = rx_local - pw / 2.0
    edy = ry_local - ph / 2.0
    rad = math.radians(-float(result.angle))
    ca = math.cos(rad); sa = math.sin(rad)
    sc = float(result.scale) if result.scale else 1.0
    ex = float(result.x) + sc * (edx * ca - edy * sa)
    ey = float(result.y) + sc * (edx * sa + edy * ca)
    return ex, ey, float(result.angle) + r_ang_off


def save_model(model: PatMaxModel, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = os.path.splitext(path)[0]

    np_data = {}
    for attr in ("patch_bgr","patch_gray","edge_image","thumbnail","mask"):
        arr = getattr(model, attr, None)
        if arr is not None:
            np_data[attr] = arr
    if np_data:
        np.savez_compressed(base + ".npz", **np_data)

    meta = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in model.__dict__.items()
            if not isinstance(getattr(model,k,None), np.ndarray)
            and not isinstance(getattr(model,k,None), type(None))
            or k in ("train_roi",)}
    # Remove numpy arrays + non-serializable từ meta
    for key in ("patch_bgr","patch_gray","edge_image","thumbnail","mask",
                "precomputed_templates"):
        meta.pop(key, None)

    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(path: str) -> Optional[PatMaxModel]:
    base  = os.path.splitext(path)[0]
    jpath = base + ".json"
    npath = base + ".npz"
    if not os.path.exists(jpath):
        return None
    try:
        with open(jpath) as f:
            meta = json.load(f)
        model = PatMaxModel(
            trained          = meta.get("trained", False),
            train_roi        = tuple(meta["train_roi"]) if meta.get("train_roi") else None,
            origin_x         = meta.get("origin_x", 0.0),
            origin_y         = meta.get("origin_y", 0.0),
            pattern_w        = meta.get("pattern_w", 0),
            pattern_h        = meta.get("pattern_h", 0),
            edge_count       = meta.get("edge_count", 0),
            model_hash       = meta.get("model_hash", ""),
            accept_threshold = meta.get("accept_threshold", 0.5),
            angle_low        = meta.get("angle_low", 0.0),
            angle_high       = meta.get("angle_high", 0.0),
            angle_step       = meta.get("angle_step", 5.0),
            scale_low        = meta.get("scale_low", 1.0),
            scale_high       = meta.get("scale_high", 1.0),
            scale_step       = meta.get("scale_step", 0.1),
            num_results      = meta.get("num_results", 1),
            overlap_threshold= meta.get("overlap_threshold", 0.5),
            canny_low        = meta.get("canny_low", 50),
            canny_high       = meta.get("canny_high", 150),
            shape_type       = meta.get("shape_type", "rect"),
            shape_data       = meta.get("shape_data"),
            train_mode       = meta.get("train_mode", "evaluate"),
            extra_refs       = list(meta.get("extra_refs") or []),
        )
        if os.path.exists(npath):
            npz = np.load(npath)
            for attr in ("patch_bgr","patch_gray","edge_image","thumbnail","mask"):
                if attr in npz:
                    setattr(model, attr, npz[attr])
        # Regenerate precomputed templates nếu mode == "create"
        if model.train_mode == "create" and model.is_valid():
            n = precompute_templates(model)
            print(f"[PatMax Load] regenerated {n} DOF templates")
        return model
    except Exception as e:
        print(f"[PatMax] load_model error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  PATMAX ALIGN TOOL — Algorithm + Train Mode dispatcher
#  Cognex VisionPro xấp xỉ behavioral (engine có sẵn không phải proprietary).
# ═══════════════════════════════════════════════════════════════════

# (w_ncc, w_edge, w_sqdiff) cho từng algorithm — bám sát đặc tính public:
#   PatMax: edge-heavy (geometric)
#   PatQuick: NCC-heavy (raw intensity, nhanh)
#   PatMax & PatQuick: 2-pass (PatQuick coarse → PatMax fine)
#   High Sensitivity: cực edge-heavy + step nhỏ
#   PatFlex: như PatMax + cho phép cell-level offset (non-rigid xấp xỉ)
#   Perspective PatMax: như PatMax + 4-corner refinement (homography)
ALGO_WEIGHTS = {
    # (w_ncc, w_edge, w_sqdiff) — phải tổng ~1 để score nằm trong [0,1].
    # Cân bằng lại sau khi mask được truyền vào matchTemplate (NCC chính
    # xác hơn) → giảm w_edge, tăng w_ncc/w_sqdiff để score thực tế cao hơn
    # với cùng pattern. PatMax giữ tinh thần edge-heavy nhưng không cực đoan.
    "PatMax":                       (0.40, 0.40, 0.20),
    "PatQuick":                     (0.70, 0.15, 0.15),
    "PatMax & PatQuick":            (0.45, 0.35, 0.20),  # final fine-pass
    "PatFlex":                      (0.40, 0.40, 0.20),
    "PatMax - High Sensitivity":    (0.30, 0.55, 0.15),
    "Perspective PatMax":           (0.40, 0.40, 0.20),
}

# (angle_step_factor, scale_step_factor, peaks_factor)
ALGO_GRID = {
    "PatMax":                       (1.0, 1.0, 1.0),
    "PatQuick":                     (2.0, 2.0, 0.5),
    "PatMax & PatQuick":            (1.0, 1.0, 1.0),  # 2-pass managed bên dưới
    "PatFlex":                      (1.0, 1.0, 1.0),
    "PatMax - High Sensitivity":    (0.5, 0.5, 2.0),
    "Perspective PatMax":           (1.0, 1.0, 1.0),
}


def _shape_model_pair(model: PatMaxModel,
                      train_mode_align: str
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chọn cặp (patch_gray, edge_image) phù hợp với Train Mode.

    - "Image": dùng raw patch + edge gốc (default).
    - "Shape Models with Image": chuẩn hoá raw patch (CLAHE) để giảm
      ảnh hưởng intensity, edge giữ nguyên — engine vẫn xài cả 2.
    - "Shape Models with Transform": dùng edge làm 'patch' (giả grayscale),
      khiến NCC cũng so trên edge → robust tuyệt đối với lighting nhưng
      kén pattern có nhiều cạnh rõ.
    """
    p_gray = model.patch_gray
    p_edge = model.edge_image
    if p_gray is None or p_edge is None:
        return p_gray, p_edge

    if train_mode_align == "Shape Models with Image":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(p_gray), p_edge

    if train_mode_align == "Shape Models with Transform":
        # patch_gray ← edge dilated (giảm aliasing khi xoay)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_thick = cv2.dilate(p_edge, kernel)
        return edge_thick, p_edge

    return p_gray, p_edge


def _ensure_precomputed(model: PatMaxModel,
                        weights: Tuple[float, float, float],
                        patch_gray: np.ndarray,
                        edge_image: np.ndarray,
                        angle_low: float, angle_high: float, angle_step: float,
                        scale_low: float, scale_high: float, scale_step: float
                        ) -> List[Dict]:
    """Build (hoặc cache) precomputed templates cho path 'Shape Models with Transform'."""
    cache_key = (round(angle_low, 2), round(angle_high, 2), round(angle_step, 2),
                 round(scale_low, 3), round(scale_high, 3), round(scale_step, 3),
                 weights, tuple(patch_gray.shape))
    cached = getattr(model, "_align_tmpl_cache", None)
    if cached and cached.get("key") == cache_key:
        return cached["templates"]
    angles, scales = _build_search_grid(angle_low, angle_high, angle_step,
                                        scale_low, scale_high, scale_step)
    tmpls = [_build_template(patch_gray, edge_image, a, s, weights,
                              mask=model.mask)
             for s in scales for a in angles]
    model._align_tmpl_cache = {"key": cache_key, "templates": tmpls}
    print(f"[PatMax Align] precomputed {len(tmpls)} oriented templates "
          f"(angles={len(angles)} × scales={len(scales)})")
    return tmpls


def _refine_perspective(gray_img: np.ndarray,
                        model_patch: np.ndarray,
                        result: PatMaxResult,
                        max_iter: int = 6) -> PatMaxResult:
    """
    Perspective PatMax — refine 4 góc bằng local search homography.
    Mỗi corner thử ±3px; chọn config tăng NCC nhất. Lặp tối đa max_iter.
    Trả PatMaxResult với corners cập nhật (origin giữ nguyên).
    """
    if not result.corners or len(result.corners) != 4:
        return result
    H, W = gray_img.shape[:2]
    th, tw = model_patch.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    corners = [list(c) for c in result.corners]

    def _score(corners_xy):
        dst = np.float32(corners_xy)
        try:
            Hm = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return -1.0
        warped = cv2.warpPerspective(model_patch, Hm, (W, H),
                                     borderMode=cv2.BORDER_REPLICATE)
        # NCC trong bbox của corners
        xs = [int(p[0]) for p in corners_xy]
        ys = [int(p[1]) for p in corners_xy]
        x0 = max(0, min(xs)); x1 = min(W, max(xs))
        y0 = max(0, min(ys)); y1 = min(H, max(ys))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return -1.0
        a = gray_img[y0:y1, x0:x1].astype(np.float32)
        b = warped[y0:y1, x0:x1].astype(np.float32)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float((a * b).mean())

    best = _score(corners)
    if best < 0:
        return result
    step = 3
    for _ in range(max_iter):
        improved = False
        for i in range(4):
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                trial = [list(c) for c in corners]
                trial[i][0] += dx; trial[i][1] += dy
                s = _score(trial)
                if s > best + 1e-3:
                    best = s
                    corners = trial
                    improved = True
        if not improved:
            step = max(1, step // 2)
            if step == 1:
                # one more pass at finest step
                for i in range(4):
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        trial = [list(c) for c in corners]
                        trial[i][0] += dx; trial[i][1] += dy
                        s = _score(trial)
                        if s > best + 1e-4:
                            best = s
                            corners = trial
                break
    result.corners = [(float(c[0]), float(c[1])) for c in corners]
    # Cập nhật score (kết hợp với rigid score)
    result.score = max(result.score, float((result.score + best) / 2.0))
    return result


def _refine_patflex(gray_img: np.ndarray,
                    model: PatMaxModel,
                    result: PatMaxResult,
                    cells: int = 2) -> PatMaxResult:
    """
    PatFlex — chia template thành cells × cells ô và cho phép mỗi ô shift
    ±max_shift px tìm vị trí khớp tốt nhất; score = avg cell NCC.
    Xấp xỉ non-rigid matching. Chỉ áp dụng khi rigid result đã tốt.
    """
    patch = model.patch_gray
    if patch is None or result.score <= 0:
        return result
    th, tw = patch.shape[:2]
    H, W = gray_img.shape[:2]
    # Dựng patch transformed (xoay/scale theo result)
    rad = math.radians(-result.angle)
    s = result.scale or 1.0
    # Để đơn giản: làm việc với patch gốc (không xoay), giả định góc nhỏ
    cell_w = max(8, tw // cells)
    cell_h = max(8, th // cells)
    cx = result.x; cy = result.y
    # Top-left của bbox patch (chưa xoay)
    base_x = cx - tw / 2.0
    base_y = cy - th / 2.0
    max_shift = max(2, int(round(min(tw, th) * 0.08)))
    ncc_sum = 0.0; cnt = 0
    for cy_idx in range(cells):
        for cx_idx in range(cells):
            cx0 = int(cx_idx * cell_w); cy0 = int(cy_idx * cell_h)
            cw = min(cell_w, tw - cx0); ch = min(cell_h, th - cy0)
            if cw < 6 or ch < 6:
                continue
            tmpl = patch[cy0:cy0 + ch, cx0:cx0 + cw]
            # Vùng tìm trong ảnh quanh vị trí dự kiến
            sx0 = int(base_x + cx0 - max_shift)
            sy0 = int(base_y + cy0 - max_shift)
            sx1 = sx0 + cw + 2 * max_shift
            sy1 = sy0 + ch + 2 * max_shift
            sx0 = max(0, sx0); sy0 = max(0, sy0)
            sx1 = min(W, sx1); sy1 = min(H, sy1)
            if sx1 - sx0 <= cw or sy1 - sy0 <= ch:
                continue
            search = gray_img[sy0:sy1, sx0:sx1]
            try:
                r = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
                ncc_sum += float(r.max()); cnt += 1
            except cv2.error:
                continue
    if cnt > 0:
        flex_score = ncc_sum / cnt
        # Blend với rigid score (giữ rigid làm baseline)
        result.score = float(0.5 * result.score + 0.5 * max(result.score, flex_score))
    return result


def _auto_pyramid_level(image_shape: Tuple[int, int],
                        pattern_shape: Tuple[int, int],
                        requested_ds: int) -> int:
    """Pick pyramid downscale tự động (Cognex-style).

    Aim cho pattern dim nhỏ nhất ~48 px ở mức coarsest. Pattern lớn → ds lớn
    → match nhanh hơn ds² lần; refinement step bù sai số ±ds pixel.
    User-requested ds > 1 được tôn trọng nguyên gốc.
    """
    if requested_ds > 1:
        return requested_ds
    ph, pw = pattern_shape[:2]
    min_pat = max(8, min(pw, ph))
    target = 48
    if min_pat <= target * 1.5:
        return 1
    ds = min_pat // target
    if ds >= 4:
        return 4
    if ds >= 2:
        return 2
    return 1


def _refine_candidate(gray_full: np.ndarray,
                      model: PatMaxModel,
                      patch_gray_full: np.ndarray,
                      edge_image_full: np.ndarray,
                      mask_full: Optional[np.ndarray],
                      cand: Dict,
                      ds: int,
                      weights: Tuple[float, float, float],
                      ang_step_coarse: float,
                      sc_step_coarse: float,
                      channels: Tuple[bool, bool, bool],
                      canny_lo: int,
                      canny_hi: int,
                      angle_range: float = 0.0,
                      scale_range: float = 0.0) -> Dict:
    """Refine 1 candidate ở full-resolution (sau coarse search ds-downscale).

    cand: (cx, cy, nW, nH, angle, scale) đã map về full-res. Refine = crop
    window quanh candidate + match angle/scale lân cận. Trả best dict.

    angle_range, scale_range: total range user đã search ở coarse pass.
    Nếu range = 0 → chỉ refine translation (1 template duy nhất → cực nhanh).
    Nếu range > 0 → thêm ±half coarse step để bù quantization angle/scale.
    """
    H, W = gray_full.shape[:2]
    base_cx = float(cand["cx"]); base_cy = float(cand["cy"])
    base_angle = float(cand["angle"]); base_scale = float(cand["scale"])
    pw_full = patch_gray_full.shape[1]
    ph_full = patch_gray_full.shape[0]
    # Template bbox sau xoay (axis-aligned)
    rad = math.radians(base_angle)
    cos_a = abs(math.cos(rad)); sin_a = abs(math.sin(rad))
    nW_est = int(pw_full * base_scale * cos_a + ph_full * base_scale * sin_a) + 4
    nH_est = int(pw_full * base_scale * sin_a + ph_full * base_scale * cos_a) + 4
    pad = max(4, ds * 2 + 4)
    half_w = nW_est / 2.0 + pad
    half_h = nH_est / 2.0 + pad
    x0 = max(0, int(round(base_cx - half_w)))
    y0 = max(0, int(round(base_cy - half_h)))
    x1 = min(W, int(round(base_cx + half_w + 1)))
    y1 = min(H, int(round(base_cy + half_h + 1)))
    if x1 - x0 < nW_est + 4 or y1 - y0 < nH_est + 4:
        return cand
    sub_gray = gray_full[y0:y1, x0:x1]
    # Canny chỉ tính khi edge channel ON (tiết kiệm ~2-5ms với window 600×400)
    sub_edge = (cv2.Canny(sub_gray, canny_lo, canny_hi)
                if channels[1] else np.empty((0, 0), dtype=np.uint8))

    # Chỉ mở rộng angle/scale grid khi user thực sự đã search range > step.
    # Mặc định angle=0 / scale=1 → 1 template duy nhất, refinement = O(1).
    fine_a_step = max(0.5, ang_step_coarse * 0.5)
    fine_s_step = max(0.01, sc_step_coarse * 0.5)
    angles: List[float] = [base_angle]
    if angle_range > 0.5 and ang_step_coarse > 0.5:
        angles.extend([base_angle - fine_a_step, base_angle + fine_a_step])
    scales: List[float] = [base_scale]
    if scale_range > 0.01 and sc_step_coarse > 0.01:
        scales.extend([max(0.1, base_scale - fine_s_step),
                       base_scale + fine_s_step])

    best = dict(cand)
    for sc in scales:
        for ang in angles:
            t = _build_template(patch_gray_full, edge_image_full,
                                ang, sc, weights, mask=mask_full)
            t["_channels"] = channels
            res = _match_template(sub_gray, sub_edge, t, 1)
            if not res:
                continue
            r = res[0]
            if r["score"] >= best["score"]:
                r["cx"] += x0; r["cy"] += y0
                r["tx"] += x0; r["ty"] += y0
                best = r
    return best


def run_patmax_align(image: np.ndarray,
                     model: PatMaxModel,
                     *,
                     algorithm: str = "PatMax & PatQuick",
                     train_mode_align: str = "Image",
                     accept_threshold: float = 0.5,
                     angle_low: float = 0.0, angle_high: float = 0.0,
                     angle_step: float = 5.0,
                     scale_low: float = 1.0, scale_high: float = 1.0,
                     scale_step: float = 0.1,
                     num_results: int = 1,
                     overlap_threshold: float = 0.5,
                     coarse_downscale: int = 1,
                     build_score_map: bool = True,
                     ) -> Tuple[List[PatMaxResult], np.ndarray]:
    """Dispatcher cho 6 algorithm × 3 train mode của PatMax Align Tool.

    ``coarse_downscale`` (1/2/4): downscale image+patch+edge trước khi match.
    Tăng tốc ~ds² lần, sai số ±ds pixel. Kết quả x/y/w/h được scale về
    full-resolution trước khi trả về.
    """
    if not model.is_valid():
        return [], _empty_vis(image)

    weights = ALGO_WEIGHTS.get(algorithm, ALGO_WEIGHTS["PatMax"])
    a_fac, s_fac, peak_fac = ALGO_GRID.get(algorithm, ALGO_GRID["PatMax"])
    ang_step_eff = max(0.5, angle_step * a_fac)
    sc_step_eff  = max(0.01, scale_step * s_fac)

    # PatQuick: NCC-only (skip edge & SQDIFF) → ~3× nhanh hơn full 3-channel.
    # Các algorithm khác giữ full 3-channel để bám sát behavior Cognex.
    if algorithm == "PatQuick":
        channels = (True, False, False)
    else:
        channels = (True, True, True)

    # Train Mode → chọn cặp (patch, edge) phù hợp. GIỮ full-res references
    # cho bước refinement sau coarse search.
    patch_full, edge_full = _shape_model_pair(model, train_mode_align)
    if patch_full is None or edge_full is None:
        return [], _empty_vis(image)
    mask_full = getattr(model, "mask", None)

    bgr      = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H_full, W_full = gray_full.shape

    # Auto-pyramid: nếu user để ds=1 và ảnh đủ lớn → tự chọn ds 2/4 để giảm
    # ~ds² lần chi phí matchTemplate. Refinement bước sau sẽ bù sai số.
    requested_ds = max(1, int(coarse_downscale))
    ds = _auto_pyramid_level(gray_full.shape, patch_full.shape, requested_ds)

    if ds > 1:
        gray_img = cv2.resize(gray_full, (W_full // ds, H_full // ds),
                              interpolation=cv2.INTER_AREA)
        ph2 = max(4, patch_full.shape[0] // ds)
        pw2 = max(4, patch_full.shape[1] // ds)
        patch_gray = cv2.resize(patch_full, (pw2, ph2), interpolation=cv2.INTER_AREA)
        edge_image = cv2.resize(edge_full, (pw2, ph2), interpolation=cv2.INTER_AREA)
        if mask_full is not None:
            try:
                model = copy.copy(model)
                model.mask = cv2.resize(mask_full, (pw2, ph2),
                                        interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass
    else:
        gray_img = gray_full
        patch_gray, edge_image = patch_full, edge_full
    H, W = gray_img.shape
    # Bỏ qua Canny nếu edge channel OFF (PatQuick) — tiết kiệm trên ảnh lớn.
    if channels[1]:
        img_edges = cv2.Canny(gray_img, model.canny_low, model.canny_high)
    else:
        img_edges = np.empty((0, 0), dtype=np.uint8)

    # ─── Build templates: precompute nếu Shape Models with Transform ───
    if train_mode_align == "Shape Models with Transform":
        templates = _ensure_precomputed(
            model, weights, patch_gray, edge_image,
            angle_low, angle_high, ang_step_eff,
            scale_low, scale_high, sc_step_eff)
    else:
        angles, scales = _build_search_grid(angle_low, angle_high, ang_step_eff,
                                            scale_low, scale_high, sc_step_eff)
        templates = [_build_template(patch_gray, edge_image, a, s, weights,
                                       mask=model.mask)
                     for s in scales for a in angles]

    # Tag templates với channels để _match_template biết bỏ qua edge/SQDIFF.
    for _t in templates:
        _t["_channels"] = channels

    print(f"[PatMax Align] algorithm={algorithm}  train_mode={train_mode_align}  "
          f"weights={weights}  channels={channels}  templates={len(templates)}  "
          f"ds={ds} (req={requested_ds})  "
          f"step(ang={ang_step_eff:.2f}, sc={sc_step_eff:.3f})")

    # ─── Special path: PatMax & PatQuick — 2-pass coarse→fine ───
    # Khi user không search angle/scale, 2-pass thoái hoá thành 1-pass (1
    # template duy nhất) — không cần seeds/refine, dùng generic path luôn.
    ang_range_user = max(0.0, angle_high - angle_low)
    sc_range_user  = max(0.0, scale_high - scale_low)
    use_two_pass = (algorithm == "PatMax & PatQuick"
                    and (ang_range_user > 0.5 or sc_range_user > 0.01))
    if use_two_pass:
        # Pass 1: PatQuick (NCC-only, coarse) — single channel cho tốc độ
        wq = ALGO_WEIGHTS["PatQuick"]
        ang_step_coarse = max(1.0, ang_step_eff * 2.0)
        sc_step_coarse  = max(0.05, sc_step_eff * 2.0)
        ang_q, sc_q = _build_search_grid(angle_low, angle_high, ang_step_coarse,
                                         scale_low, scale_high, sc_step_coarse)
        tmpls_q = [_build_template(patch_gray, edge_image, a, s, wq,
                                     mask=model.mask)
                   for s in sc_q for a in ang_q]
        for _t in tmpls_q:
            _t["_channels"] = (True, False, False)   # NCC-only cho coarse pass
        cands_q: List[Dict] = []
        for t in tmpls_q:
            cands_q.extend(_match_template(gray_img, img_edges, t,
                                           max(num_results * 2, 4)))
        cands_q.sort(key=lambda d: -d["score"])
        # Pick TOP-K seeds (loose threshold)
        seeds = []
        loose = max(0.1, accept_threshold * 0.6)
        for c in cands_q:
            if c["score"] < loose:
                break
            ok = True
            for s_seed in seeds:
                d = math.hypot(c["cx"] - s_seed["cx"], c["cy"] - s_seed["cy"])
                if d < min(c["nW"], c["nH"]) * 0.5:
                    ok = False; break
            if ok:
                seeds.append(c)
            if len(seeds) >= max(num_results + 1, 3):   # giới hạn seed để fine pass nhanh
                break
        print(f"[PatMax Align] PatQuick pass: {len(cands_q)} cands, "
              f"{len(seeds)} seeds")
        # Pass 2: PatMax fine — quanh seed angles ±0.5*ang_step_coarse
        # (đủ phủ vùng quantization của coarse mà không over-search).
        candidates: List[Dict] = []
        score_map = np.zeros((H, W), dtype=np.float32)
        best_any = 0.0
        # Fine grid: ±half coarse step, step = ang_step_eff → tối đa ~3×3=9 tpls/seed
        for seed in seeds:
            seed_a = seed["angle"]; seed_s = seed["scale"]
            a_lo = seed_a - ang_step_coarse * 0.5
            a_hi = seed_a + ang_step_coarse * 0.5
            s_lo = max(scale_low, seed_s - sc_step_coarse * 0.5)
            s_hi = min(scale_high, seed_s + sc_step_coarse * 0.5)
            ang_f, sc_f = _build_search_grid(a_lo, a_hi, ang_step_eff,
                                             s_lo, s_hi, sc_step_eff)
            for sc in sc_f:
                for ang in ang_f:
                    t = _build_template(patch_gray, edge_image, ang, sc, weights,
                                          mask=model.mask)
                    t["_channels"] = channels   # PatMax fine: full 3-channel
                    res_t = _match_template(gray_img, img_edges, t,
                                            max(2, int(num_results * peak_fac) + 1))
                    for r in res_t:
                        # Chỉ giữ peaks gần seed center (giảm drift)
                        dpx = math.hypot(r["cx"] - seed["cx"], r["cy"] - seed["cy"])
                        if dpx > min(r["nW"], r["nH"]) * 0.6:
                            continue
                        best_any = max(best_any, r["score"])
                        cy_i = min(int(r["cy"]), H - 1)
                        cx_i = min(int(r["cx"]), W - 1)
                        if score_map[cy_i, cx_i] < r["s_ncc"]:
                            score_map[cy_i, cx_i] = r["s_ncc"]
                        if r["score"] >= accept_threshold:
                            candidates.append(r)
        print(f"[PatMax Align] PatMax fine pass: best={best_any:.4f}  "
              f"cands={len(candidates)}")
    else:
        # ─── Generic 1-pass path (PatMax / PatQuick / HiSens / PatFlex / Perspective) ───
        candidates: List[Dict] = []
        score_map = np.zeros((H, W), dtype=np.float32)
        best_any = 0.0
        peaks_per_template = max(int(num_results * 4 * peak_fac), 8)
        for t in templates:
            results_t = _match_template(gray_img, img_edges, t, peaks_per_template)
            for r in results_t:
                best_any = max(best_any, r["score"])
                if (0 <= r["ty"] < H - r["nH"]) and (0 <= r["tx"] < W - r["nW"]):
                    cy_i = min(int(r["cy"]), H - 1)
                    cx_i = min(int(r["cx"]), W - 1)
                    if score_map[cy_i, cx_i] < r["s_ncc"]:
                        score_map[cy_i, cx_i] = r["s_ncc"]
                if r["score"] >= accept_threshold:
                    candidates.append(r)
        print(f"[PatMax Align] best={best_any:.4f}  "
              f"cands_above_th={len(candidates)}")

    # ─── NMS ───
    candidates.sort(key=lambda d: -d["score"])
    kept: List[Dict] = []
    for cand in candidates:
        overlap = False
        for k in kept:
            dist = math.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"])
            min_dim = min(cand["nW"], cand["nH"], k["nW"], k["nH"])
            if dist < min_dim * overlap_threshold:
                overlap = True; break
        if not overlap:
            kept.append(cand)
        if len(kept) >= num_results:
            break

    # Map ds→full-res cho mọi toạ độ trước khi build result/origin
    if ds > 1:
        for d in kept:
            d["cx"] *= ds; d["cy"] *= ds
            d["tx"] *= ds; d["ty"] *= ds
            d["nW"] *= ds; d["nH"] *= ds

    # ─── Full-resolution refinement ───
    # Sau coarse search ds-downscale, toạ độ trên có sai số ±ds px. Refine
    # mỗi candidate bằng cách crop window quanh nó ở gray_full và chạy
    # _match_template với patch full-res. Window nhỏ → rẻ nhưng đưa độ chính
    # xác về sub-pixel (≈ Cognex pyramid refinement).
    if ds > 1 and kept:
        refined: List[Dict] = []
        ang_range_used = max(0.0, angle_high - angle_low)
        sc_range_used  = max(0.0, scale_high - scale_low)
        for k in kept:
            r = _refine_candidate(
                gray_full, model,
                patch_full, edge_full, mask_full,
                k, ds, weights,
                ang_step_eff, sc_step_eff,
                channels,
                model.canny_low, model.canny_high,
                angle_range=ang_range_used,
                scale_range=sc_range_used,
            )
            refined.append(r)
        kept = refined

    # Build PatMaxResult list (origin-aware như run_patmax)
    pdx = float(model.origin_x) - float(model.pattern_w) / 2.0
    pdy = float(model.origin_y) - float(model.pattern_h) / 2.0
    results: List[PatMaxResult] = []
    for d in kept:
        corners = _rotated_corners(d["cx"], d["cy"], d["nW"], d["nH"], d["angle"])
        rad_o = math.radians(-d["angle"])
        ca = math.cos(rad_o); sa = math.sin(rad_o)
        s = d["scale"] if d["scale"] else 1.0
        ox_t = float(d["cx"]) + s * (pdx * ca - pdy * sa)
        oy_t = float(d["cy"]) + s * (pdx * sa + pdy * ca)
        results.append(PatMaxResult(
            found=True, score=d["score"],
            x=d["cx"], y=d["cy"],
            angle=d["angle"], scale=d["scale"],
            width=float(d["nW"]), height=float(d["nH"]),
            corners=corners,
            origin_x=ox_t, origin_y=oy_t,
        ))

    # ─── Algorithm-specific refinement ───
    # Refine luôn ở full-res — gray_full mới đủ chi tiết để tinh chỉnh.
    refine_gray = gray_full if ds > 1 else gray_img
    if results and algorithm == "Perspective PatMax":
        for i, r in enumerate(results):
            results[i] = _refine_perspective(refine_gray, model.patch_gray, r)
    elif results and algorithm == "PatFlex":
        for i, r in enumerate(results):
            results[i] = _refine_patflex(refine_gray, model, r)

    # Skip score-map visualization khi caller không cần — heatmap blending
    # tốn ~90ms với ảnh 8MP. proc_patmax_align discard sm_vis nên đặt
    # build_score_map=False ở production path.
    if build_score_map:
        sm_blurred = cv2.GaussianBlur(score_map, (15, 15), 0)
        if ds > 1:
            sm_blurred = cv2.resize(sm_blurred, (W_full, H_full),
                                    interpolation=cv2.INTER_LINEAR)
        sm_vis = _score_map_vis(bgr, sm_blurred)
    else:
        sm_vis = bgr
    return results, sm_vis
