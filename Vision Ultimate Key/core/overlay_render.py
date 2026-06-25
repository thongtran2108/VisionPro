"""
core/overlay_render.py
Render ảnh COMPOSITE = ảnh gốc (Acquire Image) + overlay annotation của các
tool đã chọn. Dùng CHUNG cho:
  • Image Viewer  — hiển thị "Results" overlay trên ảnh gốc.
  • Save Image    — lưu ra file ĐÚNG NHƯ đang thấy ở Image Viewer.

Tách ra core để 2 nơi xài chung 1 thuật toán → ảnh lưu giống hệt ảnh xem.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def vis_of(node):
    """Frame có annotation để render cho node: `_display_image` (overlay riêng)
    trước, fallback `image` (port clean). `or` không an toàn với numpy →
    dùng is-None."""
    v = node.outputs.get("_display_image")
    if v is None:
        v = node.outputs.get("image")
    return v


def acquire_root_image(graph):
    """Output 'image' của node Acquire Image đầu tiên (ảnh gốc/base)."""
    for nid, n in graph.nodes.items():
        if getattr(n.tool, "category", "") == "Acquire Image" \
                and "image" in n.outputs:
            return n.outputs["image"]
    return None


def node_input_image(graph, node):
    """Output 'image' của upstream gần nhất nối vào port 'image' của node."""
    for c in graph.connections:
        if c.dst_id == node.node_id and c.dst_port == "image":
            src = graph.nodes.get(c.src_id)
            if src and "image" in src.outputs:
                return src.outputs["image"]
    return None


def node_pipeline_root(graph, node_id: str) -> Optional[str]:
    """Acquire/Camera root mà node thuộc về — đi ngược upstream theo port
    'image'. None nếu node không thuộc pipeline Acquire nào."""
    if node_id not in graph.nodes:
        return None
    roots = {nid for nid, n in graph.nodes.items()
             if getattr(n.tool, "category", "") == "Acquire Image"}
    if node_id in roots:
        return node_id
    visited = set()
    cur = node_id
    for _ in range(64):
        if cur in visited:
            break
        visited.add(cur)
        if cur in roots:
            return cur
        upstream = None
        for c in graph.connections:
            if c.dst_id == cur and c.dst_port == "image":
                upstream = c.src_id
                break
        if upstream is None:
            return None
        cur = upstream
    return None


def overlay_diff(base: np.ndarray, before: np.ndarray,
                 after: np.ndarray) -> np.ndarray:
    """Compose pixel khác biệt (before→after) lên base. Lấy annotation mà
    upstream-tool đã vẽ rồi áp lên ảnh hiển thị.

    before/after có thể khác channel count (vd PatMax nhận gray HxW, output
    BGR HxWx3); chỉ cần match H,W rồi up-convert gray → BGR trước absdiff.
    """
    if (before is None or after is None
            or before.shape[:2] != after.shape[:2]
            or before.shape[:2] != base.shape[:2]):
        return base
    import cv2
    b = before if before.ndim == 3 else cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    a = after if after.ndim == 3 else cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Pixel coi như annotation nếu lệch ≥ 20 (loại noise nhỏ)
    mask = (gray > 20)
    if not mask.any():
        return base
    out = base.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out[mask] = a[mask]
    return out


def overlayable_nodes(graph, exclude_id: Optional[str] = None) -> List[Tuple[str, object]]:
    """List (node_id, node) các tool có thể overlay — node có output image,
    KHÔNG thuộc category 'Acquire Image' (đó là base), và khác exclude_id.
    Dùng cho picker 'Results' của Save Image. Kiểm tra output PORT (static)
    để hiện ngay cả khi pipeline chưa chạy."""
    out: List[Tuple[str, object]] = []
    for nid, n in graph.nodes.items():
        if nid == exclude_id:
            continue
        if getattr(n.tool, "category", "") == "Acquire Image":
            continue
        if any(p.name == "image" for p in n.tool.outputs):
            out.append((nid, n))
    return out


def render_composite(graph, overlay_node_ids: List[str],
                     base_image: Optional[np.ndarray] = None
                     ) -> Optional[np.ndarray]:
    """Ảnh COMPOSITE = base (ảnh gốc Acquire) + overlay annotation của mỗi
    node trong overlay_node_ids — GIỐNG HỆT Image Viewer "Results".

    base:
      • base_image nếu truyền vào, ngược lại
      • ảnh gốc của pipeline chứa overlay đầu tiên (hỗ trợ multi-camera),
        fallback Acquire root đầu tiên trong graph.

    Trả ảnh BGR (numpy) hoặc None nếu không có base.
    """
    overlay_node_ids = [nid for nid in (overlay_node_ids or [])
                        if nid in graph.nodes]

    base = base_image
    if base is None:
        # Base theo pipeline của overlay đầu tiên (multi-camera đúng nhánh).
        for nid in overlay_node_ids:
            root = node_pipeline_root(graph, nid)
            if root is not None and "image" in graph.nodes[root].outputs:
                base = graph.nodes[root].outputs["image"]
                break
    if base is None:
        base = acquire_root_image(graph)
    if base is None or not isinstance(base, np.ndarray):
        return None

    import cv2
    comp = base.copy()
    if comp.ndim == 2:
        comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
    for nid in overlay_node_ids:
        n = graph.nodes.get(nid)
        if n is None:
            continue
        before = node_input_image(graph, n)
        after = vis_of(n)
        if before is not None and after is not None:
            comp = overlay_diff(comp, before, after)
    return comp
