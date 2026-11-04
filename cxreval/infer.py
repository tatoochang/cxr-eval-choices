"""Tile inference with an Ultralytics RT-DETR model, and the image-storage degradation."""
import tempfile
from pathlib import Path

import cv2

from . import config as C
from .tiling import tile_offsets


def degrade(img, cond):
    """'none' | 'down512' | 'down512_jpeg75': 1024 -> 512 (INTER_AREA) [-> JPEG q75] -> 1024 (bilinear)."""
    if cond == "none":
        return img
    s = cv2.resize(img, (C.DEGRADE_SIZE, C.DEGRADE_SIZE), interpolation=cv2.INTER_AREA)
    if cond == "down512_jpeg75":
        s = cv2.imdecode(cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, C.DEGRADE_JPEG_QUALITY])[1], cv2.IMREAD_COLOR)
    return cv2.resize(s, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def predict_tile(model, path, device="0"):
    r = model.predict(str(path), imgsz=C.INFER["imgsz"], conf=C.INFER["conf"], iou=C.INFER["iou"],
                      max_det=C.INFER["max_det"], augment=False, verbose=False, device=device)[0]
    out = []
    if r.boxes is not None and len(r.boxes):
        for b, cf in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist()):
            cx, cy, w, h = [v * C.TILE for v in b]
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(float(C.TILE), cx + w / 2), min(float(C.TILE), cy + h / 2)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, float(cf)))
    return out


def predict_radiograph(model, img, device="0", tmpdir=None):
    """1024 x 1024 BGR image -> {tile k: [tile-local detections]}.
    Tiles are written as JPEG q95 first, exactly as during training and in the paper."""
    tmpdir = Path(tmpdir or tempfile.mkdtemp())
    per_tile = {}
    for k, ox, oy in tile_offsets():
        p = tmpdir / f"tile{k}.jpg"
        cv2.imwrite(str(p), img[oy:oy + C.TILE, ox:ox + C.TILE], [cv2.IMWRITE_JPEG_QUALITY, C.TILE_JPEG_QUALITY])
        per_tile[str(k)] = predict_tile(model, p, device)
    return per_tile
