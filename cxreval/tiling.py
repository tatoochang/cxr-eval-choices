"""3 x 3 overlapping 512-px tiles of a 1024-px radiograph and the box-to-tile rule."""
import cv2

from . import config as C


def tile_offsets():
    """[(k, ox, oy)] for k = 0..8, row-major (k = ty * 3 + tx)."""
    return [(ty * C.GRID + tx, tx * C.STRIDE, ty * C.STRIDE) for ty in range(C.GRID) for tx in range(C.GRID)]


def boxes_in_tile(boxes, ox, oy):
    """Image boxes [(x1,y1,x2,y2)] -> tile-local boxes. A box is kept, clipped to the tile, when at least
    50 % of its area lies inside the tile and its shorter clipped side is >= 5 px."""
    kept = []
    for x1, y1, x2, y2 in boxes:
        cx1, cy1 = max(x1, ox), max(y1, oy)
        cx2, cy2 = min(x2, ox + C.TILE), min(y2, oy + C.TILE)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if ((cx2 - cx1) * (cy2 - cy1) >= C.TILE_KEEP_AREA * (x2 - x1) * (y2 - y1)
                and min(cx2 - cx1, cy2 - cy1) >= C.TILE_MIN_SIDE):
            kept.append((cx1 - ox, cy1 - oy, cx2 - ox, cy2 - oy))
    return kept


def clip_to_image(boxes, W=1024, H=1024):
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def write_tiles(img, stem, img_dir, lab_dir=None, boxes=None):
    """Writes {stem}_t{k}.jpg (JPEG q95) and, if boxes are given, the matching YOLO tile labels."""
    for k, ox, oy in tile_offsets():
        cv2.imwrite(str(img_dir / f"{stem}_t{k}.jpg"), img[oy:oy + C.TILE, ox:ox + C.TILE].copy(),
                    [cv2.IMWRITE_JPEG_QUALITY, C.TILE_JPEG_QUALITY])
        if lab_dir is not None:
            lines = [f"0 {(a + c) / 2 / C.TILE:.6f} {(b + d) / 2 / C.TILE:.6f} {(c - a) / C.TILE:.6f} {(d - b) / C.TILE:.6f}\n"
                     for a, b, c, d in boxes_in_tile(boxes or [], ox, oy)]
            (lab_dir / f"{stem}_t{k}.txt").write_text("".join(lines))
