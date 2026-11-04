"""Reader-box fusion, curation of regional boxes, reference standards and data splits (VinDr-CXR).

VinDr-CXR provides boxes drawn independently by three radiologists per radiograph, without adjudication.
  1. boxes of the different readers that overlap at IoU >= 0.5 are fused into one lesion (weighted boxes
     fusion of reader boxes, Solovyev et al. 2021); each lesion keeps the number of readers who marked it;
  2. regional boxes drawn around a whole lung field are removed by geometric rules (see `curate`);
  3. any-reader reference = every curated lesion; consensus reference = lesions marked by >= 2 readers
     (fusion and curation repeated on that subset).
"""
import hashlib
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- reader-box fusion
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fuse_reader_boxes(boxes_by_reader, iou_thr=C.READER_WBF_IOU):
    """{reader_id: [(x1,y1,x2,y2)]} -> [(x1,y1,x2,y2,n_readers)].
    Larger boxes seed clusters; a fused box is the mean of its members."""
    entries = [(box, src) for src, boxes in boxes_by_reader.items() for box in boxes]
    if not entries:
        return []
    entries.sort(key=lambda e: -((e[0][2] - e[0][0]) * (e[0][3] - e[0][1])))
    clusters = []
    for box, src in entries:
        best_idx, best_iou = -1, iou_thr
        for i, cl in enumerate(clusters):
            score = iou(cl["fused"], box)
            if score >= best_iou:
                best_idx, best_iou = i, score
        if best_idx < 0:
            clusters.append({"fused": box, "members": [box], "sources": {src}})
        else:
            cl = clusters[best_idx]
            cl["members"].append(box)
            cl["sources"].add(src)
            m = cl["members"]
            cl["fused"] = tuple(sum(b[j] for b in m) / len(m) for j in range(4))
    return [(*cl["fused"], len(cl["sources"])) for cl in clusters]


def fuse_dataframe(df, iou_thr=C.READER_WBF_IOU):
    """df columns: image_id, rad_id, x_min, y_min, x_max, y_max (original DICOM pixels)."""
    records = []
    for image_id, group in df.groupby("image_id", sort=True):
        by_rad = defaultdict(list)
        for row in group.itertuples(index=False):
            by_rad[row.rad_id].append((float(row.x_min), float(row.y_min), float(row.x_max), float(row.y_max)))
        for x1, y1, x2, y2, n in fuse_reader_boxes(by_rad, iou_thr):
            records.append({"image_id": image_id, "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                            "n_rads": n, "n_rads_total": len(by_rad)})
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------- curation of regional boxes
REASONS = {
    "degenerate": "zero or negative area",
    "extreme_ar": "aspect ratio > 6 or < 1/6",
    "encloses": "encloses >= 2 other boxes of the same radiograph (regional box)",
    "field_box": "one side > 45 % of the image (hemithorax-sized)",
    "oversize": "equivalent side > 30 % of the image (about 10 cm)",
}


def annotate_geometry(fused, dims):
    """Adds relative size and the number of enclosed boxes. dims: image_id, rows, cols."""
    df = fused.merge(dims, on="image_id", how="left", validate="many_to_one")
    df["bw"] = df.x_max - df.x_min
    df["bh"] = df.y_max - df.y_min
    df["rel_w"] = df.bw / df.cols
    df["rel_h"] = df.bh / df.rows
    df["rel_eq"] = np.sqrt(df.bw * df.bh) / np.sqrt(df.cols * df.rows)
    df["area_frac"] = (df.bw * df.bh) / (df.cols * df.rows)
    df["ar"] = df.bw / df.bh.replace(0, np.nan)
    enclosed = np.zeros(len(df), dtype=int)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, grp in df.groupby("image_id", sort=False):
        boxes = grp[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        for a in range(len(grp)):
            ax1, ay1, ax2, ay2 = boxes[a]
            n = 0
            for b in range(len(grp)):
                if a == b:
                    continue
                bx1, by1, bx2, by2 = boxes[b]
                if (ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2
                        and (ax2 - ax1) * (ay2 - ay1) > 2 * (bx2 - bx1) * (by2 - by1)):
                    n += 1
            enclosed[pos[grp.index[a]]] = n
    df["n_enclosed"] = enclosed
    return df


def curate(df, max_rel_eq=C.CURATE_MAX_REL_EQ, max_side=C.CURATE_MAX_SIDE, max_ar=C.CURATE_MAX_AR,
           max_enclosed=C.CURATE_MAX_ENCLOSED):
    """-> (kept, removed with a 'reason' column). The first matching reason (in REASONS order) is recorded."""
    reason = pd.Series(pd.NA, index=df.index, dtype="object")

    def mark(mask, key):
        reason.loc[mask & reason.isna()] = key

    mark((df.bw <= 0) | (df.bh <= 0), "degenerate")
    mark((df.ar > max_ar) | (df.ar < 1 / max_ar), "extreme_ar")
    mark(df.n_enclosed >= max_enclosed, "encloses")
    mark((df.rel_w > max_side) | (df.rel_h > max_side), "field_box")
    mark(df.rel_eq > max_rel_eq, "oversize")
    removed = df[reason.notna()].copy()
    removed["reason"] = reason[reason.notna()]
    return df[reason.isna()].copy(), removed


def build_reference(raw, dims, min_readers=1):
    """raw: class-8 reader boxes; dims: image_id, rows, cols. -> (kept lesions, removed boxes)."""
    fused = fuse_dataframe(raw)
    if min_readers > 1:
        fused = fused[fused["n_rads"] >= min_readers].copy()
    return curate(annotate_geometry(fused, dims))


# ---------------------------------------------------------------- coordinates
def letterbox_params(orig_h, orig_w, target=C.IMG_SIZE):
    """Same scale for x and y, image centred: canvas = pixel * scale + pad."""
    scale = target / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    return scale, (target - new_w) // 2, (target - new_h) // 2


def to_yolo(box, rows, cols, size=C.IMG_SIZE):
    """Original-pixel (x1,y1,x2,y2) -> YOLO (cx,cy,w,h) on the letterboxed canvas."""
    scale, pad_x, pad_y = letterbox_params(rows, cols, size)
    x1, y1 = box[0] * scale + pad_x, box[1] * scale + pad_y
    x2, y2 = box[2] * scale + pad_x, box[3] * scale + pad_y
    clamp = lambda v, lo=0.0: max(lo, min(1.0, v))
    return (clamp((x1 + x2) / 2 / size), clamp((y1 + y2) / 2 / size),
            clamp(abs(x2 - x1) / size, 1e-6), clamp(abs(y2 - y1) / size, 1e-6))


def yolo_lines(kept_for_image, rows, cols):
    return ["0 %.6f %.6f %.6f %.6f" % to_yolo((b.x_min, b.y_min, b.x_max, b.y_max), rows, cols)
            for b in kept_for_image.itertuples(index=False)]


# ---------------------------------------------------------------- splits
def split_of(image_id):
    """Held-out assignment by hash (independent of labels and of any random seed)."""
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 100
    return "test" if h >= 100 - C.TEST_PERCENT else "trainval"


def make_folds(trainval_ids):
    """-> {image_id: fold}. Sorted ids shuffled with random.Random(42); consecutive blocks of 150."""
    ids = sorted(trainval_ids)
    rng = random.Random(C.FOLD_SEED)
    rng.shuffle(ids)
    size = len(ids) // C.N_FOLDS
    return {iid: min(k // size, C.N_FOLDS - 1) for k, iid in enumerate(ids)}
