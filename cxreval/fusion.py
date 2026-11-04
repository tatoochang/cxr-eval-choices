"""Fusion of tile detections into radiograph detections, and of the five fold models into an ensemble.
A detection is a tuple (x1, y1, x2, y2, conf) in 1024-px radiograph coordinates."""
import numpy as np

from . import config as C


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tiles_to_image(dets_per_tile):
    """{k: [tile-local dets]} -> all detections in radiograph coordinates (no fusion)."""
    out = []
    for k, dl in dets_per_tile.items():
        k = int(k)
        ox, oy = (k % C.GRID) * C.STRIDE, (k // C.GRID) * C.STRIDE
        out += [(x1 + ox, y1 + oy, x2 + ox, y2 + oy, c) for x1, y1, x2, y2, c in dl]
    return out


def nms(dets, iou_thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(iou_xyxy(d, k) <= iou_thr for k in keep):
            keep.append(d)
    return keep


def wbf(dets, iou_thr=0.5, conf_type="max"):
    """Single-model weighted boxes fusion (Solovyev et al. 2021). conf_type 'max' or 'avg' (= mean)."""
    if not dets:
        return []
    clusters, fused = [], []
    for b in sorted(dets, key=lambda d: -d[4]):
        best_j, best_iou = -1, iou_thr
        for j, fb in enumerate(fused):
            v = iou_xyxy(b, fb)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            clusters[best_j].append(b)
            cl = clusters[best_j]
            ws = sum(x[4] for x in cl)
            fused[best_j] = [sum(x[k] * x[4] for x in cl) / ws for k in range(4)]
        else:
            clusters.append([b])
            fused.append(list(b[:4]))
    out = []
    for cl, fb in zip(clusters, fused):
        confs = [x[4] for x in cl]
        out.append((fb[0], fb[1], fb[2], fb[3], float(np.mean(confs)) if conf_type == "avg" else float(max(confs))))
    return out


TILE_FUSIONS = {
    "NMS, IoU 0.5": lambda r: nms(r, 0.5),
    "NMS, IoU 0.3": lambda r: nms(r, 0.3),
    "NMS, IoU 0.1": lambda r: nms(r, 0.1),
    "WBF-max, IoU 0.5": lambda r: wbf(r, 0.5, "max"),
    "WBF-max, IoU 0.3": lambda r: wbf(r, 0.3, "max"),
    "WBF-mean, IoU 0.5": lambda r: wbf(r, 0.5, "avg"),
    "No fusion": lambda r: list(r),
}


def ensemble_wbf(model_dets, iou_thr=0.5, n_models=C.N_MODELS):
    """Cross-model WBF. model_dets: one list of detections per model.
    Fused confidence = sum over models of that model's highest confidence in the cluster / n_models."""
    allb = sorted(((d, m) for m, ds in enumerate(model_dets) for d in ds), key=lambda t: -t[0][4])
    clusters, fused = [], []
    for b, m in allb:
        bj, bi = -1, iou_thr
        for j, fb in enumerate(fused):
            v = iou_xyxy(b, fb)
            if v > bi:
                bi, bj = v, j
        if bj >= 0:
            clusters[bj].append((b, m))
            ws = sum(x[4] for x, _ in clusters[bj])
            fused[bj] = [sum(x[k] * x[4] for x, _ in clusters[bj]) / ws for k in range(4)]
        else:
            clusters.append([(b, m)])
            fused.append(list(b[:4]))
    out = []
    for cl, fb in zip(clusters, fused):
        best = {}
        for x, m in cl:
            best[m] = max(best.get(m, 0.0), x[4])
        out.append((*fb, sum(best.values()) / n_models))
    return out
