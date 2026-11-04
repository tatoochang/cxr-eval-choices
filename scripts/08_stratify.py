#!/usr/bin/env python3
"""Step 8 (optional). Lesion characteristics and recall by size, side and lung zone (Table 8), and the share of
false positives outside the dilated lung mask (Results 3.3). Needs the lung masks of step 7.

Internal: consensus lesions of the 750 cross-validation radiographs, IoU >= 0.5, at the 1 FP/image threshold.
External (optional): localisation criterion at the external 1 FP/image threshold.

Usage: python scripts/08_stratify.py --work work --cv detections/cv.json.gz --masks work/lungmask/vindr \
          [--ext-dets detections/external_nih_original.json.gz --ext-labels <dir> --ext-masks <dir>]
Output: work/results/stratified.json
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import config as C, froc as F, io  # noqa: E402
from cxreval.fusion import ensemble_wbf, nms, tiles_to_image  # noqa: E402

SIZE_BINS = [(0, 20, "<20"), (20, 40, "20-39"), (40, 80, "40-79"), (80, 1e9, ">=80")]


def dilated(mask, d=C.LUNG_DILATE_PX):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    return cv2.dilate(mask, k) > 0


def inside(m, b):
    cx, cy = int(min(1023, max(0, (b[0] + b[2]) / 2))), int(min(1023, max(0, (b[1] + b[3]) / 2)))
    return bool(m[cy, cx])


def fp_outside(dets, G, ids, maskdir):
    """False positives (IoU >= 0.5 criterion) at the 1 FP/image threshold whose centre is outside the dilated lung mask."""
    t = F.operating_points(dets, G, ids, "iou05", (1.0,))[0]["threshold"]
    fp = out = 0
    for i in ids:
        m = dilated(cv2.imread(str(Path(maskdir) / f"{i}.png"), 0))
        P = sorted([d for d in dets[i] if d[4] >= t], key=lambda d: -d[4])
        for d, (_, tp) in zip(P, F.match(P, G[i], "iou05")[0]):
            if tp == 0:
                fp += 1
                out += not inside(m, d)
    return {"threshold": t, "fp": fp, "fp_outside_lung": out}


def stratify(dets, G, ids, rule, maskdir):
    t = F.operating_points(dets, G, ids, rule, (1.0,))[0]["threshold"]
    rows = []
    for i in ids:
        raw = cv2.imread(str(Path(maskdir) / f"{i}.png"), 0)
        m = dilated(raw)
        hit = F.match([d for d in dets[i] if d[4] >= t], G[i], rule)[1]
        ys = np.where((raw > 0).any(1))[0]
        y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, 1023)
        for j, g in enumerate(G[i]):
            s = ((g[2] - g[0]) * (g[3] - g[1])) ** 0.5
            z = ((g[1] + g[3]) / 2 - y0) / max(y1 - y0, 1)
            rows.append({"size": next(n for a, b, n in SIZE_BINS if a <= s < b),
                         "side": "image_left" if (g[0] + g[2]) / 2 < 512 else "image_right",
                         "zone": "upper" if z < 1 / 3 else "middle" if z < 2 / 3 else "lower",
                         "outside_lung": not inside(m, g), "hit": hit[j]})
    df = pd.DataFrame(rows)
    out = {"threshold": t, "rule": rule, "n": int(len(df)), "recall": float(df.hit.mean()),
           "lesions_per_image": dict(Counter(str(len(G[i])) for i in ids if G[i]))}
    for col in ("size", "side", "zone", "outside_lung"):
        out[col] = {str(k): {"n": int(len(g)), "pct": float(100 * len(g) / len(df)), "recall": float(g.hit.mean())}
                    for k, g in df.groupby(col)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--cv", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--ext-dets")
    ap.add_argument("--ext-labels")
    ap.add_argument("--ext-masks")
    a = ap.parse_args()
    W = Path(a.work)
    D = io.read_json(a.cv)
    det = {i: nms(tiles_to_image({k: [tuple(x) for x in v] for k, v in e["tiles"].items()}), C.TILE_FUSION_IOU)
           for m in D["models"].values() for i, e in m.items()}
    ids = sorted(det)
    IG = io.load_labels(W / "labels/consensus", ids)
    R = {"internal": stratify(det, IG, ids, "iou05", a.masks) | {"fp_outside": fp_outside(det, IG, ids, a.masks)}}
    if a.ext_dets:
        E = io.read_json(a.ext_dets)
        eids = E["ids"]
        single = [{i: nms(tiles_to_image({k: [tuple(x) for x in v] for k, v in E["models"][str(f)][i]["tiles"].items()}), C.TILE_FUSION_IOU)
                   for i in eids} for f in range(C.N_MODELS)]
        ens = {i: ensemble_wbf([s[i] for s in single]) for i in eids}
        EG = io.load_labels(a.ext_labels, eids)
        R["external"] = stratify(ens, EG, eids, "centre_dup", a.ext_masks) | {"fp_outside": fp_outside(ens, EG, eids, a.ext_masks)}
    io.write_json(W / "results/stratified.json", R, indent=1)
    for k, v in R.items():
        print(k, "recall %.3f" % v["recall"], {z: round(x["recall"], 2) for z, x in v["zone"].items()},
              "FP outside lung %d/%d" % (v["fp_outside"]["fp_outside_lung"], v["fp_outside"]["fp"]))


if __name__ == "__main__":
    main()
