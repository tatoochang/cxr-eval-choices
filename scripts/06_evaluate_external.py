#!/usr/bin/env python3
"""Step 6. External validation from stored detections (CPU only).

The external reference boxes of the paper (1,280 lesions on 465 NIH ChestX-ray14 radiographs, drawn by
radiologists of a tertiary medical centre) are not public; they are available from the corresponding author
on reasonable request. With them, or with your own external set, run:

  python scripts/06_evaluate_external.py --dets detections/external_nih_original.json.gz \
      --labels <dir with <image_id>.txt YOLO boxes on the 1024 canvas> --nih-patients --out work/results/external.json

  --nih-patients   NIH image ids are <patient>_<follow-up>; use the first 8 characters as patient id
  --patients CSV   otherwise: columns image_id, patient_id (patient-level bootstrap); omit for image-level

Sections: ensemble (IoU 0.5 and localisation, patient-level CI; Table 6), single_models (mean +- SD),
replication (fusion rules and single-tile protocol per fold model; Table 4), box_area_ratio (Fig. 7c),
size_tertiles and lesions_per_image (Results 3.7).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import config as C, froc as F, io  # noqa: E402
from cxreval.fusion import TILE_FUSIONS, ensemble_wbf, iou_xyxy, nms, tiles_to_image  # noqa: E402
from cxreval.tiling import boxes_in_tile  # noqa: E402


def threshold_at(dets, G, ids, rule, target=1.0):
    return [r for r in F.operating_points(dets, G, ids, rule, (target,))][0]["threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--patients", default=None)
    ap.add_argument("--nih-patients", action="store_true")
    ap.add_argument("--out", default="work/results/external.json")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    D = io.read_json(a.dets)
    ids = D["ids"]
    G = io.load_labels(a.labels, ids)
    if a.nih_patients:
        groups = np.asarray([i[:8] for i in ids])
    elif a.patients:
        pm = pd.read_csv(a.patients).set_index("image_id").patient_id.astype(str).to_dict()
        groups = np.asarray([pm[i] for i in ids])
    else:
        groups = None

    raw = {f: {i: tiles_to_image({k: [tuple(x) for x in v] for k, v in D["models"][str(f)][i]["tiles"].items()}) for i in ids}
           for f in range(C.N_MODELS)}
    single = {f: {i: nms(raw[f][i], C.TILE_FUSION_IOU) for i in ids} for f in range(C.N_MODELS)}
    ens = {i: ensemble_wbf([single[f][i] for f in range(C.N_MODELS)]) for i in ids}
    R = {"n_images": len(ids), "n_patients": int(len(set(groups))) if groups is not None else None,
         "n_lesions": int(sum(len(g) for g in G.values()))}

    R["ensemble"] = {rule: F.froc(*F.counts(ens, G, ids, rule), groups=groups, n_boot=a.n_boot) for rule in F.RULES}
    R["single_models"] = {}
    for rule in ("iou05", "centre_dup"):
        v = np.asarray([F.recalls(*F.counts(single[f], G, ids, rule)) for f in range(C.N_MODELS)])
        R["single_models"][rule] = {str(k): [float(v[:, n].mean()), float(v[:, n].std(ddof=1))] for n, k in enumerate(C.FP_TARGETS)}

    # replication of the internal methodological findings, per fold model
    G0 = {i: boxes_in_tile(G[i], 0, 0) for i in ids}
    R["replication"] = {"tile0_lesions": int(sum(len(g) for g in G0.values()))}
    for rule in ("iou05", "centre_dup"):
        rows = {}
        for name, fn in TILE_FUSIONS.items():
            v = np.asarray([F.recalls(*F.counts({i: fn(raw[f][i]) for i in ids}, G, ids, rule)) for f in range(C.N_MODELS)])
            rows[name] = {str(k): [float(v[:, n].mean()), float(v[:, n].std(ddof=1))] for n, k in enumerate(C.FP_TARGETS)}
        v = np.asarray([F.recalls(*F.counts({i: [tuple(x) for x in D["models"][str(f)][i]["tiles"]["0"]] for i in ids}, G0, ids, rule))
                        for f in range(C.N_MODELS)])
        rows["Single tile (FP per tile)"] = {str(k): [float(v[:, n].mean()), float(v[:, n].std(ddof=1))] for n, k in enumerate(C.FP_TARGETS)}
        R["replication"][rule] = rows

    # reference box area / detection box area on lesions found by the centre rule (first containing box)
    t = threshold_at(ens, G, ids, "iou05")
    pairs = []
    for i in ids:
        used = [False] * len(G[i])
        for p in sorted([p for p in ens[i] if p[4] >= t], key=lambda p: -p[4]):
            cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            for j, g in enumerate(G[i]):
                if not used[j] and g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
                    used[j] = True
                    pairs.append(((g[2] - g[0]) * (g[3] - g[1]) / ((p[2] - p[0]) * (p[3] - p[1])), iou_xyxy(p, g)))
                    break
    pairs = np.asarray(pairs)
    R["box_area_ratio"] = {"threshold": t, "n": int(len(pairs)), "median": float(np.median(pairs[:, 0])),
                           "iqr": np.percentile(pairs[:, 0], [25, 75]).tolist(), "share_iou_ge_0.5": float(np.mean(pairs[:, 1] >= 0.5))}

    # recall by lesion size tertile and by number of lesions per radiograph, at the 1 FP/image centre threshold
    tc = threshold_at(ens, G, ids, "centre_dup")
    rows = []
    for i in ids:
        used = [False] * len(G[i])
        for p in sorted([d for d in ens[i] if d[4] >= tc], key=lambda d: -d[4]):
            cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            for j, g in enumerate(G[i]):
                if not used[j] and g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
                    used[j] = True
                    break
        rows += [(((g[2] - g[0]) * (g[3] - g[1])) ** 0.5, u) for g, u in zip(G[i], used)]
    s, h = np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows])
    cuts = np.quantile(s, [1 / 3, 2 / 3])
    R["size_tertiles"] = {"threshold": tc, "cuts_px": cuts.tolist(), "bins": [
        {"name": n, "n": int(m.sum()), "median_side_px": float(np.median(s[m])), "recall": float(h[m].mean())}
        for n, m in (("small", s < cuts[0]), ("medium", (s >= cuts[0]) & (s < cuts[1])), ("large", s >= cuts[1]))]}
    # recall at 1 FP per image computed within each subgroup of radiographs (localisation criterion)
    many = [i for i in ids if len(G[i]) > 3]
    few = [i for i in ids if len(G[i]) <= 3]
    rec = lambda L: F.recalls(*F.counts(ens, G, L, "centre_dup"), targets=[1.0])[0]
    R["lesions_per_image"] = {"gt3": {"radiographs": len(many), "recall_at_1fp": rec(many)},
                              "le3": {"radiographs": len(few), "recall_at_1fp": rec(few)}}

    io.write_json(a.out, R, indent=1)
    for rule in ("iou05", "centre_dup"):
        e = R["ensemble"][rule]
        print(f"{rule:10s} @1FP {e['recall']['1.0']:.3f} [{e['ci95']['1.0'][0]:.3f}, {e['ci95']['1.0'][1]:.3f}]  @4FP {e['recall']['4.0']:.3f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
