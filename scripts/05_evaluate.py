#!/usr/bin/env python3
"""Step 5. All internal analyses of the paper from stored detections (CPU only, a few minutes).

  python scripts/05_evaluate.py --work work --cv detections/cv.json.gz --heldout detections/heldout.json.gz \
      [--degraded down512=detections/cv_down512.json.gz --degraded down512_jpeg75=detections/cv_down512_jpeg75.json.gz]

Results go to work/results/internal.json. Section -> paper:
  reference_standard    fixed predictions scored against any-reader and consensus boxes (Results 3.1, Fig. 3)
  cv                    pooled and per-fold recall, NMS fusion, consensus, IoU >= 0.5 (Table 3)
  fusion_rules          tile-fusion comparison and the single-tile protocol (Table 4)
  hit_criteria          IoU 0.5 / IoU 0.3 / centre / centre with duplicates ignored (Table 6, Fig. 7)
  operating_points      threshold, recall, PPV, marks per 100 radiographs (Table 7)
  ci_width_vs_n         width of the 95 % CI against validation-set size
  heldout               five-model ensemble and single models on the 70 held-out radiographs (Table 6)
  degraded              image-storage experiment on the internal radiographs (Table 6)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import config as C, froc as F, io  # noqa: E402
from cxreval.fusion import TILE_FUSIONS, ensemble_wbf, nms, tiles_to_image  # noqa: E402
from cxreval.tiling import boxes_in_tile, clip_to_image  # noqa: E402


def as_tuples(x):
    return [tuple(d) for d in x]


def fused(entry, fn=lambda r: nms(r, C.TILE_FUSION_IOU)):
    """One radiograph entry of a detection file -> fused radiograph detections."""
    if "tiles" in entry:
        return fn(tiles_to_image({k: as_tuples(v) for k, v in entry["tiles"].items()}))
    return as_tuples(entry["nms"])


def cv_dets(D, fn=None):
    """Each validation radiograph scored by its own fold model."""
    out, fold_of = {}, {}
    for f, m in D["models"].items():
        for i, e in m.items():
            out[i] = fused(e) if fn is None else fused(e, fn)
            fold_of[i] = int(f)
    return out, fold_of


def ens_dets(D):
    ids = D["ids"]
    pm = {f: {i: fused(D["models"][str(f)][i]) for i in ids} for f in range(C.N_MODELS)}
    return {i: ensemble_wbf([pm[f][i] for f in range(C.N_MODELS)]) for i in ids}, pm


def single_tile(D, G):
    """Legacy protocol: only tile 0 (top left) scored against tile-clipped boxes; FP per tile."""
    det, gt = {}, {}
    for f, m in D["models"].items():
        for i, e in m.items():
            det[i] = as_tuples(e["tiles"]["0"])
            gt[i] = boxes_in_tile(clip_to_image(G[i]), 0, 0)
    return det, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--cv", required=True)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--degraded", action="append", default=[], help="name=path of a cv detection file")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    W = Path(a.work)
    sp = pd.read_csv(W / "splits.csv")
    R = {}

    # ------------------------------------------------ cross-validation radiographs
    D = io.read_json(a.cv)
    det, fold_of = cv_dets(D)
    # fold-major order (fold 0 ids sorted, then fold 1, ...): the bootstrap resamples positions, so this
    # order is needed to reproduce the reported intervals exactly
    ids = sorted(det, key=lambda i: (fold_of[i], i))
    G = {r: io.load_labels(W / "labels" / r, ids) for r in ("consensus", "any_reader")}
    cnt = {r: F.counts(det, G[r], ids) for r in G}
    R["reference_standard"] = {r: F.froc(*cnt[r], n_boot=a.n_boot) for r in G}
    R["reference_standard"]["paired_diff_consensus_minus_any"] = F.paired_diff(cnt["consensus"], cnt["any_reader"], n_boot=a.n_boot)

    folds = np.asarray([fold_of[i] for i in ids])
    TP, FP, GT = cnt["consensus"]
    R["cv"] = {"pooled": R["reference_standard"]["consensus"], "per_fold": {}}
    for f in range(C.N_MODELS):
        m = folds == f
        R["cv"]["per_fold"][str(f)] = F.froc(TP[m], FP[m], GT[m], n_boot=a.n_boot)
    per = [R["cv"]["per_fold"][str(f)]["recall"]["1.0"] for f in range(C.N_MODELS)]
    R["cv"]["fold_mean_sd_at_1fp"] = [float(np.mean(per)), float(np.std(per))]

    R["fusion_rules"] = {}
    for rule in ("iou05", "centre_dup"):
        rows = {}
        for name, fn in TILE_FUSIONS.items():
            d, _ = cv_dets(D, fn)
            rows[name] = F.froc(*F.counts(d, G["consensus"], ids, rule), n_boot=a.n_boot)["recall"]
        d0, g0 = single_tile(D, G["consensus"])
        st = F.froc(*F.counts(d0, g0, ids, rule), n_boot=a.n_boot)
        rows["Single tile (FP per tile)"] = st["recall"]
        rows["single_tile_lesions"] = st["n_lesions"]
        R["fusion_rules"][rule] = rows

    R["hit_criteria"] = {rule: F.froc(*F.counts(det, G["consensus"], ids, rule), n_boot=a.n_boot) for rule in F.RULES}
    R["operating_points"] = {rule: F.operating_points(det, G["consensus"], ids, rule) for rule in ("iou05", "centre_dup")}
    R["ci_width_vs_n"] = F.ci_width_vs_n(TP, FP, GT)

    # ------------------------------------------------ image-storage experiment
    if a.degraded:
        R["degraded"] = {}
        for spec in a.degraded:
            name, path = spec.split("=", 1)
            d, _ = cv_dets(io.read_json(path))
            # this analysis was run with the ids in plain sorted order (affects only the bootstrap intervals)
            R["degraded"][name] = {rule: F.froc(*F.counts(d, G["consensus"], sorted(ids), rule), n_boot=a.n_boot)
                                   for rule in ("iou05", "centre_dup")}

    # ------------------------------------------------ held-out radiographs
    if a.heldout:
        H = io.read_json(a.heldout)
        hids = H["ids"]
        assert sorted(hids) == sorted(sp.image_id[sp.split == "test"]), "held-out ids do not match splits.csv"
        ens, pm = ens_dets(H)
        R["heldout"] = {}
        for r in ("consensus", "any_reader"):
            HG = io.load_labels(W / "labels" / r, hids)
            R["heldout"][r] = {"ensemble": {rule: F.froc(*F.counts(ens, HG, hids, rule), n_boot=a.n_boot) for rule in ("iou05", "centre_dup")},
                               "single_models_at_1fp": [F.froc(*F.counts(pm[f], HG, hids), n_boot=10)["recall"]["1.0"] for f in range(C.N_MODELS)]}

    io.write_json(W / "results/internal.json", R, indent=1)
    rs = R["reference_standard"]
    print("reference standard @1FP: any-reader %.3f, consensus %.3f, diff %+.3f" % (
        rs["any_reader"]["recall"]["1.0"], rs["consensus"]["recall"]["1.0"], rs["paired_diff_consensus_minus_any"]["1.0"]["diff"]))
    c = R["cv"]["pooled"]
    print("CV pooled @1FP %.3f [%.3f, %.3f], @4FP %.3f" % (c["recall"]["1.0"], *c["ci95"]["1.0"], c["recall"]["4.0"]))
    if "heldout" in R:
        h = R["heldout"]["consensus"]["ensemble"]["iou05"]
        print("held-out @1FP %.3f [%.3f, %.3f]" % (h["recall"]["1.0"], *h["ci95"]["1.0"]))
    print("saved", W / "results/internal.json")


if __name__ == "__main__":
    main()
