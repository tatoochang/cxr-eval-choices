"""Hit criteria, lesion-level FROC, bootstrap confidence intervals and operating points.

Matching is greedy in decreasing confidence; each reference box can be matched once.
  iou05       hit if IoU >= 0.5 with an unmatched reference box (pre-specified criterion)
  iou03       same with IoU >= 0.3
  centre      hit if the detection centre lies inside an unmatched reference box
  centre_dup  as 'centre', and further detections centred on an already found lesion are ignored
              (neither TP nor FP). This is the localisation criterion of the paper.
Every radiograph contributes false positives, including those without a reference lesion.
"""
import numpy as np

from . import config as C
from .fusion import iou_xyxy

RULES = ("iou05", "iou03", "centre", "centre_dup")


def centre_in(p, g):
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]


def match(dets, gts, rule="iou05"):
    """-> ([(conf, 1 | 0 | None)], hit flags per reference box). None = ignored duplicate."""
    hit = [False] * len(gts)
    out = []
    for p in sorted(dets, key=lambda d: -d[4]):
        best, bj = -1.0, -1
        for j, g in enumerate(gts):
            if hit[j]:
                continue
            if rule in ("iou05", "iou03"):
                v = iou_xyxy(p, g)
                if v >= (0.5 if rule == "iou05" else 0.3) and v > best:
                    best, bj = v, j
            elif centre_in(p, g):
                v = iou_xyxy(p, g)
                if v > best:
                    best, bj = v, j
        if bj >= 0:
            hit[bj] = True
            out.append((p[4], 1))
        elif rule == "centre_dup" and any(hit[j] and centre_in(p, g) for j, g in enumerate(gts)):
            out.append((p[4], None))
        else:
            out.append((p[4], 0))
    return out, hit


def per_image_counts(labeled, thresholds=C.THRESHOLDS):
    thr = np.asarray(thresholds)
    tp, fp = np.zeros(len(thr)), np.zeros(len(thr))
    for conf, t in labeled:
        if t is None:
            continue
        m = thr <= conf
        if t:
            tp += m
        else:
            fp += m
    return tp, fp


def counts(dets_by_img, gt_by_img, ids, rule="iou05"):
    """-> TP [N, T], FP [N, T], GT [N] over the 128 confidence thresholds."""
    TP, FP, GT = [], [], []
    for i in ids:
        tp, fp = per_image_counts(match(dets_by_img[i], gt_by_img[i], rule)[0])
        TP.append(tp); FP.append(fp); GT.append(len(gt_by_img[i]))
    return np.asarray(TP), np.asarray(FP), np.asarray(GT)


def recall_at_fp(avg_fp, recall, target):
    """Linear interpolation of recall at the target mean number of FP per image."""
    pts = sorted(zip(avg_fp, recall), key=lambda x: x[0])
    for j in range(len(pts) - 1):
        f1, f2 = pts[j][0], pts[j + 1][0]
        if f1 <= target <= f2:
            if f2 > f1:
                w = (target - f1) / (f2 - f1)
                return float(pts[j][1] + w * (pts[j + 1][1] - pts[j][1]))
            return float(pts[j][1])
    return float(min(pts, key=lambda x: abs(x[0] - target))[1])


def recalls(TP, FP, GT, idx=None, targets=C.FP_TARGETS):
    if idx is None:
        idx = np.arange(len(GT))
    r = TP[idx].sum(0) / GT[idx].sum()
    a = FP[idx].sum(0) / len(idx)
    return [recall_at_fp(a, r, k) for k in targets]


def resample(n, groups, rng):
    """Image-level resample, or cluster (patient-level) resample when groups are given."""
    if groups is None:
        return rng.integers(0, n, n)
    u = np.unique(groups)
    mem = [np.where(groups == g)[0] for g in u]
    return np.concatenate([mem[k] for k in rng.integers(0, len(u), len(u))])


def froc(TP, FP, GT, groups=None, n_boot=C.N_BOOT, seed=C.BOOT_SEED, targets=C.FP_TARGETS):
    """Recall at each target FP rate with bootstrap 95 % CI (2.5th / 97.5th percentiles)."""
    pt = recalls(TP, FP, GT, targets=targets)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = resample(len(GT), groups, rng)
        if GT[idx].sum() > 0:
            bs.append(recalls(TP, FP, GT, idx, targets))
    bs = np.asarray(bs)
    return {"n_images": int(len(GT)), "n_lesions": int(GT.sum()),
            "recall": {str(k): float(pt[n]) for n, k in enumerate(targets)},
            "ci95": {str(k): [float(np.percentile(bs[:, n], 2.5)), float(np.percentile(bs[:, n], 97.5))]
                     for n, k in enumerate(targets)}}


def paired_diff(A, B, n_boot=C.N_BOOT, seed=C.BOOT_SEED, targets=C.FP_TARGETS):
    """A, B = (TP, FP, GT) for the same images scored two ways; CI of recall(A) - recall(B)."""
    pa, pb = recalls(*A, targets=targets), recalls(*B, targets=targets)
    rng = np.random.default_rng(seed)
    N = len(A[2])
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if A[2][idx].sum() == 0 or B[2][idx].sum() == 0:
            continue
        d.append(np.subtract(recalls(*A, idx, targets), recalls(*B, idx, targets)))
    d = np.asarray(d)
    return {str(k): {"diff": float(pa[n] - pb[n]), "ci95": [float(np.percentile(d[:, n], 2.5)), float(np.percentile(d[:, n], 97.5))]}
            for n, k in enumerate(targets)}


def operating_points(dets_by_img, gt_by_img, ids, rule="iou05", targets=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """For each target: the lowest threshold whose mean FP per image does not exceed the target."""
    TP, FP, GT = counts(dets_by_img, gt_by_img, ids, rule)
    tp, fp = TP.sum(0), FP.sum(0)
    afp = fp / len(ids)
    rows = []
    for k in targets:
        j = min((j for j in range(len(afp)) if afp[j] <= k), key=lambda j: C.THRESHOLDS[j])
        rows.append({"fp_target": k, "threshold": C.THRESHOLDS[j], "recall": float(tp[j] / GT.sum()),
                     "ppv": float(tp[j] / max(tp[j] + fp[j], 1)),
                     "marks_per_100": float(100 * (tp[j] + fp[j]) / len(ids)),
                     "pct_radiographs_marked": float(100 * ((TP[:, j] + FP[:, j]) > 0).sum() / len(ids))})
    return rows


def ci_width_vs_n(TP, FP, GT, sizes=(50, 100, 150, 200, 300, 400, 500, 750), draws=600, seed=1):
    """Width of the 95 % interval for recall at 1 FP per image when the validation set has n radiographs."""
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        vals = []
        for _ in range(draws):
            idx = rng.choice(len(GT), n, replace=True)
            if GT[idx].sum() == 0:
                continue
            vals.append(recall_at_fp(FP[idx].sum(0) / n, TP[idx].sum(0) / GT[idx].sum(), 1.0))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out.append({"n": n, "width": float(hi - lo)})
    return out
