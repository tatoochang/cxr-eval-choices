import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import annotations as A, froc as F  # noqa: E402
from cxreval.fusion import ensemble_wbf, nms, tiles_to_image, wbf  # noqa: E402
from cxreval.tiling import boxes_in_tile  # noqa: E402

GT = [(100, 100, 140, 140)]


def test_iou_vs_centre():
    tight = (110, 110, 130, 130, 0.9)            # centred on the lesion, IoU 0.25
    assert F.match([tight], GT, "iou05")[0] == [(0.9, 0)]
    assert F.match([tight], GT, "centre")[0] == [(0.9, 1)]


def test_duplicates_ignored_only_by_centre_dup():
    d = [(100, 100, 140, 140, 0.9), (105, 105, 135, 135, 0.8)]
    assert F.match(d, GT, "centre")[0] == [(0.9, 1), (0.8, 0)]
    assert F.match(d, GT, "centre_dup")[0] == [(0.9, 1), (0.8, None)]


def test_recall_interpolation():
    assert F.recall_at_fp([0.0, 2.0], [0.2, 0.6], 1.0) == 0.4


def test_froc_counts_every_image():
    dets = {"a": [(100, 100, 140, 140, 0.9)], "b": [(0, 0, 10, 10, 0.95)]}
    gts = {"a": GT, "b": []}
    TP, FP, GT_ = F.counts(dets, gts, ["a", "b"])
    assert GT_.tolist() == [1, 0] and FP[1, 0] == 1 and TP[0, 0] == 1


def test_fusion():
    tiles = {"0": [(300, 300, 340, 340, 0.8)], "1": [(44, 300, 84, 340, 0.6)]}   # same lesion seen by two tiles
    raw = tiles_to_image(tiles)
    assert len(raw) == 2 and len(nms(raw)) == 1
    assert wbf(raw, 0.5, "max")[0][4] == 0.8 and abs(wbf(raw, 0.5, "avg")[0][4] - 0.7) < 1e-9
    assert abs(ensemble_wbf([[raw[0]], [], [], [], []])[0][4] - 0.16) < 1e-9


def test_tile_rule():
    assert boxes_in_tile([(500, 100, 540, 140)], 0, 0) == []          # 30 % inside tile 0
    assert boxes_in_tile([(490, 100, 520, 140)], 0, 0) == [(490, 100, 512, 140)]


def test_reader_fusion_counts_readers():
    out = A.fuse_reader_boxes({"R1": [(0, 0, 10, 10)], "R2": [(1, 1, 10, 10)], "R3": [(50, 50, 60, 60)]})
    assert sorted(o[4] for o in out) == [1, 2]


def test_split_is_stable():
    assert A.split_of("0005e8e3701dfb1dd93d53e2ff537b6e") in ("test", "trainval")
    f = A.make_folds([f"id{i:03d}" for i in range(750)])
    assert np.bincount(list(f.values())).tolist() == [150] * 5
