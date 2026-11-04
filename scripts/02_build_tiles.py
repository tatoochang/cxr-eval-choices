#!/usr/bin/env python3
"""Step 2. Training tiles for each fold: 9 tiles of 512 px (stride 256) per radiograph, any-reader labels.

Output: work/tiles/fold<k>/{images,labels}/{train,val}/<id>_t<k>.{jpg,txt} and work/tiles/fold<k>/data.yaml
Each fold has 600 x 9 = 5,400 training and 150 x 9 = 1,350 validation tiles.

Usage: python scripts/02_build_tiles.py --work work
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import io  # noqa: E402
from cxreval.tiling import clip_to_image, write_tiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    W = Path(a.work).resolve()
    sp = pd.read_csv(W / "splits.csv")
    tv = sp[sp.split == "trainval"]
    for k in map(int, a.folds.split(",")):
        root = W / "tiles" / f"fold{k}"
        for part in ("train", "val"):
            (root / "images" / part).mkdir(parents=True, exist_ok=True)
            (root / "labels" / part).mkdir(parents=True, exist_ok=True)
        for r in tv.itertuples(index=False):
            part = "val" if int(r.fold) == k else "train"
            img = cv2.imread(str(W / "images" / f"{r.image_id}.png"))
            boxes = clip_to_image(io.load_yolo_xyxy(W / "labels/any_reader" / f"{r.image_id}.txt"))
            write_tiles(img, r.image_id, root / "images" / part, root / "labels" / part, boxes)
        (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: nodule_mass\n")
        print(f"fold {k}: {len(list((root / 'images/train').glob('*.jpg')))} train / "
              f"{len(list((root / 'images/val').glob('*.jpg')))} val tiles", flush=True)


if __name__ == "__main__":
    main()
