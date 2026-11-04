#!/usr/bin/env python3
"""Step 4. Tile inference with the five fold models. Writes raw per-tile detections (confidence >= 0.001);
all fusion and scoring happens in step 5, without a GPU.

  --set cv        each fold model on its own 150 validation radiographs (750 in total)
  --set heldout   all five models on the 70 held-out radiographs
  --set external  all five models on every *.png / *.jpg in --images-dir (1024 x 1024 radiographs)
  --degrade       none | down512 | down512_jpeg75   image-storage experiment (applied before tiling)

Output (gzipped JSON): {"ids": [...], "models": {"0": {image_id: {"tiles": {"0": [[x1,y1,x2,y2,conf], ...], ...}}}}}
Tile boxes are in tile-local pixels; tile k has offset ((k % 3) * 256, (k // 3) * 256).

Usage: python scripts/04_infer.py --work work --set cv --out work/detections/cv.json.gz
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import config as C, io  # noqa: E402
from cxreval.infer import degrade, predict_radiograph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--set", required=True, choices=["cv", "heldout", "external"])
    ap.add_argument("--images-dir", default=None, help="radiograph folder (default work/images)")
    ap.add_argument("--degrade", default="none", choices=["none", "down512", "down512_jpeg75"])
    ap.add_argument("--weights-tmpl", default="{work}/runs/fold{f}/weights/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import RTDETR
    W = Path(a.work)
    img_dir = Path(a.images_dir) if a.images_dir else W / "images"
    if a.set == "external":
        files = {p.stem: p for p in sorted(img_dir.glob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        ids = sorted(files)
        todo = {f: ids for f in range(C.N_MODELS)}
    else:
        sp = pd.read_csv(W / "splits.csv")
        files = {i: img_dir / f"{i}.png" for i in sp.image_id}
        if a.set == "heldout":
            ids = sorted(sp.image_id[sp.split == "test"])
            todo = {f: ids for f in range(C.N_MODELS)}
        else:
            tv = sp[sp.split == "trainval"]
            ids = sorted(tv.image_id)
            todo = {f: sorted(tv.image_id[tv.fold.astype(int) == f]) for f in range(C.N_MODELS)}
    tmp = Path(tempfile.mkdtemp())
    out = {"set": a.set, "degrade": a.degrade, "ids": ids, "models": {}}
    for f, fids in todo.items():
        model = RTDETR(a.weights_tmpl.format(work=W, f=f))
        res = {}
        for n, i in enumerate(fids, 1):
            img = cv2.imread(str(files[i]))
            if img.shape[:2] != (C.IMG_SIZE, C.IMG_SIZE):
                sys.exit(f"{files[i]}: expected {C.IMG_SIZE} x {C.IMG_SIZE}, got {img.shape[:2]}")
            res[i] = {"tiles": predict_radiograph(model, degrade(img, a.degrade), a.device, tmp)}
            if n % 50 == 0:
                print(f"  model {f}: {n}/{len(fids)}", flush=True)
        out["models"][str(f)] = res
        print(f"model {f} done ({len(fids)} radiographs)", flush=True)
    io.write_json(a.out, out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
