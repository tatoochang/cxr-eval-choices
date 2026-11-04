#!/usr/bin/env python3
"""Step 3. Train one fold model: RT-DETR-L from the COCO-pretrained rtdetr-l.pt released with Ultralytics.

Settings are those of the paper (cxreval/config.py TRAIN; full args in configs/ultralytics_args_fold0.yaml):
150-epoch budget, early stopping after 20 epochs without improvement of validation fitness, best checkpoint kept.
About 2 GPU hours per fold on one RTX PRO 6000 (96 GB). GPU non-determinism means retrained weights
will not be bit-identical; expect recall to vary by a few hundredths per fold.

Usage: python scripts/03_train.py --work work --fold 0 --weights rtdetr-l.pt
       (--patience 0 reproduces the full-budget sensitivity analysis)
Output: work/runs/fold<k>/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--weights", default="rtdetr-l.pt")
    ap.add_argument("--patience", type=int, default=C.TRAIN["patience"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    from ultralytics import RTDETR
    W = Path(a.work).resolve()
    args = dict(C.TRAIN, patience=a.patience)
    model = RTDETR(a.weights)
    model.train(data=str(W / "tiles" / f"fold{a.fold}" / "data.yaml"), project=str(W / "runs"),
                name=a.name or f"fold{a.fold}", exist_ok=True, device=a.device, workers=a.workers,
                box=7.5, cls=0.5, dfl=1.5, mosaic=1.0, plots=True, val=True, verbose=True, **args)


if __name__ == "__main__":
    main()
