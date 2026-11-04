#!/usr/bin/env python3
"""Step 7 (optional). Lung masks with the open-source TorchXRayVision 1.5.5 PSPNet (ChestX-Det).
Mask = left lung OR right lung, sigmoid > 0.5, predicted at 512 px and resized to 1024 px (nearest).
Used to count false positives outside the lungs and for the lung-zone analysis of step 8.

Usage: python scripts/07_lung_mask.py --src work/images --out work/lungmask/vindr [--device cuda]
Output: <out>/<image_id>.png (0/255, 1024 x 1024)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchxrayvision as xrv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = xrv.baseline_models.chestx_det.PSPNet().to(a.device).eval()
    li = [model.targets.index("Left Lung"), model.targets.index("Right Lung")]
    files = sorted(Path(a.src).glob(a.glob))
    for n, p in enumerate(files, 1):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = xrv.datasets.normalize(x, 255)[None, None]
        with torch.no_grad():
            y = torch.sigmoid(model(torch.from_numpy(x).to(a.device)))[0, li].amax(0).cpu().numpy()
        cv2.imwrite(str(out / f"{p.stem}.png"),
                    cv2.resize((y > 0.5).astype(np.uint8) * 255, (1024, 1024), interpolation=cv2.INTER_NEAREST))
        if n % 100 == 0:
            print(n, "/", len(files), flush=True)


if __name__ == "__main__":
    main()
