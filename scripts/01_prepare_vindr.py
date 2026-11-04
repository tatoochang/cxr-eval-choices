#!/usr/bin/env python3
"""Step 1. VinDr-CXR (VinBigData release) -> 1024-px PNGs, reference standards and splits.

Inputs (download yourself, see README):
  --train-csv   train.csv of the VinBigData Chest X-ray Abnormalities Detection release
  --dicom-dir   folder with the training DICOMs (<image_id>.dicom); only the 826 radiographs
                with a Nodule/Mass box are read

Outputs under --work:
  annotations/  class8_raw.csv, image_dims.csv, lesions_any_reader.csv, lesions_consensus.csv,
                removed_boxes.csv, summary.json
  labels/any_reader/<id>.txt   any-reader reference (= training labels), YOLO format on the 1024 canvas
  labels/consensus/<id>.txt    consensus reference (>= 2 of 3 readers)
  images/<id>.png              letterboxed 8-bit radiographs (skip with --skip-images)
  splits.csv                   image_id, split (trainval | test), fold (0-4 for trainval)
The split is checked against splits/splits.csv shipped with this repository.

Usage: python scripts/01_prepare_vindr.py --train-csv train.csv --dicom-dir train/ --work work
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxreval import annotations as A, config as C, io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--skip-images", action="store_true")
    a = ap.parse_args()
    W, D = Path(a.work), Path(a.dicom_dir)
    (W / "annotations").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(a.train_csv)
    raw = raw[raw.class_id == C.TARGET_CLASS][["image_id", "class_name", "class_id", "rad_id",
                                              "x_min", "y_min", "x_max", "y_max"]].copy()
    raw.to_csv(W / "annotations/class8_raw.csv", index=False)
    print(f"reader boxes: {len(raw)} on {raw.image_id.nunique()} radiographs")

    import pydicom
    dims_path = W / "annotations/image_dims.csv"
    if dims_path.exists():
        dims = pd.read_csv(dims_path)
    else:
        rows = []
        for iid in sorted(raw.image_id.unique()):
            ds = pydicom.dcmread(str(D / f"{iid}.dicom"), stop_before_pixels=True)
            rows.append({"image_id": iid, "rows": int(ds.Rows), "cols": int(ds.Columns)})
        dims = pd.DataFrame(rows)
        dims.to_csv(dims_path, index=False)

    fused = A.fuse_dataframe(raw)
    anyr, removed = A.build_reference(raw, dims, min_readers=1)
    cons, _ = A.build_reference(raw, dims, min_readers=C.CONSENSUS_MIN_READERS)
    anyr.to_csv(W / "annotations/lesions_any_reader.csv", index=False)
    cons.to_csv(W / "annotations/lesions_consensus.csv", index=False)
    removed.to_csv(W / "annotations/removed_boxes.csv", index=False)

    ids = sorted(anyr.image_id.unique())      # radiographs left without any box are excluded
    split = {i: A.split_of(i) for i in ids}
    folds = A.make_folds([i for i in ids if split[i] == "trainval"])
    sp = pd.DataFrame([{"image_id": i, "split": split[i], "fold": folds.get(i, "")} for i in ids])
    sp.to_csv(W / "splits.csv", index=False)
    ref = pd.read_csv(REPO / "splits/splits.csv", keep_default_na=False)
    if not sp.astype(str).equals(ref.astype(str)):
        sys.exit("ERROR: the split differs from splits/splits.csv - check the input files")
    print("split matches splits/splits.csv")

    dmap = {r.image_id: (r.rows, r.cols) for r in dims.itertuples(index=False)}
    for name, df in (("any_reader", anyr), ("consensus", cons)):
        out = W / "labels" / name
        out.mkdir(parents=True, exist_ok=True)
        by = dict(list(df.groupby("image_id")))
        for i in ids:
            lines = A.yolo_lines(by[i], *dmap[i]) if i in by else []
            (out / f"{i}.txt").write_text("\n".join(lines) + "\n" if lines else "")

    S = {"reader_boxes": int(len(raw)), "radiographs_with_box": int(raw.image_id.nunique()),
         "fused_lesions": int(len(fused)), "fused_by_n_readers": {int(k): int(v) for k, v in fused.n_rads.value_counts().sort_index().items()},
         "removed_regional_boxes": {k: int(v) for k, v in removed.reason.value_counts().items()},
         "radiographs": len(ids), "any_reader_lesions": int(len(anyr)), "consensus_lesions": int(len(cons)),
         "trainval": int((sp.split == "trainval").sum()), "test": int((sp.split == "test").sum()),
         "consensus_lesions_trainval": int(cons.image_id.map(split).eq("trainval").sum()),
         "consensus_lesions_test": int(cons.image_id.map(split).eq("test").sum())}
    io.write_json(W / "annotations/summary.json", S, indent=1)
    print(S)

    if not a.skip_images:
        from PIL import Image
        from cxreval.dicom import letterbox, window_dicom
        (W / "images").mkdir(exist_ok=True)
        for n, i in enumerate(ids, 1):
            Image.fromarray(letterbox(window_dicom(pydicom.dcmread(str(D / f"{i}.dicom"))), C.IMG_SIZE)).save(W / "images" / f"{i}.png")
            if n % 100 == 0:
                print(f"  PNG {n}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
