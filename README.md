# Evaluation choices and the measured performance of tile-based nodule detectors on chest radiographs

Code, data splits and detector outputs for the paper

> *Evaluation choices determine the measured performance of tile-based pulmonary nodule detectors on chest radiographs: reference standard, hit criterion and image storage.*

The paper keeps one set of detector outputs fixed and measures how three evaluation choices change the
reported sensitivity:

| Choice | Compared | Effect on recall at 1 FP per image |
|---|---|---|
| Reference standard | any-reader boxes vs boxes marked by ≥ 2 of 3 readers | 0.422 → 0.699 |
| Hit criterion | IoU ≥ 0.5 vs detection centre inside the lesion box | 0.382 → 0.734 (external) |
| Image storage | original 1024-px PNG vs 512-px JPEG copies | 0.734 → 0.617 (external) |

The detector is RT-DETR-L (Ultralytics 8.4.155) trained on 3 × 3 overlapping 512-px tiles in five-fold
cross-validation on VinDr-CXR, and evaluated on whole radiographs by lesion-level FROC with bootstrap
confidence intervals.

---

## What is in this repository

```
cxreval/            library: reader-box fusion and curation, tiling, fusion, matching, FROC, bootstrap
scripts/            numbered steps 01–08 and check_against_paper.py
splits/splits.csv   the 820 VinDr-CXR image ids used, held-out / cross-validation split and fold (0–4)
detections/         raw per-tile detections of the five fold models (gzipped JSON, see "Detection files")
configs/            the full Ultralytics training arguments of the paper's models
expected/           the numbers reported in the paper, used by check_against_paper.py
tests/              unit tests of the matching and FROC code
```

Not included:

* **Images and reader annotations.** VinDr-CXR and NIH ChestX-ray14 must be obtained from their providers under
  their own terms (links below). Step 1 rebuilds every label file from the original VinDr-CXR annotation table.
* **The external reference boxes** (1,280 lesions on 465 NIH ChestX-ray14 radiographs, drawn by radiologists of a
  tertiary medical centre). They are available from the corresponding author on reasonable request.
  The external detector outputs are included, so the external analysis can be rerun once the boxes are available.
* **Trained weights.** Available from the corresponding author on reasonable request; step 3 retrains them.

## Installation

Python ≥ 3.10. The paper used Python 3.14, PyTorch 2.14 (CUDA 13), Ultralytics 8.4.155, OpenCV 5.0 and NumPy 2.5
on one NVIDIA RTX PRO 6000 (96 GB). Steps 5, 6 and 8 need no GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # evaluation only (CPU)
pip install -r requirements-gpu.txt      # training and inference (installs ultralytics==8.4.155)
python -m pytest tests                   # optional
```

## Data

* **VinDr-CXR** — VinBigData Chest X-ray Abnormalities Detection release (Kaggle competition
  `vinbigdata-chest-xray-abnormalities-detection`): `train.csv` and the training DICOMs. Only the 826 radiographs
  with at least one *Nodule/Mass* box (class id 8) are used. Nguyen et al., *Sci Data* 2022, doi:10.1038/s41597-022-01498-w.
* **NIH ChestX-ray14** — original 1024 × 1024 PNGs from the NIH Clinical Center release. Wang et al., CVPR 2017.
  The 465 image ids used externally are listed in `detections/external_nih_original.json.gz` (`"ids"`; release asset, see "Detection files").

## Quick check without a GPU (about 15 minutes)

This reproduces the internal results of the paper from the released detections. It needs only `train.csv`;
the DICOMs are read for their image size (headers only).

```bash
python scripts/01_prepare_vindr.py --train-csv train.csv --dicom-dir train/ --work work --skip-images
python scripts/05_evaluate.py --work work --cv detections/cv.json.gz --heldout detections/heldout.json.gz \
    --degraded down512=detections/cv_down512.json.gz --degraded down512_jpeg75=detections/cv_down512_jpeg75.json.gz
python scripts/check_against_paper.py --work work
```

Step 1 prints the annotation counts of the paper (2,580 reader boxes → 1,865 fused lesions → 67 regional boxes
removed → 820 radiographs, 1,798 any-reader and 514 consensus lesions) and stops if the split differs from
`splits/splits.csv`. With the released detections, `check_against_paper.py` should report every internal value as
matching.

## Full reproduction

| Step | Command | Time | Output |
|---|---|---|---|
| 1 | `python scripts/01_prepare_vindr.py --train-csv train.csv --dicom-dir train/ --work work` | 10 min | PNGs, labels, `splits.csv` |
| 2 | `python scripts/02_build_tiles.py --work work` | 5 min | `work/tiles/fold<k>/` |
| 3 | `python scripts/03_train.py --work work --fold <k> --weights rtdetr-l.pt` (k = 0…4) | ≈ 2 h per fold | `work/runs/fold<k>/weights/best.pt` |
| 4 | `python scripts/04_infer.py --work work --set cv --out work/detections/cv.json.gz` | 15 min | per-tile detections |
|   | `... --set heldout --out work/detections/heldout.json.gz` | 5 min | |
|   | `... --set cv --degrade down512_jpeg75 --out work/detections/cv_down512_jpeg75.json.gz` (and `down512`) | 15 min | |
|   | `... --set external --images-dir <NIH PNGs> --out work/detections/external.json.gz` | 25 min | |
| 5 | `python scripts/05_evaluate.py --work work --cv work/detections/cv.json.gz --heldout ...` | 5 min | `work/results/internal.json` |
| 6 | `python scripts/06_evaluate_external.py --dets ... --labels <boxes> --nih-patients --out work/results/external_original.json` | 15 min | `external_*.json` |
| 7 | `python scripts/07_lung_mask.py --src work/images --out work/lungmask/vindr` | 5 min | lung masks |
| 8 | `python scripts/08_stratify.py --work work --cv ... --masks work/lungmask/vindr` | 2 min | `work/results/stratified.json` |

`rtdetr-l.pt` is the COCO-pretrained checkpoint distributed with Ultralytics. Retraining on a GPU is not
bit-reproducible, so a retrained model gives recalls that differ by a few hundredths per fold; with the released
detections, steps 5–8 reproduce the paper exactly. For the external JPEG condition, upsample the 512-px JPEG copies
to 1024 px with bilinear interpolation before step 4.

### Where each result comes from

| Paper | Script → key |
|---|---|
| Annotation counts, Table 1 | `01` → `work/annotations/summary.json` |
| Reference standard on fixed predictions (Fig. 3) | `05` → `reference_standard` |
| Cross-validation, per fold (Table 3) | `05` → `cv` |
| Fusion rules, single-tile protocol (Table 4) | `05` → `fusion_rules`; `06` → `replication` |
| Held-out and external evaluation (Table 6) | `05` → `heldout`, `degraded`, `hit_criteria`; `06` → `ensemble` |
| Operating points (Table 7) | `05` → `operating_points` |
| CI width against validation-set size | `05` → `ci_width_vs_n` |
| Box area ratio (Fig. 7c), size tertiles | `06` → `box_area_ratio`, `size_tertiles`, `lesions_per_image` |
| Lesion characteristics (Table 8), FPs outside the lungs | `08` |

## Method details that matter for reproduction

* **Reference standards.** Reader boxes are fused per radiograph when IoU ≥ 0.5 (larger boxes seed clusters, the
  fused box is the mean of its members). Regional boxes are removed when a box encloses ≥ 2 other boxes, has one
  side > 45 % of the image, or an equivalent side > 30 % of the image; radiographs left without boxes are excluded.
  The consensus reference repeats fusion and curation on lesions marked by ≥ 2 readers.
* **Coordinates.** DICOMs are windowed with the header window, letterboxed to 1024 × 1024 (same scale for x and y,
  centred) with Lanczos resampling; labels are on that canvas.
* **Splits.** Held-out: `md5(image_id) % 100 ≥ 90` (70 radiographs). Folds: the 750 remaining ids, sorted, shuffled
  with `random.Random(42)`, blocks of 150.
* **Tiles.** 3 × 3 tiles of 512 px, stride 256, saved as JPEG quality 95 for training *and* inference. A box belongs to
  a tile if ≥ 50 % of its area lies inside and its shorter clipped side is ≥ 5 px.
* **Inference.** Input size 1024, confidence ≥ 0.001, ≤ 100 boxes per tile; tile boxes are shifted to radiograph
  coordinates and fused by NMS at IoU 0.5. Held-out and external: the five fold models are combined by weighted boxes
  fusion (IoU 0.5), confidence = Σ (each model's highest confidence in the cluster) / 5.
* **Matching and FROC.** Greedy in decreasing confidence, each reference box matched once; 128 thresholds
  (9 fixed + 120 log-spaced between 0.001 and 0.99); recall at a target FP rate by linear interpolation. Every
  radiograph contributes false positives. Bootstrap: 1,000 resamples, seed 42, of radiographs (internal, held-out)
  or patients (external); the paired bootstrap uses the same resamples for both reference standards. The bootstrap
  resamples positions, so the image order matters for the exact interval: cross-validation radiographs are ordered
  fold by fold (sorted within a fold), as in `05_evaluate.py`.
* **Localisation criterion** (`centre_dup`). A detection is a hit when its centre lies inside an unmatched reference
  box; further detections centred on a lesion already found are ignored. It was added after visual review of
  external cases and is applied identically to all data; the pre-specified IoU ≥ 0.5 results are always reported too.

## Detection files

`detections/*.json.gz`:

```json
{"set": "cv", "degrade": "none", "ids": ["<image_id>", ...],
 "models": {"0": {"<image_id>": {"tiles": {"0": [[x1, y1, x2, y2, conf], ...], ..., "8": [...]}}}}}
```

Tile boxes are tile-local pixels (tile *k* is offset by `((k % 3) * 256, (k // 3) * 256)`), coordinates rounded to
0.001 px and confidences to 1e-6. For `cv`, model *k* contains only the validation radiographs of fold *k*.
`cv_down512*.json.gz` store the NMS-fused radiograph detections (`"nms"`) instead of tiles.

| File | Content |
|---|---|
| `cv.json.gz` | 750 cross-validation radiographs, each by its own fold model |
| `heldout.json.gz` | 70 held-out radiographs, all five models |
| `cv_down512.json.gz`, `cv_down512_jpeg75.json.gz` | internal radiographs after 512-px downsampling (± JPEG q75) |
| `external_nih_original.json.gz` | 465 NIH radiographs, original PNGs, all five models |
| `external_nih_jpeg512.json.gz` | the same radiographs from 512-px JPEG copies |

The two external files (≈ 37 MB each) are attached to the
[v1.0.0 release](https://github.com/tatoochang/cxr-eval-choices/releases/tag/v1.0.0) rather than stored in the
repository. Download them into `detections/`:

```bash
gh release download v1.0.0 -R tatoochang/cxr-eval-choices -p 'external_nih_*.json.gz' -D detections
```

## Citation

See `CITATION.cff`. Please also cite VinDr-CXR and NIH ChestX-ray14.

## License

Code: MIT (see `LICENSE`). The detection files are released under CC BY 4.0. The datasets keep their own licenses.
