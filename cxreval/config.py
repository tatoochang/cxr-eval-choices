"""All fixed settings used in the paper. Change nothing here if you want to reproduce the reported numbers."""
import numpy as np

# ---- data
TARGET_CLASS = 8            # VinBigData class id 8 = "Nodule/Mass"
IMG_SIZE = 1024             # radiographs are letterboxed to 1024 x 1024
TEST_PERCENT = 10           # held-out set: md5(image_id) % 100 >= 90
N_FOLDS = 5
FOLD_SEED = 42              # random.Random(42).shuffle of the sorted trainval ids, 150 per fold

# ---- reader-box fusion and curation
READER_WBF_IOU = 0.5        # reader boxes with IoU >= 0.5 are one lesion
CONSENSUS_MIN_READERS = 2   # consensus reference: >= 2 of 3 readers
CURATE_MAX_REL_EQ = 0.30    # equivalent side > 30 % of the image -> removed
CURATE_MAX_SIDE = 0.45      # one side > 45 % of the image width/height -> removed
CURATE_MAX_AR = 6.0         # aspect ratio > 6 or < 1/6 -> removed
CURATE_MAX_ENCLOSED = 2     # box enclosing >= 2 other boxes -> removed

# ---- tiling
TILE = 512
STRIDE = 256
GRID = 3                    # 3 x 3 = 9 tiles per radiograph
TILE_KEEP_AREA = 0.5        # a box is kept in a tile if >= 50 % of its area falls inside ...
TILE_MIN_SIDE = 5.0         # ... and its shorter clipped side is >= 5 px
TILE_JPEG_QUALITY = 95

# ---- training (Ultralytics 8.4.155, RT-DETR-L, COCO-pretrained rtdetr-l.pt)
TRAIN = dict(epochs=150, imgsz=1280, batch=8, optimizer="AdamW", lr0=1e-4, lrf=0.01, cos_lr=True,
             warmup_epochs=1.0, weight_decay=5e-4, close_mosaic=10, patience=20, seed=42,
             deterministic=True, amp=True)

# ---- inference
INFER = dict(imgsz=1024, conf=0.001, iou=0.7, max_det=100)
TILE_FUSION_IOU = 0.5       # NMS across tiles
N_MODELS = 5                # five fold models are combined for held-out / external data

# ---- evaluation
LEGACY_THR = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
THRESHOLDS = sorted(set(LEGACY_THR + [float(x) for x in np.logspace(np.log10(0.001), np.log10(0.99), 120)]))  # 128
FP_TARGETS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_BOOT = 1000
BOOT_SEED = 42

# ---- image-storage experiment
DEGRADE_SIZE = 512          # INTER_AREA down, bilinear back up to 1024
DEGRADE_JPEG_QUALITY = 75

# ---- lung mask (TorchXRayVision 1.5.5 PSPNet)
LUNG_DILATE_PX = 48
