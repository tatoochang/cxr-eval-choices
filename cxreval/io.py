"""Small file helpers: YOLO labels and (gzipped) JSON."""
import gzip
import json
from pathlib import Path


def load_yolo_xyxy(path, W=1024, H=1024):
    """YOLO label file (cls cx cy w h, normalised) -> [(x1, y1, x2, y2)] in pixels. Missing file -> []."""
    path = Path(path)
    out = []
    if not path.exists():
        return out
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def load_labels(label_dir, ids, W=1024, H=1024):
    return {i: load_yolo_xyxy(Path(label_dir) / f"{i}.txt", W, H) for i in ids}


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def write_json(path, obj, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        json.dump(obj, f, indent=indent)
