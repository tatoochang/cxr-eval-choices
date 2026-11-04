"""DICOM -> 8-bit letterboxed PNG, exactly as used for the paper."""
import numpy as np
from PIL import Image

from .annotations import letterbox_params


def window_dicom(dcm):
    """Uses the header WindowCenter/WindowWidth when plausible, otherwise min-max scaling."""
    raw = dcm.pixel_array.astype(np.float32)
    vmin, vmax = float(raw.min()), float(raw.max())
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)
    if wc is not None and ww is not None:
        first = lambda v: float(v[0]) if hasattr(v, "__getitem__") and not isinstance(v, (str, bytes)) else float(v)
        wc, ww = first(wc), first(ww)
        lo, hi = wc - ww / 2, wc + ww / 2
        if ww > 100 and hi > vmin and lo < vmax:
            img = np.clip(raw, lo, hi)
            return ((img - lo) / (hi - lo) * 255).astype(np.uint8)
    if vmax > vmin:
        return ((raw - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return np.zeros_like(raw, dtype=np.uint8)


def letterbox(img, target=1024):
    """Aspect-preserving Lanczos resize onto a black target x target canvas."""
    h, w = img.shape[:2]
    scale, pad_x, pad_y = letterbox_params(h, w, target)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))
    canvas = np.zeros((target, target), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas
