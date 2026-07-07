from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

CATEGORIES = [f"WM_{i}" for i in range(1, 9)]

CATEGORY_RANGES = {
    "WM_1": (1, 25),
    "WM_2": (26, 50),
    "WM_3": (51, 75),
    "WM_4": (76, 100),
    "WM_5": (101, 125),
    "WM_6": (126, 150),
    "WM_7": (151, 175),
    "WM_8": (176, 200),
}

EPS = 1e-8


def numeric_suffix(p: Path) -> int:
    """Extract the trailing integer id from a filename like '42.png' -> 42."""
    m = re.search(r"(\d+)$", p.stem)
    if not m:
        raise ValueError(p)
    return int(m.group(1))


def sorted_pngs(d: Path):
    return sorted(d.glob("*.png"), key=numeric_suffix)


def load_rgb(p: Path):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"), np.float32) / 255.0


def save_rgb(x, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(x * 255, 0, 255).round().astype(np.uint8)).save(p)


def to_pil(x):
    return Image.fromarray(np.clip(x * 255, 0, 255).round().astype(np.uint8))


def from_pil(im):
    return np.asarray(im.convert("RGB"), np.float32) / 255.0


def resize_np(x, size):
    return from_pil(to_pil(x).resize(size, Image.Resampling.BILINEAR))


def grayscale(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def high_pass(x, sigma=1.5):
    return x - gaussian_filter(x, sigma=(sigma, sigma, 0), mode="reflect")


def rgb_to_ycbcr(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], -1)


def ycbcr_to_rgb(x):
    y = x[..., 0]
    cb = x[..., 1] - 0.5
    cr = x[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], -1), 0, 1)


def png_roundtrip(x):
    buf = io.BytesIO()
    to_pil(x).save(buf, format="PNG")
    buf.seek(0)
    return from_pil(Image.open(buf))


def jpeg_roundtrip(x, q=95):
    buf = io.BytesIO()
    to_pil(x).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return from_pil(Image.open(buf))


def load_dataset(root: Path):
    """Load the `clean_targets/` and `watermarked_sources/<category>/` folders."""
    root = root.resolve()
    source_root = root / "watermarked_sources"
    clean_root = root / "clean_targets"
    if not source_root.is_dir() or not clean_root.is_dir():
        raise FileNotFoundError("dataset needs clean_targets/ and watermarked_sources/")
    src = {c: [load_rgb(p) for p in sorted_pngs(source_root / c)] for c in CATEGORIES}
    clean = {numeric_suffix(p): load_rgb(p) for p in sorted_pngs(clean_root)}
    return src, clean


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def category_for_id(i: int) -> str:
    """Look up which watermark category a clean-target id (1..200) belongs to."""
    for c, (lo, hi) in CATEGORY_RANGES.items():
        if lo <= i <= hi:
            return c
    raise ValueError(i)


def _wavelet_denoise(channel, wavelet="db4", level=3):
    """Wavelet BayesShrink denoiser used by the 'denoiser' content estimate."""
    import pywt

    coeffs = pywt.wavedec2(channel, wavelet, level=level, mode="periodization")
    finest = coeffs[-1][-1]
    sigma = np.median(np.abs(finest)) / 0.6745  # robust noise std (MAD)
    out = [coeffs[0]]
    for details in coeffs[1:]:
        thresholded = []
        for d in details:
            var = np.var(d)
            sigma_x = np.sqrt(max(var - sigma ** 2, 1e-12))
            thresh = sigma ** 2 / sigma_x  # BayesShrink, per subband
            thresholded.append(pywt.threshold(d, thresh, mode="soft"))
        out.append(tuple(thresholded))
    denoised = pywt.waverec2(out, wavelet, mode="periodization")
    return denoised[: channel.shape[0], : channel.shape[1]]


def estimate_content(channel, method):
    """Estimate the clean content of a single channel, for residual extraction.
    'highpass' = gaussian blur; 'denoiser' = wavelet BayesShrink (median-filter
    fallback if PyWavelets is unavailable)."""
    if method == "highpass":
        return gaussian_filter(channel, 1.5, mode="reflect")
    if method == "denoiser":
        try:
            return _wavelet_denoise(channel)
        except ImportError:
            return median_filter(channel, size=3, mode="reflect")
    raise ValueError(method)
