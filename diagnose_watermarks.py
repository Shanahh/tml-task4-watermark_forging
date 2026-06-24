from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pywt
from PIL import Image, ImageEnhance, ImageFilter
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CATEGORIES = [f"WM_{i}" for i in range(1, 9)]
EPS = 1e-8


@dataclass
class CategorySummary:
    category: str
    resolution: str
    source_count: int
    negative_count: int

    spatial_consistency: float
    spatial_consistency_ci_low: float
    spatial_consistency_ci_high: float

    phase_coherence_mean: float
    phase_coherence_top10pct: float

    spectral_magnitude_consistency: float
    radial_profile_consistency: float
    radial_profile_effect: float
    angular_profile_consistency: float
    angular_profile_effect: float

    translation_xcorr_mean: float
    periodicity_peak_ratio: float

    residual_auc: float
    residual_auc_std: float
    residual_auc_perm_p: float

    patch_auc: float
    patch_auc_std: float

    wavelet_best_auc: float
    wavelet_best_band: str

    y_auc: float
    cb_auc: float
    cr_auc: float

    lsb_auc: float
    jpeg_dct_auc: float

    energy_distance_residual: float
    wasserstein_residual: float
    mmd_residual: float

    rgb_auc: float
    content_leakage_gap: float

    resize_survival: float
    jpeg90_survival: float
    jpeg75_survival: float
    blur03_survival: float
    blur05_survival: float
    brightness_survival: float
    flip_survival: float

    recommendation: str
    confidence: str
    notes: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("diagnostics"))
    p.add_argument("--blur-sigma", type=float, default=1.5)
    p.add_argument("--signature-size", type=int, default=64)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--bootstraps", type=int, default=200)
    p.add_argument(
        "--max-negatives",
        type=int,
        default=100,
        help="Maximum same-resolution clean negatives per category.",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=100,
        help="Maximum image pairs for pairwise metrics.",
    )
    return p.parse_args()


def numeric_suffix(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    if not m:
        raise ValueError(f"Could not parse numeric suffix from {path.name}")
    return int(m.group(1))


def sorted_pngs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=numeric_suffix)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))


def from_pil(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def resize_np(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return from_pil(to_pil(image).resize(size, Image.Resampling.BILINEAR))


def high_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    return image - gaussian_filter(
        image, sigma=(sigma, sigma, 0), mode="reflect"
    )


def grayscale(image: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return np.zeros_like(vector) if norm < EPS else vector / norm


def residual_signature(
    image: np.ndarray, sigma: float, signature_size: int
) -> np.ndarray:
    residual = high_pass(image, sigma)
    encoded = np.clip(residual * 4.0 + 0.5, 0, 1)
    small = resize_np(encoded, (signature_size, signature_size))
    return normalize_vector((small - 0.5) / 4.0)


def sampled_pairs(n: int, maximum: int, rng: np.random.Generator):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > maximum:
        indices = rng.choice(len(pairs), size=maximum, replace=False)
        pairs = [pairs[int(i)] for i in indices]
    return pairs


def mean_pairwise_cosine(
    vectors: list[np.ndarray],
    max_pairs: int,
    rng: np.random.Generator,
) -> float:
    pairs = sampled_pairs(len(vectors), max_pairs, rng)
    if not pairs:
        return float("nan")
    return float(np.mean([vectors[i] @ vectors[j] for i, j in pairs]))


def bootstrap_ci(
    values: list[np.ndarray],
    metric: Callable[[list[np.ndarray]], float],
    bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if bootstraps <= 0:
        return float("nan"), float("nan")
    estimates = []
    n = len(values)
    for _ in range(bootstraps):
        idx = rng.integers(0, n, size=n)
        sample = [values[int(i)] for i in idx]
        try:
            estimates.append(metric(sample))
        except Exception:
            continue
    if not estimates:
        return float("nan"), float("nan")
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def phase_coherence(images: list[np.ndarray], sigma: float) -> tuple[float, float]:
    residuals = np.stack([grayscale(high_pass(x, sigma)) for x in images])
    spectra = np.fft.fft2(residuals, axes=(1, 2))
    unit = spectra / (np.abs(spectra) + EPS)
    coherence = np.abs(np.mean(unit, axis=0))

    h, w = coherence.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    values = coherence[(radius >= 0.02) & (radius <= 0.95)]
    k = max(1, int(math.ceil(0.10 * values.size)))
    return float(values.mean()), float(np.partition(values, -k)[-k:].mean())


def log_spectral_magnitude(image: np.ndarray, sigma: float, size: int = 64):
    residual = grayscale(high_pass(image, sigma))
    spectrum = np.fft.fftshift(np.fft.fft2(residual))
    magnitude = np.log1p(np.abs(spectrum))
    normalized = (magnitude - magnitude.mean()) / (magnitude.std() + EPS)
    pseudo_rgb = np.repeat(normalized[..., None], 3, axis=2)
    pseudo_rgb = np.clip(pseudo_rgb / 8.0 + 0.5, 0, 1)
    small = resize_np(pseudo_rgb, (size, size))[..., 0]
    return normalize_vector((small - 0.5) * 8.0)


def radial_profile(image: np.ndarray, sigma: float, bins: int = 32):
    residual = grayscale(high_pass(image, sigma))
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius /= radius.max() + EPS
    edges = np.linspace(0, 1, bins + 1)
    out = np.zeros(bins, np.float64)
    for i in range(bins):
        mask = (radius >= edges[i]) & (radius < edges[i + 1])
        out[i] = np.log1p(power[mask].mean()) if np.any(mask) else 0.0
    return normalize_vector(out)


def angular_profile(image: np.ndarray, sigma: float, bins: int = 24):
    residual = grayscale(high_pass(image, sigma))
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual))) ** 2
    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    angle = np.arctan2(yy - cy, xx - cx)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask_radius = (radius >= 0.05 * radius.max()) & (radius <= 0.95 * radius.max())
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    out = np.zeros(bins, np.float64)
    for i in range(bins):
        mask = mask_radius & (angle >= edges[i]) & (angle < edges[i + 1])
        out[i] = np.log1p(power[mask].mean()) if np.any(mask) else 0.0
    return normalize_vector(out)


def standardized_centroid_distance(pos: np.ndarray, neg: np.ndarray) -> float:
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)
    pooled = np.sqrt(0.5 * (pos.var(axis=0) + neg.var(axis=0)) + EPS)
    return float(np.linalg.norm(mean_diff / pooled) / math.sqrt(pos.shape[1]))


def translation_invariant_xcorr(
    images: list[np.ndarray],
    sigma: float,
    max_shift: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> float:
    residuals = []
    for image in images:
        r = grayscale(high_pass(image, sigma))
        r = resize_np(np.repeat((np.clip(r * 4 + .5, 0, 1))[..., None], 3, 2), (64, 64))[..., 0]
        r = (r - 0.5) / 4.0
        r -= r.mean()
        r /= np.linalg.norm(r) + EPS
        residuals.append(r)

    pairs = sampled_pairs(len(residuals), max_pairs, rng)
    scores = []
    shift = max(1, int(round(max_shift * 64 / images[0].shape[1])))
    center = 63
    for i, j in pairs:
        corr = fftconvolve(residuals[i], residuals[j][::-1, ::-1], mode="full")
        window = corr[
            center - shift : center + shift + 1,
            center - shift : center + shift + 1,
        ]
        scores.append(float(np.max(window)))
    return float(np.mean(scores)) if scores else float("nan")


def periodicity_peak_ratio(image: np.ndarray, sigma: float) -> float:
    r = grayscale(high_pass(image, sigma))
    r -= r.mean()
    corr = fftconvolve(r, r[::-1, ::-1], mode="full")
    corr /= np.sum(r * r) + EPS
    cy, cx = np.array(corr.shape) // 2
    exclusion = max(3, min(image.shape[:2]) // 64)
    corr[
        cy - exclusion : cy + exclusion + 1,
        cx - exclusion : cx + exclusion + 1,
    ] = 0.0
    absolute = np.abs(corr)
    peak = float(np.max(absolute))
    background = float(np.median(absolute) + EPS)
    return peak / background


def block_reduce_mean(array: np.ndarray, grid: int = 8) -> np.ndarray:
    if array.ndim == 2:
        array = array[..., None]
    h, w, _ = array.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    features = []
    for yi in range(grid):
        for xi in range(grid):
            block = array[ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1]]
            features.extend(block.mean(axis=(0, 1)).tolist())
    return np.asarray(features, dtype=np.float32)


def residual_features(image: np.ndarray, sigma: float) -> np.ndarray:
    residual = high_pass(image, sigma)
    gray = grayscale(residual)
    spatial = block_reduce_mean(residual, 8)
    spatial_abs = block_reduce_mean(np.abs(residual), 8)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    fourier = block_reduce_mean(spectrum, 8)

    encoded = np.repeat(gray[..., None], 3, axis=2) * 0.25 + 0.5
    resized = resize_np(np.clip(encoded, 0, 1), (64, 64))[..., 0]
    resized = (resized - 0.5) / 0.25
    dct = dctn(resized, type=2, norm="ortho")[:16, :16]
    dct[0, 0] = 0

    flat = residual.reshape(-1, 3)
    stats = np.concatenate(
        [
            flat.mean(axis=0),
            flat.std(axis=0),
            np.percentile(flat, [5, 25, 50, 75, 95], axis=0).reshape(-1),
            np.asarray(
                [
                    np.mean(np.abs(gray)),
                    np.std(gray),
                    np.percentile(np.abs(gray), 90),
                    np.percentile(np.abs(gray), 99),
                ]
            ),
        ]
    )
    return np.concatenate(
        [spatial, spatial_abs, fourier, dct.reshape(-1), stats]
    ).astype(np.float32)


def rgb_features(image: np.ndarray) -> np.ndarray:
    pooled = block_reduce_mean(resize_np(image, (32, 32)), 4)
    flat = image.reshape(-1, 3)
    stats = np.concatenate(
        [
            flat.mean(axis=0),
            flat.std(axis=0),
            np.percentile(flat, [5, 25, 50, 75, 95], axis=0).reshape(-1),
        ]
    )
    return np.concatenate([pooled, stats]).astype(np.float32)


def ycbcr_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def channel_residual_features(image: np.ndarray, sigma: float, channel: int):
    arr = ycbcr_channels(image)[channel]
    residual = arr - gaussian_filter(arr, sigma=sigma, mode="reflect")
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(residual))))
    return np.concatenate(
        [
            block_reduce_mean(residual, 8),
            block_reduce_mean(np.abs(residual), 8),
            block_reduce_mean(spectrum, 8),
            np.asarray(
                [
                    residual.mean(),
                    residual.std(),
                    np.mean(np.abs(residual)),
                    np.percentile(np.abs(residual), 90),
                    np.percentile(np.abs(residual), 99),
                ]
            ),
        ]
    ).astype(np.float32)


def wavelet_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    gray = grayscale(image)
    coeffs = pywt.wavedec2(gray, wavelet="db2", level=3)
    features = []
    names = []
    for level, detail in enumerate(coeffs[1:], start=1):
        for orientation, band in zip(("H", "V", "D"), detail):
            features.extend(
                [
                    float(np.mean(np.abs(band))),
                    float(np.std(band)),
                    float(np.mean(band * band)),
                    float(np.percentile(np.abs(band), 90)),
                ]
            )
            names.extend(
                [
                    f"L{level}{orientation}_meanabs",
                    f"L{level}{orientation}_std",
                    f"L{level}{orientation}_energy",
                    f"L{level}{orientation}_p90",
                ]
            )
    return np.asarray(features, np.float32), names


def lsb_features(image: np.ndarray) -> np.ndarray:
    uint = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    feats = []
    for channel in range(3):
        for bit in range(2):
            plane = ((uint[..., channel] >> bit) & 1).astype(np.float32)
            p = plane.mean()
            entropy = -(p * np.log2(p + EPS) + (1 - p) * np.log2(1 - p + EPS))
            horizontal = np.mean(plane[:, 1:] == plane[:, :-1])
            vertical = np.mean(plane[1:, :] == plane[:-1, :])
            transitions_h = np.mean(plane[:, 1:] != plane[:, :-1])
            transitions_v = np.mean(plane[1:, :] != plane[:-1, :])
            feats.extend([p, entropy, horizontal, vertical, transitions_h, transitions_v])
    return np.asarray(feats, np.float32)


def jpeg_dct_features(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    h, w = gray.shape
    h8, w8 = h - h % 8, w - w % 8
    gray = gray[:h8, :w8] - 0.5
    coeffs = []
    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            coeffs.append(dctn(gray[y : y + 8, x : x + 8], type=2, norm="ortho"))
    c = np.stack(coeffs)
    selected = [
        (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1),
        (2, 2), (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2),
    ]
    feats = []
    for y, x in selected:
        v = c[:, y, x]
        feats.extend(
            [
                v.mean(),
                v.std(),
                np.mean(np.abs(v)),
                np.mean(v > 0),
                np.mean(np.abs(v) < 1e-4),
                np.mean((np.round(np.abs(v) * 255).astype(np.int64) % 2) == 0),
            ]
        )
    return np.asarray(feats, np.float32)


def patch_features(image: np.ndarray, sigma: float, grid: int = 4) -> np.ndarray:
    h, w = image.shape[:2]
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    feats = []
    for yi in range(grid):
        for xi in range(grid):
            patch = image[ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1]]
            residual = high_pass(patch, sigma)
            gray = grayscale(residual)
            spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
            feats.extend(
                [
                    np.mean(np.abs(gray)),
                    np.std(gray),
                    np.percentile(np.abs(gray), 90),
                    np.mean(spectrum),
                    np.std(spectrum),
                ]
            )
    return np.asarray(feats, np.float32)


def make_model(seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    C=0.25,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def cross_validated_detector(
    pos: np.ndarray,
    neg: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, float, float]:
    x = np.concatenate([pos, neg])
    y = np.concatenate(
        [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
    )
    n_splits = min(folds, int(min(np.sum(y == 0), np.sum(y == 1))))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accuracies = [], []
    for train, test in splitter.split(x, y):
        model = make_model(seed)
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        aucs.append(roc_auc_score(y[test], prob))
        accuracies.append(accuracy_score(y[test], prob >= 0.5))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accuracies))


def permutation_p_value(
    pos: np.ndarray,
    neg: np.ndarray,
    observed_auc: float,
    folds: int,
    seed: int,
    permutations: int,
) -> float:
    if permutations <= 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    labels = np.concatenate(
        [np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)]
    )
    rng = np.random.default_rng(seed)
    null = []
    for i in range(permutations):
        shuffled = rng.permutation(labels)
        n_splits = min(folds, int(min(np.sum(shuffled == 0), np.sum(shuffled == 1))))
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + i + 1
        )
        fold_aucs = []
        for train, test in splitter.split(x, shuffled):
            model = make_model(seed + i + 1)
            model.fit(x[train], shuffled[train])
            prob = model.predict_proba(x[test])[:, 1]
            fold_aucs.append(roc_auc_score(shuffled[test], prob))
        null.append(float(np.mean(fold_aucs)))
    return float((1 + np.sum(np.asarray(null) >= observed_auc)) / (1 + permutations))


def best_single_feature_auc(
    pos: np.ndarray, neg: np.ndarray, names: list[str]
) -> tuple[float, str]:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    x = np.concatenate([pos, neg])
    best_auc, best_name = 0.5, "none"
    for i, name in enumerate(names):
        auc = roc_auc_score(y, x[:, i])
        auc = max(auc, 1.0 - auc)
        if auc > best_auc:
            best_auc, best_name = float(auc), name
    return best_auc, best_name


def rbf_mmd(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.vstack([pos, neg]).astype(np.float64)
    sample = x[: min(100, len(x))]
    dists = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=2)
    positive = dists[dists > 0]
    bandwidth = np.median(positive) if positive.size else 1.0
    gamma = 1.0 / (2.0 * bandwidth + EPS)

    def kernel(a, b):
        return np.exp(-gamma * np.sum((a[:, None] - b[None, :]) ** 2, axis=2))

    kxx = kernel(pos, pos)
    kyy = kernel(neg, neg)
    kxy = kernel(pos, neg)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def distribution_distances(
    pos: np.ndarray, neg: np.ndarray
) -> tuple[float, float, float]:
    # Compress to scalar standardized detector-like projections for robust 1D distances.
    scaler = StandardScaler().fit(np.vstack([pos, neg]))
    p = scaler.transform(pos)
    n = scaler.transform(neg)
    centroid = p.mean(axis=0) - n.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + EPS
    pp = p @ centroid
    nn = n @ centroid
    return (
        float(energy_distance(pp, nn)),
        float(wasserstein_distance(pp, nn)),
        rbf_mmd(p, n),
    )


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return from_pil(Image.open(buffer))


def transform_image(image: np.ndarray, name: str) -> np.ndarray:
    pil = to_pil(image)
    h, w = image.shape[:2]
    if name == "jpeg90":
        return jpeg_roundtrip(image, 90)
    if name == "jpeg75":
        return jpeg_roundtrip(image, 75)
    if name == "blur03":
        return from_pil(pil.filter(ImageFilter.GaussianBlur(radius=0.3)))
    if name == "blur05":
        return from_pil(pil.filter(ImageFilter.GaussianBlur(radius=0.5)))
    if name == "resize":
        d = max(32, min(h, w) // 2)
        return from_pil(
            pil.resize((d, d), Image.Resampling.BILINEAR).resize(
                (w, h), Image.Resampling.BILINEAR
            )
        )
    if name == "brightness":
        return from_pil(ImageEnhance.Brightness(pil).enhance(1.05))
    if name == "flip":
        return from_pil(pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    raise ValueError(name)


def detector_survival(
    source_images: list[np.ndarray],
    negative_images: list[np.ndarray],
    sigma: float,
    seed: int,
) -> dict[str, float]:
    pos = np.stack([residual_features(x, sigma) for x in source_images])
    neg = np.stack([residual_features(x, sigma) for x in negative_images])
    model = make_model(seed)
    model.fit(np.vstack([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])

    base_scores = model.predict_proba(pos)[:, 1]
    base_margin = np.mean(base_scores - 0.5)
    if abs(base_margin) < EPS:
        base_margin = EPS

    output = {}
    for name in ("resize", "jpeg90", "jpeg75", "blur03", "blur05", "brightness", "flip"):
        transformed = np.stack(
            [residual_features(transform_image(x, name), sigma) for x in source_images]
        )
        scores = model.predict_proba(transformed)[:, 1]
        margin = np.mean(scores - 0.5)
        output[name] = float(np.clip(margin / base_margin, -2.0, 2.0))
    return output


def category_specificity_matrix(
    category_images: dict[str, list[np.ndarray]],
    clean_by_resolution: dict[tuple[int, int], list[np.ndarray]],
    sigma: float,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    labels = CATEGORIES + ["clean"]
    matrix = np.zeros((len(CATEGORIES), len(labels)), dtype=np.float64)
    for row, category in enumerate(CATEGORIES):
        positives = category_images[category]
        resolution = (positives[0].shape[1], positives[0].shape[0])
        negatives = clean_by_resolution[resolution]
        pos_f = np.stack([residual_features(x, sigma) for x in positives])
        neg_f = np.stack([residual_features(x, sigma) for x in negatives])
        model = make_model(seed + row)
        model.fit(
            np.vstack([pos_f, neg_f]),
            np.r_[np.ones(len(pos_f)), np.zeros(len(neg_f))],
        )
        for col, target in enumerate(CATEGORIES):
            target_f = np.stack(
                [residual_features(x, sigma) for x in category_images[target]]
            )
            matrix[row, col] = model.predict_proba(target_f)[:, 1].mean()
        matrix[row, -1] = model.predict_proba(neg_f)[:, 1].mean()
    return labels, matrix


def choose_recommendation(metrics: dict) -> tuple[str, str, str]:
    candidates = []

    if metrics["translation_xcorr"] >= 0.10 or metrics["spatial"] >= 0.05:
        candidates.append(("robust_spatial_template", 3))
    if (
        metrics["phase_top"] >= 0.48
        or metrics["spectral_mag"] >= 0.12
        or metrics["radial_effect"] >= 0.80
        or metrics["angular_effect"] >= 0.80
    ):
        candidates.append(("spectral_template_or_band_attack", 3))
    if metrics["periodicity"] >= 25:
        candidates.append(("periodic_or_tiled_template", 3))
    if metrics["wavelet_auc"] >= 0.80:
        candidates.append(("wavelet_band_attack", 3))
    if max(metrics["y_auc"], metrics["cb_auc"], metrics["cr_auc"]) >= 0.82:
        candidates.append(("channel_specific_residual_attack", 3))
    if metrics["jpeg_dct_auc"] >= 0.82:
        candidates.append(("jpeg_dct_coefficient_attack", 3))
    if metrics["lsb_auc"] >= 0.82:
        candidates.append(("bit_plane_attack", 3))
    if (
        metrics["residual_auc"] >= 0.80
        and metrics["rgb_auc"] - metrics["residual_auc"] <= 0.12
        and metrics["perm_p"] <= 0.05
    ):
        candidates.append(("residual_surrogate_detector_pgd", 4))
    if metrics["patch_auc"] >= 0.80:
        candidates.append(("local_patch_surrogate_attack", 3))

    if not candidates:
        recommendation = "content_adaptive_or_latent_watermark"
        confidence = "low"
    else:
        recommendation, score = max(candidates, key=lambda item: item[1])
        evidence_count = len(candidates)
        confidence = "high" if evidence_count >= 2 or score >= 4 else "medium"

    notes = []
    if metrics["rgb_auc"] - metrics["residual_auc"] > 0.15:
        notes.append("Raw RGB substantially outperforms residual features; possible content/provenance leakage.")
    if metrics["perm_p"] > 0.05:
        notes.append("Residual-detector AUC is not significant under the configured permutation test.")
    if metrics["jpeg90_survival"] > 0.7:
        notes.append("Residual-detector evidence survives mild JPEG.")
    if metrics["resize_survival"] < 0.25:
        notes.append("Evidence is resize-sensitive.")
    if metrics["flip_survival"] < 0.25:
        notes.append("Evidence is orientation-sensitive.")
    if not notes:
        notes.append("No major validation warning was triggered.")
    return recommendation, confidence, " ".join(notes)


def analyze_category(
    category: str,
    source_images: list[np.ndarray],
    negative_images: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[CategorySummary, dict]:
    rng = np.random.default_rng(args.seed + int(category.split("_")[1]))

    signatures = [
        residual_signature(x, args.blur_sigma, args.signature_size)
        for x in source_images
    ]
    spatial = mean_pairwise_cosine(signatures, args.max_pairs, rng)
    ci_low, ci_high = bootstrap_ci(
        signatures,
        lambda sample: mean_pairwise_cosine(sample, args.max_pairs, rng),
        args.bootstraps,
        rng,
    )

    phase_mean, phase_top = phase_coherence(source_images, args.blur_sigma)

    spectral_vectors = [
        log_spectral_magnitude(x, args.blur_sigma) for x in source_images
    ]
    spectral_mag = mean_pairwise_cosine(spectral_vectors, args.max_pairs, rng)

    pos_radial = np.stack([radial_profile(x, args.blur_sigma) for x in source_images])
    neg_radial = np.stack([radial_profile(x, args.blur_sigma) for x in negative_images])
    radial_consistency = mean_pairwise_cosine(
        [v for v in pos_radial], args.max_pairs, rng
    )
    radial_effect = standardized_centroid_distance(pos_radial, neg_radial)

    pos_angular = np.stack([angular_profile(x, args.blur_sigma) for x in source_images])
    neg_angular = np.stack([angular_profile(x, args.blur_sigma) for x in negative_images])
    angular_consistency = mean_pairwise_cosine(
        [v for v in pos_angular], args.max_pairs, rng
    )
    angular_effect = standardized_centroid_distance(pos_angular, neg_angular)

    xcorr = translation_invariant_xcorr(
        source_images, args.blur_sigma, 16, args.max_pairs, rng
    )
    periodicity = float(
        np.mean([periodicity_peak_ratio(x, args.blur_sigma) for x in source_images])
    )

    pos_res = np.stack([residual_features(x, args.blur_sigma) for x in source_images])
    neg_res = np.stack([residual_features(x, args.blur_sigma) for x in negative_images])
    residual_auc, residual_std, _ = cross_validated_detector(
        pos_res, neg_res, args.cv_folds, args.seed
    )
    perm_p = permutation_p_value(
        pos_res,
        neg_res,
        residual_auc,
        args.cv_folds,
        args.seed,
        args.permutations,
    )

    pos_patch = np.stack([patch_features(x, args.blur_sigma) for x in source_images])
    neg_patch = np.stack([patch_features(x, args.blur_sigma) for x in negative_images])
    patch_auc, patch_std, _ = cross_validated_detector(
        pos_patch, neg_patch, args.cv_folds, args.seed
    )

    pos_wave = []
    neg_wave = []
    wave_names = None
    for x in source_images:
        f, names = wavelet_features(x)
        pos_wave.append(f)
        wave_names = names
    for x in negative_images:
        f, _ = wavelet_features(x)
        neg_wave.append(f)
    pos_wave = np.stack(pos_wave)
    neg_wave = np.stack(neg_wave)
    wavelet_auc, wavelet_band = best_single_feature_auc(
        pos_wave, neg_wave, wave_names or []
    )

    channel_aucs = []
    for channel in range(3):
        p = np.stack(
            [channel_residual_features(x, args.blur_sigma, channel) for x in source_images]
        )
        n = np.stack(
            [channel_residual_features(x, args.blur_sigma, channel) for x in negative_images]
        )
        channel_aucs.append(
            cross_validated_detector(p, n, args.cv_folds, args.seed)[0]
        )

    pos_lsb = np.stack([lsb_features(x) for x in source_images])
    neg_lsb = np.stack([lsb_features(x) for x in negative_images])
    lsb_auc = cross_validated_detector(
        pos_lsb, neg_lsb, args.cv_folds, args.seed
    )[0]

    pos_dct = np.stack([jpeg_dct_features(x) for x in source_images])
    neg_dct = np.stack([jpeg_dct_features(x) for x in negative_images])
    jpeg_dct_auc = cross_validated_detector(
        pos_dct, neg_dct, args.cv_folds, args.seed
    )[0]

    pos_rgb = np.stack([rgb_features(x) for x in source_images])
    neg_rgb = np.stack([rgb_features(x) for x in negative_images])
    rgb_auc = cross_validated_detector(
        pos_rgb, neg_rgb, args.cv_folds, args.seed
    )[0]

    e_dist, w_dist, mmd = distribution_distances(pos_res, neg_res)
    survival = detector_survival(
        source_images, negative_images, args.blur_sigma, args.seed
    )

    metrics = {
        "spatial": spatial,
        "phase_top": phase_top,
        "spectral_mag": spectral_mag,
        "radial_effect": radial_effect,
        "angular_effect": angular_effect,
        "translation_xcorr": xcorr,
        "periodicity": periodicity,
        "residual_auc": residual_auc,
        "patch_auc": patch_auc,
        "wavelet_auc": wavelet_auc,
        "y_auc": channel_aucs[0],
        "cb_auc": channel_aucs[1],
        "cr_auc": channel_aucs[2],
        "lsb_auc": lsb_auc,
        "jpeg_dct_auc": jpeg_dct_auc,
        "rgb_auc": rgb_auc,
        "perm_p": perm_p,
        "resize_survival": survival["resize"],
        "jpeg90_survival": survival["jpeg90"],
        "flip_survival": survival["flip"],
    }
    recommendation, confidence, notes = choose_recommendation(metrics)

    resolution = f"{source_images[0].shape[1]}x{source_images[0].shape[0]}"
    summary = CategorySummary(
        category=category,
        resolution=resolution,
        source_count=len(source_images),
        negative_count=len(negative_images),
        spatial_consistency=spatial,
        spatial_consistency_ci_low=ci_low,
        spatial_consistency_ci_high=ci_high,
        phase_coherence_mean=phase_mean,
        phase_coherence_top10pct=phase_top,
        spectral_magnitude_consistency=spectral_mag,
        radial_profile_consistency=radial_consistency,
        radial_profile_effect=radial_effect,
        angular_profile_consistency=angular_consistency,
        angular_profile_effect=angular_effect,
        translation_xcorr_mean=xcorr,
        periodicity_peak_ratio=periodicity,
        residual_auc=residual_auc,
        residual_auc_std=residual_std,
        residual_auc_perm_p=perm_p,
        patch_auc=patch_auc,
        patch_auc_std=patch_std,
        wavelet_best_auc=wavelet_auc,
        wavelet_best_band=wavelet_band,
        y_auc=channel_aucs[0],
        cb_auc=channel_aucs[1],
        cr_auc=channel_aucs[2],
        lsb_auc=lsb_auc,
        jpeg_dct_auc=jpeg_dct_auc,
        energy_distance_residual=e_dist,
        wasserstein_residual=w_dist,
        mmd_residual=mmd,
        rgb_auc=rgb_auc,
        content_leakage_gap=rgb_auc - residual_auc,
        resize_survival=survival["resize"],
        jpeg90_survival=survival["jpeg90"],
        jpeg75_survival=survival["jpeg75"],
        blur03_survival=survival["blur03"],
        blur05_survival=survival["blur05"],
        brightness_survival=survival["brightness"],
        flip_survival=survival["flip"],
        recommendation=recommendation,
        confidence=confidence,
        notes=notes,
    )

    detail = {
        "wavelet_feature_names": wave_names,
        "wavelet_positive_mean": pos_wave.mean(axis=0).tolist(),
        "wavelet_negative_mean": neg_wave.mean(axis=0).tolist(),
        "radial_positive_mean": pos_radial.mean(axis=0).tolist(),
        "radial_negative_mean": neg_radial.mean(axis=0).tolist(),
        "angular_positive_mean": pos_angular.mean(axis=0).tolist(),
        "angular_negative_mean": neg_angular.mean(axis=0).tolist(),
        "transform_survival": survival,
    }
    return summary, detail


def print_table(results: list[CategorySummary]) -> None:
    headers = [
        "Cat", "Res", "ResAUC", "p", "Patch", "Wave", "DCT",
        "Y", "Cb", "Cr", "Phase", "SpecMag", "Recommendation",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.category,
                r.resolution,
                f"{r.residual_auc:.3f}",
                f"{r.residual_auc_perm_p:.3f}",
                f"{r.patch_auc:.3f}",
                f"{r.wavelet_best_auc:.3f}",
                f"{r.jpeg_dct_auc:.3f}",
                f"{r.y_auc:.3f}",
                f"{r.cb_auc:.3f}",
                f"{r.cr_auc:.3f}",
                f"{r.phase_coherence_top10pct:.3f}",
                f"{r.spectral_magnitude_consistency:.3f}",
                r.recommendation,
            ]
        )
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def write_csv(path: Path, rows: list[CategorySummary]) -> None:
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_matrix_csv(
    path: Path, row_names: list[str], col_names: list[str], matrix: np.ndarray
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["detector"] + col_names)
        for name, values in zip(row_names, matrix):
            writer.writerow([name] + [f"{v:.8f}" for v in values])


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    clean_dir = root / "clean_targets"
    source_root = root / "watermarked_sources"
    if not clean_dir.is_dir() or not source_root.is_dir():
        raise FileNotFoundError(
            "Dataset must contain clean_targets/ and watermarked_sources/."
        )

    clean_paths = sorted_pngs(clean_dir)
    clean_by_resolution_paths: dict[tuple[int, int], list[Path]] = {}
    for path in clean_paths:
        clean_by_resolution_paths.setdefault(image_size(path), []).append(path)

    category_images: dict[str, list[np.ndarray]] = {}
    clean_by_resolution: dict[tuple[int, int], list[np.ndarray]] = {}
    summaries: list[CategorySummary] = []
    details: dict[str, dict] = {}

    for category in CATEGORIES:
        paths = sorted_pngs(source_root / category)
        if len(paths) != 25:
            raise RuntimeError(
                f"{category}: expected 25 source images, found {len(paths)}"
            )
        sizes = Counter(image_size(path) for path in paths)
        if len(sizes) != 1:
            raise RuntimeError(f"{category}: mixed source resolutions: {dict(sizes)}")
        resolution = next(iter(sizes))

        negative_paths = clean_by_resolution_paths.get(resolution, [])
        if not negative_paths:
            raise RuntimeError(
                f"{category}: no clean images found at resolution {resolution}"
            )
        negative_paths = negative_paths[: args.max_negatives]

        source_images = [load_rgb(path) for path in paths]
        negative_images = [load_rgb(path) for path in negative_paths]
        category_images[category] = source_images
        clean_by_resolution.setdefault(resolution, negative_images)

        print(f"Analyzing {category} ({resolution[0]}x{resolution[1]})...")
        summary, detail = analyze_category(
            category, source_images, negative_images, args
        )
        summaries.append(summary)
        details[category] = detail

    matrix_labels, matrix = category_specificity_matrix(
        category_images, clean_by_resolution, args.blur_sigma, args.seed
    )

    print()
    print_table(summaries)

    write_csv(output / "watermark_diagnostics_extended.csv", summaries)
    (output / "watermark_diagnostics_extended.json").write_text(
        json.dumps(
            {
                "summary": [asdict(item) for item in summaries],
                "details": details,
                "specificity_matrix": {
                    "rows": CATEGORIES,
                    "columns": matrix_labels,
                    "values": matrix.tolist(),
                },
            },
            indent=2,
        )
    )
    write_matrix_csv(
        output / "category_specificity_matrix.csv",
        CATEGORIES,
        matrix_labels,
        matrix,
    )

    print(f"\nSaved: {output / 'watermark_diagnostics_extended.json'}")
    print(f"Saved: {output / 'watermark_diagnostics_extended.csv'}")
    print(f"Saved: {output / 'category_specificity_matrix.csv'}")

    print("\nRecommendations:")
    for item in summaries:
        print(
            f"\n{item.category} [{item.resolution}] "
            f"{item.recommendation} ({item.confidence})"
        )
        print(f"  {item.notes}")


if __name__ == "__main__":
    main()
