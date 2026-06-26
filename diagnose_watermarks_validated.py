#!/usr/bin/env python3
"""Validated, out-of-fold diagnostics for the 8 unknown watermark groups.

For each category, trains small logistic-regression detectors (out-of-fold,
so reported AUCs are not inflated by overfitting) on several feature families
(raw residual, per-channel residual, LSB bit-planes, block-DCT coefficients),
runs permutation significance tests, checks robustness to common image
transforms, and builds a cross-category specificity matrix.

A feature should only be treated as a usable attack target when:
  1. its out-of-fold AUC is high;
  2. its permutation p-value is low;
  3. the intended category scores higher than clean images;
  4. the intended category scores higher than other watermark groups;
  5. the signal survives a PNG decode-and-save round trip;
  6. the signal is not explained by raw image content.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    CATEGORIES,
    EPS,
    from_pil,
    grayscale,
    high_pass,
    jpeg_roundtrip,
    load_dataset,
    png_roundtrip,
    resize_np,
    rgb_to_ycbcr,
    to_pil,
    write_json,
)

DCT_FEATURE_COORDS = [
    (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2),
    (0, 3), (3, 0), (1, 3), (3, 1),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("diagnostics_validated"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--bootstraps", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def model(seed):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000, class_weight="balanced", C=0.25,
            solver="liblinear", random_state=seed,
        )),
    ])


# --------------------------------------------------------------------------
# Feature extractors
# --------------------------------------------------------------------------

def block_mean(a, grid=8):
    if a.ndim == 2:
        a = a[..., None]
    h, w, _ = a.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    out = []
    for yi in range(grid):
        for xi in range(grid):
            block = a[ys[yi] : ys[yi + 1], xs[xi] : xs[xi + 1]]
            out.extend(block.mean((0, 1)).tolist())
    return np.asarray(out, np.float32)


def residual_features(x):
    r = high_pass(x)
    g = grayscale(r)
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))

    encoded = np.repeat(g[..., None], 3, 2) * 0.25 + 0.5
    small = resize_np(np.clip(encoded, 0, 1), (64, 64))[..., 0]
    small = (small - 0.5) / 0.25
    dct = dctn(small, type=2, norm="ortho")[:16, :16]
    dct[0, 0] = 0

    return np.concatenate(
        [block_mean(r), block_mean(np.abs(r)), block_mean(spec), dct.ravel()]
    ).astype(np.float32)


def channel_features(x, ch):
    a = rgb_to_ycbcr(x)[..., ch]
    r = a - gaussian_filter(a, 1.5, mode="reflect")
    spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(r))))
    extra = [r.mean(), r.std(), np.mean(np.abs(r)), np.percentile(np.abs(r), 90)]
    return np.concatenate(
        [block_mean(r), block_mean(np.abs(r)), block_mean(spec), extra]
    ).astype(np.float32)


def lsb_features(x):
    u = np.clip(x * 255, 0, 255).round().astype(np.uint8)
    feats = []
    for c in range(3):
        for bit in range(2):
            plane = ((u[..., c] >> bit) & 1).astype(np.float32)
            q = plane.mean()
            entropy = -(q * np.log2(q + EPS) + (1 - q) * np.log2(1 - q + EPS))
            feats += [
                q,
                entropy,
                np.mean(plane[:, 1:] == plane[:, :-1]),
                np.mean(plane[1:] == plane[:-1]),
            ]
    return np.asarray(feats, np.float32)


def dct_features(x):
    g = grayscale(x)
    h, w = g.shape
    g = g[: h - h % 8, : w - w % 8] - 0.5
    blocks = []
    for y in range(0, g.shape[0], 8):
        for z in range(0, g.shape[1], 8):
            blocks.append(dctn(g[y : y + 8, z : z + 8], type=2, norm="ortho"))
    stacked = np.stack(blocks)

    feats = []
    for a, b in DCT_FEATURE_COORDS:
        v = stacked[:, a, b]
        feats += [v.mean(), v.std(), np.mean(np.abs(v)), np.mean(v > 0)]
    return np.asarray(feats, np.float32)


# --------------------------------------------------------------------------
# Spectral / periodicity diagnostics
# --------------------------------------------------------------------------

def spectral_signature(x):
    g = grayscale(high_pass(x))
    m = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    m = (m - m.mean()) / (m.std() + EPS)
    small = resize_np(np.repeat(np.clip(m / 8 + 0.5, 0, 1)[..., None], 3, 2), (64, 64))[..., 0]
    v = ((small - 0.5) * 8).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + EPS)


def pairwise(vectors):
    return np.asarray(
        [float(vectors[i] @ vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    )


def phase_top(xs):
    r = np.stack([grayscale(high_pass(x)) for x in xs])
    z = np.fft.fft2(r, axes=(1, 2))
    coherence = np.abs(np.mean(z / (np.abs(z) + EPS), 0))
    h, w = coherence.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    values = coherence[(radius >= 0.02) & (radius <= 0.95)]
    k = max(1, math.ceil(0.1 * values.size))
    return float(np.partition(values, -k)[-k:].mean())


def periodicity(x):
    r = grayscale(high_pass(x))
    r -= r.mean()
    autocorr = fftconvolve(r, r[::-1, ::-1], mode="full") / (np.sum(r * r) + EPS)
    cy, cx = np.array(autocorr.shape) // 2
    exclude = max(3, min(x.shape[:2]) // 64)
    mask = np.ones_like(autocorr, dtype=bool)
    mask[cy - exclude : cy + exclude + 1, cx - exclude : cx + exclude + 1] = False
    values = np.abs(autocorr[mask])
    return float(np.percentile(values, 99.9) / (np.sqrt(np.mean(values * values)) + EPS))


# --------------------------------------------------------------------------
# Out-of-fold detector evaluation, permutation tests, transform survival
# --------------------------------------------------------------------------

def cv_oof(pos, neg, fn, folds, seed):
    P = np.stack([fn(x) for x in pos])
    N = np.stack([fn(x) for x in neg])
    X = np.vstack([P, N])
    y = np.r_[np.ones(len(P), int), np.zeros(len(N), int)]

    splitter = StratifiedKFold(n_splits=min(folds, len(pos)), shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for train_idx, test_idx in splitter.split(X, y):
        m = model(seed)
        m.fit(X[train_idx], y[train_idx])
        pred[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    return float(roc_auc_score(y, pred)), X, y


def perm_p(X, y, observed_auc, folds, seed, count):
    rng = np.random.default_rng(seed)
    values = []
    for i in range(count):
        y_perm = rng.permutation(y)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + i + 1)
        pred = np.zeros(len(y))
        for train_idx, test_idx in splitter.split(X, y_perm):
            m = model(seed + i + 1)
            m.fit(X[train_idx], y_perm[train_idx])
            pred[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        values.append(roc_auc_score(y_perm, pred))
    return float((1 + np.sum(np.asarray(values) >= observed_auc)) / (1 + count))


def transform(name, x):
    pil = to_pil(x)
    h, w = x.shape[:2]
    if name == "png":
        return png_roundtrip(x)
    if name == "jpeg95":
        return jpeg_roundtrip(x, 95)
    if name == "jpeg75":
        return jpeg_roundtrip(x, 75)
    if name == "resize":
        d = max(32, min(h, w) // 2)
        small = pil.resize((d, d), Image.Resampling.BILINEAR)
        return from_pil(small.resize((w, h), Image.Resampling.BILINEAR))
    if name == "blur05":
        return from_pil(pil.filter(ImageFilter.GaussianBlur(0.5)))
    if name == "brightness":
        return from_pil(ImageEnhance.Brightness(pil).enhance(1.05))
    if name == "flip":
        return from_pil(pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    raise ValueError(name)


TRANSFORM_NAMES = ["png", "jpeg95", "jpeg75", "resize", "blur05", "brightness", "flip"]


def oof_survival(pos, neg, fn, folds, seed):
    images = pos + neg
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    F = np.stack([fn(x) for x in images])

    splitter = StratifiedKFold(n_splits=min(folds, len(pos)), shuffle=True, random_state=seed)
    base = np.zeros(len(y))
    out = {name: np.zeros(len(y)) for name in TRANSFORM_NAMES}

    for train_idx, test_idx in splitter.split(F, y):
        m = model(seed)
        m.fit(F[train_idx], y[train_idx])
        base[test_idx] = m.predict_proba(F[test_idx])[:, 1]
        for name in TRANSFORM_NAMES:
            transformed = np.stack([fn(transform(name, images[i])) for i in test_idx])
            out[name][test_idx] = m.predict_proba(transformed)[:, 1]

    pos_idx = np.where(y == 1)[0]
    baseline_margin = np.mean(base[pos_idx] - 0.5) + EPS
    return {
        name: float(np.mean(out[name][pos_idx] - 0.5) / baseline_margin)
        for name in TRANSFORM_NAMES
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

FEATURE_FUNCTIONS = {
    "residual": residual_features,
    "Y": lambda x: channel_features(x, 0),
    "Cb": lambda x: channel_features(x, 1),
    "Cr": lambda x: channel_features(x, 2),
    "LSB": lsb_features,
    "DCT": dct_features,
}
SIGNIFICANCE_TESTED_FEATURES = {"residual", "Cb", "LSB", "DCT"}
PROVENANCE_TRANSFORMS = ["png", "jpeg95", "jpeg75"]


def diagnose_category(category, src, by_resolution, args, rng):
    pos = src[category]
    neg = by_resolution[(pos[0].shape[1], pos[0].shape[0])]
    row = {"category": category, "resolution": f"{pos[0].shape[1]}x{pos[0].shape[0]}"}

    for name, fn in FEATURE_FUNCTIONS.items():
        auc, X, y = cv_oof(pos, neg, fn, args.folds, args.seed)
        row[f"{name}_auc"] = auc
        if name in SIGNIFICANCE_TESTED_FEATURES:
            row[f"{name}_perm_p"] = perm_p(X, y, auc, args.folds, args.seed, args.permutations)

    wm_signatures = pairwise([spectral_signature(x) for x in pos])
    clean_signatures = pairwise([spectral_signature(x) for x in neg])
    row["spectral_consistency_wm"] = float(wm_signatures.mean())
    row["spectral_consistency_clean"] = float(clean_signatures.mean())
    row["spectral_consistency_delta"] = float(wm_signatures.mean() - clean_signatures.mean())

    bootstrap_means = [
        float(rng.choice(wm_signatures, size=len(wm_signatures), replace=True).mean())
        for _ in range(args.bootstraps)
    ]
    row["spectral_ci_low"] = float(np.percentile(bootstrap_means, 2.5))
    row["spectral_ci_high"] = float(np.percentile(bootstrap_means, 97.5))

    row["phase_top_wm"] = phase_top(pos)
    row["phase_top_clean"] = phase_top(neg[: len(pos)])
    row["phase_top_delta"] = row["phase_top_wm"] - row["phase_top_clean"]

    p_periodicity = np.asarray([periodicity(x) for x in pos])
    q_periodicity = np.asarray([periodicity(x) for x in neg])
    pooled_std = np.sqrt(0.5 * (p_periodicity.var() + q_periodicity.var())) + EPS
    row["periodicity_z"] = float((p_periodicity.mean() - q_periodicity.mean()) / pooled_std)

    row["transform_survival_residual"] = oof_survival(pos, neg, residual_features, args.folds, args.seed)
    row["provenance_residual_auc"] = {
        name: cv_oof(
            [transform(name, x) for x in pos],
            [transform(name, x) for x in neg],
            residual_features,
            args.folds,
            args.seed,
        )[0]
        for name in PROVENANCE_TRANSFORMS
    }

    return row


def specificity_matrix(src, by_resolution, args):
    labels = CATEGORIES + ["clean"]
    M = np.zeros((len(CATEGORIES), len(labels)))

    for row_idx, category in enumerate(CATEGORIES):
        pos = src[category]
        neg = by_resolution[(pos[0].shape[1], pos[0].shape[0])]
        P = np.stack([residual_features(x) for x in pos])
        N = np.stack([residual_features(x) for x in neg])
        X = np.vstack([P, N])
        y = np.r_[np.ones(len(P), int), np.zeros(len(N), int)]

        splitter = StratifiedKFold(
            n_splits=min(args.folds, len(pos)), shuffle=True, random_state=args.seed + row_idx
        )
        pred = np.zeros(len(y))
        fold_models = []
        for train_idx, test_idx in splitter.split(X, y):
            m = model(args.seed + row_idx)
            m.fit(X[train_idx], y[train_idx])
            pred[test_idx] = m.predict_proba(X[test_idx])[:, 1]
            fold_models.append(m)

        M[row_idx, row_idx] = pred[: len(P)].mean()
        M[row_idx, -1] = pred[len(P) :].mean()

        for col_idx, other_category in enumerate(CATEGORIES):
            if other_category == category:
                continue
            F = np.stack([residual_features(x) for x in src[other_category]])
            M[row_idx, col_idx] = np.mean([m.predict_proba(F)[:, 1].mean() for m in fold_models])

    return labels, M


def write_csv_outputs(out_dir, rows, labels, M):
    with (out_dir / "validated_summary.csv").open("w", newline="") as f:
        keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, dict)})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if not isinstance(v, dict)})

    with (out_dir / "oof_specificity_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector"] + labels)
        for i, category in enumerate(CATEGORIES):
            writer.writerow([category] + list(M[i]))


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    src, clean = load_dataset(args.dataset)
    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault((im.shape[1], im.shape[0]), []).append(im)

    rng = np.random.default_rng(args.seed)
    rows = [diagnose_category(c, src, by_resolution, args, rng) for c in CATEGORIES]
    for c, row in zip(CATEGORIES, rows):
        print(c, row)

    labels, M = specificity_matrix(src, by_resolution, args)

    write_json(
        out_dir / "validated_diagnostics.json",
        {"summary": rows, "specificity_matrix": {"rows": CATEGORIES, "columns": labels, "values": M.tolist()}},
    )
    write_csv_outputs(out_dir, rows, labels, M)


if __name__ == "__main__":
    main()
