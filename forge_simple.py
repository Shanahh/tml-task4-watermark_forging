#!/usr/bin/env python3
"""The simplest possible forging approach: raw mean-difference templates.

No high-pass filtering, no YCbCr/channel decomposition, no per-channel
calibration, no denoiser, no block statistics -- just "Can Simple Averaging
Defeat Modern Watermarks?" (Yang et al., NeurIPS 2024) applied literally in
the forging direction, with no embellishment:

    delta   = mean(25 source images of category k)  -  mean(clean images at
              the same resolution)
    forged  = clean_target + s * delta

Averaging cancels each source's differing content, leaving (an estimate of)
the one thing they all share: the watermark itself. Applied identically to
ALL 8 categories via the exact WM_k -> clean-image-id mapping from the
assignment (common.CATEGORY_RANGES / category_for_id) -- including WM_2/7/8,
which earlier high-pass-based diagnostics found nothing in, but that
assumption (the watermark being a high-frequency residual) was never itself
validated.

This exists to test that assumption directly: every other attack in this
repo high-pass filters before averaging. If raw averaging does no better
than 0.22, the high-pass assumption probably isn't the problem. If it does
the same or better with far less machinery, it's the floor everything else
should have been validated against.

Validated on the leaderboard: a single shared strength took the score from
0.22 -> 0.265 (s=0.3) -> 0.295 (s=0.5), then plateaued at s=0.6. Strength is
now per-category (--wm1-strength ... --wm8-strength, still nothing else
added) since the 8 watermarks are independent and likely have different real
embedding strengths, so a single shared value is necessarily a compromise --
the same lesson learned with the calibrated specialized pipeline, reapplied
here without bringing back any of that pipeline's other machinery. Defaults
are all 0.5 (the best uniform value found so far), so running with no
overrides reproduces that submission exactly.

Per-category delta EXTRACTION (--wmN-extraction, default raw):

    raw      delta = mean(sources) - mean(clean_pool)           [default]
    denoise  delta = mean over sources of [source - denoise(source)],
             then rescaled to the raw delta's per-category RMS.

The 'denoise' estimate removes each source's OWN content (via a wavelet
denoiser -- the Watermark Copy Attack extractor) before averaging, instead
of subtracting a foreign clean pool's mean. That drops the content-bias term
in the raw estimate (mean-content-of-sources minus mean-content-of-pool,
which does not fully cancel with only 25 sources), so for a noise-like
watermark it should give a cleaner delta DIRECTION -- the thing that helps
categories whose score is flat or falling in strength (delta points slightly
wrong), where more magnitude cannot help. NOTE: this is NOT the high-pass
mistake -- high-pass removes low frequencies (incl. low-freq watermark); a
denoiser keeps the noise-like watermark and removes content (opposite
selectivity).

The denoise residual is ~200-500x smaller in RMS than the raw delta (the raw
delta is dominated by low-frequency content-bias the denoiser excludes), so
it is rescaled to the raw delta's RMS. This keeps the strength range
identical to the raw attack AND makes raw-vs-denoise a clean single-variable
comparison: same perturbation amplitude, only the DIRECTION differs. Opt-in
per category; default raw reproduces the tuned result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import CATEGORIES, load_dataset, save_rgb, category_for_id
from forge_specialized import estimate_content

EXTRACTION_CHOICES = ("raw", "denoise")


def arg_name(category, suffix):
    return f"{category.lower().replace('_', '')}_{suffix}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("simple_candidates"))
    for category in CATEGORIES:
        stem = category.lower().replace("_", "")
        p.add_argument(f"--{stem}-strength", type=float, default=0.5)
        p.add_argument(f"--{stem}-extraction", choices=EXTRACTION_CHOICES, default="raw")
    return p.parse_args()


def raw_delta(sources, clean_pool):
    return np.mean(sources, axis=0) - np.mean(clean_pool, axis=0)


def denoise_delta(sources):
    """Per-source watermark estimate source - denoise(source), averaged.
    Denoises each RGB channel independently."""
    residuals = []
    for x in sources:
        residual = np.empty_like(x)
        for c in range(x.shape[-1]):
            residual[..., c] = x[..., c] - estimate_content(x[..., c], "denoiser")
        residuals.append(residual)
    return np.mean(residuals, axis=0)


def rms(a):
    return float(np.sqrt((a ** 2).mean()))


def main():
    args = parse_args()
    src, clean = load_dataset(args.dataset)
    strengths = {c: getattr(args, arg_name(c, "strength")) for c in CATEGORIES}
    extractions = {c: getattr(args, arg_name(c, "extraction")) for c in CATEGORIES}
    print("config:", " ".join(f"{c}={strengths[c]}/{extractions[c]}" for c in CATEGORIES))

    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault(im.shape[:2], []).append(im)

    deltas = {}
    for category in CATEGORIES:
        sources = src[category]
        rd = raw_delta(sources, by_resolution[sources[0].shape[:2]])
        if extractions[category] == "denoise":
            dd = denoise_delta(sources)
            # Rescale to the raw delta's RMS: same amplitude, direction-only
            # difference, and the same strength range as the raw attack.
            deltas[category] = dd * (rms(rd) / (rms(dd) + 1e-12))
        else:
            deltas[category] = rd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, x in clean.items():
        category = category_for_id(i)
        forged = np.clip(x + strengths[category] * deltas[category], 0, 1)
        save_rgb(forged, args.output_dir / f"{i}.png")

    print("saved", args.output_dir)


if __name__ == "__main__":
    main()
