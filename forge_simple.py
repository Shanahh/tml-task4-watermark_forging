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
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import CATEGORIES, load_dataset, save_rgb, category_for_id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("simple_candidates"))
    for category in CATEGORIES:
        p.add_argument(f"--{category.lower().replace('_', '')}-strength", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    src, clean = load_dataset(args.dataset)
    strengths = {c: getattr(args, f"{c.lower().replace('_', '')}_strength") for c in CATEGORIES}
    print("strengths:", " ".join(f"{c}={strengths[c]}" for c in CATEGORIES))

    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault(im.shape[:2], []).append(im)

    deltas = {}
    for category in CATEGORIES:
        sources = src[category]
        resolution = sources[0].shape[:2]
        clean_pool = by_resolution[resolution]
        deltas[category] = np.mean(sources, axis=0) - np.mean(clean_pool, axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, x in clean.items():
        category = category_for_id(i)
        forged = np.clip(x + strengths[category] * deltas[category], 0, 1)
        save_rgb(forged, args.output_dir / f"{i}.png")

    print("saved", args.output_dir)


if __name__ == "__main__":
    main()
