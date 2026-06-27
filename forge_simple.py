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
    p.add_argument("--strength", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    src, clean = load_dataset(args.dataset)

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
        delta = deltas[category_for_id(i)]
        forged = np.clip(x + args.strength * delta, 0, 1)
        save_rgb(forged, args.output_dir / f"{i}.png")

    print("saved", args.output_dir)


if __name__ == "__main__":
    main()
