#!/usr/bin/env python3
"""Sweep BM3D's sigma_psd for the 'regen' extraction and report watermark
residual CONSISTENCY per value, so you can pick sigma locally instead of
guessing across leaderboard submission cooldowns.

Consistency = ||mean(residuals)||^2 / mean(||residual||^2) across the 25
sources of a category. This is a local proxy for "how cleanly does a fixed
watermark isolate" -- higher means the per-source residuals agree with each
other more (less content noise, more shared watermark), not a guarantee about
real bit-accuracy, but it's exactly the measurement that showed regen beats
raw averaging for WM_5/WM_6 in the first place.

Usage:
    pip install bm3d
    python sweep_regen_sigma.py --dataset dataset --categories WM_5,WM_6 \
        --sigma-grid 0.0025,0.005,0.01,0.015,0.02,0.03,0.04
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--categories", default="WM_5,WM_6")
    p.add_argument("--sigma-grid", default="0.0025,0.005,0.01,0.015,0.02,0.03,0.04")
    p.add_argument("--n-sources", type=int, default=25, help="how many of the 25 sources to use (fewer = faster)")
    return p.parse_args()


def consistency(residuals):
    R = np.stack([r.ravel() for r in residuals])
    m = R.mean(0)
    return float((m @ m) / (np.mean(np.sum(R * R, 1)) + 1e-12))


def bm3d_residual(x, sigma, bm3d):
    r = np.empty_like(x)
    for c in range(x.shape[-1]):
        r[..., c] = x[..., c] - bm3d.bm3d(x[..., c], sigma_psd=sigma)
    return r


def main():
    args = parse_args()
    try:
        import bm3d
    except ImportError as e:
        raise SystemExit("needs BM3D: pip install bm3d") from e

    src, _ = load_dataset(args.dataset)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    sigmas = [float(v) for v in args.sigma_grid.split(",")]

    print(f"{'cat':5} {'sigma':>7} {'consistency':>12}")
    print("-" * 30)
    for category in categories:
        sources = src[category][: args.n_sources]
        best_sigma, best_score = None, -1.0
        for sigma in sigmas:
            residuals = [bm3d_residual(x, sigma, bm3d) for x in sources]
            score = consistency(residuals)
            marker = ""
            if score > best_score:
                best_score, best_sigma = score, sigma
            print(f"{category:5} {sigma:7.4f} {score:12.4f}")
        print(f"  -> best for {category}: sigma={best_sigma} (consistency={best_score:.4f})\n")


if __name__ == "__main__":
    main()
