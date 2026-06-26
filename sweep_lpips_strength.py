#!/usr/bin/env python3
"""Find the real usable strength budget per specialized category by directly
measuring LPIPS, instead of guessing a conservative value and eyeballing the
output.

The conservative strength grid this pipeline started with (0.0025-0.02) was
never validated against where LPIPS actually starts climbing -- "doesn't
look visually disturbed" at a given strength is a sign of unclaimed Sdet
budget, not evidence the strength is well-calibrated. Since
Sqlt = exp(-8*LPIPS), this prints/saves a table of strength -> mean LPIPS ->
Sqlt per category so you can see exactly where the curve bends, instead of
picking a strength blind.

Requires the `lpips` package (pip install lpips) -- there is no MSE-proxy
fallback here, since the entire point of this tool is the real metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import CATEGORY_RANGES, load_dataset, write_json
from forge_specialized import (
    WM3_CHANNELS,
    apply_channel,
    apply_dct,
    apply_lsb_pair,
    apply_luma,
    channel_template,
    dct_block_stats,
    lsb_template,
    phase_template,
    select_dct_coords,
)

SPECIALIZED_CATEGORIES = ["WM_1", "WM_3", "WM_4", "WM_5", "WM_6"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--categories", default=",".join(SPECIALIZED_CATEGORIES))
    p.add_argument(
        "--strength-grid",
        default="0.0025,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5",
        help="wide by design -- the point is to find where LPIPS bends, not to confirm a guess",
    )
    p.add_argument("--wm4-threshold", type=float, default=0.45)
    p.add_argument("--wm6-coeff-count", type=int, default=8)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--output", type=Path, default=Path("lpips_strength_sweep.json"))
    return p.parse_args()


def build_context(src, by_resolution, args):
    ctx = {}
    if "WM_1" in src:
        ctx["wm1_cb"] = channel_template(src["WM_1"], 1)
    if "WM_3" in src:
        ctx["wm3"] = {ch: channel_template(src["WM_3"], ch) for ch in WM3_CHANNELS}
    if "WM_4" in src:
        ctx["wm4_phase"] = phase_template(src["WM_4"], args.wm4_threshold)
    if "WM_5" in src:
        ctx["wm5_cb_residual"] = channel_template(src["WM_5"], 1)
        ctx["wm5_cr_residual"] = channel_template(src["WM_5"], 2)
        ctx["wm5_cb_lsb"] = lsb_template(src["WM_5"], 1)
        ctx["wm5_cr_lsb"] = lsb_template(src["WM_5"], 2)
    if "WM_6" in src:
        resolution = (src["WM_6"][0].shape[1], src["WM_6"][0].shape[0])
        ctx["wm6_stats"] = dct_block_stats(src["WM_6"])
        ctx["wm6_clean_stats"] = dct_block_stats(by_resolution[resolution])
        ctx["wm6_coords"] = select_dct_coords(ctx["wm6_stats"], ctx["wm6_clean_stats"], args.wm6_coeff_count)
    return ctx


def forge(category, x, strength, ctx):
    if category == "WM_1":
        return apply_channel(x, ctx["wm1_cb"], 1, strength)
    if category == "WM_3":
        y = x
        for ch in WM3_CHANNELS:
            y = apply_channel(y, ctx["wm3"][ch], ch, strength)
        return y
    if category == "WM_4":
        return apply_luma(x, ctx["wm4_phase"], strength)
    if category == "WM_5":
        y = apply_channel(x, ctx["wm5_cb_residual"], 1, strength)
        y = apply_channel(y, ctx["wm5_cr_residual"], 2, strength)
        return apply_lsb_pair(y, ctx["wm5_cb_lsb"], ctx["wm5_cr_lsb"])
    if category == "WM_6":
        return apply_dct(x, ctx["wm6_coords"], ctx["wm6_stats"], ctx["wm6_clean_stats"], min(1, strength * 25))
    raise ValueError(category)


def main():
    args = parse_args()
    try:
        import lpips
        import torch
    except ImportError as e:
        raise SystemExit(
            "this tool requires torch and the lpips package (pip install lpips) -- "
            "there is no proxy-metric fallback since the point is the real metric"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    for param in net.parameters():
        param.requires_grad_(False)

    def lpips_distance(a, b):
        ta = torch.from_numpy(np.transpose(a, (2, 0, 1))).unsqueeze(0).float().to(device)
        tb = torch.from_numpy(np.transpose(b, (2, 0, 1))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            return float(net(ta, tb, normalize=True).mean())

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    strengths = [float(v) for v in args.strength_grid.split(",")]

    src, clean = load_dataset(args.dataset)
    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault((im.shape[1], im.shape[0]), []).append(im)
    ctx = build_context(src, by_resolution, args)

    results = []
    for category in categories:
        lo, hi = CATEGORY_RANGES[category]
        targets = [(i, x) for i, x in clean.items() if lo <= i <= hi]

        for strength in strengths:
            distances = [lpips_distance(forge(category, x, strength, ctx), x) for _, x in targets]
            mean_lpips = float(np.mean(distances))
            sqlt = float(np.exp(-8 * mean_lpips))
            results.append({
                "category": category,
                "strength": strength,
                "mean_lpips": mean_lpips,
                "sqlt": sqlt,
            })
            print(f"{category}  strength={strength:<8g}  mean_lpips={mean_lpips:.4f}  Sqlt={sqlt:.4f}")

    write_json(args.output, results)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
