#!/usr/bin/env python3
"""Verify that the WM_5 LSB attack actually reproduces its intended bit
pattern after a real PNG save/reload, instead of just trusting the
in-memory computation.

This check exists because of a real bug found in practice: an earlier
version of apply_lsb() set the bit on the *derived floating-point* Cb/Cr
value and converted back to RGB. save_rgb() then rounds that RGB to uint8
for the PNG, and re-deriving Cb/Cr from the rounded RGB after reload does
not reproduce the intended byte -- empirically, 100% of bits were lost this
way. apply_lsb() now works directly in the persisted integer RGB domain
instead, but this script exists so that regressions (or similar bugs in
other channel-domain attacks) get caught automatically rather than silently
capping the real submission score the way this one did.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import CATEGORY_RANGES, load_dataset, load_rgb, rgb_to_ycbcr, save_rgb
from forge_specialized import apply_channel, apply_lsb, apply_lsb_pair, channel_template, lsb_template


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--scratch-dir", type=Path, default=Path("lsb_roundtrip_scratch"))
    p.add_argument("--n-images", type=int, default=25, help="how many WM_5 targets to check (max 25)")
    p.add_argument("--wm5-strength", type=float, default=0.0071, help="must match forge_specialized.py's --wm5-strength")
    return p.parse_args()


def check_channel(x, template, ch, scratch_path):
    """Apply the LSB attack, save it as a real PNG, reload it, and check
    whether the reloaded image's Cb/Cr byte actually has the intended LSB."""
    forged = apply_lsb(x, template, ch)
    save_rgb(forged, scratch_path)
    reloaded = load_rgb(scratch_path)
    recovered_bit = np.round(rgb_to_ycbcr(reloaded)[..., ch] * 255).astype(np.uint8) & 1
    return float(np.mean(recovered_bit != template))


def main():
    args = parse_args()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    src, clean = load_dataset(args.dataset)
    lo, hi = CATEGORY_RANGES["WM_5"]
    target_ids = [i for i in clean if lo <= i <= hi][: args.n_images]

    cb_template = lsb_template(src["WM_5"], 1)
    cr_template = lsb_template(src["WM_5"], 2)
    # also exercise the combined attack's residual half, to confirm it
    # doesn't somehow interfere with the LSB half when applied together
    cb_residual = channel_template(src["WM_5"], 1)
    cr_residual = channel_template(src["WM_5"], 2)

    cb_mismatches, cr_mismatches, combined_mismatches = [], [], []
    for i in target_ids:
        x = clean[i]
        cb_mismatches.append(check_channel(x, cb_template, 1, args.scratch_dir / f"{i}_cb.png"))
        cr_mismatches.append(check_channel(x, cr_template, 2, args.scratch_dir / f"{i}_cr.png"))

        combined = apply_channel(x, cb_residual, 1, args.wm5_strength)
        combined = apply_channel(combined, cr_residual, 2, args.wm5_strength)
        combined = apply_lsb_pair(combined, cb_template, cr_template)
        save_rgb(combined, args.scratch_dir / f"{i}_combined.png")
        reloaded = load_rgb(args.scratch_dir / f"{i}_combined.png")
        ycc = rgb_to_ycbcr(reloaded)
        cb_bit = np.round(ycc[..., 1] * 255).astype(np.uint8) & 1
        cr_bit = np.round(ycc[..., 2] * 255).astype(np.uint8) & 1
        mismatch = float(np.mean((cb_bit != cb_template) | (cr_bit != cr_template)))
        combined_mismatches.append(mismatch)

    print(f"checked {len(target_ids)} WM_5 targets")
    print(f"Cb LSB-only mean bit mismatch after real PNG round trip: {np.mean(cb_mismatches):.6f}")
    print(f"Cr LSB-only mean bit mismatch after real PNG round trip: {np.mean(cr_mismatches):.6f}")
    print(f"Combined (residual+LSB) mean bit mismatch after real PNG round trip: {np.mean(combined_mismatches):.6f}")

    threshold = 0.01
    if max(np.mean(cb_mismatches), np.mean(cr_mismatches), np.mean(combined_mismatches)) > threshold:
        print(f"FAIL: bit mismatch exceeds {threshold:.0%} -- the LSB attack is not reliably "
              "surviving the save/reload round trip, investigate before submitting")
    else:
        print("PASS: LSB bits reliably survive the PNG round trip")


if __name__ == "__main__":
    main()
