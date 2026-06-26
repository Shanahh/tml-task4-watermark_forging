#!/usr/bin/env python3
"""Calibrate per-category attack strength by matching the genuine watermark's
OWN amplitude, read directly off the 25 source images, instead of maximizing
the perturbation within an LPIPS budget.

Why this exists
---------------
The score is Sdet * Sqlt. We can measure Sqlt locally (LPIPS), but we have no
local measurement of Sdet (bit accuracy of the real, unseen decoder). Every
strength choice so far has therefore been "how hard can we push within the
quality budget" -- maximize subject to LPIPS. That is the wrong objective: for
an additive, content-independent watermark the correct strength is the one
that REPRODUCES the genuine watermark's amplitude, not the largest one quality
allows. Overshooting puts the forgery outside the distribution of genuine
watermarked images, which a normalizing / sign-based decoder can read *worse*,
while also costing Sqlt -- a double loss.

The model and the measurement
-----------------------------
For WM_1/3/4/5 the watermark is additive and content-independent: every source
is clean_content + delta, the same delta. The extracted template t-hat points
in the delta direction. Projecting a residual onto t-hat collapses an image to
a single number = "how much watermark is present along its own axis":

  - The 25 sources form a tight cluster: mean mu_s, std sigma_s. This IS the
    genuine watermark amplitude and its natural spread.
  - A clean target projects to ~0 (no watermark; only content leakage).
  - A forgery clean + s*t projects linearly in s. The strength s* whose
    forgery projection equals mu_s is the amplitude-calibrated strength: it
    makes the forgery as watermarked as a genuine source, no more, no less.

This tool reports, per category and channel, mu_s +/- sigma_s, the calibrated
s*, and -- crucially -- where the currently-configured strength lands relative
to the genuine cluster, in units of sigma_s. A current strength sitting many
sigma above mu_s is overshooting (likely hurting Sdet); far below is
under-driving.

WM_6 is intentionally skipped: its DCT distribution-matching attack already
calibrates to the source coefficient distribution by construction (it
interpolates toward the WM_6 source stats and saturates at full match), so
amplitude is not a free knob there.

This is a read-only diagnostic -- it changes nothing, it only tells you which
strength to pass to forge_specialized.py / run_pipeline.sh.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from common import CATEGORY_RANGES, EPS, grayscale, load_dataset, write_json
from forge_specialized import (
    EXTRACTION_METHODS,
    apply_channel,
    apply_luma,
    channel_residual,
    channel_template,
    phase_template,
)

CHANNEL_LABELS = {0: "Y", 1: "Cb", 2: "Cr"}
# Reference strengths used to fit the (strength -> projection) line. Spanning
# 0 plus a few small values keeps the fit in the near-linear regime.
REF_STRENGTHS = [0.0, 0.01, 0.02, 0.04]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("strength_calibration.json"))
    # The strengths currently in use, so the report can show where they land.
    # Defaults mirror forge_specialized.py's defaults.
    p.add_argument("--wm1-strength", type=float, default=0.0024)
    p.add_argument("--wm3-y-strength", type=float, default=0.0079)
    p.add_argument("--wm3-cb-strength", type=float, default=0.0017)
    p.add_argument("--wm3-cr-strength", type=float, default=0.0017)
    p.add_argument("--wm4-strength", type=float, default=0.0091)
    p.add_argument("--wm5-strength", type=float, default=0.0071)
    p.add_argument("--wm4-threshold", type=float, default=0.45)
    p.add_argument(
        "--extraction",
        default="highpass",
        choices=EXTRACTION_METHODS,
        help="must match the --extraction passed to forge_specialized.py: the genuine "
        "amplitude (and so s*) differs between extraction methods because they "
        "produce different template directions.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# Residual operators -- must match how each template's sources are built in
# forge_specialized.py, so the projection is in the right domain. channel_
# residual is imported from forge_specialized so the two stay identical; only
# luma_residual (for WM_4's phase template) is local.
# --------------------------------------------------------------------------

def luma_residual(x):
    g = grayscale(x)
    return g - gaussian_filter(g, 1.5, mode="reflect")


def project(residual, t_hat):
    return float(np.sum(residual * t_hat))


# --------------------------------------------------------------------------
# Per (category, channel) attack specs: template + matching residual + apply
# --------------------------------------------------------------------------

def build_specs(src, current_strengths, wm4_threshold, method):
    specs = []

    for cat, channels in [("WM_1", [1]), ("WM_3", [0, 1, 2]), ("WM_5", [1, 2])]:
        for ch in channels:
            template = channel_template(src[cat], ch, method)
            specs.append({
                "category": cat,
                "label": CHANNEL_LABELS[ch],
                "template": template,
                "residual": (lambda x, ch=ch: channel_residual(x, ch, method)),
                "apply": (lambda x, s, template=template, ch=ch: apply_channel(x, template, ch, s)),
                "current_strength": current_strengths[(cat, ch)],
            })

    phase_t = phase_template(src["WM_4"], wm4_threshold)
    specs.append({
        "category": "WM_4",
        "label": "Y-phase",
        "template": phase_t,
        "residual": luma_residual,
        "apply": (lambda x, s, template=phase_t: apply_luma(x, template, s)),
        "current_strength": current_strengths[("WM_4", 0)],
    })

    return specs


def calibrate(spec, sources, clean_targets):
    template = spec["template"]
    t_hat = template / (np.sqrt(np.sum(template ** 2)) + EPS)
    residual = spec["residual"]
    apply = spec["apply"]

    genuine = np.array([project(residual(s), t_hat) for s in sources])
    mu_s, sigma_s = float(genuine.mean()), float(genuine.std())

    # Fit forgery projection as a linear function of strength: proj = a + k*s.
    strengths = np.array(REF_STRENGTHS, dtype=float)
    proj_means = np.array([
        np.mean([project(residual(apply(x, s)), t_hat) for x in clean_targets])
        for s in strengths
    ])
    k, a = np.polyfit(strengths, proj_means, 1)
    k, a = float(k), float(a)

    s_star = (mu_s - a) / k if abs(k) > EPS else float("nan")

    s_now = spec["current_strength"]
    proj_now = a + k * s_now
    z_now = (proj_now - mu_s) / (sigma_s + EPS)

    if z_now > 1.5:
        verdict = "OVERSHOOT (above the genuine watermark cluster -- likely hurting Sdet)"
    elif z_now < -1.5:
        verdict = "UNDERSHOOT (below the genuine watermark cluster -- leaving Sdet on the table)"
    else:
        verdict = "match (within the genuine cluster)"

    return {
        "category": spec["category"],
        "channel": spec["label"],
        "genuine_mean": mu_s,
        "genuine_std": sigma_s,
        "content_baseline": a,
        "gain_per_strength": k,
        "calibrated_strength": s_star,
        "current_strength": s_now,
        "current_z_vs_genuine": float(z_now),
        "verdict": verdict,
    }


def main():
    args = parse_args()
    # Keyed by (category, channel) so WM_3's per-channel strengths are
    # reported accurately (its Y / Cb / Cr genuine amplitudes differ).
    current_strengths = {
        ("WM_1", 1): args.wm1_strength,
        ("WM_3", 0): args.wm3_y_strength,
        ("WM_3", 1): args.wm3_cb_strength,
        ("WM_3", 2): args.wm3_cr_strength,
        ("WM_4", 0): args.wm4_strength,
        ("WM_5", 1): args.wm5_strength,
        ("WM_5", 2): args.wm5_strength,
    }

    src, clean = load_dataset(args.dataset)
    specs = build_specs(src, current_strengths, args.wm4_threshold, args.extraction)

    results = []
    print(f"extraction = {args.extraction}")
    print(f"{'cat':5} {'chan':7} {'genuine mu+/-sd':>20} {'s*':>9} {'s_now':>8} {'z@now':>7}  verdict")
    print("-" * 100)
    for spec in specs:
        lo, hi = CATEGORY_RANGES[spec["category"]]
        clean_targets = [clean[i] for i in range(lo, hi + 1)]
        r = calibrate(spec, src[spec["category"]], clean_targets)
        results.append(r)
        print(
            f"{r['category']:5} {r['channel']:7} "
            f"{r['genuine_mean']:9.3f}+/-{r['genuine_std']:<7.3f} "
            f"{r['calibrated_strength']:9.4f} {r['current_strength']:8.4f} "
            f"{r['current_z_vs_genuine']:7.2f}  {r['verdict']}"
        )

    write_json(args.output, results)
    print(f"\nwrote {args.output}")

    # Emit a copy-pasteable forge_specialized.py flag string with the
    # calibrated s* values. WM_3 takes per-channel flags; the other channel
    # categories take one flag (WM_5's Cb/Cr are averaged since the attack
    # uses one strength for both).
    by_cat_ch = {(r["category"], r["channel"]): r["calibrated_strength"] for r in results}
    flags = [f"--wm1-strength {by_cat_ch[('WM_1', 'Cb')]:.4f}"]
    flags.append(f"--wm3-y-strength {by_cat_ch[('WM_3', 'Y')]:.4f}")
    flags.append(f"--wm3-cb-strength {by_cat_ch[('WM_3', 'Cb')]:.4f}")
    flags.append(f"--wm3-cr-strength {by_cat_ch[('WM_3', 'Cr')]:.4f}")
    flags.append(f"--wm4-strength {by_cat_ch[('WM_4', 'Y-phase')]:.4f}")
    wm5 = 0.5 * (by_cat_ch[("WM_5", "Cb")] + by_cat_ch[("WM_5", "Cr")])
    flags.append(f"--wm5-strength {wm5:.4f}")

    print("\nCalibrated strengths for forge_specialized.py "
          f"(--extraction {args.extraction}):")
    print("  " + " ".join(flags))


if __name__ == "__main__":
    main()
