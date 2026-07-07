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
    p.add_argument("--wm1-strength", type=float, default=0.002)
    p.add_argument("--wm3-y-strength", type=float, default=0.007)
    p.add_argument("--wm3-cb-strength", type=float, default=0.001)
    p.add_argument("--wm3-cr-strength", type=float, default=0.001)
    p.add_argument("--wm4-strength", type=float, default=0.009)
    p.add_argument("--wm5-strength", type=float, default=0.007)
    p.add_argument("--wm4-threshold", type=float, default=0.4)
    p.add_argument(
        "--extraction",
        default="highpass",
        choices=EXTRACTION_METHODS,
        help="must match the --extraction passed to forge_specialized.py: the genuine "
        "amplitude (and so s*) differs between extraction methods because they "
        "produce different template directions.",
    )
    return p.parse_args()


def luma_residual(x):
    g = grayscale(x)
    return g - gaussian_filter(g, 1.5, mode="reflect")


def project(residual, t_hat):
    return float(np.sum(residual * t_hat))


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
