from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    CATEGORIES,
    CATEGORY_RANGES,
    high_pass,
    load_dataset,
    rgb_to_ycbcr,
    save_rgb,
    ycbcr_to_rgb,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("baseline_candidates"))
    p.add_argument("--strength-grid", default="0.01,0.02,0.04,0.08")
    p.add_argument("--categories", default=",".join(CATEGORIES))
    return p.parse_args()


def mean_residual_template(xs):
    residual = np.mean(np.stack([high_pass(rgb_to_ycbcr(x)) for x in xs]), 0)
    return residual / (np.std(residual) + 1e-8)


def apply_template(x, template, strength):
    y = rgb_to_ycbcr(x)
    y[..., 0] = np.clip(y[..., 0] + strength * template[..., 0], 0, 1)
    y[..., 1] = np.clip(y[..., 1] + strength * template[..., 1], 0, 1)
    y[..., 2] = np.clip(y[..., 2] + strength * template[..., 2], 0, 1)
    return ycbcr_to_rgb(y)


def main():
    args = parse_args()
    src, clean = load_dataset(args.dataset)

    categories = [c.strip() for c in args.categories.split(",")]
    templates = {c: mean_residual_template(src[c]) for c in categories}

    for strength in [float(v) for v in args.strength_grid.split(",")]:
        out_dir = args.output_dir / f"strength_{strength:g}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, x in clean.items():
            category = next(
                (c for c, (lo, hi) in CATEGORY_RANGES.items() if lo <= i <= hi), None
            )
            y = apply_template(x, templates[category], strength) if category in templates else x
            save_rgb(y, out_dir / f"{i}.png")

        print("saved", out_dir)


if __name__ == "__main__":
    main()
