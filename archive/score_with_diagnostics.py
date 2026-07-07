from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import CATEGORIES, CATEGORY_RANGES, load_dataset, load_rgb, write_json
from diagnose_watermarks_validated import FEATURE_FUNCTIONS, model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--specialized-dir", type=Path, required=True,
                    help="forge_specialized.py output dir (flat <id>.png files)")
    p.add_argument("--categories", default="WM_1,WM_3,WM_4,WM_5,WM_6")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("score_with_diagnostics.json"))
    return p.parse_args()


def train_classifier(pos, neg, feature_fn, seed):
    X = np.stack([feature_fn(x) for x in pos] + [feature_fn(x) for x in neg])
    y = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    m = model(seed)
    m.fit(X, y)
    return m


def main():
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    src, clean = load_dataset(args.dataset)
    by_resolution = {}
    for i, im in clean.items():
        by_resolution.setdefault(im.shape[:2], []).append((i, im))

    results = []
    print(f"{'cat':5} {'feature':9} {'clean_p':>8} {'forged_p':>9} {'lift':>7}  verdict")
    print("-" * 70)

    for category in categories:
        lo, hi = CATEGORY_RANGES[category]
        pos = src[category]
        resolution = pos[0].shape[:2]
        neg = [im for i, im in by_resolution[resolution] if not (lo <= i <= hi)]
        if not neg:
            print(f"{category}: no negatives at resolution {resolution} excluding its own ids")
            neg = [im for i, im in by_resolution[resolution]]

        clean_targets = {i: clean[i] for i in range(lo, hi + 1)}
        forged_targets = {
            i: load_rgb(args.specialized_dir / f"{i}.png") for i in range(lo, hi + 1)
        }

        for feature_name, feature_fn in FEATURE_FUNCTIONS.items():
            clf = train_classifier(pos, neg, feature_fn, args.seed)

            clean_probs = [clf.predict_proba(feature_fn(x)[None, :])[0, 1] for x in clean_targets.values()]
            forged_probs = [clf.predict_proba(feature_fn(x)[None, :])[0, 1] for x in forged_targets.values()]

            mean_clean = float(np.mean(clean_probs))
            mean_forged = float(np.mean(forged_probs))
            lift = mean_forged - mean_clean

            if lift > 0.2 and mean_forged > 0.5:
                verdict = "registers strongly"
            elif lift > 0.05:
                verdict = "registers weakly"
            else:
                verdict = "DOES NOT REGISTER"

            results.append({
                "category": category, "feature": feature_name,
                "clean_prob": mean_clean, "forged_prob": mean_forged,
                "lift": lift, "verdict": verdict,
            })
            print(f"{category:5} {feature_name:9} {mean_clean:8.3f} {mean_forged:9.3f} {lift:7.3f}  {verdict}")

    write_json(args.output, results)
    print(f"\nwrote {args.output}")

    print("\nPer-category summary (best-registering feature):")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for category, rows in by_cat.items():
        best = max(rows, key=lambda r: r["lift"])
        if best["lift"] > 0.2 and best["forged_prob"] > 0.5:
            summary = (f"registers strongly via {best['feature']} "
                       f"(clean={best['clean_prob']:.2f} -> forged={best['forged_prob']:.2f}) "
                       "-- likely a forging-pipeline bug")
        elif best["lift"] > 0.05:
            summary = (f"registers only weakly (best: {best['feature']}, lift={best['lift']:.3f}) "
                       "-- ambiguous")
        else:
            summary = ("DOES NOT REGISTER on any feature -- the forging pipeline may be "
                       "broken, OR even own classifier can't be fooled, ")
        print(f"  {category}: {summary}")


if __name__ == "__main__":
    main()
