#!/usr/bin/env python3
"""Sanity-check a submission zip: for each category, report the mean/max
pixel difference against the matching clean target. A category showing
~0 diff almost certainly means it silently fell back to the clean image
(e.g. a stale routing.json pointing at a path that no longer exists)."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from common import CATEGORIES, CATEGORY_RANGES, load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    _, clean = load_dataset(args.dataset)

    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        assert names == {f"{i}.png" for i in range(1, 201)}, "zip doesn't contain exactly 1.png..200.png"
        assert not any("/" in n for n in names), "zip has subfolders"

        for category in CATEGORIES:
            lo, hi = CATEGORY_RANGES[category]
            diffs = []
            for i in range(lo, hi + 1):
                with zf.open(f"{i}.png") as f:
                    submitted = np.asarray(Image.open(f).convert("RGB"), np.float32)
                clean_img = np.asarray(Image.fromarray((clean[i] * 255).astype(np.uint8)), np.float32)
                diffs.append(np.abs(submitted - clean_img))
            mean_diff = np.mean([d.mean() for d in diffs])
            max_diff = np.max([d.max() for d in diffs])
            flag = "  <-- LOOKS UNMODIFIED" if mean_diff < 0.05 else ""
            print(f"{category}: mean abs diff={mean_diff:.4f}  max abs diff={max_diff:.1f}{flag}")


if __name__ == "__main__":
    main()
