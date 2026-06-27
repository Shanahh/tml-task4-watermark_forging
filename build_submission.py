#!/usr/bin/env python3
"""Assemble the final 200-image submission zip from per-category candidate
directories, described by a routing JSON file mapping category -> directory.

Any category omitted from the routing file (or whose candidate file is
missing for a given id) falls back to the unmodified clean target.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from common import category_for_id, load_dataset, save_rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--routing",
        type=Path,
        required=True,
        help="JSON file mapping category (WM_1..WM_8) to a candidate directory "
        "containing <id>.png files",
    )
    p.add_argument("--output-dir", type=Path, default=Path("final_submission"))
    p.add_argument("--zip", type=Path, default=Path("final_submission.zip"))
    return p.parse_args()


def main():
    args = parse_args()
    _, clean = load_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    routing = {k: Path(v) for k, v in json.loads(args.routing.read_text()).items()}

    for i, x in clean.items():
        category = category_for_id(i)
        candidate_dir = routing.get(category)
        src = candidate_dir / f"{i}.png" if candidate_dir else None
        dst = args.output_dir / f"{i}.png"
        if src and src.exists():
            shutil.copy2(src, dst)
        else:
            save_rgb(x, dst)

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 201):
            zf.write(args.output_dir / f"{i}.png", arcname=f"{i}.png")

    print("saved", args.zip)


if __name__ == "__main__":
    main()
