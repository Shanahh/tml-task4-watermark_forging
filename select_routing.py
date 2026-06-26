#!/usr/bin/env python3
"""Build routing.json for build_submission.py, automatically falling back to
the mean-residual baseline for any surrogate+PGD category whose transfer
check (check_surrogate_transfer.py) did not come back "likely to transfer".

Categories with a validated hand-crafted attack (WM_1/3/4/5/6 by default) are
always routed to the specialized candidates -- the transfer check only
applies to the surrogate-driven categories, since the hand-crafted attacks
don't rely on a black-box proxy model at all. WM_3 has a surrogate+PGD path
too (kept for ablation comparison, since it was the original mechanism used
for it), but defaults to the hand-crafted attack here since that sidesteps
the transferability question entirely and WM_3's own diagnostics are strong
enough (Y/Cb/Cr_auc all ~0.97-0.99) to justify it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import CATEGORIES, write_json

DEFAULT_SPECIALIZED_CATEGORIES = ["WM_1", "WM_3", "WM_4", "WM_5", "WM_6"]
DEFAULT_SURROGATE_CATEGORIES = ["WM_2", "WM_7", "WM_8"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--specialized-dir", type=Path, required=True)
    p.add_argument("--specialized-strength", default="0.005")
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--baseline-strength", default="0.02")
    p.add_argument("--pgd-dir", type=Path, required=True)
    p.add_argument("--pgd-eps", default="0.007843")
    p.add_argument("--transfer-checks-dir", type=Path, required=True)
    p.add_argument("--specialized-categories", default=",".join(DEFAULT_SPECIALIZED_CATEGORIES))
    p.add_argument("--surrogate-categories", default=",".join(DEFAULT_SURROGATE_CATEGORIES))
    p.add_argument("--output", type=Path, default=Path("routing.json"))
    return p.parse_args()


def load_verdict(transfer_checks_dir, category):
    path = transfer_checks_dir / f"transfer_check_{category.lower()}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("verdict", "")


def main():
    args = parse_args()
    specialized_categories = [c.strip() for c in args.specialized_categories.split(",") if c.strip()]
    surrogate_categories = [c.strip() for c in args.surrogate_categories.split(",") if c.strip()]

    specialized_path = args.specialized_dir / f"strength_{args.specialized_strength}"
    baseline_path = args.baseline_dir / f"strength_{args.baseline_strength}"

    routing = {}
    for category in CATEGORIES:
        if category in specialized_categories:
            routing[category] = str(specialized_path)
            continue

        if category in surrogate_categories:
            verdict = load_verdict(args.transfer_checks_dir, category)
            if verdict is None:
                print(f"{category}: no transfer check found -- falling back to baseline (be cautious)")
                routing[category] = str(baseline_path)
            elif verdict.startswith("likely to transfer"):
                pgd_path = args.pgd_dir / category.lower() / f"eps_{args.pgd_eps}"
                print(f"{category}: transfer check passed -- routing to {pgd_path}")
                routing[category] = str(pgd_path)
            else:
                print(f"{category}: transfer check failed ({verdict!r}) -- falling back to baseline")
                routing[category] = str(baseline_path)
            continue

        # Any category not explicitly classified falls back to the baseline.
        routing[category] = str(baseline_path)

    write_json(args.output, routing)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
