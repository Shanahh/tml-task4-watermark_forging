from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import CATEGORIES, CATEGORY_RANGES, load_dataset, write_json
from forge_pgd import ARCHITECTURES, high_pass_torch, load_ensemble, load_quality_loss, optimize

COLLAPSE_STD_THRESHOLD = 0.02
TRANSFER_RATIO_THRESHOLD = 0.3
GENUINE_FLIP_RATE_THRESHOLD = 0.3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--attack-models", type=Path, required=True)
    p.add_argument(
        "--attack-archs",
        default="cnn_a",
        help="comma-separated architectures to attack simultaneously, e.g. cnn_a,cnn_b",
    )
    p.add_argument("--holdout-models", type=Path, required=True)
    p.add_argument("--holdout-arch", default="cnn_b", choices=list(ARCHITECTURES))
    p.add_argument("--output", type=Path, default=Path("transfer_check.json"))
    p.add_argument("--eps", type=float, default=0.0078431373)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--step-size", type=float, default=0.0009803922)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--lpips-weight", type=float, default=10.0)
    p.add_argument("--mse-weight", type=float, default=10.0)
    p.add_argument("--tv-weight", type=float, default=0.05)
    return p.parse_args()


def predict_prob(models, image, device):
    x = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logit = torch.stack([m(high_pass_torch(x)) for m in models]).mean()
    return float(torch.sigmoid(logit))


def main():
    args = parse_args()
    attack_archs = [a.strip() for a in args.attack_archs.split(",")]
    for arch in attack_archs:
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch!r}, available: {list(ARCHITECTURES)}")
    if args.holdout_arch in attack_archs:
        print(
            f"WARNING: holdout arch {args.holdout_arch!r} is also one of the attack archs "
            f"{attack_archs} -- this holdout is not independent and the result is meaningless."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attack_models = load_ensemble(args.attack_models, args.category, attack_archs, device)
    holdout_models = load_ensemble(args.holdout_models, args.category, [args.holdout_arch], device)

    _, clean = load_dataset(args.dataset)
    lo, hi = CATEGORY_RANGES[args.category]
    quality = load_quality_loss(args, device)

    rows = []
    for i, x in clean.items():
        if not (lo <= i <= hi):
            continue
        forged = optimize(x, attack_models, args.eps, args, device, quality)
        rows.append({
            "id": i,
            "clean_attack_prob": predict_prob(attack_models, x, device),
            "forged_attack_prob": predict_prob(attack_models, forged, device),
            "clean_holdout_prob": predict_prob(holdout_models, x, device),
            "forged_holdout_prob": predict_prob(holdout_models, forged, device),
        })

    attack_lift = float(np.mean([r["forged_attack_prob"] - r["clean_attack_prob"] for r in rows]))
    holdout_lift = float(np.mean([r["forged_holdout_prob"] - r["clean_holdout_prob"] for r in rows]))
    transfer_ratio = float(holdout_lift / attack_lift) if attack_lift > 1e-6 else 0.0

    genuine_flip_rate = float(np.mean([
        r["clean_holdout_prob"] < 0.5 and r["forged_holdout_prob"] >= 0.5 for r in rows
    ]))

    holdout_baseline_std = float(np.std([r["clean_holdout_prob"] for r in rows]))
    holdout_uninformative = holdout_baseline_std < COLLAPSE_STD_THRESHOLD

    if holdout_uninformative:
        verdict = (
            "inconclusive -- holdout model collapsed to a near-constant output "
            "regardless of input, it cannot judge transfer either way"
        )
    elif transfer_ratio >= TRANSFER_RATIO_THRESHOLD and genuine_flip_rate >= GENUINE_FLIP_RATE_THRESHOLD:
        verdict = "likely to transfer"
    elif transfer_ratio > 0:
        verdict = (
            "weak or partial transfer -- some signal carries over to the holdout model, "
            "but not enough to cross its decision boundary at this eps/step budget"
        )
    else:
        verdict = "unlikely to transfer -- attack probably overfits the attack ensemble"

    summary = {
        "category": args.category,
        "attack_archs": attack_archs,
        "holdout_arch": args.holdout_arch,
        "eps": args.eps,
        "n_images": len(rows),
        "attack_ensemble_mean_lift": attack_lift,
        "holdout_ensemble_mean_lift": holdout_lift,
        "holdout_baseline_std": holdout_baseline_std,
        "transfer_ratio": transfer_ratio,
        "holdout_genuine_flip_rate": genuine_flip_rate,
        "verdict": verdict,
        "rows": rows,
    }

    write_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
