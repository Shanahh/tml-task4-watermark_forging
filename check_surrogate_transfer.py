#!/usr/bin/env python3
"""Sanity-check whether a surrogate-driven PGD attack is likely to transfer to
the real, unseen watermark detector before spending a submission on it.

The idea: craft PGD perturbations against one surrogate ensemble ("attack"),
then measure whether a *second*, independently trained ensemble (different
architecture and/or seed, never used in the attack) also flips its prediction
on the same images. If your own two surrogates don't agree, the perturbation
is very unlikely to transfer to the real hidden detector either -- it is
overfit to whatever the attack ensemble happened to learn (which, per the
validated diagnostics, may be image content/provenance rather than the
watermark itself for categories like WM_2/7/8).

Typical usage:

    python train_surrogate.py --dataset DATA --category WM_2 --arch cnn_a \
        --output-dir surrogates
    python train_surrogate.py --dataset DATA --category WM_2 --arch cnn_b \
        --output-dir surrogates

    python check_surrogate_transfer.py --dataset DATA --category WM_2 \
        --attack-models surrogates --attack-arch cnn_a \
        --holdout-models surrogates --holdout-arch cnn_b \
        --eps 0.0078431373
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import CATEGORIES, CATEGORY_RANGES, load_dataset, write_json
from forge_pgd import high_pass_torch, load_models, load_quality_loss, optimize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--attack-models", type=Path, required=True)
    p.add_argument("--attack-arch", default="cnn_a")
    p.add_argument("--holdout-models", type=Path, required=True)
    p.add_argument("--holdout-arch", default="cnn_b")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attack_models = load_models(
        args.attack_models / args.category.lower() / args.attack_arch, args.attack_arch, device
    )
    holdout_models = load_models(
        args.holdout_models / args.category.lower() / args.holdout_arch, args.holdout_arch, device
    )

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

    attack_lift = np.mean([r["forged_attack_prob"] - r["clean_attack_prob"] for r in rows])
    holdout_lift = np.mean([r["forged_holdout_prob"] - r["clean_holdout_prob"] for r in rows])
    transfer_rate = float(np.mean([r["forged_holdout_prob"] >= 0.5 for r in rows]))
    transfer_ratio = float(holdout_lift / attack_lift) if attack_lift > 1e-6 else 0.0

    summary = {
        "category": args.category,
        "eps": args.eps,
        "n_images": len(rows),
        "attack_ensemble_mean_lift": float(attack_lift),
        "holdout_ensemble_mean_lift": float(holdout_lift),
        "transfer_ratio": transfer_ratio,
        "holdout_flip_rate": transfer_rate,
        "verdict": (
            "likely to transfer" if transfer_ratio > 0.3 and transfer_rate > 0.5
            else "unlikely to transfer -- attack probably overfits the attack ensemble"
        ),
        "rows": rows,
    }

    write_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
