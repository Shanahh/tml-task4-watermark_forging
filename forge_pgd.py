#!/usr/bin/env python3
"""Surrogate-classifier + constrained PGD forging attack.

Used for watermark categories with no validated hand-crafted statistical
signal (WM_2, WM_3, WM_7, WM_8): a small ensemble of residual classifiers
(trained by train_surrogate.py) stands in for the unknown real detector, and
each clean target is perturbed within an L_inf budget to maximize the
ensemble's "watermarked" logit while staying perceptually close to the
original.

--archs accepts a comma-separated list (e.g. "cnn_a,cnn_b") to attack several
structurally different architectures simultaneously. This is the standard
transferability trick: a perturbation that fools a diverse set of models at
once tends to generalize to an unseen model far better than one optimized
against a single architecture. See check_surrogate_transfer.py to validate
this against a held-out architecture not included in --archs.

The quality term optimizes real LPIPS directly (Sqlt = exp(-8*LPIPS) is the
actual scoring function), falling back to an MSE+TV proxy if the `lpips`
package is not installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import CATEGORIES, CATEGORY_RANGES, load_dataset, save_rgb
from train_surrogate import ARCHITECTURES, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument(
        "--archs",
        default="cnn_a",
        help="comma-separated architectures to attack simultaneously, e.g. cnn_a,cnn_b "
        f"(available: {','.join(ARCHITECTURES)})",
    )
    p.add_argument("--output-dir", type=Path, default=Path("pgd_candidates"))
    p.add_argument("--eps-grid", default="0.0039215686,0.0078431373,0.011764706")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--step-size", type=float, default=0.0009803922)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--lpips-weight", type=float, default=10.0)
    p.add_argument("--mse-weight", type=float, default=10.0)
    p.add_argument("--tv-weight", type=float, default=0.05)
    return p.parse_args()


def high_pass_torch(x):
    kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=x.dtype, device=x.device)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(x.shape[1], 1, 5, 5)
    return x - F.conv2d(x, kernel, padding=2, groups=x.shape[1])


def total_variation(d):
    return torch.mean(torch.abs(d[:, :, 1:] - d[:, :, :-1])) + torch.mean(
        torch.abs(d[:, :, :, 1:] - d[:, :, :, :-1])
    )


def load_models(model_dir, arch, device):
    models = []
    for path in sorted(model_dir.glob("detector_*.pt")):
        # weights_only=False: these checkpoints are written by train_surrogate.py
        # in this repo, never loaded from an untrusted source.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(arch).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models.append(model)
    if not models:
        raise FileNotFoundError(model_dir)
    return models


def load_ensemble(models_root, category, archs, device):
    """Load and concatenate the detector ensembles for every architecture in
    `archs` (e.g. ["cnn_a", "cnn_b"]) into a single flat list of models, all
    of which optimize() will average logits over."""
    models = []
    for arch in archs:
        models += load_models(models_root / category.lower() / arch, arch, device)
    return models


def load_quality_loss(args, device):
    try:
        import lpips

        net = lpips.LPIPS(net=args.lpips_net).to(device).eval()
        for param in net.parameters():
            param.requires_grad_(False)

        def quality(x, original):
            return args.lpips_weight * net(x, original, normalize=True).mean()

        print(f"using LPIPS({args.lpips_net}) quality loss, weight={args.lpips_weight}")
        return quality
    except ImportError:
        print(
            "lpips package not found, falling back to MSE+TV quality proxy "
            "(pip install lpips for the real metric)"
        )

        def quality(x, original):
            d = x - original
            return args.mse_weight * torch.mean(d * d) + args.tv_weight * total_variation(d)

        return quality


def optimize(image, models, eps, args, device, quality):
    original = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).float().to(device)
    x = original.clone().detach().requires_grad_(True)

    for _ in range(args.steps):
        logit = torch.stack([m(high_pass_torch(x)) for m in models]).mean()
        loss = F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
        loss = loss + quality(x, original)
        grad = torch.autograd.grad(loss, x)[0]
        x = (x.detach() - args.step_size * grad.sign())
        x = torch.max(torch.min(x, original + eps), original - eps).clamp(0, 1)
        x.requires_grad_(True)

    return np.transpose(x.detach().cpu().numpy()[0], (1, 2, 0))


def main():
    args = parse_args()
    archs = [a.strip() for a in args.archs.split(",")]
    for arch in archs:
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch!r}, available: {list(ARCHITECTURES)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(args.models, args.category, archs, device)
    print(f"attacking ensemble of {len(models)} models across architectures {archs}")
    _, clean = load_dataset(args.dataset)
    lo, hi = CATEGORY_RANGES[args.category]
    quality = load_quality_loss(args, device)

    for eps in [float(v) for v in args.eps_grid.split(",")]:
        out_dir = args.output_dir / args.category.lower() / f"eps_{eps:.6f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, x in clean.items():
            if lo <= i <= hi:
                y = optimize(x, models, eps, args, device, quality)
            else:
                y = x
            save_rgb(y, out_dir / f"{i}.png")

        print("saved", out_dir)


if __name__ == "__main__":
    main()
