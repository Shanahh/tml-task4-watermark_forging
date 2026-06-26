#!/usr/bin/env python3
"""Train an ensemble of watermarked-vs-clean residual classifiers for one
watermark category. Used as the surrogate detector that forge_pgd.py attacks
with constrained PGD for categories with no validated hand-crafted signal
(WM_2, WM_3, WM_7, WM_8).

Two distinct architectures (--arch cnn_a / cnn_b) are provided so that an
independent, structurally different ensemble can be trained for the same
category and used purely to sanity-check transfer (see
check_surrogate_transfer.py) before spending a real submission on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import CATEGORIES, high_pass, load_dataset


class ResidualDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = np.transpose(high_pass(self.images[i]), (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.float32)


class ResidualCNN(nn.Module):
    """Default architecture: 5 conv layers, two stride-2 downsamples."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class ResidualCNNAlt(nn.Module):
    """Structurally different architecture (larger kernels, one downsample,
    different channel widths) used to build an independent surrogate for
    transfer sanity-checking, not as the primary attack model."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


ARCHITECTURES = {"cnn_a": ResidualCNN, "cnn_b": ResidualCNNAlt}


def build_model(arch):
    return ARCHITECTURES[arch]()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--output-dir", type=Path, default=Path("surrogates"))
    p.add_argument("--arch", default="cnn_a", choices=list(ARCHITECTURES))
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parse_args()
    # Nested by architecture so independent ensembles (e.g. cnn_a vs cnn_b)
    # for the same category never overwrite each other's checkpoints.
    out_dir = args.output_dir / args.category.lower() / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    src, clean = load_dataset(args.dataset)
    pos = src[args.category]
    neg = [x for x in clean.values() if x.shape[:2] == pos[0].shape[:2]]
    images = pos + neg
    labels = np.r_[np.ones(len(pos), np.float32), np.zeros(len(neg), np.float32)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = []

    for member in range(args.ensemble_size):
        seed = args.seed + member
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

        model = build_model(args.arch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        best_auc, best_state = -1, None

        for epoch in range(args.epochs):
            model.train()
            train_loader = DataLoader(
                ResidualDataset([images[i] for i in train_idx], labels[train_idx]),
                batch_size=args.batch_size,
                shuffle=True,
            )
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_labels, val_probs = [], []
            val_loader = DataLoader(
                ResidualDataset([images[i] for i in val_idx], labels[val_idx]),
                batch_size=args.batch_size,
            )
            with torch.no_grad():
                for x, y in val_loader:
                    val_probs += torch.sigmoid(model(x.to(device))).cpu().tolist()
                    val_labels += y.tolist()

            auc = float(roc_auc_score(val_labels, val_probs))
            print(args.category, args.arch, member, epoch + 1, auc)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        path = out_dir / f"detector_{member}.pt"
        torch.save(
            {"state_dict": best_state, "val_auc": best_auc, "seed": seed,
             "category": args.category, "arch": args.arch},
            path,
        )
        metadata.append({"path": str(path), "val_auc": best_auc, "seed": seed, "arch": args.arch})

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
