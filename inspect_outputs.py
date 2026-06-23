#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", type=Path, default=Path("clean_targets"))
    p.add_argument("--forged-dir", type=Path, default=Path("submission_temp"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mse_values, mae_values, linf_values = [], [], []

    for number in range(1, 201):
        clean = np.asarray(
            Image.open(args.clean_dir / f"{number}.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        forged = np.asarray(
            Image.open(args.forged_dir / f"{number}.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0

        if clean.shape != forged.shape:
            raise RuntimeError(f"Shape mismatch for {number}.png")

        delta = forged - clean
        mse_values.append(float(np.mean(delta ** 2)))
        mae_values.append(float(np.mean(np.abs(delta))))
        linf_values.append(float(np.max(np.abs(delta))))

    for name, values in [
        ("MSE", mse_values),
        ("MAE", mae_values),
        ("L_inf", linf_values),
    ]:
        array = np.asarray(values)
        print(
            f"{name}: mean={array.mean():.8f}, "
            f"median={np.median(array):.8f}, max={array.max():.8f}"
        )


if __name__ == "__main__":
    main()
