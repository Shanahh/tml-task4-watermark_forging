from __future__ import annotations

import argparse
import json
import shutil

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch

from utils import (
    locate_dataset,
    load_preference_model,
    build_lpips,
    np_to_tensor,
    load_np,
    save_png,
    extract_or_load,
    build_bank,
    plant,
    rank_candidates,
    refine,
    package,
    natural_key,
    parse_float_list,
)

from utils import Candidate



CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

MODEL_SIZE = 768


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=Path("."))
    p.add_argument("--wmforger-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", type=Path, default=Path("wmforger_cache"))
    p.add_argument("--output-dir", type=Path, default=Path("submission_temp"))
    p.add_argument("--submission", type=Path, default=Path("submission.zip"))
    p.add_argument("--force-extract", action="store_true")

    p.add_argument("--extract-steps", type=int, default=75)
    p.add_argument("--extract-lr", type=float, default=0.05)
    p.add_argument(
        "--strengths",
        type=parse_float_list,
        default=[0.65, 0.85, 1.0, 1.2, 1.45, 1.75],
    )
    p.add_argument(
        "--epsilons",
        type=parse_float_list,
        default=[6.0, 8.0, 10.0, 12.0],
    )
    p.add_argument("--keep-individuals", type=int, default=5)
    p.add_argument("--max-candidates", type=int, default=160)

    p.add_argument("--rank-lpips-weight", type=float, default=2.0)
    p.add_argument("--rank-mse-weight", type=float, default=10.0)

    p.add_argument("--refine-steps", type=int, default=12)
    p.add_argument("--refine-lr", type=float, default=0.5)
    p.add_argument("--refine-epsilon", type=float, default=12.0)
    p.add_argument("--refine-lpips-weight", type=float, default=1.5)
    p.add_argument("--refine-mse-weight", type=float, default=8.0)
    p.add_argument("--refine-anchor-weight", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available.")

    device = torch.device(args.device)
    root = locate_dataset(args.dataset)
    model = load_preference_model(
        args.wmforger_root, args.checkpoint, device
    )
    lpips_model = build_lpips(device)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    diagnostics = {}
    target_dir = root / "clean_targets"

    for category, start, stop in CATEGORIES:
        source_paths = sorted(
            (root / "watermarked_sources" / category).glob("*.png"),
            key=natural_key,
        )
        if len(source_paths) != 25:
            raise RuntimeError(
                f"Expected 25 sources in {category}, found {len(source_paths)}"
            )

        residuals = extract_or_load(
            category,
            source_paths,
            args.cache_dir,
            model,
            device,
            args.extract_steps,
            args.extract_lr,
            args.force_extract,
        )
        bank = build_bank(residuals, args.keep_individuals)
        diagnostics[category] = {"templates": list(bank), "targets": {}}

        for number in tqdm(range(start, stop + 1), desc=f"forge {category}"):
            target_path = target_dir / f"{number}.png"
            clean = np_to_tensor(load_np(target_path), device)

            candidates = []
            for template_name, residual in bank.items():
                for eps in args.epsilons:
                    for strength in args.strengths:
                        for masked in (False, True):
                            forged, delta = plant(
                                clean, residual, strength, eps, masked
                            )
                            name = (
                                f"{template_name}_s{strength:g}_e{eps:g}_"
                                f"{'tex' if masked else 'plain'}"
                            )
                            candidates.append(Candidate(name, forged, delta))

            if len(candidates) > args.max_candidates:
                idx = np.linspace(
                    0, len(candidates) - 1, args.max_candidates, dtype=int
                )
                candidates = [candidates[i] for i in idx]

            ranked = rank_candidates(
                clean,
                candidates,
                model,
                lpips_model,
                args.rank_lpips_weight,
                args.rank_mse_weight,
            )
            selected = ranked[0]

            final = refine(
                clean,
                selected,
                model,
                lpips_model,
                args.refine_steps,
                args.refine_lr,
                args.refine_epsilon,
                args.refine_lpips_weight,
                args.refine_mse_weight,
                args.refine_anchor_weight,
            )
            save_png(args.output_dir / f"{number}.png", final)

            diagnostics[category]["targets"][str(number)] = {
                "selected": selected.name,
                "score": selected.score,
                "preference_drop": selected.preference_drop,
                "lpips": selected.lpips_value,
                "mse": selected.mse_value,
                "top5": [
                    {
                        "name": c.name,
                        "score": c.score,
                        "preference_drop": c.preference_drop,
                        "lpips": c.lpips_value,
                        "mse": c.mse_value,
                    }
                    for c in ranked[:5]
                ],
            }

            del candidates, ranked, selected, final, clean
            if device.type == "cuda":
                torch.cuda.empty_cache()

    package(args.output_dir, args.submission)
    diagnostics_path = args.submission.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    print(f"Created {args.submission}")
    print(f"Created {diagnostics_path}")


if __name__ == "__main__":
    main()
