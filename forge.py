#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import torch
from tqdm import tqdm

from config import CATEGORY_RANGES, CATEGORY_RESOLUTIONS
from data import load_rgb, verify_dataset_root
from model_utils import load_img2img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=Path("."))
    p.add_argument("--run-dir", type=Path, default=Path("wm_lora_runs"))
    p.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--adapter-scale", type=float, default=0.75)
    p.add_argument(
        "--strength",
        type=float,
        default=0.10,
        help="Img2img noise strength. Lower preserves image quality better.",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", type=Path, default=Path("submission_temp"))
    p.add_argument("--submission", type=Path, default=Path("submission.zip"))
    return p.parse_args()


def package_submission(output_dir: Path, destination: Path) -> None:
    paths = sorted(output_dir.glob("*.png"), key=lambda p: int(p.stem))
    expected = [f"{i}.png" for i in range(1, 201)]
    actual = [path.name for path in paths]

    if actual != expected:
        raise RuntimeError(
            f"Invalid output set. Missing={sorted(set(expected)-set(actual))}; "
            f"extra={sorted(set(actual)-set(expected))}"
        )

    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.run_dir = args.run_dir.resolve()
    verify_dataset_root(args.dataset)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if args.steps * args.strength < 1.0:
        raise ValueError(
            "steps * strength must be at least 1, otherwise img2img may run "
            "zero denoising steps."
        )

    device = torch.device(args.device)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    for category, (start, stop) in CATEGORY_RANGES.items():
        adapter_dir = args.run_dir / "adapters" / category / "final"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"Missing adapter: {adapter_dir}. Run train.py first."
            )

        pipe = load_img2img(
            args.model_id,
            adapter_dir,
            device,
            args.precision,
            args.adapter_scale,
        )
        resolution = CATEGORY_RESOLUTIONS[category]

        for number in tqdm(range(start, stop + 1), desc=f"forge {category}"):
            path = args.dataset / "clean_targets" / f"{number}.png"
            image = load_rgb(path)

            if image.size != (resolution, resolution):
                raise RuntimeError(
                    f"{path} is {image.size}, expected {resolution}x{resolution}"
                )

            generator = torch.Generator(device=device).manual_seed(
                args.seed + number
            )
            forged = pipe(
                prompt="",
                image=image,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=generator,
            ).images[0]

            if forged.size != image.size:
                forged = forged.resize(image.size)

            forged.save(args.output_dir / f"{number}.png", compress_level=4)

        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    package_submission(args.output_dir, args.submission)
    args.submission.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "adapter_scale": args.adapter_scale,
                "strength": args.strength,
                "steps": args.steps,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"Created {args.submission}")


if __name__ == "__main__":
    main()
