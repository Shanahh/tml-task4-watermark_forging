from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from utils import load_rgb, save_rgb, robust_template, texture_mask


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


def forge(
    target: np.ndarray,
    template: np.ndarray,
    strength: float,
    max_delta_255: float,
) -> np.ndarray:
    if target.shape != template.shape:
        raise ValueError(
            f"Target shape {target.shape} differs from source/template shape {template.shape}."
        )

    delta = strength * template * texture_mask(target)

    max_delta = max_delta_255 / 255.0
    delta = np.clip(delta, -max_delta, max_delta)

    headroom = np.minimum(target, 1.0 - target)
    attenuation = np.clip(headroom / (2.0 * max_delta + 1e-8), 0.25, 1.0)
    delta *= attenuation

    return np.clip(target + delta, 0.0, 1.0)


def extract_dataset(zip_path: Path, dataset_dir: Path) -> None:
    if dataset_dir.exists():
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(zip_path.parent)


def locate_dataset(dataset_dir: Path) -> Path:
    if (dataset_dir / "watermarked_sources").is_dir():
        return dataset_dir

    matches = [
        path.parent
        for path in dataset_dir.rglob("watermarked_sources")
        if path.is_dir()
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not locate watermarked_sources below {dataset_dir}"
    )


def package_flat_pngs(output_dir: Path, submission_path: Path) -> None:
    images = sorted(output_dir.glob("*.png"), key=lambda p: int(p.stem))
    expected = [f"{i}.png" for i in range(1, 201)]
    actual = [p.name for p in images]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"Invalid output set. Missing={missing}, extra={extra}")

    with zipfile.ZipFile(
        submission_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in images:
            archive.write(path, arcname=path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("Dataset.zip"))
    parser.add_argument("--dataset", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("submission_temp"))
    parser.add_argument("--submission", type=Path, default=Path("submission.zip"))
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Global watermark scale. Recommended sweep: 0.7, 1.0, 1.3, 1.6.",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=10.0,
        help="Maximum per-channel pixel change on the 0..255 scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_dataset(args.zip, args.dataset)
    dataset_root = locate_dataset(args.dataset)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    target_dir = dataset_root / "clean_targets"
    total = 0

    for source_name, target_start, target_stop in CATEGORIES:
        source_dir = dataset_root / "watermarked_sources" / source_name
        source_paths = sorted(source_dir.glob("*.png"))
        if len(source_paths) != 25:
            raise RuntimeError(
                f"Expected 25 images in {source_dir}, found {len(source_paths)}"
            )

        print(f"Estimating shared pattern for {source_name}...")
        sources = [load_rgb(path) for path in source_paths]
        template = robust_template(sources)

        diagnostic = template / (2.0 * np.max(np.abs(template)) + 1e-8) + 0.5
        save_rgb(args.output_dir / f"_template_{source_name}.png", diagnostic)

        for number in range(target_start, target_stop + 1):
            target_path = target_dir / f"{number}.png"
            if not target_path.exists():
                raise FileNotFoundError(target_path)

            target = load_rgb(target_path)
            forged = forge(target, template, args.strength, args.max_delta)
            save_rgb(args.output_dir / target_path.name, forged)
            total += 1

    for path in args.output_dir.glob("_template_*.png"):
        path.unlink()

    package_flat_pngs(args.output_dir, args.submission)
    print(f"Created {args.submission} with {total} forged images.")


if __name__ == "__main__":
    main()
