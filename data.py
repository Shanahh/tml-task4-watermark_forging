from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse numeric suffix from {path.name}")
    return int(match.group(1))


def sorted_pngs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=numeric_suffix)


def verify_dataset_root(root: Path) -> None:
    required = [root / "clean_targets", root / "watermarked_sources"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing dataset directories: {missing}")


def verify_resolution(paths: list[Path], resolution: int) -> None:
    if not paths:
        raise RuntimeError("No PNG images found.")

    wrong = []
    for path in paths:
        with Image.open(path) as image:
            if image.size != (resolution, resolution):
                wrong.append((str(path), image.size))

    if wrong:
        raise RuntimeError(
            f"Expected {resolution}x{resolution}; examples of mismatches: {wrong[:5]}"
        )


class WatermarkedDataset(Dataset):
    """
    Full-image dataset with no geometric augmentation.

    This is deliberate: watermark signals may be tied to exact coordinates or
    frequency bins, so random crops/flips can destroy the signal being learned.
    """

    def __init__(self, source_dir: Path, resolution: int, repeats: int = 200):
        self.paths = sorted_pngs(source_dir)
        verify_resolution(self.paths, resolution)
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index % len(self.paths)]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = pil_to_tensor(image).float() / 255.0

        # Stable Diffusion VAE input range.
        return tensor * 2.0 - 1.0


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()
