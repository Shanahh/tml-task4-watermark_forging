#!/usr/bin/env python3
"""
Inspect image dimensions and basic dataset statistics.

Expected structure:
    clean_targets/
    watermarked_sources/
        WM_1/
        ...
        WM_8/

Run:
    python inspect_dataset.py
    python inspect_dataset.py --dataset .
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("."),
        help="Directory containing clean_targets and watermarked_sources.",
    )
    return parser.parse_args()


def find_images(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path),
    )


def inspect_images(paths: list[Path]) -> tuple[list[dict], list[tuple[Path, str]]]:
    records: list[dict] = []
    errors: list[tuple[Path, str]] = []

    for path in paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
                records.append(
                    {
                        "path": path,
                        "width": width,
                        "height": height,
                        "mode": image.mode,
                        "format": image.format or "unknown",
                    }
                )
        except (OSError, UnidentifiedImageError) as exc:
            errors.append((path, str(exc)))

    return records, errors


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_records_summary(name: str, records: list[dict]) -> None:
    print_section(name)

    if not records:
        print("No readable images found.")
        return

    size_counts = Counter((r["width"], r["height"]) for r in records)
    mode_counts = Counter(r["mode"] for r in records)
    format_counts = Counter(r["format"] for r in records)

    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    pixels = [r["width"] * r["height"] for r in records]

    print(f"Images:              {len(records)}")
    print(f"Unique dimensions:   {len(size_counts)}")
    print(f"Width range:         {min(widths)} to {max(widths)}")
    print(f"Height range:        {min(heights)} to {max(heights)}")
    print(f"Pixel-count range:   {min(pixels):,} to {max(pixels):,}")

    print("\nDimension distribution:")
    for (width, height), count in sorted(
        size_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        percentage = 100.0 * count / len(records)
        orientation = (
            "square"
            if width == height
            else "landscape"
            if width > height
            else "portrait"
        )
        print(
            f"  {width:4d} x {height:<4d} "
            f"{count:4d} images "
            f"({percentage:6.2f}%)  {orientation}"
        )

    print("\nImage modes:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode:<10} {count}")

    print("\nFile formats:")
    for image_format, count in sorted(format_counts.items()):
        print(f"  {image_format:<10} {count}")


def print_group_statistics(
    records: list[dict],
    watermarked_root: Path,
) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        relative = record["path"].relative_to(watermarked_root)
        group = relative.parts[0] if relative.parts else "unknown"
        groups[group].append(record)

    print_section("Watermarked source statistics by category")

    for group in sorted(groups):
        group_records = groups[group]
        size_counts = Counter(
            (r["width"], r["height"]) for r in group_records
        )

        print(f"\n{group}: {len(group_records)} images")
        for (width, height), count in sorted(size_counts.items()):
            print(f"  {width:4d} x {height:<4d}: {count}")


def print_resolution_cross_table(
    clean_records: list[dict],
    source_records: list[dict],
) -> None:
    clean_sizes = Counter(
        (record["width"], record["height"]) for record in clean_records
    )
    source_sizes = Counter(
        (record["width"], record["height"]) for record in source_records
    )

    all_sizes = sorted(set(clean_sizes) | set(source_sizes))

    print_section("Clean targets versus watermarked sources")

    print(f"{'Dimensions':<16} {'Clean':>8} {'Sources':>10} {'Total':>8}")
    print("-" * 46)

    for width, height in all_sizes:
        clean_count = clean_sizes[(width, height)]
        source_count = source_sizes[(width, height)]
        print(
            f"{width}x{height:<10} "
            f"{clean_count:>8} "
            f"{source_count:>10} "
            f"{clean_count + source_count:>8}"
        )


def print_nonstandard_images(records: list[dict]) -> None:
    expected = {(128, 128), (256, 256), (512, 512)}
    unusual = [
        record
        for record in records
        if (record["width"], record["height"]) not in expected
    ]

    print_section("Images outside expected 128/256/512 square sizes")

    if not unusual:
        print("None.")
        return

    for record in unusual:
        print(
            f"{record['width']}x{record['height']}  "
            f"{record['mode']:<6}  {record['path']}"
        )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()

    clean_dir = dataset / "clean_targets"
    watermarked_dir = dataset / "watermarked_sources"

    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {clean_dir}")

    if not watermarked_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {watermarked_dir}")

    clean_paths = find_images(clean_dir)
    source_paths = find_images(watermarked_dir)

    clean_records, clean_errors = inspect_images(clean_paths)
    source_records, source_errors = inspect_images(source_paths)

    print(f"Dataset root: {dataset}")

    print_records_summary("Clean targets", clean_records)
    print_records_summary("Watermarked sources", source_records)
    print_group_statistics(source_records, watermarked_dir)
    print_resolution_cross_table(clean_records, source_records)
    print_nonstandard_images(clean_records + source_records)

    errors = clean_errors + source_errors
    print_section("Unreadable or corrupt images")

    if not errors:
        print("None.")
    else:
        for path, error in errors:
            print(f"{path}: {error}")

    print_section("Validation")

    print(f"Clean target count:       {len(clean_records)}")
    print(f"Watermarked source count: {len(source_records)}")

    if len(clean_records) != 200:
        print("WARNING: Expected 200 clean target images.")

    if len(source_records) != 200:
        print("WARNING: Expected 200 watermarked source images.")

    expected_groups = {f"WM_{index}" for index in range(1, 9)}
    actual_groups = {
        path.relative_to(watermarked_dir).parts[0]
        for path in source_paths
        if path.relative_to(watermarked_dir).parts
    }

    missing_groups = expected_groups - actual_groups
    extra_groups = actual_groups - expected_groups

    if missing_groups:
        print(f"WARNING: Missing groups: {sorted(missing_groups)}")

    if extra_groups:
        print(f"WARNING: Unexpected groups: {sorted(extra_groups)}")

    if not missing_groups and not extra_groups:
        print("All expected WM_1 through WM_8 groups are present.")


if __name__ == "__main__":
    main()
    from pathlib import Path
    from PIL import Image

    categories = [
        ("WM_1", 1, 25),
        ("WM_2", 26, 50),
        ("WM_3", 51, 75),
        ("WM_4", 76, 100),
        ("WM_5", 101, 125),
        ("WM_6", 126, 150),
        ("WM_7", 151, 175),
        ("WM_8", 176, 200),
    ]

    for name, start, stop in categories:
        sizes = {}
        for number in range(start, stop + 1):
            path = Path("./dataset/clean_targets") / f"{number}.png"
            with Image.open(path) as image:
                sizes[image.size] = sizes.get(image.size, 0) + 1

        print(name, sizes)