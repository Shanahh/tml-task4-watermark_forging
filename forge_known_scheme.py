#!/usr/bin/env python3
"""Forge a group using the REAL encoder of a known, identified open-source
watermarking scheme, instead of any statistical delta estimate.

Use identify_scheme.py first to find candidates: it flags a (library, method,
length) combo as a possible match only when decoding all 25 sources of a
group agrees on a BALANCED, non-degenerate bit string (not near-all-0s/1s,
which is a decode-bias artifact) with high consistency and a clear gap over
a clean-image control. Confirmed so far: WM_2 decodes consistently under
imwatermark's rivaGan method to the 32-bit message
"00010000101111110011101011101000" (22/25 sources exact, rest off by <=6
bits; round-trip encode-then-decode on clean targets recovers the message
with 100% accuracy at ~0.013 mean LPIPS locally).

Also confirmed: WM_7 decodes consistently under TrustMark (model_type='Q',
use_ECC=False, MODE='binary') to the 100-bit message
"1000100100110010100110011110110010010010111110000111011111111101011001111101011011010001010011100001"
(agreement 1.000 vs 0.612 control, balanced). Round-trip verified: encoding
this message with TrustMark's own encoder onto WM_7's clean targets recovers
it with 100% bit accuracy at ~0.0016 mean LPIPS locally (Sqlt~0.987) -- even
better than WM_2's RivaGAN result.

This script is deliberately OPT-IN and EXPLICIT per category: you must name
the scheme and paste the exact recovered message yourself after inspecting
identify_scheme.py's output, rather than have this auto-trust a detection.
Any category not given a scheme falls back to an existing candidate
directory (e.g. simple_candidates/) unchanged -- so this composes with
everything already tuned there instead of replacing it.

Usage:
    python forge_known_scheme.py --dataset dataset --base-dir simple_candidates \
        --output-dir known_scheme_candidates \
        --wm2-scheme rivaGan --wm2-message 00010000101111110011101011101000 \
        --wm7-scheme trustmark --wm7-message 1000100100110010100110011110110010010010111110000111011111111101011001111101011011010001010011100001
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from common import CATEGORIES, CATEGORY_RANGES, load_dataset, save_rgb

IMWATERMARK_METHODS = ("dwtDct", "dwtDctSvd", "rivaGan")
BLIND_WATERMARK_METHOD = "blind_watermark"
TRUSTMARK_METHOD = "trustmark"
SCHEME_CHOICES = ("none",) + IMWATERMARK_METHODS + (BLIND_WATERMARK_METHOD, TRUSTMARK_METHOD)


def arg_name(category, suffix):
    return f"{category.lower().replace('_', '')}_{suffix}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("known_scheme_candidates"))
    p.add_argument("--base-dir", type=Path, default=None,
                    help="existing candidate directory (flat <id>.png) used for any category "
                    "without a --wmN-scheme override; falls back to the unmodified clean target "
                    "if not given")
    for category in CATEGORIES:
        stem = category.lower().replace("_", "")
        p.add_argument(f"--{stem}-scheme", choices=SCHEME_CHOICES, default="none")
        p.add_argument(f"--{stem}-message", default=None,
                        help="0/1 bit string to embed, e.g. 00010000101111110011101011101000 "
                        "(imwatermark methods) or any string (blind_watermark, encoded as UTF-8 bits)")
        # blind_watermark hyperparameters
        p.add_argument(f"--{stem}-password-wm", type=int, default=1)
        p.add_argument(f"--{stem}-password-img", type=int, default=1)
        p.add_argument(f"--{stem}-block-shape", default="4,4")
        # trustmark hyperparameters
        p.add_argument(f"--{stem}-trustmark-model-type", default="Q", choices=["C", "Q", "B", "P"])
    return p.parse_args()


def bits_from_string(s):
    return [int(ch) for ch in s.strip()]


def forge_imwatermark(clean_targets, method, bits):
    from imwatermark import WatermarkEncoder

    if method == "rivaGan":
        WatermarkEncoder.loadModel()
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", bits)

    forged = {}
    for i, x in clean_targets.items():
        bgr = np.clip(x * 255, 0, 255).round().astype(np.uint8)[..., ::-1].copy()
        encoded_bgr = encoder.encode(bgr, method=method)
        forged[i] = encoded_bgr[..., ::-1].astype(np.float32) / 255.0
    return forged


def forge_blind_watermark(clean_targets, message, password_wm, password_img, block_shape):
    from blind_watermark import WaterMark

    forged = {}
    for i, x in clean_targets.items():
        bgr = np.clip(x * 255, 0, 255).round().astype(np.uint8)[..., ::-1].copy()
        wm = WaterMark(password_wm=password_wm, password_img=password_img, block_shape=block_shape)
        wm.read_img(img=bgr)
        wm.read_wm(message, mode="str")
        encoded_bgr = wm.embed()
        forged[i] = encoded_bgr[..., ::-1].astype(np.float32) / 255.0
    return forged


def forge_trustmark(clean_targets, message, model_type):
    from trustmark import TrustMark
    from PIL import Image

    tm = TrustMark(verbose=False, model_type=model_type, use_ECC=False)
    forged = {}
    for i, x in clean_targets.items():
        u8 = np.clip(x * 255, 0, 255).round().astype(np.uint8)
        encoded = tm.encode(Image.fromarray(u8), message, MODE="binary")
        forged[i] = np.asarray(encoded, dtype=np.float32) / 255.0
    return forged


def main():
    args = parse_args()
    _, clean = load_dataset(args.dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        stem = category.lower().replace("_", "")
        scheme = getattr(args, arg_name(category, "scheme"))
        lo, hi = CATEGORY_RANGES[category]
        ids = range(lo, hi + 1)

        if scheme == "none":
            for i in ids:
                if args.base_dir is not None and (args.base_dir / f"{i}.png").exists():
                    shutil.copy2(args.base_dir / f"{i}.png", args.output_dir / f"{i}.png")
                else:
                    save_rgb(clean[i], args.output_dir / f"{i}.png")
            print(f"{category}: passthrough ({'base-dir' if args.base_dir else 'clean'})")
            continue

        message = getattr(args, arg_name(category, "message"))
        if message is None:
            raise SystemExit(f"--{stem}-message is required when --{stem}-scheme={scheme}")

        targets = {i: clean[i] for i in ids}
        if scheme in IMWATERMARK_METHODS:
            bits = bits_from_string(message)
            forged = forge_imwatermark(targets, scheme, bits)
        elif scheme == TRUSTMARK_METHOD:
            model_type = getattr(args, arg_name(category, "trustmark_model_type"))
            forged = forge_trustmark(targets, message, model_type)
        else:
            password_wm = getattr(args, arg_name(category, "password_wm"))
            password_img = getattr(args, arg_name(category, "password_img"))
            block_shape = tuple(int(v) for v in getattr(args, arg_name(category, "block_shape")).split(","))
            forged = forge_blind_watermark(targets, message, password_wm, password_img, block_shape)

        for i, y in forged.items():
            save_rgb(y, args.output_dir / f"{i}.png")
        print(f"{category}: forged with {scheme} (message={message})")

    print("saved", args.output_dir)


if __name__ == "__main__":
    main()
