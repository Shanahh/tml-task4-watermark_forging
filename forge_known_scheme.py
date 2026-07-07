from __future__ import annotations

import argparse
import importlib
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import CATEGORIES, CATEGORY_RANGES, load_dataset, save_rgb

IMWATERMARK_METHODS = ("dwtDct", "dwtDctSvd", "rivaGan")
BLIND_WATERMARK_METHOD = "blind_watermark"
TRUSTMARK_METHOD = "trustmark"
SCHEME_CHOICES = ("none",) + IMWATERMARK_METHODS + (BLIND_WATERMARK_METHOD, TRUSTMARK_METHOD)

REQUIRED_MODULE = {
    "dwtDct": "imwatermark", "dwtDctSvd": "imwatermark", "rivaGan": "imwatermark",
    BLIND_WATERMARK_METHOD: "blind_watermark",
    TRUSTMARK_METHOD: "trustmark",
}
INSTALL_HINT = {
    "imwatermark": "pip install invisible-watermark onnxruntime",
    "blind_watermark": "pip install blind-watermark",
    "trustmark": "pip install trustmark",
}


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


def preflight_check(args):
    """Verify every library needed by the requested --wmN-scheme flags is
    importable BEFORE writing anything."""
    missing = {}
    for category in CATEGORIES:
        scheme = getattr(args, arg_name(category, "scheme"))
        if scheme == "none":
            continue
        module_name = REQUIRED_MODULE[scheme]
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.setdefault(module_name, []).append(category)

    if missing:
        lines = ["Missing dependencies for the requested schemes -- nothing has been written:"]
        for module_name, categories in missing.items():
            lines.append(f"  {module_name} (needed for {', '.join(categories)}): {INSTALL_HINT[module_name]}")
        raise SystemExit("\n".join(lines))


def main():
    args = parse_args()
    preflight_check(args)

    _, clean = load_dataset(args.dataset)

    staging_dir = Path(tempfile.mkdtemp(prefix="forge_known_scheme_"))

    for category in CATEGORIES:
        stem = category.lower().replace("_", "")
        scheme = getattr(args, arg_name(category, "scheme"))
        lo, hi = CATEGORY_RANGES[category]
        ids = range(lo, hi + 1)

        if scheme == "none":
            for i in ids:
                if args.base_dir is not None and (args.base_dir / f"{i}.png").exists():
                    shutil.copy2(args.base_dir / f"{i}.png", staging_dir / f"{i}.png")
                else:
                    save_rgb(clean[i], staging_dir / f"{i}.png")
            print(f"{category}: passthrough ({'base-dir' if args.base_dir else 'clean'})")
            continue

        message = getattr(args, arg_name(category, "message"))
        if message is None:
            shutil.rmtree(staging_dir, ignore_errors=True)
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
            save_rgb(y, staging_dir / f"{i}.png")
        print(f"{category}: forged with {scheme} (message={message})")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.move(str(staging_dir), str(args.output_dir))

    print("saved", args.output_dir)


if __name__ == "__main__":
    main()
