from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import CATEGORIES, load_dataset


def to_bgr_u8(rgb_float):
    u8 = np.clip(rgb_float * 255, 0, 255).round().astype(np.uint8)
    return u8[..., ::-1].copy()  # RGB -> BGR, contiguous for cv2


def bit_agreement(bit_lists):
    """Mean fraction of bits matching the per-bit majority vote, across a
    list of equal-length 0/1 arrays, plus the majority string itself and its
    balance (fraction of 1-bits). 1.0 agreement = perfect, ~0.5 = random."""
    if not bit_lists:
        return None, None, None
    M = np.stack(bit_lists).astype(np.int64)
    majority = (M.mean(0) >= 0.5).astype(np.int64)
    return float((M == majority).mean()), majority, float(majority.mean())


def add_result(results, category, library, method, params, bit_lists_src, bit_lists_ctrl):
    agreement, majority, balance = bit_agreement(bit_lists_src)
    control, _, _ = bit_agreement(bit_lists_ctrl) if bit_lists_ctrl else (None, None, None)
    results.append({
        "category": category, "library": library, "method": method, "params": params,
        "agreement": agreement, "control": control, "balance": balance,
        "majority": "".join(map(str, majority)) if majority is not None else None,
    })


def imwatermark_decode_all(images_bgr, method, length, **configs):
    from imwatermark import WatermarkDecoder

    decoder = WatermarkDecoder("bits", length)
    if method == "rivaGan":
        WatermarkDecoder.loadModel()
    bit_lists = []
    for img in images_bgr:
        try:
            bits = decoder.decode(img, method=method, **configs)
            bit_lists.append(np.asarray(bits, dtype=np.int64))
        except Exception:
            return None
    return bit_lists


def run_imwatermark(sources_bgr, control_bgr, results, category):
    configs = [("dwtDct", n) for n in (32, 48, 64, 100)]
    configs += [("dwtDctSvd", n) for n in (32, 48, 64, 100)]
    configs += [("rivaGan", 32)]

    min_side = min(min(im.shape[:2]) for im in sources_bgr)
    if min_side < 256:
        print(f"  [imwatermark] {category}: images smaller than 256x256, library requires "
              "at least that -- skipping default-param pass (would need upscaling, which "
              "risks false negatives)")
        return

    for method, length in configs:
        src_bits = imwatermark_decode_all(sources_bgr, method, length)
        if src_bits is None:
            continue
        ctrl_bits = imwatermark_decode_all(control_bgr, method, length)
        add_result(results, category, "imwatermark", method, f"len={length}", src_bits, ctrl_bits)


# --------------------------------------------------------------------------
# scale/block/channel parameter sweep for dwtDct and dwtDctSvd.
# --------------------------------------------------------------------------

def run_dct_scale_sweep(sources_bgr, control_bgr, results, category, scale_grid, block_grid, length_grid):
    min_side = min(min(im.shape[:2]) for im in sources_bgr)
    if min_side < 256:
        print(f"  [dct-sweep] {category}: images smaller than 256x256 -- skipping")
        return

    # scales index: [Y, Cb, Cr] but both methods structurally only read
    # indices 0 (Y) and 1 (Cb) -- Cr (index 2) is always dead code.
    channel_configs = {
        "Cb-only": lambda s: [0, s, 0],
        "Y-only": lambda s: [s, 0, 0],
    }

    for method in ("dwtDct", "dwtDctSvd"):
        for channel_name, make_scales in channel_configs.items():
            for scale in scale_grid:
                for block in block_grid:
                    for length in length_grid:
                        scales = make_scales(scale)
                        src_bits = imwatermark_decode_all(
                            sources_bgr, method, length, scales=scales, block=block)
                        if src_bits is None:
                            continue
                        ctrl_bits = imwatermark_decode_all(
                            control_bgr, method, length, scales=scales, block=block)
                        params = f"scale={scale} block={block} ch={channel_name} len={length}"
                        add_result(results, category, "imwatermark", method, params, src_bits, ctrl_bits)


# --------------------------------------------------------------------------
# blind-watermark: DWT+DCT+SVD, default password
# --------------------------------------------------------------------------

def blind_watermark_decode_all(images_bgr, length):
    from blind_watermark import WaterMark

    bit_lists = []
    for img in images_bgr:
        try:
            wm = WaterMark(password_wm=1, password_img=1)
            soft = wm.extract(embed_img=img, wm_shape=length, mode="bit")
            bit_lists.append((np.asarray(soft) >= 0.5).astype(np.int64))
        except Exception:
            return None
    return bit_lists


def run_blind_watermark(sources_bgr, control_bgr, results, category):
    for length in (32, 48, 64, 100, 128):
        src_bits = blind_watermark_decode_all(sources_bgr, length)
        if src_bits is None:
            continue
        ctrl_bits = blind_watermark_decode_all(control_bgr, length)
        add_result(results, category, "blind_watermark", "dwt_dct_svd", f"len={length}", src_bits, ctrl_bits)


def trustmark_decode_all(images_bgr, tm):
    from PIL import Image

    bit_lists = []
    for img_bgr in images_bgr:
        try:
            pil_img = Image.fromarray(img_bgr[..., ::-1])  # BGR -> RGB for PIL
            decoded, detected, _ = tm.decode(pil_img, MODE="binary")
            if not detected:
                return None
            bit_lists.append(np.array([int(c) for c in decoded], dtype=np.int64))
        except Exception:
            return None
    return bit_lists


def run_trustmark(sources_bgr, control_bgr, results, category, model_types):
    for model_type in model_types:
        try:
            from trustmark import TrustMark
            tm = TrustMark(verbose=False, model_type=model_type, use_ECC=False)
        except Exception as e:
            print(f"  [trustmark] {category}: failed to load model_type={model_type}: {e}")
            continue

        src_bits = trustmark_decode_all(sources_bgr, tm)
        if src_bits is None:
            print(f"  [trustmark] {category}: model_type={model_type} did not detect a watermark on all sources")
            continue
        ctrl_bits = trustmark_decode_all(control_bgr, tm)
        add_result(results, category, "trustmark", model_type, "binary/no-ECC", src_bits, ctrl_bits)


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--categories", default=",".join(CATEGORIES))
    p.add_argument("--n-control", type=int, default=25, help="how many clean images to use as the negative control")
    p.add_argument("--skip-imwatermark", action="store_true")
    p.add_argument("--skip-blind-watermark", action="store_true")
    p.add_argument("--skip-trustmark", action="store_true")
    p.add_argument("--skip-dct-sweep", action="store_true")
    p.add_argument("--trustmark-model-types", default="Q",
                    help="comma-separated subset of C,Q,B,P -- each downloads its own checkpoint on first use")
    p.add_argument("--scale-grid", default="10,20,30,36,50,75,100,150,200")
    p.add_argument("--block-grid", default="4")
    p.add_argument("--dct-sweep-lengths", default="32,100")
    return p.parse_args()


def main():
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    model_types = [m.strip() for m in args.trustmark_model_types.split(",") if m.strip()]
    scale_grid = [float(v) for v in args.scale_grid.split(",")]
    block_grid = [int(v) for v in args.block_grid.split(",")]
    length_grid = [int(v) for v in args.dct_sweep_lengths.split(",")]

    src, clean = load_dataset(args.dataset)
    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault(im.shape[:2], []).append(im)

    results = []
    for category in categories:
        print(f"=== {category} ===")
        sources_bgr = [to_bgr_u8(x) for x in src[category]]
        resolution = src[category][0].shape[:2]
        control_bgr = [to_bgr_u8(x) for x in by_resolution[resolution][: args.n_control]]

        if not args.skip_imwatermark:
            run_imwatermark(sources_bgr, control_bgr, results, category)
        if not args.skip_dct_sweep:
            run_dct_scale_sweep(sources_bgr, control_bgr, results, category, scale_grid, block_grid, length_grid)
        if not args.skip_blind_watermark:
            run_blind_watermark(sources_bgr, control_bgr, results, category)
        if not args.skip_trustmark:
            run_trustmark(sources_bgr, control_bgr, results, category, model_types)

    print(f"\n{'category':6} {'library':14} {'method':10} {'params':30} {'agree':>7} {'control':>8} {'balance':>8}  verdict")
    print("-" * 120)
    flagged = []
    for r in results:
        if r["agreement"] is None:
            continue
        gap = r["agreement"] - (r["control"] if r["control"] is not None else 0.5)
        degenerate = r["balance"] is not None and not (0.3 <= r["balance"] <= 0.7)
        if r["agreement"] > 0.9 and gap > 0.15 and not degenerate:
            verdict = "POSSIBLE MATCH"
            flagged.append(r)
        elif r["agreement"] > 0.9 and gap > 0.15 and degenerate:
            verdict = "degenerate majority, likely decode-bias artifact"
        else:
            verdict = ""
        ctrl_str = f"{r['control']:.3f}" if r["control"] is not None else "n/a"
        bal_str = f"{r['balance']:.2f}" if r["balance"] is not None else "n/a"
        print(f"{r['category']:6} {r['library']:14} {r['method']:10} {r['params']:30} "
              f"{r['agreement']:7.3f} {ctrl_str:>8} {bal_str:>8}  {verdict}")

    print(f"\n{len(flagged)} possible matches flagged.")
    for r in flagged:
        print(f"  {r['category']} / {r['library']} / {r['method']} ({r['params']}): majority bits = {r['majority']}")
    if flagged:
        print("Confirm by round-tripping")


if __name__ == "__main__":
    main()
