#!/usr/bin/env python3
"""Test whether each of the 8 unidentified watermark groups is actually an
instance of a well-known, OPEN-SOURCE watermarking scheme, rather than
estimating the watermark statistically.

Rationale: the task's reference papers (WMCopier, Watermark Copy Attack, etc.)
all benchmark against a small set of standard schemes with public pip-
installable encode/decode implementations (DwtDct, DwtDctSvd, RivaGAN via the
`invisible-watermark` package; a separate DWT+DCT+SVD blind scheme via
`blind-watermark`). If a group's 25 sources actually came from one of these,
running that scheme's OWN decoder on them should recover the SAME message
bits consistently across all 25 images -- not because we estimated anything,
but because we're reading out the real embedding with the real algorithm.
That is qualitatively different from (and far stronger than) any statistical
delta estimate: if a match is found, the SAME library's encoder can then embed
that exact message on the clean targets directly, for near-perfect bit
accuracy at near-zero perceptual cost.

Method: for each candidate scheme/parameter combination, decode all 25
sources of a group and measure bit-AGREEMENT with the per-bit majority vote
across those 25 decodes. Compare against the same measurement on 25 clean
(unwatermarked) images of matching resolution as a negative control. A
scheme/param combo is flagged as a plausible match only if source agreement
is both high in absolute terms AND clearly above the clean-image control
(ruling out a scheme that just happens to produce stable-looking output on
any image, e.g. because of block-level DC energy rather than an actual
embedded bit).

This is a SCREENING tool, not a guarantee -- a match should be confirmed by
visually inspecting the recovered message for structure (e.g. a short cycle
that decodes to plausible bytes across sources) before trusting it, and its
absence does not rule out a scheme this script doesn't happen to try (custom/
proprietary schemes, or ones needing a secret key we don't have).

Requires: pip install invisible-watermark blind-watermark onnxruntime
"""
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
    balance (fraction of 1-bits). 1.0 agreement = perfect, ~0.5 = random.

    A high agreement with a DEGENERATE majority (nearly all 0s or all 1s) is
    a decode-bias artifact, not a real embedded message -- some blind
    decoders default toward one bit value on typical natural-image
    statistics regardless of whether any watermark of theirs is present.
    Confirmed empirically: WM_1 vs dwtDct produced agreement=0.91 with a
    majority string of ~95% ones (and the "recovered message" changed
    structure between length=48 and length=100, itself a red flag) -- while
    WM_2 vs rivaGan produced agreement=0.99 with a balanced, non-degenerate
    32-bit string (16/32 ones) that round-trips perfectly through the real
    encoder. Only the latter pattern is trustworthy."""
    if not bit_lists:
        return None, None, None
    M = np.stack(bit_lists).astype(np.int64)
    majority = (M.mean(0) >= 0.5).astype(np.int64)
    return float((M == majority).mean()), majority, float(majority.mean())


# --------------------------------------------------------------------------
# invisible-watermark (imwatermark): dwtDct, dwtDctSvd, rivaGan
# --------------------------------------------------------------------------

def imwatermark_decode_all(images_bgr, method, length):
    from imwatermark import WatermarkDecoder

    decoder = WatermarkDecoder("bits", length)
    if method == "rivaGan":
        WatermarkDecoder.loadModel()
    bit_lists = []
    for img in images_bgr:
        try:
            bits = decoder.decode(img, method=method)
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
              "at least that -- skipping (would need upscaling, which risks false negatives)")
        return

    for method, length in configs:
        src_bits = imwatermark_decode_all(sources_bgr, method, length)
        if src_bits is None:
            continue
        ctrl_bits = imwatermark_decode_all(control_bgr, method, length)
        agreement, majority, balance = bit_agreement(src_bits)
        control, _, _ = bit_agreement(ctrl_bits) if ctrl_bits else (None, None, None)
        results.append({
            "category": category, "library": "imwatermark", "method": method, "length": length,
            "agreement": agreement, "control": control, "balance": balance,
            "majority": "".join(map(str, majority)) if majority is not None else None,
        })


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
        agreement, majority, balance = bit_agreement(src_bits)
        control, _, _ = bit_agreement(ctrl_bits) if ctrl_bits else (None, None, None)
        results.append({
            "category": category, "library": "blind_watermark", "method": "dwt_dct_svd", "length": length,
            "agreement": agreement, "control": control, "balance": balance,
            "majority": "".join(map(str, majority)) if majority is not None else None,
        })


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--categories", default=",".join(CATEGORIES))
    p.add_argument("--n-control", type=int, default=25, help="how many clean images to use as the negative control")
    p.add_argument("--skip-imwatermark", action="store_true")
    p.add_argument("--skip-blind-watermark", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

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
        if not args.skip_blind_watermark:
            run_blind_watermark(sources_bgr, control_bgr, results, category)

    print(f"\n{'category':6} {'library':14} {'method':10} {'len':>4} {'agree':>7} {'control':>8} {'balance':>8}  verdict")
    print("-" * 95)
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
            verdict = "degenerate majority -- likely decode-bias artifact, not a real message"
        else:
            verdict = ""
        ctrl_str = f"{r['control']:.3f}" if r["control"] is not None else "n/a"
        bal_str = f"{r['balance']:.2f}" if r["balance"] is not None else "n/a"
        print(f"{r['category']:6} {r['library']:14} {r['method']:10} {r['length']:4} "
              f"{r['agreement']:7.3f} {ctrl_str:>8} {bal_str:>8}  {verdict}")

    print(f"\n{len(flagged)} possible match(es) flagged (balanced majority + high agreement + clear control gap).")
    for r in flagged:
        print(f"  {r['category']} / {r['library']} / {r['method']} (len={r['length']}): majority bits = {r['majority']}")
    if flagged:
        print("Confirm each by round-tripping: use the SAME library's encoder to embed this exact message")
        print("on the matching clean targets, decode again, and check both bit accuracy and real LPIPS.")


if __name__ == "__main__":
    main()
