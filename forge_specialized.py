#!/usr/bin/env python3
"""Hand-crafted, non-neural forging attacks for the watermark groups that show
a validated, domain-specific statistical signal in diagnose_watermarks_validated.py:

    WM_1  Cb-channel high-pass residual template
    WM_3  combined Y/Cb/Cr high-pass residual templates (all three channels
          show strong, independent evidence -- Y_auc/Cb_auc/Cr_auc all
          ~0.97-0.99 -- so this sidesteps the surrogate+PGD transferability
          question entirely for this category, the same way WM_1's Cb
          template does for its one strong channel)
    WM_4  coherent Fourier-phase template
    WM_5  Cb/Cr residual template *and* Cb/Cr LSB bit-plane copy (combined,
          since both domains show independent, significant evidence and the
          LSB edit costs essentially no extra perceptual budget)
    WM_6  block-DCT coefficient distribution matching

WM_3 also has a surrogate-classifier + PGD path (train_surrogate.py /
forge_pgd.py) kept around for ablation comparison, but the hand-crafted
attack here is the default routing choice since it doesn't depend on a
black-box proxy model's transferability to the real detector.

WM_2, WM_7, WM_8 are left untouched here; they are handled by the
surrogate-classifier + PGD pipeline only, since they show no validated
hand-crafted signal at all.

Strength is per-category (and per-channel for WM_3), and the defaults are
amplitude-CALIBRATED, not chosen for maximum perceptual budget. calibrate_
strength.py measures the genuine watermark's own amplitude from the 25
sources (by projecting each source's residual onto the extracted template)
and the default strengths below reproduce that amplitude, so the forgery is
"as watermarked as a genuine source" rather than as strong as the LPIPS
budget allows. This matters because overshooting the genuine amplitude puts
the forgery outside the distribution of real watermarked images -- which can
lower the real decoder's bit accuracy (Sdet) AND costs Sqlt, a double loss.
An earlier LPIPS-budget-driven sweep (sweep_lpips_strength.py) is still
useful for seeing where quality collapses, but the genuine amplitude, not
the quality knee, is the correct strength target. Re-run calibrate_strength.py
and update the defaults if the templates change.

WM_3's three channels have different genuine amplitudes (Y is ~4.6x Cb/Cr),
so it takes three separate strengths (--wm3-y/cb/cr-strength) instead of one.

This script produces exactly one candidate set per invocation (no more
strength_<value> subfolders) -- if you want to compare strength choices,
run it multiple times with different --output-dir values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, median_filter

from common import EPS, grayscale, load_dataset, rgb_to_ycbcr, save_rgb, ycbcr_to_rgb


# --------------------------------------------------------------------------
# Content estimation for residual extraction.
#
# The channel-residual attacks recover the watermark as (channel - estimated
# clean content). How the clean content is estimated determines how cleanly
# the watermark is isolated:
#
#   "highpass" (default): clean ~ gaussian-blurred channel. Cheap, but it
#       keeps sharp edges/texture (high-frequency *content*) in the residual,
#       leaking content into the template, and discards any low-frequency
#       watermark component.
#   "denoiser": clean ~ a proper edge-preserving denoiser. For a noise-like
#       (spread-spectrum) watermark this isolates delta far better, because the
#       denoiser removes the noise-like watermark across all bands while
#       preserving content (edges/texture) -- i.e. the residual is the
#       watermark, not the content. This is the classic Watermark Copy Attack
#       (Kutter et al.) estimator. Wavelet (BayesShrink) when PyWavelets is
#       available, scipy median-filter fallback otherwise.
#
# This is opt-in via --extraction; the default preserves the prior behavior
# exactly, so an existing calibrated run is unchanged.
# --------------------------------------------------------------------------

EXTRACTION_METHODS = ("highpass", "denoiser")


def _wavelet_denoise(channel, wavelet="db4", level=3):
    import pywt

    coeffs = pywt.wavedec2(channel, wavelet, level=level, mode="periodization")
    finest = coeffs[-1][-1]
    sigma = np.median(np.abs(finest)) / 0.6745  # robust noise std (MAD)
    out = [coeffs[0]]
    for details in coeffs[1:]:
        thresholded = []
        for d in details:
            var = np.var(d)
            sigma_x = np.sqrt(max(var - sigma ** 2, 1e-12))
            thresh = sigma ** 2 / sigma_x  # BayesShrink, per subband
            thresholded.append(pywt.threshold(d, thresh, mode="soft"))
        out.append(tuple(thresholded))
    denoised = pywt.waverec2(out, wavelet, mode="periodization")
    return denoised[: channel.shape[0], : channel.shape[1]]


def estimate_content(channel, method):
    if method == "highpass":
        return gaussian_filter(channel, 1.5, mode="reflect")
    if method == "denoiser":
        try:
            return _wavelet_denoise(channel)
        except ImportError:
            # scipy-only fallback: median filter is edge-preserving and needs
            # no extra dependency. Less ideal than wavelet for spread-spectrum
            # watermarks but still far better than a gaussian blur.
            return median_filter(channel, size=3, mode="reflect")
    raise ValueError(method)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("specialized_candidates"))
    # Defaults are the amplitude-calibrated strengths from calibrate_strength.py:
    # the strength at which the forgery's projection onto the watermark template
    # matches the genuine watermark's own projection, measured from the 25
    # sources. This reproduces the real watermark amplitude rather than
    # maximizing within an LPIPS budget -- overshooting puts the forgery outside
    # the genuine watermark distribution (hurting Sdet) and costs Sqlt. Re-run
    # calibrate_strength.py and update these if the templates change.
    p.add_argument("--wm1-strength", type=float, default=0.0024)
    # WM_3's three channels have different genuine amplitudes (Y ~4.6x Cb/Cr),
    # so they get separate strengths rather than one shared value.
    p.add_argument("--wm3-y-strength", type=float, default=0.0079)
    p.add_argument("--wm3-cb-strength", type=float, default=0.0017)
    p.add_argument("--wm3-cr-strength", type=float, default=0.0017)
    p.add_argument("--wm4-strength", type=float, default=0.0091)
    p.add_argument("--wm5-strength", type=float, default=0.0071)
    # WM_6 is distribution-matched, not amplitude-additive: its interpolation
    # factor saturates at strength>=0.04 (full match to the source DCT
    # distribution), so this is left at the saturation point, not calibrated.
    p.add_argument("--wm6-strength", type=float, default=0.04)
    p.add_argument("--wm4-threshold", type=float, default=0.45)
    p.add_argument("--wm6-coeff-count", type=int, default=8)
    p.add_argument(
        "--extraction",
        default="highpass",
        choices=EXTRACTION_METHODS,
        help="how the channel-residual attacks (WM_1/3/5) estimate clean content. "
        "'highpass' is the default and preserves prior behavior; 'denoiser' is "
        "opt-in and isolates spread-spectrum watermarks more cleanly. NOTE: the "
        "calibrated default strengths were derived for 'highpass' -- re-run "
        "calibrate_strength.py --extraction denoiser and pass the new strengths.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# WM_1, WM_3, WM_5: channel residual templates (content estimated per
# --extraction; see estimate_content above).
# --------------------------------------------------------------------------

def channel_residual(x, ch, method="highpass"):
    c = rgb_to_ycbcr(x)[..., ch]
    return c - estimate_content(c, method)


def channel_template(xs, ch, method="highpass"):
    residuals = [channel_residual(x, ch, method) for x in xs]
    template = np.median(np.stack(residuals), 0)
    return (template - template.mean()) / (template.std() + EPS)


def apply_channel(x, template, ch, strength):
    y = rgb_to_ycbcr(x)
    y[..., ch] = np.clip(y[..., ch] + strength * template, 0, 1)
    return ycbcr_to_rgb(y)


# --------------------------------------------------------------------------
# WM_3: combined Y/Cb/Cr channel templates (reuses channel_template/
# apply_channel above, which already work for any YCbCr channel index)
# --------------------------------------------------------------------------

WM3_CHANNELS = (0, 1, 2)  # Y, Cb, Cr -- all independently strong for WM_3


# --------------------------------------------------------------------------
# WM_5: Cb/Cr LSB bit-plane templates
# --------------------------------------------------------------------------

def lsb_template(xs, ch):
    bits = []
    for x in xs:
        c = np.round(rgb_to_ycbcr(x)[..., ch] * 255).astype(np.uint8)
        bits.append(c & 1)
    return (np.mean(np.stack(bits), 0) >= 0.5).astype(np.uint8)


def apply_lsb(x, template, ch, max_delta=6):
    """Set the LSB of the saved Cb/Cr byte to `template`, exactly, after the
    PNG save/reload round trip.

    Setting the bit on the *derived* floating-point YCbCr value and
    converting back to RGB does NOT work: save_rgb rounds the resulting RGB
    to uint8 for the PNG, and re-deriving YCbCr from that rounded RGB after
    reload essentially never reproduces the intended Cb/Cr byte exactly
    (confirmed empirically: 100% bit mismatch in practice). Cb is dominated
    by the Blue channel (weight 0.5) and Cr by the Red channel (weight 0.5),
    so instead this perturbs only that one RGB channel, by the smallest
    integer delta, until the resulting Cb/Cr byte -- computed exactly as it
    will be after the uint8 RGB round trip -- has the desired LSB.
    """
    assert ch in (1, 2)
    driver = 2 if ch == 1 else 0  # Cb <- Blue, Cr <- Red

    rgb_u8 = np.clip(np.round(x * 255), 0, 255).astype(np.int16)
    r, g, b = (rgb_u8[..., 0].astype(np.float64), rgb_u8[..., 1].astype(np.float64),
               rgb_u8[..., 2].astype(np.float64))

    best_delta = np.zeros(rgb_u8.shape[:2], dtype=np.int16)
    found = np.zeros(rgb_u8.shape[:2], dtype=bool)

    for delta in sorted(range(-max_delta, max_delta + 1), key=abs):
        candidate = np.clip(rgb_u8[..., driver].astype(np.int16) + delta, 0, 255).astype(np.float64)
        rr, gg, bb = r, g, b
        if driver == 2:
            bb = candidate
        else:
            rr = candidate
        if ch == 1:
            component = -0.168736 * rr / 255 - 0.331264 * gg / 255 + 0.5 * bb / 255 + 0.5
        else:
            component = 0.5 * rr / 255 - 0.418688 * gg / 255 - 0.081312 * bb / 255 + 0.5
        byte = np.round(component * 255).astype(np.int16) & 1
        match = (byte == template) & ~found
        best_delta = np.where(match, delta, best_delta)
        found = found | match
        if found.all():
            break

    out = rgb_u8.astype(np.int16).copy()
    out[..., driver] = np.clip(out[..., driver] + best_delta, 0, 255)
    return out.astype(np.float32) / 255.0


def apply_lsb_pair(x, cb_template, cr_template, max_delta=4):
    """Set both the Cb and Cr LSBs at once, exactly, after the PNG round trip.

    Calling apply_lsb() twice in sequence (once per channel) does NOT work:
    fixing Cr nudges the Red channel, but Cb's formula has a non-zero Red
    coefficient (-0.168736), so that nudge can flip the Cb bit that was just
    set (confirmed empirically: ~9% mismatch on the combined attack output,
    vs <0.3% for either channel fixed in isolation). This solves for a joint
    (delta_red, delta_blue) pair per pixel that satisfies both bits at once,
    smallest combined magnitude first, instead of fixing them independently.
    """
    rgb_u8 = np.clip(np.round(x * 255), 0, 255).astype(np.int16)
    r0, g0, b0 = (rgb_u8[..., 0].astype(np.float64), rgb_u8[..., 1].astype(np.float64),
                  rgb_u8[..., 2].astype(np.float64))

    best_dr = np.zeros(rgb_u8.shape[:2], dtype=np.int16)
    best_db = np.zeros(rgb_u8.shape[:2], dtype=np.int16)
    found = np.zeros(rgb_u8.shape[:2], dtype=bool)

    deltas = range(-max_delta, max_delta + 1)
    candidates = sorted(
        ((dr, db) for dr in deltas for db in deltas),
        key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])),
    )

    for dr, db in candidates:
        rr = np.clip(r0 + dr, 0, 255)
        bb = np.clip(b0 + db, 0, 255)
        cb_norm = -0.168736 * rr / 255 - 0.331264 * g0 / 255 + 0.5 * bb / 255 + 0.5
        cr_norm = 0.5 * rr / 255 - 0.418688 * g0 / 255 - 0.081312 * bb / 255 + 0.5
        cb_byte = np.round(cb_norm * 255).astype(np.int16) & 1
        cr_byte = np.round(cr_norm * 255).astype(np.int16) & 1
        match = (cb_byte == cb_template) & (cr_byte == cr_template) & ~found
        best_dr = np.where(match, dr, best_dr)
        best_db = np.where(match, db, best_db)
        found = found | match
        if found.all():
            break

    out = rgb_u8.astype(np.int16).copy()
    out[..., 0] = np.clip(out[..., 0] + best_dr, 0, 255)
    out[..., 2] = np.clip(out[..., 2] + best_db, 0, 255)
    return out.astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# WM_4: coherent Fourier-phase template
# --------------------------------------------------------------------------

def phase_template(xs, threshold):
    residuals = []
    for x in xs:
        g = grayscale(x)
        residuals.append(g - gaussian_filter(g, 1.5, mode="reflect"))
    z = np.fft.fft2(np.stack(residuals), axes=(1, 2))
    unit = z / (np.abs(z) + EPS)
    mean_unit = unit.mean(0)
    coherence = np.abs(mean_unit)
    phase = np.angle(mean_unit)
    magnitude = np.median(np.abs(z), 0)
    mask = coherence >= threshold
    mask[0, 0] = False
    template = np.fft.ifft2(mask * magnitude * np.exp(1j * phase)).real
    return (template - template.mean()) / (template.std() + EPS)


def apply_luma(x, template, strength):
    y = rgb_to_ycbcr(x)
    y[..., 0] = np.clip(y[..., 0] + strength * template, 0, 1)
    return ycbcr_to_rgb(y)


# --------------------------------------------------------------------------
# WM_6: block-DCT coefficient distribution matching
# --------------------------------------------------------------------------

DCT_COORDS = [
    (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2),
    (0, 3), (3, 0), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3),
]


def dct_block_stats(xs):
    samples = {coord: [] for coord in DCT_COORDS}
    for x in xs:
        g = grayscale(x)
        h, w = g.shape
        g = g[: h - h % 8, : w - w % 8] - 0.5
        for y in range(0, g.shape[0], 8):
            for z in range(0, g.shape[1], 8):
                block = dctn(g[y : y + 8, z : z + 8], type=2, norm="ortho")
                for coord in DCT_COORDS:
                    samples[coord].append(block[coord])
    return {
        coord: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v) + EPS),
            "sign": float(np.mean(np.asarray(v) > 0)),
        }
        for coord, v in samples.items()
    }


def select_dct_coords(wm_stats, clean_stats, n):
    scored = []
    for coord in DCT_COORDS:
        a, b = wm_stats[coord], clean_stats[coord]
        score = (
            abs(a["mean"] - b["mean"]) / b["std"]
            + abs(a["std"] - b["std"]) / b["std"]
            + abs(a["sign"] - b["sign"])
        )
        scored.append((score, coord))
    return [coord for _, coord in sorted(scored, reverse=True)[:n]]


def apply_dct(x, coords, wm_stats, clean_stats, strength):
    ycc = rgb_to_ycbcr(x)
    y = ycc[..., 0].copy()
    h, w = y.shape
    out = y.copy()
    for by in range(0, h - h % 8, 8):
        for bx in range(0, w - w % 8, 8):
            block = dctn(y[by : by + 8, bx : bx + 8] - 0.5, type=2, norm="ortho")
            for coord in coords:
                z = (block[coord] - clean_stats[coord]["mean"]) / clean_stats[coord]["std"]
                target = wm_stats[coord]["mean"] + z * wm_stats[coord]["std"]
                block[coord] = (1 - strength) * block[coord] + strength * target
            out[by : by + 8, bx : bx + 8] = idctn(block, type=2, norm="ortho") + 0.5
    ycc[..., 0] = np.clip(out, 0, 1)
    return ycbcr_to_rgb(ycc)


def main():
    args = parse_args()
    src, clean = load_dataset(args.dataset)

    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault((im.shape[1], im.shape[0]), []).append(im)

    method = args.extraction
    wm1_cb_template = channel_template(src["WM_1"], 1, method)
    wm3_channel_templates = {ch: channel_template(src["WM_3"], ch, method) for ch in WM3_CHANNELS}
    wm4_phase_template = phase_template(src["WM_4"], args.wm4_threshold)
    wm5_cb_template = channel_template(src["WM_5"], 1, method)
    wm5_cr_template = channel_template(src["WM_5"], 2, method)
    wm5_cb_lsb = lsb_template(src["WM_5"], 1)
    wm5_cr_lsb = lsb_template(src["WM_5"], 2)

    wm6_resolution = (src["WM_6"][0].shape[1], src["WM_6"][0].shape[0])
    wm6_stats = dct_block_stats(src["WM_6"])
    wm6_clean_stats = dct_block_stats(by_resolution[wm6_resolution])
    wm6_coords = select_dct_coords(wm6_stats, wm6_clean_stats, args.wm6_coeff_count)
    print("WM6 DCT coeffs", wm6_coords)
    wm3_strengths = {0: args.wm3_y_strength, 1: args.wm3_cb_strength, 2: args.wm3_cr_strength}
    print(
        f"extraction={method} | strengths: WM_1={args.wm1_strength} "
        f"WM_3(Y/Cb/Cr)={args.wm3_y_strength}/{args.wm3_cb_strength}/{args.wm3_cr_strength} "
        f"WM_4={args.wm4_strength} WM_5={args.wm5_strength} WM_6={args.wm6_strength}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, x in clean.items():
        if 1 <= i <= 25:
            y = apply_channel(x, wm1_cb_template, 1, args.wm1_strength)
        elif 51 <= i <= 75:
            y = x
            for ch in WM3_CHANNELS:
                y = apply_channel(y, wm3_channel_templates[ch], ch, wm3_strengths[ch])
        elif 76 <= i <= 100:
            y = apply_luma(x, wm4_phase_template, args.wm4_strength)
        elif 101 <= i <= 125:
            y = apply_channel(x, wm5_cb_template, 1, args.wm5_strength)
            y = apply_channel(y, wm5_cr_template, 2, args.wm5_strength)
            y = apply_lsb_pair(y, wm5_cb_lsb, wm5_cr_lsb)
        elif 126 <= i <= 150:
            y = apply_dct(x, wm6_coords, wm6_stats, wm6_clean_stats, min(1, args.wm6_strength * 25))
        else:
            y = x
        save_rgb(y, args.output_dir / f"{i}.png")

    print("saved", args.output_dir)


if __name__ == "__main__":
    main()
