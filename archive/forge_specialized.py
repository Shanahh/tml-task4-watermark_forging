from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, median_filter

from common import EPS, grayscale, load_dataset, rgb_to_ycbcr, save_rgb, ycbcr_to_rgb

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
            return median_filter(channel, size=3, mode="reflect")
    raise ValueError(method)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("specialized_candidates"))
    p.add_argument("--wm1-strength", type=float, default=0.002)
    p.add_argument("--wm3-y-strength", type=float, default=0.007)
    p.add_argument("--wm3-cb-strength", type=float, default=0.001)
    p.add_argument("--wm3-cr-strength", type=float, default=0.001)
    p.add_argument("--wm4-strength", type=float, default=0.009)
    p.add_argument("--wm5-strength", type=float, default=0.007)
    p.add_argument("--wm6-strength", type=float, default=0.04)
    p.add_argument("--wm4-threshold", type=float, default=0.4)
    p.add_argument("--wm6-coeff-count", type=int, default=8)
    p.add_argument(
        "--extraction",
        default="highpass",
        choices=EXTRACTION_METHODS,
        help="how the channel-residual attacks (WM_1/3/5) estimate clean content. "
        "'highpass' is the default and preserves prior behavior; 'denoiser' is "
        "opt-in and isolates spread-spectrum watermarks more cleanly.",
    )
    p.add_argument(
        "--block-match-strength",
        type=float,
        default=0.0,
        help="OPT-IN (default 0 = off, byte-identical to before): additionally apply "
        "block-distribution matching (see apply_block_match) on top of the global "
        "template for WM_1/3/4. score_with_diagnostics.py showed amplitude-calibrated "
        "global templates only weakly reproduce the genuine watermark's block-level "
        "signature",
    )
    p.add_argument("--block-match-grid", type=int, default=8)
    return p.parse_args()


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


def block_grid_means(residual, grid=8):
    h, w = residual.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    means = np.zeros((grid, grid))
    for yi in range(grid):
        for xi in range(grid):
            means[yi, xi] = residual[ys[yi]:ys[yi + 1], xs[xi]:xs[xi + 1]].mean()
    return means, ys, xs


def expand_grid(values, ys, xs, shape):
    out = np.zeros(shape)
    grid = values.shape[0]
    for yi in range(grid):
        for xi in range(grid):
            out[ys[yi]:ys[yi + 1], xs[xi]:xs[xi + 1]] = values[yi, xi]
    return out


def block_match_stats(xs, ch, method, grid=8):
    cell_means = [block_grid_means(channel_residual(x, ch, method), grid)[0] for x in xs]
    stacked = np.stack(cell_means)
    return stacked.mean(0), stacked.std(0) + EPS


def apply_block_match(x, ch, method, ws_mean, ws_std, cs_mean, cs_std, strength, grid=8):
    ycc = rgb_to_ycbcr(x)
    channel = ycc[..., ch]
    residual = channel - estimate_content(channel, method)
    cur_mean, ys, xs = block_grid_means(residual, grid)
    z = (cur_mean - cs_mean) / cs_std
    target_mean = ws_mean + z * ws_std
    offset = expand_grid((target_mean - cur_mean) * strength, ys, xs, channel.shape)
    ycc[..., ch] = np.clip(channel + offset, 0, 1)
    return ycbcr_to_rgb(ycc)


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
        f"WM_4={args.wm4_strength} WM_5={args.wm5_strength} WM_6={args.wm6_strength} "
        f"block_match_strength={args.block_match_strength}"
    )

    block_stats = {}
    if args.block_match_strength > 0:
        grid = args.block_match_grid
        for cat, ch in [("WM_1", 1), ("WM_3", 0), ("WM_3", 1), ("WM_3", 2), ("WM_4", 0)]:
            resolution = (src[cat][0].shape[1], src[cat][0].shape[0])
            ws_mean, ws_std = block_match_stats(src[cat], ch, method, grid)
            cs_mean, cs_std = block_match_stats(by_resolution[resolution], ch, method, grid)
            block_stats[(cat, ch)] = (ws_mean, ws_std, cs_mean, cs_std)

    def maybe_block_match(y, cat, ch):
        if args.block_match_strength <= 0:
            return y
        ws_mean, ws_std, cs_mean, cs_std = block_stats[(cat, ch)]
        return apply_block_match(
            y, ch, method, ws_mean, ws_std, cs_mean, cs_std,
            args.block_match_strength, args.block_match_grid,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, x in clean.items():
        if 1 <= i <= 25:
            y = apply_channel(x, wm1_cb_template, 1, args.wm1_strength)
            y = maybe_block_match(y, "WM_1", 1)
        elif 51 <= i <= 75:
            y = x
            for ch in WM3_CHANNELS:
                y = apply_channel(y, wm3_channel_templates[ch], ch, wm3_strengths[ch])
            for ch in WM3_CHANNELS:
                y = maybe_block_match(y, "WM_3", ch)
        elif 76 <= i <= 100:
            y = apply_luma(x, wm4_phase_template, args.wm4_strength)
            y = maybe_block_match(y, "WM_4", 0)
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
