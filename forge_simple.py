#!/usr/bin/env python3
"""The simplest possible forging approach: raw mean-difference templates.

No high-pass filtering, no YCbCr/channel decomposition, no per-channel
calibration, no denoiser, no block statistics -- just "Can Simple Averaging
Defeat Modern Watermarks?" (Yang et al., NeurIPS 2024) applied literally in
the forging direction, with no embellishment:

    delta   = mean(25 source images of category k)  -  mean(clean images at
              the same resolution)
    forged  = clean_target + s * delta

Averaging cancels each source's differing content, leaving (an estimate of)
the one thing they all share: the watermark itself. Applied identically to
ALL 8 categories via the exact WM_k -> clean-image-id mapping from the
assignment (common.CATEGORY_RANGES / category_for_id) -- including WM_2/7/8,
which earlier high-pass-based diagnostics found nothing in, but that
assumption (the watermark being a high-frequency residual) was never itself
validated.

This exists to test that assumption directly: every other attack in this
repo high-pass filters before averaging. If raw averaging does no better
than 0.22, the high-pass assumption probably isn't the problem. If it does
the same or better with far less machinery, it's the floor everything else
should have been validated against.

Validated on the leaderboard: a single shared strength took the score from
0.22 -> 0.265 (s=0.3) -> 0.295 (s=0.5), then plateaued at s=0.6. Strength is
now per-category (--wm1-strength ... --wm8-strength, still nothing else
added) since the 8 watermarks are independent and likely have different real
embedding strengths, so a single shared value is necessarily a compromise --
the same lesson learned with the calibrated specialized pipeline, reapplied
here without bringing back any of that pipeline's other machinery. Defaults
are all 0.5 (the best uniform value found so far), so running with no
overrides reproduces that submission exactly.

Per-category delta EXTRACTION (--wmN-extraction, default raw):

    raw       delta = mean(sources) - mean(clean_pool)          [default]
    denoise   delta = mean over sources of [source - wavelet_denoise(source)]
    regen     same, but with a STRONG denoiser (BM3D) -- the copy attack
    wmcopier  raw delta with the clean-image content subspace projected out

All non-raw variants are rescaled to the raw delta's per-category RMS, so the
strength range is identical and raw-vs-X is a clean single-variable
comparison: same perturbation amplitude, only the DIRECTION differs.

'regen' is the regeneration/copy attack: remove each source's content with
BM3D (a strong denoiser) and average the residuals, giving a clean watermark
with no foreign-pool content-bias. Measured watermark-consistency across
sources (||mean residual||^2 / mean ||residual||^2) shows this is a 3-4x
cleaner watermark than raw averaging for the copyable groups WM_5 (0.13->0.39)
and WM_6 (0.09->0.31), a weak win for WM_4, and NO help for WM_1/2/3/7/8
(their residual is not consistent under any denoiser -- content-adaptive or
semantic watermarks with no fixed pattern to copy; WM_3 is actually best with
raw). So regen is a targeted win for the additive groups, not a universal
fix. Needs `pip install bm3d`. Tune --wmN-regen-sigma (BM3D noise std, default
0.02; ~watermark amplitude to keep).

'denoise' removes each source's OWN content (wavelet Watermark-Copy-Attack
extractor) before averaging, dropping the content-bias term (mean-content-of-
sources minus mean-content-of-pool, which does not fully cancel with 25
sources). NOT the high-pass mistake -- a denoiser keeps the noise-like
watermark and removes content, the opposite selectivity from high-pass.

'wmcopier' is a FEASIBLE LINEAR ADAPTATION of WMCopier (Dong et al., NeurIPS
2025), NOT the paper's diffusion method. The real WMCopier trains an
unconditional diffusion model on a large self-generated watermarked dataset
to separate watermark from content -- infeasible here (black-box, 25 samples,
no watermark encoder). Its core idea, though, is separating the watermark
from image content. This does that linearly: the raw delta = watermark +
content-bias, where the content-bias lies in the subspace of natural-image
variation. We estimate that subspace as the top-k principal components of the
clean-image pool and project it out of the delta, leaving the part orthogonal
to content (more likely the watermark). Hyperparameter --wmN-components (k,
default 8) trades bias (small k leaves content-bias in) against variance
(large k also removes any watermark energy that overlaps content directions).
k=0 reduces to the raw delta. Best suited to the content-contaminated groups
(those that wanted LOW strength in the raw sweep, e.g. wm4/wm8), and wm1
(flat to strength = direction, not magnitude, is the limit).

Opt-in per category; default raw reproduces the tuned result.

Per-category LPIPS CAP (--wmN-lpips-cap, default 0 = off = fixed strength):

Instead of one fixed strength for all 25 images in a group, cap mode gives
each image the LARGEST strength that keeps its LPIPS under the cap -- so every
image ends at the same quality Sqlt = exp(-8*cap) but the maximum detection
strength it can afford. This is a strictly better allocation of a fixed
quality budget than a single group strength (more watermark on images that
tolerate it, less on those that don't), which directly targets the score
mean(Sdet*Sqlt). The per-image strength is found by binary search in
[0, --max-strength]; the group's --wmN-strength is ignored when its cap > 0.
A good starting cap for a group is the LPIPS its best fixed strength was
already producing -- run with --report-lpips to print that per group.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import CATEGORIES, load_dataset, save_rgb, category_for_id, estimate_content

EXTRACTION_CHOICES = ("raw", "denoise", "wmcopier", "regen")


def arg_name(category, suffix):
    return f"{category.lower().replace('_', '')}_{suffix}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("simple_candidates"))
    for category in CATEGORIES:
        stem = category.lower().replace("_", "")
        p.add_argument(f"--{stem}-strength", type=float, default=0.5)
        p.add_argument(f"--{stem}-extraction", choices=EXTRACTION_CHOICES, default="raw")
        p.add_argument(f"--{stem}-components", type=int, default=8,
                        help="wmcopier extraction: number of clean-content PCs to project out")
        p.add_argument(f"--{stem}-regen-sigma", type=float, default=0.02,
                        help="regen extraction: BM3D denoiser noise std (the watermark amplitude "
                             "to keep in the residual)")
        p.add_argument(f"--{stem}-lpips-cap", type=float, default=0.0,
                        help="if >0, per-image strength is binary-searched to this LPIPS cap "
                             "instead of using the fixed strength")
    p.add_argument("--max-strength", type=float, default=3.0,
                    help="ceiling for the per-image binary search in LPIPS-cap mode")
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg"])
    p.add_argument("--report-lpips", action="store_true",
                    help="print each group's mean LPIPS of the final output (guides cap selection)")
    return p.parse_args()


def load_lpips(net_name):
    import torch
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = lpips.LPIPS(net=net_name).to(device).eval()
    for param in net.parameters():
        param.requires_grad_(False)

    def distance(a, b):
        ta = torch.from_numpy(np.transpose(a, (2, 0, 1))).unsqueeze(0).float().to(device)
        tb = torch.from_numpy(np.transpose(b, (2, 0, 1))).unsqueeze(0).float().to(device)
        with torch.no_grad():
            return float(net(ta, tb, normalize=True))

    return distance


def cap_strength(x, delta, cap, max_strength, lpips_distance, iters=12):
    """Largest strength s in [0, max_strength] with LPIPS(x, clip(x+s*delta))
    <= cap, by binary search. If even max_strength stays under the cap (delta
    too weak to reach it), returns max_strength."""
    if lpips_distance(x, np.clip(x + max_strength * delta, 0, 1)) <= cap:
        return max_strength
    lo, hi = 0.0, max_strength
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if lpips_distance(x, np.clip(x + mid * delta, 0, 1)) <= cap:
            lo = mid
        else:
            hi = mid
    return lo


def raw_delta(sources, clean_pool):
    return np.mean(sources, axis=0) - np.mean(clean_pool, axis=0)


def denoise_delta(sources):
    """Per-source watermark estimate source - denoise(source), averaged.
    Denoises each RGB channel independently with the wavelet denoiser."""
    residuals = []
    for x in sources:
        residual = np.empty_like(x)
        for c in range(x.shape[-1]):
            residual[..., c] = x[..., c] - estimate_content(x[..., c], "denoiser")
        residuals.append(residual)
    return np.mean(residuals, axis=0)


def regen_delta(sources, sigma):
    """Regeneration/copy-attack extraction: source - denoise(source) averaged,
    but with a STRONG denoiser (BM3D) instead of the weak wavelet. BM3D removes
    each source's content while leaving the watermark in the residual, so with
    no foreign-pool content-bias term the averaged residual is a much cleaner
    watermark for content-independent (additive) watermarks. Verified to
    isolate a 3-4x more consistent watermark than raw averaging for the
    copyable groups (WM_5, WM_6); does nothing for content-adaptive/semantic
    groups (WM_1/2/7/8), whose per-source residual is not consistent under any
    denoiser. `sigma` is BM3D's noise std -- roughly the watermark amplitude to
    preserve in the residual."""
    try:
        import bm3d
    except ImportError as e:
        raise SystemExit("regen extraction needs BM3D: pip install bm3d") from e
    residuals = []
    for x in sources:
        residual = np.empty_like(x)
        for c in range(x.shape[-1]):
            residual[..., c] = x[..., c] - bm3d.bm3d(x[..., c], sigma_psd=sigma)
        residuals.append(residual)
    return np.mean(residuals, axis=0)


def wmcopier_delta(sources, clean_pool, n_components):
    """WMCopier-inspired linear watermark/content separation: take the raw
    mean-difference delta and project out the top-n_components principal
    directions of the clean-image pool (the natural-image-content subspace
    that the content-bias term lives in), keeping the part orthogonal to
    content. n_components=0 returns the raw delta unchanged.

    The clean-pool PCs are computed via the Gram trick (eigendecomposition of
    the small N x N covariance of the flattened, mean-centered clean images),
    so this is cheap even at full image resolution.
    """
    delta = raw_delta(sources, clean_pool)
    if n_components <= 0:
        return delta

    # np.errstate: large-K matmul spuriously trips NumPy 2.x's FP-state check
    # ("divide by zero encountered in matmul") though the result is exact.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        flat = np.stack([c.ravel() for c in clean_pool]).astype(np.float64)
        centered = flat - flat.mean(0)
        gram = centered @ centered.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1][:n_components]

        d = delta.ravel().astype(np.float64)
        for j in order:
            if eigvals[j] <= 1e-12:
                continue
            pc = centered.T @ eigvecs[:, j]
            pc /= np.linalg.norm(pc) + 1e-12
            d -= (d @ pc) * pc
    return d.reshape(delta.shape).astype(delta.dtype)


def rms(a):
    return float(np.sqrt((a ** 2).mean()))


def main():
    args = parse_args()
    src, clean = load_dataset(args.dataset)
    strengths = {c: getattr(args, arg_name(c, "strength")) for c in CATEGORIES}
    extractions = {c: getattr(args, arg_name(c, "extraction")) for c in CATEGORIES}
    components = {c: getattr(args, arg_name(c, "components")) for c in CATEGORIES}
    regen_sigmas = {c: getattr(args, arg_name(c, "regen_sigma")) for c in CATEGORIES}
    caps = {c: getattr(args, arg_name(c, "lpips_cap")) for c in CATEGORIES}

    def describe(c):
        tag = extractions[c]
        if tag == "wmcopier":
            tag += f"(k={components[c]})"
        elif tag == "regen":
            tag += f"(sig={regen_sigmas[c]})"
        knob = f"cap={caps[c]}" if caps[c] > 0 else f"s={strengths[c]}"
        return f"{c}:{knob}/{tag}"
    print("config:", " ".join(describe(c) for c in CATEGORIES))

    by_resolution = {}
    for im in clean.values():
        by_resolution.setdefault(im.shape[:2], []).append(im)

    deltas = {}
    for category in CATEGORIES:
        sources = src[category]
        clean_pool = by_resolution[sources[0].shape[:2]]
        rd = raw_delta(sources, clean_pool)
        method = extractions[category]
        if method == "denoise":
            estimate = denoise_delta(sources)
        elif method == "wmcopier":
            estimate = wmcopier_delta(sources, clean_pool, components[category])
        elif method == "regen":
            estimate = regen_delta(sources, regen_sigmas[category])
        else:
            estimate = rd
        # Rescale any non-raw estimate to the raw delta's RMS: same amplitude,
        # direction-only difference, same strength range as the raw attack.
        if method != "raw":
            estimate = estimate * (rms(rd) / (rms(estimate) + 1e-12))
        deltas[category] = estimate

    # LPIPS is only needed if some group uses a cap, or we're reporting.
    need_lpips = any(caps[c] > 0 for c in CATEGORIES) or args.report_lpips
    lpips_distance = load_lpips(args.lpips_net) if need_lpips else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_lpips = {c: [] for c in CATEGORIES}
    for i, x in clean.items():
        category = category_for_id(i)
        delta = deltas[category]
        if caps[category] > 0:
            s = cap_strength(x, delta, caps[category], args.max_strength, lpips_distance)
        else:
            s = strengths[category]
        forged = np.clip(x + s * delta, 0, 1)
        if lpips_distance is not None:
            group_lpips[category].append(lpips_distance(x, forged))
        save_rgb(forged, args.output_dir / f"{i}.png")

    print("saved", args.output_dir)

    if lpips_distance is not None:
        print("per-group mean LPIPS (Sqlt = exp(-8*LPIPS)):")
        for c in CATEGORIES:
            vals = group_lpips[c]
            if vals:
                mean_lp = float(np.mean(vals))
                print(f"  {c}: mean_lpips={mean_lp:.4f}  Sqlt={np.exp(-8 * mean_lp):.4f}")


if __name__ == "__main__":
    main()
