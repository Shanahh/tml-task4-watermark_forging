from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
import omegaconf
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
import zipfile
from scipy.ndimage import gaussian_filter
import math
from PIL import Image
from pathlib import Path
from tqdm import tqdm


MODEL_SIZE = 768



@dataclass
class Candidate:
    name: str
    image: torch.Tensor
    delta: torch.Tensor
    score: float = -1e30
    preference_drop: float = 0.0
    lpips_value: float = 0.0
    mse_value: float = 0.0


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def natural_key(path: Path):
    digits = "".join(c for c in path.stem if c.isdigit())
    return (int(digits) if digits else 10**9, path.stem)


def locate_dataset(root: Path) -> Path:
    root = root.resolve()
    if (root / "clean_targets").is_dir() and (root / "watermarked_sources").is_dir():
        return root
    for wm_dir in root.rglob("watermarked_sources"):
        parent = wm_dir.parent
        if (parent / "clean_targets").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not find clean_targets and watermarked_sources below {root}"
    )


def load_np(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def np_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def tensor_to_np(image: torch.Tensor) -> np.ndarray:
    return image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()


def resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        x, size=size, mode="bilinear", align_corners=False, antialias=True
    )


def save_png(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(image.detach().clamp(0, 1).squeeze(0).cpu()).save(
        path, compress_level=4
    )


def load_preference_model(
    wmforger_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    wmforger_root = wmforger_root.resolve()
    checkpoint = checkpoint.resolve()

    if not (wmforger_root / "wmforger").is_dir():
        raise FileNotFoundError(f"Invalid WmForger root: {wmforger_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    sys.path.insert(0, str(wmforger_root))
    from wmforger.models import build_extractor  # type: ignore

    config = omegaconf.OmegaConf.load(wmforger_root / "configs" / "extractor.yaml")
    model_type = "convnext_tiny"
    model = build_extractor(
        model_type,
        config[model_type],
        img_size=256,
        nbits=0,
    )
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model"]
    model.load_state_dict(state)
    model = model.eval().to(device)

    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_input(image: torch.Tensor) -> torch.Tensor:
    return resize(image, (MODEL_SIZE, MODEL_SIZE))


@torch.no_grad()
def preference_score(model: torch.nn.Module, image: torch.Tensor) -> float:
    return float(model(model_input(image)).mean().item())


def extract_watermark(
    source_np: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    steps: int,
    lr: float,
) -> np.ndarray:
    """
    Official WmForger direction:
      maximize preference score to clean source;
      watermark = source - cleaned.
    """
    h, w = source_np.shape[:2]
    source = np_to_tensor(source_np, device)
    working = model_input(source).detach()

    param = torch.nn.Parameter(torch.zeros_like(working))
    optimizer = torch.optim.SGD([param], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        cleaned = (working + param).clamp(0, 1)
        loss = -model(cleaned).mean()
        loss.backward()
        optimizer.step()

    cleaned = resize((working + param).clamp(0, 1).detach(), (h, w))
    return (source_np - tensor_to_np(cleaned)).astype(np.float32)


def clean_residual(residual: np.ndarray) -> np.ndarray:
    residual = residual.astype(np.float32)
    residual -= np.median(residual, axis=(0, 1), keepdims=True)
    residual -= 0.65 * gaussian_filter(
        residual, sigma=(6.0, 6.0, 0), mode="reflect"
    )
    return residual.astype(np.float32)


def signature(residual: np.ndarray, size: int = 64) -> np.ndarray:
    hp = residual - gaussian_filter(
        residual, sigma=(1.5, 1.5, 0), mode="reflect"
    )
    scale = max(float(np.percentile(np.abs(hp), 99.0)), 1e-8)
    vis = np.clip(hp / scale * 0.5 + 0.5, 0, 1)
    small = Image.fromarray((vis * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    vec = np.asarray(small, dtype=np.float32).reshape(-1)
    vec -= vec.mean()
    vec /= max(float(np.linalg.norm(vec)), 1e-8)
    return vec


def consistency_order(residuals: list[np.ndarray]) -> list[int]:
    sigs = np.stack([signature(x) for x in residuals])
    sim = sigs @ sigs.T
    score = (sim.sum(1) - 1.0) / max(len(residuals) - 1, 1)
    return list(np.argsort(-score))


def trimmed_mean(stack: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    ordered = np.sort(stack, axis=0)
    trim = int(math.floor(len(stack) * fraction))
    if trim == 0:
        return ordered.mean(0)
    return ordered[trim:-trim].mean(0)


def fourier_template(stack: np.ndarray) -> np.ndarray:
    spectra = np.fft.fft2(stack, axes=(1, 2))
    amplitude = np.median(np.abs(spectra), axis=0)
    unit = spectra / (np.abs(spectra) + 1e-8)
    phase_mean = unit.mean(0)
    coherence = np.abs(phase_mean)
    phase = phase_mean / (coherence + 1e-8)

    h, w = stack.shape[1:3]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    band = (
        (1.0 - np.exp(-((radius / 0.018) ** 4)))
        * np.exp(-((radius / 0.95) ** 8))
    )[..., None]

    reliability = np.clip((coherence - 0.08) / 0.55, 0, 1) ** 1.5
    merged = amplitude * phase * reliability * band
    result = np.fft.ifft2(merged, axes=(0, 1)).real.astype(np.float32)
    result -= result.mean(axis=(0, 1), keepdims=True)
    return result


def energy_match(template: np.ndarray, residuals: list[np.ndarray]) -> np.ndarray:
    target_rms = np.median(
        [np.sqrt(np.mean(x * x, dtype=np.float64)) for x in residuals]
    )
    current_rms = np.sqrt(np.mean(template * template, dtype=np.float64))
    if current_rms < 1e-10:
        return template
    return (template * (target_rms / current_rms)).astype(np.float32)


def build_bank(
    residuals: list[np.ndarray],
    keep_individuals: int,
) -> dict[str, np.ndarray]:
    residuals = [clean_residual(x) for x in residuals]
    order = consistency_order(residuals)
    ordered = [residuals[i] for i in order]
    stack = np.stack(ordered)

    top_n = max(5, len(ordered) // 2)
    top = np.stack(ordered[:top_n])

    bank = {
        "median": np.median(stack, axis=0).astype(np.float32),
        "trimmed": trimmed_mean(stack).astype(np.float32),
        "fourier": fourier_template(stack),
        "top_median": np.median(top, axis=0).astype(np.float32),
        "top_trimmed": trimmed_mean(top, 0.15).astype(np.float32),
    }

    for key in list(bank):
        bank[key] = energy_match(bank[key], residuals)

    for rank, item in enumerate(ordered[:keep_individuals], start=1):
        bank[f"individual_{rank:02d}"] = item

    return bank


def texture_mask(clean: torch.Tensor) -> torch.Tensor:
    y = (
        0.2126 * clean[:, 0:1]
        + 0.7152 * clean[:, 1:2]
        + 0.0722 * clean[:, 2:3]
    )
    blur = F.avg_pool2d(y, 5, stride=1, padding=2)
    texture = (y - blur).abs()
    p95 = torch.quantile(texture.flatten(1), 0.95, dim=1).view(-1, 1, 1, 1)
    texture = (texture / p95.clamp_min(1e-6)).clamp(0, 1)
    texture = F.avg_pool2d(texture, 3, stride=1, padding=1)
    return 0.72 + 0.48 * texture


def plant(
    clean: torch.Tensor,
    residual_np: np.ndarray,
    strength: float,
    epsilon_255: float,
    masked: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = np_to_tensor(residual_np, clean.device)
    if residual.shape[-2:] != clean.shape[-2:]:
        residual = resize(residual, clean.shape[-2:])

    delta = residual * strength
    if masked:
        delta *= texture_mask(clean)

    epsilon = epsilon_255 / 255.0
    delta = delta.clamp(-epsilon, epsilon)
    forged = (clean + delta).clamp(0, 1)
    return forged, forged - clean


def build_lpips(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Install LPIPS: pip install lpips") from exc
    model = lpips.LPIPS(net="alex").eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def lpips_distance(model, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = resize(a, (256, 256))
    b = resize(b, (256, 256))
    return model(a * 2 - 1, b * 2 - 1).mean()


@torch.no_grad()
def rank_candidates(
    clean: torch.Tensor,
    candidates: list[Candidate],
    preference_model: torch.nn.Module,
    lpips_model,
    lpips_weight: float,
    mse_weight: float,
) -> list[Candidate]:
    clean_pref = preference_score(preference_model, clean)

    for c in candidates:
        forged_pref = preference_score(preference_model, c.image)
        c.preference_drop = clean_pref - forged_pref
        c.lpips_value = float(lpips_distance(lpips_model, clean, c.image).item())
        c.mse_value = float(F.mse_loss(c.image, clean).item())
        c.score = (
            c.preference_drop
            - lpips_weight * c.lpips_value
            - mse_weight * c.mse_value
        )

    return sorted(candidates, key=lambda x: x.score, reverse=True)


def refine(
    clean: torch.Tensor,
    selected: Candidate,
    preference_model: torch.nn.Module,
    lpips_model,
    steps: int,
    lr_255: float,
    epsilon_255: float,
    lpips_weight: float,
    mse_weight: float,
    anchor_weight: float,
) -> torch.Tensor:
    if steps <= 0:
        return selected.image

    epsilon = epsilon_255 / 255.0
    anchor = selected.delta.detach()
    delta = selected.delta.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr_255 / 255.0)

    best = selected.image.detach()
    best_loss = float("inf")

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        forged = (clean + delta).clamp(0, 1)
        loss = (
            preference_model(model_input(forged)).mean()
            + lpips_weight * lpips_distance(lpips_model, clean, forged)
            + mse_weight * F.mse_loss(forged, clean)
            + anchor_weight * F.mse_loss(delta, anchor)
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
            current = float(loss.item())
            if current < best_loss:
                best_loss = current
                best = (clean + delta).clamp(0, 1).detach().clone()

    return best


def extract_or_load(
    category: str,
    source_paths: list[Path],
    cache_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    steps: int,
    lr: float,
    force: bool,
) -> list[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{category}.npz"

    if cache_path.exists() and not force:
        data = np.load(cache_path)
        return [data[f"residual_{i:02d}"] for i in range(len(source_paths))]

    payload = {}
    residuals = []

    for i, path in enumerate(tqdm(source_paths, desc=f"extract {category}")):
        residual = extract_watermark(load_np(path), model, device, steps, lr)
        residuals.append(residual)
        payload[f"residual_{i:02d}"] = residual

    np.savez_compressed(cache_path, **payload)
    return residuals


def package(output_dir: Path, submission: Path) -> None:
    expected = [f"{i}.png" for i in range(1, 201)]
    paths = sorted(output_dir.glob("*.png"), key=natural_key)
    actual = [p.name for p in paths]

    if actual != expected:
        raise RuntimeError(
            f"Wrong outputs. Missing={sorted(set(expected)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected))}"
        )

    with zipfile.ZipFile(
        submission, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)

