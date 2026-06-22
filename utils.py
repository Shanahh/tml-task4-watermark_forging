import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter



def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path, compress_level=4)

def high_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    return image - gaussian_filter(image, sigma=(sigma, sigma, 0), mode="reflect")


def radial_bandpass(height: int, width: int, low: float, high: float) -> np.ndarray:
    """Smooth radial frequency mask. Frequencies are normalized to Nyquist."""
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy) / 0.5

    low_gate = 1.0 - np.exp(-((radius / max(low, 1e-6)) ** 4))
    high_gate = np.exp(-((radius / max(high, 1e-6)) ** 8))
    return (low_gate * high_gate).astype(np.float32)


def robust_template(source_images: list[np.ndarray]) -> np.ndarray:
    """
    Estimate a shared watermark pattern.

    Spatial median suppresses image-dependent content.
    Fourier phase coherence keeps frequencies consistently aligned across images.
    """
    shape = source_images[0].shape
    if any(image.shape != shape for image in source_images):
        raise ValueError("All source images in a watermark category must have the same size.")

    residual_sets = []
    for sigma in (0.7, 1.4, 2.8):
        residuals = np.stack([high_pass(image, sigma) for image in source_images])
        residuals -= np.median(residuals, axis=(1, 2), keepdims=True)
        residual_sets.append(residuals)

    spatial = (
        0.50 * np.median(residual_sets[0], axis=0)
        + 0.30 * np.median(residual_sets[1], axis=0)
        + 0.20 * np.median(residual_sets[2], axis=0)
    )


    residuals = residual_sets[1]
    spectra = np.fft.fft2(residuals, axes=(1, 2))
    unit_phase = spectra / (np.abs(spectra) + 1e-8)
    coherence = np.abs(np.mean(unit_phase, axis=0))
    mean_spectrum = np.mean(spectra, axis=0)

    h, w, _ = spatial.shape
    band = radial_bandpass(h, w, low=0.018, high=0.90)[..., None]

    reliability = np.clip((coherence - 0.12) / 0.55, 0.0, 1.0) ** 1.5
    fourier = np.fft.ifft2(
        mean_spectrum * reliability * band,
        axes=(0, 1),
    ).real.astype(np.float32)

    template = 0.65 * spatial + 0.35 * fourier

    template -= gaussian_filter(template, sigma=(5.0, 5.0, 0), mode="reflect")
    template -= np.mean(template, axis=(0, 1), keepdims=True)

    scale = np.percentile(np.abs(template), 99.5)
    if scale < 1e-8:
        raise RuntimeError("Estimated watermark template is effectively zero.")
    template = template / scale * (4.0 / 255.0)
    return template.astype(np.float32)


def texture_mask(target: np.ndarray) -> np.ndarray:
    """
    Put more perturbation in textured regions where LPIPS is less sensitive,
    while retaining enough signal in smooth regions for detection.
    """
    luminance = (
        0.2126 * target[..., 0]
        + 0.7152 * target[..., 1]
        + 0.0722 * target[..., 2]
    )
    texture = np.abs(
        luminance - gaussian_filter(luminance, sigma=1.3, mode="reflect")
    )
    p95 = max(float(np.percentile(texture, 95.0)), 1e-6)
    texture = np.clip(texture / p95, 0.0, 1.0)
    texture = gaussian_filter(texture, sigma=1.0, mode="reflect")

    return (0.55 + 0.65 * texture)[..., None].astype(np.float32)