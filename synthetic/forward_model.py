"""
Forward model for synthetic microscopy simulation.

Uses psfmodels for realistic PSF generation and proper convolution.
"""

import numpy as np
from scipy.signal import fftconvolve
from typing import Tuple, Optional

import sys
from pathlib import Path

# Import psfmodels via our wrapper
PSFMODELS_PATH = Path(__file__).parent.parent / "vendor" / "PSFmodels" / "src"
if PSFMODELS_PATH.exists():
    sys.path.insert(0, str(PSFMODELS_PATH))

try:
    import psfmodels as psfm
    HAS_PSFMODELS = True
except ImportError:
    HAS_PSFMODELS = False


def generate_psf(
    nz: int,
    nxy: int,
    dz: float,
    dxy: float,
    wavelength: float,
    na: float,
    ni: float = 1.0,
) -> np.ndarray:
    """
    Generate a realistic widefield PSF using psfmodels.

    Args:
        nz: Number of Z planes (should be odd)
        nxy: XY size in pixels (should be odd)
        dz: Z step in microns
        dxy: Pixel size in microns
        wavelength: Emission wavelength in microns
        na: Numerical aperture
        ni: Immersion refractive index

    Returns:
        3D PSF array, normalized to sum=1
    """
    if not HAS_PSFMODELS:
        raise ImportError("psfmodels not available")

    # Ensure odd dimensions for centering
    nz = nz if nz % 2 == 1 else nz + 1
    nxy = nxy if nxy % 2 == 1 else nxy + 1

    psf = psfm.make_psf(
        z=nz,
        nx=nxy,
        dxy=dxy,
        dz=dz,
        wvl=wavelength,
        NA=na,
        ni=ni,
        ni0=ni,
        model='vectorial'
    )

    # Normalize to sum=1 (energy conservation)
    psf = psf / psf.sum()

    return psf.astype(np.float32)


def convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve image with PSF (forward model).

    Args:
        image: Ground truth image
        psf: Point spread function (should sum to 1)

    Returns:
        Convolved (blurred) image
    """
    # FFT convolution
    blurred = fftconvolve(image, psf, mode='same')

    # Ensure non-negative (numerical errors can cause tiny negatives)
    blurred = np.clip(blurred, 0, None)

    return blurred.astype(np.float32)


def add_poisson_noise(image: np.ndarray, peak_snr: float, seed: int = None) -> np.ndarray:
    """
    Add Poisson noise to simulate photon shot noise.

    Args:
        image: Input image (assumed to be in arbitrary intensity units)
        peak_snr: Signal-to-noise ratio at peak intensity
                  SNR = sqrt(N) for Poisson, so N = SNR^2 photons at peak
        seed: Random seed

    Returns:
        Noisy image
    """
    if seed is not None:
        np.random.seed(seed)

    if image.max() == 0:
        return image.copy()

    # Scale image so peak corresponds to SNR^2 photons
    peak_photons = peak_snr ** 2
    scale = peak_photons / image.max()

    # Convert to photon counts
    photon_image = image * scale

    # Apply Poisson noise
    noisy_photons = np.random.poisson(photon_image.astype(np.float64))

    # Scale back
    noisy = noisy_photons / scale

    return noisy.astype(np.float32)


def add_gaussian_noise(image: np.ndarray, snr: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise (read noise approximation).

    Args:
        image: Input image
        snr: Signal-to-noise ratio (peak / noise_std)
        seed: Random seed

    Returns:
        Noisy image
    """
    if seed is not None:
        np.random.seed(seed)

    if image.max() == 0:
        return image.copy()

    noise_std = image.max() / snr
    noise = np.random.normal(0, noise_std, image.shape)

    noisy = image + noise
    noisy = np.clip(noisy, 0, None)  # No negative values

    return noisy.astype(np.float32)


def simulate_microscopy(
    ground_truth: np.ndarray,
    psf: np.ndarray,
    snr: float = 20,
    noise_type: str = "poisson",
    background: float = 0.0,
    seed: int = None,
) -> np.ndarray:
    """
    Full forward model: blur + noise + background.

    Args:
        ground_truth: True object
        psf: Point spread function
        snr: Signal-to-noise ratio
        noise_type: "poisson" or "gaussian"
        background: Constant background to add
        seed: Random seed

    Returns:
        Simulated microscopy image
    """
    # Convolve
    blurred = convolve(ground_truth, psf)

    # Add background
    if background > 0:
        blurred = blurred + background * blurred.max()

    # Add noise
    if noise_type == "poisson":
        noisy = add_poisson_noise(blurred, snr, seed)
    elif noise_type == "gaussian":
        noisy = add_gaussian_noise(blurred, snr, seed)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy


def perturb_psf(
    psf: np.ndarray,
    sigma_error: float = 0.0,
    z_error: float = 0.0,
) -> np.ndarray:
    """
    Create a perturbed PSF for mismatch testing.

    Args:
        psf: Original PSF
        sigma_error: Fractional error in lateral width (e.g., 0.1 = 10% wider)
        z_error: Fractional error in axial width

    Returns:
        Perturbed PSF
    """
    from scipy.ndimage import zoom

    # Calculate zoom factors
    # Positive error = wider PSF = zoom > 1
    z_zoom = 1 + z_error
    xy_zoom = 1 + sigma_error

    # Apply anisotropic zoom
    zoomed = zoom(psf, (z_zoom, xy_zoom, xy_zoom), order=1)

    # Crop or pad to original size
    result = np.zeros_like(psf)
    src_shape = zoomed.shape
    dst_shape = psf.shape

    # Calculate copy regions
    src_start = [max(0, (s - d) // 2) for s, d in zip(src_shape, dst_shape)]
    src_end = [min(s, src_start[i] + d) for i, (s, d) in enumerate(zip(src_shape, dst_shape))]
    dst_start = [max(0, (d - s) // 2) for s, d in zip(src_shape, dst_shape)]
    dst_end = [dst_start[i] + (src_end[i] - src_start[i]) for i in range(3)]

    result[dst_start[0]:dst_end[0], dst_start[1]:dst_end[1], dst_start[2]:dst_end[2]] = \
        zoomed[src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]

    # Renormalize
    if result.sum() > 0:
        result = result / result.sum()

    return result.astype(np.float32)
