"""
Point Spread Function (PSF) handling for deconvolution.
"""

import numpy as np
import torch
from .utils import to_tensor, get_device


def load_psf(path, normalize=True):
    """
    Load a PSF from a TIFF file.

    Args:
        path: Path to PSF TIFF file
        normalize: If True, normalize PSF to sum to 1

    Returns:
        torch.Tensor: PSF as a 3D tensor (Z, Y, X)
    """
    import tifffile as tf

    psf = tf.imread(path).astype(np.float64)

    # Ensure 3D
    if psf.ndim == 2:
        psf = psf[np.newaxis, ...]

    if normalize:
        psf = psf / psf.sum()

    return to_tensor(psf)


def generate_gaussian_psf(shape, sigma, normalize=True):
    """
    Generate a 3D Gaussian PSF.

    Args:
        shape: (nz, ny, nx) shape of the PSF
        sigma: (sigma_z, sigma_y, sigma_x) standard deviations
        normalize: If True, normalize PSF to sum to 1

    Returns:
        torch.Tensor: Gaussian PSF
    """
    nz, ny, nx = shape
    sigma_z, sigma_y, sigma_x = sigma

    z = torch.arange(nz, device=get_device()) - nz // 2
    y = torch.arange(ny, device=get_device()) - ny // 2
    x = torch.arange(nx, device=get_device()) - nx // 2

    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    psf = torch.exp(
        -0.5 * ((zz / sigma_z) ** 2 + (yy / sigma_y) ** 2 + (xx / sigma_x) ** 2)
    )

    if normalize:
        psf = psf / psf.sum()

    return psf.float()


def generate_born_wolf_psf(shape, wavelength_nm, na, pixel_size_um, z_step_um,
                           n_medium=1.33, normalize=True):
    """
    Generate a theoretical PSF based on the Born-Wolf model.

    Args:
        shape: (nz, ny, nx) shape of the PSF
        wavelength_nm: Emission wavelength in nanometers
        na: Numerical aperture
        pixel_size_um: Lateral pixel size in micrometers
        z_step_um: Axial step size in micrometers
        n_medium: Refractive index of medium (default 1.33 for water)
        normalize: If True, normalize PSF to sum to 1

    Returns:
        torch.Tensor: Theoretical PSF
    """
    nz, ny, nx = shape
    device = get_device()

    wavelength_um = wavelength_nm / 1000.0
    k = 2 * np.pi * n_medium / wavelength_um

    # Lateral coordinates
    y = (torch.arange(ny, device=device) - ny // 2) * pixel_size_um
    x = (torch.arange(nx, device=device) - nx // 2) * pixel_size_um
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    rho = torch.sqrt(xx**2 + yy**2)

    # Axial coordinates
    z = (torch.arange(nz, device=device) - nz // 2) * z_step_um

    # Build PSF slice by slice
    psf = torch.zeros((nz, ny, nx), device=device, dtype=torch.float32)

    alpha = np.arcsin(na / n_medium)

    for i, zi in enumerate(z):
        # Simplified Born-Wolf using Airy pattern approximation
        # For more accurate PSF, use numerical integration of Debye integral
        v = k * rho * np.sin(alpha)
        u = k * zi * (np.sin(alpha) ** 2)

        # Airy pattern (lateral)
        airy = torch.ones_like(v)
        mask = v > 0
        airy[mask] = (2 * torch.special.bessel_j1(v[mask]) / v[mask]) ** 2

        # Defocus (axial)
        defocus = torch.sinc(u / (2 * np.pi)) ** 2

        psf[i] = airy * defocus

    if normalize:
        psf = psf / psf.sum()

    return psf


def compute_otf(psf):
    """
    Compute the Optical Transfer Function from a PSF.

    Args:
        psf: Point Spread Function tensor

    Returns:
        torch.Tensor: OTF (complex)
    """
    return torch.fft.rfftn(psf)


def center_psf(psf):
    """
    Center a PSF so its maximum is at the origin.

    Args:
        psf: PSF tensor

    Returns:
        torch.Tensor: Centered PSF
    """
    # Find the index of maximum value
    max_idx = torch.argmax(psf)
    max_coords = np.unravel_index(max_idx.cpu().item(), psf.shape)

    # Calculate shifts needed
    shifts = [-(coord - size // 2) for coord, size in zip(max_coords, psf.shape)]

    # Roll the PSF
    for dim, shift in enumerate(shifts):
        psf = torch.roll(psf, int(shift), dims=dim)

    return psf
