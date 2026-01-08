"""
PSF Generator - thin wrapper around psfmodels.

All optical parameters must be explicitly provided.
"""
import sys
from pathlib import Path

import numpy as np

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
    nx: int,
    dxy: float,
    dz: float,
    wavelength: float,
    na: float,
    ni: float = 1.0,
) -> np.ndarray:
    """
    Generate a theoretical PSF using psfmodels.

    Args:
        nz: Number of Z planes
        nx: XY size in pixels (will be made odd if even)
        dxy: Pixel size in microns
        dz: Z step size in microns
        wavelength: Emission wavelength in microns
        na: Numerical aperture
        ni: Immersion medium refractive index (default 1.0 for air)

    Returns:
        3D PSF array (nz, nx, nx), normalized, centered
    """
    if not HAS_PSFMODELS:
        raise ImportError("psfmodels not installed. Install with: pip install psfmodels")

    # Ensure odd dimensions (required by psfmodels for centering)
    nx = nx if nx % 2 == 1 else nx + 1

    psf = psfm.make_psf(
        z=nz,
        nx=nx,
        dxy=dxy,
        dz=dz,
        wvl=wavelength,
        NA=na,
        ni=ni,
        ni0=ni,
        model='vectorial'
    )

    return psf.astype(np.float32)


def generate_confocal_psf(
    nz: int,
    nx: int,
    dxy: float,
    dz: float,
    wavelength: float,
    na: float,
    ni: float = 1.0,
) -> np.ndarray:
    """
    Generate a confocal PSF (widefield PSF squared).

    Args:
        Same as generate_psf

    Returns:
        3D confocal PSF array, normalized to max=1
    """
    psf = generate_psf(nz, nx, dxy, dz, wavelength, na, ni)
    psf_confocal = psf ** 2
    psf_confocal = psf_confocal / psf_confocal.max()
    return psf_confocal.astype(np.float32)
