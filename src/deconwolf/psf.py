"""PSF generation for microscopy deconvolution."""
import math

import numpy as np

try:
    import psfmodels as psfm
    HAS_PSFMODELS = True
except ImportError:
    HAS_PSFMODELS = False


def compute_psf_size(nz_acquisition, dxy, dz, wavelength, na, ni=1.0):
    """Compute PSF dimensions from optical parameters.

    Lateral: 6 Airy radii each side (~97% encircled energy).
    Axial: max of deconwolf heuristic (2*Nz-1) and optics-based extent
           (6x axial FWHM each side from focus).

    Args:
        nz_acquisition: Number of z-planes in the acquired stack
        dxy: Pixel size in µm
        dz: Z-step size in µm
        wavelength: Emission wavelength in µm
        na: Numerical aperture
        ni: Immersion medium refractive index (default 1.0 for air)

    Returns:
        (nz_psf, nxy_psf) — both odd integers
    """
    # Lateral: Rayleigh criterion, 6x Airy radius each side
    r_airy_px = 0.61 * wavelength / na / dxy
    nxy = 2 * math.ceil(6 * r_airy_px) + 1

    # Axial: optics-based with signal-processing heuristic as floor
    axial_fwhm_um = 2 * ni * wavelength / (na ** 2)
    nz_optical = 2 * math.ceil(6 * axial_fwhm_um / dz) + 1
    nz_signal = 2 * nz_acquisition - 1
    nz = max(nz_optical, nz_signal)

    # Ensure odd
    if nz % 2 == 0:
        nz += 1
    if nxy % 2 == 0:
        nxy += 1

    return nz, nxy


def generate_psf(
    nz: int,
    nxy: int,
    dxy: float,
    dz: float,
    wavelength: float,
    na: float,
    ni: float = 1.0,
) -> np.ndarray:
    """
    Generate a theoretical PSF using the vectorial model.

    Args:
        nz: Number of z-planes (should be odd)
        nxy: Lateral size in pixels (should be odd)
        dxy: Pixel size in µm
        dz: Z-step size in µm
        wavelength: Emission wavelength in µm (e.g., 0.525 for green)
        na: Numerical aperture
        ni: Refractive index of immersion medium (1.0=air, 1.515=oil)

    Returns:
        PSF array (Z, Y, X), normalized to sum=1, float32

    Example:
        >>> # 20x air objective, 488nm excitation (~525nm emission)
        >>> psf = generate_psf(
        ...     nz=31, nxy=31,
        ...     dxy=0.752, dz=1.5,
        ...     wavelength=0.525, na=0.8
        ... )
    """
    if not HAS_PSFMODELS:
        raise ImportError(
            "psfmodels required for PSF generation. "
            "Install with: pip install psfmodels"
        )

    # Ensure odd sizes for centered PSF
    nxy = nxy if nxy % 2 == 1 else nxy + 1
    nz = nz if nz % 2 == 1 else nz + 1

    psf = psfm.make_psf(
        z=nz,
        nx=nxy,
        dxy=dxy,
        dz=dz,
        wvl=wavelength,
        NA=na,
        ni=ni,
        ni0=ni,
        model="vectorial",
    )

    # Normalize to sum=1
    psf = psf / psf.sum()

    return psf.astype(np.float32)


def wavelength_from_channel(channel_name: str) -> float:
    """
    Estimate emission wavelength from channel name.

    Common fluorophores and their approximate emission peaks:
        405nm excitation -> ~450nm emission (DAPI)
        488nm excitation -> ~525nm emission (GFP, FITC)
        561nm excitation -> ~590nm emission (RFP, mCherry)
        638nm excitation -> ~670nm emission (Cy5, far red)

    Args:
        channel_name: Channel name like "Fluorescence 488 nm Ex"

    Returns:
        Estimated emission wavelength in µm
    """
    # Extract excitation wavelength from name
    import re
    match = re.search(r"(\d{3})\s*nm", channel_name)
    if not match:
        raise ValueError(f"Cannot parse wavelength from: {channel_name}")

    excitation = int(match.group(1))

    # Approximate Stokes shift (emission is red-shifted from excitation)
    emission_map = {
        405: 450,   # DAPI
        488: 525,   # GFP
        561: 590,   # RFP
        638: 670,   # Cy5
        730: 780,   # IR
    }

    # Find closest match or estimate
    if excitation in emission_map:
        emission = emission_map[excitation]
    else:
        # Rough estimate: ~50nm Stokes shift
        emission = excitation + 50

    return emission / 1000  # Convert nm to µm
