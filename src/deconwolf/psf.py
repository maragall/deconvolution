"""PSF generation for microscopy deconvolution."""
import numpy as np

try:
    import psfmodels as psfm
    HAS_PSFMODELS = True
except ImportError:
    HAS_PSFMODELS = False


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
