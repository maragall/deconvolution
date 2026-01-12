"""PSF generation - thin wrapper around psfmodels."""
import numpy as np

try:
    import psfmodels as psfm
except ImportError:
    raise ImportError("psfmodels required: pip install psfmodels")


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
    Generate a theoretical PSF.

    Args:
        nz: Number of Z planes
        nxy: XY size in pixels (will be made odd)
        dxy: Pixel size in microns
        dz: Z step in microns
        wavelength: Emission wavelength in microns
        na: Numerical aperture
        ni: Immersion refractive index (1.0=air, 1.515=oil)

    Returns:
        3D PSF array (nz, nxy, nxy), float32
    """
    nxy = nxy if nxy % 2 == 1 else nxy + 1
    psf = psfm.make_psf(
        z=nz, nx=nxy, dxy=dxy, dz=dz,
        wvl=wavelength, NA=na, ni=ni, ni0=ni,
        model='vectorial'
    )
    psf = psf / psf.sum()  # Normalize to sum=1
    return psf.astype(np.float32)
