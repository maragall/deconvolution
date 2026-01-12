"""
deconv_tool - 3D Microscopy Deconvolution

Usage:
    # Low-level API
    from deconv_tool import deconvolve, generate_psf
    psf = generate_psf(nz=31, nxy=31, dxy=0.1, dz=0.3, wavelength=0.525, na=0.8)
    result = deconvolve(image, psf)

    # High-level API with Squid acquisition
    from deconv_tool import open_acquisition
    acq = open_acquisition("/path/to/squid/data")
    for fov in acq.iter_fovs():
        stack = acq.get_stack(fov, channel="488")
"""
from .deconv import deconvolve
from .psf import generate_psf
from .readers import open_acquisition, SquidAcquisition, SquidMetadata

__all__ = [
    "deconvolve",
    "generate_psf",
    "open_acquisition",
    "SquidAcquisition",
    "SquidMetadata",
]
