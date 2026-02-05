"""Deconwolf - Microscopy deconvolution.

Simple API for 3D fluorescence microscopy deconvolution using deconwolf.

Example:
    >>> from deconwolf import deconvolve, generate_psf, open_acquisition
    >>>
    >>> # Open acquisition and get metadata
    >>> acq = open_acquisition("/path/to/data")
    >>> meta = acq.metadata
    >>>
    >>> # Generate PSF from metadata
    >>> psf = generate_psf(
    ...     nz=31, nxy=31,
    ...     dxy=meta.dxy, dz=meta.dz,
    ...     wavelength=0.525, na=meta.na
    ... )
    >>>
    >>> # Deconvolve each FOV
    >>> for fov in acq.iter_fovs():
    ...     stack = acq.get_stack(fov, channel="488")
    ...     result = deconvolve(stack, psf)
"""
from .core import deconvolve
from .psf import generate_psf
from .readers import open_acquisition, Metadata, FOV, AcquisitionReader

__all__ = [
    "deconvolve",
    "generate_psf",
    "open_acquisition",
    "Metadata",
    "FOV",
    "AcquisitionReader",
]
__version__ = "0.1.0"
