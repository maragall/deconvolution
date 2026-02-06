"""Deconwolf - Microscopy deconvolution.

Simple API for 3D fluorescence microscopy deconvolution using deconwolf.

Example:
    >>> from deconwolf import deconvolve, generate_psf, compute_psf_size, open_acquisition
    >>>
    >>> # Open acquisition and get metadata
    >>> acq = open_acquisition("/path/to/data")
    >>> meta = acq.metadata
    >>>
    >>> # Compute PSF dimensions from optical parameters
    >>> nz_psf, nxy_psf = compute_psf_size(
    ...     meta.nz, meta.dxy, meta.dz,
    ...     wavelength=0.525, na=meta.na, ni=1.0
    ... )
    >>>
    >>> # Generate PSF
    >>> psf = generate_psf(
    ...     nz=nz_psf, nxy=nxy_psf,
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
from .psf import compute_psf_size, generate_psf
from .readers import open_acquisition, Metadata, FOV, AcquisitionReader

__all__ = [
    "compute_psf_size",
    "deconvolve",
    "generate_psf",
    "open_acquisition",
    "Metadata",
    "FOV",
    "AcquisitionReader",
]
__version__ = "0.1.0"
