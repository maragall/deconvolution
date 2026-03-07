"""PetaKit — Microscopy deconvolution with PetaKit5D algorithms.

GPU-first via CuPy, CPU fallback via NumPy/SciPy.

Example:
    >>> from petakit import deconvolve, generate_psf, compute_psf_size, open_acquisition
    >>>
    >>> acq = open_acquisition("/path/to/data")
    >>> meta = acq.metadata
    >>>
    >>> nz_psf, nxy_psf = compute_psf_size(
    ...     meta.nz, meta.dxy, meta.dz,
    ...     wavelength=0.525, na=meta.na, ni=1.0
    ... )
    >>> psf = generate_psf(
    ...     nz=nz_psf, nxy=nxy_psf,
    ...     dxy=meta.dxy, dz=meta.dz,
    ...     wavelength=0.525, na=meta.na
    ... )
    >>>
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
__version__ = "0.0.0"
