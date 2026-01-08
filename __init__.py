"""
Deconv Tool - Minimal Microscopy Deconvolution

Usage:
    from deconv_tool import OpticalParams, deconvolve_tiff

    params = OpticalParams(
        na=0.4,
        wavelength=0.488,  # microns
        dxy=0.276,         # microns
        dz=0.454,          # microns
    )

    deconvolve_tiff(
        input_path="input.tif",
        output_path="output.zarr",
        params=params,
        is_confocal=False,
    )
"""

__version__ = "0.1.0"

from .models import OpticalParams
from .psf import generate_psf, generate_confocal_psf
from .deconv import deconvolve, RLGCDeconvolver
from .pipeline import run_deconvolution, deconvolve_tiff, save_zarr_pyramid

__all__ = [
    "OpticalParams",
    "generate_psf",
    "generate_confocal_psf",
    "deconvolve",
    "RLGCDeconvolver",
    "run_deconvolution",
    "deconvolve_tiff",
    "save_zarr_pyramid",
]
