"""
GPU-accelerated deconvolution with Richardson-Lucy and Gradient Consensus methods.
"""

from .core import DeconvolutionEngine, ChunkedDeconvolver
from .operators import Crop3D, PSFConvolution
from .psf import load_psf, generate_gaussian_psf
from .io import (
    load_tiff_stack, save_tiff_stack,
    open_tiff_stack, get_stack_info,
    LazyTiffStack, TiffStackWriter
)

__version__ = "0.1.0"
__all__ = [
    "DeconvolutionEngine",
    "ChunkedDeconvolver",
    "Crop3D",
    "PSFConvolution",
    "load_psf",
    "generate_gaussian_psf",
    "load_tiff_stack",
    "save_tiff_stack",
    "open_tiff_stack",
    "get_stack_info",
    "LazyTiffStack",
    "TiffStackWriter",
]
