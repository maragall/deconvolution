"""
Core data models for the deconvolution pipeline.

No hardcoded defaults - all parameters must be explicitly provided.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class OpticalParams:
    """
    Optical parameters for PSF generation.

    All parameters must be explicitly provided.
    Units: wavelength, dxy, dz in microns.
    """
    na: float          # Numerical aperture
    wavelength: float  # Emission wavelength in microns
    dxy: float         # Pixel size in microns
    dz: float          # Z step in microns
    ni: float = 1.0    # Immersion refractive index (1.0 for air, 1.515 for oil)


@dataclass
class DeconvolutionInput:
    """Input for deconvolution: image + optical parameters."""
    image: np.ndarray      # 3D image (Z, Y, X)
    params: OpticalParams  # Optical parameters for PSF generation
    is_confocal: bool = False  # True for confocal (PSF squared)
