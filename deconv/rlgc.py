"""
RLGC Deconvolution Adapter

Thin wrapper around opm-processing-v2 RLGC implementation.
Does NOT modify PSF - assumes PSF is properly formed (centered, no background).
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add opm-processing to path if available
OPM_PROCESSING_PATH = Path(__file__).parent.parent / "vendor" / "opm-processing-v2" / "src"
if OPM_PROCESSING_PATH.exists():
    sys.path.insert(0, str(OPM_PROCESSING_PATH))

try:
    from opm_processing.imageprocessing.rlgc import chunked_rlgc
    HAS_RLGC = True
except ImportError:
    HAS_RLGC = False


class RLGCDeconvolver:
    """
    Richardson-Lucy Gradient Consensus Deconvolver.

    Thin wrapper around opm-processing-v2 RLGC implementation.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        crop_z: int = 128,
        overlap_z: int = 32,
        safe_mode: bool = True,
        verbose: int = 1,
    ):
        if not HAS_RLGC:
            raise ImportError(
                "opm-processing-v2 not found. "
                f"Expected at: {OPM_PROCESSING_PATH}"
            )

        self.gpu_id = gpu_id
        self.crop_z = crop_z
        self.overlap_z = overlap_z
        self.safe_mode = safe_mode
        self.verbose = verbose

    def deconvolve(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        crop_z: Optional[int] = None,
        overlap_z: Optional[int] = None,
    ) -> np.ndarray:
        """
        Deconvolve an image with the given PSF.

        Args:
            image: 3D image array (Z, Y, X)
            psf: 3D PSF array (Z, Y, X) - must be centered with max at geometric center
            crop_z: Override default crop_z
            overlap_z: Override default overlap_z

        Returns:
            Deconvolved image as float32 array
        """
        image = image.astype(np.float32)
        psf = psf.astype(np.float32)

        cz = crop_z if crop_z is not None else self.crop_z
        oz = overlap_z if overlap_z is not None else self.overlap_z

        # Ensure PSF fits in crop size
        if cz < psf.shape[0]:
            cz = psf.shape[0]

        result = chunked_rlgc(
            image=image,
            psf=psf,
            gpu_id=self.gpu_id,
            crop_z=cz,
            overlap_z=oz,
            safe_mode=self.safe_mode,
            verbose=self.verbose,
        )

        return result.astype(np.float32)


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    gpu_id: int = 0,
    crop_z: int = 128,
    overlap_z: int = 32,
    safe_mode: bool = True,
    verbose: int = 1,
) -> np.ndarray:
    """Convenience function for one-off deconvolution."""
    deconvolver = RLGCDeconvolver(
        gpu_id=gpu_id,
        crop_z=crop_z,
        overlap_z=overlap_z,
        safe_mode=safe_mode,
        verbose=verbose,
    )
    return deconvolver.deconvolve(image, psf)
