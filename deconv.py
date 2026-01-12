"""RLGC Deconvolution - GPU only."""
import sys
from pathlib import Path

import cupy as cp
import numpy as np

OPM_PATH = Path(__file__).parent.parent / "opm-processing-v2" / "src"
if OPM_PATH.exists():
    sys.path.insert(0, str(OPM_PATH))

from opm_processing.imageprocessing.rlgc import chunked_rlgc


def _auto_chunk(gpu_id: int, psf_shape: tuple, image_shape: tuple) -> tuple:
    """Auto-detect chunk/overlap from GPU memory, PSF shape, and image shape."""
    cp.cuda.Device(gpu_id).use()
    free_mem, _ = cp.cuda.Device(gpu_id).mem_info

    # ~60 bytes/voxel (10 arrays × 4 bytes × 1.5 FFT overhead)
    cube = int((free_mem * 0.5 / 60) ** (1/3))
    cube = max((cube // 32) * 32, 64)

    # Overlap >= PSF/2, but ensure 2*overlap < crop_size
    oz, oy, ox = [max(16, s // 2 + 1) for s in psf_shape]
    cz = max(cube, psf_shape[0] + oz * 2)
    cy = max(cube, psf_shape[1] + oy * 2)
    cx = max(cube, psf_shape[2] + ox * 2)

    # If image dimension is small, don't chunk that axis
    if image_shape[0] <= cz:
        cz, oz = image_shape[0], 0
    if image_shape[1] <= cy:
        cy, oy = image_shape[1], 0
    if image_shape[2] <= cx:
        cx, ox = image_shape[2], 0

    return cz, cy, cx, oz, oy, ox


def deconvolve(image: np.ndarray, psf: np.ndarray, gpu_id: int = 0) -> np.ndarray:
    """
    Deconvolve a 3D image with the given PSF.

    Args:
        image: 3D array (Z, Y, X)
        psf: 3D PSF array (Z, Y, X)
        gpu_id: GPU device (default: 0)

    Returns:
        Deconvolved image (float32)
    """
    image = image.astype(np.float32)
    psf = psf.astype(np.float32)

    cz, cy, cx, oz, oy, ox = _auto_chunk(gpu_id, psf.shape, image.shape)

    # Scale to ~10k photons for binomial splitting
    scale = 10000.0 / image.max() if image.max() > 0 else 1.0
    scaled = image * scale

    # Background for sparse images
    bg = scaled.max() * 0.01 if np.median(scaled) == 0 else 0.0
    if bg > 0:
        scaled = scaled + bg

    result = chunked_rlgc(
        image=scaled, psf=psf, gpu_id=gpu_id,
        crop_z=cz, crop_y=cy, crop_x=cx,
        overlap_z=oz, overlap_y=oy, overlap_x=ox,
        safe_mode=True, verbose=0,
    )

    if bg > 0:
        result = np.clip(result - bg, 0, None)
    return (result / scale).astype(np.float32)
