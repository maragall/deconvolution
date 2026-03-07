"""Core deconvolution function."""

import numpy as np

from . import engine


class PetakitError(RuntimeError):
    """Raised when deconvolution fails."""
    pass


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    method: str = "omw",
    iterations: int | None = None,
    gpu: bool = True,
    verbose: bool = False,
    avail_memory_gb: float | None = None,
) -> np.ndarray:
    """Deconvolve a 3D microscopy image.

    Args:
        image: 3D array (Z, Y, X)
        psf: 3D PSF array, should be normalized (sum to 1)
        method: "omw" (high throughput, default) or "rl" (max resolution)
        iterations: Number of iterations (default: 2 for omw, 15 for rl)
        gpu: Try GPU, fall back to CPU (default True)
        verbose: Print progress
        avail_memory_gb: Override available memory (GB) for tiling decisions.
            None means auto-detect.

    Returns:
        Deconvolved image as float32 (Z, Y, X)
    """
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D (Z,Y,X), got shape {image.shape}")
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D (Z,Y,X), got shape {psf.shape}")

    if method == "omw":
        return engine.omw(
            image, psf,
            n_iter=iterations if iterations is not None else 2,
            gpu=gpu, verbose=verbose,
            avail_memory_gb=avail_memory_gb,
        )
    elif method == "rl":
        return engine.rl(
            image, psf,
            n_iter=iterations if iterations is not None else 15,
            gpu=gpu, verbose=verbose,
            avail_memory_gb=avail_memory_gb,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'omw' or 'rl'.")
