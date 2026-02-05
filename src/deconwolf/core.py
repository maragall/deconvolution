"""Core deconvolution function."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from .binary import find_binary


class DeconwolfError(RuntimeError):
    """Raised when deconwolf execution fails."""
    def __init__(self, message: str, returncode: int, stderr: str):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    relerror: float = 0.001,
    maxiter: int = 200,
    iterations: int | None = None,
    method: str = "shb",
    tilesize: int | None = None,
    threads: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Deconvolve a 3D image using deconwolf.

    Deconwolf uses adaptive stopping by default (relerror + maxiter).
    Set `iterations` to use fixed iteration count instead.

    Args:
        image: 3D array (Z, Y, X)
        psf: 3D PSF array, should be normalized (sum to 1)
        relerror: Convergence threshold (default 0.001). Stop when relative
                  change between iterations falls below this value.
        maxiter: Maximum iterations for adaptive stopping (default 200)
        iterations: If set, use fixed iteration count instead of adaptive
        method: Algorithm - "shb" (default), "rl", "shbcl2" (GPU)
        tilesize: Tile size for large images (memory efficiency)
        threads: Number of CPU threads (default: auto)
        verbose: Print deconwolf output

    Returns:
        Deconvolved image as float32 (Z, Y, X)

    Raises:
        ValueError: Invalid input dimensions
        DeconwolfError: Deconwolf execution failed

    Example:
        >>> result = deconvolve(image, psf, relerror=0.001, maxiter=100)
    """
    # Validate inputs
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D (Z,Y,X), got shape {image.shape}")
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D (Z,Y,X), got shape {psf.shape}")

    dw = find_binary()

    with tempfile.TemporaryDirectory(prefix="deconwolf_") as tmpdir:
        tmp = Path(tmpdir)
        img_path = tmp / "image.tif"
        psf_path = tmp / "psf.tif"
        out_path = tmp / "output.tif"

        # Write inputs as float32 TIFF
        tifffile.imwrite(img_path, image.astype(np.float32), imagej=True)
        tifffile.imwrite(psf_path, psf.astype(np.float32), imagej=True)

        # Build command
        cmd = [dw]

        # Iteration control
        if iterations is not None:
            cmd.extend(["--iter", str(iterations)])
        else:
            cmd.extend(["--relerror", str(relerror)])
            cmd.extend(["--maxiter", str(maxiter)])

        # Method and output format
        cmd.extend(["--method", method])
        cmd.append("--float")
        cmd.append("--overwrite")
        cmd.extend(["--out", str(out_path)])

        # Optional parameters
        if tilesize is not None:
            cmd.extend(["--tilesize", str(tilesize)])
        if threads is not None:
            cmd.extend(["--threads", str(threads)])

        # Input files
        cmd.extend([str(img_path), str(psf_path)])

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        # Execute
        result = subprocess.run(cmd, capture_output=True, text=True)

        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

        if result.returncode != 0:
            raise DeconwolfError(
                f"deconwolf failed with code {result.returncode}",
                returncode=result.returncode,
                stderr=result.stderr,
            )

        return tifffile.imread(out_path).astype(np.float32)
