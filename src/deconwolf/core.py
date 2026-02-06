"""Core deconvolution function."""
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from .binary import find_binary


def _get_library_env(dw_path: str) -> dict:
    """Build environment with library paths for subprocess.

    Handles platform-specific library path variables:
    - Linux: LD_LIBRARY_PATH
    - Windows: PATH
    """
    import sys
    env = os.environ.copy()
    system = platform.system().lower()

    # Common library locations (priority order)
    lib_paths = [
        Path(dw_path).parent / "lib",      # Next to binary (bundled)
        Path(dw_path).parent,               # Same dir as binary
        Path.home() / ".local" / "lib",     # User local (Linux)
    ]

    # Add conda environment lib path if running in conda
    conda_prefix = Path(sys.prefix)
    conda_lib = conda_prefix / "lib"
    if conda_lib.exists():
        lib_paths.append(conda_lib)

    # Find conda root (navigate up from envs/name or use directly)
    if "envs" in conda_prefix.parts:
        conda_root = conda_prefix.parent.parent
    else:
        conda_root = conda_prefix

    pkgs_dir = conda_root / "pkgs"
    if pkgs_dir.exists():
        # Find jpeg-9* package for libjpeg.so.9
        for pkg in pkgs_dir.glob("jpeg-9*/lib"):
            if pkg.exists():
                lib_paths.append(pkg)
                break

    if system == "linux":
        lib_paths.append(Path("/usr/local/lib"))
        key = "LD_LIBRARY_PATH"
        separator = ":"
    elif system == "windows":
        key = "PATH"
        separator = ";"
    else:
        return env

    # Filter to existing directories
    existing_paths = [str(p) for p in lib_paths if p.exists()]

    if not existing_paths:
        return env

    current = env.get(key, "")
    env[key] = separator.join(existing_paths + ([current] if current else []))

    return env


class DeconwolfError(RuntimeError):
    """Raised when deconwolf execution fails."""
    def __init__(self, message: str, returncode: int, stderr: str):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)


def deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    relerror: float = 0.02,
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
        relerror: Convergence threshold (default 0.02). Stop when relative
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
        >>> result = deconvolve(image, psf, relerror=0.02, maxiter=100)
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

        # Execute with library paths set for the subprocess
        env = _get_library_env(dw)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

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
