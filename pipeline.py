"""
Minimal Deconvolution Pipeline

Input: TIFF + optical parameters (manually provided)
Output: Zarr pyramid
"""
from pathlib import Path
import numpy as np
import tifffile
import zarr
from numcodecs import Blosc

from .models import OpticalParams
from .psf import generate_psf, generate_confocal_psf
from .deconv import RLGCDeconvolver


def load_tiff(path: str | Path) -> np.ndarray:
    """Load a TIFF file as numpy array."""
    return tifffile.imread(path)


def save_zarr_pyramid(
    data: np.ndarray,
    output_path: str | Path,
    num_levels: int = 4,
    chunk_size: tuple = (32, 256, 256),
) -> None:
    """
    Save data as zarr pyramid with multiple resolution levels.

    Args:
        data: 3D array (Z, Y, X)
        output_path: Output .zarr path
        num_levels: Number of pyramid levels
        chunk_size: Chunk size for zarr storage
    """
    output_path = Path(output_path)
    compressor = Blosc(cname='zstd', clevel=3)

    # Create zarr group
    root = zarr.open_group(output_path, mode='w')

    # Full resolution
    root.create_dataset(
        '0',
        data=data,
        chunks=chunk_size,
        compressor=compressor,
        dtype=data.dtype,
    )

    # Downsampled levels
    current = data
    for level in range(1, num_levels):
        # Downsample by 2 in Y and X (keep Z)
        if current.shape[1] > 1 and current.shape[2] > 1:
            current = current[:, ::2, ::2]
            root.create_dataset(
                str(level),
                data=current,
                chunks=chunk_size,
                compressor=compressor,
                dtype=current.dtype,
            )
        else:
            break


def run_deconvolution(
    image: np.ndarray,
    params: OpticalParams,
    is_confocal: bool = False,
    psf_nx: int = 101,
    gpu_id: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run deconvolution on a 3D image.

    Args:
        image: 3D image array (Z, Y, X)
        params: Optical parameters (na, wavelength, dxy, dz, ni)
        is_confocal: True for confocal microscopy
        psf_nx: PSF lateral size in pixels
        gpu_id: GPU device ID
        verbose: Print progress

    Returns:
        Tuple of (deconvolved_image, psf)
    """
    if verbose:
        print(f"Image shape: {image.shape}")
        print(f"Parameters: NA={params.na}, λ={params.wavelength}µm, dxy={params.dxy}µm, dz={params.dz}µm")

    # Generate PSF
    nz = image.shape[0]
    if is_confocal:
        if verbose:
            print("Generating confocal PSF...")
        psf = generate_confocal_psf(
            nz=nz,
            nx=psf_nx,
            dxy=params.dxy,
            dz=params.dz,
            wavelength=params.wavelength,
            na=params.na,
            ni=params.ni,
        )
    else:
        if verbose:
            print("Generating widefield PSF...")
        psf = generate_psf(
            nz=nz,
            nx=psf_nx,
            dxy=params.dxy,
            dz=params.dz,
            wavelength=params.wavelength,
            na=params.na,
            ni=params.ni,
        )

    if verbose:
        print(f"PSF shape: {psf.shape}")

    # Run deconvolution
    if verbose:
        print("Running RLGC deconvolution...")

    deconvolver = RLGCDeconvolver(
        gpu_id=gpu_id,
        verbose=1 if verbose else 0,
    )
    result = deconvolver.deconvolve(image, psf)

    if verbose:
        print(f"Done. Result range: [{result.min():.1f}, {result.max():.1f}]")

    return result, psf


def deconvolve_tiff(
    input_path: str | Path,
    output_path: str | Path,
    params: OpticalParams,
    is_confocal: bool = False,
    save_pyramid: bool = True,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Deconvolve a TIFF file and save result.

    Args:
        input_path: Path to input TIFF
        output_path: Path for output (.tif or .zarr)
        params: Optical parameters
        is_confocal: True for confocal microscopy
        save_pyramid: If output is .zarr, save as pyramid
        gpu_id: GPU device ID

    Returns:
        Deconvolved image array
    """
    print(f"Loading: {input_path}")
    image = load_tiff(input_path)

    result, psf = run_deconvolution(
        image=image,
        params=params,
        is_confocal=is_confocal,
        gpu_id=gpu_id,
    )

    output_path = Path(output_path)
    if output_path.suffix == '.zarr':
        print(f"Saving zarr pyramid: {output_path}")
        save_zarr_pyramid(result, output_path)
    else:
        print(f"Saving TIFF: {output_path}")
        tifffile.imwrite(output_path, result)

    return result
