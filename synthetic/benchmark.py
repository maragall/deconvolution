"""
Synthetic benchmark runner.

Wraps DeconvTest for data generation, runs deconvolution pipeline,
and computes metrics comparing output to ground truth.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import fftconvolve

# Add vendor path for DeconvTest
DECONVTEST_PATH = Path(__file__).parent.parent / "vendor" / "DeconvTest"
if DECONVTEST_PATH.exists():
    sys.path.insert(0, str(DECONVTEST_PATH))

# Import DeconvTest modules directly (avoid __init__.py which needs helper_lib)
try:
    # Import directly from module files to avoid helper_lib dependency
    import importlib.util

    # NumPy 2.0 compatibility: restore removed aliases
    if not hasattr(np, 'int_'):
        np.int_ = np.intp
    if not hasattr(np, 'round_'):
        np.round_ = np.round

    def _import_module_directly(module_path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _input_objects = _import_module_directly(
        DECONVTEST_PATH / "DeconvTest" / "modules" / "input_objects.py",
        "input_objects"
    )
    _psfs = _import_module_directly(
        DECONVTEST_PATH / "DeconvTest" / "modules" / "psfs.py",
        "psfs"
    )
    _noise = _import_module_directly(
        DECONVTEST_PATH / "DeconvTest" / "modules" / "noise.py",
        "noise"
    )

    generate_ellipsoid = _input_objects.generate_ellipsoid
    generate_spiky_cell = _input_objects.generate_spiky_cell
    generate_gaussian_psf = _psfs.gaussian
    add_poisson_noise = _noise.add_poisson_noise
    add_gaussian_noise = _noise.add_gaussian_noise

    HAS_DECONVTEST = True
except Exception as e:
    HAS_DECONVTEST = False
    _DECONVTEST_ERROR = str(e)

# Import our deconvolution pipeline
from ..deconv.rlgc import RLGCDeconvolver
from .metrics import evaluate
from .test_suite import DEFAULT_TEST_CASES, QUICK_TEST_CASES, MISMATCH_TEST_CASES


def _check_deconvtest():
    """Raise error if DeconvTest not available."""
    if not HAS_DECONVTEST:
        raise ImportError(
            f"DeconvTest not available. Error: {_DECONVTEST_ERROR}\n"
            f"Expected at: {DECONVTEST_PATH}"
        )


def generate_phantom(
    phantom_type: str,
    cell_size: Tuple[float, float, float],
    voxel_size: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic phantom using DeconvTest.

    Args:
        phantom_type: "ellipsoid" or "spiky_cell"
        cell_size: (z, y, x) size in micrometers
        voxel_size: Voxel size in micrometers (isotropic)
        seed: Random seed for reproducibility

    Returns:
        3D phantom array
    """
    _check_deconvtest()

    if seed is not None:
        np.random.seed(seed)

    size_z, size_y, size_x = cell_size

    # DeconvTest expects voxel_size as array [z, y, x]
    voxel_size_arr = np.array([voxel_size, voxel_size, voxel_size])

    if phantom_type == "ellipsoid":
        phantom = generate_ellipsoid(
            input_voxel_size=voxel_size_arr,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )
    elif phantom_type == "spiky_cell":
        phantom = generate_spiky_cell(
            input_voxel_size=voxel_size_arr,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
        )
    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")

    return phantom.astype(np.float32)


def generate_psf(
    sigma: float,
    aspect_ratio: float,
) -> np.ndarray:
    """
    Generate a Gaussian PSF using DeconvTest.

    Args:
        sigma: Standard deviation in xy (pixels)
        aspect_ratio: Z/XY elongation ratio

    Returns:
        3D PSF array
    """
    _check_deconvtest()

    psf = generate_gaussian_psf(sigma=sigma, aspect_ratio=aspect_ratio)
    return psf.astype(np.float32)


def crop_psf_to_fit(psf: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Crop PSF to fit within image dimensions if necessary.

    Centers the crop around the PSF maximum.
    """
    cropped = psf.copy()
    for dim in range(3):
        if cropped.shape[dim] > image_shape[dim]:
            excess = cropped.shape[dim] - image_shape[dim]
            start = excess // 2
            end = start + image_shape[dim]
            slices = [slice(None)] * 3
            slices[dim] = slice(start, end)
            cropped = cropped[tuple(slices)]
    return cropped


def convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve image with PSF using FFT.

    Args:
        image: Input image
        psf: Point spread function

    Returns:
        Convolved image, normalized to [0, 255]
    """
    convolved = fftconvolve(image, psf, mode="same")
    # Ensure non-negative (FFT can produce small negative values due to numerical errors)
    convolved = np.clip(convolved, 0, None)
    # Normalize to [0, 255]
    if convolved.max() > 0:
        convolved = convolved / convolved.max() * 255
    return convolved.astype(np.float32)


def add_noise(
    image: np.ndarray,
    snr: float,
    noise_type: str = "poisson",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add noise to image.

    Args:
        image: Input image
        snr: Signal-to-noise ratio
        noise_type: "poisson" or "gaussian"
        seed: Random seed

    Returns:
        Noisy image
    """
    _check_deconvtest()

    if seed is not None:
        np.random.seed(seed)

    if noise_type == "poisson":
        noisy = add_poisson_noise(image, snr=snr)
    elif noise_type == "gaussian":
        noisy = add_gaussian_noise(image, snr=snr)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy.astype(np.float32)


def generate_test_case(
    phantom_type: str = "ellipsoid",
    cell_size: Tuple[float, float, float] = (10, 7, 7),
    voxel_size: float = 0.3,
    psf_sigma: float = 1.0,
    psf_aspect_ratio: float = 3.0,
    snr: float = 20,
    noise_type: str = "poisson",
    seed: int = 42,
    name: Optional[str] = None,
) -> Dict:
    """
    Generate a complete synthetic test case.

    Args:
        phantom_type: "ellipsoid" or "spiky_cell"
        cell_size: (z, y, x) size in micrometers
        voxel_size: Voxel size in micrometers
        psf_sigma: PSF sigma in pixels
        psf_aspect_ratio: PSF z/xy elongation
        snr: Signal-to-noise ratio
        noise_type: "poisson" or "gaussian"
        seed: Random seed for reproducibility
        name: Optional test case name

    Returns:
        Dictionary with ground_truth, psf, synthetic_image, and params
    """
    np.random.seed(seed)

    # Generate ground truth phantom
    ground_truth = generate_phantom(phantom_type, cell_size, voxel_size, seed)

    # Generate PSF
    psf = generate_psf(psf_sigma, psf_aspect_ratio)

    # Forward model: convolve and add noise
    convolved = convolve(ground_truth, psf)
    synthetic_image = add_noise(convolved, snr, noise_type, seed + 1)

    return {
        "name": name or f"{phantom_type}_snr{snr}",
        "ground_truth": ground_truth,
        "psf": psf,
        "synthetic_image": synthetic_image,
        "params": {
            "phantom_type": phantom_type,
            "cell_size": cell_size,
            "voxel_size": voxel_size,
            "psf_sigma": psf_sigma,
            "psf_aspect_ratio": psf_aspect_ratio,
            "snr": snr,
            "noise_type": noise_type,
            "seed": seed,
        },
    }


def run_mismatch_test(
    phantom_type: str = "ellipsoid",
    cell_size: Tuple[float, float, float] = (10, 7, 7),
    voxel_size: float = 0.3,
    true_psf_sigma: float = 1.0,
    assumed_psf_sigma: float = 1.2,
    true_psf_aspect_ratio: float = 3.0,
    assumed_psf_aspect_ratio: float = 3.0,
    snr: float = 20,
    noise_type: str = "poisson",
    seed: int = 42,
    name: Optional[str] = None,
    gpu_id: int = 0,
    verbose: int = 0,
) -> Dict:
    """
    Run a PSF mismatch test.

    Generates synthetic data with one PSF but deconvolves with a different one.

    Args:
        true_psf_sigma: PSF sigma used for data generation
        assumed_psf_sigma: PSF sigma used for deconvolution
        true_psf_aspect_ratio: PSF aspect ratio for generation
        assumed_psf_aspect_ratio: PSF aspect ratio for deconvolution
        ... (other args same as generate_test_case)

    Returns:
        Dictionary with metrics and timing
    """
    np.random.seed(seed)

    # Generate ground truth and synthetic data with TRUE PSF
    ground_truth = generate_phantom(phantom_type, cell_size, voxel_size, seed)
    true_psf = generate_psf(true_psf_sigma, true_psf_aspect_ratio)
    convolved = convolve(ground_truth, true_psf)
    synthetic_image = add_noise(convolved, snr, noise_type, seed + 1)

    # Generate ASSUMED PSF for deconvolution
    assumed_psf = generate_psf(assumed_psf_sigma, assumed_psf_aspect_ratio)

    # Crop PSF to fit image if necessary
    assumed_psf = crop_psf_to_fit(assumed_psf, synthetic_image.shape)

    # Deconvolve with assumed PSF
    deconvolver = RLGCDeconvolver(gpu_id=gpu_id, verbose=verbose)
    start_time = time.time()
    deconvolved = deconvolver.deconvolve(synthetic_image, assumed_psf)
    elapsed = time.time() - start_time

    # Evaluate
    metrics = evaluate(deconvolved, ground_truth)

    return {
        "name": name or f"mismatch_s{true_psf_sigma}_to_{assumed_psf_sigma}",
        "metrics": metrics,
        "elapsed_time": elapsed,
        "params": {
            "phantom_type": phantom_type,
            "true_psf_sigma": true_psf_sigma,
            "assumed_psf_sigma": assumed_psf_sigma,
            "true_psf_aspect_ratio": true_psf_aspect_ratio,
            "assumed_psf_aspect_ratio": assumed_psf_aspect_ratio,
            "snr": snr,
        },
    }


def run_benchmark(
    test_cases: Optional[List[Dict]] = None,
    gpu_id: int = 0,
    verbose: int = 0,
    include_mismatch: bool = False,
) -> List[Dict]:
    """
    Run benchmark on a set of test cases.

    Args:
        test_cases: List of test case configurations. If None, uses DEFAULT_TEST_CASES
        gpu_id: GPU device ID
        verbose: Verbosity level (0=silent, 1=progress)
        include_mismatch: Also run mismatch tests

    Returns:
        List of result dictionaries with name, metrics, and timing
    """
    if test_cases is None:
        test_cases = DEFAULT_TEST_CASES

    results = []
    deconvolver = RLGCDeconvolver(gpu_id=gpu_id, verbose=0)

    for tc in test_cases:
        name = tc.get("name", "unnamed")
        if verbose:
            print(f"  {name}: ", end="", flush=True)

        # Generate test case
        data = generate_test_case(**tc)

        # Crop PSF to fit image if necessary
        psf_for_deconv = crop_psf_to_fit(data["psf"], data["synthetic_image"].shape)

        # Evaluate blurred input (baseline)
        blur_metrics = evaluate(data["synthetic_image"], data["ground_truth"])

        # Deconvolve
        start_time = time.time()
        deconvolved = deconvolver.deconvolve(data["synthetic_image"], psf_for_deconv)
        elapsed = time.time() - start_time

        # Evaluate deconvolved
        metrics = evaluate(deconvolved, data["ground_truth"])

        # Compute improvement
        improvement = metrics["psnr"] - blur_metrics["psnr"]

        if verbose:
            print(
                f"PSNR={metrics['psnr']:.1f} dB ({improvement:+.1f})  "
                f"SSIM={metrics['ssim']:.2f}  "
                f"({elapsed:.1f}s)"
            )

        results.append({
            "name": name,
            "metrics": metrics,
            "blur_metrics": blur_metrics,
            "improvement_db": improvement,
            "elapsed_time": elapsed,
            "params": data["params"],
        })

    # Optionally run mismatch tests
    if include_mismatch:
        if verbose:
            print("\nPSF Mismatch Tests:")
        for tc in MISMATCH_TEST_CASES:
            result = run_mismatch_test(**tc, gpu_id=gpu_id, verbose=0)
            if verbose:
                m = result["metrics"]
                print(
                    f"  {result['name']}: "
                    f"PSNR={m['psnr']:.1f} dB  "
                    f"SSIM={m['ssim']:.2f}  "
                    f"Pearson={m['pearson']:.2f}  "
                    f"({result['elapsed_time']:.1f}s)"
                )
            results.append(result)

    return results


def results_to_dataframe(results: List[Dict]):
    """Convert results list to pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for DataFrame output")

    rows = []
    for r in results:
        row = {"name": r["name"], "elapsed_time": r["elapsed_time"]}
        row.update(r["metrics"])
        rows.append(row)

    return pd.DataFrame(rows)
