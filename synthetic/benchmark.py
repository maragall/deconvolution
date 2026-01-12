"""
Synthetic benchmark for deconvolution validation.

Uses:
- Continuous phantoms (Gaussian blobs, not binary)
- psfmodels for realistic PSF generation
- Proper forward model with Poisson noise
- Comprehensive test cases including PSF mismatch
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .phantoms import gaussian_blob, multi_blob, bead_phantom, cell_phantom
from .forward_model import generate_psf, simulate_microscopy, perturb_psf
from .metrics import evaluate
from ..deconv import deconvolve as rlgc_deconvolve


# Default optical parameters (realistic widefield microscopy)
DEFAULT_OPTICS = {
    "wavelength": 0.525,  # Green emission (microns)
    "na": 0.8,            # Moderate NA objective
    "ni": 1.0,            # Air immersion
    "dxy": 0.1,           # 100nm pixels
    "dz": 0.3,            # 300nm Z step
}


def create_test_case(
    phantom_type: str = "multi_blob",
    phantom_shape: Tuple[int, int, int] = (64, 128, 128),
    psf_size: Tuple[int, int] = (31, 31),  # (nz, nxy)
    snr: float = 20,
    noise_type: str = "poisson",
    optics: Dict = None,
    psf_error: Dict = None,  # {"sigma_error": 0.1, "z_error": 0.1}
    seed: int = 42,
    name: str = None,
) -> Dict:
    """
    Create a complete test case for benchmarking.

    Args:
        phantom_type: "gaussian_blob", "multi_blob", "beads", "cell"
        phantom_shape: Volume dimensions (z, y, x)
        psf_size: PSF dimensions (nz, nxy)
        snr: Signal-to-noise ratio
        noise_type: "poisson" or "gaussian"
        optics: Optical parameters (uses DEFAULT_OPTICS if None)
        psf_error: PSF mismatch parameters for testing robustness
        seed: Random seed
        name: Test case name

    Returns:
        Dictionary with ground_truth, psf, blurred, noisy, deconv_psf, params
    """
    if optics is None:
        optics = DEFAULT_OPTICS.copy()

    np.random.seed(seed)

    # Generate phantom
    if phantom_type == "gaussian_blob":
        ground_truth = gaussian_blob(
            phantom_shape,
            sigma=(8, 12, 12),
            intensity=1.0
        )
    elif phantom_type == "multi_blob":
        ground_truth = multi_blob(
            phantom_shape,
            n_blobs=5,
            sigma_range=(4, 10),
            seed=seed
        )
    elif phantom_type == "beads":
        ground_truth = bead_phantom(
            phantom_shape,
            n_beads=15,
            bead_sigma=1.5,
            seed=seed
        )
    elif phantom_type == "cell":
        ground_truth = cell_phantom(
            phantom_shape,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")

    # Generate TRUE PSF (for forward model)
    true_psf = generate_psf(
        nz=psf_size[0],
        nxy=psf_size[1],
        dz=optics["dz"],
        dxy=optics["dxy"],
        wavelength=optics["wavelength"],
        na=optics["na"],
        ni=optics["ni"],
    )

    # Generate PSF for deconvolution (may have error)
    if psf_error:
        deconv_psf = perturb_psf(
            true_psf,
            sigma_error=psf_error.get("sigma_error", 0),
            z_error=psf_error.get("z_error", 0),
        )
    else:
        deconv_psf = true_psf.copy()

    # Forward model: blur
    from .forward_model import convolve
    blurred = convolve(ground_truth, true_psf)

    # Add noise
    noisy = simulate_microscopy(
        ground_truth,
        true_psf,
        snr=snr,
        noise_type=noise_type,
        seed=seed + 1,
    )

    return {
        "name": name or f"{phantom_type}_snr{snr}",
        "ground_truth": ground_truth,
        "true_psf": true_psf,
        "deconv_psf": deconv_psf,
        "blurred": blurred,
        "noisy": noisy,
        "params": {
            "phantom_type": phantom_type,
            "phantom_shape": phantom_shape,
            "psf_size": psf_size,
            "snr": snr,
            "noise_type": noise_type,
            "optics": optics,
            "psf_error": psf_error,
            "seed": seed,
        },
    }


# Standard test cases
STANDARD_TESTS = [
    # Basic phantoms at SNR=20
    {"name": "blob_snr20", "phantom_type": "gaussian_blob", "snr": 20},
    {"name": "multi_snr20", "phantom_type": "multi_blob", "snr": 20},
    {"name": "beads_snr20", "phantom_type": "beads", "snr": 20},
    {"name": "cell_snr20", "phantom_type": "cell", "snr": 20},

    # SNR sweep
    {"name": "multi_snr50", "phantom_type": "multi_blob", "snr": 50},
    {"name": "multi_snr10", "phantom_type": "multi_blob", "snr": 10},
    {"name": "multi_snr5", "phantom_type": "multi_blob", "snr": 5},

    # No noise (ideal case - MUST improve)
    {"name": "multi_nonoise", "phantom_type": "multi_blob", "snr": 1000},
]

# PSF mismatch tests
MISMATCH_TESTS = [
    # 10% error (should still work well)
    {"name": "mismatch_10pct", "phantom_type": "multi_blob", "snr": 20,
     "psf_error": {"sigma_error": 0.1, "z_error": 0.1}},

    # 20% error (should still improve)
    {"name": "mismatch_20pct", "phantom_type": "multi_blob", "snr": 20,
     "psf_error": {"sigma_error": 0.2, "z_error": 0.2}},

    # 50% error (stress test)
    {"name": "mismatch_50pct", "phantom_type": "multi_blob", "snr": 20,
     "psf_error": {"sigma_error": 0.5, "z_error": 0.5}},
]

# Quick test for fast validation
QUICK_TESTS = [
    {"name": "quick_blob", "phantom_type": "gaussian_blob", "snr": 20,
     "phantom_shape": (32, 64, 64), "psf_size": (21, 21)},
    {"name": "quick_nonoise", "phantom_type": "gaussian_blob", "snr": 1000,
     "phantom_shape": (32, 64, 64), "psf_size": (21, 21)},
]


def run_single_test(
    test_config: Dict,
    gpu_id: int = 0,
    verbose: bool = False,
) -> Dict:
    """Run a single test case and return results."""
    data = create_test_case(**test_config)
    name = data["name"]

    if verbose:
        print(f"  {name}: ", end="", flush=True)

    blur_metrics = evaluate(data["blurred"], data["ground_truth"])
    noisy_metrics = evaluate(data["noisy"], data["ground_truth"])

    start_time = time.time()
    deconvolved = rlgc_deconvolve(data["noisy"], data["deconv_psf"], gpu_id=gpu_id)
    elapsed = time.time() - start_time

    deconv_metrics = evaluate(deconvolved, data["ground_truth"])
    improvement_vs_blur = deconv_metrics["psnr"] - blur_metrics["psnr"]
    improvement_vs_noisy = deconv_metrics["psnr"] - noisy_metrics["psnr"]

    if verbose:
        status = "OK" if improvement_vs_blur > 0 else "FAIL"
        print(
            f"PSNR={deconv_metrics['psnr']:.1f}dB "
            f"(vs blur: {improvement_vs_blur:+.1f}, vs noisy: {improvement_vs_noisy:+.1f}) "
            f"SSIM={deconv_metrics['ssim']:.3f} "
            f"[{status}] ({elapsed:.1f}s)"
        )

    return {
        "name": name,
        "blur_metrics": blur_metrics,
        "noisy_metrics": noisy_metrics,
        "deconv_metrics": deconv_metrics,
        "improvement_vs_blur": improvement_vs_blur,
        "improvement_vs_noisy": improvement_vs_noisy,
        "elapsed_time": elapsed,
        "params": data["params"],
        "passed": improvement_vs_blur > 0,
    }


def run_benchmark(
    test_configs: List[Dict] = None,
    include_mismatch: bool = False,
    gpu_id: int = 0,
    verbose: bool = True,
) -> List[Dict]:
    """Run complete benchmark suite."""
    if test_configs is None:
        test_configs = STANDARD_TESTS.copy()

    if include_mismatch:
        test_configs = test_configs + MISMATCH_TESTS

    results = []

    if verbose:
        print(f"Running {len(test_configs)} tests...")
        print()

    for config in test_configs:
        result = run_single_test(config, gpu_id=gpu_id, verbose=verbose)
        results.append(result)

    # Summary
    if verbose:
        print()
        print("=" * 60)
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        avg_improvement = np.mean([r["improvement_vs_blur"] for r in results])
        avg_ssim = np.mean([r["deconv_metrics"]["ssim"] for r in results])

        print(f"PASSED: {passed}/{total}")
        print(f"Avg improvement vs blur: {avg_improvement:+.2f} dB")
        print(f"Avg SSIM: {avg_ssim:.3f}")

        if passed < total:
            print()
            print("FAILED tests:")
            for r in results:
                if not r["passed"]:
                    print(f"  - {r['name']}: {r['improvement_vs_blur']:+.2f} dB")

    return results


def run_quick_test(gpu_id: int = 0, verbose: bool = True) -> bool:
    """
    Run quick validation test.

    Returns True if all tests pass.
    """
    results = run_benchmark(QUICK_TESTS, gpu_id=gpu_id, verbose=verbose)
    return all(r["passed"] for r in results)


# Keep backward compatibility
def generate_test_case(**kwargs):
    """Backward compatibility wrapper."""
    return create_test_case(**kwargs)


DEFAULT_TEST_CASES = STANDARD_TESTS
QUICK_TEST_CASES = QUICK_TESTS
MISMATCH_TEST_CASES = MISMATCH_TESTS
