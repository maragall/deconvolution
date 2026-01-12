"""
Synthetic benchmark suite for deconvolution validation.

Uses continuous phantoms and psfmodels for realistic PSF generation.
"""

from .benchmark import (
    create_test_case,
    run_benchmark,
    run_quick_test,
    run_single_test,
    STANDARD_TESTS,
    QUICK_TESTS,
    MISMATCH_TESTS,
    DEFAULT_TEST_CASES,
    QUICK_TEST_CASES,
    MISMATCH_TEST_CASES,
)
from .metrics import evaluate, compute_psnr, compute_ssim
from .phantoms import gaussian_blob, multi_blob, bead_phantom, cell_phantom
from .forward_model import generate_psf, simulate_microscopy, convolve

__all__ = [
    "create_test_case",
    "run_benchmark",
    "run_quick_test",
    "run_single_test",
    "evaluate",
    "compute_psnr",
    "compute_ssim",
    "generate_psf",
    "simulate_microscopy",
    "convolve",
    "gaussian_blob",
    "multi_blob",
    "bead_phantom",
    "cell_phantom",
    "STANDARD_TESTS",
    "QUICK_TESTS",
    "MISMATCH_TESTS",
    "DEFAULT_TEST_CASES",
    "QUICK_TEST_CASES",
    "MISMATCH_TEST_CASES",
]
