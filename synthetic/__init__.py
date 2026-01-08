"""
Synthetic benchmark suite for deconvolution validation.

Wraps DeconvTest for phantom generation and provides metrics
for evaluating deconvolution quality against ground truth.
"""

from .benchmark import generate_test_case, run_benchmark, run_mismatch_test
from .metrics import evaluate, compute_psnr, compute_ssim
from .test_suite import DEFAULT_TEST_CASES, QUICK_TEST_CASES, MISMATCH_TEST_CASES

__all__ = [
    "generate_test_case",
    "run_benchmark",
    "run_mismatch_test",
    "evaluate",
    "compute_psnr",
    "compute_ssim",
    "DEFAULT_TEST_CASES",
    "QUICK_TEST_CASES",
    "MISMATCH_TEST_CASES",
]
