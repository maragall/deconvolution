"""
Command-line interface for synthetic benchmarks.

Usage:
    python -m synthetic.cli              # Run all tests
    python -m synthetic.cli --quick      # Quick check (2 tests)
    python -m synthetic.cli --mismatch   # Include PSF mismatch tests
    python -m synthetic.cli --output results.csv  # Save results
"""

import argparse
import sys
from pathlib import Path

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deconv_tool.synthetic.benchmark import run_benchmark, results_to_dataframe
from deconv_tool.synthetic.test_suite import (
    DEFAULT_TEST_CASES,
    QUICK_TEST_CASES,
    MISMATCH_TEST_CASES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic deconvolution benchmarks"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test suite (2 tests)",
    )
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="Include PSF mismatch tests",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Run specific test cases by name",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Select test cases
    if args.quick:
        test_cases = QUICK_TEST_CASES
    elif args.cases:
        # Filter to specified cases
        all_cases = DEFAULT_TEST_CASES + MISMATCH_TEST_CASES
        test_cases = [tc for tc in all_cases if tc["name"] in args.cases]
        if not test_cases:
            print(f"No matching test cases found for: {args.cases}")
            print("Available cases:")
            for tc in all_cases:
                print(f"  - {tc['name']}")
            sys.exit(1)
    else:
        test_cases = DEFAULT_TEST_CASES

    verbose = 0 if args.quiet else 1

    if verbose:
        print(f"Running benchmark ({len(test_cases)} test cases)...")
        if args.mismatch:
            print(f"  + {len(MISMATCH_TEST_CASES)} mismatch tests")
        print()

    # Run benchmark
    results = run_benchmark(
        test_cases=test_cases,
        gpu_id=args.gpu,
        verbose=verbose,
        include_mismatch=args.mismatch,
    )

    # Save results if requested
    if args.output:
        try:
            df = results_to_dataframe(results)
            df.to_csv(args.output, index=False)
            if verbose:
                print(f"\nResults saved to: {args.output}")
        except ImportError:
            print("Warning: pandas not available, cannot save CSV")

    # Print summary
    if verbose and results:
        print("\nSummary:")
        avg_psnr = sum(r["metrics"]["psnr"] for r in results) / len(results)
        avg_ssim = sum(r["metrics"]["ssim"] for r in results) / len(results)
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        print(f"  Avg PSNR: {avg_psnr:.1f} dB")
        print(f"  Avg SSIM: {avg_ssim:.3f}")
        print(f"  Avg Time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
