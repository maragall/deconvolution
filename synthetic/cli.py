"""
Command-line interface for synthetic benchmarks.

Usage:
    python -m deconv_tool.synthetic.cli              # Run all tests
    python -m deconv_tool.synthetic.cli --quick      # Quick check (2 tests)
    python -m deconv_tool.synthetic.cli --mismatch   # Include PSF mismatch tests
"""

import argparse
import sys
from pathlib import Path

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deconv_tool.synthetic.benchmark import (
    run_benchmark,
    run_quick_test,
    STANDARD_TESTS,
    QUICK_TESTS,
    MISMATCH_TESTS,
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

    verbose = not args.quiet

    # Quick test mode
    if args.quick:
        print("Running quick test...")
        success = run_quick_test(gpu_id=args.gpu, verbose=verbose)
        sys.exit(0 if success else 1)

    # Select test cases
    if args.cases:
        all_cases = STANDARD_TESTS + MISMATCH_TESTS
        test_cases = [tc for tc in all_cases if tc["name"] in args.cases]
        if not test_cases:
            print(f"No matching test cases found for: {args.cases}")
            print("Available cases:")
            for tc in all_cases:
                print(f"  - {tc['name']}")
            sys.exit(1)
    else:
        test_cases = STANDARD_TESTS

    # Run benchmark
    results = run_benchmark(
        test_configs=test_cases,
        include_mismatch=args.mismatch,
        gpu_id=args.gpu,
        verbose=verbose,
    )

    # Check if all passed
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
