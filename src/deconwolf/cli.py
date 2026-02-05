"""Deconwolf CLI - Command-line interface for batch deconvolution."""
import argparse
from pathlib import Path

import tifffile

from .readers import open_acquisition
from .psf import generate_psf, wavelength_from_channel
from .core import deconvolve


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deconvolve microscopy images using deconwolf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all FOVs for channel 488
  deconwolf /path/to/acquisition --channel 488

  # Process with custom parameters
  deconwolf /path/to/acquisition --channel 488 --relerror 0.01 --maxiter 100

  # Use GPU acceleration
  deconwolf /path/to/acquisition --channel 488 --method shbcl2
""",
    )

    parser.add_argument(
        "acquisition",
        type=Path,
        help="Path to acquisition folder",
    )
    parser.add_argument(
        "--channel", "-c",
        required=True,
        help="Channel to process (e.g., 488)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (default: {acquisition}/deconvolved)",
    )
    parser.add_argument(
        "--relerror",
        type=float,
        default=0.001,
        help="Convergence threshold (default: 0.001)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum iterations (default: 200)",
    )
    parser.add_argument(
        "--method",
        choices=["shb", "rl", "shbcl2"],
        default="shb",
        help="Algorithm (default: shb, use shbcl2 for GPU)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Open acquisition
    print(f"Loading acquisition: {args.acquisition}")
    acq = open_acquisition(args.acquisition)
    meta = acq.metadata

    print(f"  Format: {acq.format_name}")
    print(f"  Channels: {meta.channels}")
    print(f"  Metadata: NA={meta.na}, dxy={meta.dxy:.3f}µm, dz={meta.dz:.1f}µm")

    # Set output directory
    output_dir = args.output or (args.acquisition / "deconvolved")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    # Generate PSF
    wavelength = wavelength_from_channel(f"Fluorescence {args.channel} nm Ex")
    print(f"\nGenerating PSF (λ={wavelength*1000:.0f}nm, NA={meta.na})")

    psf = generate_psf(
        nz=31, nxy=31,
        dxy=meta.dxy, dz=meta.dz,
        wavelength=wavelength, na=meta.na,
    )

    # Process FOVs
    fovs = list(acq.iter_fovs())
    print(f"\nProcessing {len(fovs)} FOVs...")

    for i, fov in enumerate(fovs, 1):
        print(f"  [{i}/{len(fovs)}] {fov}...", end=" ", flush=True)

        stack = acq.get_stack(fov, args.channel)
        result = deconvolve(
            stack, psf,
            relerror=args.relerror,
            maxiter=args.maxiter,
            method=args.method,
            verbose=args.verbose,
        )

        out_path = output_dir / f"{fov}_ch{args.channel}_deconv.tiff"
        tifffile.imwrite(out_path, result, imagej=True)
        print("done")

    print(f"\nComplete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
