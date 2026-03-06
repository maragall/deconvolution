"""Deconwolf CLI - Command-line interface for batch deconvolution."""
import argparse
from pathlib import Path

import tifffile

from .readers import open_acquisition
from .psf import generate_psf, wavelength_from_channel
from .core import deconvolve
from .engine import gpu_info


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deconvolve microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OMW deconvolution (recommended, 2 iterations)
  deconwolf /path/to/acquisition --channel 488

  # RL deconvolution (max resolution, 15 iterations)
  deconwolf /path/to/acquisition --channel 488 --method rl

  # Force CPU (no GPU)
  deconwolf /path/to/acquisition --channel 488 --no-gpu
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
        "--method",
        choices=["omw", "rl"],
        default="omw",
        help="Algorithm: omw (high throughput, default) or rl (max resolution)",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=None,
        help="Number of iterations (default: 2 for omw, 15 for rl)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU computation (GPU is used by default if available)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # GPU status
    use_gpu = not args.no_gpu
    print(gpu_info())

    # Open acquisition
    print(f"Loading acquisition: {args.acquisition}")
    acq = open_acquisition(args.acquisition)
    meta = acq.metadata

    print(f"  Format: {acq.format_name}")
    print(f"  Channels: {meta.channels}")
    print(f"  Metadata: NA={meta.na}, dxy={meta.dxy:.3f}um, dz={meta.dz:.1f}um")

    # Set output directory
    output_dir = args.output or (args.acquisition / "deconvolved")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    # Generate PSF
    wavelength = wavelength_from_channel(f"Fluorescence {args.channel} nm Ex")
    print(f"\nGenerating PSF (wl={wavelength*1000:.0f}nm, NA={meta.na})")

    psf = generate_psf(
        nz=31, nxy=31,
        dxy=meta.dxy, dz=meta.dz,
        wavelength=wavelength, na=meta.na,
    )

    # Process FOVs
    n_iter = args.iterations
    method = args.method
    print(f"\nMethod: {method}, iterations: {n_iter or ('2' if method == 'omw' else '15')}")

    fovs = list(acq.iter_fovs())
    print(f"Processing {len(fovs)} FOVs...")

    for i, fov in enumerate(fovs, 1):
        print(f"  [{i}/{len(fovs)}] {fov}...", end=" ", flush=True)

        stack = acq.get_stack(fov, args.channel)
        result = deconvolve(
            stack, psf,
            method=method,
            iterations=n_iter,
            gpu=use_gpu,
            verbose=args.verbose,
        )

        out_path = output_dir / f"{fov}_ch{args.channel}_deconv.tiff"
        tifffile.imwrite(out_path, result, imagej=True)
        print("done")

    print(f"\nComplete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
