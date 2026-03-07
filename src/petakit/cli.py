"""PetaKit CLI - Command-line interface for batch deconvolution."""
import argparse
from pathlib import Path

import tifffile

from .readers import open_acquisition
from .psf import (
    generate_psf, wavelength_from_channel, compute_psf_size, infer_immersion_index,
)
from .core import deconvolve
from .engine import gpu_info


def run_batch(acq_path, channel, output_dir=None, method="omw",
              iterations=None, use_gpu=True, verbose=False):
    """Run batch deconvolution on an acquisition directory.

    Args:
        acq_path: Path to acquisition folder.
        channel: Channel to process (e.g., "488").
        output_dir: Output directory (default: {acq_path}/deconvolved).
        method: "omw" or "rl".
        iterations: Number of iterations (None = method default).
        use_gpu: Try GPU acceleration.
        verbose: Print detailed output.

    Raises:
        ValueError: If channel is not found in acquisition metadata.
    """
    acq_path = Path(acq_path)
    print(gpu_info())

    print(f"Loading acquisition: {acq_path}")
    acq = open_acquisition(acq_path)
    meta = acq.metadata

    print(f"  Format: {acq.format_name}")
    print(f"  Channels: {meta.channels}")
    print(f"  Metadata: NA={meta.na}, dxy={meta.dxy:.3f}um, dz={meta.dz:.1f}um")

    if channel not in meta.channels:
        raise ValueError(
            f"Channel '{channel}' not found. Available: {meta.channels}"
        )

    output_dir = Path(output_dir) if output_dir else (acq_path / "deconvolved")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    wavelength = wavelength_from_channel(f"Fluorescence {channel} nm Ex")
    ni = infer_immersion_index(meta.na)

    nz_psf, nxy_psf = compute_psf_size(
        meta.nz, dxy=meta.dxy, dz=meta.dz,
        wavelength=wavelength, na=meta.na, ni=ni,
    )
    print(f"\nGenerating PSF (wl={wavelength*1000:.0f}nm, NA={meta.na}, "
          f"ni={ni}, size={nz_psf}x{nxy_psf}x{nxy_psf})")

    psf = generate_psf(
        nz=nz_psf, nxy=nxy_psf,
        dxy=meta.dxy, dz=meta.dz,
        wavelength=wavelength, na=meta.na, ni=ni,
    )

    print(f"\nMethod: {method}, iterations: "
          f"{iterations or ('2' if method == 'omw' else '15')}")

    fovs = list(acq.iter_fovs())
    print(f"Processing {len(fovs)} FOVs...")

    for i, fov in enumerate(fovs, 1):
        print(f"  [{i}/{len(fovs)}] {fov}...", end=" ", flush=True)

        stack = acq.get_stack(fov, channel)
        result = deconvolve(
            stack, psf,
            method=method,
            iterations=iterations,
            gpu=use_gpu,
            verbose=verbose,
        )

        out_path = output_dir / f"{fov}_ch{channel}_deconv.tiff"
        tifffile.imwrite(out_path, result, imagej=True)
        print("done")

    print(f"\nComplete! Results saved to: {output_dir}")


def _run_check():
    """Print install health and GPU status, then exit."""
    import numpy, scipy, tifffile
    print(f"petakit  : installed")
    print(f"numpy    : {numpy.__version__}")
    print(f"scipy    : {scipy.__version__}")
    print(f"tifffile : {tifffile.__version__}")
    try:
        import cupy
        print(f"cupy     : {cupy.__version__}")
    except Exception:
        print("cupy     : not installed")
    print(gpu_info())
    print("\nReady to deconvolve.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deconvolve microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OMW deconvolution (recommended, 2 iterations)
  petakit /path/to/acquisition --channel 488

  # RL deconvolution (max resolution, 15 iterations)
  petakit /path/to/acquisition --channel 488 --method rl

  # Force CPU (no GPU)
  petakit /path/to/acquisition --channel 488 --no-gpu
""",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Print install health and GPU status, then exit",
    )
    parser.add_argument(
        "acquisition",
        nargs="?",
        type=Path,
        help="Path to acquisition folder",
    )
    parser.add_argument(
        "--channel", "-c",
        default=None,
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

    if args.check:
        _run_check()
        return

    if args.acquisition is None:
        parser.error("the following arguments are required: acquisition")
    if args.channel is None:
        parser.error("the following arguments are required: --channel/-c")

    try:
        run_batch(
            acq_path=args.acquisition,
            channel=args.channel,
            output_dir=args.output,
            method=args.method,
            iterations=args.iterations,
            use_gpu=not args.no_gpu,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
