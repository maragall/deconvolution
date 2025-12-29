#!/usr/bin/env python3
"""
Command-line interface for GPU-accelerated deconvolution.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated deconvolution with Richardson-Lucy or Gradient Consensus',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument('input', type=str, help='Input TIFF file or directory')
    parser.add_argument('-o', '--output', type=str, help='Output directory (default: input_deconvolved)')
    parser.add_argument('--psf', type=str, help='PSF file (TIFF). If not provided, uses Gaussian approximation.')

    # Deconvolution parameters
    parser.add_argument('-m', '--method', choices=['rl', 'gc', 'both'], default='gc',
                        help='Method: rl=Richardson-Lucy, gc=Gradient Consensus, both=run both')
    parser.add_argument('-i', '--iterations', type=int, default=50,
                        help='Number of iterations')
    parser.add_argument('--background', type=float, default=0.0,
                        help='Background signal level')

    # PSF parameters (if no PSF file provided)
    parser.add_argument('--sigma-z', type=float, default=2.0,
                        help='PSF sigma in Z (pixels) for Gaussian PSF')
    parser.add_argument('--sigma-xy', type=float, default=1.5,
                        help='PSF sigma in XY (pixels) for Gaussian PSF')
    parser.add_argument('--psf-size', type=int, default=31,
                        help='PSF size in pixels (should be odd)')

    # Data selection
    parser.add_argument('--channels', type=str, nargs='+',
                        help='Channels to process (e.g., 488 561)')
    parser.add_argument('--z-range', type=int, nargs=2,
                        help='Z range to process (start end)')

    # Output options
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save intermediate results every N iterations (0=disabled)')
    parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32',
                        help='Computation dtype')

    args = parser.parse_args()

    # Import after arg parsing for faster --help
    from deconvolution import (
        DeconvolutionEngine, PSFConvolution,
        load_tiff_stack, save_tiff_stack,
        load_psf, generate_gaussian_psf
    )
    from deconvolution.utils import get_device

    print(f"Using device: {get_device()}")

    # Setup output directory
    input_path = Path(args.input)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_deconvolved"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {input_path}...")
    z_range = tuple(args.z_range) if args.z_range else None
    data = load_tiff_stack(input_path, channels=args.channels, z_range=z_range)

    # Handle multiple channels
    if isinstance(data, dict):
        channels = data
    else:
        channels = {'default': data}

    # Load or generate PSF
    if args.psf:
        print(f"Loading PSF from {args.psf}...")
        psf = load_psf(args.psf)
    else:
        print(f"Generating Gaussian PSF (sigma_z={args.sigma_z}, sigma_xy={args.sigma_xy})...")
        # Determine PSF shape based on data
        sample = list(channels.values())[0]
        psf_z = min(args.psf_size, sample.shape[0])
        psf_size = args.psf_size
        if psf_z % 2 == 0:
            psf_z += 1
        if psf_size % 2 == 0:
            psf_size += 1

        psf = generate_gaussian_psf(
            shape=(psf_z, psf_size, psf_size),
            sigma=(args.sigma_z, args.sigma_xy, args.sigma_xy)
        )

    print(f"PSF shape: {tuple(psf.shape)}")

    # Determine methods to run
    if args.method == 'both':
        methods = ['richardson_lucy', 'gradient_consensus']
    elif args.method == 'rl':
        methods = ['richardson_lucy']
    else:
        methods = ['gradient_consensus']

    # Process each channel
    for ch_name, ch_data in channels.items():
        print(f"\nProcessing channel: {ch_name}")
        print(f"  Data shape: {tuple(ch_data.shape)}")
        print(f"  Data range: [{ch_data.min():.1f}, {ch_data.max():.1f}]")

        # Setup PSF convolution operator
        psf_op = PSFConvolution(psf, image_shape=tuple(ch_data.shape))
        engine = DeconvolutionEngine(psf_op, background=args.background)

        for method in methods:
            method_short = 'rl' if method == 'richardson_lucy' else 'gc'
            print(f"\n  Running {method} ({args.iterations} iterations)...")

            # Progress callback
            def progress_callback(i, estimate):
                if (i + 1) % 10 == 0:
                    print(f"    Iteration {i + 1}/{args.iterations}")
                if args.save_every > 0 and (i + 1) % args.save_every == 0:
                    interim_path = output_dir / f"{ch_name}_{method_short}_iter{i+1:04d}.tif"
                    save_tiff_stack(interim_path, estimate)

            # Run deconvolution
            result = engine.deconvolve(
                ch_data,
                iterations=args.iterations,
                method=method,
                callback=progress_callback
            )

            # Save result
            output_name = f"{ch_name}_{method_short}.tif" if ch_name != 'default' else f"deconvolved_{method_short}.tif"
            output_path = output_dir / output_name
            print(f"  Saving to {output_path}...")
            save_tiff_stack(output_path, result)

            print(f"  Result range: [{result.min():.1f}, {result.max():.1f}]")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
