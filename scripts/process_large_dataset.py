#!/usr/bin/env python3
"""
Process large datasets using chunked/streaming deconvolution.

This script demonstrates memory-efficient processing that works on CPU
without loading the entire dataset into memory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from deconvolution import (
    ChunkedDeconvolver,
    open_tiff_stack, get_stack_info,
    generate_gaussian_psf, load_psf
)
from deconvolution.utils import get_device


def main():
    parser = argparse.ArgumentParser(
        description='Process large datasets with chunked deconvolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input', help='Input directory with TIFF slices')
    parser.add_argument('-o', '--output', help='Output TIFF file path')
    parser.add_argument('--channel', default='488', help='Channel to process')

    parser.add_argument('-m', '--method', choices=['rl', 'gc'], default='rl',
                        help='Method: rl=Richardson-Lucy, gc=Gradient Consensus')
    parser.add_argument('-i', '--iterations', type=int, default=30,
                        help='Iterations per chunk')
    parser.add_argument('--chunk-size', type=int, default=20,
                        help='Z-slices per chunk')
    parser.add_argument('--mode', choices=['3d', '2d'], default='2d',
                        help='3D chunked or 2D slice-by-slice processing')

    parser.add_argument('--psf', help='PSF file (TIFF)')
    parser.add_argument('--sigma-z', type=float, default=2.0, help='PSF sigma Z')
    parser.add_argument('--sigma-xy', type=float, default=1.5, help='PSF sigma XY')
    parser.add_argument('--background', type=float, default=0.0, help='Background level')

    args = parser.parse_args()

    print(f"Device: {get_device()}")

    # Get dataset info without loading
    input_path = Path(args.input)
    print(f"\nScanning dataset: {input_path}")
    info = get_stack_info(input_path, channels=[args.channel])
    print(f"  Shape: {info['shape']}")
    print(f"  Files: {info['num_files']}")

    # Open lazy stack
    stack = open_tiff_stack(input_path, channels=[args.channel])
    print(f"  Lazy stack ready: {len(stack)} slices")

    # Setup output
    if args.output:
        output_path = Path(args.output)
    else:
        method_str = 'rl' if args.method == 'rl' else 'gc'
        output_path = input_path.parent / f"{input_path.name}_{args.channel}_{method_str}.tif"

    # Load or generate PSF
    if args.psf:
        print(f"\nLoading PSF from {args.psf}")
        psf = load_psf(args.psf)
    else:
        print(f"\nGenerating Gaussian PSF (sigma_z={args.sigma_z}, sigma_xy={args.sigma_xy})")
        psf_z = 11 if args.mode == '3d' else 1
        psf = generate_gaussian_psf(
            shape=(psf_z, 21, 21),
            sigma=(args.sigma_z, args.sigma_xy, args.sigma_xy)
        )
    print(f"  PSF shape: {tuple(psf.shape)}")

    # Create chunked processor
    deconvolver = ChunkedDeconvolver(
        psf=psf,
        chunk_size=args.chunk_size,
        background=args.background
    )

    method = 'richardson_lucy' if args.method == 'rl' else 'gradient_consensus'

    # Progress callback
    def progress_3d(chunk_idx, total_chunks, z_start, z_end):
        print(f"  Chunk {chunk_idx + 1}/{total_chunks}: Z [{z_start}, {z_end})")

    def progress_2d(slice_idx, total_slices):
        if (slice_idx + 1) % 10 == 0 or slice_idx == 0:
            print(f"  Slice {slice_idx + 1}/{total_slices}")

    # Process
    print(f"\nProcessing with {args.mode.upper()} mode, {args.iterations} iterations per {'chunk' if args.mode == '3d' else 'slice'}")
    print(f"Output: {output_path}")

    if args.mode == '3d':
        deconvolver.process_stack(
            stack, output_path,
            iterations=args.iterations,
            method=method,
            progress_callback=progress_3d
        )
    else:
        deconvolver.process_2d_slices(
            stack, output_path,
            iterations=args.iterations,
            method=method,
            progress_callback=progress_2d
        )

    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
