#!/usr/bin/env python3
"""
Benchmark comparing full-memory vs chunked deconvolution processing.

Measures:
- Peak GPU memory usage
- Total processing time
- Memory efficiency
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse


def benchmark_full_memory(data_path, n_slices, iterations, psf):
    """Load all data into GPU memory and process at once."""
    from deconvolution import (
        DeconvolutionEngine, PSFConvolution,
        load_tiff_stack
    )
    from deconvolution.utils import reset_peak_memory, get_gpu_memory_info

    reset_peak_memory()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Load all data into GPU memory
    print(f"  Loading {n_slices} slices into GPU memory...")
    data = load_tiff_stack(data_path, channels=['488'], z_range=(0, n_slices))

    load_time = time.perf_counter() - start_time
    mem_after_load = get_gpu_memory_info()
    print(f"  Load time: {load_time:.2f}s")
    print(f"  GPU memory after load: {mem_after_load['allocated']:.0f} MB")

    # Setup deconvolution
    psf_op = PSFConvolution(psf, image_shape=tuple(data.shape))
    engine = DeconvolutionEngine(psf_op, background=0)

    # Run deconvolution
    print(f"  Running {iterations} iterations...")
    deconv_start = time.perf_counter()
    result = engine.deconvolve(data, iterations=iterations, method='richardson_lucy')
    torch.cuda.synchronize()
    deconv_time = time.perf_counter() - deconv_start

    total_time = time.perf_counter() - start_time
    peak_memory = get_gpu_memory_info()['max_allocated']

    # Cleanup
    del data, result, engine, psf_op
    torch.cuda.empty_cache()

    return {
        'load_time': load_time,
        'deconv_time': deconv_time,
        'total_time': total_time,
        'peak_memory_mb': peak_memory,
    }


def benchmark_chunked(data_path, n_slices, iterations, psf, chunk_size):
    """Process data in chunks with streaming I/O."""
    from deconvolution import ChunkedDeconvolver, open_tiff_stack
    from deconvolution.utils import reset_peak_memory, get_gpu_memory_info
    import tempfile

    reset_peak_memory()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Open lazy stack (no GPU memory used yet)
    print(f"  Opening lazy stack...")
    stack = open_tiff_stack(data_path, channels=['488'])

    # Limit to n_slices for fair comparison
    class SliceLimitedStack:
        def __init__(self, stack, limit):
            self._stack = stack
            self._limit = limit

        def __len__(self):
            return min(len(self._stack), self._limit)

        def __getitem__(self, idx):
            return self._stack[idx]

        def load_chunk(self, start, end):
            end = min(end, self._limit)
            return self._stack.load_chunk(start, end)

        @property
        def shape(self):
            s = self._stack.shape
            return (min(s[0], self._limit), s[1], s[2])

    limited_stack = SliceLimitedStack(stack, n_slices)

    load_time = time.perf_counter() - start_time
    mem_after_open = get_gpu_memory_info()
    print(f"  Open time: {load_time:.4f}s (lazy, minimal memory)")
    print(f"  GPU memory after open: {mem_after_open['allocated']:.0f} MB")

    # Create chunked processor
    deconvolver = ChunkedDeconvolver(
        psf=psf,
        chunk_size=chunk_size,
        background=0
    )

    # Run chunked deconvolution
    print(f"  Running chunked processing (chunk_size={chunk_size})...")
    deconv_start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix='.tif', delete=True) as tmp:
        def progress(chunk_idx, total, z_start, z_end):
            print(f"    Chunk {chunk_idx+1}/{total}: Z[{z_start}:{z_end}]")

        deconvolver.process_stack(
            limited_stack,
            tmp.name,
            iterations=iterations,
            method='richardson_lucy',
            progress_callback=progress
        )

    torch.cuda.synchronize()
    deconv_time = time.perf_counter() - deconv_start
    total_time = time.perf_counter() - start_time
    peak_memory = get_gpu_memory_info()['max_allocated']

    # Cleanup
    torch.cuda.empty_cache()

    return {
        'load_time': load_time,
        'deconv_time': deconv_time,
        'total_time': total_time,
        'peak_memory_mb': peak_memory,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark deconvolution approaches')
    parser.add_argument('input', help='Input directory with TIFF slices')
    parser.add_argument('--slices', type=int, default=20, help='Number of Z slices to process')
    parser.add_argument('--iterations', type=int, default=20, help='Deconvolution iterations')
    parser.add_argument('--chunk-size', type=int, default=10, help='Chunk size for chunked processing')
    parser.add_argument('--skip-full', action='store_true', help='Skip full-memory benchmark')

    args = parser.parse_args()

    from deconvolution import generate_gaussian_psf
    from deconvolution.utils import get_device

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"\nBenchmark parameters:")
    print(f"  Data: {args.input}")
    print(f"  Slices: {args.slices}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Chunk size: {args.chunk_size}")

    # Generate PSF
    psf = generate_gaussian_psf(
        shape=(11, 21, 21),
        sigma=(2.0, 1.5, 1.5)
    )
    print(f"  PSF shape: {tuple(psf.shape)}")

    results = {}

    # Benchmark full-memory approach
    if not args.skip_full:
        print(f"\n{'='*60}")
        print("FULL-MEMORY APPROACH")
        print('='*60)
        try:
            results['full'] = benchmark_full_memory(
                args.input, args.slices, args.iterations, psf
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  OUT OF MEMORY! Cannot fit {args.slices} slices in GPU memory.")
                results['full'] = {'error': 'OOM'}
            else:
                raise

    # Benchmark chunked approach
    print(f"\n{'='*60}")
    print("CHUNKED APPROACH")
    print('='*60)
    results['chunked'] = benchmark_chunked(
        args.input, args.slices, args.iterations, psf, args.chunk_size
    )

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print('='*60)

    if 'full' in results and 'error' not in results['full']:
        print(f"\nFull-memory approach:")
        print(f"  Peak GPU memory: {results['full']['peak_memory_mb']:.0f} MB")
        print(f"  Load time:       {results['full']['load_time']:.2f}s")
        print(f"  Deconv time:     {results['full']['deconv_time']:.2f}s")
        print(f"  Total time:      {results['full']['total_time']:.2f}s")
    elif 'full' in results:
        print(f"\nFull-memory approach: OUT OF MEMORY")

    print(f"\nChunked approach (chunk_size={args.chunk_size}):")
    print(f"  Peak GPU memory: {results['chunked']['peak_memory_mb']:.0f} MB")
    print(f"  Load time:       {results['chunked']['load_time']:.4f}s (lazy)")
    print(f"  Deconv time:     {results['chunked']['deconv_time']:.2f}s")
    print(f"  Total time:      {results['chunked']['total_time']:.2f}s")

    if 'full' in results and 'error' not in results['full']:
        memory_savings = (1 - results['chunked']['peak_memory_mb'] / results['full']['peak_memory_mb']) * 100
        time_overhead = (results['chunked']['total_time'] / results['full']['total_time'] - 1) * 100
        print(f"\nComparison:")
        print(f"  Memory savings: {memory_savings:.1f}%")
        print(f"  Time overhead:  {time_overhead:+.1f}%")


if __name__ == '__main__':
    main()
