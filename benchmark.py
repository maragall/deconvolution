#!/usr/bin/env python3
"""
Benchmark: Full-memory vs Chunked deconvolution processing.

Reports:
- Peak GPU memory usage
- Processing time
- Memory efficiency comparison
"""

import time
import torch
from pathlib import Path
from deconvolution import deconvolve, load_tiff, save_tiff, gaussian_psf

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path("/home/cephla/Downloads/20250802_andrea_test_R0/R0")
OUTPUT_DIR = Path("/home/cephla/Downloads/deconvolution/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test parameters
N_SLICES = [5, 10, 20, 50]  # Different dataset sizes to test
ITERATIONS = 30
METHODS = ['richardson_lucy', 'gradient_consensus']

# =============================================================================
# Benchmark Functions
# =============================================================================

def get_memory_mb():
    """Get current GPU memory in MB."""
    return torch.cuda.memory_allocated() / 1024**2

def get_peak_memory_mb():
    """Get peak GPU memory in MB."""
    return torch.cuda.max_memory_allocated() / 1024**2

def benchmark_full_memory(files, psf, method, iterations):
    """Load all data into GPU, process at once."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load
    t0 = time.perf_counter()
    stack = torch.stack([load_tiff(f) for f in files])
    load_time = time.perf_counter() - t0
    mem_after_load = get_memory_mb()

    # Deconvolve
    t1 = time.perf_counter()
    result = deconvolve(stack, psf, iterations=iterations, method=method)
    torch.cuda.synchronize()
    deconv_time = time.perf_counter() - t1

    peak_mem = get_peak_memory_mb()

    del stack, result
    torch.cuda.empty_cache()

    return {
        'load_time': load_time,
        'deconv_time': deconv_time,
        'total_time': load_time + deconv_time,
        'mem_after_load_mb': mem_after_load,
        'peak_memory_mb': peak_mem,
    }

def benchmark_slice_by_slice(files, psf, method, iterations):
    """Process each slice independently (2D deconvolution)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Use center slice of PSF for 2D
    psf_2d = psf[psf.shape[0] // 2] if psf.ndim == 3 else psf

    results = []
    t0 = time.perf_counter()

    for f in files:
        # Load single slice
        slice_data = load_tiff(f)

        # Deconvolve
        result = deconvolve(slice_data, psf_2d, iterations=iterations, method=method)
        results.append(result.cpu())  # Move to CPU immediately

        del slice_data, result

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    peak_mem = get_peak_memory_mb()

    del results
    torch.cuda.empty_cache()

    return {
        'total_time': total_time,
        'peak_memory_mb': peak_mem,
    }

# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("=" * 70)
    print("DECONVOLUTION BENCHMARK")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Data: {DATA_DIR}")
    print(f"Iterations: {ITERATIONS}")
    print()

    # Get all 488nm files
    all_files = sorted(DATA_DIR.glob("*488*.tiff"))
    print(f"Total files available: {len(all_files)}")

    # Check single image size
    test_img = load_tiff(all_files[0])
    print(f"Image size: {test_img.shape} = {test_img.numel() * 4 / 1024**2:.1f} MB per slice")
    del test_img
    torch.cuda.empty_cache()

    # Generate PSF
    psf = gaussian_psf(shape=(5, 21, 21), sigma=(2.0, 1.5, 1.5))
    print(f"PSF shape: {tuple(psf.shape)}")
    print()

    # Results storage
    results = []

    for n_slices in N_SLICES:
        files = all_files[:n_slices]
        print("=" * 70)
        print(f"TESTING WITH {n_slices} SLICES")
        print("=" * 70)

        for method in METHODS:
            method_short = 'RL' if method == 'richardson_lucy' else 'GC'

            # Full memory approach
            print(f"\n[{method_short}] Full-memory loading...")
            try:
                r_full = benchmark_full_memory(files, psf, method, ITERATIONS)
                print(f"  Load time:     {r_full['load_time']:.2f}s")
                print(f"  Deconv time:   {r_full['deconv_time']:.2f}s")
                print(f"  Total time:    {r_full['total_time']:.2f}s")
                print(f"  Mem after load: {r_full['mem_after_load_mb']:.0f} MB")
                print(f"  Peak memory:   {r_full['peak_memory_mb']:.0f} MB")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OUT OF MEMORY!")
                    r_full = {'error': 'OOM', 'peak_memory_mb': float('inf')}
                else:
                    raise

            # Slice-by-slice approach
            print(f"\n[{method_short}] Slice-by-slice processing...")
            r_slice = benchmark_slice_by_slice(files, psf, method, ITERATIONS)
            print(f"  Total time:    {r_slice['total_time']:.2f}s")
            print(f"  Peak memory:   {r_slice['peak_memory_mb']:.0f} MB")

            # Comparison
            if 'error' not in r_full:
                mem_savings = (1 - r_slice['peak_memory_mb'] / r_full['peak_memory_mb']) * 100
                time_diff = (r_slice['total_time'] / r_full['total_time'] - 1) * 100
                print(f"\n  Comparison:")
                print(f"    Memory savings: {mem_savings:.1f}%")
                print(f"    Time overhead:  {time_diff:+.1f}%")

            results.append({
                'n_slices': n_slices,
                'method': method_short,
                'full_memory': r_full,
                'slice_by_slice': r_slice,
            })

    # Summary table
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Slices':<8} {'Method':<6} {'Full Mem (MB)':<14} {'Slice (MB)':<12} {'Full Time':<10} {'Slice Time':<10}")
    print("-" * 70)

    for r in results:
        full_mem = r['full_memory'].get('peak_memory_mb', float('inf'))
        slice_mem = r['slice_by_slice']['peak_memory_mb']
        full_time = r['full_memory'].get('total_time', float('inf'))
        slice_time = r['slice_by_slice']['total_time']

        full_mem_str = f"{full_mem:.0f}" if full_mem < float('inf') else "OOM"
        full_time_str = f"{full_time:.2f}s" if full_time < float('inf') else "N/A"

        print(f"{r['n_slices']:<8} {r['method']:<6} {full_mem_str:<14} {slice_mem:<12.0f} {full_time_str:<10} {slice_time:.2f}s")


if __name__ == '__main__':
    main()
