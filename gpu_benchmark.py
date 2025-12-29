#!/usr/bin/env python3
"""
GPU Deconvolution Benchmark - Memory, Timing, and Visualization.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from deconvolution import (
    DeconvolutionEngine, ChunkedDeconvolver,
    open_tiff_stack, generate_gaussian_psf, save_tiff_stack,
    PSFConvolution
)

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path("/home/cephla/Downloads/20250802_andrea_test_R0/R0")
OUTPUT_DIR = Path("/home/cephla/Downloads/deconvolution/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test parameters
N_SLICES_LIST = [5, 10, 20]
ITERATIONS = 30
METHODS = ['richardson_lucy', 'gradient_consensus']


def get_memory_stats():
    """Get GPU memory statistics."""
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'peak_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
    }


def reset_memory():
    """Reset GPU memory tracking."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_full_stack(stack, psf, method, iterations, n_slices):
    """Benchmark loading full stack into GPU and processing."""
    reset_memory()

    # Load data
    t0 = time.perf_counter()
    data = stack.load_chunk(0, n_slices)
    load_time = time.perf_counter() - t0
    mem_after_load = get_memory_stats()

    # Create engine
    psf_op = PSFConvolution(psf, image_shape=tuple(data.shape))
    engine = DeconvolutionEngine(psf_op)

    # Deconvolve
    t1 = time.perf_counter()
    result = engine.deconvolve(data, iterations=iterations, method=method)
    torch.cuda.synchronize()
    deconv_time = time.perf_counter() - t1

    peak_mem = get_memory_stats()['peak_mb']

    # Cleanup
    result_cpu = result.cpu().numpy()
    del data, result, engine
    reset_memory()

    return {
        'load_time': load_time,
        'deconv_time': deconv_time,
        'total_time': load_time + deconv_time,
        'mem_after_load_mb': mem_after_load['allocated_mb'],
        'peak_memory_mb': peak_mem,
        'result': result_cpu,
    }


def benchmark_slice_by_slice(stack, psf, method, iterations, n_slices):
    """Benchmark processing each slice independently."""
    reset_memory()

    # Use center slice of PSF for 2D
    psf_2d = psf[psf.shape[0] // 2]

    results = []
    t0 = time.perf_counter()

    for i in range(n_slices):
        # Load single slice
        slice_data = torch.from_numpy(stack[i].astype(np.float32)).cuda()

        # Create engine for this slice
        psf_op = PSFConvolution(psf_2d, image_shape=tuple(slice_data.shape))
        engine = DeconvolutionEngine(psf_op)

        # Deconvolve
        result = engine.deconvolve(slice_data, iterations=iterations, method=method)
        results.append(result.cpu().numpy())

        del slice_data, result, engine

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    peak_mem = get_memory_stats()['peak_mb']

    result_stack = np.stack(results, axis=0)
    del results
    reset_memory()

    return {
        'total_time': total_time,
        'peak_memory_mb': peak_mem,
        'result': result_stack,
    }


def visualize_results(original, deconvolved, method, output_path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Middle slice for visualization
    mid_z = original.shape[0] // 2

    # Original
    orig_slice = original[mid_z]
    vmin, vmax = np.percentile(orig_slice, [1, 99])

    axes[0, 0].imshow(orig_slice, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Original (slice {mid_z})')
    axes[0, 0].axis('off')

    # Deconvolved
    deconv_slice = deconvolved[mid_z]
    vmin_d, vmax_d = np.percentile(deconv_slice, [1, 99])

    axes[0, 1].imshow(deconv_slice, cmap='gray', vmin=vmin_d, vmax=vmax_d)
    axes[0, 1].set_title(f'Deconvolved - {method}')
    axes[0, 1].axis('off')

    # Difference (normalized)
    diff = deconv_slice - orig_slice
    vmax_diff = np.percentile(np.abs(diff), 99)
    axes[0, 2].imshow(diff, cmap='RdBu', vmin=-vmax_diff, vmax=vmax_diff)
    axes[0, 2].set_title('Difference')
    axes[0, 2].axis('off')

    # Line profiles
    mid_y = orig_slice.shape[0] // 2
    axes[1, 0].plot(orig_slice[mid_y, :], label='Original', alpha=0.7)
    axes[1, 0].plot(deconv_slice[mid_y, :], label='Deconvolved', alpha=0.7)
    axes[1, 0].set_title('Horizontal Profile')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Pixel')
    axes[1, 0].set_ylabel('Intensity')

    mid_x = orig_slice.shape[1] // 2
    axes[1, 1].plot(orig_slice[:, mid_x], label='Original', alpha=0.7)
    axes[1, 1].plot(deconv_slice[:, mid_x], label='Deconvolved', alpha=0.7)
    axes[1, 1].set_title('Vertical Profile')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Pixel')
    axes[1, 1].set_ylabel('Intensity')

    # Histograms
    axes[1, 2].hist(orig_slice.flatten(), bins=100, alpha=0.5, label='Original', density=True)
    axes[1, 2].hist(deconv_slice.flatten(), bins=100, alpha=0.5, label='Deconvolved', density=True)
    axes[1, 2].set_title('Intensity Distribution')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")


def plot_benchmark_summary(results, output_path):
    """Plot benchmark summary charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract data for plotting
    methods = list(set(r['method'] for r in results))
    n_slices_list = sorted(set(r['n_slices'] for r in results))

    # Peak Memory comparison
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        full_mem = [r['full']['peak_memory_mb'] for r in method_results]
        slice_mem = [r['slice']['peak_memory_mb'] for r in method_results]
        n_slices = [r['n_slices'] for r in method_results]

        label = 'RL' if method == 'richardson_lucy' else 'GC'
        axes[0].plot(n_slices, full_mem, 'o-', label=f'{label} Full Stack')
        axes[0].plot(n_slices, slice_mem, 's--', label=f'{label} Slice-by-Slice')

    axes[0].set_xlabel('Number of Slices')
    axes[0].set_ylabel('Peak GPU Memory (MB)')
    axes[0].set_title('Peak Memory Usage')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Timing comparison
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        full_time = [r['full']['total_time'] for r in method_results]
        slice_time = [r['slice']['total_time'] for r in method_results]
        n_slices = [r['n_slices'] for r in method_results]

        label = 'RL' if method == 'richardson_lucy' else 'GC'
        axes[1].plot(n_slices, full_time, 'o-', label=f'{label} Full Stack')
        axes[1].plot(n_slices, slice_time, 's--', label=f'{label} Slice-by-Slice')

    axes[1].set_xlabel('Number of Slices')
    axes[1].set_ylabel('Total Time (s)')
    axes[1].set_title('Processing Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Memory per slice
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        full_mem_per = [r['full']['peak_memory_mb'] / r['n_slices'] for r in method_results]
        slice_mem_per = [r['slice']['peak_memory_mb'] / r['n_slices'] for r in method_results]
        n_slices = [r['n_slices'] for r in method_results]

        label = 'RL' if method == 'richardson_lucy' else 'GC'
        axes[2].plot(n_slices, full_mem_per, 'o-', label=f'{label} Full Stack')
        axes[2].plot(n_slices, slice_mem_per, 's--', label=f'{label} Slice-by-Slice')

    axes[2].set_xlabel('Number of Slices')
    axes[2].set_ylabel('Memory per Slice (MB)')
    axes[2].set_title('Memory Efficiency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved benchmark summary: {output_path}")


def main():
    print("=" * 70)
    print("GPU DECONVOLUTION BENCHMARK")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Data: {DATA_DIR}")
    print(f"Iterations: {ITERATIONS}")
    print()

    # Load stack lazily
    stack = open_tiff_stack(DATA_DIR, channels=['488'])
    print(f"Stack shape: {stack.shape}")
    print(f"Total slices available: {len(stack)}")

    # Check single image size
    test_slice = stack[0]
    print(f"Slice shape: {test_slice.shape}")
    print(f"Memory per slice: {test_slice.nbytes / 1024**2:.2f} MB")

    # Generate PSF
    psf = generate_gaussian_psf(shape=(5, 21, 21), sigma=(2.0, 1.5, 1.5))
    print(f"PSF shape: {tuple(psf.shape)}")
    print()

    # Store original data for visualization
    original_data = stack.load_chunk(0, N_SLICES_LIST[0]).cpu().numpy()

    # Results storage
    all_results = []

    for n_slices in N_SLICES_LIST:
        print("=" * 70)
        print(f"TESTING WITH {n_slices} SLICES")
        print("=" * 70)

        for method in METHODS:
            method_short = 'RL' if method == 'richardson_lucy' else 'GC'

            # Full memory approach
            print(f"\n[{method_short}] Full-stack loading...")
            try:
                r_full = benchmark_full_stack(stack, psf, method, ITERATIONS, n_slices)
                print(f"  Load time:     {r_full['load_time']:.2f}s")
                print(f"  Deconv time:   {r_full['deconv_time']:.2f}s")
                print(f"  Total time:    {r_full['total_time']:.2f}s")
                print(f"  Mem after load: {r_full['mem_after_load_mb']:.0f} MB")
                print(f"  Peak memory:   {r_full['peak_memory_mb']:.0f} MB")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OUT OF MEMORY!")
                    r_full = {'error': 'OOM', 'peak_memory_mb': float('inf'), 'total_time': float('inf')}
                else:
                    raise

            # Slice-by-slice approach
            print(f"\n[{method_short}] Slice-by-slice processing...")
            r_slice = benchmark_slice_by_slice(stack, psf, method, ITERATIONS, n_slices)
            print(f"  Total time:    {r_slice['total_time']:.2f}s")
            print(f"  Peak memory:   {r_slice['peak_memory_mb']:.0f} MB")

            # Comparison
            if 'error' not in r_full:
                mem_savings = (1 - r_slice['peak_memory_mb'] / r_full['peak_memory_mb']) * 100
                time_diff = (r_slice['total_time'] / r_full['total_time'] - 1) * 100
                print(f"\n  Comparison:")
                print(f"    Memory savings: {mem_savings:.1f}%")
                print(f"    Time overhead:  {time_diff:+.1f}%")

                # Create visualization for first test case
                if n_slices == N_SLICES_LIST[0]:
                    visualize_results(
                        original_data,
                        r_full['result'],
                        f"{method_short} ({ITERATIONS} iters)",
                        OUTPUT_DIR / f"comparison_{method_short}.png"
                    )

            all_results.append({
                'n_slices': n_slices,
                'method': method,
                'full': r_full,
                'slice': r_slice,
            })

    # Summary table
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Slices':<8} {'Method':<6} {'Full Mem (MB)':<14} {'Slice (MB)':<12} {'Full Time':<10} {'Slice Time':<10}")
    print("-" * 70)

    for r in all_results:
        full_mem = r['full'].get('peak_memory_mb', float('inf'))
        slice_mem = r['slice']['peak_memory_mb']
        full_time = r['full'].get('total_time', float('inf'))
        slice_time = r['slice']['total_time']

        full_mem_str = f"{full_mem:.0f}" if full_mem < float('inf') else "OOM"
        full_time_str = f"{full_time:.2f}s" if full_time < float('inf') else "N/A"
        method_short = 'RL' if r['method'] == 'richardson_lucy' else 'GC'

        print(f"{r['n_slices']:<8} {method_short:<6} {full_mem_str:<14} {slice_mem:<12.0f} {full_time_str:<10} {slice_time:.2f}s")

    # Plot summary
    plot_benchmark_summary(all_results, OUTPUT_DIR / "benchmark_summary.png")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
