#!/usr/bin/env python3
"""
Test script for Andrea dataset - processes a small crop to verify functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from deconvolution import (
    DeconvolutionEngine, PSFConvolution,
    load_tiff_stack, save_tiff_stack,
    generate_gaussian_psf
)
from deconvolution.utils import get_device, to_numpy

print(f"Device: {get_device()}")
print(f"PyTorch version: {torch.__version__}")

# Paths
data_dir = Path("/home/cephla/Downloads/20250802_andrea_test_R0/R0/")
output_dir = Path("/home/cephla/Downloads/deconvolution/test_output")
output_dir.mkdir(exist_ok=True)

# Load subset of data
print("\nLoading data (10 z-slices, 488nm channel)...")
data = load_tiff_stack(data_dir, channels=['488'], z_range=(0, 10))
print(f"  Full data shape: {tuple(data.shape)}")
print(f"  Data range: [{data.min():.1f}, {data.max():.1f}]")

# Crop to manageable size for testing
crop_size = 512
z_start, y_start, x_start = 0, 1500, 2000
crop = data[z_start:, y_start:y_start+crop_size, x_start:x_start+crop_size].clone()
print(f"\nUsing crop: {tuple(crop.shape)} at position (z={z_start}, y={y_start}, x={x_start})")

# Save input crop for reference
save_tiff_stack(output_dir / "input_crop.tif", crop)
print(f"  Saved input crop to {output_dir / 'input_crop.tif'}")

# Generate PSF
print("\nGenerating Gaussian PSF...")
psf = generate_gaussian_psf(
    shape=(11, 21, 21),  # z, y, x
    sigma=(2.0, 1.5, 1.5)  # sigma_z, sigma_y, sigma_x
)
print(f"  PSF shape: {tuple(psf.shape)}")

# Setup deconvolution
psf_op = PSFConvolution(psf, image_shape=tuple(crop.shape))
engine = DeconvolutionEngine(psf_op, background=100)  # Small background offset

# Run Richardson-Lucy
print("\nRunning Richardson-Lucy (30 iterations)...")
def rl_callback(i, est):
    if (i + 1) % 10 == 0:
        print(f"  Iteration {i+1}: range [{est.min():.1f}, {est.max():.1f}]")

result_rl = engine.deconvolve(crop, iterations=30, method='richardson_lucy', callback=rl_callback)
save_tiff_stack(output_dir / "result_rl.tif", result_rl)
print(f"  Saved to {output_dir / 'result_rl.tif'}")

# Run Gradient Consensus
print("\nRunning Gradient Consensus (30 iterations)...")
def gc_callback(i, est):
    if (i + 1) % 10 == 0:
        print(f"  Iteration {i+1}: range [{est.min():.1f}, {est.max():.1f}]")

result_gc = engine.deconvolve(crop, iterations=30, method='gradient_consensus', callback=gc_callback)
save_tiff_stack(output_dir / "result_gc.tif", result_gc)
print(f"  Saved to {output_dir / 'result_gc.tif'}")

# Save comparison stack
print("\nSaving comparison stack...")
import numpy as np
comparison = torch.stack([crop, result_rl, result_gc], dim=0)  # 3 x Z x Y x X
save_tiff_stack(output_dir / "comparison_input_rl_gc.tif", comparison)
print(f"  Saved to {output_dir / 'comparison_input_rl_gc.tif'}")

print(f"\n{'='*60}")
print("Results summary:")
print(f"  Input:   range [{crop.min():.1f}, {crop.max():.1f}]")
print(f"  RL:      range [{result_rl.min():.1f}, {result_rl.max():.1f}]")
print(f"  GC:      range [{result_gc.min():.1f}, {result_gc.max():.1f}]")
print(f"\nOutput saved to: {output_dir}")
print("Open the TIFF files in ImageJ/Fiji to view results.")
