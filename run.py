#!/usr/bin/env python3
"""
Run deconvolution on test data.
"""

import torch
from pathlib import Path
from deconvolution import deconvolve, load_tiff, save_tiff, gaussian_psf

# Config
DATA_DIR = Path("/home/cephla/Downloads/20250802_andrea_test_R0/R0")
OUTPUT_DIR = Path("/home/cephla/Downloads/deconvolution/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load a few slices (adjust range as needed)
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Loading data...")

files = sorted(DATA_DIR.glob("*488*.tiff"))[:10]  # First 10 slices, 488nm channel
stack = torch.stack([load_tiff(f) for f in files])
print(f"Stack shape: {stack.shape}")

# Generate PSF
psf = gaussian_psf(shape=(5, 21, 21), sigma=(2.0, 1.5, 1.5))
print(f"PSF shape: {psf.shape}")

# Run deconvolution
def progress(i, est):
    if (i + 1) % 10 == 0:
        print(f"  Iteration {i + 1}")

print("\nRichardson-Lucy:")
torch.cuda.reset_peak_memory_stats()
result_rl = deconvolve(stack, psf, iterations=30, method='richardson_lucy', callback=progress)
print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\nGradient Consensus:")
torch.cuda.reset_peak_memory_stats()
result_gc = deconvolve(stack, psf, iterations=30, method='gradient_consensus', callback=progress)
print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Save
save_tiff(OUTPUT_DIR / "input.tif", stack)
save_tiff(OUTPUT_DIR / "result_rl.tif", result_rl)
save_tiff(OUTPUT_DIR / "result_gc.tif", result_gc)

print(f"\nSaved to {OUTPUT_DIR}")
