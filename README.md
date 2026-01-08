# deconv_tool

Minimal microscopy deconvolution tool.

## Installation

```bash
git clone --recursive https://github.com/maragall/deconvolution.git
cd deconvolution
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- CUDA-capable GPU
- Dependencies: `zarr`, `tifffile`, `numpy`, `cupy`

## Usage

```python
from deconv_tool import OpticalParams, deconvolve_tiff

# All parameters must be explicitly provided (no hidden defaults)
params = OpticalParams(
    na=0.4,           # Numerical aperture
    wavelength=0.488, # Emission wavelength in microns
    dxy=0.276,        # Pixel size in microns
    dz=0.454,         # Z step in microns
    ni=1.0,           # Immersion refractive index (1.0=air, 1.515=oil)
)

# Deconvolve and save as zarr pyramid
deconvolve_tiff(
    input_path="input.tif",
    output_path="output.zarr",
    params=params,
    is_confocal=False,  # True for confocal microscopy
)
```

## Architecture

```
deconv_tool/
├── models.py      # OpticalParams dataclass
├── psf/           # PSF generation (wrapper around psfmodels)
├── deconv/        # Deconvolution (wrapper around RLGC)
├── pipeline.py    # High-level functions (load, deconvolve, save)
├── formats/       # Format readers
└── vendor/        # Bundled dependencies (PSFmodels, opm-processing-v2)
```

### Design Principles

1. **No hidden defaults** - All optical parameters must be explicitly provided
2. **Thin wrappers** - PSF and RLGC code are not modified, only wrapped
3. **Open/closed** - New formats can be added without modifying existing code

## API

### Low-level

```python
from deconv_tool import generate_psf, generate_confocal_psf, RLGCDeconvolver

# Generate PSF
psf = generate_psf(nz=50, nx=101, dxy=0.276, dz=0.454, wavelength=0.488, na=0.4)

# Or confocal PSF (widefield squared)
psf = generate_confocal_psf(nz=50, nx=101, dxy=0.276, dz=0.454, wavelength=0.488, na=0.4)

# Deconvolve
deconvolver = RLGCDeconvolver(gpu_id=0)
result = deconvolver.deconvolve(image, psf)
```

### High-level

```python
from deconv_tool import run_deconvolution, save_zarr_pyramid

# Run deconvolution
result, psf = run_deconvolution(image, params, is_confocal=False)

# Save as zarr pyramid
save_zarr_pyramid(result, "output.zarr")
```

## License

MIT License
