# Deconwolf Python Wrapper

A Python wrapper and GUI for [**deconwolf**](https://github.com/elgw/deconwolf), a 3D deconvolution tool for fluorescence microscopy.

> **Attribution**: This project is a wrapper around **deconwolf**, developed by Erik Wernersson et al. The core deconvolution algorithms are entirely from the original deconwolf project. See [Wernersson et al., Nature Methods 2024](https://www.nature.com/articles/s41592-024-02294-7).

## Installation

```bash
# Clone the repository
git clone https://github.com/maragall/deconvolution
cd deconvolution

# Create conda environment
conda env create -f environment.yml
conda activate deconwolf

# Note: You need the deconwolf binary (dw) installed
# See https://github.com/elgw/deconwolf for installation instructions
```

## Usage

### GUI

```bash
deconwolf-gui
```

1. Browse to your acquisition folder
2. Select channel to process
3. Click "Run Deconvolution"

### Command Line

```bash
# Process all FOVs for channel 488
deconwolf /path/to/acquisition --channel 488

# With custom parameters
deconwolf /path/to/acquisition --channel 488 --relerror 0.01 --maxiter 100

# GPU acceleration (requires OpenCL)
deconwolf /path/to/acquisition --channel 488 --method shbcl2
```

### Python API

```python
from deconwolf import deconvolve, generate_psf, open_acquisition

# Open acquisition
acq = open_acquisition("/path/to/data")
meta = acq.metadata

# Generate PSF from metadata
psf = generate_psf(
    nz=31, nxy=31,
    dxy=meta.dxy, dz=meta.dz,
    wavelength=0.525, na=meta.na
)

# Deconvolve each FOV
for fov in acq.iter_fovs():
    stack = acq.get_stack(fov, channel="488")
    result = deconvolve(stack, psf)
```

## Supported Acquisition Formats

- **OME-TIFF**: `ome_tiff/*.ome.tiff`
- **Individual TIFF**: `*_Fluorescence_*_nm_Ex.tiff`

## Deconvolution Parameters

All parameters are passed directly to deconwolf. See the [deconwolf documentation](https://elgw.github.io/deconwolf/) for details.

| Parameter | Default | Description |
|-----------|---------|-------------|
| relerror | 0.001 | Convergence threshold (adaptive stopping) |
| maxiter | 200 | Maximum iterations |
| method | shb | Algorithm: `shb` (CPU), `rl` (Richardson-Lucy), `shbcl2` (GPU/OpenCL) |

## Credits

- **Deconwolf**: Erik Wernersson and contributors - https://github.com/elgw/deconwolf
- **Citation**: Wernersson et al., "Deconwolf—a GPU-accelerated deconvolution tool for microscopy", Nature Methods 2024

## License

This wrapper is MIT licensed. The deconwolf binary has its own license (GPL v3) - see the [deconwolf repository](https://github.com/elgw/deconwolf).
