# Deconwolf Python Wrapper

A Python wrapper and GUI for [**deconwolf**](https://github.com/elgw/deconwolf), a 3D deconvolution tool for fluorescence microscopy.

> **Attribution**: This project is a wrapper around **deconwolf**, developed by Erik Wernersson et al. The core deconvolution algorithms are entirely from the original deconwolf project. See [Wernersson et al., Nature Methods 2024](https://www.nature.com/articles/s41592-024-02294-7).

## Installation

The deconwolf binary is **bundled** — no manual binary setup required.

### Linux

```bash
git clone git@github.com:maragall/deconvolution.git
cd deconvolution

# Install system libraries (FFTW3 and libpng are not bundled)
sudo apt-get install -y libfftw3-single3 libpng16-16

# Create Python environment
conda env create -f environment.yml
conda activate deconwolf
```

### Windows

```powershell
git clone git@github.com:maragall/deconvolution.git
cd deconvolution

# Create Python environment
conda env create -f environment.yml
conda activate deconwolf
```

All required DLLs are bundled in `bin\windows-x86_64\`.

### Verify

```bash
python -c "from deconwolf.binary import find_binary; print(find_binary())"
```

This should print the path to the bundled `dw` binary.

### Advanced: using your own deconwolf build

If you want to use a custom build instead of the bundled binary, the wrapper searches for `dw` in this order:

1. Bundled binary at `bin/<platform>/dw`
2. System `PATH`

To override, set the `DW_PATH` environment variable to point to your binary. To build deconwolf from source, see the [deconwolf repository](https://github.com/elgw/deconwolf).

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
deconwolf /path/to/acquisition --channel 488 --relerror 0.02 --maxiter 100

# GPU acceleration (requires OpenCL)
deconwolf /path/to/acquisition --channel 488 --method shbcl2
```

### Python API

```python
from deconwolf import deconvolve, generate_psf, open_acquisition
from deconwolf.psf import compute_psf_size

# Open acquisition
acq = open_acquisition("/path/to/data")
meta = acq.metadata

# Compute PSF dimensions from optical parameters
nz_psf, nxy_psf = compute_psf_size(
    meta.nz, meta.dxy, meta.dz,
    wavelength=0.525, na=meta.na, ni=1.0
)

# Generate PSF
psf = generate_psf(
    nz=nz_psf, nxy=nxy_psf,
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
| relerror | 0.02 | Convergence threshold (adaptive stopping) |
| maxiter | 200 | Maximum iterations |
| method | shb | Algorithm: `shb` (CPU), `rl` (Richardson-Lucy), `shbcl2` (GPU/OpenCL) |

## Credits

- **Deconwolf**: Erik Wernersson and contributors - https://github.com/elgw/deconwolf
- **Citation**: Wernersson et al., "Deconwolf—a GPU-accelerated deconvolution tool for microscopy", Nature Methods 2024

## License

This wrapper is MIT licensed. The deconwolf binary has its own license (GPL v3) - see the [deconwolf repository](https://github.com/elgw/deconwolf).
