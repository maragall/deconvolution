# Deconwolf Python Wrapper

A Python wrapper and GUI for [**deconwolf**](https://github.com/elgw/deconwolf), a 3D deconvolution tool for fluorescence microscopy.

> **Attribution**: This project is a wrapper around **deconwolf**, developed by Erik Wernersson et al. The core deconvolution algorithms are entirely from the original deconwolf project. See [Wernersson et al., Nature Methods 2024](https://www.nature.com/articles/s41592-024-02294-7).

## Installation

### Step 1: Clone and set up the Python environment

```bash
git clone git@github.com:maragall/deconvolution.git
cd deconvolution

conda env create -f environment.yml
conda activate deconwolf
```

### Step 2: Get the deconwolf binary

This wrapper needs the `dw` binary from [deconwolf](https://github.com/elgw/deconwolf). Choose one of the options below.

#### Option A: Use the bundled binary (Linux x86_64, simplest)

A pre-built `dw` binary is included at `bin/linux-x86_64/dw` with its custom libraries. You just need the system dependencies it links against:

```bash
sudo apt-get install -y libfftw3-single3 libpng16-16
```

Verify it works:

```bash
./bin/linux-x86_64/dw --help
```

#### Option B: Build from source (any Linux distro, latest version)

```bash
# 1. Install build dependencies (Ubuntu/Debian)
sudo apt-get install -y cmake pkg-config gcc git \
  libfftw3-dev libfftw3-single3 libgsl-dev libomp-dev libpng-dev libtiff-dev

# 2. Clone and build deconwolf
git clone https://github.com/elgw/deconwolf.git /tmp/deconwolf
cd /tmp/deconwolf
mkdir builddir && cd builddir
cmake -DENABLE_GPU=OFF ..
cmake --build .

# 3. Either install system-wide:
sudo cmake --install . --prefix /usr

# Or copy into this project's bin/ directory:
mkdir -p /path/to/deconvolution/bin/linux-x86_64
cp dw /path/to/deconvolution/bin/linux-x86_64/
```

Add `-DENABLE_GPU=ON` in the cmake step if you have OpenCL and want GPU acceleration.

#### Option C: Point to an existing installation

If you already have `dw` installed elsewhere, either:

- Add it to your `PATH`, or
- Set the `DW_PATH` environment variable:
  ```bash
  export DW_PATH=/path/to/dw
  ```

### How binary discovery works

The wrapper searches for `dw` in this order:

1. `DW_PATH` environment variable
2. Bundled binary at `bin/<platform>/dw`
3. System `PATH`

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
