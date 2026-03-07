# PetaKit

Microscopy deconvolution with PetaKit5D algorithms (Richardson-Lucy, OTF-Masked Wiener).
GPU-first via CuPy, CPU fallback via NumPy/SciPy.

## Installation

```bash
git clone git@github.com:maragall/deconvolution.git
cd deconvolution

conda env create -f environment.yml
conda activate petakit
```

### Verify

```bash
python -c "from petakit import deconvolve; print('OK')"
```

### GPU (optional)

```bash
pip install cupy-cuda12x
```

## Usage

### GUI

```bash
petakit-gui
```

1. Browse to your acquisition folder
2. Select channel to process
3. Click "Run Deconvolution"

### Command Line

```bash
# OMW deconvolution (recommended, 2 iterations)
petakit /path/to/acquisition --channel 488

# RL deconvolution (max resolution, 15 iterations)
petakit /path/to/acquisition --channel 488 --method rl

# Force CPU (no GPU)
petakit /path/to/acquisition --channel 488 --no-gpu

# Verbose output
petakit /path/to/acquisition --channel 488 -v
```

### Python API

```python
from petakit import deconvolve, generate_psf, compute_psf_size, open_acquisition

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

## Deconvolution Methods

| Method | Iterations | Use case |
|--------|-----------|----------|
| `omw` (default) | 2 | High throughput, fast convergence |
| `rl` | 15 | Maximum resolution, more iterations |

GPU is auto-detected via CuPy. Use `--no-gpu` to force CPU.

## Credits

- **PetaKit5D**: Ruan et al., Nature Methods 2024

## License

MIT
