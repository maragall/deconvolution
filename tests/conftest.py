"""Shared test fixtures for petakit tests."""
import json

import numpy as np
import pytest
import tifffile


@pytest.fixture
def synthetic_stack():
    """8x32x32 float32 volume with a Gaussian blob."""
    z, y, x = np.mgrid[:8, :32, :32]
    blob = np.exp(-((z - 4)**2 + (y - 16)**2 + (x - 16)**2) / 8.0)
    return blob.astype(np.float32)


@pytest.fixture
def synthetic_psf():
    """7x15x15 normalized Gaussian PSF."""
    z, y, x = np.mgrid[:7, :15, :15]
    psf = np.exp(-((z - 3)**2 / 2.0 + (y - 7)**2 / 4.0 + (x - 7)**2 / 4.0))
    psf = psf / psf.sum()
    return psf.astype(np.float32)


def _acq_params_json():
    """Standard acquisition_parameters.json content."""
    return {
        "dz(um)": 1.5,
        "Nz": 8,
        "Nt": 1,
        "objective": {
            "magnification": 20.0,
            "NA": 0.8,
        },
        "sensor_pixel_size_um": 7.52,
    }


@pytest.fixture
def individual_acquisition(tmp_path):
    """Fake individual-TIFF acquisition directory."""
    # Write metadata
    json_path = tmp_path / "acquisition_parameters.json"
    json_path.write_text(json.dumps(_acq_params_json()), encoding="utf-8")

    # Create FOV directory with 8 z-slices, 1 channel
    fov_dir = tmp_path / "0"
    fov_dir.mkdir()

    for z in range(8):
        img = np.random.randint(100, 200, (32, 32), dtype=np.uint16)
        fname = f"manual0_0_{z}_Fluorescence_488_nm_Ex.tiff"
        tifffile.imwrite(fov_dir / fname, img)

    return tmp_path


@pytest.fixture
def ometiff_acquisition(tmp_path):
    """Fake OME-TIFF acquisition directory."""
    # Write metadata
    json_path = tmp_path / "acquisition_parameters.json"
    json_path.write_text(json.dumps(_acq_params_json()), encoding="utf-8")

    # Create OME-TIFF with 2 channels, 4 z-planes
    ome_dir = tmp_path / "ome_tiff"
    ome_dir.mkdir()

    nz, nc, ny, nx = 4, 2, 32, 32
    data = np.random.randint(100, 200, (nc, nz, ny, nx), dtype=np.uint16)

    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="test">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT"
            SizeX="32" SizeY="32" SizeC="2" SizeZ="4" SizeT="1"
            Type="uint16">
      <Channel ID="Channel:0:0" Name="Fluorescence 488 nm Ex" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" Name="Fluorescence 561 nm Ex" SamplesPerPixel="1"/>
      <TiffData FirstC="0" FirstZ="0" FirstT="0"/>
    </Pixels>
  </Image>
</OME>"""

    filepath = ome_dir / "manual0_0.ome.tiff"
    tifffile.imwrite(
        filepath,
        data,
        photometric='minisblack',
        metadata={'axes': 'CZYX'},
        description=ome_xml,
    )

    return tmp_path
