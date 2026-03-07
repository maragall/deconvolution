"""Reader for individual TIFF format (one file per z-slice per channel)."""
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV


def detect_individual(root: Path) -> bool:
    """Check if directory contains individual TIFF format."""
    # Look for pattern: *_Fluorescence_*_nm_Ex.tiff in subdirectories
    for subdir in root.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            for f in subdir.glob("*_Fluorescence_*_nm_Ex.tiff"):
                return True
    return False


def open_individual(root: Path) -> "IndividualReader":
    """Open an acquisition in individual TIFF format."""
    root = Path(root)

    # Find all TIFF files and extract channels
    channels = set()
    pattern = re.compile(r"Fluorescence_(\d+)_nm_Ex")

    for subdir in root.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*_Fluorescence_*_nm_Ex.tiff"):
                if match := pattern.search(f.name):
                    channels.add(match.group(1))

    channels = sorted(channels, key=int)

    # Load metadata
    json_path = root / "acquisition_parameters.json"
    if not json_path.exists():
        json_path = root / "acquisition parameters.json"

    metadata = Metadata.from_acquisition_json(json_path, channels)

    return IndividualReader(root, metadata)


class IndividualReader(AcquisitionReader):
    """
    Reader for individual TIFF format.

    Directory structure:
        root/
            acquisition_parameters.json
            0/  (region folder)
                coordinates.csv
                manual0_0_0_Fluorescence_405_nm_Ex.tiff
                manual0_0_0_Fluorescence_488_nm_Ex.tiff
                manual0_0_1_Fluorescence_405_nm_Ex.tiff
                ...

    Filename pattern: {region}_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
    """

    @property
    def format_name(self) -> str:
        return "individual"

    def iter_fovs(self) -> Iterator[FOV]:
        """Yield unique FOVs."""
        seen = set()
        pattern = re.compile(r"^([a-zA-Z]+\d+)_(\d+)_\d+_Fluorescence")

        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue

            for f in sorted(subdir.glob("*_Fluorescence_*_nm_Ex.tiff")):
                if match := pattern.match(f.name):
                    region = match.group(1)
                    fov_idx = int(match.group(2))
                    key = (region, fov_idx)

                    if key not in seen:
                        seen.add(key)
                        yield FOV(region=region, index=fov_idx)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """Load z-stack for given FOV and channel."""
        # Find all files for this FOV and channel
        pattern = f"{fov.region}_{fov.index}_*_Fluorescence_{channel}_nm_Ex.tiff"
        files = []

        for subdir in self.root.iterdir():
            if subdir.is_dir():
                files.extend(subdir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files found for FOV {fov}, channel {channel}"
            )

        # Sort by z-index
        z_pattern = re.compile(rf"{fov.region}_{fov.index}_(\d+)_Fluorescence")
        files_with_z = []
        for f in files:
            if match := z_pattern.search(f.name):
                z = int(match.group(1))
                files_with_z.append((z, f))

        files_with_z.sort(key=lambda x: x[0])

        # Stack images
        slices = [tifffile.imread(f) for _, f in files_with_z]
        return np.stack(slices, axis=0).astype(np.float32)
