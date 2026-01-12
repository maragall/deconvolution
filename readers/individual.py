"""Individual images format reader.

Structure:
    root/
    ├── acquisition parameters.json
    └── {timepoint}/
        ├── coordinates.csv
        └── {region}_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
"""
from pathlib import Path
from typing import Iterator
import re

import numpy as np
import tifffile

from .base import SquidAcquisition, SquidMetadata, FOV


class IndividualReader(SquidAcquisition):
    """Reader for individual TIFF format (one file per FOV/channel/z)."""

    @property
    def format_name(self) -> str:
        return "individual"

    def _parse_filename(self, path: Path) -> tuple[str, int, int, str]:
        """
        Parse filename like 'manual0_0_0_Fluorescence_488_nm_Ex.tiff'.
        Returns (region, fov, z_level, wavelength).
        """
        match = re.match(
            r"(.+)_(\d+)_(\d+)_Fluorescence_(\d+)_nm_Ex\.tiff?$",
            path.name
        )
        if not match:
            raise ValueError(f"Cannot parse filename: {path.name}")
        return match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)

    def _get_timepoint_dirs(self) -> list[Path]:
        """Get sorted list of timepoint directories."""
        dirs = []
        for p in self.root.iterdir():
            if p.is_dir() and p.name.isdigit():
                dirs.append(p)
        return sorted(dirs, key=lambda x: int(x.name))

    def iter_fovs(self) -> Iterator[FOV]:
        """Iterate over unique FOVs (region, fov, timepoint combinations)."""
        seen = set()
        for tp_dir in self._get_timepoint_dirs():
            timepoint = int(tp_dir.name)
            for tiff_path in tp_dir.glob("*_Fluorescence_*_nm_Ex.tif*"):
                region, fov, _, _ = self._parse_filename(tiff_path)
                key = (region, fov, timepoint)
                if key not in seen:
                    seen.add(key)
                    yield FOV(region=region, fov=fov, timepoint=timepoint)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """
        Load a z-stack for given FOV and channel.
        Returns (Z, Y, X) array.
        """
        tp_dir = self.root / str(fov.timepoint)
        z_planes = []

        for z in range(self.metadata.nz):
            # Try both .tiff and .tif
            for ext in [".tiff", ".tif"]:
                path = tp_dir / f"{fov.region}_{fov.fov}_{z}_Fluorescence_{channel}_nm_Ex{ext}"
                if path.exists():
                    z_planes.append(tifffile.imread(path))
                    break
            else:
                raise FileNotFoundError(
                    f"Missing z-plane: {fov.region}_{fov.fov}_{z}_Fluorescence_{channel}_nm_Ex"
                )

        return np.stack(z_planes, axis=0)


def detect_individual(root: Path) -> bool:
    """Check if root is an individual images acquisition."""
    if not (root / "acquisition parameters.json").exists():
        return False
    tp0 = root / "0"
    if not tp0.is_dir():
        return False
    return any(tp0.glob("*_Fluorescence_*_nm_Ex.tif*"))


def open_individual(root: Path) -> IndividualReader:
    """Open an individual images acquisition."""
    # Discover channels from files in first timepoint
    tp0 = root / "0"
    channels = set()
    pattern = re.compile(r"_Fluorescence_(\d+)_nm_Ex\.tiff?$")

    for path in tp0.glob("*_Fluorescence_*_nm_Ex.tif*"):
        match = pattern.search(path.name)
        if match:
            channels.add(match.group(1))

    channels = sorted(channels, key=int)  # Sort by wavelength

    params_path = root / "acquisition parameters.json"
    metadata = SquidMetadata.from_json(params_path, channels)
    return IndividualReader(root, metadata)
