"""Multi-page TIFF format reader.

Structure:
    root/
    ├── acquisition parameters.json
    ├── coordinates.csv
    └── {timepoint}/
        └── {region}_{fov}_stack.tiff  (pages = channels × z-levels)
"""
from pathlib import Path
from typing import Iterator
import re

import numpy as np
import tifffile

from .base import SquidAcquisition, SquidMetadata, FOV


class MultiPageReader(SquidAcquisition):
    """Reader for multi-page TIFF format (channels × z stacked)."""

    @property
    def format_name(self) -> str:
        return "multipage"

    def _parse_filename(self, path: Path) -> tuple[str, int]:
        """Extract region and fov from filename like 'current_0_stack.tiff'."""
        match = re.match(r"(.+)_(\d+)_stack\.tiff?$", path.name)
        if not match:
            raise ValueError(f"Cannot parse filename: {path.name}")
        return match.group(1), int(match.group(2))

    def _get_timepoint_dirs(self) -> list[Path]:
        """Get sorted list of timepoint directories (named 0, 1, 2, ...)."""
        dirs = []
        for p in self.root.iterdir():
            if p.is_dir() and p.name.isdigit():
                dirs.append(p)
        return sorted(dirs, key=lambda x: int(x.name))

    def iter_fovs(self) -> Iterator[FOV]:
        """Iterate over all FOVs in the acquisition."""
        for tp_dir in self._get_timepoint_dirs():
            timepoint = int(tp_dir.name)
            for tiff_path in sorted(tp_dir.glob("*_stack.tif*")):
                region, fov = self._parse_filename(tiff_path)
                yield FOV(region=region, fov=fov, timepoint=timepoint)

    def _get_tiff_path(self, fov: FOV) -> Path:
        """Get path to the multi-page TIFF for a FOV."""
        tp_dir = self.root / str(fov.timepoint)
        # Try both .tiff and .tif extensions
        for ext in [".tiff", ".tif"]:
            path = tp_dir / f"{fov.region}_{fov.fov}_stack{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No stack found for {fov}")

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """
        Load a z-stack for given FOV and channel.

        Multi-page TIFFs store pages as: [C0Z0, C0Z1, ..., C0Zn, C1Z0, C1Z1, ...]
        Returns (Z, Y, X) array.
        """
        path = self._get_tiff_path(fov)

        # Count pages and read all
        with tifffile.TiffFile(path) as tif:
            n_pages = len(tif.pages)
        data = tifffile.imread(path, key=range(n_pages))  # Shape: (n_pages, Y, X)

        n_channels = len(self.metadata.channels)
        n_z = self.metadata.nz

        # Find channel index
        if channel not in self.metadata.channels:
            raise ValueError(f"Unknown channel: {channel}. Available: {self.metadata.channels}")
        ch_idx = self.metadata.channels.index(channel)

        # Extract z-planes for this channel
        # Pages are ordered: C0Z0, C0Z1, ..., C0Zn, C1Z0, ...
        start = ch_idx * n_z
        end = start + n_z
        return data[start:end]


def detect_multipage(root: Path) -> bool:
    """Check if root is a multi-page TIFF acquisition."""
    # Must have acquisition parameters
    if not (root / "acquisition parameters.json").exists():
        return False
    # Check first timepoint directory for *_stack.tiff files
    tp0 = root / "0"
    if not tp0.is_dir():
        return False
    return any(tp0.glob("*_stack.tif*"))


def open_multipage(root: Path) -> MultiPageReader:
    """Open a multi-page TIFF acquisition."""
    import json

    # Read params first to get nz
    params_path = root / "acquisition parameters.json"
    with open(params_path) as f:
        params = json.load(f)
    nz = params["Nz"]

    # Count pages in first stack to infer n_channels
    tp0 = root / "0"
    first_stack = next(tp0.glob("*_stack.tif*"))
    with tifffile.TiffFile(first_stack) as tif:
        n_pages = len(tif.pages)

    n_channels = n_pages // nz

    # Generate channel names (we don't know actual wavelengths from this format)
    # User may need to specify or we use generic names
    channels = [f"ch{i}" for i in range(n_channels)]

    metadata = SquidMetadata.from_json(params_path, channels)
    return MultiPageReader(root, metadata)
