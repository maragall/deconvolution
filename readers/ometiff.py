"""OME-TIFF format reader.

Structure:
    root/
    ├── acquisition parameters.json
    ├── {timepoint}/
    │   └── coordinates.csv
    └── ome_tiff/
        └── {region}_{fov}.ome.tiff  (contains all C, Z, T)
"""
from pathlib import Path
from typing import Iterator
import re
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

from .base import SquidAcquisition, SquidMetadata, FOV


class OmeTiffReader(SquidAcquisition):
    """Reader for OME-TIFF format."""

    def __init__(self, root: Path, metadata: SquidMetadata):
        super().__init__(root, metadata)
        self._ome_dir = root / "ome_tiff"

    @property
    def format_name(self) -> str:
        return "ometiff"

    def _parse_filename(self, path: Path) -> tuple[str, int]:
        """Parse filename like 'manual0_0.ome.tiff'. Returns (region, fov)."""
        match = re.match(r"(.+)_(\d+)\.ome\.tiff?$", path.name)
        if not match:
            raise ValueError(f"Cannot parse filename: {path.name}")
        return match.group(1), int(match.group(2))

    def iter_fovs(self) -> Iterator[FOV]:
        """Iterate over all FOVs. Each OME-TIFF contains all timepoints."""
        for path in sorted(self._ome_dir.glob("*.ome.tif*")):
            region, fov = self._parse_filename(path)
            for t in range(self.metadata.nt):
                yield FOV(region=region, fov=fov, timepoint=t)

    def _get_ome_path(self, fov: FOV) -> Path:
        """Get path to OME-TIFF for a FOV."""
        for ext in [".ome.tiff", ".ome.tif"]:
            path = self._ome_dir / f"{fov.region}_{fov.fov}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No OME-TIFF found for {fov.region}_{fov.fov}")

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """
        Load a z-stack for given FOV, timepoint, and channel.
        OME-TIFF dimension order is typically XYCZT.
        Returns (Z, Y, X) array.
        """
        path = self._get_ome_path(fov)

        with tifffile.TiffFile(path) as tif:
            # Get shape info from OME metadata
            if tif.ome_metadata:
                # Parse from OME-XML to get dimension order
                pass

            # Read as array - tifffile handles OME dimension ordering
            data = tif.asarray()

        # Expected shape from OME: (T, C, Z, Y, X) or (C, Z, Y, X) if T=1
        # Find channel index
        ch_idx = self.metadata.channels.index(channel)

        if data.ndim == 5:  # TCZYX
            return data[fov.timepoint, ch_idx]
        elif data.ndim == 4:  # CZYX (single timepoint)
            return data[ch_idx]
        elif data.ndim == 3:  # ZYX (single channel, single timepoint)
            return data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")


def detect_ometiff(root: Path) -> bool:
    """Check if root is an OME-TIFF acquisition."""
    if not (root / "acquisition parameters.json").exists():
        return False
    ome_dir = root / "ome_tiff"
    if not ome_dir.is_dir():
        return False
    return any(ome_dir.glob("*.ome.tif*"))


def open_ometiff(root: Path) -> OmeTiffReader:
    """Open an OME-TIFF acquisition."""
    ome_dir = root / "ome_tiff"
    first_ome = next(ome_dir.glob("*.ome.tif*"))

    # Extract channel names from OME-XML
    channels = []
    with tifffile.TiffFile(first_ome) as tif:
        if tif.ome_metadata:
            root_xml = ET.fromstring(tif.ome_metadata)
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            for ch in root_xml.findall(".//ome:Channel", ns):
                name = ch.get("Name", "")
                # Extract wavelength from name like "Fluorescence 488 nm Ex"
                match = re.search(r"(\d+)\s*nm", name)
                if match:
                    channels.append(match.group(1))
                else:
                    channels.append(name)

    if not channels:
        # Fallback: read from array shape
        data = tifffile.imread(first_ome)
        if data.ndim >= 4:
            n_ch = data.shape[-4] if data.ndim == 5 else data.shape[0]
            channels = [f"ch{i}" for i in range(n_ch)]

    params_path = root / "acquisition parameters.json"
    metadata = SquidMetadata.from_json(params_path, channels)
    return OmeTiffReader(root, metadata)
