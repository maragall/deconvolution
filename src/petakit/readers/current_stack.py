"""Reader for 'current_N_stack.tiff' format (e.g. MERFISH acquisitions).

Directory structure:
    root/
        acquisition parameters.json
        configurations.xml
        0/  (single FOV folder)
            coordinates.csv
            current_0_stack.tiff   # channel0_z0
            current_1_stack.tiff   # channel0_z1
            ...
            current_7_stack.tiff   # channel0_z7
            current_8_stack.tiff   # channel1_z0
            ...

Files are sequential: for each round (timepoint), channels cycle with Nz
slices each.  Channel order comes from configurations.xml (Selected="true"
fluorescence modes).
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV


def detect_current_stack(root: Path) -> bool:
    """Check if directory contains current_*_stack.tiff format."""
    for subdir in root.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            if list(subdir.glob("current_*_stack.tiff")):
                return True
    return False


def _parse_fluorescence_channels(root: Path) -> list[str]:
    """Extract selected fluorescence channel wavelengths from configurations.xml."""
    xml_path = root / "configurations.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"configurations.xml not found at {root}. "
            "Required for current_stack format to identify channels."
        )

    tree = ET.parse(xml_path)
    pattern = re.compile(r"Fluorescence\s+(\d+)\s+nm\s+Ex")
    channels = []
    for mode in tree.getroot().findall(".//mode"):
        if mode.get("Selected") == "true":
            m = pattern.match(mode.get("Name", ""))
            if m:
                channels.append(m.group(1))
    if not channels:
        raise ValueError("No selected fluorescence channels found in configurations.xml")
    return channels


def open_current_stack(root: Path) -> "CurrentStackReader":
    """Open an acquisition in current_stack format."""
    root = Path(root)
    channels = _parse_fluorescence_channels(root)

    # Count total selected modes (including non-fluorescence) for stride
    xml_path = root / "configurations.xml"
    tree = ET.parse(xml_path)
    n_selected = sum(
        1 for mode in tree.getroot().findall(".//mode")
        if mode.get("Selected") == "true"
    )

    # Build channel-to-mode-index mapping (position among all selected modes)
    pattern = re.compile(r"Fluorescence\s+(\d+)\s+nm\s+Ex")
    channel_mode_index = {}
    mode_idx = 0
    for mode in tree.getroot().findall(".//mode"):
        if mode.get("Selected") == "true":
            m = pattern.match(mode.get("Name", ""))
            if m:
                channel_mode_index[m.group(1)] = mode_idx
            mode_idx += 1

    json_path = root / "acquisition_parameters.json"
    if not json_path.exists():
        json_path = root / "acquisition parameters.json"

    metadata = Metadata.from_acquisition_json(json_path, channels)

    return CurrentStackReader(root, metadata, n_selected, channel_mode_index)


class CurrentStackReader(AcquisitionReader):
    """Reader for current_N_stack.tiff sequential format."""

    def __init__(self, root, metadata, n_selected_modes, channel_mode_index):
        super().__init__(root, metadata)
        self._n_modes = n_selected_modes
        self._channel_idx = channel_mode_index
        self._nz = metadata.nz

    @property
    def format_name(self) -> str:
        return "current_stack"

    def _fov_dirs(self):
        """Yield sorted numeric subdirectories."""
        dirs = [d for d in self.root.iterdir() if d.is_dir() and d.name.isdigit()]
        return sorted(dirs, key=lambda d: int(d.name))

    def iter_fovs(self) -> Iterator[FOV]:
        """Each numeric subdirectory is one FOV."""
        for d in self._fov_dirs():
            yield FOV(region="current", index=int(d.name))

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """Load z-stack for a given FOV and channel (first timepoint)."""
        fov_dir = self.root / str(fov.index)
        if not fov_dir.exists():
            raise FileNotFoundError(f"FOV directory not found: {fov_dir}")

        mode_idx = self._channel_idx[channel]
        # Files for channel in round 0: start at mode_idx * nz
        start = mode_idx * self._nz

        slices = []
        for z in range(self._nz):
            fpath = fov_dir / f"current_{start + z}_stack.tiff"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing z-slice: {fpath}")
            slices.append(tifffile.imread(fpath))

        return np.stack(slices, axis=0).astype(np.float32)
