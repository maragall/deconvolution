"""Reader for *_stack.tiff multi-page format (Squid MULTI_PAGE_TIFF output).

Each file is {region}_{fov}_stack.tiff containing all z-planes and channels
as individual pages. Page metadata is embedded as JSON in ImageDescription
with keys: z_level, channel, region_id, fov.
"""
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV

_PATTERN = re.compile(r"^(.+?)_(\d+)_stack\.tiff$")


def detect_currentstack(root: Path) -> bool:
    """Check if directory contains *_stack.tiff multi-page files."""
    for subdir in root.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            for f in subdir.glob("*_stack.tiff"):
                if _PATTERN.match(f.name):
                    return True
    return False


def open_currentstack(root: Path) -> "CurrentStackReader":
    """Open a *_stack.tiff acquisition."""
    root = Path(root)

    # Find the digit-named subdirectory containing the stack files
    plane_dir = None
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name.isdigit():
            if any(f for f in subdir.glob("*_stack.tiff") if _PATTERN.match(f.name)):
                plane_dir = subdir
                break

    if plane_dir is None:
        raise FileNotFoundError("No *_stack.tiff files found")

    # Read metadata from first file to discover channels and z-levels
    first_file = next(f for f in sorted(plane_dir.glob("*_stack.tiff")) if _PATTERN.match(f.name))
    channels = []
    nz = 0
    with tifffile.TiffFile(str(first_file)) as tif:
        ch_set = set()
        z_set = set()
        for page in tif.pages:
            meta = json.loads(page.description)
            ch_set.add(meta["channel"])
            z_set.add(meta["z_level"])
        nz = len(z_set)
        # Extract wavelength numbers for channel names, preserving order
        wl_pattern = re.compile(r"(\d{3})\s*nm")
        for ch in sorted(ch_set):
            m = wl_pattern.search(ch)
            channels.append(m.group(1) if m else ch)

    # Load acquisition metadata
    json_path = root / "acquisition parameters.json"
    metadata = Metadata.from_acquisition_json(json_path, channels=channels)
    metadata.nz = nz

    return CurrentStackReader(root, metadata, plane_dir)


class CurrentStackReader(AcquisitionReader):
    """Reader for multi-page *_stack.tiff files (Squid format)."""

    def __init__(self, root: Path, metadata: Metadata, plane_dir: Path):
        super().__init__(root, metadata)
        self._plane_dir = plane_dir

    @property
    def format_name(self) -> str:
        return "currentstack"

    def iter_fovs(self) -> Iterator[FOV]:
        """Yield FOVs discovered from filenames."""
        seen = set()
        for f in sorted(self._plane_dir.glob("*_stack.tiff")):
            m = _PATTERN.match(f.name)
            if m:
                region, idx = m.group(1), int(m.group(2))
                key = (region, idx)
                if key not in seen:
                    seen.add(key)
                    yield FOV(region=region, index=idx)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """Load z-stack for a specific FOV and channel.

        Args:
            fov: FOV identifier
            channel: Wavelength string (e.g. "638") — matched against page metadata
        """
        path = self._plane_dir / f"{fov.region}_{fov.index}_stack.tiff"
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")

        slices = {}
        with tifffile.TiffFile(str(path)) as tif:
            for page in tif.pages:
                meta = json.loads(page.description)
                # Match channel by wavelength substring
                if channel in meta["channel"]:
                    slices[meta["z_level"]] = page.asarray()

        if not slices:
            raise ValueError(
                f"Channel '{channel}' not found in {path.name}"
            )

        stack = np.stack([slices[z] for z in sorted(slices)], axis=0)
        return stack.astype(np.float32)

    def get_plane(self, fov: FOV, channel: str, z: int) -> np.ndarray:
        """Load a single z-plane efficiently (reads one page)."""
        path = self._plane_dir / f"{fov.region}_{fov.index}_stack.tiff"
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")

        with tifffile.TiffFile(str(path)) as tif:
            for page in tif.pages:
                meta = json.loads(page.description)
                if channel in meta["channel"] and meta["z_level"] == z:
                    return page.asarray().astype(np.float32)

        raise FileNotFoundError(f"Plane z={z}, channel '{channel}' not found in {path.name}")
