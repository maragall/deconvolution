"""Reader for OME-TIFF format."""
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

from .base import AcquisitionReader, Metadata, FOV


def detect_ometiff(root: Path) -> bool:
    """Check if directory contains OME-TIFF format."""
    ome_dir = root / "ome_tiff"
    if ome_dir.exists():
        return any(ome_dir.glob("*.ome.tiff"))
    return False


def open_ometiff(root: Path) -> "OMETiffReader":
    """Open an acquisition in OME-TIFF format."""
    root = Path(root)
    ome_dir = root / "ome_tiff"

    # Get channels from first OME-TIFF file
    first_file = next(ome_dir.glob("*.ome.tiff"))
    with tifffile.TiffFile(first_file) as tif:
        ome = tif.ome_metadata
        # Extract channel names from OME-XML
        channels = _parse_channels_from_ome(ome)

    # Load metadata from JSON
    json_path = root / "acquisition_parameters.json"
    if not json_path.exists():
        json_path = root / "acquisition parameters.json"

    metadata = Metadata.from_acquisition_json(json_path, channels)

    return OMETiffReader(root, metadata)


def _parse_channels_from_ome(ome_xml: str) -> list[str]:
    """Extract channel wavelengths from OME-XML."""
    import re
    channels = []

    # Match channel names like "Fluorescence 488 nm Ex"
    pattern = re.compile(r'Name="Fluorescence (\d+) nm Ex"')
    for match in pattern.finditer(ome_xml):
        channels.append(match.group(1))

    return sorted(set(channels), key=int)


class OMETiffReader(AcquisitionReader):
    """
    Reader for OME-TIFF format.

    Directory structure:
        root/
            acquisition_parameters.json
            ome_tiff/
                manual0_0.ome.tiff   # Contains all channels and z-planes
                manual0_1.ome.tiff
                manual1_0.ome.tiff
                ...

    Each OME-TIFF contains dimensions: (C, Z, Y, X) or similar.
    """

    @property
    def format_name(self) -> str:
        return "ometiff"

    @property
    def ome_dir(self) -> Path:
        return self.root / "ome_tiff"

    def iter_fovs(self) -> Iterator[FOV]:
        """Yield unique FOVs from OME-TIFF files."""
        pattern = re.compile(r"^([a-zA-Z]+\d+)_(\d+)\.ome\.tiff$")

        for f in sorted(self.ome_dir.glob("*.ome.tiff")):
            if match := pattern.match(f.name):
                region = match.group(1)
                fov_idx = int(match.group(2))
                yield FOV(region=region, index=fov_idx)

    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """Load z-stack for given FOV and channel."""
        # Find the OME-TIFF file
        filename = f"{fov.region}_{fov.index}.ome.tiff"
        filepath = self.ome_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"OME-TIFF not found: {filepath}")

        # Read the file
        with tifffile.TiffFile(filepath) as tif:
            # Get channel index
            channel_idx = self._get_channel_index(tif.ome_metadata, channel)

            # Read data - typically shaped as (Z, C, Y, X) or (C, Z, Y, X)
            data = tif.asarray()

            # Handle different dimension orders
            # OME-TIFF from this system appears to be (Z*C, Y, X) interleaved
            # or could be (C, Z, Y, X)
            if data.ndim == 3:
                # Likely interleaved: reshape to (C, Z, Y, X) then extract channel
                n_channels = len(self.metadata.channels)
                n_z = data.shape[0] // n_channels
                data = data.reshape(n_z, n_channels, data.shape[1], data.shape[2])
                # Now (Z, C, Y, X)
                stack = data[:, channel_idx, :, :]
            elif data.ndim == 4:
                # Already separated, determine order from shape
                # Assume (Z, C, Y, X) if second dim matches channel count
                if data.shape[1] == len(self.metadata.channels):
                    stack = data[:, channel_idx, :, :]
                else:
                    # Try (C, Z, Y, X)
                    stack = data[channel_idx, :, :, :]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

        return stack.astype(np.float32)

    def _get_channel_index(self, ome_xml: str, channel: str) -> int:
        """Get channel index from OME-XML."""
        pattern = re.compile(r'Name="Fluorescence (\d+) nm Ex"')
        channels = []
        for match in pattern.finditer(ome_xml):
            channels.append(match.group(1))

        # Remove duplicates while preserving order
        seen = set()
        unique_channels = []
        for ch in channels:
            if ch not in seen:
                seen.add(ch)
                unique_channels.append(ch)

        if channel not in unique_channels:
            raise ValueError(
                f"Channel {channel} not found. Available: {unique_channels}"
            )

        return unique_channels.index(channel)
