"""Abstract acquisition reader interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json

import numpy as np


@dataclass
class Metadata:
    """Microscope acquisition parameters."""
    dxy: float          # Pixel size in µm
    dz: float           # Z-step in µm
    na: float           # Numerical aperture
    magnification: float
    channels: list[str] # Channel names (e.g., ["405", "488", "561"])
    nz: int             # Number of z-planes
    nt: int             # Number of timepoints

    @classmethod
    def from_acquisition_json(cls, path: Path, channels: list[str]) -> "Metadata":
        """
        Parse acquisition_parameters.json.

        Expected JSON structure:
            {
                "dz(um)": 1.5,
                "Nz": 10,
                "Nt": 1,
                "objective": {
                    "magnification": 20.0,
                    "NA": 0.8
                },
                "sensor_pixel_size_um": 7.52
            }
        """
        with open(path) as f:
            params = json.load(f)

        obj = params["objective"]
        mag = obj["magnification"]
        sensor_px = params["sensor_pixel_size_um"]

        return cls(
            dxy=sensor_px / mag,
            dz=params["dz(um)"],
            na=obj["NA"],
            magnification=mag,
            channels=channels,
            nz=params["Nz"],
            nt=params.get("Nt", 1),
        )


@dataclass
class FOV:
    """Field of view identifier."""
    region: str
    index: int
    timepoint: int = 0

    def __str__(self) -> str:
        return f"{self.region}_{self.index}"


class AcquisitionReader(ABC):
    """
    Abstract base for acquisition format readers.

    Subclasses must implement:
        - format_name: str property
        - iter_fovs(): Iterator[FOV]
        - get_stack(fov, channel): np.ndarray

    Example usage:
        acq = open_acquisition("/path/to/data")
        for fov in acq.iter_fovs():
            stack = acq.get_stack(fov, channel="488")
            # stack is (Z, Y, X) array
    """

    def __init__(self, root: Path, metadata: Metadata):
        self.root = root
        self.metadata = metadata

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return format identifier (e.g., 'individual', 'ometiff')."""
        pass

    @abstractmethod
    def iter_fovs(self) -> Iterator[FOV]:
        """Iterate over all FOVs in the acquisition."""
        pass

    @abstractmethod
    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """
        Load a z-stack for given FOV and channel.

        Args:
            fov: Field of view identifier
            channel: Channel name (e.g., "488")

        Returns:
            3D array with shape (Z, Y, X)
        """
        pass

    def get_all_channels(self, fov: FOV) -> dict[str, np.ndarray]:
        """Load all channels for a FOV."""
        return {ch: self.get_stack(fov, ch) for ch in self.metadata.channels}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"format='{self.format_name}', "
            f"fovs={sum(1 for _ in self.iter_fovs())}, "
            f"channels={self.metadata.channels})"
        )
