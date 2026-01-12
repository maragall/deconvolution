"""Base classes for Squid acquisition readers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json

import numpy as np


@dataclass
class SquidMetadata:
    """Microscope parameters extracted from acquisition_parameters.json."""
    dxy: float  # µm
    dz: float   # µm
    na: float
    magnification: float
    sensor_pixel_size_um: float
    nz: int
    nt: int
    channels: list[str]  # e.g. ["405", "488", "561"]

    @classmethod
    def from_json(cls, path: Path, channels: list[str]) -> "SquidMetadata":
        """Parse acquisition parameters.json."""
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
            sensor_pixel_size_um=sensor_px,
            nz=params["Nz"],
            nt=params["Nt"],
            channels=channels,
        )


@dataclass
class FOV:
    """A single field of view identifier."""
    region: str
    fov: int
    timepoint: int


class SquidAcquisition(ABC):
    """Abstract base for Squid acquisition formats."""

    def __init__(self, root: Path, metadata: SquidMetadata):
        self.root = root
        self.metadata = metadata

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return format identifier."""
        pass

    @abstractmethod
    def iter_fovs(self) -> Iterator[FOV]:
        """Iterate over all FOVs in the acquisition."""
        pass

    @abstractmethod
    def get_stack(self, fov: FOV, channel: str) -> np.ndarray:
        """Load a z-stack for given FOV and channel. Returns (Z, Y, X) array."""
        pass

    def get_all_channels(self, fov: FOV) -> dict[str, np.ndarray]:
        """Load all channels for a FOV. Returns {channel: (Z,Y,X) array}."""
        return {ch: self.get_stack(fov, ch) for ch in self.metadata.channels}
