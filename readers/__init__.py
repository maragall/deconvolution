"""Squid acquisition format readers."""
from .base import SquidAcquisition, SquidMetadata
from .detect import open_acquisition

__all__ = ["open_acquisition", "SquidAcquisition", "SquidMetadata"]
