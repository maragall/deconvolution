"""Auto-detect acquisition format and open."""
from pathlib import Path

from .base import AcquisitionReader
from .individual import detect_individual, open_individual
from .ometiff import detect_ometiff, open_ometiff


def detect_format(root: Path) -> str | None:
    """
    Detect acquisition format from directory structure.

    Returns:
        "ometiff", "individual", or None if unknown
    """
    root = Path(root)

    if detect_ometiff(root):
        return "ometiff"
    if detect_individual(root):
        return "individual"
    return None


def open_acquisition(path: str | Path) -> AcquisitionReader:
    """
    Open a microscopy acquisition, auto-detecting the format.

    Supports:
        - OME-TIFF format (ome_tiff/*.ome.tiff)
        - Individual TIFF format (*_Fluorescence_*_nm_Ex.tiff)

    Args:
        path: Path to acquisition root directory

    Returns:
        AcquisitionReader for iterating FOVs and loading data

    Raises:
        FileNotFoundError: Path does not exist
        ValueError: Unknown acquisition format

    Example:
        >>> acq = open_acquisition("/path/to/acquisition")
        >>> print(f"Format: {acq.format_name}")
        >>> print(f"Channels: {acq.metadata.channels}")
        >>>
        >>> for fov in acq.iter_fovs():
        ...     stack = acq.get_stack(fov, channel="488")
        ...     print(f"FOV {fov}: shape={stack.shape}")
    """
    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    fmt = detect_format(root)

    if fmt == "ometiff":
        return open_ometiff(root)
    elif fmt == "individual":
        return open_individual(root)
    else:
        raise ValueError(
            f"Unknown acquisition format at {root}.\n"
            f"Expected one of:\n"
            f"  - ome_tiff/*.ome.tiff (OME-TIFF format)\n"
            f"  - */*_Fluorescence_*_nm_Ex.tiff (Individual format)"
        )
