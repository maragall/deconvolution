"""Auto-detect Squid acquisition format and open."""
from pathlib import Path

from .base import SquidAcquisition
from .multipage import detect_multipage, open_multipage
from .individual import detect_individual, open_individual
from .ometiff import detect_ometiff, open_ometiff


def detect_format(root: Path) -> str | None:
    """
    Detect acquisition format from directory structure.
    Returns: "multipage", "individual", "ometiff", or None if unknown.
    """
    root = Path(root)

    if detect_ometiff(root):
        return "ometiff"
    if detect_multipage(root):
        return "multipage"
    if detect_individual(root):
        return "individual"
    return None


def open_acquisition(path: str | Path) -> SquidAcquisition:
    """
    Open a Squid acquisition, auto-detecting the format.

    Args:
        path: Path to acquisition root directory

    Returns:
        SquidAcquisition object for iterating and loading data

    Example:
        acq = open_acquisition("/path/to/acquisition")
        print(f"Format: {acq.format_name}")
        print(f"dxy={acq.metadata.dxy}, dz={acq.metadata.dz}, NA={acq.metadata.na}")

        for fov in acq.iter_fovs():
            stack = acq.get_stack(fov, channel="488")  # (Z, Y, X) array
    """
    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    fmt = detect_format(root)

    if fmt == "ometiff":
        return open_ometiff(root)
    elif fmt == "multipage":
        return open_multipage(root)
    elif fmt == "individual":
        return open_individual(root)
    else:
        raise ValueError(
            f"Unknown acquisition format at {root}. "
            "Expected: acquisition parameters.json + (ome_tiff/ | *_stack.tiff | *_Fluorescence_*_nm_Ex.tiff)"
        )
