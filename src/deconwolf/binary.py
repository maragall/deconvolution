"""Find deconwolf binary."""
import os
import platform
import shutil
from pathlib import Path


class DeconwolfNotFoundError(FileNotFoundError):
    """Raised when deconwolf binary cannot be found."""
    pass


def find_binary(custom_path: str | None = None) -> str:
    """
    Find the deconwolf (dw) binary.

    Search order:
        1. custom_path if provided
        2. DW_PATH environment variable
        3. Bundled binary in package bin/ directory
        4. System PATH

    Args:
        custom_path: Optional explicit path to dw binary

    Returns:
        Path to dw binary

    Raises:
        DeconwolfNotFoundError: If binary not found
    """
    # 1. Custom path
    if custom_path:
        if os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
            return custom_path
        raise DeconwolfNotFoundError(f"Binary not found at: {custom_path}")

    # 2. Environment variable
    if env_path := os.environ.get("DW_PATH"):
        if os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path

    # 3. Bundled binary
    pkg_root = Path(__file__).parent.parent.parent
    system = platform.system().lower()

    if system == "linux":
        bundled = pkg_root / "bin" / "linux-x86_64" / "dw"
    elif system == "windows":
        bundled = pkg_root / "bin" / "windows-x86_64" / "dw.exe"
    else:
        bundled = None

    if bundled and bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)

    # 4. System PATH
    if found := shutil.which("dw"):
        return found

    raise DeconwolfNotFoundError(
        "deconwolf binary 'dw' not found.\n"
        "Options:\n"
        "  1. Install deconwolf and add to PATH\n"
        "  2. Set DW_PATH environment variable\n"
        "  3. Place binary in bin/<platform>/ directory"
    )
