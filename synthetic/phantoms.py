"""
Continuous phantom generators for realistic deconvolution testing.

These phantoms are continuous-valued (like real microscopy data),
not binary masks (which are worst-case for deconvolution).
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_blob(
    shape: tuple,
    center: tuple = None,
    sigma: tuple = (5, 5, 5),
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Generate a 3D Gaussian blob.

    Args:
        shape: Output volume shape (z, y, x)
        center: Blob center (z, y, x). If None, uses volume center.
        sigma: Gaussian sigma for each axis (z, y, x)
        intensity: Peak intensity

    Returns:
        3D array with Gaussian blob
    """
    if center is None:
        center = tuple(s // 2 for s in shape)

    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center
    sz, sy, sx = sigma

    blob = np.exp(
        -((z - cz) ** 2 / (2 * sz ** 2) +
          (y - cy) ** 2 / (2 * sy ** 2) +
          (x - cx) ** 2 / (2 * sx ** 2))
    )

    return (blob * intensity).astype(np.float32)


def multi_blob(
    shape: tuple,
    n_blobs: int = 5,
    sigma_range: tuple = (3, 8),
    intensity_range: tuple = (0.5, 1.0),
    seed: int = None,
) -> np.ndarray:
    """
    Generate multiple Gaussian blobs at random positions.

    Args:
        shape: Output volume shape (z, y, x)
        n_blobs: Number of blobs
        sigma_range: (min, max) sigma for blobs
        intensity_range: (min, max) intensity for blobs
        seed: Random seed

    Returns:
        3D array with multiple blobs
    """
    if seed is not None:
        np.random.seed(seed)

    volume = np.zeros(shape, dtype=np.float32)

    for _ in range(n_blobs):
        # Random center (avoid edges)
        margin = max(sigma_range) * 2
        center = tuple(
            np.random.randint(margin, s - margin) if s > 2 * margin else s // 2
            for s in shape
        )

        # Random sigma (isotropic for simplicity)
        sigma_val = np.random.uniform(*sigma_range)
        sigma = (sigma_val, sigma_val, sigma_val)

        # Random intensity
        intensity = np.random.uniform(*intensity_range)

        blob = gaussian_blob(shape, center, sigma, intensity)
        volume = np.maximum(volume, blob)  # Max blending

    return volume


def bead_phantom(
    shape: tuple,
    n_beads: int = 20,
    bead_sigma: float = 2.0,
    intensity_range: tuple = (0.5, 1.0),
    seed: int = None,
) -> np.ndarray:
    """
    Generate a bead phantom (small point-like objects).

    Simulates fluorescent beads commonly used for PSF measurement.

    Args:
        shape: Output volume shape (z, y, x)
        n_beads: Number of beads
        bead_sigma: Gaussian sigma for beads (small = point-like)
        intensity_range: (min, max) intensity
        seed: Random seed

    Returns:
        3D array with bead phantom
    """
    if seed is not None:
        np.random.seed(seed)

    volume = np.zeros(shape, dtype=np.float32)

    margin = int(bead_sigma * 4)
    for _ in range(n_beads):
        center = tuple(
            np.random.randint(margin, s - margin) if s > 2 * margin else s // 2
            for s in shape
        )

        intensity = np.random.uniform(*intensity_range)
        sigma = (bead_sigma, bead_sigma, bead_sigma)

        blob = gaussian_blob(shape, center, sigma, intensity)
        volume = np.maximum(volume, blob)

    return volume


def filament_phantom(
    shape: tuple,
    n_filaments: int = 3,
    thickness: float = 2.0,
    seed: int = None,
) -> np.ndarray:
    """
    Generate filament-like structures (e.g., microtubules, actin).

    Args:
        shape: Output volume shape (z, y, x)
        n_filaments: Number of filaments
        thickness: Filament thickness (Gaussian sigma)
        seed: Random seed

    Returns:
        3D array with filament phantom
    """
    if seed is not None:
        np.random.seed(seed)

    volume = np.zeros(shape, dtype=np.float32)

    for _ in range(n_filaments):
        # Random start and end points
        start = np.array([np.random.randint(0, s) for s in shape])
        end = np.array([np.random.randint(0, s) for s in shape])

        # Draw line
        n_points = int(np.linalg.norm(end - start) * 2)
        if n_points < 2:
            continue

        for t in np.linspace(0, 1, n_points):
            point = start + t * (end - start)
            point = tuple(int(p) for p in point)

            # Check bounds
            if all(0 <= point[i] < shape[i] for i in range(3)):
                volume[point] = 1.0

    # Smooth to create thickness
    volume = gaussian_filter(volume, sigma=thickness)
    if volume.max() > 0:
        volume = volume / volume.max()

    return volume.astype(np.float32)


def cell_phantom(
    shape: tuple,
    cell_radius: tuple = (15, 20, 20),
    nucleus_radius: tuple = (8, 10, 10),
    n_vesicles: int = 10,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a simplified cell phantom with nucleus and vesicles.

    Args:
        shape: Output volume shape (z, y, x)
        cell_radius: Cell ellipsoid radii (z, y, x)
        nucleus_radius: Nucleus ellipsoid radii
        n_vesicles: Number of cytoplasmic vesicles
        seed: Random seed

    Returns:
        3D array with cell phantom
    """
    if seed is not None:
        np.random.seed(seed)

    center = tuple(s // 2 for s in shape)

    # Cell body (dim)
    cell = gaussian_blob(shape, center, cell_radius, intensity=0.3)

    # Nucleus (brighter)
    nucleus = gaussian_blob(shape, center, nucleus_radius, intensity=0.8)

    # Vesicles
    vesicles = np.zeros(shape, dtype=np.float32)
    for _ in range(n_vesicles):
        # Random position within cell
        offset = tuple(
            np.random.randint(-r // 2, r // 2) for r in cell_radius
        )
        vesicle_center = tuple(c + o for c, o in zip(center, offset))
        vesicle = gaussian_blob(shape, vesicle_center, (2, 2, 2), intensity=1.0)
        vesicles = np.maximum(vesicles, vesicle)

    # Combine
    phantom = np.maximum(cell, nucleus)
    phantom = np.maximum(phantom, vesicles)

    return phantom
