"""
Test case definitions for synthetic benchmarks.
"""

# Quick test cases for fast validation
QUICK_TEST_CASES = [
    {
        "name": "ellipsoid_snr20",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    {
        "name": "spiky_snr20",
        "phantom_type": "spiky_cell",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
]

# Default test cases covering various scenarios
DEFAULT_TEST_CASES = [
    # Baseline - ellipsoid
    {
        "name": "ellipsoid_snr20",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    # Baseline - spiky
    {
        "name": "spiky_snr20",
        "phantom_type": "spiky_cell",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    # SNR sweep - low noise
    {
        "name": "ellipsoid_snr50",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 50,
    },
    # SNR sweep - high noise
    {
        "name": "ellipsoid_snr5",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 5,
    },
    # PSF variation - wider PSF
    {
        "name": "ellipsoid_wide_psf",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 2.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    # PSF variation - high aspect ratio
    {
        "name": "ellipsoid_high_aspect",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 5.0,
        "snr": 20,
    },
    # Size variation - large cell
    {
        "name": "ellipsoid_large",
        "phantom_type": "ellipsoid",
        "cell_size": (20, 15, 15),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    # Size variation - small cell
    {
        "name": "ellipsoid_small",
        "phantom_type": "ellipsoid",
        "cell_size": (5, 4, 4),
        "voxel_size": 0.3,
        "psf_sigma": 1.0,
        "psf_aspect_ratio": 3.0,
        "snr": 20,
    },
]

# PSF mismatch test cases
# Format: (true_psf_sigma, assumed_psf_sigma, true_aspect_ratio, assumed_aspect_ratio)
MISMATCH_TEST_CASES = [
    {
        "name": "mismatch_sigma_under",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "true_psf_sigma": 1.0,
        "assumed_psf_sigma": 0.8,  # Underestimate
        "true_psf_aspect_ratio": 3.0,
        "assumed_psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    {
        "name": "mismatch_sigma_over",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "true_psf_sigma": 1.0,
        "assumed_psf_sigma": 1.2,  # Overestimate
        "true_psf_aspect_ratio": 3.0,
        "assumed_psf_aspect_ratio": 3.0,
        "snr": 20,
    },
    {
        "name": "mismatch_aspect_under",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "true_psf_sigma": 1.0,
        "assumed_psf_sigma": 1.0,
        "true_psf_aspect_ratio": 3.0,
        "assumed_psf_aspect_ratio": 2.5,  # Underestimate
        "snr": 20,
    },
    {
        "name": "mismatch_aspect_over",
        "phantom_type": "ellipsoid",
        "cell_size": (10, 7, 7),
        "voxel_size": 0.3,
        "true_psf_sigma": 1.0,
        "assumed_psf_sigma": 1.0,
        "true_psf_aspect_ratio": 3.0,
        "assumed_psf_aspect_ratio": 3.5,  # Overestimate
        "snr": 20,
    },
]
