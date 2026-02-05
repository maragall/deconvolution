"""Tests for core deconvolution module."""
import numpy as np
import pytest


class TestDeconvolve:
    """Tests for deconvolve function."""

    def test_requires_3d_image(self):
        """Deconvolve should reject 2D images."""
        from deconwolf.core import deconvolve

        image_2d = np.random.rand(64, 64).astype(np.float32)
        psf = np.random.rand(11, 11, 11).astype(np.float32)

        with pytest.raises(ValueError, match="3D"):
            deconvolve(image_2d, psf)

    def test_requires_3d_psf(self):
        """Deconvolve should reject 2D PSF."""
        from deconwolf.core import deconvolve

        image = np.random.rand(16, 64, 64).astype(np.float32)
        psf_2d = np.random.rand(11, 11).astype(np.float32)

        with pytest.raises(ValueError, match="3D"):
            deconvolve(image, psf_2d)


class TestBinaryFinder:
    """Tests for binary discovery."""

    def test_custom_path_not_found(self):
        """Should raise error for non-existent custom path."""
        from deconwolf.binary import find_binary, DeconwolfNotFoundError

        with pytest.raises(DeconwolfNotFoundError):
            find_binary("/nonexistent/path/dw")
