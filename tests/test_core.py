"""Tests for core deconvolution module."""
import numpy as np
import pytest


class TestDeconvolve:
    """Tests for deconvolve function."""

    def test_requires_3d_image(self):
        from petakit.core import deconvolve

        image_2d = np.random.rand(64, 64).astype(np.float32)
        psf = np.random.rand(11, 11, 11).astype(np.float32)

        with pytest.raises(ValueError, match="3D"):
            deconvolve(image_2d, psf)

    def test_requires_3d_psf(self):
        from petakit.core import deconvolve

        image = np.random.rand(16, 64, 64).astype(np.float32)
        psf_2d = np.random.rand(11, 11).astype(np.float32)

        with pytest.raises(ValueError, match="3D"):
            deconvolve(image, psf_2d)

    def test_deconvolve_omw_runs(self, synthetic_stack, synthetic_psf):
        from petakit.core import deconvolve

        result = deconvolve(synthetic_stack, synthetic_psf, method="omw", gpu=False)
        assert result.shape == synthetic_stack.shape
        assert result.dtype == np.float32

    def test_deconvolve_rl_runs(self, synthetic_stack, synthetic_psf):
        from petakit.core import deconvolve

        result = deconvolve(
            synthetic_stack, synthetic_psf, method="rl", iterations=2, gpu=False,
        )
        assert result.shape == synthetic_stack.shape
        assert result.dtype == np.float32

    def test_deconvolve_unknown_method(self, synthetic_stack, synthetic_psf):
        from petakit.core import deconvolve

        with pytest.raises(ValueError, match="Unknown method"):
            deconvolve(synthetic_stack, synthetic_psf, method="bad")
