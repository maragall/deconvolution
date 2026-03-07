"""Tests for deconvolution engine."""
import numpy as np
import pytest
from scipy.ndimage import convolve

from petakit.engine import (
    crop_psf_to_image,
    psf2otf,
    _omw_backprojector,
    _estimate_peak_gb,
    rl,
    omw,
)


class TestCropPsf:
    def test_noop_when_smaller(self, synthetic_psf, synthetic_stack):
        result = crop_psf_to_image(synthetic_psf, synthetic_stack.shape)
        assert result.shape == synthetic_psf.shape
        np.testing.assert_array_equal(result, synthetic_psf)

    def test_crops_when_larger(self):
        psf = np.ones((20, 20, 20), dtype=np.float32)
        result = crop_psf_to_image(psf, (10, 10, 10))
        assert result.shape == (10, 10, 10)


class TestPsf2Otf:
    def test_output_shape(self, synthetic_psf):
        out_shape = (16, 32, 32)
        otf = psf2otf(synthetic_psf, out_shape, np.fft.fftn)
        assert otf.shape == out_shape


class TestOmwBackprojector:
    def test_shape_matches_input(self, synthetic_psf):
        bp = _omw_backprojector(synthetic_psf)
        assert bp.shape == synthetic_psf.shape
        assert bp.dtype == np.float32

    def test_nonzero(self, synthetic_psf):
        bp = _omw_backprojector(synthetic_psf)
        assert np.any(bp != 0)


class TestEstimatePeakGb:
    def test_known_shape(self):
        # 100x100x100 = 1e6 voxels * 40 bytes = 40 MB = 0.04 GB
        result = _estimate_peak_gb((100, 100, 100))
        assert abs(result - 0.04) < 0.001


class TestRL:
    def test_preserves_shape(self, synthetic_stack, synthetic_psf):
        result = rl(synthetic_stack, synthetic_psf, n_iter=2, gpu=False)
        assert result.shape == synthetic_stack.shape

    def test_nonnegative(self, synthetic_stack, synthetic_psf):
        result = rl(synthetic_stack, synthetic_psf, n_iter=2, gpu=False)
        assert np.all(result >= 0)

    def test_improves_on_blurred(self, synthetic_psf):
        # Create ground truth and blur it
        z, y, x = np.mgrid[:8, :32, :32]
        truth = np.exp(-((z - 4)**2 + (y - 16)**2 + (x - 16)**2) / 4.0).astype(np.float32)
        blurred = convolve(truth, synthetic_psf, mode='constant').astype(np.float32)

        result = rl(blurred, synthetic_psf, n_iter=5, gpu=False)

        error_before = np.mean((blurred - truth) ** 2)
        error_after = np.mean((result - truth) ** 2)
        assert error_after < error_before


class TestOMW:
    def test_preserves_shape(self, synthetic_stack, synthetic_psf):
        result = omw(synthetic_stack, synthetic_psf, n_iter=2, gpu=False)
        assert result.shape == synthetic_stack.shape

    def test_nonnegative(self, synthetic_stack, synthetic_psf):
        result = omw(synthetic_stack, synthetic_psf, n_iter=2, gpu=False)
        assert np.all(result >= 0)

    def test_improves_on_blurred(self, synthetic_psf):
        z, y, x = np.mgrid[:8, :32, :32]
        truth = np.exp(-((z - 4)**2 + (y - 16)**2 + (x - 16)**2) / 4.0).astype(np.float32)
        blurred = convolve(truth, synthetic_psf, mode='constant').astype(np.float32)

        result = omw(blurred, synthetic_psf, n_iter=2, gpu=False)

        error_before = np.mean((blurred - truth) ** 2)
        error_after = np.mean((result - truth) ** 2)
        assert error_after < error_before


class TestTiling:
    def test_tiled_matches_direct(self, synthetic_stack, synthetic_psf):
        """Force tiling with tiny memory budget and compare to direct."""
        direct = rl(synthetic_stack, synthetic_psf, n_iter=2, gpu=False)
        tiled = rl(synthetic_stack, synthetic_psf, n_iter=2, gpu=False,
                   avail_memory_gb=0.0001)

        # Tiled result should be close (not identical due to boundary effects)
        np.testing.assert_allclose(tiled, direct, rtol=0.3, atol=0.01)
