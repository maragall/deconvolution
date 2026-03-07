"""Tests for PSF generation module."""
import numpy as np
import pytest


def _has_psfmodels():
    try:
        import psfmodels
        return True
    except ImportError:
        return False


class TestGeneratePSF:
    @pytest.mark.skipif(not _has_psfmodels(), reason="psfmodels not installed")
    def test_generates_normalized_psf(self):
        from petakit.psf import generate_psf

        psf = generate_psf(
            nz=11, nxy=11,
            dxy=0.1, dz=0.3,
            wavelength=0.525, na=0.8,
        )

        assert psf.shape == (11, 11, 11)
        assert abs(psf.sum() - 1.0) < 0.001
        assert psf.dtype == np.float32

    @pytest.mark.skipif(not _has_psfmodels(), reason="psfmodels not installed")
    def test_ensures_odd_sizes(self):
        from petakit.psf import generate_psf

        psf = generate_psf(
            nz=10, nxy=10,
            dxy=0.1, dz=0.3,
            wavelength=0.525, na=0.8,
        )

        assert psf.shape[0] % 2 == 1
        assert psf.shape[1] % 2 == 1
        assert psf.shape[2] % 2 == 1


class TestWavelengthFromChannel:
    def test_parses_standard_channels(self):
        from petakit.psf import wavelength_from_channel

        assert wavelength_from_channel("Fluorescence 488 nm Ex") == 0.525
        assert wavelength_from_channel("Fluorescence 405 nm Ex") == 0.450
        assert wavelength_from_channel("Fluorescence 561 nm Ex") == 0.590

    def test_raises_for_invalid_channel(self):
        from petakit.psf import wavelength_from_channel

        with pytest.raises(ValueError):
            wavelength_from_channel("InvalidChannel")


class TestComputePsfSize:
    def test_results_are_odd(self):
        from petakit.psf import compute_psf_size

        nz, nxy = compute_psf_size(10, dxy=0.376, dz=1.5, wavelength=0.525, na=0.8)
        assert nz % 2 == 1
        assert nxy % 2 == 1

    def test_higher_na_smaller_lateral(self):
        from petakit.psf import compute_psf_size

        _, nxy_low = compute_psf_size(10, dxy=0.376, dz=1.5, wavelength=0.525, na=0.4)
        _, nxy_high = compute_psf_size(10, dxy=0.376, dz=1.5, wavelength=0.525, na=1.4, ni=1.515)
        assert nxy_high < nxy_low


class TestInferImmersionIndex:
    def test_air(self):
        from petakit.psf import infer_immersion_index
        assert infer_immersion_index(0.8) == 1.0

    def test_water(self):
        from petakit.psf import infer_immersion_index
        assert infer_immersion_index(1.2) == 1.33

    def test_oil(self):
        from petakit.psf import infer_immersion_index
        assert infer_immersion_index(1.45) == 1.515
