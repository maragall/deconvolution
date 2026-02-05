"""Tests for PSF generation module."""
import numpy as np
import pytest


class TestGeneratePSF:
    """Tests for generate_psf function."""

    @pytest.mark.skipif(
        not _has_psfmodels(),
        reason="psfmodels not installed"
    )
    def test_generates_normalized_psf(self):
        """PSF should be normalized to sum=1."""
        from deconwolf.psf import generate_psf

        psf = generate_psf(
            nz=11, nxy=11,
            dxy=0.1, dz=0.3,
            wavelength=0.525, na=0.8,
        )

        assert psf.shape == (11, 11, 11)
        assert abs(psf.sum() - 1.0) < 0.001
        assert psf.dtype == np.float32

    @pytest.mark.skipif(
        not _has_psfmodels(),
        reason="psfmodels not installed"
    )
    def test_ensures_odd_sizes(self):
        """PSF sizes should be made odd for centering."""
        from deconwolf.psf import generate_psf

        psf = generate_psf(
            nz=10, nxy=10,  # Even sizes
            dxy=0.1, dz=0.3,
            wavelength=0.525, na=0.8,
        )

        assert psf.shape[0] % 2 == 1  # z should be odd
        assert psf.shape[1] % 2 == 1  # y should be odd
        assert psf.shape[2] % 2 == 1  # x should be odd


class TestWavelengthFromChannel:
    """Tests for wavelength extraction from channel names."""

    def test_parses_standard_channels(self):
        """Should parse standard fluorescence channel names."""
        from deconwolf.psf import wavelength_from_channel

        assert wavelength_from_channel("Fluorescence 488 nm Ex") == 0.525
        assert wavelength_from_channel("Fluorescence 405 nm Ex") == 0.450
        assert wavelength_from_channel("Fluorescence 561 nm Ex") == 0.590

    def test_raises_for_invalid_channel(self):
        """Should raise error for unparseable channel names."""
        from deconwolf.psf import wavelength_from_channel

        with pytest.raises(ValueError):
            wavelength_from_channel("InvalidChannel")


def _has_psfmodels():
    """Check if psfmodels is installed."""
    try:
        import psfmodels
        return True
    except ImportError:
        return False
