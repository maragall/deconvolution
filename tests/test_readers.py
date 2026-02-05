"""Tests for acquisition readers."""
import pytest
from pathlib import Path


class TestDetectFormat:
    """Tests for format detection."""

    def test_returns_none_for_empty_dir(self, tmp_path):
        """Should return None for directory without acquisitions."""
        from deconwolf.readers import detect_format

        result = detect_format(tmp_path)
        assert result is None


class TestMetadata:
    """Tests for Metadata dataclass."""

    def test_from_acquisition_json(self, tmp_path):
        """Should parse acquisition_parameters.json."""
        from deconwolf.readers.base import Metadata

        # Create test JSON
        json_content = """{
            "dz(um)": 1.5,
            "Nz": 10,
            "Nt": 1,
            "objective": {
                "magnification": 20.0,
                "NA": 0.8,
                "tube_lens_f_mm": 180.0,
                "name": "20x"
            },
            "sensor_pixel_size_um": 7.52
        }"""

        json_path = tmp_path / "acquisition_parameters.json"
        json_path.write_text(json_content)

        meta = Metadata.from_acquisition_json(json_path, channels=["488", "561"])

        assert meta.dxy == pytest.approx(7.52 / 20.0)
        assert meta.dz == 1.5
        assert meta.na == 0.8
        assert meta.nz == 10
        assert meta.channels == ["488", "561"]
