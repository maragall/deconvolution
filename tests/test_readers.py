"""Tests for acquisition readers."""
import pytest
from pathlib import Path


class TestDetectFormat:
    def test_returns_none_for_empty_dir(self, tmp_path):
        from petakit.readers import detect_format

        result = detect_format(tmp_path)
        assert result is None


class TestMetadata:
    def test_from_acquisition_json(self, tmp_path):
        from petakit.readers.base import Metadata

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

    def test_missing_json_gives_clear_error(self, tmp_path):
        from petakit.readers.base import Metadata

        missing_path = tmp_path / "acquisition_parameters.json"

        with pytest.raises(FileNotFoundError, match="acquisition_parameters.json not found"):
            Metadata.from_acquisition_json(missing_path, channels=["488"])

    def test_missing_fields_gives_clear_error(self, tmp_path):
        from petakit.readers.base import Metadata

        json_path = tmp_path / "acquisition_parameters.json"
        json_path.write_text('{"objective": {"magnification": 20, "NA": 0.8}}')

        with pytest.raises(ValueError, match="missing required fields"):
            Metadata.from_acquisition_json(json_path, channels=["488"])


class TestIndividualReader:
    def test_detect(self, individual_acquisition):
        from petakit.readers.individual import detect_individual

        assert detect_individual(individual_acquisition) is True

    def test_iter_fovs(self, individual_acquisition):
        from petakit.readers.individual import open_individual

        reader = open_individual(individual_acquisition)
        fovs = list(reader.iter_fovs())
        assert len(fovs) == 1
        assert str(fovs[0]) == "manual0_0"

    def test_get_stack_shape(self, individual_acquisition):
        from petakit.readers.individual import open_individual

        reader = open_individual(individual_acquisition)
        fovs = list(reader.iter_fovs())
        stack = reader.get_stack(fovs[0], "488")

        assert stack.ndim == 3
        assert stack.shape == (8, 32, 32)
        assert stack.dtype.name == "float32"


class TestOMETiffReader:
    def test_detect(self, ometiff_acquisition):
        from petakit.readers.ometiff import detect_ometiff

        assert detect_ometiff(ometiff_acquisition) is True

    def test_get_stack_shape(self, ometiff_acquisition):
        from petakit.readers.ometiff import open_ometiff

        reader = open_ometiff(ometiff_acquisition)
        fovs = list(reader.iter_fovs())
        assert len(fovs) >= 1

        stack = reader.get_stack(fovs[0], "488")
        assert stack.ndim == 3
        assert stack.shape[0] == 4  # 4 z-planes
        assert stack.shape[1] == 32
        assert stack.shape[2] == 32
        assert stack.dtype.name == "float32"


class TestOpenAcquisition:
    def test_individual(self, individual_acquisition):
        from petakit.readers import open_acquisition

        acq = open_acquisition(individual_acquisition)
        assert acq.format_name == "individual"

    def test_unknown_format(self, tmp_path):
        from petakit.readers import open_acquisition

        with pytest.raises(ValueError, match="Unknown acquisition format"):
            open_acquisition(tmp_path)
