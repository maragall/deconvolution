"""Tests for CLI batch processing."""
import pytest

from petakit.cli import run_batch


class TestRunBatch:
    def test_runs_on_individual(self, individual_acquisition, tmp_path):
        output_dir = tmp_path / "output"
        run_batch(
            acq_path=individual_acquisition,
            channel="488",
            output_dir=output_dir,
            method="omw",
            iterations=1,
            use_gpu=False,
        )

        outputs = list((output_dir / "ome_tiff").glob("*.ome.tiff"))
        assert len(outputs) == 1

    def test_invalid_channel_raises(self, individual_acquisition):
        with pytest.raises(ValueError, match="not found"):
            run_batch(
                acq_path=individual_acquisition,
                channel="999",
                use_gpu=False,
            )

    def test_creates_output_dir(self, individual_acquisition, tmp_path):
        output_dir = tmp_path / "nested" / "output"
        run_batch(
            acq_path=individual_acquisition,
            channel="488",
            output_dir=output_dir,
            iterations=1,
            use_gpu=False,
        )

        assert output_dir.exists()
        assert len(list((output_dir / "ome_tiff").glob("*.ome.tiff"))) == 1
