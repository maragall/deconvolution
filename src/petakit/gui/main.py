"""PetaKit GUI - Simple PyQt interface for microscopy deconvolution."""
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
    QGroupBox, QSpinBox, QTextEdit, QCheckBox,
)
from PyQt5.QtCore import QThread, pyqtSignal

from ..readers import open_acquisition
from ..psf import compute_psf_size, generate_psf, wavelength_from_channel, infer_immersion_index
from ..core import deconvolve
from ..engine import gpu_info


class DeconvolutionWorker(QThread):
    """Background worker for batch deconvolution."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, acq, channel, output_dir, params):
        super().__init__()
        self.acq = acq
        self.channel = channel
        self.output_dir = Path(output_dir)
        self.params = params

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            meta = self.acq.metadata
            wavelength = wavelength_from_channel(f"Fluorescence {self.channel} nm Ex")

            ni = infer_immersion_index(meta.na)

            nz_psf, nxy_psf = compute_psf_size(
                meta.nz, meta.dxy, meta.dz, wavelength, meta.na, ni,
            )

            psf = generate_psf(
                nz=nz_psf, nxy=nxy_psf,
                dxy=meta.dxy, dz=meta.dz,
                wavelength=wavelength, na=meta.na,
                ni=ni,
            )

            fovs = list(self.acq.iter_fovs())
            for i, fov in enumerate(fovs):
                self.progress.emit(i + 1, len(fovs), f"Processing {fov}")

                stack = self.acq.get_stack(fov, self.channel)
                result = deconvolve(stack, psf, **self.params)

                import tifffile
                out_path = self.output_dir / f"{fov}_deconv.tiff"
                tifffile.imwrite(out_path, result, imagej=True)

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PetaKit")
        self.setMinimumWidth(500)

        self.acq = None
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # GPU status
        self.gpu_label = QLabel(gpu_info())
        self.gpu_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.gpu_label)

        # Acquisition selection
        acq_group = QGroupBox("Acquisition")
        acq_layout = QVBoxLayout(acq_group)

        path_row = QHBoxLayout()
        self.path_label = QLabel("No acquisition loaded")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_acquisition)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(self.browse_btn)
        acq_layout.addLayout(path_row)

        self.info_label = QLabel("")
        acq_layout.addWidget(self.info_label)

        layout.addWidget(acq_group)

        # Channel selection
        channel_group = QGroupBox("Processing")
        channel_layout = QVBoxLayout(channel_group)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.setEnabled(False)
        channel_row.addWidget(self.channel_combo, 1)
        channel_layout.addLayout(channel_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self.output_label = QLabel("(auto)")
        self.output_btn = QPushButton("Change...")
        self.output_btn.clicked.connect(self._select_output)
        output_row.addWidget(self.output_label, 1)
        output_row.addWidget(self.output_btn)
        channel_layout.addLayout(output_row)

        layout.addWidget(channel_group)

        # Method and parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["omw (high throughput)", "rl (max resolution)"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        method_row.addWidget(self.method_combo)
        params_layout.addLayout(method_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(2)
        iter_row.addWidget(self.iter_spin)
        params_layout.addLayout(iter_row)

        self.nogpu_check = QCheckBox("Force CPU (disable GPU)")
        params_layout.addWidget(self.nogpu_check)

        layout.addWidget(params_group)

        # Run button
        self.run_btn = QPushButton("Run Deconvolution")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_deconvolution)
        layout.addWidget(self.run_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _on_method_changed(self, method):
        if "omw" in method.lower():
            self.iter_spin.setValue(2)
        else:
            self.iter_spin.setValue(15)

    def _browse_acquisition(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Acquisition Folder"
        )
        if path:
            self._load_acquisition(path)

    def _load_acquisition(self, path):
        try:
            self.acq = open_acquisition(path)
            self.path_label.setText(path)

            meta = self.acq.metadata
            self.info_label.setText(
                f"Format: {self.acq.format_name} | "
                f"NA={meta.na}, dxy={meta.dxy:.3f}um, dz={meta.dz:.1f}um"
            )

            self.channel_combo.clear()
            self.channel_combo.addItems(meta.channels)
            self.channel_combo.setEnabled(True)

            self.output_label.setText(str(Path(path) / "deconvolved"))
            self.run_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_label.setText(path)

    def _run_deconvolution(self):
        if not self.acq:
            return

        channel = self.channel_combo.currentText()
        output_dir = self.output_label.text()

        method = "omw" if "omw" in self.method_combo.currentText().lower() else "rl"

        params = {
            "method": method,
            "iterations": self.iter_spin.value(),
            "gpu": not self.nogpu_check.isChecked(),
        }

        self.worker = DeconvolutionWorker(self.acq, channel, output_dir, params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.worker.start()

    def _on_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_finished(self):
        self.status_label.setText("Complete!")
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)


def main():
    """GUI entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
