"""Deconwolf GUI - Simple PyQt interface for microscopy deconvolution."""
import math
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
    QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, QCheckBox,
)
from PyQt5.QtCore import QThread, pyqtSignal

from ..readers import open_acquisition
from ..psf import compute_psf_size, generate_psf, wavelength_from_channel
from ..core import deconvolve


class DeconvolutionWorker(QThread):
    """Background worker for batch deconvolution."""
    progress = pyqtSignal(int, int, str)  # current, total, message
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

            # Generate PSF
            meta = self.acq.metadata
            wavelength = wavelength_from_channel(f"Fluorescence {self.channel} nm Ex")

            # Auto-detect immersion medium based on NA
            # NA cannot exceed refractive index of immersion medium
            if meta.na <= 1.0:
                ni = 1.0    # air
            elif meta.na <= 1.33:
                ni = 1.33   # water
            else:
                ni = 1.515  # oil

            nz_psf, nxy_psf = compute_psf_size(
                meta.nz, meta.dxy, meta.dz, wavelength, meta.na, ni,
            )

            psf = generate_psf(
                nz=nz_psf, nxy=nxy_psf,
                dxy=meta.dxy, dz=meta.dz,
                wavelength=wavelength, na=meta.na,
                ni=ni,
            )

            # Process each FOV
            fovs = list(self.acq.iter_fovs())
            for i, fov in enumerate(fovs):
                self.progress.emit(i + 1, len(fovs), f"Processing {fov}")

                stack = self.acq.get_stack(fov, self.channel)
                result = deconvolve(stack, psf, **self.params)

                # Save result
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
        self.setWindowTitle("Deconwolf")
        self.setMinimumWidth(500)

        self.acq = None
        self.worker = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

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

        # Advanced settings (collapsible)
        self.advanced_check = QCheckBox("Advanced Settings")
        self.advanced_check.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_check)

        self.advanced_group = QGroupBox()
        self.advanced_group.setVisible(False)
        adv_layout = QVBoxLayout(self.advanced_group)

        # Relerror
        rel_row = QHBoxLayout()
        rel_row.addWidget(QLabel("Relerror:"))
        self.relerror_spin = QDoubleSpinBox()
        self.relerror_spin.setDecimals(3)
        self.relerror_spin.setRange(0.001, 0.5)
        self.relerror_spin.setSingleStep(0.01)
        self.relerror_spin.setValue(0.02)
        rel_row.addWidget(self.relerror_spin)
        adv_layout.addLayout(rel_row)

        # Maxiter
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max iterations:"))
        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(10, 500)
        self.maxiter_spin.setValue(200)
        max_row.addWidget(self.maxiter_spin)
        adv_layout.addLayout(max_row)

        # Method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["shb", "rl", "shbcl2 (GPU)"])
        method_row.addWidget(self.method_combo)
        adv_layout.addLayout(method_row)

        layout.addWidget(self.advanced_group)

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

    def _toggle_advanced(self, checked):
        self.advanced_group.setVisible(checked)

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
                f"NA={meta.na}, dxy={meta.dxy:.3f}µm, dz={meta.dz:.1f}µm"
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

        method = self.method_combo.currentText()
        if "GPU" in method:
            method = "shbcl2"

        params = {
            "relerror": self.relerror_spin.value(),
            "maxiter": self.maxiter_spin.value(),
            "method": method,
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
