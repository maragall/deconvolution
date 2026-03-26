"""PetaKit GUI - Simple PyQt interface for microscopy deconvolution."""
import sys
import tempfile
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar,
    QGroupBox, QSpinBox, QTextEdit, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

from ..readers import open_acquisition
from ..psf import compute_psf_size, generate_psf, wavelength_from_channel, infer_immersion_index
from ..core import deconvolve
from ..engine import gpu_info


# ── Helpers ───────────────────────────────────────────────────────────────

def _save_stack_ome(path, result, channel_name):
    """Normalize to uint16 and save as OME-TIFF with channel metadata."""
    import numpy as np
    import tifffile
    p_low, p_high = np.percentile(result, (0.1, 99.9))
    if p_high > p_low:
        clipped = np.clip(result, p_low, p_high)
        u16 = ((clipped - p_low) / (p_high - p_low) * 65535).astype(np.uint16)
    else:
        u16 = np.zeros_like(result, dtype=np.uint16)
    tifffile.imwrite(
        path, u16, ome=True,
        metadata={"axes": "ZYX", "Channel": {"Name": channel_name}},
    )


def _make_psf(acq, channel):
    """Generate PSF from acquisition metadata and channel wavelength."""
    meta = acq.metadata
    wavelength = wavelength_from_channel(f"Fluorescence {channel} nm Ex")
    ni = infer_immersion_index(meta.na)
    nz_psf, nxy_psf = compute_psf_size(
        meta.nz, meta.dxy, meta.dz, wavelength, meta.na, ni,
    )
    return generate_psf(
        nz=nz_psf, nxy=nxy_psf,
        dxy=meta.dxy, dz=meta.dz,
        wavelength=wavelength, na=meta.na, ni=ni,
    )


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ── Comparison Window ─────────────────────────────────────────────────────

class ComparisonWindow(QWidget):
    """Side-by-side raw vs deconvolved viewer using two LightweightViewer.

    Left viewer (raw) is the "leader" — its FOV slider, Z position, and
    contrast are mirrored to the right viewer (deconvolved) via polling,
    following the pattern from Cephla-Lab/ndviewer.
    """

    _POLL_MS = 80

    def __init__(self, raw_path, deconv_path, channel_name):
        super().__init__()
        from ndviewer_light import LightweightViewer
        from PyQt5.QtCore import QTimer

        self.setWindowTitle(f"PetaKit — Fluorescence {channel_name} nm Ex")
        self.resize(1400, 700)

        main_layout = QVBoxLayout(self)
        viewers_layout = QHBoxLayout()

        # Left: Raw
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Raw"))
        self._left = LightweightViewer()
        left_col.addWidget(self._left, 1)
        viewers_layout.addLayout(left_col, 1)

        # Right: Deconvolved
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Deconvolved"))
        self._right = LightweightViewer()
        right_col.addWidget(self._right, 1)
        viewers_layout.addLayout(right_col, 1)

        main_layout.addLayout(viewers_layout, 1)

        # Show window first so ndv canvas initializes
        self.show()

        # Load datasets
        self._left.load_dataset(raw_path)
        self._right.load_dataset(deconv_path)

        # Default to 3D view for z-stacks
        for viewer in (self._left, self._right):
            if viewer.ndv_viewer:
                viewer.ndv_viewer.display_model.visible_axes = (-3, -2, -1)

        # Hide right viewer's LightweightViewer sliders — left drives both
        self._right._fov_slider_container.setVisible(False)
        self._right._time_container.setVisible(False)

        # Sync: left FOV slider → right viewer
        self._left._fov_slider.valueChanged.connect(self._sync_fov)

        # Track last-seen state for change detection
        self._last_clims = {}
        self._last_visible_axes = None

        # Start polling for Z + contrast + 3D mode sync
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._poll_sync)
        self._sync_timer.start(self._POLL_MS)

    def _sync_fov(self, value):
        """When left FOV changes, load same FOV on right and re-hide sliders."""
        self._right.load_fov(value)
        # ndv_viewer may be recreated after load_fov — re-hide its sliders
        if self._right.ndv_viewer:
            try:
                self._right.ndv_viewer.widget().dims_sliders.hide()
            except Exception:
                pass

    def _poll_sync(self):
        """Mirror left viewer's Z position and contrast to right viewer."""
        left_ndv = getattr(self._left, "ndv_viewer", None)
        right_ndv = getattr(self._right, "ndv_viewer", None)
        if not left_ndv or not right_ndv:
            return

        # ── Position sync (all shared dimensions) ─────────────────────
        try:
            left_idx = dict(left_ndv.display_model.current_index)
            right_idx = right_ndv.display_model.current_index
            for key, val in left_idx.items():
                if key in right_idx and right_idx[key] != val:
                    right_idx[key] = val
        except Exception:
            pass

        # ── Contrast / CLIMs sync ─────────────────────────────────────
        try:
            left_luts = left_ndv.display_model.luts
            right_luts = right_ndv.display_model.luts
            for ch_idx in left_luts:
                if ch_idx not in right_luts:
                    continue
                left_clims = left_luts[ch_idx].clims
                key = repr(left_clims)
                if key != self._last_clims.get(ch_idx):
                    self._last_clims[ch_idx] = key
                    right_luts[ch_idx].clims = left_clims
        except Exception:
            pass

        # ── 3D mode sync (visible_axes) ───────────────────────────────
        try:
            left_va = left_ndv.display_model.visible_axes
            if left_va != self._last_visible_axes:
                self._last_visible_axes = left_va
                right_ndv.display_model.visible_axes = left_va
        except Exception:
            pass

    def closeEvent(self, event):
        self._sync_timer.stop()
        super().closeEvent(event)


# ── Workers ───────────────────────────────────────────────────────────────

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
            ome_dir = self.output_dir / "ome_tiff"
            ome_dir.mkdir(parents=True, exist_ok=True)
            psf = _make_psf(self.acq, self.channel)
            channel_name = f"Fluorescence {self.channel} nm Ex"

            fovs = list(self.acq.iter_fovs())
            t0 = time.perf_counter()
            bytes_processed = 0
            for i, fov in enumerate(fovs):
                self.progress.emit(
                    i, len(fovs), f"[{i+1}/{len(fovs)}] {fov}...",
                )
                stack = self.acq.get_stack(fov, self.channel)
                result = deconvolve(stack, psf, **self.params)
                bytes_processed += result.nbytes

                _save_stack_ome(ome_dir / f"{fov}.ome.tiff", result, channel_name)

                elapsed = time.perf_counter() - t0
                gb_per_min = (bytes_processed / 1e9) / (elapsed / 60)
                remaining = (len(fovs) - i - 1) * (elapsed / (i + 1))
                self.progress.emit(
                    i + 1, len(fovs),
                    f"[{i+1}/{len(fovs)}] {fov}  |  "
                    f"{gb_per_min:.2f} GB/min  |  "
                    f"ETA {_fmt_time(remaining)}",
                )

            elapsed = time.perf_counter() - t0
            gb_total = bytes_processed / 1e9
            gb_per_min = gb_total / (elapsed / 60)
            self.progress.emit(
                len(fovs), len(fovs),
                f"Done — {gb_total:.1f} GB in {_fmt_time(elapsed)} "
                f"({gb_per_min:.2f} GB/min)",
            )
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class PreviewWorker(QThread):
    """Score FOVs, pick top ~5, deconvolve, save raw+deconv as OME-TIFF."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, str)  # (raw_dir, deconv_dir)
    error = pyqtSignal(str)

    def __init__(self, acq, channel, params, num_preview=5):
        super().__init__()
        self.acq = acq
        self.channel = channel
        self.params = params
        self.num_preview = num_preview

    def run(self):
        try:
            import numpy as np
            fovs = list(self.acq.iter_fovs())
            mid_z = self.acq.metadata.nz // 2

            # Score a subsample of FOVs to keep it fast
            import random
            max_to_score = min(len(fovs), 30)
            if len(fovs) > max_to_score:
                candidates = random.sample(fovs, max_to_score)
            else:
                candidates = fovs

            self.progress.emit(f"Scoring {len(candidates)} FOVs...")
            scores = []
            for fov in candidates:
                plane = self.acq.get_plane(fov, self.channel, mid_z)
                score = float(np.mean(plane) * np.std(plane))
                scores.append((score, fov))

            scores.sort(key=lambda x: x[0], reverse=True)
            selected = [fov for _, fov in scores[:self.num_preview]]

            # Generate PSF
            psf = _make_psf(self.acq, self.channel)
            channel_name = f"Fluorescence {self.channel} nm Ex"

            # Create temp dirs with ome_tiff subdirs
            raw_base = tempfile.mkdtemp(prefix="petakit_preview_raw_")
            deconv_base = tempfile.mkdtemp(prefix="petakit_preview_deconv_")
            raw_dir = Path(raw_base) / "ome_tiff"
            deconv_dir = Path(deconv_base) / "ome_tiff"
            raw_dir.mkdir()
            deconv_dir.mkdir()

            for i, fov in enumerate(selected):
                self.progress.emit(
                    f"Preview [{i+1}/{len(selected)}] {fov}..."
                )
                stack = self.acq.get_stack(fov, self.channel)
                result = deconvolve(stack, psf, **self.params)

                _save_stack_ome(raw_dir / f"{fov}.ome.tiff", stack, channel_name)
                _save_stack_ome(deconv_dir / f"{fov}.ome.tiff", result, channel_name)

            self.finished.emit(raw_base, deconv_base)

        except Exception as e:
            self.error.emit(str(e))


class RawExportWorker(QThread):
    """Export raw stacks matching deconvolved output for comparison."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)  # raw_dir
    error = pyqtSignal(str)

    def __init__(self, acq, channel, deconv_dir):
        super().__init__()
        self.acq = acq
        self.channel = channel
        self.deconv_dir = Path(deconv_dir)

    def run(self):
        try:
            channel_name = f"Fluorescence {self.channel} nm Ex"
            raw_base = tempfile.mkdtemp(prefix="petakit_raw_export_")
            raw_dir = Path(raw_base) / "ome_tiff"
            raw_dir.mkdir()

            # Match FOVs that have deconvolved output
            deconv_files = list((self.deconv_dir / "ome_tiff").glob("*.ome.tiff"))
            fovs = list(self.acq.iter_fovs())
            fov_by_name = {str(fov): fov for fov in fovs}

            for i, df in enumerate(deconv_files):
                # Extract FOV name from filename (e.g. "current_0.ome.tiff" → "current_0")
                fov_name = df.stem.replace(".ome", "")
                self.progress.emit(f"Exporting raw [{i+1}/{len(deconv_files)}]...")
                if fov_name in fov_by_name:
                    fov = fov_by_name[fov_name]
                    stack = self.acq.get_stack(fov, self.channel)
                    _save_stack_ome(raw_dir / df.name, stack, channel_name)

            self.finished.emit(raw_base)

        except Exception as e:
            self.error.emit(str(e))


# ── Main Window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PetaKit")
        self.setMinimumWidth(500)

        self.acq = None
        self.worker = None
        self._last_channel = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # GPU status
        self.gpu_label = QLabel(gpu_info())
        self.gpu_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.gpu_label)

        # Acquisition selection (drop zone + browse)
        acq_group = QGroupBox("Acquisition")
        acq_layout = QVBoxLayout(acq_group)

        self.drop_label = QLabel("Drop acquisition folder here\nor click to browse")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setFixedHeight(60)
        self.drop_label.setStyleSheet(
            "QLabel { border: 2px dashed #aaa; border-radius: 6px; "
            "color: #888; background: #fafafa; }"
        )
        self.drop_label.setCursor(Qt.PointingHandCursor)
        self.drop_label.setAcceptDrops(True)
        self.drop_label.mousePressEvent = lambda _: self._browse_acquisition()
        self.drop_label.dragEnterEvent = self._drag_enter
        self.drop_label.dragLeaveEvent = self._drag_leave
        self.drop_label.dropEvent = self._drop
        acq_layout.addWidget(self.drop_label)

        self.info_label = QLabel("")
        acq_layout.addWidget(self.info_label)

        # Preview button (right under drop zone)
        self.preview_btn = QPushButton("Preview (5 FOVs)")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._run_preview)
        acq_layout.addWidget(self.preview_btn)

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

        # View output button (visible after deconvolution completes)
        self.view_btn = QPushButton("View Output")
        self.view_btn.setVisible(False)
        self.view_btn.clicked.connect(self._view_output)
        layout.addWidget(self.view_btn)

        layout.addStretch()

    # ── Params ────────────────────────────────────────────────────────

    def _on_method_changed(self, method):
        if "omw" in method.lower():
            self.iter_spin.setValue(2)
        else:
            self.iter_spin.setValue(15)

    def _get_params(self):
        method = "omw" if "omw" in self.method_combo.currentText().lower() else "rl"
        return {
            "method": method,
            "iterations": self.iter_spin.value(),
            "gpu": not self.nogpu_check.isChecked(),
        }

    # ── Drag and drop ─────────────────────────────────────────────────

    def _drag_enter(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                    event.acceptProposedAction()
                    self.drop_label.setStyleSheet(
                        "QLabel { border: 2px dashed #4a90d9; border-radius: 6px; "
                        "color: #4a90d9; background: #e8f0fe; }"
                    )
                    return
        event.ignore()

    def _drag_leave(self, event):
        self.drop_label.setStyleSheet(
            "QLabel { border: 2px dashed #aaa; border-radius: 6px; "
            "color: #888; background: #fafafa; }"
        )

    def _drop(self, event):
        self._drag_leave(event)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).is_dir():
                self._load_acquisition(path)
                return

    # ── Acquisition ───────────────────────────────────────────────────

    def _browse_acquisition(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Acquisition Folder"
        )
        if path:
            self._load_acquisition(path)

    def _load_acquisition(self, path):
        try:
            self.acq = open_acquisition(path)
            self.drop_label.setText(Path(path).name)
            self.drop_label.setStyleSheet(
                "QLabel { border: 2px solid #4a90d9; border-radius: 6px; "
                "color: #333; background: #e8f0fe; }"
            )

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
            self.preview_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def _select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_label.setText(path)

    # ── Buttons disable/enable ────────────────────────────────────────

    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.preview_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)

    # ── Full deconvolution ────────────────────────────────────────────

    def _run_deconvolution(self):
        if not self.acq:
            return
        self._last_channel = self.channel_combo.currentText()
        self.worker = DeconvolutionWorker(
            self.acq, self._last_channel,
            self.output_label.text(), self._get_params(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_deconv_finished)
        self.worker.error.connect(self._on_error)
        self._set_running(True)
        self.worker.start()

    def _on_progress(self, current, total, message):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_deconv_finished(self):
        self._set_running(False)
        self.view_btn.setVisible(True)

    # ── Preview ───────────────────────────────────────────────────────

    def _run_preview(self):
        if not self.acq:
            return
        self._last_channel = self.channel_combo.currentText()
        self._preview_worker = PreviewWorker(
            self.acq, self._last_channel, self._get_params(),
        )
        self._preview_worker.progress.connect(
            lambda msg: self.status_label.setText(msg)
        )
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_error)
        self._set_running(True)
        self.progress_bar.setMaximum(0)  # indeterminate
        self._preview_worker.start()

    def _on_preview_finished(self, raw_dir, deconv_dir):
        self._set_running(False)
        self.status_label.setText("Preview ready")
        self._preview_window = ComparisonWindow(
            raw_dir, deconv_dir, self._last_channel,
        )

    # ── View output (comparison) ──────────────────────────────────────

    def _view_output(self):
        """Export raw stacks then open side-by-side comparison."""
        output_dir = self.output_label.text()
        if not Path(output_dir).is_dir():
            self.status_label.setText("Output directory not found")
            return
        if not self.acq or not self._last_channel:
            self.status_label.setText("No acquisition loaded")
            return
        try:
            from ndviewer_light import LightweightViewer  # noqa: F401
        except ImportError:
            self.status_label.setText(
                "ndviewer_light not installed — pip install ndviewer-light"
            )
            return

        self._raw_export_worker = RawExportWorker(
            self.acq, self._last_channel, output_dir,
        )
        self._raw_export_worker.progress.connect(
            lambda msg: self.status_label.setText(msg)
        )
        self._raw_export_worker.finished.connect(
            lambda raw_dir: self._open_comparison(raw_dir, output_dir)
        )
        self._raw_export_worker.error.connect(self._on_error)
        self.status_label.setText("Exporting raw data for comparison...")
        self.view_btn.setEnabled(False)
        self._raw_export_worker.start()

    def _open_comparison(self, raw_dir, deconv_dir):
        self.view_btn.setEnabled(True)
        self.status_label.setText("")
        self._comparison_window = ComparisonWindow(
            raw_dir, deconv_dir, self._last_channel,
        )

    # ── Error handling ────────────────────────────────────────────────

    def _on_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self._set_running(False)


def main():
    """GUI entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
