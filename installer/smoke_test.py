"""Post-freeze smoke tests for the bundled PetaKit application."""
import os
import sys
import tempfile


def _test(name, fn):
    """Run a single test, print PASS/FAIL, return success bool."""
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name} -- {e}")
        return False


def run():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    results = []

    def t_numpy():
        import numpy as np
        arr = np.arange(12).reshape(3, 4)
        assert arr.sum() == 66

    def t_scipy():
        from scipy.ndimage import zoom
        import numpy as np
        result = zoom(np.ones((10, 10)), 0.5)
        assert result.shape == (5, 5)

    def t_tifffile():
        import numpy as np
        import tifffile
        arr = np.zeros((10, 10), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, arr)
            data = tifffile.imread(path)
            assert data.shape == (10, 10)
        finally:
            os.unlink(path)

    def t_psfmodels():
        import psfmodels  # noqa: F401

    def t_petakit():
        import petakit  # noqa: F401
        from petakit.readers import open_acquisition  # noqa: F401
        from petakit.psf import generate_psf, compute_psf_size  # noqa: F401
        from petakit.core import deconvolve  # noqa: F401

    def t_ndviewer():
        from ndviewer_light import LightweightViewer  # noqa: F401

    def t_ndv():
        import ndv  # noqa: F401

    def t_vispy():
        import vispy  # noqa: F401

    def t_pyqt5():
        from PyQt5.QtWidgets import QApplication  # noqa: F401

    def t_psutil():
        import psutil  # noqa: F401

    def t_cupy():
        import cupy as cp
        import cupyx.scipy.fft as cufft  # noqa: F401
        # Just verify import — GPU may not be present on CI
        cp.cuda.runtime  # noqa: B018

    tests = [
        ("import numpy", t_numpy),
        ("scipy.ndimage zoom", t_scipy),
        ("tifffile read/write", t_tifffile),
        ("import psfmodels", t_psfmodels),
        ("petakit core imports", t_petakit),
        ("ndviewer_light", t_ndviewer),
        ("import ndv", t_ndv),
        ("import vispy", t_vispy),
        ("PyQt5 QApplication", t_pyqt5),
        ("import psutil", t_psutil),
        ("cupy + cupyx.scipy.fft", t_cupy),
    ]

    for name, fn in tests:
        results.append(_test(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} smoke tests passed.")
    sys.exit(0 if all(results) else 1)
