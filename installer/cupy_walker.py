"""Trace which CuPy/CuPyX modules petakit actually uses at runtime.

Outputs two things:
  1. The list of hiddenimports for the PyInstaller spec
  2. The list of .so files that can be excluded (not loaded)

Usage:
  python installer/cupy_walker.py
"""
import os
import sys


def walk():
    """Import cupy the way petakit.engine does, return used modules."""
    before = set(sys.modules.keys())

    import numpy as np
    import cupy as cp
    import cupyx.scipy.fft as cufft

    # Reproduce every cupy call from petakit/engine.py
    arr = cp.asarray(np.zeros((4, 4, 4), dtype=np.float32))
    _ = cp.maximum(arr, 0)
    _ = cp.zeros((4, 4, 4), dtype=cp.float32)
    _ = cp.zeros_like(arr)
    _ = cp.real(arr)
    _ = cp.conj(arr)
    flat = arr.ravel()
    _ = cp.dot(flat, flat)
    _ = cp.finfo(cp.float32)
    _ = cp.pad(arr, [(0, 1), (0, 1), (0, 1)])
    _ = cp.roll(arr, [1, 1, 1], axis=(0, 1, 2))
    result = cufft.fftn(arr)
    _ = cufft.ifftn(result)
    _ = cp.asnumpy(arr)
    _ = cp.cuda.runtime.getDeviceCount()

    after = set(sys.modules.keys())
    cupy_mods = sorted(m for m in (after - before)
                       if m.startswith(('cupy', 'cupyx')))
    return cupy_mods


def find_excludable_sos(used_modules):
    """Find .so files NOT loaded by the traced modules."""
    import cupy
    import cupy_backends

    loaded_sos = set()
    for name in used_modules:
        mod = sys.modules.get(name)
        if mod:
            f = getattr(mod, '__file__', None)
            if f and f.endswith(('.so', '.pyd')):
                loaded_sos.add(os.path.abspath(f))

    cupy_dir = os.path.dirname(cupy.__file__)
    cb_dir = os.path.dirname(cupy_backends.__file__)

    all_sos = []
    for d in [cupy_dir, cb_dir]:
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith(('.so', '.pyd')):
                    all_sos.append(os.path.abspath(os.path.join(root, f)))

    not_loaded = sorted(set(all_sos) - loaded_sos)
    return not_loaded, loaded_sos


def main():
    print("=== CuPy Import Walker for PetaKit ===\n")

    used = walk()
    print(f"Traced {len(used)} cupy/cupyx modules.\n")

    print("--- hiddenimports for .spec ---")
    for m in used:
        print(f"    '{m}',")

    not_loaded, loaded = find_excludable_sos(used)

    loaded_size = sum(os.path.getsize(f) for f in loaded)
    excluded_size = sum(os.path.getsize(f) for f in not_loaded)

    print(f"\n--- Bundle: {len(loaded)} .so files ({loaded_size/1e6:.0f} MB) ---")
    print(f"--- Exclude: {len(not_loaded)} .so files ({excluded_size/1e6:.0f} MB) ---\n")

    if not_loaded:
        print("Excludable .so basenames (for spec excludes list):")
        for f in not_loaded:
            basename = os.path.basename(f)
            sz = os.path.getsize(f)
            print(f"    '{basename}',  # {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
