"""Pure Python deconvolution engine (PetaKit5D algorithms).

Implements Richardson-Lucy with Biggs-Andrews acceleration and
OTF-Masked Wiener (OMW) back projector generation.

GPU-first by default: tries CuPy, falls back to CPU silently.
Matches PetaKit5D's useGPU=true + gpuDeviceCount strategy.

References:
    PetaKit5D — Ruan et al., Nature Methods 2024
    decon_lucy_function.m, decon_lucy_omw_function.m,
    omw_backprojector_generation.m, decon_psf2otf.m
"""

import gc

import numpy as np
import scipy.fft
from numpy.fft import fftshift, ifftshift
from scipy.ndimage import (
    binary_opening,
    binary_closing,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label,
)


# ── GPU backend ──────────────────────────────────────────────────────


def _get_array_module(gpu=True):
    """GPU-first, CPU fallback. Matches PetaKit5D's useGPU=true default.

    Args:
        gpu: If True, try CuPy/GPU. If False, force CPU.

    Returns:
        (xp, fft_mod) — array module and FFT module.
    """
    if gpu:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                import cupyx.scipy.fft as cufft

                return cp, cufft
        except Exception:
            pass
    return np, scipy.fft


def _to_numpy(arr, xp):
    """Move array back to CPU numpy."""
    if xp is np:
        return arr
    return xp.asnumpy(arr)


def gpu_info():
    """Return GPU status string for display."""
    try:
        import cupy as cp

        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            name = cp.cuda.runtime.getDeviceProperties(0)['name']
            if isinstance(name, bytes):
                name = name.decode()
            return f"GPU: {name} (detected)"
    except Exception:
        pass
    return "GPU: not available (using CPU)"


# ── PSF/OTF utilities ────────────────────────────────────────────────


def crop_psf_to_image(psf, im_shape):
    """Center-crop PSF if larger than image in any dimension."""
    p_shape = np.array(psf.shape)
    i_shape = np.array(im_shape)
    if np.all(p_shape <= i_shape):
        return psf
    slices = []
    for p, i in zip(p_shape, i_shape):
        if p > i:
            start = (p - i) // 2
            slices.append(slice(start, start + i))
        else:
            slices.append(slice(None))
    return psf[tuple(slices)]


def psf2otf(psf, out_shape, fft_func):
    """Convert PSF to OTF: zero-pad, circshift center to origin, FFT.

    Port of PetaKit5D's decon_psf2otf.m.
    """
    psf = crop_psf_to_image(psf, out_shape)
    xp = type(psf).__module__
    if xp == "cupy":
        import cupy

        psf_shape = cupy.array(psf.shape)
        pad_size = cupy.array(out_shape) - psf_shape
        padded = cupy.pad(psf, [(0, int(p)) for p in pad_size])
        shift = -(psf_shape // 2)
        padded = cupy.roll(padded, shift.astype(int).tolist(), axis=(0, 1, 2))
    else:
        psf_shape = np.array(psf.shape)
        pad_size = np.array(out_shape) - psf_shape
        padded = np.pad(psf, [(0, int(p)) for p in pad_size])
        shift = -(psf_shape // 2)
        padded = np.roll(padded, shift.astype(int), axis=(0, 1, 2))
    return fft_func(padded)


# ── Richardson-Lucy with Biggs-Andrews acceleration ──────────────────


def _rl_core(image, psf, n_iter=15, gpu=True, verbose=False):
    """RL core — processes a single volume that fits in memory."""
    xp, fft_mod = _get_array_module(gpu)

    image = xp.maximum(xp.asarray(image, dtype=xp.float32), 0)
    psf_np = np.asarray(psf, dtype=np.float32) if xp is not np else psf.astype(np.float32)
    psf_np = psf_np / psf_np.sum()
    psf_d = xp.asarray(psf_np)
    out_shape = image.shape

    H = psf2otf(psf_d, out_shape, fft_mod.fftn)

    J_2 = image.copy()
    J_3 = xp.zeros_like(image)
    J_4 = xp.zeros(image.size, dtype=xp.float32)
    lam = 0.0
    eps = float(xp.finfo(xp.float32).eps)
    Y = None

    for k in range(1, n_iter + 1):
        if k > 2 and Y is not None:
            diff = (J_2 - Y).ravel()
            dot_num = float(xp.dot(diff, J_4))
            dot_den = float(xp.dot(J_4, J_4))
            lam = max(min(dot_num / (dot_den + eps), 1.0), 0.0)
            J_4 = diff
        elif k == 2 and Y is not None:
            J_4 = (J_2 - Y).ravel()

        Y = xp.maximum(J_2 + lam * (J_2 - J_3), 0)

        # Forward: convolve estimate with PSF
        ReBlurred = xp.maximum(xp.real(fft_mod.ifftn(H * fft_mod.fftn(Y))), eps)
        # Ratio + backward: convolve with conj(OTF)
        ratio = image / ReBlurred
        J_3 = J_2.copy()
        J_2 = xp.maximum(Y * xp.real(fft_mod.ifftn(xp.conj(H) * fft_mod.fftn(ratio))), 0)

        if verbose:
            print(f"  RL iter {k}/{n_iter}")

    result = _to_numpy(J_2, xp)
    return result.astype(np.float32)


def rl(image, psf, n_iter=15, gpu=True, verbose=False, avail_memory_gb=None):
    """Accelerated Richardson-Lucy deconvolution.

    Port of PetaKit5D's decon_lucy_function.m with Biggs-Andrews
    lambda extrapolation for faster convergence.
    Automatically tiles along Z if the volume exceeds available memory.

    Args:
        image: 3D array (Z, Y, X), float
        psf: 3D PSF array, will be normalized to sum=1
        n_iter: Number of iterations (default 15)
        gpu: Try GPU, fall back to CPU (default True)
        verbose: Print per-iteration progress
        avail_memory_gb: Override available memory (GB) for tiling decisions.
            None means auto-detect.

    Returns:
        Deconvolved image as float32 (Z, Y, X), CPU numpy array.
    """
    peak = _estimate_peak_gb(image.shape)
    avail = avail_memory_gb if avail_memory_gb is not None else _available_memory_gb()

    if peak < avail * 0.8:
        return _rl_core(image, psf, n_iter=n_iter, gpu=gpu, verbose=verbose)

    if verbose:
        print(f"  Volume needs ~{peak:.1f} GB, available ~{avail:.1f} GB — tiling")

    return _tile_z(image, psf, _rl_core, avail_memory_gb=avail,
                   verbose=verbose, n_iter=n_iter, gpu=gpu)


# ── OMW back projector generation (always CPU) ───────────────────────


def _omw_backprojector(psf, alpha=0.005, otf_cum_thresh=0.9,
                       hann_bounds=(0.8, 1.0)):
    """Generate OTF-masked Wiener back projector PSF.

    Always runs on CPU (scipy.ndimage morphology ops).
    Port of PetaKit5D's omw_backprojector_generation.m.

    Args:
        psf: 3D PSF array (CPU numpy), normalized to sum=1.
        alpha: Wiener regularization parameter.
        otf_cum_thresh: Cumulative OTF intensity threshold for support mask.
        hann_bounds: (lower, upper) bounds for cosine-squared apodization.

    Returns:
        Back projector PSF as float32 array (same shape as input PSF).
    """
    from numpy.fft import fftn, ifftn

    psf = psf.astype(np.float32)
    psf = psf / psf.sum()

    OTF = fftn(ifftshift(psf))
    abs_OTF = np.abs(OTF)
    abs_OTF_c = fftshift(abs_OTF)

    # Segment OTF support by cumulative intensity
    OTF_vals = np.sort(abs_OTF.ravel())[::-1]
    cum = np.cumsum(OTF_vals.astype(np.float64))
    total = cum[-1]
    tind = np.searchsorted(cum, total * otf_cum_thresh)
    dc_thresh = max(OTF_vals[tind] / abs_OTF.ravel()[0], 1e-3)

    OTF_mask = abs_OTF_c > abs_OTF.ravel()[0] * dc_thresh

    # Morphological cleanup
    struct = generate_binary_structure(3, 2)
    OTF_mask = binary_opening(OTF_mask, struct, iterations=2)
    OTF_mask = binary_closing(OTF_mask, struct, iterations=2)
    OTF_mask = binary_opening(OTF_mask, struct, iterations=2)

    # Keep largest connected component
    labeled, n_labels = label(OTF_mask)
    if n_labels > 1:
        sizes = [np.sum(labeled == i) for i in range(1, n_labels + 1)]
        largest = np.argmax(sizes) + 1
        OTF_mask = labeled == largest

    if not np.any(OTF_mask):
        OTF_mask = abs_OTF_c > abs_OTF.ravel()[0] * 1e-4

    # Distance-based apodization (cosine-squared window)
    dist_inside = distance_transform_edt(OTF_mask)
    max_dist = dist_inside.max()
    if max_dist > 0:
        bw_dist = 1.0 - dist_inside / max_dist
    else:
        bw_dist = np.ones_like(dist_inside)

    l, u = hann_bounds
    mask = np.where(
        bw_dist < l,
        1.0,
        np.where(
            bw_dist < u,
            np.cos(np.pi * (bw_dist - l) / 2 / (u - l)) ** 2,
            0.0,
        ),
    )
    mask_shifted = ifftshift(mask).astype(np.float32)

    # Wiener filter × mask
    OTF_bp_w = np.conj(OTF) / (abs_OTF ** 2 + alpha)
    OTF_bp_omw = mask_shifted * OTF_bp_w

    bp = fftshift(np.real(ifftn(OTF_bp_omw)))
    return bp.astype(np.float32)


# ── Z-tiling for large volumes ───────────────────────────────────────


def _estimate_peak_gb(shape):
    """Estimate peak memory (GB) for OMW/RL on a volume of given shape.

    Simultaneous arrays at peak: image + J (float32) +
    OTF_f + OTF_b + 2 FFT temporaries (complex64).
    scipy.fft preserves float32 -> complex64 (8 bytes per element).
    """
    n = int(np.prod(shape))
    # 2 real(f32=4B) + 4 complex(c64=8B) = 40 bytes/voxel
    return n * 40 / 1e9


def _available_memory_gb():
    """Estimate available system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        pass
    import os
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / 1e9 * 0.7  # use 70% of total
        except (ValueError, OSError):
            pass
    return 8.0  # conservative default


def _psf_support_z(psf, threshold=0.01):
    """Measure PSF support in Z (number of slices with significant energy)."""
    z_profile = np.sum(psf, axis=(1, 2))
    z_profile = z_profile / z_profile.max()
    above = np.where(z_profile > threshold)[0]
    if len(above) == 0:
        return psf.shape[0]
    return above[-1] - above[0] + 1


def _tile_z(image, psf, deconv_func, avail_memory_gb=None, verbose=False, **kwargs):
    """Process large volume by tiling along Z with overlap.

    Border is based on the PSF's actual support in Z (where energy > 1%),
    not the full padded PSF dimension.
    """
    nz, ny, nx = image.shape
    avail = avail_memory_gb if avail_memory_gb is not None else _available_memory_gb()
    budget = avail * 0.7  # 30% headroom

    # Find max total tile Z that fits in memory budget
    max_tile_z = nz
    while max_tile_z > 4:
        if _estimate_peak_gb((max_tile_z, ny, nx)) < budget:
            break
        max_tile_z = max_tile_z // 2

    if max_tile_z >= nz:
        return deconv_func(image, psf, verbose=verbose, **kwargs)

    # Allocate border from the tile budget (up to 1/3 of tile for overlap)
    support_z = _psf_support_z(psf)
    ideal_border = max(support_z // 2 + 2, 3)
    border = min(ideal_border, max_tile_z // 3)
    chunk_nz = max(max_tile_z - 2 * border, 2)

    if verbose:
        print(f"  Z-tiling: {nz} slices -> chunks of {chunk_nz} (border={border})")

    result = np.zeros_like(image, dtype=np.float32)
    z = 0
    tile_idx = 0
    while z < nz:
        z_end = min(z + chunk_nz, nz)

        # Expand tile with overlap borders
        z_load_start = max(z - border, 0)
        z_load_end = min(z_end + border, nz)

        tile = image[z_load_start:z_load_end]

        if verbose:
            print(f"  Tile {tile_idx}: z=[{z_load_start}:{z_load_end}] "
                  f"({tile.shape[0]} slices, {tile.nbytes/1e9:.2f} GB)")

        deconv_tile = deconv_func(tile, psf, verbose=verbose, **kwargs)

        # Trim borders and place into result
        trim_start = z - z_load_start
        trim_end = trim_start + (z_end - z)
        result[z:z_end] = deconv_tile[trim_start:trim_end]

        del tile, deconv_tile
        gc.collect()

        z = z_end
        tile_idx += 1

    return result


# ── OMW deconvolution ────────────────────────────────────────────────


def _omw_core(image, psf, n_iter=2, alpha=0.005, otf_cum_thresh=0.9,
              hann_bounds=(0.8, 1.0), gpu=True, verbose=False):
    """OMW core — processes a single volume that fits in memory."""
    xp, fft_mod = _get_array_module(gpu)

    # Normalize PSF on CPU
    psf_np = np.asarray(psf, dtype=np.float32) if xp is not np else psf.astype(np.float32)
    psf_np = psf_np / psf_np.sum()

    # Back projector: always CPU (scipy.ndimage)
    if verbose:
        print("  Computing OMW back projector...")
    bp_psf = _omw_backprojector(psf_np, alpha, otf_cum_thresh, hann_bounds)

    # Move to device
    image_d = xp.maximum(xp.asarray(image, dtype=xp.float32), 0)
    OTF_f = psf2otf(xp.asarray(psf_np), image.shape, fft_mod.fftn)
    OTF_b = psf2otf(xp.asarray(bp_psf), image.shape, fft_mod.fftn)

    J = image_d.copy()
    eps = float(xp.finfo(xp.float32).eps)

    for k in range(1, n_iter + 1):
        CX = xp.maximum(xp.real(fft_mod.ifftn(fft_mod.fftn(J) * OTF_f)), eps)
        J = xp.maximum(xp.real(fft_mod.ifftn(fft_mod.fftn(image_d / CX) * OTF_b)) * J, 0)

        if verbose:
            print(f"  OMW iter {k}/{n_iter}")

    result = _to_numpy(J, xp)
    return result.astype(np.float32)


def omw(image, psf, n_iter=2, alpha=0.005, otf_cum_thresh=0.9,
        hann_bounds=(0.8, 1.0), gpu=True, verbose=False, avail_memory_gb=None):
    """Richardson-Lucy with OTF-Masked Wiener back projector.

    Port of PetaKit5D's decon_lucy_omw_function.m.
    Back projector computed once on CPU, RL iterations on GPU if available.
    Automatically tiles along Z if the volume exceeds available memory.

    Args:
        image: 3D array (Z, Y, X), float
        psf: 3D PSF array, will be normalized to sum=1
        n_iter: Number of iterations (default 2, OMW converges fast)
        alpha: Wiener regularization for back projector
        otf_cum_thresh: OTF cumulative threshold for mask
        hann_bounds: Apodization bounds
        gpu: Try GPU, fall back to CPU (default True)
        verbose: Print progress
        avail_memory_gb: Override available memory (GB) for tiling decisions.
            None means auto-detect.

    Returns:
        Deconvolved image as float32 (Z, Y, X), CPU numpy array.
    """
    peak = _estimate_peak_gb(image.shape)
    avail = avail_memory_gb if avail_memory_gb is not None else _available_memory_gb()

    if peak < avail * 0.8:
        return _omw_core(image, psf, n_iter=n_iter, alpha=alpha,
                         otf_cum_thresh=otf_cum_thresh, hann_bounds=hann_bounds,
                         gpu=gpu, verbose=verbose)

    if verbose:
        print(f"  Volume needs ~{peak:.1f} GB, available ~{avail:.1f} GB — tiling")

    return _tile_z(image, psf, _omw_core, avail_memory_gb=avail,
                   verbose=verbose, n_iter=n_iter,
                   alpha=alpha, otf_cum_thresh=otf_cum_thresh,
                   hann_bounds=hann_bounds, gpu=gpu)
