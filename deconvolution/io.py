"""
I/O utilities for loading and saving image stacks.
Supports lazy loading to avoid memory issues with large datasets.
"""

import os
import re
import numpy as np
import torch
from pathlib import Path
from .utils import to_tensor, to_numpy


class LazyTiffStack:
    """
    Lazy-loading TIFF stack that only loads slices when accessed.

    This avoids loading entire datasets into memory.
    """

    def __init__(self, files, dtype='float32'):
        """
        Args:
            files: List of paths to TIFF slice files (sorted by Z)
            dtype: Data type for loaded arrays
        """
        import tifffile as tf

        self.files = [Path(f) for f in files]
        self.dtype = dtype
        self._shape = None
        self._slice_shape = None

        # Read first file to get slice shape
        if self.files:
            with tf.TiffFile(str(self.files[0])) as tif:
                page = tif.pages[0]
                self._slice_shape = (page.shape[0], page.shape[1])

    @property
    def shape(self):
        """Return (Z, Y, X) shape without loading data."""
        if self._shape is None and self._slice_shape is not None:
            self._shape = (len(self.files), *self._slice_shape)
        return self._shape

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Load slice(s) on demand."""
        import tifffile as tf

        if isinstance(idx, int):
            # Single slice
            if idx < 0:
                idx = len(self) + idx
            return tf.imread(str(self.files[idx])).astype(self.dtype)

        elif isinstance(idx, slice):
            # Range of slices
            start, stop, step = idx.indices(len(self))
            indices = range(start, stop, step)

            if len(indices) == 0:
                return np.array([], dtype=self.dtype)

            # Load slices one by one
            slices = []
            for i in indices:
                slices.append(tf.imread(str(self.files[i])).astype(self.dtype))

            return np.stack(slices, axis=0)

        elif isinstance(idx, tuple):
            # Handle multi-dimensional indexing (z, y, x)
            z_idx = idx[0]
            rest = idx[1:]

            data = self[z_idx]  # Load Z slices

            if rest and data.size > 0:
                # Apply remaining indices
                if data.ndim == 2:
                    return data[rest]
                else:
                    return data[(slice(None),) + rest]
            return data

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def load_chunk(self, z_start, z_end):
        """Load a chunk of Z slices into memory as a tensor."""
        chunk = self[z_start:z_end]
        return to_tensor(chunk)

    def iter_chunks(self, chunk_size, overlap=0):
        """
        Iterate over Z-chunks with optional overlap.

        Yields:
            (z_start, z_end, chunk_tensor)
        """
        n_slices = len(self)
        z = 0

        while z < n_slices:
            z_end = min(z + chunk_size, n_slices)
            chunk = self.load_chunk(z, z_end)
            yield z, z_end, chunk

            z = z_end - overlap
            if z_end == n_slices:
                break


class TiffStackWriter:
    """
    Streaming TIFF stack writer that writes slices incrementally.
    """

    def __init__(self, path, shape=None, dtype='float32', imagej=True):
        """
        Args:
            path: Output file path
            shape: Expected (Z, Y, X) shape (optional, for pre-allocation)
            dtype: Output data type
            imagej: Write ImageJ-compatible metadata
        """
        import tifffile as tf

        self.path = Path(path)
        self.dtype = dtype
        self.imagej = imagej
        self._writer = None
        self._slices_written = 0
        self._min_val = float('inf')
        self._max_val = float('-inf')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write_slice(self, data):
        """Write a single 2D slice."""
        import tifffile as tf

        data = to_numpy(data).astype(self.dtype)

        # Track min/max for metadata
        self._min_val = min(self._min_val, float(np.percentile(data, 1)))
        self._max_val = max(self._max_val, float(np.percentile(data, 99)))

        if self._writer is None:
            self._writer = tf.TiffWriter(str(self.path), bigtiff=True)

        self._writer.write(data, contiguous=True)
        self._slices_written += 1

    def write_chunk(self, data):
        """Write a 3D chunk (multiple slices)."""
        data = to_numpy(data)

        if data.ndim == 2:
            self.write_slice(data)
        else:
            for i in range(data.shape[0]):
                self.write_slice(data[i])

    def close(self):
        """Close the writer and finalize the file."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def open_tiff_stack(path, channels=None, pattern=None):
    """
    Open a TIFF stack for lazy loading (does not load into memory).

    Args:
        path: Path to directory containing TIFF slices
        channels: List of channel identifiers to include (e.g., ['488'])
        pattern: Regex pattern to filter files

    Returns:
        dict: Mapping from channel name to LazyTiffStack
              or single LazyTiffStack if only one channel
    """
    path = Path(path)

    if path.is_file():
        # Single file - wrap in list
        return LazyTiffStack([path])

    if not path.is_dir():
        raise ValueError(f"Path does not exist: {path}")

    # Find TIFF files
    files = sorted([f for f in path.iterdir() if f.suffix.lower() in ('.tif', '.tiff')])

    if not files:
        raise ValueError(f"No TIFF files found in {path}")

    if pattern:
        regex = re.compile(pattern)
        files = [f for f in files if regex.search(f.name)]

    # Group by channel
    channel_files = {}
    for f in files:
        match = re.search(r'(\d{3})_nm', f.name)
        ch = match.group(1) if match else 'default'

        if channels is None or ch in channels:
            if ch not in channel_files:
                channel_files[ch] = []
            channel_files[ch].append(f)

    # Create lazy stacks
    result = {ch: LazyTiffStack(sorted(ch_files))
              for ch, ch_files in channel_files.items()}

    if len(result) == 1:
        return list(result.values())[0]

    return result


def load_tiff_stack(path, pattern=None, channels=None, z_range=None, dtype='float32'):
    """
    Load a TIFF stack into memory.

    WARNING: For large datasets, use open_tiff_stack() instead for lazy loading.

    Args:
        path: Path to a single TIFF file or directory containing TIFF slices
        pattern: Regex pattern to filter files (for directory mode)
        channels: List of channel identifiers to load (e.g., ['488', '561'])
        z_range: Tuple (start, end) to load subset of z-slices
        dtype: Output data type

    Returns:
        torch.Tensor or dict of tensors
    """
    import tifffile as tf

    path = Path(path)

    if path.is_file():
        data = tf.imread(str(path)).astype(dtype)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if z_range is not None:
            data = data[z_range[0]:z_range[1]]
        return to_tensor(data)

    # Use lazy loading then load requested range
    stacks = open_tiff_stack(path, channels=channels, pattern=pattern)

    if isinstance(stacks, dict):
        result = {}
        for ch, stack in stacks.items():
            if z_range is not None:
                result[ch] = stack.load_chunk(z_range[0], z_range[1])
            else:
                result[ch] = stack.load_chunk(0, len(stack))
        return result if len(result) > 1 else list(result.values())[0]
    else:
        if z_range is not None:
            return stacks.load_chunk(z_range[0], z_range[1])
        else:
            return stacks.load_chunk(0, len(stacks))


def save_tiff_stack(path, data, imagej=True, metadata=None):
    """
    Save a tensor as a TIFF stack.

    Args:
        path: Output file path
        data: Tensor or numpy array (Z, Y, X) or (T, Z, Y, X) or list of tensors
        imagej: If True, save in ImageJ-compatible format
        metadata: Optional metadata dict
    """
    import tifffile as tf

    # Convert to numpy
    if isinstance(data, list):
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data)
        data = to_numpy(data)
    else:
        data = to_numpy(data)

    data = np.asarray(data).astype('float32')

    # Ensure 5D for ImageJ (TZCYX)
    while data.ndim < 5:
        data = np.expand_dims(data, -3)

    # Compute display range
    lo, hi = np.percentile(data[-1], [1, 99])

    meta = {'min': lo, 'max': hi}
    if metadata:
        meta.update(metadata)

    tf.imwrite(str(path), data, imagej=imagej, metadata=meta)


def get_stack_info(path, channels=None):
    """
    Get information about a TIFF stack without loading it.

    Args:
        path: Path to directory or file
        channels: Optional channel filter

    Returns:
        dict with shape, dtype, num_files, etc.
    """
    stacks = open_tiff_stack(path, channels=channels)

    if isinstance(stacks, dict):
        return {ch: {'shape': stack.shape, 'num_files': len(stack)}
                for ch, stack in stacks.items()}
    else:
        return {'shape': stacks.shape, 'num_files': len(stacks)}
