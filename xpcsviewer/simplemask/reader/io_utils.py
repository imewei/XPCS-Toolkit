"""Raw-frame IO helpers shared across format loaders.

ponytail: multiprocessing-only frame averaging. Upstream pySimpleMask adds a
free-threaded-Python (no-GIL) fast path that bypasses HDF5 entirely via
os.pread + a ctypes LZ4 binding for LZ4-chunked datasets. That optimization
is tied to their specific production filesystem and HDF5 filter setup; add
it here if profiling on this project's actual deployment shows the
multiprocessing.Pool path below is a real bottleneck.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression plugins
import numpy as np

logger = logging.getLogger(__name__)


def _cast_to_signed(arr: np.ndarray) -> np.ndarray:
    """Cast an unsigned integer array to its signed counterpart.

    Preserves bit patterns so detector overflow values (e.g. 65535 for
    uint16) become negative rather than staying as large positive counts.
    """
    if arr.dtype.kind == "u":
        return arr.astype(np.dtype(f"int{arr.dtype.itemsize * 8}"))
    return arr


def process_chunk(file_path, dataset_name, start_idx, end_idx):
    """Return the per-pixel float32 sum over ``[start_idx, end_idx)`` of a dataset."""
    with h5py.File(file_path, "r") as f:
        chunk = _cast_to_signed(f[dataset_name][start_idx:end_idx])
        return np.sum(chunk, axis=0, dtype=np.float32)


def resolve_frame_range(total_frames, start_frame, num_frames):
    """Clamp a requested frame range to what the dataset contains.

    ``num_frames`` semantics, shared by every format loader:
      * ``> 0`` -- exactly that many frames (clamped to the end);
      * ``0`` / ``None`` -- all remaining frames from ``start_frame``;
      * ``< 0`` -- a representative subset (``max(1000, total_frames // 5)``).

    Returns the number of frames to read, starting at ``start_frame``.
    """
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame must be between 0 and {total_frames - 1}")
    if num_frames is None or num_frames == 0:
        num_frames = total_frames - start_frame
    elif num_frames < 0:
        num_frames = max(1000, total_frames // 5)
    if start_frame + num_frames > total_frames:
        num_frames = total_frames - start_frame
    return num_frames


def average_frames_parallel(
    file_path,
    dataset_name="/entry/data/data",
    start_frame=0,
    num_frames=-1,
    chunk_size=32,
    num_processes=None,
):
    """Return the per-pixel mean image over a range of frames in a 3-D HDF5 stack.

    See :func:`resolve_frame_range` for ``num_frames`` semantics.

    Returns:
        np.ndarray: 2-D ``float32`` mean image.
    """
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        if dataset.ndim != 3:
            raise ValueError("expected a 3-D (frame, y, x) dataset")

        total_frames = dataset.shape[0]
        logger.info("Total frames in dataset: %d", total_frames)
        num_frames = resolve_frame_range(total_frames, start_frame, num_frames)

        # Small ranges are cheaper to read in a single process.
        if num_frames < chunk_size:
            frames = _cast_to_signed(dataset[start_frame : start_frame + num_frames])
            return (np.sum(frames, axis=0, dtype=np.float32) / num_frames).astype(
                np.float32
            )

        if num_processes is None:
            num_processes = max(1, cpu_count() // 2)

        stop = start_frame + num_frames
        chunks = [
            (file_path, dataset_name, i, min(i + chunk_size, stop))
            for i in range(start_frame, stop, chunk_size)
        ]
    # h5py file is closed here; all needed metadata is in local variables.

    num_processes = min(len(chunks), num_processes)
    logger.info("using %d cores to load %d frames", num_processes, num_frames)
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, chunks)

    return (sum(results) / num_frames).astype(np.float32)
