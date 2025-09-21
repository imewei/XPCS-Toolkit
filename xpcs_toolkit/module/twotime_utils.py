from __future__ import annotations

# Standard library imports
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import Pool

# Third-party imports
import h5py
import numpy as np

from xpcs_toolkit.utils.logging_config import get_logger

# Local imports
from ..fileIO.aps_8idi import key as key_map
from ..fileIO.hdf_reader import _connection_pool

key_map = key_map["nexus"]
logger = get_logger(__name__)

# Global process pool for reuse
_process_pool = None
_pool_size = None
_shared_arrays = {}


def get_optimal_worker_count() -> int:
    """Get optimal number of worker processes based on system resources."""
    cpu_count = mp.cpu_count()
    # Use 75% of available CPUs, minimum 1, maximum 16
    optimal = max(1, min(16, int(cpu_count * 0.75)))
    logger.debug(f"Optimal worker count: {optimal} (CPU count: {cpu_count})")
    return optimal


def get_process_pool(num_workers: int | None = None) -> ProcessPoolExecutor:
    """
    Get or create a reusable process pool.

    Args:
        num_workers: Number of worker processes. If None, uses optimal count.

    Returns:
        ProcessPoolExecutor instance
    """
    global _process_pool, _pool_size

    if num_workers is None:
        num_workers = get_optimal_worker_count()

    # Create new pool if none exists or size changed
    if _process_pool is None or _pool_size != num_workers:
        if _process_pool is not None:
            _process_pool.shutdown(wait=False)

        _process_pool = ProcessPoolExecutor(max_workers=num_workers)
        _pool_size = num_workers
        logger.info(f"Created process pool with {num_workers} workers")

    return _process_pool


def shutdown_process_pool():
    """Shutdown the global process pool."""
    global _process_pool, _pool_size
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None
        _pool_size = None
        logger.info("Process pool shutdown")


def create_shared_array(
    name: str, shape: tuple[int, ...], dtype=np.float32
) -> np.ndarray:
    """
    Create a shared memory array for multiprocessing.

    Args:
        name: Unique name for the array
        shape: Shape of the array
        dtype: Data type

    Returns:
        Numpy array backed by shared memory
    """
    try:
        from multiprocessing import shared_memory

        # Calculate size
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

        # Create numpy array
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Store reference to prevent garbage collection
        _shared_arrays[name] = (shm, array)

        logger.debug(f"Created shared array '{name}' with shape {shape}")
        return array

    except ImportError:
        # Fallback to regular array if shared_memory not available
        logger.warning("shared_memory not available, using regular arrays")
        return np.empty(shape, dtype=dtype)


def get_shared_array(name: str) -> np.ndarray | None:
    """Get a previously created shared array by name."""
    if name in _shared_arrays:
        return _shared_arrays[name][1]
    return None


def cleanup_shared_arrays():
    """Clean up all shared memory arrays."""
    global _shared_arrays
    for name, (shm, _array) in _shared_arrays.items():
        try:
            shm.close()
            shm.unlink()
            logger.debug(f"Cleaned up shared array '{name}'")
        except Exception as e:
            logger.warning(f"Error cleaning up shared array '{name}': {e}")

    _shared_arrays.clear()


def read_single_c2_enhanced(args: tuple) -> tuple[np.ndarray, int, str]:
    """
    Vectorized version of read_single_c2 with optimized matrix operations.

    Args:
        args: Tuple of (full_path, index_str, max_size, correct_diag, progress_callback)

    Returns:
        Tuple of (c2_matrix, sampling_rate, status_message)
    """
    full_path, index_str, max_size, correct_diag = args[:4]
    progress_callback = args[4] if len(args) > 4 else None

    try:
        c2_prefix = key_map["c2_prefix"]

        if progress_callback:
            progress_callback(f"Reading {index_str}")

        with h5py.File(full_path, "r") as f:
            if f"{c2_prefix}/{index_str}" not in f:
                return None, 0, f"Dataset {index_str} not found"

            c2_half = f[f"{c2_prefix}/{index_str}"][()]

            # Vectorized C2 matrix reconstruction
            c2 = _reconstruct_c2_matrix_vectorized(c2_half)

            # Vectorized sampling if needed
            sampling_rate = 1
            if max_size > 0 and max_size < c2.shape[0]:
                sampling_rate = (c2.shape[0] + max_size - 1) // max_size
                c2 = c2[::sampling_rate, ::sampling_rate]

        # Apply diagonal correction if requested
        if correct_diag:
            c2 = correct_diagonal_c2_vectorized(c2)

        if progress_callback:
            progress_callback(f"Completed {index_str}")

        return c2, sampling_rate, "success"

    except Exception as e:
        error_msg = f"Error reading {index_str}: {e!s}"
        logger.error(error_msg)
        return None, 0, error_msg


def process_c2_batch(
    batch_args: list[tuple], progress_callback: Callable | None = None
) -> list[tuple]:
    """
    Optimized batch processing of C2 matrices with vectorized operations.

    Args:
        batch_args: List of arguments for read_single_c2_enhanced
        progress_callback: Optional progress callback

    Returns:
        List of (c2_matrix, sampling_rate, status) tuples
    """
    results = []

    # Dynamic threshold based on system resources
    optimal_threshold = max(4, get_optimal_worker_count() // 2)

    if len(batch_args) < optimal_threshold:
        # Use sequential processing for small batches with vectorized operations
        for i, args in enumerate(batch_args):
            if progress_callback:
                progress_callback(i, len(batch_args), f"Processing {args[1]}")
            result = read_single_c2_enhanced(args)
            results.append(result)
    else:
        # Use parallel processing for larger batches
        pool = get_process_pool()

        # Batch submit for better resource utilization
        futures = []
        for args in batch_args:
            future = pool.submit(read_single_c2_enhanced, args)
            futures.append((future, args))

        # Collect results as they complete with better error handling
        completed_count = 0
        for future, args in futures:
            if progress_callback:
                progress_callback(
                    completed_count, len(batch_args), f"Completed {args[1]}"
                )

            try:
                result = future.result(timeout=300)  # 5-minute timeout per operation
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel C2 processing for {args[1]}: {e}")
                results.append((None, 0, str(e)))

            completed_count += 1

    return results


def correct_diagonal_c2_vectorized(c2_mat):
    """
    Vectorized diagonal correction for C2 matrices.
    Optimized to eliminate loops and use advanced NumPy operations.
    """
    size = c2_mat.shape[0]
    if size < 2:
        return c2_mat

    # Extract side bands using advanced indexing - more cache-friendly
    upper_diag = np.diag(c2_mat, k=1)  # Upper diagonal
    lower_diag = np.diag(c2_mat, k=-1)  # Lower diagonal

    # Vectorized diagonal value computation
    diag_val = np.zeros(size, dtype=c2_mat.dtype)
    diag_val[:-1] += upper_diag
    diag_val[1:] += lower_diag

    # Vectorized normalization - avoid creating full norm array
    diag_val[0] /= 1.0  # Edge case
    diag_val[-1] /= 1.0  # Edge case
    if size > 2:
        diag_val[1:-1] /= 2.0  # Interior points

    # In-place diagonal assignment for better memory efficiency
    np.fill_diagonal(c2_mat, diag_val)
    return c2_mat


def _reconstruct_c2_matrix_vectorized(c2_half):
    """
    Vectorized C2 matrix reconstruction from half matrix.
    Optimized for memory efficiency and cache performance.
    """
    # Use optimized transpose and addition
    c2 = c2_half + c2_half.T

    # Vectorized diagonal correction - avoid creating index arrays
    diag_vals = np.diag(c2_half)
    np.fill_diagonal(c2, diag_vals)  # Diagonal should not be doubled

    return c2


def read_single_c2(args):
    """
    Optimized single C2 reading with vectorized matrix operations.
    """
    if len(args) == 4:
        # Legacy mode: open file each time
        full_path, index_str, max_size, correct_diag = args
        c2_prefix = key_map["c2_prefix"]
        with _connection_pool.get_connection(full_path, "r") as f:
            c2_half = f[f"{c2_prefix}/{index_str}"][()]
            c2 = _reconstruct_c2_matrix_vectorized(c2_half)
            sampling_rate = 1
            if max_size > 0 and max_size < c2.shape[0]:
                sampling_rate = (c2.shape[0] + max_size - 1) // max_size
                c2 = c2[::sampling_rate, ::sampling_rate]
    else:
        # Optimized mode: reuse file handle
        # Handle both 5+ args (new) and other arg counts
        f, index_str, max_size, correct_diag = args[:4]
        c2_prefix = key_map["c2_prefix"]
        c2_half = f[f"{c2_prefix}/{index_str}"][()]
        c2 = _reconstruct_c2_matrix_vectorized(c2_half)
        sampling_rate = 1
        if max_size > 0 and max_size < c2.shape[0]:
            sampling_rate = (c2.shape[0] + max_size - 1) // max_size
            c2 = c2[::sampling_rate, ::sampling_rate]

    if correct_diag:
        c2 = correct_diagonal_c2_vectorized(c2)
    return c2, sampling_rate


@lru_cache(maxsize=16)
def get_all_c2_from_hdf(
    full_path,
    dq_selection=None,
    max_c2_num=32,
    max_size=512,
    num_workers=12,
    correct_diag=True,
):
    # t0 = time.perf_counter()
    idx_toload = []
    c2_prefix = key_map["c2_prefix"]

    # Use connection pool for single file handle across all operations
    with _connection_pool.get_connection(full_path, "r") as f:
        if c2_prefix not in f:
            return None
        idxlist = list(f[c2_prefix])
        for idx in idxlist:
            if dq_selection is not None and int(idx[4:]) not in dq_selection:
                continue
            idx_toload.append(idx)
            if max_c2_num > 0 and len(idx_toload) > max_c2_num:
                break

        # For multiprocessing, still need to use file paths due to pickling limitations
        # But for single-threaded operation, we can reuse the file handle
        if len(idx_toload) >= 6:
            # Use multiprocessing with legacy approach (file paths)
            args_list = [
                (full_path, index, max_size, correct_diag) for index in idx_toload
            ]
            with Pool(min(len(idx_toload), num_workers)) as p:
                result = p.map(read_single_c2, args_list)
        else:
            # Use single thread with shared file handle for optimization
            args_list = [(f, index, max_size, correct_diag) for index in idx_toload]
            result = [read_single_c2(args) for args in args_list]

    c2_all = np.array([res[0] for res in result])
    sampling_rate_all = {res[1] for res in result}
    assert len(sampling_rate_all) == 1, (
        f"Sampling rate not consistent {sampling_rate_all}"
    )
    sampling_rate = next(iter(sampling_rate_all))
    c2_result = {
        "c2_all": c2_all,
        "delta_t": 1.0 * sampling_rate,  # put absolute time in xpcs_file
        "acquire_period": 1.0,
        "dq_selection": dq_selection,
    }
    return c2_result


def get_all_c2_from_hdf_enhanced(
    full_path,
    dq_selection=None,
    max_c2_num=32,
    max_size=512,
    num_workers=None,
    correct_diag=True,
    progress_callback=None,
):
    """
    Enhanced version of get_all_c2_from_hdf with better multiprocessing and progress reporting.

    Args:
        full_path: Path to HDF5 file
        dq_selection: List of q-indices to load
        max_c2_num: Maximum number of C2 matrices to load
        max_size: Maximum size for C2 matrices (for downsampling)
        num_workers: Number of worker processes (auto-determined if None)
        correct_diag: Whether to apply diagonal correction
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary containing C2 data and metadata
    """
    if progress_callback:
        progress_callback(0, 100, "Scanning HDF5 file structure")

    idx_toload = []
    c2_prefix = key_map["c2_prefix"]

    try:
        with h5py.File(full_path, "r") as f:
            if c2_prefix not in f:
                logger.error(f"C2 prefix '{c2_prefix}' not found in {full_path}")
                return None

            idxlist = list(f[c2_prefix])
            logger.info(f"Found {len(idxlist)} C2 datasets in {full_path}")

            for idx in idxlist:
                if dq_selection is not None and int(idx[4:]) not in dq_selection:
                    continue
                idx_toload.append(idx)
                if max_c2_num > 0 and len(idx_toload) >= max_c2_num:
                    break

        if progress_callback:
            progress_callback(10, 100, f"Selected {len(idx_toload)} datasets to load")

    except Exception as e:
        logger.error(f"Error scanning HDF5 file {full_path}: {e}")
        return None

    if not idx_toload:
        logger.warning(f"No C2 datasets selected for loading from {full_path}")
        return None

    # Prepare arguments for parallel processing
    args_list = [(full_path, index, max_size, correct_diag) for index in idx_toload]

    if progress_callback:
        progress_callback(20, 100, "Starting C2 matrix processing")

    # Process with enhanced parallel function
    try:

        def batch_progress(current, total, message):
            if progress_callback:
                # Scale progress from 20% to 90%
                progress_pct = 20 + int((current / total) * 70)
                progress_callback(progress_pct, 100, message)

        results = process_c2_batch(args_list, batch_progress)

        if progress_callback:
            progress_callback(90, 100, "Assembling results")

        # Filter out failed results and extract data
        successful_results = [
            (c2, sr)
            for c2, sr, status in results
            if c2 is not None and status == "success"
        ]

        if not successful_results:
            logger.error(f"No C2 matrices successfully loaded from {full_path}")
            return None

        failed_count = len(results) - len(successful_results)
        if failed_count > 0:
            logger.warning(f"{failed_count} C2 matrices failed to load")

        # Extract C2 matrices and sampling rates
        c2_matrices = [res[0] for res in successful_results]
        sampling_rates = [res[1] for res in successful_results]

        # Verify sampling rate consistency
        unique_rates = set(sampling_rates)
        if len(unique_rates) != 1:
            logger.warning(f"Inconsistent sampling rates found: {unique_rates}")
            # Use the most common sampling rate
            from collections import Counter

            rate_counts = Counter(sampling_rates)
            sampling_rate = rate_counts.most_common(1)[0][0]
            logger.info(f"Using most common sampling rate: {sampling_rate}")
        else:
            sampling_rate = next(iter(unique_rates))

        # Convert to numpy array
        c2_all = np.array(c2_matrices)

        if progress_callback:
            progress_callback(100, 100, "C2 loading complete")

        c2_result = {
            "c2_all": c2_all,
            "delta_t": 1.0 * sampling_rate,
            "acquire_period": 1.0,
            "dq_selection": dq_selection,
            "loaded_indices": [
                idx_toload[i]
                for i, (_, _, status) in enumerate(results)
                if status == "success"
            ],
            "failed_count": failed_count,
            "total_requested": len(idx_toload),
        }

        logger.info(
            f"Successfully loaded {len(c2_matrices)} C2 matrices from {full_path}"
        )
        return c2_result

    except Exception as e:
        logger.error(f"Error processing C2 matrices from {full_path}: {e}")
        return None


@lru_cache(maxsize=16)
def get_single_c2_from_hdf(
    full_path, selection=0, max_size=512, t0=1, correct_diag=True
):
    c2_prefix = key_map["c2_prefix"]

    # Use connection pool and batch operations
    with _connection_pool.get_connection(full_path, "r") as f:
        if c2_prefix not in f:
            return None
        idxstr = list(f[c2_prefix])[selection]

        # Read C2 data using shared file handle (use 5 args to avoid legacy mode)
        c2_mat, sampling_rate = read_single_c2(
            (f, idxstr, max_size, correct_diag, True)
        )

        # Read G2 partials in the same file operation
        g2_full_key = key_map["c2_g2"]  # Dataset {5000, 25}
        g2_partial_key = key_map["c2_g2_segments"]  # Dataset {1000, 5, 25}

        g2_full = f[g2_full_key][()]
        g2_partial = f[g2_partial_key][()]

        g2_full = np.swapaxes(g2_full, 0, 1)
        g2_partial = np.swapaxes(g2_partial, 0, 2)

    c2_result = {
        "c2_mat": c2_mat,
        "delta_t": t0 * sampling_rate,  # put absolute time in xpcs_file
        "acquire_period": t0,
        "dq_selection": selection,
        "g2_full": g2_full[selection],
        "g2_partial": g2_partial[selection],
    }
    return c2_result


@lru_cache(maxsize=16)
def get_c2_g2partials_from_hdf(full_path):
    # t0 = time.perf_counter()
    c2_prefix = key_map["c2_prefix"]
    g2_full_key = key_map["c2_g2"]  # Dataset {5000, 25}
    g2_partial_key = key_map["c2_g2_segments"]  # Dataset {1000, 5, 25}

    # Use connection pool for optimization
    with _connection_pool.get_connection(full_path, "r") as f:
        if c2_prefix not in f:
            return None
        g2_full = f[g2_full_key][()]
        g2_partial = f[g2_partial_key][()]

    g2_full = np.swapaxes(g2_full, 0, 1)
    g2_partial = np.swapaxes(g2_partial, 0, 2)

    g2_partials = {
        "g2_full": g2_full,
        "g2_partial": g2_partial,
    }
    return g2_partials


def get_c2_stream(full_path, max_size=-1):
    """Returns (idxlist, generator) where the generator yields C2 streams."""
    c2_prefix = key_map["c2_prefix"]

    # Use connection pool for reading the index list
    with _connection_pool.get_connection(full_path, "r") as f:
        if c2_prefix in f:
            idxlist = list(f[c2_prefix])  # Extract the list of indices
        else:
            idxlist = []  # Return empty list if prefix is missing

    def generator():
        for idx in idxlist:  # Use idxlist for iteration
            c2, _sampling_rate = read_single_c2((full_path, idx, max_size, False))
            yield int(idx[3:]), c2

    return idxlist, generator()


def batch_c2_matrix_operations(c2_matrices, operations=None):
    """
    Vectorized batch operations on multiple C2 matrices.

    Args:
        c2_matrices: List or array of C2 matrices
        operations: List of operations to apply ('normalize', 'symmetrize', 'diagonal_correct')

    Returns:
        Processed C2 matrices
    """
    if operations is None:
        operations = ["symmetrize", "diagonal_correct"]

    # Convert to numpy array for vectorized operations
    c2_array = np.array(c2_matrices)

    if "normalize" in operations:
        # Vectorized normalization across all matrices
        norms = np.linalg.norm(c2_array.reshape(c2_array.shape[0], -1), axis=1)
        c2_array = c2_array / norms.reshape(-1, 1, 1)

    if "symmetrize" in operations:
        # Vectorized symmetrization
        c2_array = 0.5 * (c2_array + np.swapaxes(c2_array, -2, -1))

    if "diagonal_correct" in operations:
        # Batch diagonal correction
        for i in range(c2_array.shape[0]):
            c2_array[i] = correct_diagonal_c2_vectorized(c2_array[i])

    return c2_array


def compute_c2_statistics_vectorized(c2_matrices):
    """
    Compute statistical measures for C2 matrices using vectorized operations.

    Args:
        c2_matrices: Array of C2 matrices [batch, height, width]

    Returns:
        Dictionary with statistical measures
    """
    c2_array = np.array(c2_matrices)

    # Vectorized statistical computations
    stats = {
        "mean": np.mean(c2_array, axis=0),
        "std": np.std(c2_array, axis=0),
        "median": np.median(c2_array, axis=0),
        "min": np.min(c2_array, axis=0),
        "max": np.max(c2_array, axis=0),
        "trace": np.trace(c2_array, axis1=-2, axis2=-1),  # Batch trace computation
        "diagonal_mean": np.mean(np.diagonal(c2_array, axis1=-2, axis2=-1), axis=-1),
        "off_diagonal_mean": [],
    }

    # Vectorized off-diagonal mean computation
    for i in range(c2_array.shape[0]):
        mask = ~np.eye(c2_array.shape[-1], dtype=bool)
        off_diag_vals = c2_array[i][mask]
        stats["off_diagonal_mean"].append(np.mean(off_diag_vals))

    stats["off_diagonal_mean"] = np.array(stats["off_diagonal_mean"])

    return stats


def optimized_c2_sampling(c2_matrix, target_size, method="bilinear"):
    """
    Optimized C2 matrix downsampling using vectorized operations.

    Args:
        c2_matrix: Input C2 matrix
        target_size: Target matrix size
        method: Sampling method ('uniform', 'bilinear', 'adaptive')

    Returns:
        Downsampled C2 matrix
    """
    current_size = c2_matrix.shape[0]

    if current_size <= target_size:
        return c2_matrix

    if method == "uniform":
        # Simple uniform sampling
        step = current_size // target_size
        return c2_matrix[::step, ::step]

    if method == "bilinear":
        # Bilinear interpolation using vectorized operations
        from scipy.ndimage import zoom

        zoom_factor = target_size / current_size
        return zoom(c2_matrix, zoom_factor, order=1)

    if method == "adaptive":
        # Adaptive sampling preserving important features
        # Use block averaging for better feature preservation
        block_size = current_size // target_size

        # Vectorized block averaging
        trimmed_size = target_size * block_size
        trimmed_matrix = c2_matrix[:trimmed_size, :trimmed_size]

        # Reshape and compute block means
        reshaped = trimmed_matrix.reshape(
            target_size, block_size, target_size, block_size
        )
        return np.mean(reshaped, axis=(1, 3))

    return c2_matrix
