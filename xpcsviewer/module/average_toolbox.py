# Standard library imports
import multiprocessing
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from shutil import copyfile

# Third-party imports
import numpy as np
import pyqtgraph as pg

# sklearn is imported lazily inside functions to avoid eager loading
from tqdm import tqdm, trange

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import QObject, QtCore, Slot
from xpcsviewer.utils.logging_config import get_logger

# Local imports
from ..fileIO.hdf_reader import put
from ..helper.listmodel import ListDataModel
from ..xpcs_file import MemoryMonitor
from ..xpcs_file import XpcsFile as XF

logger = get_logger(__name__)


class WorkerSignal(QObject):
    progress = QtCore.Signal(tuple)
    values = QtCore.Signal(tuple)
    status = QtCore.Signal(tuple)
    finished = QtCore.Signal()


class AverageToolbox(QtCore.QRunnable):
    """Thread-pool runnable for averaging multiple XPCS datasets.

    Manages a file list, baseline filtering, and progress signalling
    via Qt signals. Designed to run in a ``QThreadPool`` to keep the
    GUI responsive during long averaging operations.

    Args:
        work_dir: Directory containing the HDF5 files.
        flist: List of filenames to average.
        jid: Optional job identifier for tracking.
    """

    def __init__(self, work_dir=None, flist=None, jid=None) -> None:
        if flist is None:
            flist = ["hello"]
        super().__init__()
        self.file_list = flist.copy()
        self.model = ListDataModel(self.file_list)

        self.work_dir = work_dir
        self.signals = WorkerSignal()
        from typing import Any

        self.kwargs: dict[str, Any] = {}
        if jid is None:
            self.jid = uuid.uuid4()
        else:
            self.jid = jid
        self.submit_time = time.strftime("%H:%M:%S")
        self.stime = self.submit_time
        self.etime = "--:--:--"
        self.status = "wait"
        self.baseline = np.zeros(max(len(self.model), 10), dtype=np.float32)
        self.ptr = 0
        self.short_name = self.generate_avg_fname()
        self.eta = "..."
        self.size = len(self.model)
        self._progress = "0%"
        # axis to show the baseline;
        self.ax = None
        # use one file as templelate
        self.origin_path = os.path.join(self.work_dir, self.model[0])

        self.is_killed = threading.Event()

    def kill(self):
        self.is_killed.set()

    def __str__(self) -> str:
        return str(self.jid)

    def generate_avg_fname(self):
        if len(self.model) == 0:
            return None
        fname = self.model[0]
        end = fname.rfind("_")
        if end == -1:
            end = len(fname)
        new_fname = "Avg" + fname[slice(0, end)]
        # if new_fname[-3:] not in ['.h5', 'hdf']:
        #     new_fname += '.hdf'
        return new_fname

    @Slot()
    def run(self):
        self.do_average(*self.args, **self.kwargs)

    def setup(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def do_average(
        self,
        chunk_size=256,
        save_path=None,
        avg_window=3,
        avg_qindex=0,
        avg_blmin=0.95,
        avg_blmax=1.05,
        fields=None,
    ):
        if fields is None:
            fields = ["saxs_2d"]
        self.stime = time.strftime("%H:%M:%S")
        self.status = "running"
        logger.info("average job %d starts", self.jid)
        tot_num = len(self.model)
        steps = (tot_num + chunk_size - 1) // chunk_size
        mask = np.zeros(tot_num, dtype=np.int64)

        def validate_g2_baseline(g2_data, q_idx):
            if q_idx >= g2_data.shape[1]:
                idx = 0
                logger.info("q_index is out of range; using 0 instead")
            else:
                idx = q_idx

            g2_baseline = np.mean(g2_data[-avg_window:, idx])
            if avg_blmax >= g2_baseline >= avg_blmin:
                return True, g2_baseline
            return False, g2_baseline

        result = {}
        for key in fields:
            result[key] = None

        t0 = time.perf_counter()

        # Monitor memory before starting
        initial_memory_mb, _ = MemoryMonitor.get_memory_usage()
        logger.info(
            f"Starting average with {tot_num} files, initial memory: {initial_memory_mb:.1f}MB"
        )

        # Try parallel processing if we have enough files and cores
        n_workers = min(multiprocessing.cpu_count(), max(1, tot_num // 4))
        use_parallel = tot_num >= 8 and n_workers > 1

        # Check memory pressure - disable parallel processing if memory is tight
        if MemoryMonitor.is_memory_pressure_high(0.7):
            use_parallel = False
            logger.warning(
                "High memory pressure detected, forcing sequential processing"
            )

        if use_parallel:
            logger.info(f"Using parallel processing with {n_workers} workers")
            self._process_files_parallel(
                tot_num, fields, validate_g2_baseline, avg_qindex, result, mask, t0
            )
        else:
            logger.info("Using sequential processing")
            self._process_files_sequential(
                tot_num,
                chunk_size,
                steps,
                fields,
                validate_g2_baseline,
                avg_qindex,
                result,
                mask,
                t0,
            )

        if np.sum(mask) == 0:
            logger.info("no dataset is valid; check the baseline criteria.")
            return
        for key in fields:
            if key == "saxs_1d":
                # only keep the Iq component, put method doesn't accept dict
                result["saxs_1d"] = result["saxs_1d"] / np.sum(mask)
            else:
                result[key] /= np.sum(mask)
            if key == "g2_err":
                result[key] /= np.sqrt(np.sum(mask))
            if key == "saxs_2d":
                # saxs_2d needs to be (1, height, width)
                saxs_2d = result[key]
                if saxs_2d.ndim == 2:
                    saxs_2d = np.expand_dims(saxs_2d, axis=0)
                result[key] = saxs_2d

        logger.info(
            "the valid dataset number is %d / %d", int(np.sum(mask)), int(tot_num)
        )

        # Report final memory usage and peak memory reduction
        final_memory_mb, _ = MemoryMonitor.get_memory_usage()
        memory_delta = final_memory_mb - initial_memory_mb
        logger.info(
            f"Memory usage during averaging: {memory_delta:+.1f}MB "
            f"(final: {final_memory_mb:.1f}MB)"
        )

        logger.info(f"create file: {save_path}")
        copyfile(self.origin_path, save_path)
        put(save_path, result, ftype="nexus", mode="alias")

        # Final cleanup to release memory
        del result
        try:
            from ..threading.cleanup_optimized import smart_gc_collect

            smart_gc_collect("average_toolbox_final_cleanup")
        except ImportError:
            import gc

            gc.collect()

        final_cleanup_memory_mb, _ = MemoryMonitor.get_memory_usage()
        logger.info(f"Memory after cleanup: {final_cleanup_memory_mb:.1f}MB")

        self.status = "finished"
        self.signals.status.emit((self.jid, self.status))
        self.etime = time.strftime("%H:%M:%S")
        QtCore.QMetaObject.invokeMethod(
            self.model, "layoutChanged", QtCore.Qt.QueuedConnection
        )
        self.signals.progress.emit((self.jid, 100))
        self.signals.finished.emit()
        logger.info("average job %d finished", self.jid)
        return  # Return None since we deleted result to save memory

    def _process_files_sequential(
        self,
        tot_num,
        chunk_size,
        steps,
        fields,
        validate_g2_baseline,
        avg_qindex,
        result,
        mask,
        t0,
    ):
        """Sequential file processing (original implementation)"""
        prev_percentage = 0

        for n in range(steps):
            beg = chunk_size * (n + 0)
            end = chunk_size * (n + 1)
            end = min(tot_num, end)

            for m in range(beg, end):
                if self.is_killed.is_set():
                    logger.info("the averaging instance has been killed.")
                    self._progress = "killed"
                    self.status = "killed"
                    return

                curr_percentage = int((m + 1) * 100 / tot_num)
                if curr_percentage >= prev_percentage:
                    prev_percentage = curr_percentage
                    dt = (time.perf_counter() - t0) / (m + 1)
                    eta = dt * (tot_num - m - 1)
                    self.eta = eta
                    self._progress = f"{curr_percentage}%"

                fname = self.model[m]
                try:
                    xf = XF(os.path.join(self.work_dir, fname), fields=fields)
                    flag, val = validate_g2_baseline(xf.g2, avg_qindex)
                    self.baseline[self.ptr] = val
                    self.ptr += 1
                except Exception as e:
                    logger.error(f"Error in filtering baseline calculation: {e}")
                    traceback.print_exc()
                    flag, val = False, 0
                    logger.error("file %s is damaged, skip", fname)

                if flag:
                    # Gather this file's fields, then include all-or-nothing so
                    # mask[m] reflects whether EVERY field was summed (consistent
                    # per-key normalization).
                    file_data = {}
                    for key in fields:
                        if key != "saxs_1d":
                            file_data[key] = getattr(xf, key)
                        else:
                            file_data[key] = getattr(xf, "saxs_1d")["data_raw"]
                    shapes_ok = all(
                        result[key] is None or result[key].shape == file_data[key].shape
                        for key in fields
                    )
                    if shapes_ok:
                        mask[m] = 1
                        for key in fields:
                            if result[key] is None:
                                result[key] = file_data[key].copy()
                            else:
                                result[key] += file_data[key]
                    else:
                        logger.info(f"data shape does not match, skipping {fname}")

                # Clear the XpcsFile to release memory immediately
                if "xf" in locals():
                    xf.clear_cache()
                    del xf

                # Periodic memory cleanup every 10 files (gen0 only for speed)
                if m % 10 == 0:
                    try:
                        from ..threading.cleanup_optimized import smart_gc_collect

                        smart_gc_collect("average_toolbox_periodic_cleanup")
                    except ImportError:
                        import gc

                        gc.collect(0)
                    current_memory_mb, _ = MemoryMonitor.get_memory_usage()

                    # If memory pressure is too high, trigger more aggressive cleanup
                    if MemoryMonitor.is_memory_pressure_high(0.85):
                        logger.warning(
                            f"Memory pressure high during averaging (file {m + 1}/{tot_num}), "
                            f"current memory: {current_memory_mb:.1f}MB"
                        )
                        # Force more frequent garbage collection
                        try:
                            from ..threading.cleanup_optimized import smart_gc_collect

                            smart_gc_collect("average_toolbox_memory_pressure")
                        except ImportError:
                            import gc

                            gc.collect()

                self.signals.values.emit((self.jid, val))

    def _process_files_parallel(
        self, tot_num, fields, validate_g2_baseline, avg_qindex, result, mask, t0
    ):
        """Parallel file processing using ThreadPoolExecutor"""
        # Create batches for processing
        batch_size = max(1, tot_num // (multiprocessing.cpu_count() * 2))
        batches = []
        for i in range(0, tot_num, batch_size):
            end = min(i + batch_size, tot_num)
            batches.append(list(range(i, end)))

        completed_files = 0
        prev_percentage = 0

        # Process files in batches
        # Cap at 4 workers: h5py holds the GIL during reads so extra threads
        # add context-switch overhead rather than I/O concurrency.
        with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
            # Submit batch jobs
            future_to_batch = {}
            for batch_indices in batches:
                if self.is_killed.is_set():
                    return
                future = executor.submit(
                    self._process_batch,
                    batch_indices,
                    fields,
                    validate_g2_baseline,
                    avg_qindex,
                )
                future_to_batch[future] = batch_indices

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                if self.is_killed.is_set():
                    logger.info("the averaging instance has been killed.")
                    self._progress = "killed"
                    self.status = "killed"
                    return

                batch_indices = future_to_batch[future]
                try:
                    batch_results, batch_baselines = future.result()

                    # Merge batch results into main result
                    # Write baseline at file's original index m so that
                    # completion order (as_completed) does not corrupt ordering.
                    for m, (file_result, baseline_val) in zip(
                        batch_indices,
                        zip(batch_results, batch_baselines, strict=False),
                        strict=False,
                    ):
                        self.baseline[m] = baseline_val

                        if file_result is not None:
                            for key in fields:
                                data = file_result[key]
                                if result[key] is None:
                                    result[key] = data
                                    mask[m] = 1
                                elif result[key].shape == data.shape:
                                    result[key] += data
                                    mask[m] = 1
                                else:
                                    logger.info(
                                        f"data shape does not match for key {key}"
                                    )

                        self.signals.values.emit((self.jid, baseline_val))

                    completed_files += len(batch_indices)
                    curr_percentage = int(completed_files * 100 / tot_num)

                    if curr_percentage >= prev_percentage:
                        prev_percentage = curr_percentage
                        dt = (time.perf_counter() - t0) / completed_files
                        eta = dt * (tot_num - completed_files)
                        self.eta = eta
                        self._progress = f"{curr_percentage}%"

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Handle batch failure - mark files as invalid at their
                    # original index so ordering matches the successful path.
                    for _m in batch_indices:
                        self.baseline[_m] = 0
                        self.signals.values.emit((self.jid, 0))

        # Set ptr to tot_num after all batches complete so update_plot slices
        # self.baseline[:self.ptr] correctly regardless of completion order.
        self.ptr = tot_num

    def _process_batch(self, batch_indices, fields, validate_g2_baseline, avg_qindex):
        """Process a batch of files"""
        batch_results = []
        batch_baselines = []

        for m in batch_indices:
            # DEFECT 1: check kill flag once per file so long batches can be
            # interrupted without waiting for the entire batch to finish.
            if self.is_killed.is_set():
                return None, []

            fname = self.model[m]
            try:
                xf = XF(os.path.join(self.work_dir, fname), fields=fields)
                flag, val = validate_g2_baseline(xf.g2, avg_qindex)
                batch_baselines.append(val)

                if flag:
                    file_result = {}
                    for key in fields:
                        if key != "saxs_1d":
                            file_result[key] = getattr(xf, key)
                        else:
                            file_result[key] = getattr(xf, "saxs_1d")["data_raw"]
                    batch_results.append(file_result)
                else:
                    batch_results.append(None)

                # DEFECT 4: release XpcsFile memory immediately, mirroring the
                # sequential path cleanup at lines 338-341.
                xf.clear_cache()
                del xf

            except Exception as e:
                logger.error(f"file {fname} is damaged or failed to load, skip: {e}")
                batch_results.append(None)
                batch_baselines.append(0)

        return batch_results, batch_baselines

    def initialize_plot(self, hdl):
        hdl.clear()
        t = hdl.addPlot()
        t.setLabel("bottom", "Dataset Index")
        t.setLabel("left", "g2 baseline")
        self.ax = t.plot(symbol="o")
        if "avg_blmin" in self.kwargs:
            dn = pg.InfiniteLine(
                pos=self.kwargs["avg_blmin"], angle=0, pen=pg.mkPen("r")
            )
            t.addItem(dn)
        if "avg_blmax" in self.kwargs:
            up = pg.InfiniteLine(
                pos=self.kwargs["avg_blmax"], angle=0, pen=pg.mkPen("r")
            )
            # t.addItem(pg.FillBetweenItem(dn, up))
            t.addItem(up)
        t.setMouseEnabled(x=False, y=False)

    def update_plot(self):
        if self.ax is not None:
            self.ax.setData(self.baseline[: self.ptr])
            return

    def get_pg_tree(self):
        data = {}
        for key, val in self.kwargs.items():
            if isinstance(val, np.ndarray):
                if val.size > 4096:
                    data[key] = "data size is too large"
                # suqeeze one-element array
                if val.size == 1:
                    data[key] = float(val)
            else:
                data[key] = val

        # additional keys to describe the worker
        add_keys = ["submit_time", "etime", "status", "baseline", "ptr", "eta", "size"]

        for key in add_keys:
            data[key] = self.__dict__[key]

        if self.size > 20:
            data["first_10_datasets"] = self.model[0:10]
            data["last_10_datasets"] = self.model[-10:]
        else:
            data["input_datasets"] = self.model[:]

        tree = pg.DataTreeWidget(data=data)
        tree.setWindowTitle(f"Job_{self.jid}_{self.model[0]}")
        tree.resize(600, 800)
        return tree


def _process_file_for_average(args):
    """Helper function for parallel file processing in do_average"""
    fname, work_dir, fields, avg_window, avg_qindex, avg_blmin, avg_blmax = args

    def validate_g2_baseline(g2_data, q_idx):
        idx = 0 if q_idx >= g2_data.shape[1] else q_idx
        g2_baseline = np.mean(g2_data[-avg_window:, idx])
        return avg_blmax >= g2_baseline >= avg_blmin, g2_baseline

    try:
        xf = XF(os.path.join(work_dir, fname), fields=fields)
        flag, val = validate_g2_baseline(xf.g2, avg_qindex)

        if flag:
            result = {}
            for key in fields:
                if key != "saxs_1d":
                    result[key] = getattr(xf, key)
                else:
                    data = getattr(xf, "saxs_1d")["data_raw"]
                    scale = xf.abs_cross_section_scale
                    if scale is None:
                        scale = 1.0
                    result[key] = data * scale

            scale = xf.abs_cross_section_scale if "saxs_1d" in fields else 1.0
            return True, val, result, scale if scale is not None else 1.0
        return False, val, None, 1.0
    except Exception as ec:
        logger.error(f"file {fname} is damaged, skip: {ec!s}")
        return False, 0, None, 1.0


def do_average(
    flist,
    work_dir=None,
    save_path=None,
    avg_window=3,
    avg_qindex=0,
    avg_blmin=0.95,
    avg_blmax=1.05,
    fields=None,
    n_jobs=None,
):
    """Average multiple XPCS datasets with baseline filtering.

    Reads each file in *flist*, evaluates a baseline metric at the
    given Q-index, and includes the file in the average only when
    its baseline falls within ``[avg_blmin, avg_blmax]``. Uses
    parallel processing for large file lists.

    Args:
        flist: List of HDF5 filenames to average.
        work_dir: Directory containing the files. Defaults to ``"./"``
        save_path: Output file path. Defaults to ``"AVG" + flist[0]``.
        avg_window: Smoothing window size for baseline evaluation.
        avg_qindex: Q-bin index used for baseline computation.
        avg_blmin: Minimum acceptable baseline value for inclusion.
        avg_blmax: Maximum acceptable baseline value for inclusion.
        fields: List of HDF5 field keys to average. Defaults to
            ``["saxs_2d", "saxs_1d", "g2", "g2_err"]``.
        n_jobs: Number of parallel workers. Defaults to
            ``min(len(flist), cpu_count())``.

    Returns:
        numpy.ndarray | None: Per-file baseline values as a 1-D array,
        or ``None`` if no valid datasets pass the baseline filter.
    """
    if fields is None:
        fields = ["saxs_2d", "saxs_1d", "g2", "g2_err"]
    if work_dir is None:
        work_dir = "./"

    tot_num = len(flist)

    # Monitor memory before starting
    initial_memory_mb, _ = MemoryMonitor.get_memory_usage()
    logger.info(
        f"Starting standalone average with {tot_num} files, initial memory: {initial_memory_mb:.1f}MB"
    )

    abs_cs_scale_tot = 0.0
    baseline = np.zeros(tot_num, dtype=np.float32)
    mask = np.zeros(tot_num, dtype=np.int64)

    result = {}
    for key in fields:
        result[key] = None

    # Determine number of workers
    if n_jobs is None:
        n_jobs = min(tot_num, multiprocessing.cpu_count())

    # Use parallel processing for large datasets
    if tot_num >= 4 and n_jobs > 1:
        logger.info(f"Using parallel processing with {n_jobs} workers")

        # Prepare arguments for parallel processing
        args_list = [
            (fname, work_dir, fields, avg_window, avg_qindex, avg_blmin, avg_blmax)
            for fname in flist
        ]

        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results_parallel = list(
                    tqdm(
                        executor.map(_process_file_for_average, args_list),
                        total=tot_num,
                        desc="Processing files",
                    )
                )

            # Process results from parallel execution
            for m, (flag, val, file_result, scale) in enumerate(results_parallel):
                baseline[m] = val

                if flag and file_result is not None:
                    # All-or-nothing: include this file only if every field is
                    # shape-compatible, so per-key normalization stays consistent.
                    shapes_ok = all(
                        result[key] is None
                        or result[key].shape == file_result[key].shape
                        for key in fields
                    )
                    if shapes_ok:
                        mask[m] = 1
                        for key in fields:
                            data = file_result[key]
                            if key == "saxs_1d":
                                abs_cs_scale_tot += scale
                            if result[key] is None:
                                result[key] = data
                            else:
                                result[key] += data
                    else:
                        logger.info(f"data shape does not match, skipping {flist[m]}")

        except Exception as e:
            logger.warning(f"Parallel processing failed, falling back to serial: {e}")
            # Fall back to sequential processing
            for m in trange(tot_num):
                flag, val, file_result, scale = _process_file_for_average(
                    (
                        flist[m],
                        work_dir,
                        fields,
                        avg_window,
                        avg_qindex,
                        avg_blmin,
                        avg_blmax,
                    )
                )
                baseline[m] = val

                if flag and file_result is not None:
                    # All-or-nothing: include this file only if every field is
                    # shape-compatible, so per-key normalization stays consistent.
                    shapes_ok = all(
                        result[key] is None
                        or result[key].shape == file_result[key].shape
                        for key in fields
                    )
                    if shapes_ok:
                        mask[m] = 1
                        for key in fields:
                            data = file_result[key]
                            if key == "saxs_1d":
                                abs_cs_scale_tot += scale
                            if result[key] is None:
                                result[key] = data
                            else:
                                result[key] += data
                    else:
                        logger.info(f"data shape does not match, skipping {flist[m]}")
    else:
        # Sequential processing for small datasets
        logger.info("Using sequential processing")
        for m in trange(tot_num):
            flag, val, file_result, scale = _process_file_for_average(
                (
                    flist[m],
                    work_dir,
                    fields,
                    avg_window,
                    avg_qindex,
                    avg_blmin,
                    avg_blmax,
                )
            )
            baseline[m] = val

            if flag and file_result is not None:
                # All-or-nothing: include this file only if every field is
                # shape-compatible, so per-key normalization stays consistent.
                shapes_ok = all(
                    result[key] is None or result[key].shape == file_result[key].shape
                    for key in fields
                )
                if shapes_ok:
                    mask[m] = 1
                    for key in fields:
                        data = file_result[key]
                        if key == "saxs_1d":
                            abs_cs_scale_tot += scale
                        if result[key] is None:
                            result[key] = data
                        else:
                            result[key] += data
                else:
                    logger.info(f"data shape does not match, skipping {flist[m]}")

    if np.sum(mask) == 0:
        logger.info("no dataset is valid; check the baseline criteria.")
        return None
    n_valid = np.sum(mask)
    for key in fields:
        if key == "saxs_1d":
            # abs_cs_scale_tot can be 0 if every file's scale was 0; fall back to
            # an unweighted mean instead of dividing by zero.
            if abs_cs_scale_tot > 0:
                result["saxs_1d"] /= abs_cs_scale_tot
            else:
                result["saxs_1d"] /= n_valid
        else:
            result[key] /= n_valid
        if key == "g2_err":
            result[key] /= np.sqrt(n_valid)

    logger.info("the valid dataset number is %d / %d", int(np.sum(mask)), int(tot_num))

    # Report final memory usage and peak memory reduction
    final_memory_mb, _ = MemoryMonitor.get_memory_usage()
    memory_delta = final_memory_mb - initial_memory_mb
    logger.info(
        f"Memory usage during standalone averaging: {memory_delta:+.1f}MB "
        f"(final: {final_memory_mb:.1f}MB)"
    )

    original_file = os.path.join(work_dir, flist[0])
    if save_path is None:
        save_path = "AVG" + os.path.basename(flist[0])
    logger.info(f"create file: {save_path}")
    copyfile(original_file, save_path)
    put(save_path, result, ftype="nexus", mode="alias")

    # Final cleanup
    del result
    try:
        from ..threading.cleanup_optimized import smart_gc_collect

        smart_gc_collect("average_files_final_cleanup")
    except ImportError:
        import gc

        gc.collect()

    final_cleanup_memory_mb, _ = MemoryMonitor.get_memory_usage()
    logger.info(f"Memory after cleanup: {final_cleanup_memory_mb:.1f}MB")

    return baseline
