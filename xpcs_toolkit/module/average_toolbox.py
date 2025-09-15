# Standard library imports
import multiprocessing
import os
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from shutil import copyfile

# Third-party imports
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtCore import QObject, Slot
from sklearn.cluster import KMeans as sk_kmeans
from tqdm import trange

from xpcs_toolkit.utils.logging_config import get_logger

# Local imports
from ..fileIO.hdf_reader import put
from ..helper.listmodel import ListDataModel
from ..xpcs_file import MemoryMonitor
from ..xpcs_file import XpcsFile as XF

logger = get_logger(__name__)


def average_plot_cluster(self, hdl1, num_clusters=2):
    if (
        self.meta["avg_file_list"] != tuple(self.target)
        or "avg_intt_minmax" not in self.meta
    ):
        logger.info("avg cache not exist")
        labels = ["Int_t"]
        res = self.fetch(labels, file_list=self.target)
        Int_t = res["Int_t"][:, 1, :].astype(np.float32)
        Int_t = Int_t / np.max(Int_t)
        intt_minmax = []
        for n in range(len(self.target)):
            intt_minmax.append([np.min(Int_t[n]), np.max(Int_t[n])])
        intt_minmax = np.array(intt_minmax).T.astype(np.float32)

        self.meta["avg_file_list"] = tuple(self.target)
        self.meta["avg_intt_minmax"] = intt_minmax
        self.meta["avg_intt_mask"] = np.ones(len(self.target))

    else:
        logger.info("using avg cache")
        intt_minmax = self.meta["avg_intt_minmax"]

    y_pred = sk_kmeans(n_clusters=num_clusters).fit_predict(intt_minmax.T)
    freq = np.bincount(y_pred)
    self.meta["avg_intt_mask"] = y_pred == y_pred[freq.argmax()]
    valid_num = np.sum(y_pred == y_pred[freq.argmax()])
    title = "%d / %d" % (valid_num, y_pred.size)
    hdl1.show_scatter(
        intt_minmax, color=y_pred, xlabel="Int-t min", ylabel="Int-t max", title=title
    )


class WorkerSignal(QObject):
    progress = QtCore.Signal(tuple)
    values = QtCore.Signal(tuple)
    status = QtCore.Signal(tuple)


class AverageToolbox(QtCore.QRunnable):
    def __init__(self, work_dir=None, flist=None, jid=None) -> None:
        if flist is None:
            flist = ["hello"]
        super().__init__()
        self.file_list = flist.copy()
        self.model = ListDataModel(self.file_list)

        self.work_dir = work_dir
        self.signals = WorkerSignal()
        self.kwargs = {}
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

        self.is_killed = False

    def kill(self):
        self.is_killed = True

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

        logger.info("the valid dataset number is %d / %d" % (np.sum(mask), tot_num))

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
        self.model.layoutChanged.emit()
        self.signals.progress.emit((self.jid, 100))
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
                if self.is_killed:
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
                    self._progress = "%d%%" % (curr_percentage)

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
                    for key in fields:
                        if key != "saxs_1d":
                            data = xf.__getattr__(key)
                        else:
                            data = xf.__getattr__("saxs_1d")["data_raw"]
                        if result[key] is None:
                            result[key] = data.copy()  # Ensure we own the data
                            mask[m] = 1
                        elif result[key].shape == data.shape:
                            result[key] += data
                            mask[m] = 1
                        else:
                            logger.info(
                                f"data shape does not match for key {key}, {fname}"
                            )

                # Clear the XpcsFile to release memory immediately
                if "xf" in locals():
                    xf.clear_cache()
                    del xf

                # Periodic memory cleanup every 10 files
                if m % 10 == 0:
                    try:
                        from ..threading.cleanup_optimized import smart_gc_collect

                        smart_gc_collect("average_toolbox_periodic_cleanup")
                    except ImportError:
                        import gc

                        gc.collect()
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
        with ThreadPoolExecutor(
            max_workers=min(len(batches), multiprocessing.cpu_count())
        ) as executor:
            # Submit batch jobs
            future_to_batch = {}
            for batch_indices in batches:
                if self.is_killed:
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
                if self.is_killed:
                    logger.info("the averaging instance has been killed.")
                    self._progress = "killed"
                    self.status = "killed"
                    return

                batch_indices = future_to_batch[future]
                try:
                    batch_results, batch_baselines = future.result()

                    # Merge batch results into main result
                    for m, (file_result, baseline_val) in zip(
                        batch_indices,
                        zip(batch_results, batch_baselines, strict=False),
                        strict=False,
                    ):
                        self.baseline[self.ptr] = baseline_val
                        self.ptr += 1

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
                        self._progress = "%d%%" % curr_percentage

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Handle batch failure - mark files as invalid
                    for m in batch_indices:
                        self.baseline[self.ptr] = 0
                        self.ptr += 1
                        self.signals.values.emit((self.jid, 0))

    def _process_batch(self, batch_indices, fields, validate_g2_baseline, avg_qindex):
        """Process a batch of files"""
        batch_results = []
        batch_baselines = []

        for m in batch_indices:
            fname = self.model[m]
            try:
                xf = XF(os.path.join(self.work_dir, fname), fields=fields)
                flag, val = validate_g2_baseline(xf.g2, avg_qindex)
                batch_baselines.append(val)

                if flag:
                    file_result = {}
                    for key in fields:
                        if key != "saxs_1d":
                            file_result[key] = xf.__getattr__(key)
                        else:
                            file_result[key] = xf.__getattr__("saxs_1d")["data_raw"]
                    batch_results.append(file_result)
                else:
                    batch_results.append(None)

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
        tree.setWindowTitle("Job_%d_%s" % (self.jid, self.model[0]))
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
                    result[key] = xf.at(key)
                else:
                    data = xf.at("saxs_1d")["data_raw"]
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
                    trange(
                        executor.map(_process_file_for_average, args_list),
                        total=tot_num,
                        desc="Processing files",
                    )
                )

            # Process results from parallel execution
            for m, (flag, val, file_result, scale) in enumerate(results_parallel):
                baseline[m] = val

                if flag and file_result is not None:
                    mask[m] = 1
                    for key in fields:
                        data = file_result[key]
                        if key == "saxs_1d":
                            abs_cs_scale_tot += scale

                        if result[key] is None:
                            result[key] = data
                        elif result[key].shape == data.shape:
                            result[key] += data
                        else:
                            logger.info(
                                f"data shape does not match for key {key}, {flist[m]}"
                            )
                            mask[m] = 0

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
                    mask[m] = 1
                    for key in fields:
                        data = file_result[key]
                        if key == "saxs_1d":
                            abs_cs_scale_tot += scale

                        if result[key] is None:
                            result[key] = data
                        elif result[key].shape == data.shape:
                            result[key] += data
                        else:
                            logger.info(
                                f"data shape does not match for key {key}, {flist[m]}"
                            )
                            mask[m] = 0
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
                mask[m] = 1
                for key in fields:
                    data = file_result[key]
                    if key == "saxs_1d":
                        abs_cs_scale_tot += scale

                    if result[key] is None:
                        result[key] = data
                    elif result[key].shape == data.shape:
                        result[key] += data
                    else:
                        logger.info(
                            f"data shape does not match for key {key}, {flist[m]}"
                        )
                        mask[m] = 0

    if np.sum(mask) == 0:
        logger.info("no dataset is valid; check the baseline criteria.")
        return None
    for key in fields:
        if key == "saxs_1d":
            result["saxs_1d"] /= abs_cs_scale_tot
        else:
            result[key] /= np.sum(mask)
        if key == "g2_err":
            result[key] /= np.sqrt(np.sum(mask))

    logger.info("the valid dataset number is %d / %d" % (np.sum(mask), tot_num))

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
