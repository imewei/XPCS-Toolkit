# Threading and Reliability Audit Report

**Generated:** 2026-02-19
**Author:** sre agent (Phase 2 audit)
**Scope:** Qt threading safety, signal/slot hygiene, race conditions, deadlocks, resource leaks

---

## Executive Summary

The threading architecture uses two distinct subsystems: a `QThreadPool`-backed
`WorkerManager` for plot workers and Bayesian fits, and a `concurrent.futures`
`ThreadPoolExecutor`-backed `UnifiedThreadingManager` for general tasks.  The
primary `QThreadPool` path (plot workers, Bayesian workers) is largely correct:
signals cross the thread boundary via Qt's auto-queued mechanism, and
`BaseAsyncWorker.run()` never touches the GUI directly.  Several concrete
issues were found, ordered by severity below.

---

## P0 — Critical: Crash or data-loss risk

### P0-1: Missing `@Slot` decorators on `QMetaObject.invokeMethod` targets
**File:** `xpcsviewer/threading/unified_threading.py:314-329`

```python
def _emit_task_started_queued(self, task_id: str, task_type: str) -> None: ...
def _emit_task_failed_queued(self, task_id: str, error_msg: str) -> None: ...
def _emit_task_completed_queued(self, task_id: str, result: object) -> None: ...
```

`QMetaObject.invokeMethod` with `Qt.QueuedConnection` requires that the
target method is registered as a Qt meta-method (i.e., has a `@Slot`
decorator or is registered via `Q_INVOKABLE`).  Without `@Slot`, the
invocation silently fails: the queued event is never posted to the main
thread's event loop, so `task_started`, `task_completed`, and `task_failed`
signals are never emitted.  Contrast with the two methods that *do* have
`@Slot` (`_emit_memory_pressure_queued` at line 331, `_emit_load_changed_queued`
at line 341).

**Impact:** All `UnifiedThreadingManager`-submitted tasks silently fail to
notify the GUI.  Any code relying on `task_completed` or `task_failed` is
unreachable.

**Fix:** Add `@Slot(str, str)`, `@Slot(str, str)`, and `@Slot(str, object)`
decorators to the three affected methods.

---

### P0-2: `cancel_all_operations` races with in-flight completion callbacks
**File:** `xpcsviewer/threading/async_kernel.py:263-279`

```python
def cancel_all_operations(self):
    self.worker_manager.cancel_all_workers()  # (1) sets cancel flags
    for operation_id in list(self._signal_connections.keys()):
        self._disconnect_signals(operation_id)  # (2) disconnects lambdas
    self.active_operations.clear()  # (3) clears dict
```

Between steps (1) and (2), a worker that is past its cancellation check can
complete and emit `signals.finished`.  The lambda slot
`lambda result: self._on_plot_ready(operation_id, result)` executes on the
main thread via the queued connection *after* step (3) has run.
`_on_plot_ready` then tries `del self.active_operations[operation_id]` on
an already-empty dict — `KeyError` is swallowed by the `if operation_id in
self.active_operations` guard, but the `plot_ready` signal is still emitted,
delivering a stale result to `XpcsViewer.on_async_plot_ready`.

**Impact:** Stale plot data applied to the GUI after the user has cancelled;
potential mismatched operation IDs causing plot corruption.

**Reproduction:** Start a slow plot (e.g., Bayesian fit + G2 plot), click
Cancel All immediately; observe that the plot is partially or fully applied
despite cancellation.

**Fix:** In `_on_plot_ready` (and all completion handlers), check
`self._signal_connections` for the `operation_id` before emitting
`plot_ready`; if signals are already disconnected the result is stale.

---

## P1 — High: Observable incorrect behaviour

### P1-1: `_emit_operation_progress` iterates `active_operations` without a lock
**File:** `xpcsviewer/threading/async_kernel.py:377-389`

```python
def _emit_operation_progress(self, worker_id, current, total, message):
    for op_id, w_id in self.active_operations.items():  # no lock
        ...
```

`_emit_operation_progress` is called from a `lambda` connected to
`WorkerManager.worker_progress`, which is emitted from within
`WorkerManager._on_worker_progress`.  That slot runs on the main thread
(queued connection from the worker thread), so concurrent access is
unlikely in practice.  However, `cancel_all_operations` and
`cancel_operation` mutate `active_operations` also on the main thread, and
because Qt processes events in FIFO order, a cancellation event and a
progress event for the same operation can arrive back-to-back.  Iterating
while a deletion is occurring in an adjacent event handler causes
`RuntimeError: dictionary changed size during iteration`.

**Fix:** Iterate over `list(self.active_operations.items())`.

---

### P1-2: `XpcsViewer.active_plot_operations` mutated from lambda callbacks without synchronisation
**File:** `xpcsviewer/xpcs_viewer.py:1268, 1281`

`on_async_plot_ready` does `del self.active_plot_operations[operation_id]`
and `update_plot_async` reads and mutates the same dict in the same
method.  Both run on the main thread (Qt event loop), which means there is
no true data race, but `on_async_plot_ready` calls
`del self.active_plot_operations[operation_id]` unconditionally after the
`try/except` block at line 1268, while `on_async_operation_error` at
line 1281 guards with `if operation_id in`.  If a result arrives for an
operation that was already cancelled via `cancel_async_operation` (which
does `del self.active_plot_operations[operation_id]`), line 1268 raises
`KeyError`.

**Reproduction:** Cancel a plot operation just before it completes; the
finished signal can still arrive via the queued connection.

**Fix:** Change `del self.active_plot_operations[operation_id]` at line 1268
to `self.active_plot_operations.pop(operation_id, None)`.

---

### P1-3: `TwotimePlotWorker` reads from `ViewerKernel._current_dset_cache` (a `WeakValueDictionary`) from a worker thread
**File:** `xpcsviewer/threading/plot_workers.py:287`

```python
current_dset = self.viewer_kernel._get_cached_dataset(xfile.fname, xfile)
```

`ViewerKernel._current_dset_cache` is a `weakref.WeakValueDictionary`.
Python's `weakref.WeakValueDictionary` is not thread-safe: its internal
finalizer callbacks run on the GC thread, which can delete entries during
the `.get()` call, causing `KeyError` inside the CPython implementation.
Additionally, `plot_twotime` on the *main* thread writes to the same
cache (`self._current_dset_cache[xfile.fname] = current_dset` at line 744)
concurrently with the worker's read.

**Note:** The comment at line 289-290 explicitly says "Do NOT write cache
from worker thread", which shows awareness; the read is not guarded.

**Fix:** Protect `_current_dset_cache` accesses with a `threading.Lock` in
`ViewerKernel`, or pass the required data to the worker at construction time
rather than accessing shared mutable state.

---

### P1-4: `_JIT_CACHE` and `_PARTITION_JIT_CACHE` are module-level dicts without locks
**Files:** `xpcsviewer/simplemask/qmap.py:33`, `xpcsviewer/simplemask/utils.py:28`

```python
_JIT_CACHE: dict[str, Callable] = {}
_PARTITION_JIT_CACHE: dict[str, Callable] = {}
```

Both caches use an optimistic check-then-set pattern:

```python
if cache_key not in _JIT_CACHE:
    ...  # compile JIT function
    _JIT_CACHE[cache_key] = compiled_fn
return _JIT_CACHE[cache_key]
```

If `SimpleMaskKernel` is ever called from a worker thread (e.g., a future
`QMapPlotWorker` that computes a Q-map in the background), two threads can
simultaneously pass the `if cache_key not in` guard, both compile the
function, and the second write races with the first.  In CPython, individual
dict assignments are GIL-protected, so the result is a valid entry (no
corruption), but two redundant JIT compilations are triggered and the
first result is silently discarded — wasting seconds of compile time.

**Current status:** `compute_qmap` is called from `SimpleMaskKernel`, which
currently runs on the main thread only.  The risk is latent but will
materialise if Q-map computation is offloaded to a worker thread.

**Fix:** Add a `threading.Lock` per cache, or use `functools.lru_cache`
with a stable hashable key (string-only cache keys already support this).

---

### P1-5: `_g2_bayesian_worker_active` flag not reset on `cancelled` signal
**File:** `xpcsviewer/xpcs_viewer.py:3900-3907, 3909-3938`

```python
worker.signals.finished.connect(self._on_g2_bayesian_finished)
worker.signals.error.connect(self._on_g2_bayesian_error)
# No connection for worker.signals.cancelled
```

`_on_g2_bayesian_finished` and `_on_g2_bayesian_error` both reset
`self._g2_bayesian_worker_active = False`.  However, `BayesianFitWorker`
inherits `BaseAsyncWorker.run()`, which emits `signals.cancelled` on
`InterruptedError` — not `signals.finished` or `signals.error`.  If
`_fit_g2_bayesian` is cancelled (e.g., via `cancel_all_operations` or
`thread_pool.waitForDone` on shutdown), the flag stays `True` forever,
permanently disabling the "Fit Bayesian" button.

The identical issue exists for `_diff_bayesian_worker_active` at line 4060-4067.

**Fix:** Connect `worker.signals.cancelled` to a handler that resets the
flag and re-enables the button, for both G2 and diffusion workers.

---

### P1-6: Shutdown ordering — `UnifiedThreadingManager` shut down before `QThreadPool.waitForDone`
**File:** `xpcsviewer/xpcs_viewer.py:909-950`

`closeEvent` first calls `async_vk.cancel_all_operations()` then calls
`thread_pool.waitForDone(5000)`.  The `UnifiedThreadingManager` (separate
`ThreadPoolExecutor` pools) is shut down *after* `waitForDone` via the
`shutdown_unified_threading()` import loop.  Meanwhile, `UnifiedThreadingManager`
has its own `_monitor_thread` (daemon=True) that runs `_monitor_system()`.

If `_monitor_system()` fires a `QMetaObject.invokeMethod` while the Qt
event loop is tearing down, the posted event targets a `QObject`
(`UnifiedThreadingManager` itself) that may have already been deleted from
the C++ side.  Setting `_monitoring_active = False` then joining with
`timeout=2.0` in `shutdown()` mitigates this, but the 2-second join happens
*after* the Qt objects are potentially destroyed.

**Fix:** Call `shutdown_unified_threading()` before `thread_pool.waitForDone`
in `closeEvent`.

---

## P2 — Medium: Reliability concern or technical debt

### P2-1: `AsyncDataPreloader.start_preload` connects `data_loaded` without tracking or cleanup
**File:** `xpcsviewer/threading/async_kernel.py:476`

```python
self.async_kernel.data_loaded.connect(self._on_preload_completed)
```

The connection is removed in `_on_preload_completed` via `.disconnect()`.
If `start_preload` is called again before the previous preload completes,
the method returns early (`if self.is_preloading: return`), so double
connection is avoided.  However, if the preload operation is cancelled or
errors before `_on_preload_completed` fires, the signal is never disconnected,
creating a persistent leaked connection.  Every subsequent `data_loaded`
emission (from any operation, not just "preload_data") will call the stale
`_on_preload_completed`.

**Fix:** Guard `_on_preload_completed` with a try/finally that disconnects
the signal, or use `auto_reconnect=False` and track the connection object.

---

### P2-2: Lambda slots in `WorkerManager.submit_worker` capture `worker_id` by value but hold a reference to `worker`
**File:** `xpcsviewer/threading/async_workers.py:568-584`

```python
worker.signals.started.connect(lambda wid, priority: self._on_worker_started(worker_id))
```

The lambda captures `worker_id` (a string — fine) but `worker.signals` holds
a reference to the `WorkerSignals` QObject, which in turn holds the lambda.
The lambda captures `self` (WorkerManager) — creating a reference cycle:
`WorkerManager -> lambda -> WorkerManager`.  CPython's cycle collector
handles this, but the cycle delays destruction and is unnecessary.

Additionally, once the worker completes and is removed from `active_workers`,
the `worker.signals` QObject is the only remaining reference holder.  If
the worker's `QRunnable` is auto-deleted by `QThreadPool` (via
`setAutoDelete(True)` at line 189), the `signals` QObject may be deleted
from the pool thread, triggering Qt signal-related assertions on macOS.

**Note:** `setAutoDelete` at line 189 says `self.setAutoDelete(True)` despite
the docstring comment "Set auto-delete to False so we can reuse workers".
This inconsistency should be resolved.

**Fix:** Use `functools.partial` or explicit `@Slot` methods instead of
capturing lambdas, and decide consistently on `setAutoDelete`.

---

### P2-3: `WorkerManager.worker_results` and `worker_errors` grow unbounded
**File:** `xpcsviewer/threading/async_workers.py:631, 643`

```python
self.worker_results[worker_id] = result  # never evicted
self.worker_errors[worker_id] = ...  # never evicted
```

For a long-running application session, completed-worker results accumulate
in memory indefinitely.  The `shutdown()` method clears them, but there is
no per-entry TTL or size cap.

**Fix:** Use `collections.deque(maxlen=N)` or evict entries older than a
threshold on each completion.

---

### P2-4: `ProgressManager.request_cancel` only updates the UI, not the actual worker
**File:** `xpcsviewer/threading/progress_manager.py:497-503`

```python
@Slot(str)
def request_cancel(self, operation_id: str):
    logger.info(f"Cancel requested for operation: {operation_id}")
    # This signal should be connected to the actual cancellation mechanism
    # For now, just update the UI
    self.cancel_operation_ui(operation_id)
```

The comment admits this is incomplete.  `operation_cancelled` is emitted by
`cancel_operation_ui`, which is connected at line 389:

```python
self.progress_manager.operation_cancelled.connect(self.cancel_async_operation)
```

`cancel_async_operation` then calls `async_vk.cancel_operation(operation_id)`.
This creates a multi-hop, event-loop-dependent chain; if `cancel_async_operation`
is reached, it does cancel the worker.  However, the indirect path means
that if the progress dialog's Cancel button is clicked during teardown (event
loop draining), the cancellation signal may never be delivered.

**Fix:** `request_cancel` should directly call the cancellation mechanism
(or at minimum document the full signal chain explicitly).

---

### P2-5: `UnifiedThreadingManager._adjust_pool_size` calls `pool._max_workers` — private attribute
**File:** `xpcsviewer/threading/unified_threading.py:511-519`

```python
current_workers = pool._max_workers
```

`ThreadPoolExecutor._max_workers` is a CPython implementation detail, not
part of the public API.  Python 3.13+ may remove or rename it.

**Fix:** Store `max_workers` as a local attribute in the `_pool_configs` dict
instead of reading from the private executor attribute.

---

### P2-6: `get_performance_stats` iterates `pool._threads` — private attribute
**File:** `xpcsviewer/threading/unified_threading.py:566`

```python
"active_workers": len([t for t in pool._threads if t.is_alive()]),
```

Same concern as P2-5.  `pool._threads` is an implementation detail.

---

## Summary Table

| ID    | Severity | Component                          | File:Line                                   | Issue |
|-------|----------|------------------------------------|---------------------------------------------|-------|
| P0-1  | P0       | UnifiedThreadingManager            | unified_threading.py:314-329               | Missing `@Slot` on invokeMethod targets — signals silently dropped |
| P0-2  | P0       | AsyncViewerKernel                  | async_kernel.py:263-279                    | Race: cancelled op result delivered post-cancel |
| P1-1  | P1       | AsyncViewerKernel                  | async_kernel.py:383                        | `active_operations` iterated without lock during cancel |
| P1-2  | P1       | XpcsViewer                         | xpcs_viewer.py:1268                        | `KeyError` on `del active_plot_operations` after cancel |
| P1-3  | P1       | TwotimePlotWorker / ViewerKernel   | plot_workers.py:287, viewer_kernel.py:786  | `WeakValueDictionary` read from worker thread without lock |
| P1-4  | P1       | simplemask/qmap, simplemask/utils  | qmap.py:33, utils.py:28                   | Module-level JIT caches without locks (latent, not currently racy) |
| P1-5  | P1       | XpcsViewer Bayesian workers        | xpcs_viewer.py:3900-3907, 4060-4067       | Bayesian active flag not reset on `cancelled` signal |
| P1-6  | P1       | closeEvent shutdown ordering       | xpcs_viewer.py:909-950                    | UnifiedThreadingManager shut down after QThreadPool wait |
| P2-1  | P2       | AsyncDataPreloader                 | async_kernel.py:476                        | Preload signal connection leaked on cancel/error |
| P2-2  | P2       | WorkerManager                      | async_workers.py:568-584, 189             | Lambda capture + auto-delete inconsistency |
| P2-3  | P2       | WorkerManager                      | async_workers.py:631, 643                  | Unbounded `worker_results` / `worker_errors` dicts |
| P2-4  | P2       | ProgressManager                    | progress_manager.py:497-503               | Cancel button only updates UI, relies on multi-hop signal chain |
| P2-5  | P2       | UnifiedThreadingManager            | unified_threading.py:511                   | Uses `pool._max_workers` private attribute |
| P2-6  | P2       | UnifiedThreadingManager            | unified_threading.py:566                   | Uses `pool._threads` private attribute |

---

## Confirmed Correct Patterns

The following patterns were reviewed and found to be correct:

- **QThreadPool signal crossing:** `WorkerSignals` inherits `QObject` and signals
  are emitted from the pool thread; Qt auto-detects the cross-thread connection
  and delivers them via queued events to the main thread.  `on_async_plot_ready`,
  `apply_*_result`, and all PyQtGraph calls only execute in main-thread slots.

- **UnifiedThreadingManager → Qt main thread marshalling:** `_execute_task` and
  `_handle_task_completion` use `QMetaObject.invokeMethod` with
  `Qt.QueuedConnection` for the `_emit_*_queued` methods that *do* have `@Slot`
  decorators (memory pressure, load changed).  The pattern is correct; only the
  task-lifecycle emitters are missing decorators (P0-1).

- **BayesianFitWorker.do_work():** Contains no GUI operations.  All widget
  updates happen in `_on_g2_bayesian_finished` / `_on_g2_bayesian_error`,
  which are connected via the default auto-connection and therefore execute on
  the main thread.

- **HDF5 fork safety:** `h5py` reads happen inside worker `do_work()` methods
  (thread pool, not forked processes).  `h5py` with the default HDF5 threading
  model (serialized) is safe for concurrent reads from multiple threads as long
  as each thread does not share `h5py.File` objects.  Each `XpcsFile` reads
  with its own file handle — verified in `xpcs_file.py`.

- **`cancel_worker` thread safety:** `WorkerManager.cancel_worker` acquires
  `_workers_lock` before reading `active_workers`, then calls `worker.cancel()`
  (which sets a `threading.Event`) outside the lock.  Correct pattern.

- **Double-checked locking for singleton:** `get_unified_threading_manager()`
  uses double-checked locking with `_global_threading_manager_lock`.  Correct
  for CPython due to the GIL, and also correct in principle for any Python
  implementation that makes dict assignment atomic under a lock.

- **`closeEvent` waits for workers:** `thread_pool.waitForDone(5000)` at
  line 922 ensures plot workers finish (or time out) before Qt objects are
  destroyed.  Correct.

---

## Debugger Agent Supplementary Findings (Phase 2)

**Scope:** Worker lifecycle completeness, resource cleanup chains, XpcsFile lifecycle,
cache consistency, and shutdown sequence correctness.

---

### THR-A1 (P2) — `BayesianFitWorker` cancellation during MCMC not observable in-flight

**File:** `xpcsviewer/threading/async_workers.py:356`, `xpcsviewer/threading/bayesian_worker.py:82`

`check_cancelled()` is called in two places: once by `BaseAsyncWorker.run()` immediately
*after* `do_work()` returns (line 356), and once by `BayesianFitWorker.do_work()` after
`fit_func` returns (bayesian_worker.py:82). During NumPyro NUTS warmup/sampling (which can
take seconds to minutes), `cancel()` sets the `threading.Event` but the MCMC loop has no way
to observe it. The cancellation only takes effect when the fit completes.

```python
# bayesian_worker.py
fit_result = self.fit_func(self.x, self.y, self.yerr, **self.sampler_kwargs)
self.check_cancelled()  # too late — entire MCMC has run
```

**Impact:** The "fitting..." button state and `_g2_bayesian_worker_active` flag are locked for
the duration of the full MCMC run even if the user has cancelled. The flag is correctly reset
when the `cancelled` signal eventually fires (confirmed P1-5 is fixed in the SRE findings),
but UX is poor.

**Fix:** NumPyro MCMC does not expose an interrupt hook in 0.15.x. Document this limitation;
optionally add a host-callback via `jax.debug.callback` to check the cancel flag periodically.

---

### THR-A2 (P1) — `ObjectRegistry.clear_all()` drops references without calling cleanup

**File:** `xpcsviewer/threading/cleanup_optimized.py:52-55`

```python
def clear_all(self):
    with self._lock:
        self.objects.clear()  # drops all refs — no cleanup called
```

`XpcsFile` registers itself via `register_for_cleanup()` at `__init__:224`. On shutdown,
`schedule_type_cleanup("XpcsFile", HIGH)` (safe_shutdown step 3) only triggers
`SmartGarbageCollector.collect()` (i.e., `gc.collect()`), which does **not** call
`XpcsFile.close()`. The `ObjectRegistry.clear_all()` path also skips cleanup.

The computation cache entries for `XpcsFile` (keyed by `_cache_prefix`) in the shared
`_memory_manager` singleton are never explicitly evicted when `XpcsFile.close()` is called,
because `clear_cache()` itself acknowledges it cannot selectively evict by prefix
(line 3093: "We can't directly clear specific entries by prefix").

**Impact:** On session end, computation cache entries for all `XpcsFile` instances linger in
the shared memory manager until the process exits. For long sessions with many files loaded,
this can be 10s of MB.

**Fix:** Add an `XpcsFile.close()` invocation inside `schedule_type_cleanup` for `XpcsFile`
objects, or add a `memory_manager.evict_by_prefix(prefix)` method and call it from `close()`.

---

### THR-A3 (P1) — `CleanupScheduler.execute_pending_cleanup()` holds lock during task execution

**File:** `xpcsviewer/threading/cleanup_optimized.py:111-130`

```python
def execute_pending_cleanup(self):
    with self._lock:                  # lock acquired
        for task in self.cleanup_tasks[:]:
            if current_time >= task["scheduled_time"]:
                try:
                    task["func"]()    # cleanup runs INSIDE lock
```

Any cleanup function that internally calls `schedule_cleanup()` (which also acquires
`self._lock`) will deadlock. Similarly, any thread that calls `schedule_cleanup()` while
`execute_pending_cleanup()` is running will block for the full duration of all pending tasks.

**Fix:** Collect due tasks under the lock, then release before executing:
```python
with self._lock:
    due = [t for t in self.cleanup_tasks[:] if current_time >= t["scheduled_time"]]
    for t in due:
        self.cleanup_tasks.remove(t)
for t in due:
    try:
        t["func"]()
    except Exception as e:
        logger.warning(f"Cleanup task {t['name']} failed: {e}")
```

---

### THR-A4 (P2) — FFT cache stale after `Int_t` data reload

**File:** `xpcsviewer/xpcs_file.py:3014-3025`

`_compute_int_t_fft_cached()` caches under key `f"{self._cache_prefix}_int_t_fft"`. The
`_cache_prefix` is set once at `__init__:238` using a UUID and never changes. If `load_data()`
is called a second time (e.g., after a file modification or forced reload), `self.Int_t` is
updated but the cache key is identical to the previous load, so the old FFT is returned.

**Impact:** Silent wrong results for FFT-derived intensity metrics. Low probability in current
UI (no explicit "reload" button), but becomes a bug if reload functionality is added.

**Fix:** Increment a `_data_version` counter on each `load_data()` call and include it in the
FFT cache key: `f"{self._cache_prefix}_int_t_fft_v{self._data_version}"`.

---

### THR-A5 (P2) — `WorkerManager` lambda connections never disconnected; `worker_results` grows unbounded

**File:** `xpcsviewer/threading/async_workers.py:557-593, 631`

`submit_worker()` connects 5 lambda slots to `worker.signals.*`. These are never explicitly
disconnected. When the worker's `QRunnable` is destroyed by `QThreadPool` (auto-delete),
`WorkerSignals` (a child `QObject`) is also destroyed, which disconnects all slots. This is
safe. However, the result is stored in `self.worker_results[worker_id]` indefinitely
(line 631). For long sessions, this dict accumulates thousands of entries and their payloads
(NumPy arrays, plot dicts).

This duplicates SRE finding P2-3. Confirmed from debugger analysis.

**Fix:** Evict completed entries from `worker_results` after they have been consumed
(or use `deque(maxlen=200)`).

---

### THR-A6 (P2) — `shutdown_unified_threading()` absent from `safe_shutdown()`

**File:** `xpcsviewer/cli_main.py:39-143`

`safe_shutdown()` (invoked on SIGINT/SIGTERM) does not call `shutdown_unified_threading()`.
It handles `WorkerManager` shutdown (step 1) but the `concurrent.futures` pools in
`UnifiedThreadingManager` are only shut down from `XpcsViewer.closeEvent()` via dynamic
import (lines 933-955). If the process is killed without the GUI close event (e.g., SIGKILL
preceded by SIGTERM, or test teardown), these pools linger.

This overlaps with SRE finding P1-6. Confirmed: `safe_shutdown()` and `closeEvent()` have
complementary but non-overlapping shutdown coverage.

**Fix:** Add `shutdown_unified_threading()` to `safe_shutdown()` step 1 (before HDF5 pool clear).

---

### THR-A7 (P2) — `@Slot` decorator signatures on `WorkerManager` handlers mislead static analysis

**File:** `xpcsviewer/threading/async_workers.py:628-660`

```python
@Slot(str, object)                        # expects 2 args from signal
def _on_worker_finished(self, worker_id: str, result: Any):
```

But `WorkerSignals.finished = Signal(object)` (1 arg). The lambda bridge in
`submit_worker()` provides `worker_id` from closure — so the method works at runtime. However,
the `@Slot` decorator registers the wrong signature with the Qt meta-object system, which could
cause issues if the method is ever called via `QMetaObject.invokeMethod`.

Same issue on `_on_worker_error` (`@Slot(str, str, str)`) and `_on_worker_cancelled` (`@Slot(str)`).

**Fix:** Change `@Slot(str, object)` to `@Slot(object)` (or remove `@Slot` from these private
methods entirely, since they are only called from lambda closures not directly connected as slots).

---

## Updated Summary Table (Debugger Additions)

| ID       | Severity | Area                   | File:Line                              | Description |
|----------|----------|------------------------|----------------------------------------|-------------|
| THR-A1   | P2       | Bayesian worker        | `bayesian_worker.py:82`               | Cancel not observable during MCMC NUTS sampling |
| THR-A2   | P1       | Cleanup chain          | `cleanup_optimized.py:52-55`          | `clear_all()` drops refs without calling `XpcsFile.close()` |
| THR-A3   | P1       | Cleanup chain          | `cleanup_optimized.py:111-130`        | Lock held during cleanup task execution — deadlock risk |
| THR-A4   | P2       | Cache consistency      | `xpcs_file.py:3014-3025`             | FFT cache not invalidated on data reload |
| THR-A5   | P2       | WorkerManager          | `async_workers.py:557-593, 631`       | Lambda connections + unbounded `worker_results` (confirms P2-3) |
| THR-A6   | P2       | Shutdown sequence      | `cli_main.py:39-143`                  | `shutdown_unified_threading()` missing from `safe_shutdown()` (confirms P1-6) |
| THR-A7   | P2       | API hygiene            | `async_workers.py:628-660`            | `@Slot` decorator signatures mismatch actual signal signatures |

---

### Items Verified Correct by Debugger Analysis

- **`BaseAsyncWorker.run()` error coverage:** All exception paths emit exactly one terminal
  signal (`finished`, `cancelled`, or `error`). The `signals.error` signal always fires for
  any unhandled `Exception`.

- **`XpcsFile.__enter__`/`__exit__`/`close()`:** Context manager is correctly implemented;
  `close()` clears in-memory caches without touching HDF5 handles (which are pool-managed).

- **`_g2_bayesian_worker_active` reset in error path:** Both `_on_g2_bayesian_finished` and
  `_on_g2_bayesian_error` reset the flag (lines 3911, 3934). Note: SRE finding P1-5 identified
  that the `cancelled` signal path does NOT reset it — confirmed and agreed.

- **`UnifiedThreadingManager._monitor_system` signal marshalling:** Correctly uses
  `QMetaObject.invokeMethod` with `Qt.QueuedConnection` for all Qt signal emissions from the
  raw monitoring thread.

- **`AsyncViewerKernel` signal cleanup:** `_disconnect_signals()` is called in all three
  terminal handlers (`_on_plot_ready`, `_on_operation_error`, `_on_operation_cancelled`).
  No connection leak in the plot-worker path.

---

*Debugger agent audit complete. Combined with SRE findings above.*
