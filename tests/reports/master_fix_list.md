# XPCS Viewer — Master Fix List

**Generated:** 2026-02-19
**Authors:** debugger (synthesis), sre, python-pro, jax-pro, type-analyzer agents
**Scope:** All phases (1–5). Full-codebase audit of threading, types/contracts, numerical/JAX, schema design.

---

## How to Read This Report

Each entry includes:
- **ID** — canonical identifier for cross-referencing
- **Severity** — P0 (crash/data-corruption/silent wrong result), P1 (observable wrong behaviour / reliability), P2 (technical debt / maintainability)
- **Source(s)** — which agent(s) identified it
- **Evidence** — file:line
- **Compound** — interactions with other bugs
- **Fix** — concrete recommended action

Compound bugs (where two or more issues must be fixed together to be safe) are called out explicitly in section 4.

---

## 1. P0 — Critical: Crash, Data Corruption, or Silent Wrong Result

### P0-01 — Missing `extra_fields=("diverging",)` in MCMC → KeyError after sampling

**Source:** python-pro (T-04), jax-pro (JAX-N-02)
**Evidence:** `xpcsviewer/fitting/sampler.py:198`

`mcmc.get_extra_fields()["diverging"]` is called unconditionally after MCMC sampling. NumPyro only includes `"diverging"` in the extra fields dict when `extra_fields=("diverging",)` is passed to the `MCMC` constructor. Without it, `get_extra_fields()` returns an empty dict and the `[]` access raises `KeyError`, crashing `BayesianFitWorker.do_work()`. The exception propagates to `BaseAsyncWorker.run()`, which emits `signals.error` — the Bayesian fit UI locks until the error handler resets the flag. The fit result is lost.

**Compound:** Interacts with P0-03 (NaN gradients) — if NUTS produces NaN samples, divergences are the diagnostic that would catch it. Without this fix, P0-03 is undetectable via the UI.

**Fix:**
```python
# sampler.py: MCMC constructor
mcmc = MCMC(
    kernel,
    num_warmup=config.num_warmup,
    num_samples=config.num_samples,
    num_chains=config.num_chains,
    extra_fields=("diverging",),  # ADD THIS
)
```

---

### P0-02 — `__dict__` access on frozen dataclass `QMapSchema` → AttributeError

**Source:** python-pro (T-01)
**Evidence:** `xpcsviewer/xpcs_file.py:914`

`QMapSchema` is a `@dataclass(frozen=True)`. Accessing `self.qmap.__dict__[key]` on a frozen dataclass raises `AttributeError` in Python 3.12+ because frozen dataclasses use `__slots__`-like storage and do not expose `__dict__` by default when `slots=True` is set, or raise `FrozenInstanceError` on attribute mutation. Even without slots, `__dict__` access is fragile for key names that don't match the dataclass field names.

**Compound:** Every downstream caller of `XpcsFile` methods that reads from `qmap` via dict-style access is affected. If `qmap` is a plain `dict` (legacy path), this works; if it is a `QMapSchema` (new path via `HDF5Facade`), it crashes at runtime.

**Fix:**
```python
# xpcs_file.py:914 — replace __dict__ access
# OLD: value = self.qmap.__dict__[key]
# NEW:
value = getattr(self.qmap, key)
```
Where field existence is uncertain, guard with `getattr(self.qmap, key, default)`.

---

### P0-03 — `double_exp_model` NaN gradients: `tau1 ≈ 0` → `exp(0/0)`

**Source:** jax-pro (JAX-N-03)
**Evidence:** `xpcsviewer/fitting/models.py:90`

In `double_exp_model`, `tau1` is sampled from `LogNormal(0, 2)` which has support `(0, ∞)`. NUTS can propose very small values (`tau1 ≈ 1e-300`) during warmup. The model evaluates `jnp.exp(-x / tau1)` without clipping, producing `exp(-inf)=0` or `exp(NaN)` depending on the platform. The NaN propagates into the likelihood and poisons the NUTS step. Combined with P0-01 (diverging field missing), these NaN steps are completely invisible to the diagnostics.

**Compound:** P0-01 must be fixed first to make NaN divergences visible. Also interacts with JAX-N-06 (stretched_exp beta=0.01 → log Beta(-inf)).

**Fix:**
```python
# models.py — inside double_exp_model
tau1 = numpyro.sample("tau1", dist.LogNormal(0, 2))
tau2 = numpyro.sample("tau2", dist.LogNormal(0, 2))
# Clip to prevent exp(-inf) or NaN at near-zero tau
tau1_safe = jnp.clip(tau1, a_min=1e-6)
tau2_safe = jnp.clip(tau2, a_min=1e-6)
mu = double_exp_func(x, tau1_safe, tau2_safe, baseline, c1, c2)
```

---

### P0-04 — `read_g2_data` does not coerce float32 → float64 before `G2Data` constructor

**Source:** python-pro (T-02)
**Evidence:** `xpcsviewer/io/hdf5_facade.py:421`

`G2Data.__post_init__()` enforces `dtype == float64` for all arrays. HDF5 files written by older pipeline versions store `g2` as float32. `read_g2_data()` passes the raw h5py dataset arrays directly to `G2Data(...)` without dtype coercion, so `__post_init__` raises `ValueError: g2 must be float64` on any such file. This is a silent incompatibility with all pre-migration datasets.

**Fix:**
```python
# io/hdf5_facade.py — in read_g2_data(), before constructing G2Data:
g2 = np.asarray(raw_g2, dtype=np.float64)
g2_err = np.asarray(raw_g2_err, dtype=np.float64)
tau = np.asarray(raw_tau, dtype=np.float64)
```

---

### P0-05 — `read_qmap` does not coerce float32 → float64 before `QMapSchema` construction

**Source:** python-pro (T-13)
**Evidence:** `xpcsviewer/io/hdf5_facade.py:117`, `xpcsviewer/schemas/validators.py`

Same root cause as P0-04. `QMapSchema.__post_init__()` validates `dtype == float64` for `sqmap` and `dqmap`. HDF5 files written with float32 Q-maps fail on construction. This affects all users loading legacy data.

**Fix:**
```python
# io/hdf5_facade.py — in read_qmap(), before QMapSchema(**data):
for key in ("sqmap", "dqmap", "phis"):
    if key in data:
        data[key] = np.asarray(data[key], dtype=np.float64)
```

---

### P0-06 — Qt signals emitted from raw `threading.Thread` in `_monitor_system`

**Source:** sre (SRE-1)
**Evidence:** `xpcsviewer/threading/unified_threading.py:436, 458`

`UnifiedThreadingManager._monitor_system()` runs on a raw `threading.Thread` (not a `QThread`). The current code uses `QMetaObject.invokeMethod(..., QueuedConnection)` for two signals (`_emit_memory_pressure_queued`, `_emit_load_changed_queued`) — those are correct. However, three other `invokeMethod` targets (`_emit_task_started_queued`, `_emit_task_failed_queued`, `_emit_task_completed_queued`) lack `@Slot` decorators (see P0-07), so their invocations silently fail.

**Note:** The signal-emission pattern in the monitor thread itself is correct where `@Slot` is present. The failure mode is in the missing decorators (P0-07), not in the threading model.

**Partially addressed by:** P0-07.

---

### P0-07 — `@Slot` decorators missing on `invokeMethod` targets → signals silently dropped

**Source:** sre (P0-1 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/unified_threading.py:314-329`

`_emit_task_started_queued`, `_emit_task_failed_queued`, and `_emit_task_completed_queued` are called via `QMetaObject.invokeMethod(..., Qt.QueuedConnection)`. Without `@Slot`, Qt cannot find the method in the meta-object system, so the invocation silently no-ops. All `task_started`, `task_completed`, and `task_failed` signals from `UnifiedThreadingManager` are therefore never delivered to the GUI.

**Fix:**
```python
@Slot(str, str)
def _emit_task_started_queued(self, task_id: str, task_type: str) -> None: ...


@Slot(str, str)
def _emit_task_failed_queued(self, task_id: str, error_msg: str) -> None: ...


@Slot(str, object)
def _emit_task_completed_queued(self, task_id: str, result: object) -> None: ...
```

---

### P0-08 — GUI widget updates submitted to `ThreadPoolExecutor` thread

**Source:** sre (SRE-2)
**Evidence:** `xpcsviewer/visualization_optimizer.py:852`

A GUI widget update is submitted to a `ThreadPoolExecutor` thread. All Qt widget operations must occur on the main thread. Executing them on a pool thread causes undefined behavior: segfault, assertion failure, or silent corruption depending on macOS/Qt version.

**Fix:**
```python
# Replace direct pool submission of GUI update with:
QTimer.singleShot(0, lambda: widget.update_function(data))
```

---

### P0-09 — `self.target` list read from worker threads while main thread modifies it

**Source:** sre (SRE-3)
**Evidence:** `xpcsviewer/file_locator.py:120-151`

`self.target` (the list of loaded datasets) is read by worker threads to iterate over files. The main thread can append, remove, or replace entries in this list during `load_path()` or `import_mask()`. CPython dict/list mutation is not atomic at the semantic level (only individual bytecodes are GIL-protected). A worker reading `self.target` while the main thread replaces it with a new list will observe the old list, potentially operating on stale or partially-constructed data.

**Fix:**
```python
# Worker side: snapshot at submission time
target_snapshot = list(self.target)  # under a lock if target is mutated concurrently
# Pass snapshot to worker, not the live reference
```

---

### P0-10 — Unbounded JIT cache growth in `qmap.py` from Python `if/elif` inside traced function

**Source:** jax-pro (JAX-N-01)
**Evidence:** `xpcsviewer/simplemask/qmap.py:179`

A Python `if/elif` on `orientation` inside a `@jax.jit`-decorated closure causes JAX to retrace for every new concrete value of `orientation`. Since `orientation` is a string argument (not a traced JAX array), JAX treats it as a static value and creates a new compilation entry per unique value. If `orientation` can take many values, `_JIT_CACHE` grows without bound and each new orientation triggers a full XLA recompilation.

**Fix:** Pass `orientation` as a `static_argnums` argument to ensure JAX correctly caches per-orientation compilations, or use `functools.lru_cache` on the JIT factory function (already partially done — verify the cache key includes orientation).

---

## 2. P1 — High: Observable Wrong Behaviour / Reliability

### P1-01 — Stale plot data applied to GUI after `cancel_all_operations`

**Source:** sre (P0-2 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/async_kernel.py:263-279`

Between `cancel_all_workers()` and `_disconnect_signals()`, a completing worker can emit `signals.finished`. The lambda `_on_plot_ready` fires after the operation has been removed from `active_operations`, but still emits `plot_ready`, delivering stale data to `XpcsViewer.on_async_plot_ready`. The stale result is applied to the visible plot.

**Fix:** In `_on_plot_ready`, check whether `self._signal_connections` still contains `operation_id` before emitting `plot_ready`. If already removed (cancelled), discard the result.

---

### P1-02 — `del self.active_plot_operations[operation_id]` raises `KeyError` after cancel

**Source:** sre (P1-2 in threading_audit.md)
**Evidence:** `xpcsviewer/xpcs_viewer.py:1268`

`on_async_plot_ready` calls `del self.active_plot_operations[operation_id]` unconditionally. If the operation was already cancelled via `cancel_async_operation` (which also deletes the entry), this raises `KeyError`.

**Fix:**
```python
self.active_plot_operations.pop(operation_id, None)
```

---

### P1-03 — `WeakValueDictionary` read from worker thread without lock

**Source:** sre (P1-3 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/plot_workers.py:287`, `xpcsviewer/viewer_kernel.py:786`

`ViewerKernel._current_dset_cache` is a `weakref.WeakValueDictionary`. Python's `WeakValueDictionary` is not thread-safe: its internal finalizer callback can delete entries during a `.get()` call on another thread, raising `KeyError` inside the CPython implementation. The main thread writes to this cache (`_current_dset_cache[fname] = dset`) concurrently with workers reading from it.

**Fix:** Protect all accesses to `_current_dset_cache` with a `threading.Lock` in `ViewerKernel`, or capture the required dataset reference in the worker constructor rather than reading shared mutable state during `do_work()`.

---

### P1-04 — `_g2_bayesian_worker_active` flag not reset on `signals.cancelled`

**Source:** sre (P1-5 in threading_audit.md), confirmed by debugger
**Evidence:** `xpcsviewer/xpcs_viewer.py:3900-3907`, `4060-4067`

`BayesianFitWorker` inherits `BaseAsyncWorker.run()`, which emits `signals.cancelled` on `InterruptedError`. Neither `_fit_g2_bayesian` nor `_fit_diff_bayesian` connects `worker.signals.cancelled` to a handler. If the worker is cancelled (via shutdown or `cancel_all_operations`), the flag stays `True` permanently, locking the "Fit Bayesian" button.

**Fix:**
```python
def _on_g2_bayesian_cancelled(self, worker_id, reason):
    self._g2_bayesian_worker_active = False
    self.btn_g2_bayesian.setEnabled(True)
    self.btn_g2_bayesian.setText("Fit Bayesian")


# In _fit_g2_bayesian:
worker.signals.cancelled.connect(self._on_g2_bayesian_cancelled)
```
Repeat for `_diff_bayesian_worker_active`.

---

### P1-05 — `HealthMonitor.stop_monitoring()` deadlocks: holds lock while joining thread

**Source:** sre (SRE-6)
**Evidence:** `xpcsviewer/utils/health_monitor.py:204`

`stop_monitoring()` acquires a lock and then calls `thread.join()`. The monitoring thread may attempt to acquire the same lock in its loop body, causing a deadlock. The join will hang indefinitely if the thread is waiting for the lock.

**Fix:** Set the stop flag under the lock, then release the lock before calling `join()`:
```python
with self._lock:
    self._monitoring_active = False
self._monitor_thread.join(timeout=5.0)
```

---

### P1-06 — `TwotimePlotWorker` forks child processes while parent holds HDF5 connection

**Source:** sre (SRE-7)
**Evidence:** `xpcsviewer/module/twotime_utils.py:324`

`multiprocessing.Pool` workers (forked from the parent) inherit the parent's HDF5 file descriptors. When both parent and child attempt to read the same HDF5 file, HDF5's internal state (especially with the default threading model) can be corrupted. This is a classic fork-safety violation.

**Fix:** Close all HDF5 connections before `Pool()` creation, or switch from `fork` to `spawn` start method:
```python
from multiprocessing import get_context

pool = get_context("spawn").Pool(...)
```

---

### P1-07 — `SimpleMask` signal connections doubled on window re-creation

**Source:** debugger (D01)
**Evidence:** `xpcsviewer/xpcs_viewer.py:2158`

When `open_simplemask()` is called a second time (e.g., user closes and reopens the mask editor), a new `SimpleMaskWindow` is created and `mask_exported` / `qmap_exported` signals are connected to `import_mask` / `import_partition`. The old connections (from the first window) are not disconnected. If the old `SimpleMaskWindow` is still alive (e.g., just hidden), exporting a mask will trigger `import_mask` twice.

**Fix:** Disconnect old signal connections before creating a new window, or reuse the existing window instance:
```python
if self.simplemask_window is not None:
    try:
        self.simplemask_window.mask_exported.disconnect(self.import_mask)
        self.simplemask_window.qmap_exported.disconnect(self.import_partition)
    except (RuntimeError, TypeError):
        pass
```

---

### P1-08 — Bayesian fit results not cleared on file switch → stale diagnosis window

**Source:** debugger (D04)
**Evidence:** `xpcsviewer/xpcs_viewer.py:266`

`self._g2_bayesian_results` is a dict accumulated per Q-bin. When the user switches to a different dataset (different `XpcsFile`), the dict is not cleared. Clicking "Show Diagnosis" after switching datasets shows the old file's fit results, labeled with the new file's Q-values. This is silent data misattribution.

**Fix:** Clear `_g2_bayesian_results`, `_g2_bayesian_data`, `_diff_bayesian_result`, and `_diff_bayesian_data` in the dataset-switch handler (wherever `update_plot` / `load_path` changes the active file).

---

### P1-09 — `plot_kwargs_record` not reset on dataset switch → stale plot state

**Source:** debugger (D03)
**Evidence:** `xpcsviewer/xpcs_viewer.py:170`

`plot_kwargs_record` accumulates per-tab plot state. When the user switches datasets, this record is not reset. Subsequent plots inherit the previous dataset's axis limits, color scales, and fitting parameters, producing misleading visual continuity between unrelated datasets.

**Fix:** Reset `plot_kwargs_record` to its initial state in the dataset-switch handler.

---

### P1-10 — Session restore saves `target_files` but never re-loads them

**Source:** debugger (D02)
**Evidence:** `xpcsviewer/xpcs_viewer.py:985`

`_collect_session_state()` serializes the list of loaded file paths. `_restore_session_state()` reads them back but calls no equivalent of `load_path()`. On next launch, the session appears to restore but no files are loaded, and the target list is empty. Users lose their session context silently.

**Fix:** In `_restore_session_state()`, call `load_path(path)` for each restored file path (with error handling for moved/deleted files).

---

### P1-11 — `stability`/`intensity` `apply_*_result` methods run synchronous computation on main thread

**Source:** debugger (D07)
**Evidence:** `xpcsviewer/xpcs_viewer.py:1387`

`apply_stability_result()` and `apply_intensity_result()` include data transformation steps (normalization, reshaping) that should have been completed in the worker. These run on the main thread (Qt event loop), blocking the UI for datasets with many Q-bins or long time series.

**Fix:** Move all data transformation into the corresponding `PlotWorker.do_work()` so that `apply_*_result` only performs the final PyQtGraph/Matplotlib update.

---

### P1-12 — `_emit_operation_progress` iterates `active_operations` without snapshot

**Source:** sre (P1-1 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/async_kernel.py:383`

Iterating `self.active_operations.items()` while a cancellation event (processed in the same Qt event loop tick) can delete entries mid-iteration raises `RuntimeError: dictionary changed size during iteration`.

**Fix:**
```python
for op_id, w_id in list(self.active_operations.items()):
```

---

### P1-13 — `UnifiedThreadingManager` `_stats` dict and `qsize()` read outside `_task_lock`

**Source:** sre (SRE-5)
**Evidence:** `xpcsviewer/threading/unified_threading.py:251`

`_stats` mutations in `_handle_task_completion` are guarded by `_task_lock`, but reads in `get_performance_stats()` and the `qsize()` call on `_task_queue` are not. This produces inconsistent snapshots and potential torn reads on 32-bit counters (though CPython GIL makes this safe for simple int increments). The main risk is `qsize()` returning a stale value used for load-balancing decisions.

**Fix:** Read `_stats` under `_task_lock`; use `_task_queue.qsize()` only as an advisory hint (document it as non-authoritative).

---

### P1-14 — `closeEvent` shuts down `UnifiedThreadingManager` after `QThreadPool.waitForDone`

**Source:** sre (P1-6 in threading_audit.md), confirmed by debugger (THR-A6)
**Evidence:** `xpcsviewer/xpcs_viewer.py:909-950`

`_monitor_system()` continues running and may emit `QMetaObject.invokeMethod` calls after Qt objects begin teardown. `shutdown_unified_threading()` is called after `waitForDone(5000)` via the dynamic import loop. The 2-second join of `_monitor_thread` in `UnifiedThreadingManager.shutdown()` may fire while Qt event processing is disabled.

**Fix:** Call `shutdown_unified_threading()` before `thread_pool.waitForDone()` in `closeEvent`.

---

### P1-15 — `MaskSchema` rejects bool dtype — primary producer uses bool masks

**Source:** python-pro (T-07), type-analyzer
**Evidence:** `xpcsviewer/simplemask/simplemask_kernel.py:133`, `xpcsviewer/schemas/validators.py:715`

`MaskSchema.__post_init__()` validates that mask dtype is `int32` or `int64`. `SimpleMaskKernel` produces bool masks (`np.ndarray` with dtype `bool`). Any attempt to create a `MaskSchema` from a `SimpleMaskKernel` mask — e.g., via `HDF5Facade.write_mask()` — raises a `ValueError`. The HDF5 write path is broken for the primary use case.

**Fix:** Either accept `bool` dtype in `MaskSchema` (and document that `True` = valid pixel), or add an explicit coercion step in `SimpleMaskKernel.export_mask()`:
```python
mask_int = mask.astype(np.int32)
schema = MaskSchema(mask=mask_int, ...)
```

---

### P1-16 — `ObjectRegistry.clear_all()` drops references without calling cleanup

**Source:** debugger (THR-A2)
**Evidence:** `xpcsviewer/threading/cleanup_optimized.py:52-55`

`clear_all()` only calls `self.objects.clear()`. For `XpcsFile` instances, this means computation cache entries in the shared memory manager are never evicted, and no `close()` is called. On long sessions with many files loaded, this leaks the memory manager's computation cache indefinitely.

**Fix:** Before clearing, call cleanup on registered objects:
```python
def clear_all(self):
    with self._lock:
        objs = list(self.objects.values())
        self.objects.clear()
    for obj in objs:
        if hasattr(obj, "close"):
            try:
                obj.close()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")
```

---

### P1-17 — `CleanupScheduler.execute_pending_cleanup()` holds lock during task execution

**Source:** debugger (THR-A3)
**Evidence:** `xpcsviewer/threading/cleanup_optimized.py:111-130`

The `_lock` is held for the entire iteration including calling each cleanup function. Any cleanup function that calls `schedule_cleanup()` (which also acquires `_lock`) will deadlock.

**Fix:** Collect due tasks under the lock, then execute outside:
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

### P1-18 — `stretched_exp` beta at NLSQ lower bound → `log Beta(0.01) = -inf` → NUTS init failure

**Source:** jax-pro (JAX-N-06)
**Evidence:** `xpcsviewer/fitting/sampler.py:261`

When NLSQ returns `beta=0.01` (the lower bound), the warm-start initial value fed to NUTS results in `log p(0.01 | Beta(2,2)) = -inf`. NumPyro NUTS raises `ValueError: initial position has -inf log probability` and the Bayesian fit fails immediately without a clear error message.

**Fix:** Clamp the NLSQ warm-start value away from the prior's zero-density regions before passing to NUTS:
```python
init_params["beta"] = float(np.clip(init_params.get("beta", 0.5), 0.05, 0.95))
```

---

### P1-19 — `jax_enable_x64` called lazily but `@_maybe_jit` applied at import time → float32 leakage

**Source:** jax-pro (JAX-N-05)
**Evidence:** `xpcsviewer/backends/__init__.py:34`, `xpcsviewer/fitting/models.py:44`

`jax.config.update("jax_enable_x64", True)` is called when `get_backend()` first returns a `JAXBackend`. Model functions are decorated with `@_maybe_jit` at module import time, before `get_backend()` is called. If any code imports `fitting.models` before activating the JAX backend, the JIT-compiled functions are traced in float32 mode. Subsequent enabling of x64 does not retrace already-compiled functions.

**Fix:** Move `jax.config.update("jax_enable_x64", True)` to the top of `backends/__init__.py` (unconditional, before any imports that could trigger tracing), or add a `functools.lru_cache`-busting mechanism that forces recompilation after backend activation.

---

### P1-20 — `numpy.bytes_` unit string decoding with NumPy 2.x produces `b'nm^-1'` repr

**Source:** python-pro (T-09)
**Evidence:** `xpcsviewer/io/hdf5_facade.py:125`

h5py returns byte strings as `numpy.bytes_` objects in NumPy 2.x. Without explicit `.decode("utf-8")`, comparing against string literals (`"nm^-1"`) fails silently (the comparison returns `False` and no error is raised), causing unit validation in `QMapSchema` to accept the wrong string or reject a valid unit.

**Fix:**
```python
unit = (
    raw_unit.decode("utf-8")
    if isinstance(raw_unit, (bytes, np.bytes_))
    else str(raw_unit)
)
```
Apply at all h5py unit string reads in `hdf5_facade.py`.

---

### P1-21 — `AsyncDataPreloader` leaks signal connection on cancel/error before preload completes

**Source:** sre (P2-1 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/async_kernel.py:476`

`start_preload()` connects `async_kernel.data_loaded` to `_on_preload_completed`. If the preload errors or is cancelled before `data_loaded` fires, the connection is never removed. Every subsequent `data_loaded` emission (from any operation) invokes the stale handler, potentially applying preload results to the wrong context.

**Fix:** Wrap the connection in a one-shot pattern:
```python
def _on_preload_completed_once(self, op_id, result):
    try:
        self.async_kernel.data_loaded.disconnect(self._on_preload_completed_once)
    except (RuntimeError, TypeError):
        pass
    self._on_preload_completed(op_id, result)
```

---

### P1-22 — `g2mod.py` passes arrays to PyQtGraph without `ensure_numpy`

**Source:** python-pro (T-06)
**Evidence:** `xpcsviewer/module/g2mod.py:795`

JAX arrays passed to PyQtGraph will fail or produce incorrect plots. `ensure_numpy()` must be called at all JAX→PyQtGraph boundaries per BUG-027 pattern.

**Fix:**
```python
from xpcsviewer.backends._conversions import ensure_numpy

# Before setData/setImage calls:
plot_item.setData(ensure_numpy(x), ensure_numpy(y))
```

---

### P1-23 — `legacy.py` module-level singleton captures stale backend after `reset_backend()`

**Source:** jax-pro (JAX-N-07)
**Evidence:** `xpcsviewer/fitting/legacy.py:116`

`_xnp = get_backend()` is evaluated at module import time and stored as a module-level global. If `reset_backend()` is called later (e.g., in tests or reconfiguration), all functions in `legacy.py` continue using the original backend instance, bypassing the reset.

**Fix:**
```python
# Replace module-level singleton with per-call lookup:
def fit_with_fixed(...):
    _xnp = get_backend()   # looked up fresh each call
    ...
```

---

## 3. P2 — Medium: Technical Debt / Maintainability

### P2-01 — `@abstractmethod` on `BackendProtocol` has no enforcement effect

**Source:** python-pro (T-05), type-analyzer
**Evidence:** `xpcsviewer/backends/_base.py`

`BackendProtocol` uses `@abstractmethod` inside a `Protocol` class. `Protocol` does not use `ABCMeta`; `@abstractmethod` has no enforcement effect unless `runtime_checkable` + `isinstance` checks are added. Neither `JAXBackend` nor `NumPyBackend` inherits from `BackendProtocol`, so no static verification that all 50+ methods are implemented.

**Fix (option A):** Make backends inherit from `BackendProtocol` and use `ABCMeta`:
```python
from abc import ABC, abstractmethod
class BackendProtocol(ABC):
    @abstractmethod
    def zeros(self, ...): ...
```
**Fix (option B):** Keep as `Protocol`, remove misleading `@abstractmethod`, add mypy `--strict` to CI to enforce structural subtyping at type-check time.

---

### P2-02 — `G2Data.to_dict()` returns raw (mutable) array references

**Source:** type-analyzer
**Evidence:** `xpcsviewer/schemas/validators.py` (G2Data)

`G2Data.to_dict()` returns references to the internal arrays directly, unlike `QMapSchema.to_dict()` which returns copies. Callers who mutate the returned dict's arrays corrupt the frozen dataclass's internal state. This breaks the defensive copy pattern established by `QMapSchema`.

**Fix:**
```python
def to_dict(self):
    return {
        "g2": self.g2.copy(),
        "g2_err": self.g2_err.copy(),
        "tau": self.tau.copy(),
        "q_values": list(self.q_values),
    }
```

---

### P2-03 — Silent `"deg"` default for missing `phis_unit` in `QMapSchema`

**Source:** python-pro (T-11)
**Evidence:** `xpcsviewer/schemas/validators.py:191`

`QMapSchema.from_dict()` silently substitutes `"deg"` when `phis_unit` is absent from the source dict. This masks missing geometry metadata and can produce wrong unit labels in plots and exports.

**Fix:** Raise `ValueError` (or at minimum log a `WARNING` with the source file path) when `phis_unit` is absent, rather than silently defaulting.

---

### P2-04 — `NLSQResult` lacks invariant enforcement; setter no-ops are confusing API

**Source:** type-analyzer
**Evidence:** `xpcsviewer/fitting/results.py:184`

`NLSQResult` is a mutable dataclass with 12 legacy `_`-prefixed fields and property delegation to `native_result`. Setter properties emit `DeprecationWarning` but do nothing, making it impossible to tell programmatically whether a set took effect. `__post_init__` does not validate field consistency.

**Fix (minimal):** Convert `_`-prefixed fields to proper validated properties with `__post_init__` checking, and raise `TypeError` (not silently warning) when a setter is called on a `native_result`-backed instance.

---

### P2-05 — `FitResult.predict()` returns zeros (placeholder implementation)

**Source:** type-analyzer
**Evidence:** `xpcsviewer/fitting/results.py`

`FitResult.predict()` returns a zero array regardless of the sampled parameters. Any code that calls `predict()` for model evaluation will silently get wrong results.

**Fix:** Implement `predict()` using the posterior mean parameters and the model function, or raise `NotImplementedError` explicitly.

---

### P2-06 — `MaskAssemble` history stack is publicly mutable

**Source:** type-analyzer
**Evidence:** `xpcsviewer/simplemask/area_mask.py:338`

`mask_record`, `mask_ptr`, and `workers` are public attributes with no protection against external mutation. Direct modification of `mask_ptr` bypasses the undo/redo invariant (that `mask_ptr_min <= mask_ptr <= len(mask_record)-1`).

**Fix:** Prefix with `_` and expose through property accessors that enforce the invariant.

---

### P2-07 — `UnifiedThreadingManager` reads `pool._max_workers` and `pool._threads` (private CPython attrs)

**Source:** sre (P2-5, P2-6 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/unified_threading.py:511, 566`

`ThreadPoolExecutor._max_workers` and `._threads` are CPython implementation details not in the public API. Python 3.13+ may remove them.

**Fix:** Store `max_workers` in `_pool_configs` at pool creation time; count live threads via the public `concurrent.futures` API or by maintaining an explicit counter.

---

### P2-08 — `WorkerManager.worker_results` and `worker_errors` grow unbounded

**Source:** sre (P2-3), debugger (THR-A5)
**Evidence:** `xpcsviewer/threading/async_workers.py:631, 643`

No eviction policy. Long sessions accumulate thousands of completed-worker payloads.

**Fix:** Use `collections.OrderedDict` with a max-size eviction policy (keep last 200 entries), or clear entries immediately after they are consumed by the caller.

---

### P2-09 — Fragile `"q"` key lookup in `simplemask_kernel.py`

**Source:** python-pro (T-12)
**Evidence:** `xpcsviewer/simplemask/simplemask_kernel.py:295`

`qmap["q"]` is accessed without `dict.get()` or a guard. If the qmap dict does not contain `"q"` (e.g., a non-standard HDF5 format), this raises `KeyError` with no helpful error message.

**Fix:**
```python
q_data = self.qmap.get("q")
if q_data is None:
    raise ValueError(
        f"Q-map missing required 'q' field; available keys: {list(self.qmap.keys())}"
    )
```

---

### P2-10 — `ProgressManager.request_cancel` only updates UI, relies on multi-hop signal chain

**Source:** sre (P2-4 in threading_audit.md)
**Evidence:** `xpcsviewer/threading/progress_manager.py:497-503`

Cancellation dispatched via 3 signal hops. If the event loop drains during teardown, the chain may not complete.

**Fix:** `request_cancel` should call the actual worker manager cancellation directly or at least document the chain explicitly with a comment.

---

### P2-11 — `get_backend()` called inside model function on every NUTS evaluation

**Source:** jax-pro (JAX-N-08)
**Evidence:** `xpcsviewer/fitting/legacy.py:396`

`get_backend()` inside a JAX-traced function is called on every NUTS step, performing a module lookup each time. This was fixed previously (BUG-026) but regressed in `legacy.py`.

**Fix:** Cache `_xnp = get_backend()` at module level (with the caveat from P1-23 — use a lazy initializer that checks if backend has changed).

---

### P2-12 — Double `update_plot` on session restore

**Source:** debugger (D08)
**Evidence:** `xpcsviewer/xpcs_viewer.py:1009`

Session restore calls `load_path()` which triggers `update_plot`, then a tab-change signal also triggers `update_plot`. The second call is redundant and doubles load time on startup.

**Fix:** Block the tab-change signal during session restore using a guard flag:
```python
self._restoring_session = True
try:
    self._restore_session_state(session)
finally:
    self._restoring_session = False
```
Check the flag in `update_plot`.

---

### P2-13 — `calibration.py` uses two separate JIT compilations for grad and eval of same function

**Source:** jax-pro (audit notes)
**Evidence:** `xpcsviewer/simplemask/calibration.py:79, 268`

`jax.jit(jax.grad(loss_fn))` and `jax.jit(loss_fn)` create two separate XLA traces for the same function body. `jax.value_and_grad` produces both value and gradient in a single pass.

**Fix:**
```python
@jax.jit
def loss_and_grad(params):
    return jax.value_and_grad(loss_fn)(params)
```

---

### P2-14 — `@Slot` decorator signatures on `WorkerManager` handlers mismatch signal signatures

**Source:** debugger (THR-A7)
**Evidence:** `xpcsviewer/threading/async_workers.py:628-660`

`@Slot(str, object)` on `_on_worker_finished` (which actually receives `worker_id` from the lambda closure, not the signal). The decorator misleads mypy and static analysis. No runtime bug, but registers the wrong signature with Qt's meta-object system.

**Fix:** Remove `@Slot` decorators from private handler methods that are called only via lambda closures, not directly as slot connections.

---

## 4. Compound Bugs — Issues That Must Be Fixed Together

### COMPOUND-A — Bayesian fit is completely broken (P0-01 + P0-03 + P1-18)

All three issues affect the same code path. P0-01 crashes the fit with `KeyError` before any result is produced. P0-03 can produce NaN samples that corrupt the MCMC chain. P1-18 prevents NUTS initialization. They must all be fixed together to have a working Bayesian pipeline.

**Fix order:** P0-01 → P1-18 → P0-03 (in that order: unblock execution, then fix initialization, then fix NaN propagation).

### COMPOUND-B — Schema dtype validation rejects all legacy HDF5 data (P0-04 + P0-05 + P1-15)

`G2Data`, `QMapSchema`, and `MaskSchema` all enforce dtypes that legacy files violate. The coercion fixes (P0-04, P0-05) and the bool→int32 coercion (P1-15) must be applied together, since a user loading a legacy file triggers all three validators in sequence.

**Fix order:** P0-04 + P0-05 + P1-15 (can be done in parallel — all in `hdf5_facade.py` and `simplemask_kernel.py`).

### COMPOUND-C — `QMapSchema` access broken at two levels (P0-02 + type-analyzer note)

P0-02 (`__dict__` on frozen dataclass) and the facade callers bypassing coercion (T-02/T-13) both affect the same `QMapSchema` path. Fix P0-05 (coercion at read time) first to ensure the schema is constructed correctly, then fix P0-02 (attribute access) to ensure it is read correctly.

**Fix order:** P0-05 → P0-02.

### COMPOUND-D — Shutdown race: Bayesian worker flag left set + UI buttons disabled (P1-04 + P1-14)

If shutdown races with a Bayesian fit (P1-14 — wrong shutdown order allows Qt teardown while workers run), the cancelled signal may never be delivered (P1-04 — cancelled signal not connected). Fix P1-14 first (correct shutdown order), then P1-04 (connect cancelled handler).

**Fix order:** P1-14 → P1-04.

### COMPOUND-E — Stale plot on cancel is both a threading issue and a state issue (P1-01 + P1-09)

A stale plot applied after cancellation (P1-01) combines with stale `plot_kwargs_record` (P1-09) to produce a doubly-corrupted visible state. Fix P1-01 to prevent the stale apply, and P1-09 to reset state on dataset switch.

---

## 5. Recommended Fix Order (Dependency-Aware)

### Wave 1 — Unblock core functionality (fix before any testing)

| Priority | ID | Description |
|----------|----|-------------|
| 1 | P0-01 | MCMC `extra_fields` — Bayesian fit crashes on every run |
| 2 | P0-07 | `@Slot` on `invokeMethod` targets — `UnifiedThreadingManager` signals silently dropped |
| 3 | P0-08 | GUI updates on `ThreadPoolExecutor` thread — potential segfault |
| 4 | P0-04 | `G2Data` float32→float64 coercion — all legacy files fail |
| 5 | P0-05 | `QMapSchema` float32→float64 coercion — all legacy files fail |
| 6 | P0-02 | `__dict__` on frozen dataclass — runtime `AttributeError` |
| 7 | P1-15 | `MaskSchema` bool dtype — HDF5 mask write broken for primary use case |

### Wave 2 — Fix crash/data-corruption risks under normal use

| Priority | ID | Description |
|----------|----|-------------|
| 8  | P0-03 | NaN gradient clipping in `double_exp_model` |
| 9  | P0-09 | Target list race condition in `file_locator.py` |
| 10 | P0-10 | Unbounded JIT cache growth in `qmap.py` |
| 11 | P1-06 | HDF5 + multiprocessing fork safety in `twotime_utils.py` |
| 12 | P1-18 | stretched_exp beta at NLSQ bound → NUTS init failure |
| 13 | P1-19 | float64 enablement before JIT trace |

### Wave 3 — Fix reliability and UI correctness issues

| Priority | ID | Description |
|----------|----|-------------|
| 14 | P1-04 | Bayesian worker flag not reset on `cancelled` |
| 15 | P1-14 | Shutdown order: `UnifiedThreadingManager` before `waitForDone` |
| 16 | P1-05 | `HealthMonitor` deadlock in `stop_monitoring` |
| 17 | P1-01 | Stale plot applied after cancel |
| 18 | P1-02 | `KeyError` on `del active_plot_operations` after cancel |
| 19 | P1-03 | `WeakValueDictionary` read from worker thread |
| 20 | P1-12 | `active_operations` iteration without snapshot |
| 21 | P1-07 | SimpleMask signal doubling on re-creation |
| 22 | P1-17 | `CleanupScheduler` holds lock during task execution |
| 23 | P1-16 | `ObjectRegistry.clear_all()` skips `XpcsFile.close()` |

### Wave 4 — Fix UX and session state issues

| Priority | ID | Description |
|----------|----|-------------|
| 24 | P1-08 | Stale Bayesian results on file switch |
| 25 | P1-09 | Stale `plot_kwargs_record` on file switch |
| 26 | P1-10 | Session restore does not re-load files |
| 27 | P1-11 | Sync computation on main thread in `apply_*_result` |
| 28 | P1-20 | `numpy.bytes_` unit string decoding |
| 29 | P1-21 | `AsyncDataPreloader` signal leak on cancel |
| 30 | P1-22 | `g2mod.py` missing `ensure_numpy` |
| 31 | P1-23 | Legacy.py stale backend singleton |

### Wave 5 — Technical debt and API hygiene (can be batched as a refactor sprint)

P2-01 through P2-14 (all can be addressed in a single refactor pass without risk of regressions, since they are API/documentation issues, not runtime behavior changes).

---

## 6. Architecture Improvement Recommendations

### A. Centralize schema coercion in `HDF5Facade`

All dtype coercion (float32→float64, bytes→str, bool→int32) should happen in `HDF5Facade` before schema construction, not inside `__post_init__` validators. This follows the "validate once, at the boundary" principle. Schemas should assume correct types and focus on invariant checking.

### B. Make `BackendProtocol` enforceable

Either switch to `ABC` inheritance or add a mypy plugin to enforce structural subtyping. The current `Protocol` with `@abstractmethod` gives false confidence. Add a test that instantiates both backends and calls every documented method.

### C. Decouple `NLSQResult` legacy fields from `native_result` delegation

Replace the 12 `_`-prefixed legacy fields with a single `_legacy_params: dict | None` field, and make all property delegation explicit with clear fallback semantics. The current dual-path design (legacy `_` fields vs `native_result` delegation) makes invariants impossible to reason about.

### D. Move `safe_shutdown()` and `closeEvent` shutdown logic to a single `ApplicationShutdown` class

Currently shutdown logic is split between `cli_main.py:safe_shutdown()` (SIGTERM path) and `xpcs_viewer.py:closeEvent()` (GUI close path). They cover overlapping but non-identical sets of subsystems. A single `ApplicationShutdown` class with a well-defined order (workers → HDF5 pool → JAX cleanup → cleanup system → Qt teardown) would eliminate the current gaps.

### E. Introduce a `DatasetContext` object to carry active-file state

`_g2_bayesian_results`, `_diff_bayesian_result`, `plot_kwargs_record`, and related fields accumulate per-file state on `XpcsViewer` directly. When the user switches files, these must all be reset together. Bundling them into a `DatasetContext` dataclass makes the reset atomic and auditable.

---

*End of master fix list. 10 P0, 23 P1, 14 P2 findings. 5 compound bug groups. Recommended fix wave structure above.*
