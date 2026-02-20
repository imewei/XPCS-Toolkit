# XPCS Viewer — Architecture Map

Generated: 2026-02-19
Mapped by: explorer agent (Phase 1 audit)

---

## 1. Module Dependency Graph

```
xpcsviewer/
├── cli_main.py                  ← Entry point (CLI + GUI launch)
│     └── xpcs_viewer.py        ← Main GUI window
│
├── xpcs_viewer.py               [XpcsViewer(QMainWindow)]
│     ├── viewer_kernel.py       [ViewerKernel]
│     ├── xpcs_file.py           [XpcsFile]
│     ├── threading/
│     │     ├── async_kernel.py  [AsyncViewerKernel, AsyncDataPreloader]
│     │     ├── async_workers.py [BaseAsyncWorker, WorkerManager, PlotWorker, ...]
│     │     ├── plot_workers.py  [SaxsPlotWorker, G2PlotWorker, ...]
│     │     ├── bayesian_worker.py [BayesianFitWorker]
│     │     ├── unified_threading.py [UnifiedThreadingManager]
│     │     ├── progress_manager.py [ProgressManager]
│     │     ├── cleanup_optimized.py [OptimizedCleanupSystem]
│     │     └── gui_integration.py [ThreadingIntegrator]
│     ├── simplemask/            [SimpleMaskWindow, SimpleMaskKernel]
│     ├── fitting/               [nlsq_optimize, run_*_fit, FitResult, NLSQResult]
│     ├── backends/              [get_backend, ensure_numpy, io_adapter]
│     ├── schemas/               [QMapSchema, GeometryMetadata, G2Data, ...]
│     ├── io/                    [HDF5Facade]
│     ├── gui/
│     │     ├── state/           [SessionManager, RecentPathsManager]
│     │     ├── theme/           [ThemeManager]
│     │     ├── widgets/         [ToastManager, CommandPalette, ...]
│     │     ├── bayesian_diagnosis.py [BayesianDiagnosisWindow]
│     │     └── qt_compat.py    [Qt import shim]
│     └── utils/                 [get_logger, sanitize_path, log_timing, ...]
│
└── viewer_kernel.py             [ViewerKernel(XpcsFileModule)]
      └── xpcs_file.py           [XpcsFile]
```

### Key inter-module imports

| Consumer | Imported From | What |
|---|---|---|
| xpcs_viewer.py | threading.async_kernel | AsyncViewerKernel, AsyncDataPreloader |
| xpcs_viewer.py | threading.progress_manager | ProgressManager |
| xpcs_viewer.py | simplemask.simplemask_window | SimpleMaskWindow |
| xpcs_viewer.py | fitting | fit_single_exp, fit_double_exp |
| xpcs_viewer.py | threading.bayesian_worker | BayesianFitWorker |
| xpcs_viewer.py | gui.state | SessionManager, RecentPathsManager |
| xpcs_viewer.py | gui.theme | ThemeManager |
| threading.async_kernel | threading.async_workers | WorkerManager, BaseAsyncWorker |
| threading.plot_workers | viewer_kernel | ViewerKernel (passed in) |
| fitting.sampler | fitting.nlsq | nlsq_optimize |
| fitting.sampler | fitting.models | single_exp_model, single_exp_func, etc. |
| fitting.models | backends | get_backend (for _xnp) |
| io.hdf5_facade | schemas.validators | QMapSchema, GeometryMetadata, etc. |
| simplemask.simplemask_kernel | simplemask.qmap | compute_qmap |
| simplemask.simplemask_kernel | simplemask.area_mask | MaskAssemble |
| xpcs_file.py | fitting.legacy | fit_with_fixed, robust_curve_fit, etc. |

---

## 2. Thread Boundary Diagram

```
MAIN THREAD (Qt event loop)
=========================================================================
XpcsViewer (QMainWindow)
  |
  |-- ProgressManager          <- status bar + dialog UI
  |-- ThemeManager             <- stylesheet application
  |-- SessionManager           <- JSON serialization at startup/shutdown
  |-- SimpleMaskWindow         <- child QMainWindow, runs on main thread
  |
  |-- init_async_kernel()      <- creates AsyncViewerKernel (main thread)
  |
  |-- update_plot_async(tab)   <- dispatches to QThreadPool
  |     +-- async_vk.plot_*_async()
  |           +-- _submit_plot_worker(worker, op_id)
  |                 +-- worker_manager.submit_worker(worker)
  |                       +-- QThreadPool.start(worker)  -------->
  |                                                                |
  |  <-- worker.signals.finished -> on_async_plot_ready()  <------+
  |        +-- apply_*_result(result)  <- back on main thread via signal
  |
  |-- _fit_g2_bayesian()       <- BayesianFitWorker dispatched
  |     +-- thread_pool.start(BayesianFitWorker)  ----------------->
  |  <-- signals.finished -> _on_g2_bayesian_finished()  <----------+
  |
  +-- submit_job() / start_avg_job()   <- AverageToolbox (separate pool)

WORKER THREADS (QThreadPool -- one pool)
=========================================================================
BaseAsyncWorker.run() [slot decorated]
  |-- calls do_work()   <- subclass-specific computation
  |-- emits signals.started
  |-- emits signals.progress (rate-limited)
  |-- emits signals.finished(result)   -> main thread slot
  |-- emits signals.cancelled          -> main thread slot
  +-- emits signals.error              -> main thread slot

Plot Workers (BasePlotWorker < BaseAsyncWorker):
  SaxsPlotWorker       <- ViewerKernel.plot_saxs_2d()
  G2PlotWorker         <- ViewerKernel.plot_g2() + _perform_g2_fitting()
  TwotimePlotWorker    <- ViewerKernel.plot_twotime()
  IntensityPlotWorker  <- ViewerKernel.plot_intt()
  StabilityPlotWorker  <- ViewerKernel.plot_stability() + _compute_stability_metrics()
  QMapPlotWorker       <- ViewerKernel.plot_qmap()

BayesianFitWorker (BaseAsyncWorker):
  +-- do_work() -> fit_func(x, y, yerr)   <- NumPyro NUTS on JAX

UnifiedThreadingManager (separate from QThreadPool):
  +-- submit_task() -> concurrent.futures pools (CPU/IO/background)
  +-- _monitor_system() -> memory pressure + pool balancing

NOTE: HDF5 reads (h5py) happen in worker threads.
      All PyQtGraph/Matplotlib calls must happen on main thread.
      ensure_numpy() called at worker->main signal boundary.
```

---

## 3. Data Flow: HDF5 -> XpcsFile -> ViewerKernel -> PlotWorkers -> PyQtGraph

```
HDF5 File on Disk
       |
       v  (load_path / add_target button)
XpcsViewer.load_path(folder)
  +-- ViewerKernel.__init__(path)
       +-- XpcsFile created per .hdf file in directory
            |
            v  (add_target -> update_plot -> update_plot_async)
XpcsFile.load_data()        [HDF5 -> NumPy, in worker thread]
  |-- fields: tau, g2, g2_err, saxs_1d, Int_t, t0, ...
  |-- enhanced_reader.read_multiple_datasets(fname, fields)
  +-- fallback: batch_read_fields(fname, fields, "alias", ftype="nexus")
       |
       v
XpcsFile attributes: .tau, .g2, .g2_err, .qmap (reshaped NumPy)

       |
       v  (async dispatch via AsyncViewerKernel)
PlotWorker.do_work()       [worker thread]
  +-- ViewerKernel.plot_g2(dryrun=False)
       +-- XpcsFile.get_g2_data()
            +-- returns NumPy arrays (g2, t_el, q_labels)
  +-- result dict built

       |  (signals.finished emitted -- crosses thread boundary)
       v
XpcsViewer.on_async_plot_ready(op_id, result)   [main thread]
  +-- apply_g2_result(result)
       +-- Matplotlib/PyQtGraph plot handlers update

SAXS 2D Flow:
  XpcsFile.saxs_2d_data (property)
    +-- batch_read_fields or enhanced_reader -> raw 2D array
    +-- ensure_numpy(raw) before setImage()
  XpcsFile._compute_saxs_log_standard/streaming/chunked
    +-- log10(img + 1) on NumPy

Q-MAP Flow:
  SimpleMaskKernel.compute_qmap()
    +-- simplemask.qmap.compute_qmap(stype, metadata)
         +-- geometry -> sqmap, dqmap, phis (NumPy float64)
    +-- MaskAssemble.update_qmap(qmap)
  HDF5Facade.read_qmap(file_path) [optional persistence]
    +-- QMapSchema.__post_init__() validates shapes, dtypes, units, NaN

MASK EXPORT Flow:
  SimpleMaskWindow.export_mask_to_viewer()
    +-- emits mask_exported(np.ndarray)
  XpcsViewer.import_mask(mask)
    +-- validates shape vs detector_shape
    +-- sets xf.mask = mask for all XpcsFile in target list
    +-- calls update_plot()
```

---

## 4. Signal/Slot Connection Map

### XpcsViewer primary connections (set in __init__)

| Signal | Slot | Thread |
|---|---|---|
| tabWidget.currentChanged | update_plot() | main |
| list_view_target.clicked | update_plot() | main |
| list_view_target.doubleClicked | show_dataset() | main |
| pg_saxs.scene.sigMouseMoved | saxs2d_mouseMoved() | main |
| pushButton_plot_saxs2d.clicked | plot_saxs_2d() | main |
| pushButton_plot_saxs1d.clicked | plot_saxs_1d() | main |
| pushButton_plot_stability.clicked | plot_stability() | main |
| pushButton_plot_intt.clicked | plot_intensity_t() | main |
| g2_fitting_function.currentIndexChanged | update_g2_fitting_function() | main |
| comboBox_qmap_target.currentIndexChanged | update_plot() | main |
| btn_g2_bayesian.clicked | _fit_g2_bayesian() | main |
| btn_g2_diagnosis.clicked | _show_g2_diagnosis() | main |
| btn_diff_bayesian.clicked | _fit_diff_bayesian() | main |
| btn_diff_diagnosis.clicked | _show_diff_diagnosis() | main |
| theme_manager.theme_changed | _on_theme_changed() | main |
| progress_manager.operation_cancelled | cancel_async_operation() | main |

### AsyncViewerKernel connections (set in init_async_kernel)

| Signal | Slot | Thread |
|---|---|---|
| async_vk.plot_ready(op_id, result) | XpcsViewer.on_async_plot_ready() | main (queued) |
| async_vk.operation_error(op_id, msg) | XpcsViewer.on_async_operation_error() | main (queued) |
| async_vk.operation_progress(...) | progress_manager.update_progress() | main (queued) |

### Plot Worker Signals (per-operation, set in _submit_plot_worker)

| Signal | Slot |
|---|---|
| worker.signals.finished | lambda -> AsyncViewerKernel._on_plot_ready(op_id, result) |
| worker.signals.error | lambda -> AsyncViewerKernel._on_operation_error(op_id, ...) |
| worker.signals.cancelled | lambda -> AsyncViewerKernel._on_operation_cancelled(op_id) |

### BayesianFitWorker Signals (set in _fit_g2_bayesian)

| Signal | Slot |
|---|---|
| worker.signals.finished | XpcsViewer._on_g2_bayesian_finished() |
| worker.signals.error | XpcsViewer._on_g2_bayesian_error() |

### SimpleMaskWindow Signals

| Signal | Connected To | Where |
|---|---|---|
| mask_exported(np.ndarray) | XpcsViewer.import_mask() | open_simplemask() |
| qmap_exported(dict) | XpcsViewer.import_partition() | open_simplemask() |

---

## 5. JIT Compilation Boundary Inventory

### JAX JIT points

| Location | What is JIT-compiled | Notes |
|---|---|---|
| fitting/models.py:_maybe_jit() | single_exp_func, double_exp_func, stretched_exp_func, power_law_func | Only when JAX_AVAILABLE=True; applied at module load |
| fitting/models.py NumPyro models | Traced by NumPyro NUTS during MCMC warmup | Internal to NumPyro |
| fitting/sampler.py:_run_mcmc() | mcmc.run() -> NUTS.sample | jax.random.PRNGKey, jnp.asarray conversions |
| backends/_jax_backend.py | .jit(), .grad(), .vmap() methods | Delegates to jax.jit etc. |
| simplemask/calibration.py | minimize_with_grad (gradient descent) | Via jax.grad + JAX arrays |
| simplemask/qmap.py | Q-map array computations | Uses backend.linspace/sin/cos -- JIT if JAX backend active |

### NOT JIT-compiled (intentionally NumPy)

| Location | Reason |
|---|---|
| xpcs_file.py all methods | HDF5 I/O boundary; NumPy required for h5py |
| viewer_kernel.py all methods | PyQtGraph interface; NumPy required |
| backends/_numpy_backend.py | Fallback path; no JIT |
| nlsq_optimize() call to nlsq.fit() | Third-party NLSQ library (not JAX) |
| All plot workers | Pass NumPy to PyQtGraph/Matplotlib |
| ensure_numpy() result | Explicitly exits JAX device |

### JIT Dispatch Logic

```
XPCS_USE_JAX=1 env var
  +-- _configure_jax() called at import
       +-- get_backend() -> JAXBackend instance
            +-- backend.jit(fn) -> jax.jit(fn)
            +-- backend.grad(fn) -> jax.grad(fn)

XPCS_USE_JAX unset or 0
  +-- get_backend() -> NumPyBackend instance
       +-- backend.jit(fn) -> fn  (identity -- no-op)
       +-- MCMC still uses JAX (jax/numpyro direct import in sampler.py)
```

---

## 6. Schema Validation Points

### Where validation occurs

| Schema | Validation Trigger | Location |
|---|---|---|
| QMapSchema | HDF5Facade.read_qmap(validate=True) | io/hdf5_facade.py:80 |
| QMapSchema | QMapSchema.from_dict(d) | schemas/validators.py |
| QMapSchema.__post_init__ | Shape equality, 2D, units whitelist, float64, NaN, mask/partition shape | schemas/validators.py:69 |
| GeometryMetadata | HDF5Facade.read_geometry_metadata() | io/hdf5_facade.py |
| G2Data | HDF5Facade.read_g2_data() | io/hdf5_facade.py |
| MaskSchema | HDF5Facade.write_mask() | io/hdf5_facade.py |
| PartitionSchema | HDF5Facade.write_partition() | io/hdf5_facade.py |

### Validation bypass

- HDF5Facade(validate=False) skips QMapSchema() and returns raw dict instead.
  - Note: BUG-029 was that validate=False still called QMapSchema(**data) (now fixed).

### ensure_numpy() boundary calls (I/O boundaries)

```
Worker thread result -> main thread:
  - apply_saxs_2d_result()   -> pg_saxs.setImage(ensure_numpy(img))
  - apply_g2_result()        -> mp_g2.plot(ensure_numpy(x), ...)
  - apply_qmap_result()      -> pg_qmap.setImage(ensure_numpy(qmap))

HDF5 write (h5py):
  - HDF5Adapter.to_hdf5(arr) -> np.asarray(arr)
  - SimpleMaskKernel.save_mask() -> ensure_numpy(mask)

PyQtGraph plot:
  - PyQtGraphAdapter.to_pyqtgraph(arr) -> ensure_numpy(arr)
```

---

## 7. Entry Points

### GUI Entry Point

```
pyproject.toml: [project.scripts]
  xpcsviewer-gui = "xpcsviewer.xpcs_viewer:main_gui"
  |
  v
xpcs_viewer.py:main_gui(path, label_style)
  +-- QApplication() + XpcsViewer(path, label_style) + app.exec()
```

### CLI Entry Point

```
pyproject.toml: [project.scripts]
  xpcsviewer = "xpcsviewer.cli_main:main"
  |
  v
cli_main.py:main()
  |-- command=twotime -> _run_twotime_batch(args)   [batch, no GUI]
  +-- (future commands)

GUI also accessible from CLI:
  cli_main.py:main_gui() -> _start_gui(path, label_style)
    +-- xpcs_viewer.main_gui()
```

### Test Entry Points

```
tests/unit/simplemask/           <- kernel, qmap, area_mask, drawing_tools, mask I/O, partition
tests/integration/               <- signal export, SimpleMask integration
tests/jax_migration/backend/     <- backend detection, device tests
tests/jax_migration/fitting/     <- fitting + sampler tests
tests/jax_migration/numerical/   <- equivalence tests
tests/jax_migration/performance/ <- JIT + benchmark
tests/jax_migration/precision/   <- float64 precision
tests/jax_migration/integration/ <- mixed JAX/NumPy env
tests/end_to_end/                <- full workflow tests
```

---

## 8. SimpleMask Subsystem Detail

```
SimpleMaskWindow (QMainWindow)
  |-- kernel: SimpleMaskKernel
  |     |-- mask_kernel: MaskAssemble      <- undo/redo history
  |     |     |-- mask_record: list[np.ndarray]
  |     |     |-- mask_ptr: int            <- current position
  |     |     +-- mask_ptr_min: int        <- default mask boundary
  |     |-- qmap: dict[str, np.ndarray]
  |     |-- metadata: dict                 <- geometry params
  |     +-- mask: np.ndarray (2D)
  |
  |-- Signals:
  |     mask_exported(np.ndarray) -> XpcsViewer.import_mask()
  |     qmap_exported(dict)       -> XpcsViewer.import_partition()
  |
  |-- Drawing flow:
  |     _on_apply_drawing()
  |       -> kernel.apply_drawing(tool, roi_params)   <- returns bool mask
  |       -> kernel.mask_apply("mask_draw")            <- syncs kernel.mask
  |
  |-- Undo/redo flow:
  |     _on_mask_action("undo"/"redo")
  |       -> kernel.mask_action(action)
  |       -> mask_kernel.redo_undo(direction)          <- moves mask_ptr
  |       -> kernel.mask_apply(action)                 <- syncs kernel.mask
  |
  +-- Drawing tools (simplemask/drawing_tools.py):
        Rectangle, Circle, Polygon, Line, Ellipse, Eraser
        <- custom PyQtGraph ROI classes (pyqtgraph_mod.py: LineROI)
```

---

## 9. Fitting Pipeline Detail

```
Synchronous fit (in worker thread via G2PlotWorker):
  G2PlotWorker._perform_g2_fitting()
    +-- XpcsFile.fit_g2() / fit_g2_robust()
         +-- fitting.legacy.fit_with_fixed()
              +-- scipy.optimize.curve_fit (legacy path)
         OR
         +-- fitting.nlsq.nlsq_optimize()
              +-- nlsq.fit() (NLSQ 0.6.0)
                   +-- returns NLSQResult

Bayesian fit (in BayesianFitWorker, separate from plot workers):
  BayesianFitWorker.do_work()
    +-- fit_single_exp(x, y, yerr) / fit_double_exp(x, y, yerr)
         +-- fitting.sampler.run_single_exp_fit()
              |-- nlsq_optimize() [warm-start]
              |    +-- nlsq.fit() -> NLSQResult.params as init_params
              |-- jnp.asarray(x, y, yerr)          [NumPy -> JAX]
              +-- _run_mcmc(single_exp_model, ...)
                   |-- NUTS(model, target_accept_prob, max_tree_depth)
                   |-- MCMC(kernel, num_warmup, num_samples, num_chains)
                   +-- mcmc.run(rng_key, *model_args, init_params=...)
                        +-- returns FitResult(samples, diagnostics)

Model functions (models.py):
  single_exp_func(x, tau, baseline, contrast) -> JAX array
  double_exp_func(x, tau1, tau2, baseline, c1, c2) -> JAX array
  stretched_exp_func(x, tau, baseline, contrast, beta) -> JAX array
  power_law_func(q, tau0, alpha) -> JAX array
  (all JIT-compiled when JAX available via _maybe_jit)

  NumPyro models (priors + likelihood):
  single_exp_model(x, y, yerr):
    tau    ~ LogNormal(0, 2)
    baseline ~ Normal(1, 0.5)
    contrast ~ Beta(2, 2)
    sigma  ~ HalfNormal(0.1)
    obs    ~ Normal(single_exp_func(x, tau, baseline, contrast), sigma)
```

---

## 10. Backend Abstraction Layer

```
xpcsviewer/backends/__init__.py
  get_backend() -> BackendProtocol
    |-- checks XPCS_USE_JAX env var -> True/False
    |-- if True and JAX importable -> JAXBackend
    +-- else -> NumPyBackend

BackendProtocol (abc-like Protocol):
  .name, .supports_gpu, .supports_jit, .supports_grad
  .zeros/ones/arange/linspace/logspace/...
  .sin/cos/arctan/exp/log/sqrt/...
  .mean/std/sum/min/max/...
  .logical_and/or/not/where/...
  .jit(fn), .grad(fn), .value_and_grad(fn), .vmap(fn)

JAXBackend (_jax_backend.py):
  .jit(fn) -> jax.jit(fn)
  .grad(fn) -> jax.grad(fn)
  .array(x) -> jnp.array(x)
  .to_numpy(x) -> np.array(x)   [device->host copy]

NumPyBackend (_numpy_backend.py):
  .jit(fn) -> fn   (identity)
  .grad(fn) -> NotImplementedError
  .array(x) -> np.array(x)

Conversion utilities (_conversions.py):
  ensure_numpy(arr):
    - np.ndarray (read-only) -> copy
    - jnp.ndarray -> np.array(arr)
    - hasattr(__array__) -> np.asarray
    - fallback -> np.array(arr)

  is_jax_array(arr): checks type against jax.Array
  ensure_backend_array(arr): inverse of ensure_numpy

I/O Adapters (io_adapter.py):
  PyQtGraphAdapter.to_pyqtgraph(arr) -> ensure_numpy(arr) [+ stats]
  HDF5Adapter.to_hdf5(arr) -> np.asarray(arr)
  MatplotlibAdapter.to_matplotlib(arr) -> ensure_numpy(arr)
```

---

## 11. Summary of Key Architectural Risks

| Area | Risk | Notes |
|---|---|---|
| Thread safety | active_plot_operations dict mutated from main thread only; signal lambdas capture op_id | Verify Qt::QueuedConnection used |
| HDF5 fork safety | h5py in worker threads (not forked processes) -- safe with thread pool | Separate file handles per worker |
| JAX+NumPy coexistence | ensure_numpy() must be called at every JAX->PyQtGraph/HDF5 boundary | BUG-027 pattern |
| validate=False bypass | HDF5Facade(validate=False) returns raw dict -- callers must handle both dict and QMapSchema | Typing inconsistency |
| Bayesian flag reset | _g2_bayesian_worker_active flag reset only in _on_g2_bayesian_finished -- error path may skip reset | Verify error handler resets flag |
| Worker signal cleanup | _submit_plot_worker stores lambda slots for cleanup -- stale entries if op never completes | Check _disconnect_signals() completeness |
| NLSQResult.converged | getattr(native_result, "success", False) defaults False -- may mask silent NLSQ failures | BUG-017 noted |
