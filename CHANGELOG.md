# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Batch all-Q Bayesian fitting with GUI integration
  - `fitting/bayesian_assembly.py` coordinator for per-Q NUTS sampling
  - Batch fitting dialog in G2 Fit tab with progress tracking
  - `assemble_fit_summary()` converts Bayesian posteriors to legacy format
- NLSQ 0.6.0 adoption with full feature delegation
  - Statistical metrics: R², adjusted R², RMSE, MAE, AIC, BIC
  - Confidence intervals (95% level) and prediction intervals
  - `workflow` parameter replacing presets ('auto', 'auto_global', 'hpc')
  - `stability`, `fallback`, `compute_diagnostics`, `show_progress` parameters
  - NLSQResult property delegation to native CurveFitResult
- SVG icon system with theme-aware `currentColor` replacement and caching
- Category tab bar with separator lines between tab groups
- 12-tab layout: G2 View, G2 Fit, G2 Map as separate tabs
- Bayesian inference worker with NUTS diagnosis dialog
- ROI handles and Q-map colormap selector in SimpleMask
- Comprehensive structured logging (rate limiting, JSON aggregation, `@log_timing`)
- Type-safe schema dataclasses for HDF5 data validation
- HDF5 facade with connection pooling and batch reading
- HDF5 connection pool (`max_size=25`, health checks)
- JAX GPU installation targets for system CUDA
- 91+ new tests (Phase 1 test suite update)
- Performance profiling and bottleneck analysis test suite

### Changed

- G2 tab split into three tabs (G2 View, G2 Fit, G2 Map)
- NLSQ `preset` parameter renamed to `workflow`
- Fitting backend migrated from scipy to NLSQ 0.6.0 library
- Exception-based validation replaces tuple-return pattern (`XPCSValidationError`)
- Lambda slots replaced with named functions in async kernel
- `scripts/` reorganized into `build/`, `screenshots/`, `validation/` subdirectories
- `examples/` directory removed (content covered by notebooks)
- `validation/` directory consolidated into `scripts/validation/`

### Fixed

- NLSQResult delegation: handle `confidence_intervals` as method or property
- NLSQResult delegation: handle `prediction_interval` returning `(n, 2)` array
- JAX `TracerArrayConversionError` in model functions (use `jnp.exp` not `np.exp`)
- MCMC warm-start: use `init_to_value` instead of partial `init_params`
- Power-law fitting: correct `tau_err` and `bounds` API
- Legacy fitter: use initial guess for fixed parameters
- HDF5 fork-safety and float64 type coercions
- Qt main thread safety and resource leak prevention
- 38 P0/P1/P2 GUI issues from deep RCA analysis
- Batch coordinator initialization order (before `load_path`)
- Type safety improvements across widgets and threading modules
- Concurrency deadlocks and signal safety in threading layer

### Performance

- G2, SAXS 1D, and twotime hot paths optimized (reduced array copies)
- Mask handling and file I/O: fewer intermediate array allocations
- `@jax.jit` applied to NLSQ model functions
- `vmap`/`scan`/`fori_loop` added to JAX backend
- HDF5 connection pooling and batch reading

### Documentation

- Tutorial notebooks rewritten as self-contained with synthetic data
- Gallery screenshots refreshed for 12-tab layout
- API docs added for icons, category tab bar, Bayesian worker, exceptions
- Fitting module Sphinx docs with NLSQResult metric docstrings
- Architecture docs: dependency analysis, integration catalog

## [0.1.2] - 2026-01-06

### Added

- SimpleMask module for interactive mask editing and Q-map generation
  - Drawing tools: Rectangle, Circle, Polygon, Line, Ellipse, Eraser
  - Q-map computation from detector geometry
  - Partition (Q-binning) utilities with dynamic/static partition support
  - Undo/redo history for mask operations
  - Export masks and partitions to XPCS Viewer via signals

### Documentation

- Comprehensive SimpleMask documentation added to Sphinx
- AST-based documentation analysis and updates

## [0.1.1] - 2026-01-06

### Added

- G2 Map tab with dynamic UI creation for visualizing g2 values across Q-space
- G2 stability visualization ported from upstream
- XPCS Viewer logo with Qt icon resources

### Changed

- Window title changed from "XPCS Toolkit" to "XPCS Viewer"
- G2 Map tab positioned after G2 tab for logical workflow
- Python target version updated to 3.13
- Session manager updated to support 11 tabs
- Sphinx documentation theme switched to Furo

### Documentation

- RST documentation files converted to Markdown
- Logo added to Sphinx header

### Fixed

- Session manager properly handles increased tab count
- Qt error regression test fixture added
- Method renamed from `plot_g2map` to `plot_g2_map` for consistency
- Bandit false positive B608 suppressed with nosec comment
- Ruff linting issues resolved across codebase
- CI breakages fixed: UI generation and missing assets repaired
- Invalid file scanning paths guarded
- Performance test stabilized

## [0.1.0] - 2026-01-05

### Initial Release - xpcsviewer package

This is the first release under the new package name `xpcsviewer`.

### Features

- G2 correlation analysis with single/double exponential fitting
- SAXS 1D/2D visualization and analysis
- Two-time correlation analysis with batch processing
- HDF5/NeXus data support (APS-8IDI beamline format)
- Sample stability monitoring
- File averaging with parallel processing

### GUI Modernization

- Light/dark theme support with automatic system detection
- Session persistence - resume work where you left off
- Command palette (Ctrl+Shift+P) for quick access to all commands
- Toast notifications for non-intrusive status updates
- Keyboard shortcut manager with customizable keybindings
- Drag-and-drop file handling with visual feedback
- Theme-aware PyQtGraph and Matplotlib plot backends

### Technical

- Python 3.12+ with PySide6 GUI framework
- Centralized constants module for configuration
- Comprehensive test coverage
- CI/CD with GitHub Actions
- Sphinx documentation
