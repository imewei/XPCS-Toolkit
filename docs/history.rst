History
=======

See the full changelog in `CHANGELOG.md <https://github.com/imewei/xpcsviewer/blob/master/CHANGELOG.md>`_.

Recent Changes
--------------

Version 0.1.7 (2026-05-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Constrain JAX versions and optimize performance benchmarks
- Add codecov configuration to ignore non-code paths

Version 0.1.6 (2026-03-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- CI: bump ``astral-sh/setup-uv`` from v4 to v7 across all GitHub Actions workflows

Version 0.1.3 (2026-02-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- "Plot All Q" button in G2 Fit tab for all-Q Bayesian overlay visualization
- Export button: CSV parameters, PDF/PNG figures, netCDF ArviZ diagnostics
- Sampler config spinboxes (warmup, samples, chains) for batch Bayesian fitting
- ``fitting.viz`` module: ``plot_bayesian_all_q``, ``export_bayesian_csv``, ``export_bayesian_diagnostics``
- Batch all-Q Bayesian fitting with GUI integration (``bayesian_assembly.py``)
- NLSQ 0.6.0 adoption: R², adjusted R², RMSE, MAE, AIC, BIC metrics
- Confidence intervals and prediction intervals for fitted parameters
- SVG icon system with theme-aware ``currentColor`` replacement and caching
- Category tab bar with separator lines between tab groups
- 12-tab layout: G2 View, G2 Fit, G2 Map as separate tabs
- Q-bin spinbox with per-Q navigation in G2 Fit tab
- Compact layouts for G2 Fit, Diffusion, and G2 Map tabs
- Bayesian fit results stored on ``bayesian_fit_summary`` (no longer overwritten by NLSQ)
- G2 Fit tab auto-refreshes after "Fit All Q" batch completes
- Diffusion tab prefers Bayesian fit results when available
- JAX ``TracerArrayConversionError`` fix in NLSQ model functions
- Exception-based validation (``XPCSValidationError``)
- Tutorial notebooks rewritten as self-contained with synthetic data
- Installation docs updated to ``xpcsviewer-gui`` package name
- Dark-theme gallery screenshots
- Performance: hot path optimization for G2, SAXS 1D, twotime modules

Version 0.1.2 (2026-01-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- SimpleMask module for interactive mask editing and Q-map generation
- Drawing tools: Rectangle, Circle, Polygon, Line, Ellipse, Eraser
- Q-map computation from detector geometry
- Partition (Q-binning) utilities

Version 0.1.1 (2026-01-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- G2 Map tab with dynamic UI for Q-space visualization
- XPCS Viewer logo and branding
- Python 3.13 support
- Furo documentation theme

Version 0.1.0 (2026-01-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initial release under the ``xpcsviewer`` package name.

**Features:**

- G2 correlation analysis with single/double exponential fitting
- SAXS 1D/2D visualization and analysis
- Two-time correlation analysis with batch processing
- HDF5/NeXus data support (APS-8IDI beamline format)
- Sample stability monitoring
- File averaging with parallel processing

**GUI Modernization:**

- Light/dark theme support with automatic system detection
- Session persistence
- Command palette (Ctrl+Shift+P)
- Toast notifications
- Keyboard shortcut manager
- Drag-and-drop file handling
- Theme-aware PyQtGraph and Matplotlib backends
