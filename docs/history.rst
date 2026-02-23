History
=======

See the full changelog in `CHANGELOG.md <https://github.com/imewei/xpcsviewer/blob/master/CHANGELOG.md>`_.

Recent Changes
--------------

Version 0.1.3 (2026-02-22)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Batch all-Q Bayesian fitting with GUI integration (``bayesian_assembly.py``)
- NLSQ 0.6.0 adoption: R², adjusted R², RMSE, MAE, AIC, BIC metrics
- Confidence intervals and prediction intervals for fitted parameters
- SVG icon system with theme-aware ``currentColor`` replacement and caching
- Category tab bar with separator lines between tab groups
- 12-tab layout: G2 View, G2 Fit, G2 Map as separate tabs
- Q-bin spinbox with per-Q navigation in G2 Fit tab
- Compact layouts for G2 Fit, Diffusion, and G2 Map tabs
- JAX ``TracerArrayConversionError`` fix in NLSQ model functions
- Exception-based validation (``XPCSValidationError``)
- Tutorial notebooks rewritten as self-contained with synthetic data
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
