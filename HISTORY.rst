=======
History
=======

0.1.0 (2026-01-05)
------------------

**Initial Release - xpcsviewer package**

This is the first release under the new package name ``xpcsviewer``.

**Features:**

* G2 correlation analysis with single/double exponential fitting
* SAXS 1D/2D visualization and analysis
* Two-time correlation analysis with batch processing
* HDF5/NeXus data support (APS-8IDI beamline format)
* Sample stability monitoring
* File averaging with parallel processing

**GUI Modernization:**

* Light/dark theme support with automatic system detection
* Session persistence - resume work where you left off
* Command palette (Ctrl+Shift+P) for quick access to all commands
* Toast notifications for non-intrusive status updates
* Keyboard shortcut manager with customizable keybindings
* Drag-and-drop file handling with visual feedback
* Theme-aware PyQtGraph and Matplotlib plot backends

**Technical:**

* Python 3.12+ with PySide6 GUI framework
* Centralized constants module for configuration
* Comprehensive test coverage
* CI/CD with GitHub Actions
* Sphinx documentation
