# Changelog

All notable changes to this project will be documented in this file.

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
