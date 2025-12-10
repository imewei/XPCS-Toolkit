# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XPCS Toolkit is a Python-based GUI application for visualizing and modeling X-ray Photon Correlation Spectroscopy (XPCS) datasets. It uses PySide6/Qt for the GUI, pyqtgraph for visualization, and processes HDF5 data files from synchrotron beamlines (primarily APS 8-ID-I).

## Build & Development Commands

```bash
# Setup development environment (installs package + dev/test/docs extras + pre-commit hooks)
make dev-setup

# Run quick unit tests (default test target)
make test              # or: make test-fast

# Run core tests (unit + integration)
make test-core

# Run a single test file
pytest tests/unit/core/test_xpcs_file.py -v

# Run tests by marker
pytest -m "unit and not slow" -v
pytest -m "scientific" -v
pytest -m "gui" -v                # requires display

# Run with coverage
make coverage

# Lint (ruff, critical errors only)
make lint

# Format code (ruff)
make format

# Build Sphinx docs
make docs-build

# Launch the GUI
make run-app
# or: xpcs-toolkit / pyxpcsviewer / run_viewer
```

## Architecture

### Package Structure

```
xpcs_toolkit/
├── xpcs_file.py         # Core HDF5 data loader (XpcsFile class)
├── viewer_kernel.py     # Backend kernel coordinating analysis modules
├── xpcs_viewer.py       # Main GUI application window
├── viewer_ui.py         # Auto-generated Qt UI (do not edit)
├── file_locator.py      # File discovery and management
├── fileIO/              # HDF5 readers (APS 8-ID-I format, qmap utilities)
├── module/              # Analysis modules:
│   ├── g2mod.py         #   G2 correlation function analysis & fitting
│   ├── twotime.py       #   Two-time correlation analysis
│   ├── saxs1d.py        #   1D SAXS analysis
│   ├── saxs2d.py        #   2D SAXS visualization
│   ├── stability.py     #   Stability analysis
│   └── intt.py          #   Integrated intensity
├── plothandler/         # Matplotlib/pyqtgraph plotting backends
├── threading/           # Async workers for background processing
├── helper/              # Fitting utilities, logging, Qt models
└── utils/               # Memory management, logging config, validation
```

### Key Classes

- **XpcsFile**: Main data container; handles HDF5 reading, caching, lazy loading, and memory management
- **ViewerKernel**: Backend coordinator between GUI and analysis modules; inherits from FileLocator
- **XpcsViewer**: Main Qt window connecting UI events to ViewerKernel operations

### Analysis Flow

1. User selects HDF5 file(s) via FileLocator
2. XpcsFile loads metadata and provides lazy access to datasets
3. ViewerKernel routes requests to appropriate module (g2mod, twotime, etc.)
4. Results displayed via plothandler backends (pyqtgraph for interactive, matplotlib for export)

## Testing

Test markers defined in `pyproject.toml`:
- `unit`, `integration`, `scientific`, `gui`, `performance`, `slow`
- `error_handling`, `threading`, `numerical`, `file_io`

Key test directories:
- `tests/unit/` - Component tests (core, fileio, analysis, threading)
- `tests/scientific/` - Algorithm accuracy and physical constraint validation
- `tests/gui_interactive/` - Qt GUI tests (requires display or `QT_QPA_PLATFORM=offscreen`)

Run GUI tests headless:
```bash
QT_QPA_PLATFORM=offscreen pytest tests/gui_interactive/ -v
```

## Code Style

- Python 3.12+, Ruff enforced (line-length 88)
- Type hints expected; mypy runs in strict mode
- snake_case for functions/variables, PascalCase for classes
- Do not hand-edit `*_ui.py` or `icons_rc.py` (auto-generated)

## Commit Style

Follow Conventional Commits with scopes: `feat:`, `fix:`, `chore:`, `docs:`, e.g., `fix(ci): resolve workflow failure`
