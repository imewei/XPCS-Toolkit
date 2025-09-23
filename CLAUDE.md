# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XPCS Toolkit is an interactive Python-based visualization tool for X-ray Photon Correlation Spectroscopy (XPCS) datasets. It provides a PySide6 GUI for analyzing XPCS data stored in customized NeXus HDF5 format from APS-8IDI beamline, supporting both multi-tau and two-time correlation analysis.

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n xpcs-toolkit python==3.10.16
conda activate xpcs-toolkit

# Install in development mode with all dependencies
pip install -e .[all]

# Alternative: Install specific dependency groups
pip install -e .[dev]        # Development tools only
pip install -e .[docs]       # Documentation building tools
pip install -e .[validation] # Profiling and validation tools
pip install -e .[performance] # Performance analysis tools
```

### Building and Testing
```bash
# Run tests
python -m pytest tests/ -v
make test

# Run a single test
python -m pytest tests/unit/test_package_basics.py::test_package_version -v

# Run tests with coverage
make coverage

# Run tests on all Python versions
make test-all

# Test suite profiles (see docs/TESTING_STRATEGY.md for details)
# Run core tests only (fast feedback)
python -m pytest -m "unit or integration"

# Run everything except stress tests (recommended for development)
python -m pytest -m "not stress"

# Run CI-safe tests (excludes flaky and system-dependent tests)
python -m pytest -m "not (stress or system_dependent or flaky)"

# Lint code
make lint

# Clean build artifacts
make clean
```

### Running the Application
```bash
# Launch GUI from HDF directory
xpcs-toolkit path_to_hdf_directory

# Launch GUI from current directory
xpcs-toolkit

# Alternative commands (legacy)
pyxpcsviewer
run_viewer

# Suppress Qt connection warnings (optional)
PYXPCS_SUPPRESS_QT_WARNINGS=1 xpcs-toolkit
```

### Documentation
```bash
# Build docs
make docs

# Serve docs with auto-reload
make servedocs
```

### Packaging
```bash
# Build distribution packages
make dist

# Release to PyPI
make release
```

## Architecture Overview

### Core Components

**GUI Layer (`xpcs_viewer.py`)**
- Main application window built with PySide6
- Tab-based interface for different analysis modes (SAXS 2D/1D, G2, stability, two-time)
- Integrates PyQtGraph for real-time plotting and matplotlib for complex visualizations
- Thread pool management for non-blocking operations

**Data Management (`xpcs_file.py`)**
- `XpcsFile` class: Primary data container for XPCS datasets
- Lazy loading of large arrays (saxs_2d, saxs_2d_log) to manage memory
- Built-in fitting capabilities for G2 analysis (single/double exponential)
- ROI (Region of Interest) data extraction for custom analysis regions

**Backend Kernel (`viewer_kernel.py`)**
- `ViewerKernel` class: Bridges GUI and data processing
- Manages file collections, averaging operations, and plot state
- Coordinates between different analysis modules
- Handles background processing for compute-intensive tasks

**File I/O System (`fileIO/`)**
- `hdf_reader.py`: HDF5 file operations with connection pooling optimization
- `qmap_utils.py`: Q-space mapping and detector geometry calculations
- `aps_8idi.py`: Beamline-specific data structure definitions
- Supports both "nexus" and legacy data formats

**Analysis Modules (`module/`)**
- `g2mod.py`: Multi-tau correlation analysis and fitting
- `saxs1d.py` & `saxs2d.py`: Small-angle scattering analysis
- `twotime.py` & `twotime_utils.py`: Two-time correlation with multiprocessing
- `stability.py`: Sample stability analysis over time
- `intt.py`: Intensity vs. time analysis
- `average_toolbox.py`: File averaging with parallel processing

### Performance Optimizations

The codebase includes recent performance optimizations:

**HDF5 I/O Optimization**
- Connection pooling in `hdf_reader.py` reduces file open/close overhead
- Batch reading operations minimize I/O calls
- Cached metadata reads for repeated access

**Memory Management**
- LRU caching system for frequently accessed data
- Memory pressure detection and automatic cleanup
- Optimized array operations with minimal copying

**Concurrency**
- Async worker framework for GUI responsiveness
- Multiprocessing for CPU-intensive operations (two-time correlation, averaging)
- Background threading for plot generation and data loading

**Vectorization**
- NumPy-optimized algorithms throughout analysis modules
- Eliminated nested loops in favor of broadcasted operations
- Optimized fitting routines with parallel processing

### Data Flow Architecture

1. **File Loading**: `FileLocator` → `XpcsFile` → lazy data loading
2. **GUI Interaction**: `XpcsViewer` → `ViewerKernel` → analysis modules
3. **Plotting**: Analysis modules → PyQtGraph/matplotlib handlers → GUI
4. **Processing**: Background workers → progress signals → GUI updates

### Key Design Patterns

**Observer Pattern**: GUI components listen to kernel state changes via Qt signals
**Strategy Pattern**: Different file formats handled through pluggable readers
**Factory Pattern**: Plot handlers created based on visualization requirements
**Command Pattern**: Analysis operations encapsulated as worker tasks

### Testing Approach

- Unit tests in `tests/` directory using Python unittest framework
- Property-based testing with Hypothesis for mathematical invariants
- Performance benchmarks for optimization validation
- Comprehensive test validation framework
- Integration tests for GUI workflows (limited due to GUI complexity)
- Test data should use synthetic or minimal real XPCS datasets

For detailed testing guidance, see `docs/TESTING.md`.

### Critical Dependencies

- **PySide6**: GUI framework, specific version compatibility important
- **PyQtGraph**: Real-time plotting, custom extensions for XPCS workflows
- **h5py**: HDF5 file access, must support parallel reading
- **NumPy/SciPy**: Numerical computing, vectorized operations critical for performance
- **joblib**: Caching and parallel processing for fitting operations

### Development Notes

**Memory Considerations**: XPCS datasets can be several GB; always consider memory usage in new features
**GUI Threading**: Heavy computations must run in background threads to maintain responsiveness
**Scientific Accuracy**: All optimizations must preserve numerical precision for scientific validity
**Beamline Compatibility**: Changes to data format support require coordination with APS-8IDI beamline scientists
**Qt Warnings**: Some Qt connection warnings from PyQtGraph are expected and cosmetic; they don't affect functionality
**Logging System**: Comprehensive logging is integrated throughout the codebase - see `docs/LOGGING_SYSTEM.md` for usage guidelines
