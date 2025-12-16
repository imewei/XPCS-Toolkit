=======
History
=======

1.1.0 (2025-12-16)
------------------

**GUI Modernization Release**

* Added light/dark theme support with automatic system detection
* Implemented session persistence - resume work where you left off
* Added command palette (Ctrl+Shift+P) for quick access to all commands
* Added toast notifications for non-intrusive status updates
* Implemented keyboard shortcut manager with customizable keybindings
* Added drag-and-drop file handling with visual feedback
* Integrated theme support for PyQtGraph and Matplotlib plot backends
* Added preferences management with type-safe access
* Added recent files tracking with validation
* Updated all dependencies to latest versions
* Comprehensive test coverage for GUI modules
* Updated documentation with GUI modernization features

1.0.9 (2025-10-04)
------------------

* Fixed critical memory leak in twotime CLI batch processing
* Added context manager support to XpcsFile for automatic resource cleanup
* Implemented explicit cleanup in batch processing with periodic garbage collection
* Resolved pip-audit security job failures with documented vulnerability suppression
* Enhanced memory management documentation and best practices
* Updated security scanning configuration (pip-audit, authlib upgrade)
* Improved CI/CD security job reliability

1.0.5 (2025-09-23)
------------------

* CI/CD pipeline stabilization and cross-platform compatibility
* Windows CI test failures resolved with path normalization
* Qt threading issues fixed - eliminated fatal Python errors
* GitHub Actions workflow comprehensive fixes
* Security scanning tools updated with current syntax
* Performance test robustness with CI-aware thresholds
* Enhanced HDF5 connection pooling and memory management
* Test framework improvements with isolation and timeout protection
* Scientific computing accuracy maintained across optimizations
* Enhanced error handling and validation framework
* Security analysis integration with bandit
* Module consolidation and architecture improvements

1.0.2+ (2025-09-22)
-------------------

* Sphinx documentation with API reference
* Qt validation testing framework
* Performance optimizations (25-40% improvement)
* Development tools (ruff, pre-commit)
* PySide6 integration
* Memory management and caching
* Error handling improvements

1.0.2 (2025-09-13)
------------------

* PyQtGraph PySide6 compatibility fixes
* Test validation framework
* ROI boundary validation corrections
* Performance benchmark suite

1.0.1 (2025-08-14)
------------------

* Threading system optimizations
* Memory management improvements
* HDF5 I/O performance enhancements
* Scientific algorithm optimizations

1.0.0 (2025-06-01)
------------------

* Python 3.12+ support
* PySide6 GUI
* Enhanced analysis capabilities

0.3.0 (2025-03-02)
------------------

* First PyPI release
* Basic XPCS analysis
* Initial GUI
