GUI Components
==============

Interactive XPCS data visualization interface.

.. note::
   For complete API documentation of all GUI modules, see :doc:`../xpcs_toolkit`.

.. currentmodule:: xpcs_toolkit

Main Application
----------------

The main GUI application window built with PySide6. Provides tab-based
interface for different analysis modes (SAXS 2D/1D, G2, stability, two-time).

See :mod:`xpcs_toolkit.xpcs_viewer` for complete API documentation.

.. note::
   The GUI components have limited automated testing due to their interactive
   nature. Manual testing and user feedback are primary validation methods.

Viewer Kernel
-------------

Backend kernel that bridges GUI and data processing operations.
Manages file collections, averaging operations, and plot state.

See :mod:`xpcs_toolkit.viewer_kernel` for complete API documentation.

File Locator
------------

File discovery and management utilities for XPCS datasets.
Handles file system navigation and dataset validation.

See :mod:`xpcs_toolkit.file_locator` for complete API documentation.

Command Line Interface
----------------------

Command-line entry points for launching the GUI application.
Supports various startup configurations and directory specifications.

See :mod:`xpcs_toolkit.cli` for complete API documentation.
