API Reference
=============

This section contains the complete API reference for XPCS Toolkit.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   xpcs_toolkit
   fileio
   modules
   plotting
   threading
   utils
   gui

Overview
--------

The XPCS Toolkit provides a comprehensive Python API for X-ray Photon Correlation Spectroscopy data analysis. The main components are:

- **Core Data Classes**: :class:`~xpcs_toolkit.XpcsFile` for data container and analysis
- **Analysis Modules**: G2 correlation, SAXS analysis, two-time correlation
- **File I/O**: HDF5 data reading and Q-space mapping utilities
- **GUI Components**: PySide6-based interactive visualization
- **Utilities**: Threading, logging, memory management, and optimization tools

Quick Start
-----------

Basic usage example::

    from xpcs_toolkit import XpcsFile

    # Load XPCS data file
    xf = XpcsFile('path/to/data.hdf')

    # Access data attributes
    print(f"Analysis type: {xf.atype}")
    print(f"Data shape: {xf.saxs_2d.shape}")

    # Perform G2 analysis
    from xpcs_toolkit.module import g2mod
    g2_data = g2mod.get_data([xf])
