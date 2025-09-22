XPCS Toolkit Documentation
===========================

Interactive Python-based visualization tool for X-ray Photon Correlation
Spectroscopy (XPCS) datasets from APS-8IDI beamline.

Quick Start
-----------

.. code-block:: python

   from xpcs_toolkit import XpcsFile

   # Load XPCS dataset
   xf = XpcsFile('path/to/data.hdf')

   # Launch GUI
   from xpcs_toolkit.xpcs_viewer import main
   main()

Documentation Sections
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   modules

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/index
   contributing

.. toctree::
   :maxdepth: 1
   :caption: Project Information

   readme
   authors
   history

Features
--------

- **Data Analysis**: G2 correlation, SAXS analysis, two-time correlation
- **Interactive GUI**: Real-time plotting with PyQtGraph and matplotlib
- **Performance**: Optimized for large datasets with threading and caching
- **File Formats**: Native support for NeXus HDF5 from APS-8IDI
- **Cross-platform**: Windows, macOS, and Linux support

Analysis Types
--------------

- **Multi-tau G2 Analysis**: Time correlation function analysis
- **SAXS 1D/2D**: Small-angle X-ray scattering visualization
- **Two-time Correlation**: Advanced correlation analysis
- **Stability Analysis**: Sample stability over time
- **Intensity vs Time**: Time-series intensity analysis

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
