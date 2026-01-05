============
XPCS Viewer
============

Python-based XPCS data analysis and visualization tool.

.. image:: https://github.com/imewei/XPCSViewer/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/imewei/XPCSViewer/actions/workflows/ci.yml
   :alt: CI Status

.. image:: https://img.shields.io/badge/python-3.12%2B-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

**Features:**

* G2 correlation analysis with fitting
* SAXS 1D/2D visualization
* Two-time correlation analysis
* HDF5 data support (NeXus format)

**GUI Features:**

* Light/dark theme support with system detection
* Session persistence (resume where you left off)
* Command palette (Ctrl+Shift+P) for quick access
* Toast notifications for status updates
* Keyboard shortcut management
* Drag-and-drop file handling
* Theme-aware plots (PyQtGraph & Matplotlib)

UI notes
--------

* Menu-driven header (no quick-access toolbar); all actions live under the menus/shortcuts.
* Starts maximized with a rectangular layout and a minimum-size floor to prevent cramped controls.
* PySide6 GUI interface with modern theming
* Performance optimizations

Installation
------------

**Requirements:** Python 3.12+

.. code-block:: bash

   # Basic installation
   pip install xpcsviewer

   # Complete installation with all features and tools
   pip install xpcsviewer[all]

   # Install with specific optional dependencies
   pip install xpcsviewer[dev]        # Development tools
   pip install xpcsviewer[docs]       # Documentation building
   pip install xpcsviewer[validation] # Profiling and validation tools
   pip install xpcsviewer[performance] # Performance analysis tools

Usage
-----

**GUI (Interactive):**

.. code-block:: bash

   # Launch GUI with data path
   xpcsviewer-gui /path/to/hdf/data

   # Launch from current directory
   xpcsviewer-gui

   # With debug logging
   xpcsviewer-gui --log-level DEBUG

**CLI (Batch Processing):**

.. code-block:: bash

   # Show available commands
   xpcsviewer --help

   # Generate twotime plots for all phi angles at q=0.05
   xpcsviewer twotime --input /data --output /results --q 0.05

   # Generate high-resolution PDF plots
   xpcsviewer twotime -i /data -o /results --phi 45 --dpi 300 --format pdf

Citation
--------

Chu et al., *"pyXPCSviewer: an open-source interactive tool for X-ray photon correlation spectroscopy visualization and analysis"*, Journal of Synchrotron Radiation, (2022) 29, 1122–1129.

Development
-----------

.. code-block:: bash

   # Clone and install
   git clone https://github.com/imewei/XPCSViewer.git
   cd XPCSViewer
   pip install -e .[dev]

   # Run tests
   make test

   # Build docs
   make docs

Data Formats
------------

* NeXus HDF5 (APS-8IDI beamline)
* SAXS 2D/1D data
* G2 correlation functions
* Time series data

Testing
-------

.. code-block:: bash

   make test              # Run tests
   make test-unit         # Unit tests
   make test-integration  # Integration tests
   make coverage          # Coverage report

Documentation
-------------

- `API Reference <https://github.com/imewei/xpcsviewer/tree/master/docs>`_
- `User Guide <https://github.com/imewei/xpcsviewer/blob/master/docs/usage.rst>`_
- `Quick Start <https://github.com/imewei/xpcsviewer/blob/master/docs/user_guide/quickstart.rst>`_

.. code-block:: bash

   make docs              # Build docs
   make docs-autobuild    # Live reload docs

Configuration
-------------

Environment variables for customization:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Variable
     - Description
     - Default
   * - ``XPCS_LOG_LEVEL``
     - Logging verbosity (DEBUG, INFO, WARNING, ERROR)
     - INFO
   * - ``XPCS_CACHE_SIZE_MB``
     - Maximum cache size in MB
     - 512
   * - ``XPCS_THEME``
     - UI theme (light, dark, system)
     - system

Project Structure
-----------------

.. code-block::

   xpcsviewer/
   ├── module/            # Analysis modules
   ├── fileIO/            # HDF5 I/O
   ├── gui/               # GUI modernization
   │   ├── theme/         # Light/dark theming
   │   ├── state/         # Session & preferences
   │   ├── shortcuts/     # Keyboard shortcuts
   │   └── widgets/       # Modern UI widgets
   ├── plothandler/       # Theme-aware plotting
   ├── threading/         # Async workers
   ├── utils/             # Utilities
   └── xpcs_file.py       # Core data class

Analysis Features
-----------------

* Multi-tau G2 correlation with fitting
* Two-time correlation analysis
* SAXS 2D pattern visualization
* SAXS 1D radial averaging
* Sample stability monitoring
* File averaging tools

Gallery
-------

**Analysis Modules Showcase**

1. **Integrated 2D Scattering Pattern**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/saxs2d.png
      :alt: 2D SAXS pattern visualization

2. **1D SAXS Reduction and Analysis**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/saxs1d.png
      :alt: Radially averaged 1D SAXS data

3. **Sample Stability Assessment**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/stability.png
      :alt: Temporal stability analysis across 10 time sections

4. **Intensity vs Time Series**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/intt.png
      :alt: Intensity fluctuation monitoring

5. **File Averaging Toolbox**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/average.png
      :alt: Advanced file averaging capabilities

6. **G2 Correlation Analysis**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/g2mod.png
      :alt: Multi-tau correlation function fitting

7. **Diffusion Characterization**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/diffusion.png
      :alt: τ vs q analysis for diffusion coefficients

8. **Two-time Correlation Maps**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/twotime.png
      :alt: Interactive two-time correlation analysis

9. **HDF5 Metadata Explorer**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/hdf_info.png
      :alt: File structure and metadata viewer

License
-------

MIT License. See `CONTRIBUTING.rst <CONTRIBUTING.rst>`_ for development guidelines.
