============
XPCS Toolkit
============

Python-based XPCS data analysis and visualization tool.

.. image:: https://img.shields.io/badge/python-3.12%2B-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
   :alt: License

**Features:**

* G2 correlation analysis with fitting
* SAXS 1D/2D visualization
* Two-time correlation analysis
* HDF5 data support (NeXus format)
* PySide6 GUI interface
* Performance optimizations

Installation
------------

**Requirements:** Python 3.12+

.. code-block:: bash

   # Basic installation
   pip install xpcs-toolkit

   # Complete installation with all features and tools
   pip install xpcs-toolkit[all]

   # Install with specific optional dependencies
   pip install xpcs-toolkit[dev]        # Development tools
   pip install xpcs-toolkit[docs]       # Documentation building
   pip install xpcs-toolkit[validation] # Profiling and validation tools
   pip install xpcs-toolkit[performance] # Performance analysis tools

Usage
-----

.. code-block:: bash

   # Launch GUI
   xpcs-toolkit path_to_hdf_directory

   # Launch from current directory
   xpcs-toolkit

Citation
--------

Chu et al., *"pyXPCSviewer: an open-source interactive tool for X-ray photon correlation spectroscopy visualization and analysis"*, Journal of Synchrotron Radiation, (2022) 29, 1122–1129.

Development
-----------

.. code-block:: bash

   # Clone and install
   git clone https://github.com/imewei/XPCS-Toolkit.git
   cd XPCS-Toolkit
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

.. code-block:: bash

   make docs              # Build docs
   make docs-autobuild    # Live reload docs

Project Structure
-----------------

.. code-block::

   xpcs_toolkit/
   ├── module/            # Analysis modules
   ├── fileIO/            # HDF5 I/O
   ├── gui/               # PySide6 GUI
   ├── plothandler/       # Plotting
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

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/saxs2d.png
      :alt: 2D SAXS pattern visualization

2. **1D SAXS Reduction and Analysis**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/saxs1d.png
      :alt: Radially averaged 1D SAXS data

3. **Sample Stability Assessment**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/stability.png
      :alt: Temporal stability analysis across 10 time sections

4. **Intensity vs Time Series**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/intt.png
      :alt: Intensity fluctuation monitoring

5. **File Averaging Toolbox**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/average.png
      :alt: Advanced file averaging capabilities

6. **G2 Correlation Analysis**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/g2mod.png
      :alt: Multi-tau correlation function fitting

7. **Diffusion Characterization**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/diffusion.png
      :alt: τ vs q analysis for diffusion coefficients

8. **Two-time Correlation Maps**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/twotime.png
      :alt: Interactive two-time correlation analysis

9. **HDF5 Metadata Explorer**

   .. image:: https://raw.githubusercontent.com/imewei/XPCS-Toolkit/master/docs/images/hdf_info.png
      :alt: File structure and metadata viewer

License
-------

MIT License. See `CONTRIBUTING.rst <CONTRIBUTING.rst>`_ for development guidelines.
