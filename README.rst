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

* G2 correlation analysis across three tabs (g2 view, g2 Fit, g2 Map)
* SAXS 1D/2D visualization
* Two-time correlation analysis
* Diffusion coefficient extraction (tau-Q analysis)
* Sample stability monitoring
* Mask Editor with Q-map and Q-binning
* HDF5 data support (NeXus format)

**GUI Features:**

* Light/dark theme support with system detection
* Scalable SVG icons with theme-aware coloring
* Category tab bar grouping 12 analysis tabs
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

GPU Acceleration (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU-accelerated computations on NVIDIA GPUs with CUDA 12+:

.. code-block:: bash

   # Install JAX CPU backend (default)
   pip install xpcsviewer[jax]

   # For GPU acceleration, install JAX with CUDA support
   # This uses your system's CUDA installation (requires CUDA 12+)
   pip install jax[cuda12-local]

   # Or for CUDA 13+
   pip install jax[cuda13-local]

**Verify GPU detection:**

.. code-block:: bash

   python -c "import jax; print(jax.devices())"
   # Expected output: [CudaDevice(id=0), ...]

**Environment variables for control:**

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Variable
     - Description
     - Default
   * - ``XPCS_USE_JAX``
     - Enable JAX backend (true, false, auto)
     - auto
   * - ``JAX_PLATFORMS``
     - Force CPU/GPU (cpu, cuda)
     - (auto-detected)
   * - ``XPCS_GPU_FALLBACK``
     - Allow CPU fallback if GPU fails
     - true
   * - ``XPCS_GPU_MEMORY_FRACTION``
     - Maximum GPU memory usage (0.0-1.0)
     - 0.9

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

- `Tutorials <docs/tutorials/index.rst>`_ -- Step-by-step learning guides
- `How-To Guides <docs/how-to/index.rst>`_ -- Task-oriented instructions
- `API Reference <docs/api/index.rst>`_ -- Auto-generated from docstrings
- `Architecture <docs/architecture/index.rst>`_ -- Design decisions and diagrams

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
   ├── module/            # Analysis modules (g2, saxs, twotime, ...)
   ├── fileIO/            # HDF5 I/O and Q-map utilities
   ├── simplemask/        # Mask editor & Q-map
   ├── gui/               # GUI modernization
   │   ├── icons.py       # SVG icon loader with theme-aware colors
   │   ├── theme/         # Light/dark theming
   │   ├── state/         # Session & preferences
   │   ├── shortcuts/     # Keyboard shortcuts
   │   └── widgets/       # Modern UI widgets (incl. category tab bar)
   ├── fitting/           # Bayesian fitting (NLSQ 0.6.0)
   ├── plothandler/       # Theme-aware plotting
   ├── threading/         # Async workers
   ├── utils/             # Utilities, validation, exceptions
   └── xpcs_file.py       # Core data class

Analysis Features
-----------------

* Multi-tau G2 correlation with fitting
* Two-time correlation analysis
* SAXS 2D pattern visualization
* SAXS 1D radial averaging
* Sample stability monitoring
* File averaging tools
* Mask editing with drawing tools (Rectangle, Circle, Polygon, Line, Ellipse)
* Q-map generation from detector geometry
* Q-binning (partition) for XPCS analysis

Fitting Module (NLSQ 0.6.0)
---------------------------

Bayesian fitting with NumPyro NUTS sampler and JAX-accelerated NLSQ warm-start:

* Statistical metrics: R², adjusted R², RMSE, MAE, AIC, BIC
* Confidence intervals for parameter uncertainty
* Prediction intervals accounting for observation noise
* Models: single/double/stretched exponential, power law
* Automatic bounds inference and fallback strategies
* Model health diagnostics

.. code-block:: python

   from xpcsviewer.fitting import nlsq_fit
   import jax.numpy as jnp

   def model(x, tau, baseline, contrast):
       return baseline + contrast * jnp.exp(-2 * x / tau)

   result = nlsq_fit(
       model, x_data, y_data, y_errors,
       p0={'tau': 1.0, 'baseline': 1.0, 'contrast': 0.3},
       bounds={'tau': (0.01, 100), 'baseline': (0.9, 1.1), 'contrast': (0.1, 0.5)},
   )
   print(f"R² = {result.r_squared:.4f}")
   print(result.summary())

Gallery
-------

**Analysis Modules Showcase**

1. **SAXS 2D -- Integrated Scattering Pattern**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/saxs2d.png
      :alt: 2D SAXS pattern visualization

2. **SAXS 1D -- Radial Reduction and Analysis**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/saxs1d.png
      :alt: Radially averaged 1D SAXS data

3. **Stability -- Sample Assessment**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/stability.png
      :alt: Temporal stability analysis across time sections

4. **I(t) -- Intensity vs Time**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/intt.png
      :alt: Intensity fluctuation monitoring

5. **g2 -- Correlation Function Viewer**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/g2mod.png
      :alt: Multi-tau correlation function display

6. **g2 Fit -- Correlation Fitting**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/g2_fit.png
      :alt: G2 single/double exponential fitting

7. **g2 Map -- Correlation Map**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/g2_map.png
      :alt: G2 correlation map across Q values

8. **Diffusion -- tau(q) Characterization**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/diffusion.png
      :alt: Diffusion coefficient analysis

9. **Two-Time -- Correlation Maps**

   .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/twotime.png
      :alt: Interactive two-time correlation analysis

10. **Average -- File Averaging Toolbox**

    .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/average.png
       :alt: Advanced file averaging capabilities

11. **Metadata -- HDF5 Explorer**

    .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/hdf_info.png
       :alt: File structure and metadata viewer

License
-------

MIT License. See `CONTRIBUTING.rst <CONTRIBUTING.rst>`_ for development guidelines.
