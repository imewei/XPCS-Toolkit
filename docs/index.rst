XPCS Viewer Documentation
===========================

**XPCS Viewer** is a Python toolkit for X-ray Photon Correlation Spectroscopy (XPCS)
data analysis. It provides a PySide6 GUI and Python API for loading, visualizing, and
fitting XPCS correlation functions from HDF5 data files.

----

At a Glance
------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - **Analysis**
     - G2 correlation, SAXS 1D/2D, two-time correlation, stability, diffusion
   * - **Fitting**
     - NLSQ 0.6.0 point estimates + NumPyro NUTS Bayesian inference with ArviZ diagnostics
   * - **Backends**
     - NumPy (default) and JAX (GPU acceleration, JIT, gradients)
   * - **Mask Editor**
     - Interactive mask creation, Q-map generation, and Q-phi partitioning
   * - **Data Format**
     - HDF5 (NeXus convention) with schema-validated I/O
   * - **GUI**
     - PySide6 with light/dark themes, command palette, session persistence

Key Features
------------

**Correlation Analysis**
   G2 autocorrelation with multi-tau and two-time methods, SAXS 1D/2D visualization,
   sample stability monitoring, and diffusion coefficient extraction

**Fitting Pipeline**
   NLSQ warm-start followed by NumPyro NUTS sampling. Model selection via AIC/BIC,
   prediction intervals, and ArviZ convergence diagnostics (R-hat, ESS, BFMI)

**Backend Abstraction**
   Unified NumPy/JAX API with automatic fallback. JIT compilation and GPU acceleration
   for compute-intensive operations. ``ensure_numpy()`` at I/O boundaries

**Interactive Mask Editor**
   Drawing tools (Rectangle, Circle, Polygon, Line, Ellipse, Eraser) with undo/redo
   history, Q-map computation from detector geometry, and Q-phi partition export

Quick Start
-----------

.. code-block:: bash

   # Install
   pip install xpcsviewer

   # Launch GUI
   xpcsviewer-gui /path/to/hdf/data

   # CLI batch processing
   xpcsviewer twotime --input /data --output /results --q 0.05

.. code-block:: python

   from xpcsviewer import XpcsFile

   # Load XPCS data
   with XpcsFile("data.hdf") as xf:
       q, t_el, g2, g2_err, labels = xf.get_g2_data()
       print(f"Q bins: {len(q)}, Delay points: {len(t_el)}")

----

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   how-to/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/index

.. toctree::
   :maxdepth: 1
   :caption: Operations

   operations/index

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   authors
   history

.. toctree::
   :hidden:
   :caption: Legacy Redirects

   user_guide/index
   developer/index

----

Gallery
-------

.. only:: html

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
         :alt: Diffusion coefficient analysis

   8. **Two-time Correlation Maps**

      .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/twotime.png
         :alt: Interactive two-time correlation analysis

   9. **HDF5 Metadata Explorer**

      .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/hdf_info.png
         :alt: File structure and metadata viewer

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
