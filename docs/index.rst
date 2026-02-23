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
     - G2 correlation (g2, g2 Fit, g2 Map), SAXS 1D/2D, two-time correlation, stability, diffusion
   * - **Fitting**
     - NLSQ 0.6.0 point estimates + NumPyro NUTS Bayesian inference with ArviZ diagnostics
   * - **Backends**
     - NumPy (default) and JAX (GPU acceleration, JIT, gradients)
   * - **Mask Editor**
     - Interactive mask creation, Q-map generation, and Q-phi partitioning
   * - **Data Format**
     - HDF5 (NeXus convention) with schema-validated I/O
   * - **GUI**
     - PySide6 with light/dark themes, SVG icons, category tab bar, command palette, session persistence

Key Features
------------

**Correlation Analysis**
   G2 autocorrelation across dedicated tabs (g2 view, g2 Fit, g2 Map), two-time methods,
   SAXS 1D/2D visualization, sample stability monitoring, and diffusion coefficient extraction

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
   pip install xpcsviewer-gui

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

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: How-To Guides

   how-to/index

.. toctree::
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/index

.. toctree::
   :hidden:
   :caption: Architecture

   architecture/index

.. toctree::
   :hidden:
   :caption: Operations

   operations/index

.. toctree::
   :hidden:
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

   1. **SAXS 2D -- Integrated Scattering Pattern**

      .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/saxs2d.png
         :alt: 2D SAXS pattern visualization

   2. **SAXS 1D -- Radial Reduction and Analysis**

      .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/saxs1d.png
         :alt: Radially averaged 1D SAXS data

   3. **Stability -- Sample Stability Assessment**

      .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/stability.png
         :alt: Temporal stability analysis across time sections

   4. **I(t) -- Intensity vs Time Series**

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

   10. **Q-Map -- Q-Space Mapping**

       .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/qmap.png
          :alt: Q-space mapping and partitioning

   11. **Average -- File Averaging Toolbox**

       .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/average.png
          :alt: Advanced file averaging capabilities

   12. **Metadata -- HDF5 Explorer**

       .. image:: https://raw.githubusercontent.com/imewei/XPCSViewer/master/docs/images/hdf_info.png
          :alt: File structure and metadata viewer

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
