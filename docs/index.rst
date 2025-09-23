XPCS Toolkit Documentation
===========================

Python tool for X-ray Photon Correlation Spectroscopy (XPCS) data analysis.

Quick Start
-----------

.. code-block:: bash

   # Install
   pip install xpcs-toolkit

   # Launch GUI
   xpcs-toolkit path/to/hdf/directory

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/index
   contributing

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   readme
   authors
   history

Features
--------

* G2 correlation analysis with fitting
* SAXS 1D/2D visualization
* Two-time correlation analysis
* HDF5 data support (NeXus format)
* PySide6 GUI interface

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

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
