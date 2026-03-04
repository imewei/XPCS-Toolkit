Analysis Modules
================

Specialized modules for XPCS analysis, plotting, and data processing.

.. currentmodule:: xpcsviewer.module

.. note::

   These modules depend on PyQtGraph for interactive plotting. Some functions
   accept PyQtGraph plot items as arguments; those parameters will appear
   as mock types in the rendered documentation.

G2 Correlation Analysis
-----------------------

Multi-tau correlation analysis with single and double exponential fitting.

.. automodule:: xpcsviewer.module.g2mod
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

SAXS 1D Analysis
----------------

Radial averaging, intensity profiles, and line plotting.

.. automodule:: xpcsviewer.module.saxs1d
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

SAXS 2D Visualization
---------------------

2D scattering pattern visualization.

.. automodule:: xpcsviewer.module.saxs2d
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Two-Time Correlation
--------------------

Two-time correlation map visualization and analysis for studying temporal
dynamics beyond traditional multi-tau analysis.

.. automodule:: xpcsviewer.module.twotime
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Two-Time Utilities
~~~~~~~~~~~~~~~~~~

C2 matrix I/O and processing utilities.

.. automodule:: xpcsviewer.module.twotime_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Stability Analysis
------------------

Sample stability monitoring by comparing SAXS-1D profiles across time sections.

.. automodule:: xpcsviewer.module.stability
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Intensity vs Time
-----------------

Time series analysis of intensity fluctuations with FFT spectrum and
interactive zoom panels.

.. automodule:: xpcsviewer.module.intt
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Tau-Q Analysis
--------------

Relaxation time vs Q-value analysis for diffusion characterization.

.. automodule:: xpcsviewer.module.tauq
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

File Averaging
--------------

Parallel processing framework for averaging multiple XPCS datasets.

See :mod:`xpcsviewer.module.average_toolbox` for complete API documentation.

.. note::

   This module contains Qt-dependent classes (``AverageToolbox``, ``WorkerSignal``)
   that cannot be rendered by autodoc. Refer to the source code for API details.
