Analysis Modules
================

Specialized modules for XPCS analysis and data processing.

.. note::
   For complete API documentation of all analysis modules, see :doc:`../xpcs_toolkit.module`.

.. currentmodule:: xpcs_toolkit.module

G2 Correlation Analysis
-----------------------

Multi-tau correlation analysis with single and double exponential fitting.
Core functionality for time correlation analysis in XPCS.

See :mod:`xpcs_toolkit.module.g2mod` for complete API documentation.

SAXS Analysis
-------------

Small-angle scattering analysis in both 1D and 2D formats.

- **SAXS 1D**: Radial averaging and intensity profiles. See :mod:`xpcs_toolkit.module.saxs1d`.
- **SAXS 2D**: 2D scattering pattern analysis. See :mod:`xpcs_toolkit.module.saxs2d`.

Two-Time Correlation
--------------------

Advanced two-time correlation analysis with multiprocessing support.
Provides detailed temporal dynamics beyond traditional multi-tau analysis.

See :mod:`xpcs_toolkit.module.twotime` and :mod:`xpcs_toolkit.module.twotime_utils` for complete API documentation.

Stability Analysis
------------------

Sample stability monitoring over time. Tracks intensity fluctuations
and system stability during XPCS measurements.

See :mod:`xpcs_toolkit.module.stability` for complete API documentation.

Intensity vs Time
-----------------

Time series analysis of intensity fluctuations.
Essential for understanding temporal dynamics in XPCS datasets.

See :mod:`xpcs_toolkit.module.intt` for complete API documentation.

File Averaging
--------------

Parallel processing framework for averaging multiple XPCS datasets.
Includes background subtraction and statistical analysis.

See :mod:`xpcs_toolkit.module.average_toolbox` for complete API documentation.
