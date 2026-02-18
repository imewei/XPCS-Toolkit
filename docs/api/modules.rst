Analysis Modules
================

Specialized modules for XPCS analysis, plotting, and data processing.

.. currentmodule:: xpcsviewer.module

.. note::

   These modules depend on PyQtGraph for interactive plotting. API documentation
   is available via source code links (click ``[source]`` on any documented function).

G2 Correlation Analysis
-----------------------

Multi-tau correlation analysis with single and double exponential fitting.

See :mod:`xpcsviewer.module.g2mod` for complete API documentation.

Key functions:

- :func:`~g2mod.get_data` -- Extract G2 data from multiple XpcsFile objects
- :func:`~g2mod.pg_plot` -- Interactive PyQtGraph plot of G2 correlation functions
- :func:`~g2mod.pg_plot_one_g2` -- Plot a single G2 curve with fit overlay
- :func:`~g2mod.compute_geometry` -- Compute Q-dependent geometry parameters
- :func:`~g2mod.get_g2_stability_data` -- G2 stability analysis across time sections
- :func:`~g2mod.vectorized_g2_baseline_correction` -- Vectorized baseline correction
- :func:`~g2mod.batch_g2_normalization` -- Batch normalization of G2 data

SAXS Analysis
-------------

Small-angle scattering analysis in both 1D and 2D formats.

**SAXS 1D** -- Radial averaging, intensity profiles, and line plotting:

See :mod:`xpcsviewer.module.saxs1d` for complete API documentation.

- :func:`~saxs1d.pg_plot` -- Interactive 1D SAXS profile plot
- :func:`~saxs1d.offset_intensity` -- Apply vertical offset to intensity curves
- :func:`~saxs1d.vectorized_q_binning` -- Vectorized Q-binning operations
- :func:`~saxs1d.batch_saxs_analysis` -- Batch SAXS analysis pipeline

**SAXS 2D** -- 2D scattering pattern visualization:

See :mod:`xpcsviewer.module.saxs2d` for complete API documentation.

- :func:`~saxs2d.plot` -- Display 2D scattering pattern with colormap

Two-Time Correlation
--------------------

Two-time correlation map visualization and analysis for studying temporal
dynamics beyond traditional multi-tau analysis.

See :mod:`xpcsviewer.module.twotime` for complete API documentation.

- :func:`~twotime.plot_twotime` -- Interactive two-time correlation map
- :func:`~twotime.plot_twotime_g2` -- G2 from two-time correlation diagonal cuts
- :func:`~twotime.clean_c2_for_visualization` -- Clean C2 matrix for display
- :func:`~twotime.calculate_safe_levels` -- Calculate safe color scale levels

**Two-Time Utilities** -- C2 matrix I/O and processing:

See :mod:`xpcsviewer.module.twotime_utils` for complete API documentation.

- :func:`~twotime_utils.get_all_c2_from_hdf` -- Read all C2 matrices from HDF5
- :func:`~twotime_utils.get_single_c2_from_hdf` -- Read single C2 matrix
- :func:`~twotime_utils.get_c2_g2partials_from_hdf` -- Extract G2 partials from C2
- :func:`~twotime_utils.correct_diagonal_c2_vectorized` -- Vectorized diagonal correction

Stability Analysis
------------------

Sample stability monitoring by comparing SAXS-1D profiles across time sections.

See :mod:`xpcsviewer.module.stability` for complete API documentation.

- :func:`~stability.plot` -- Plot stability analysis across time sections

Intensity vs Time
-----------------

Time series analysis of intensity fluctuations with FFT spectrum and
interactive zoom panels.

See :mod:`xpcsviewer.module.intt` for complete API documentation.

- :func:`~intt.plot` -- Interactive intensity vs time plot with FFT
- :func:`~intt.smooth_data` -- Smooth intensity time series

Tau-Q Analysis
--------------

Relaxation time vs Q-value analysis for diffusion characterization.

See :mod:`xpcsviewer.module.tauq` for complete API documentation.

- :func:`~tauq.plot` -- Plot tau vs Q with power-law fits
- :func:`~tauq.plot_pre` -- Pre-analysis tau-Q plot

File Averaging
--------------

Parallel processing framework for averaging multiple XPCS datasets.

See :mod:`xpcsviewer.module.average_toolbox` for complete API documentation.

- :func:`~average_toolbox.do_average` -- Average multiple datasets with error propagation
- :class:`~average_toolbox.AverageToolbox` -- Interactive averaging widget
