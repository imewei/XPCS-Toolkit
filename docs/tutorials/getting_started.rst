Getting Started with XPCS Viewer
=================================

This tutorial walks you through loading XPCS data from HDF5 files, inspecting
metadata, and performing a basic G2 correlation analysis.

.. admonition:: What you'll learn

   - Load an XPCS HDF5 result file with :class:`~xpcsviewer.xpcs_file.XpcsFile`
   - Inspect metadata and the HDF5 tree structure
   - Extract G2 correlation data and apply Q/time-range filters
   - Plot G2 autocorrelation curves with Matplotlib
   - Access SAXS 1D data
   - Run a basic single-exponential G2 fit

Prerequisites
-------------

Install xpcsviewer in a virtual environment:

.. code-block:: bash

   uv sync           # If using uv (recommended)
   # or
   pip install xpcsviewer-gui

You will also need an XPCS HDF5 result file produced by a correlator (e.g.,
the APS multi-tau or two-time pipeline). The file follows the NeXus convention.

Step 1 -- Load an HDF5 File
----------------------------

The central object is :class:`~xpcsviewer.xpcs_file.XpcsFile`. It reads
the HDF5 file, loads Q-map information, and attaches all analysis fields
as instance attributes.

.. code-block:: python

   from xpcsviewer.xpcs_file import XpcsFile

   # Open a multi-tau XPCS result file
   xf = XpcsFile("/path/to/A001_Multitau_result.hdf")

   # Display a summary of loaded attributes
   print(xf)
   # Output (example):
   #   File:/path/to/A001_Multitau_result.hdf
   #      fname       : /path/to/A001_Multitau_result.hdf
   #      atype       : Multitau
   #      label       : A001
   #      g2          : (100, 24)
   #      g2_err      : (100, 24)
   #      t_el        : (100,)
   #      ...

The ``atype`` attribute tells you the analysis type (``"Multitau"``,
``"Twotime"``, or both). This is important because it determines which
downstream analysis methods are available -- for example, two-time correlation
maps are only present when ``"Twotime"`` appears in ``atype``.

Step 2 -- Inspect Metadata
---------------------------

Retrieve the full HDF5 tree structure as a dictionary:

.. code-block:: python

   info = xf.get_hdf_info()
   # info is a nested dictionary of HDF5 groups and datasets

   # Explore the analysis type
   print(f"Analysis type: {xf.atype}")

   # The Q-map object provides momentum-transfer binning info
   print(f"Q-map type: {type(xf.qmap)}")

Understanding the HDF5 structure is useful when you need to access data that
is not exposed as a top-level attribute. The ``get_hdf_info()`` method returns
the complete tree of groups and datasets so you can see what is available.

Step 3 -- Extract G2 Data
--------------------------

The :meth:`~xpcsviewer.xpcs_file.XpcsFile.get_g2_data` method extracts
the G2 correlation function, errors, delay times, and Q-bin labels:

.. code-block:: python

   # Extract G2 data for all Q bins
   q_values, t_el, g2, g2_err, labels = xf.get_g2_data()

   print(f"Number of Q bins: {len(q_values)}")
   print(f"Delay times shape: {t_el.shape}")
   print(f"G2 shape: {g2.shape}")         # (n_delay, n_q)
   print(f"G2 error shape: {g2_err.shape}")

You can restrict the range of Q values and delay times. This is useful when
you want to focus on a specific Q region or exclude early/late delay points
that may be noisy:

.. code-block:: python

   # Extract G2 for Q in [0.01, 0.05] nm^-1, t in [0.1, 100] s
   q_values, t_el, g2, g2_err, labels = xf.get_g2_data(
       qrange=(0.01, 0.05),
       trange=(0.1, 100),
   )

Step 4 -- Plot G2 Curves
--------------------------

Visualize the G2 autocorrelation functions using Matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   fig, ax = plt.subplots(figsize=(8, 5))

   for i, label in enumerate(labels):
       ax.errorbar(
           t_el, g2[:, i], yerr=g2_err[:, i],
           fmt="o", markersize=3, label=label,
       )

   ax.set_xscale("log")
   ax.set_xlabel("Delay time (s)")
   ax.set_ylabel(r"$g_2(\tau)$")
   ax.set_title("G2 Autocorrelation")
   ax.legend(fontsize=8, ncol=2)
   plt.tight_layout()
   plt.show()

The logarithmic x-axis is conventional for XPCS data because the delay times
span several orders of magnitude. Each curve corresponds to a different Q bin;
the legend labels show the Q value in reciprocal nanometers.

Step 5 -- Extract SAXS 1D Data
-------------------------------

The :meth:`~xpcsviewer.xpcs_file.XpcsFile.get_saxs1d_data` method returns
the radially averaged scattering intensity:

.. code-block:: python

   q, Iq, xlabel, ylabel = xf.get_saxs1d_data()

   fig, ax = plt.subplots()
   # Iq may have multiple frames; plot the first
   ax.loglog(q, Iq[0], "-")
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   ax.set_title("SAXS 1D Profile")
   plt.tight_layout()
   plt.show()

The SAXS 1D profile gives the azimuthally averaged scattering intensity I(q).
Plotting on a log-log scale reveals the power-law slope, which encodes
structural information about the sample.

Step 6 -- Basic G2 Fitting
----------------------------

Fit a single-exponential decay model to the G2 data:

.. include:: /_includes/g2_fitting_basics.rst

.. code-block:: python

   # Define parameter bounds: [lower_bounds, upper_bounds]
   # Order: [baseline, tau, contrast, stretching_exponent]
   bounds = [
       [0.9, 1e-6, 0.0, 0.5],   # lower bounds
       [1.1, 1e3,  0.5, 1.5],    # upper bounds
   ]

   fit_summary = xf.fit_g2(
       q_range=(0.01, 0.05),
       t_range=None,
       bounds=bounds,
       fit_func="single",
   )

   # fit_summary is a dict with keys:
   #   'fit_func', 'fit_val', 't_el', 'q_val',
   #   'fit_line', 'fit_x', 'label', etc.
   print(f"Fitted Q values: {fit_summary['q_val']}")
   print(f"Fit parameters shape: {fit_summary['fit_val'].shape}")

The ``fit_g2`` method performs NLSQ fitting across all Q bins in the specified
range. The ``fit_summary`` dictionary contains both the fitted parameters and
the smooth fit curves for visualization. See :doc:`fitting_guide` for the
full fitting pipeline including Bayesian inference.

Step 7 -- Context Manager Usage
--------------------------------

:class:`~xpcsviewer.xpcs_file.XpcsFile` supports the context manager
protocol for automatic cleanup:

.. code-block:: python

   with XpcsFile("/path/to/data.hdf") as xf:
       q_values, t_el, g2, g2_err, labels = xf.get_g2_data()
       # Work with data...

   # Resources are released when exiting the context

Using the context manager is recommended when processing many files in a loop
to ensure HDF5 file handles are released promptly.

Next Steps
----------

- :doc:`mask_editor` -- Create masks and compute Q-maps with SimpleMask
- :doc:`fitting_guide` -- Full fitting workflow with NLSQ and Bayesian inference
- :doc:`backend_selection` -- Switch between NumPy and JAX backends
- :doc:`cookbook` -- Common patterns and recipes
