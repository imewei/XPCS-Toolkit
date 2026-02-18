Getting Started with XPCS Viewer
=================================

This tutorial walks you through loading XPCS data from HDF5 files, inspecting
metadata, and performing a basic G2 correlation analysis.

Prerequisites
-------------

Install xpcsviewer in a virtual environment:

.. code-block:: bash

   uv sync           # If using uv (recommended)
   # or
   pip install xpcsviewer

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
``"Twotime"``, or both).

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

You can restrict the range of Q values and delay times:

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

Step 6 -- Basic G2 Fitting
----------------------------

Fit a single-exponential decay model to the G2 data:

.. math::

   g_2(\tau) = \text{baseline} + \text{contrast} \cdot \exp(-2\tau / \tau_c)

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

Step 7 -- Context Manager Usage
--------------------------------

:class:`~xpcsviewer.xpcs_file.XpcsFile` supports the context manager
protocol for automatic cleanup:

.. code-block:: python

   with XpcsFile("/path/to/data.hdf") as xf:
       q_values, t_el, g2, g2_err, labels = xf.get_g2_data()
       # Work with data...

   # Resources are released when exiting the context

Next Steps
----------

- :doc:`mask_editor` -- Create masks and compute Q-maps with SimpleMask
- :doc:`fitting_guide` -- Full fitting workflow with NLSQ and Bayesian inference
- :doc:`backend_selection` -- Switch between NumPy and JAX backends
- :doc:`cookbook` -- Common patterns and recipes
