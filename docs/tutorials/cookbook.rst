Cookbook: Common Patterns and Recipes
======================================

This page collects short, self-contained recipes for common xpcsviewer tasks.
Each recipe can be copied and adapted to your workflow.

.. contents:: Recipes
   :local:
   :depth: 1

Loading Multiple Files
-----------------------

Process a batch of HDF5 result files and collect G2 data:

.. code-block:: python

   from pathlib import Path
   from xpcsviewer.xpcs_file import XpcsFile

   data_dir = Path("/path/to/results")
   hdf_files = sorted(data_dir.glob("*_Multitau_result.hdf"))

   all_g2 = []
   for fpath in hdf_files:
       with XpcsFile(str(fpath)) as xf:
           q_values, t_el, g2, g2_err, labels = xf.get_g2_data()
           all_g2.append({
               "file": fpath.name,
               "q": q_values,
               "t": t_el,
               "g2": g2,
               "g2_err": g2_err,
               "labels": labels,
           })

   print(f"Loaded {len(all_g2)} files")

Comparing G2 Across Files
---------------------------

Overlay G2 curves from multiple files at the same Q bin:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   q_index = 5  # Compare the 6th Q bin

   fig, ax = plt.subplots(figsize=(8, 5))
   for entry in all_g2:
       if q_index < entry["g2"].shape[1]:
           ax.errorbar(
               entry["t"], entry["g2"][:, q_index],
               yerr=entry["g2_err"][:, q_index],
               fmt="o", ms=3, label=entry["file"],
           )

   ax.set_xscale("log")
   ax.set_xlabel("Delay time (s)")
   ax.set_ylabel(r"$g_2(\tau)$")
   ax.set_title(f"G2 comparison at Q bin {q_index}")
   ax.legend(fontsize=7)
   plt.tight_layout()
   plt.show()

Batch Fitting with NLSQ
-------------------------

Fit all Q bins across multiple files:

.. code-block:: python

   from xpcsviewer.fitting import nlsq_fit
   from xpcsviewer.fitting.models import single_exp_func

   results = []

   for entry in all_g2:
       t = entry["t"]
       for qi in range(entry["g2"].shape[1]):
           g2_col = entry["g2"][:, qi]
           err_col = entry["g2_err"][:, qi]

           result = nlsq_fit(
               model_fn=single_exp_func,
               x=t, y=g2_col, yerr=err_col,
               p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
               bounds={
                   "tau": (1e-6, 1e6),
                   "baseline": (0.5, 2.0),
                   "contrast": (0.0, 1.0),
               },
               preset="fast",
           )

           if result.converged and not result.is_fallback:
               results.append({
                   "file": entry["file"],
                   "q": entry["q"][qi],
                   "tau": result.params["tau"],
                   "tau_err": result.get_param_uncertainty("tau"),
                   "r_squared": result.r_squared,
               })

   print(f"Successful fits: {len(results)}")

Tau vs Q Analysis (Diffusion)
-------------------------------

Extract diffusion coefficients from the Q-dependence of relaxation times:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Collect tau(q) from batch fitting results
   q_arr = np.array([r["q"] for r in results])
   tau_arr = np.array([r["tau"] for r in results])
   tau_err_arr = np.array([r["tau_err"] for r in results])

   # For diffusive dynamics: 1/tau = D * q^2
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # tau vs q
   axes[0].errorbar(q_arr, tau_arr, yerr=tau_err_arr, fmt="o")
   axes[0].set_xlabel(r"$q$ (nm$^{-1}$)")
   axes[0].set_ylabel(r"$\tau$ (s)")
   axes[0].set_title(r"$\tau$ vs $q$")
   axes[0].set_yscale("log")

   # 1/tau vs q^2 (should be linear for simple diffusion)
   axes[1].errorbar(q_arr**2, 1.0/tau_arr, fmt="o")
   axes[1].set_xlabel(r"$q^2$ (nm$^{-2}$)")
   axes[1].set_ylabel(r"$1/\tau$ (s$^{-1}$)")
   axes[1].set_title(r"$1/\tau$ vs $q^2$")

   # Fit linear slope for diffusion coefficient
   valid = np.isfinite(tau_arr) & (tau_arr > 0)
   if valid.sum() > 2:
       coeffs = np.polyfit(q_arr[valid]**2, 1.0/tau_arr[valid], 1)
       D = coeffs[0]
       axes[1].plot(q_arr**2, np.polyval(coeffs, q_arr**2), "r--",
                    label=f"D = {D:.2e} nm$^2$/s")
       axes[1].legend()

   plt.tight_layout()
   plt.show()

Reading G2 Data with HDF5Facade
---------------------------------

The :class:`~xpcsviewer.io.hdf5_facade.HDF5Facade` provides schema-validated
reading:

.. code-block:: python

   from xpcsviewer.io import HDF5Facade
   from xpcsviewer.schemas import G2Data

   facade = HDF5Facade()

   # Read validated G2 data
   g2_data = facade.read_g2_data("/path/to/data.hdf")

   # g2_data is a G2Data schema instance
   print(f"G2 shape: {g2_data.g2.shape}")
   print(f"Delay times: {g2_data.delay_times.shape}")
   print(f"Q values: {g2_data.q_values}")

   # Read Q-map with validation
   qmap = facade.read_qmap("/path/to/data.hdf")
   print(f"Q-map shape: {qmap.sqmap.shape}")
   print(f"Q-map unit: {qmap.sqmap_unit}")

Creating G2Data from Arrays
-----------------------------

Construct validated data objects from raw arrays:

.. code-block:: python

   import numpy as np
   from xpcsviewer.schemas import G2Data

   # From arrays (must be float64)
   g2_data = G2Data(
       g2=np.random.rand(100, 24).astype(np.float64),
       g2_err=np.full((100, 24), 0.01, dtype=np.float64),
       delay_times=np.logspace(-3, 2, 100).astype(np.float64),
       q_values=[0.01 * (i + 1) for i in range(24)],
   )

   # From a legacy dictionary
   g2_data = G2Data.from_dict({
       "g2": g2_array,
       "g2_err": err_array,
       "delay_times": times,
       "q_values": q_list,
   })

Backend-Aware Computation
---------------------------

Write analysis code that works with both NumPy and JAX:

.. code-block:: python

   from xpcsviewer.backends import get_backend, ensure_numpy

   backend = get_backend()

   def compute_structure_factor(q, intensity, background):
       """Compute S(q) with backend-agnostic operations."""
       net = backend.clip(intensity - background, 0, None)
       norm = backend.max(net)
       return net / norm if norm > 0 else backend.zeros_like(net)

   # Compute (may use JAX or NumPy)
   sq = compute_structure_factor(q, intensity, background)

   # Convert at I/O boundary
   import matplotlib.pyplot as plt
   plt.plot(ensure_numpy(q), ensure_numpy(sq))
   plt.show()

Exporting Results to CSV
--------------------------

.. code-block:: python

   import csv

   with open("fit_results.csv", "w", newline="") as f:
       writer = csv.DictWriter(f, fieldnames=["file", "q", "tau", "tau_err", "r_squared"])
       writer.writeheader()
       writer.writerows(results)

Two-Time Correlation Analysis
-------------------------------

For files with two-time analysis data:

.. code-block:: python

   from xpcsviewer.xpcs_file import XpcsFile

   with XpcsFile("/path/to/A001_Twotime_result.hdf") as xf:
       # Check analysis type
       print(f"Analysis type: {xf.atype}")

       if "Twotime" in xf.atype:
           # Get the two-time correlation maps
           dqmap_disp, saxs, selection = xf.get_twotime_maps(
               scale="log",
               auto_crop=True,
           )

           # G2 is also available from two-time data
           q_values, t_el, g2, g2_err, labels = xf.get_g2_data()

Memory-Efficient Processing
-----------------------------

For large files, use context managers and clear caches:

.. code-block:: python

   from xpcsviewer.xpcs_file import XpcsFile

   results = []

   for fpath in hdf_files:
       with XpcsFile(str(fpath)) as xf:
           # Process data
           q_values, t_el, g2, g2_err, labels = xf.get_g2_data()

           fit_summary = xf.fit_g2(
               bounds=[[0.9, 1e-6, 0.0, 0.5], [1.1, 1e3, 0.5, 1.5]],
               fit_func="single",
           )

           # Collect only the summary, not the raw data
           results.append({
               "file": fpath.name,
               "fit_val": fit_summary["fit_val"].copy(),
               "q_val": fit_summary["q_val"].copy(),
           })

           # Clear computation caches to free memory
           xf.clear_cache("computation")

   # XpcsFile releases resources when exiting the `with` block

Custom Model Functions
-----------------------

Define a custom model and fit with NLSQ:

.. code-block:: python

   import numpy as np
   from xpcsviewer.fitting import nlsq_fit

   # Custom model: stretched exponential
   def custom_model(x, tau, baseline, contrast, beta):
       tau = np.clip(tau, 1e-30, None)
       return baseline + contrast * np.exp(-2 * (x / tau) ** beta)

   result = nlsq_fit(
       model_fn=custom_model,
       x=t, y=g2_data, yerr=g2_err,
       p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3, "beta": 1.0},
       bounds={
           "tau": (1e-6, 1e6),
           "baseline": (0.5, 2.0),
           "contrast": (0.0, 1.0),
           "beta": (0.1, 2.0),
       },
       preset="robust",
       stability="auto",
   )

   print(f"Stretching exponent beta = {result.params['beta']:.3f}")

Logging Configuration
-----------------------

Control log verbosity for debugging:

.. code-block:: python

   import os

   # Set before importing xpcsviewer
   os.environ["PYXPCS_LOG_LEVEL"] = "DEBUG"    # DEBUG/INFO/WARNING/ERROR
   os.environ["PYXPCS_LOG_FORMAT"] = "TEXT"     # TEXT or JSON

   from xpcsviewer.utils import get_logger
   logger = get_logger(__name__)
   logger.info("Starting analysis")
