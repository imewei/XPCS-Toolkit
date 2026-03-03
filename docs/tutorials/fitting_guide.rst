Fitting G2 Correlation Functions
=================================

This tutorial covers the full fitting pipeline for XPCS G2 correlation
functions: from quick NLSQ point estimates to full Bayesian inference with
NumPyro NUTS.

Overview of the Fitting Pipeline
----------------------------------

The xpcsviewer fitting module provides two complementary approaches:

1. **NLSQ** (Non-Linear Least Squares) -- Fast point estimates using
   ``nlsq 0.6.0``. Returns :class:`~xpcsviewer.fitting.results.NLSQResult`
   with R², confidence intervals, and model diagnostics.

2. **Bayesian NUTS** (No-U-Turn Sampler) -- Full posterior via NumPyro.
   Uses NLSQ warm-start for initialization. Returns
   :class:`~xpcsviewer.fitting.results.FitResult` with posterior samples,
   HDI intervals, and ArviZ diagnostics.

Physical Models
-----------------

The G2 autocorrelation for a single relaxation process is:

.. math::

   g_2(\tau) = \text{baseline} + \text{contrast} \cdot \exp(-2\tau / \tau_c)

where :math:`\tau_c` is the characteristic relaxation time, baseline is
typically near 1.0, and contrast is the Siegert factor.

For systems with two relaxation modes:

.. math::

   g_2(\tau) = \text{baseline} + c_1 \exp(-2\tau / \tau_1) + c_2 \exp(-2\tau / \tau_2)

Available model functions in :mod:`xpcsviewer.fitting.models`:

- :func:`~xpcsviewer.fitting.models.single_exp_func` -- ``baseline + contrast * exp(-2x/tau)``
- :func:`~xpcsviewer.fitting.models.double_exp_func` -- Two-component decay
- :func:`~xpcsviewer.fitting.models.stretched_exp_func` -- Stretched exponential
- :func:`~xpcsviewer.fitting.models.power_law_func` -- Power-law decay

Part 1: NLSQ Fitting
----------------------

Quick Point Estimates
^^^^^^^^^^^^^^^^^^^^^^

The :func:`~xpcsviewer.fitting.nlsq_fit` function provides NLSQ fitting with
the native ``nlsq 0.6.0`` API:

.. code-block:: python

   import numpy as np
   from xpcsviewer.fitting import nlsq_fit
   from xpcsviewer.fitting.models import single_exp_func

   # Synthetic G2 data
   np.random.seed(42)
   tau_true = 5.0
   baseline_true = 1.0
   contrast_true = 0.3
   t = np.logspace(-2, 3, 100)
   g2_clean = baseline_true + contrast_true * np.exp(-2 * t / tau_true)
   g2_noise = g2_clean + np.random.normal(0, 0.005, size=t.shape)
   g2_err = np.full_like(t, 0.005)

   result = nlsq_fit(
       model_fn=single_exp_func,
       x=t,
       y=g2_noise,
       yerr=g2_err,
       p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
       bounds={
           "tau": (1e-6, 1e6),
           "baseline": (0.5, 2.0),
           "contrast": (0.0, 1.0),
       },
   )

   print(f"Converged: {result.converged}")
   print(f"tau = {result.params['tau']:.3f}")
   print(f"baseline = {result.params['baseline']:.4f}")
   print(f"contrast = {result.params['contrast']:.4f}")

Examining Fit Quality
^^^^^^^^^^^^^^^^^^^^^^

:class:`~xpcsviewer.fitting.results.NLSQResult` provides comprehensive
fit quality metrics:

.. code-block:: python

   # Statistical metrics (delegated to native nlsq 0.6.0 CurveFitResult)
   print(f"R-squared:      {result.r_squared:.6f}")
   print(f"Adjusted R-sq:  {result.adj_r_squared:.6f}")
   print(f"RMSE:           {result.rmse:.6e}")
   print(f"MAE:            {result.mae:.6e}")
   print(f"Chi-squared:    {result.chi_squared:.4f}")

   # Model selection criteria
   print(f"AIC:            {result.aic:.2f}")
   print(f"BIC:            {result.bic:.2f}")

   # Parameter uncertainties from covariance
   for name in result.params:
       std_err = result.get_param_uncertainty(name)
       ci_lo, ci_hi = result.get_confidence_interval(name)
       print(f"  {name}: {result.params[name]:.4f} +/- {std_err:.4f}"
             f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

   # Full formatted summary
   print(result.summary())

Robust Fitting Options
^^^^^^^^^^^^^^^^^^^^^^^

Use presets, stability checks, and fallback strategies for difficult data:

.. code-block:: python

   result = nlsq_fit(
       model_fn=single_exp_func,
       x=t, y=g2_noise, yerr=g2_err,
       p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
       bounds={"tau": (1e-6, 1e6), "baseline": (0.5, 2.0), "contrast": (0, 1)},
       workflow='auto_global',    # Multi-start global optimization
       stability="auto",         # Automatic numerical fixes
       fallback=True,            # Fallback strategies for hard problems
       compute_diagnostics=True, # Enable model health report
   )

   # Check model health
   if result.diagnostics is not None:
       print(f"Health score: {result.health_score}/100")
       print(f"Is healthy: {result.is_healthy}")
       print(f"Condition number: {result.condition_number:.2f}")

Prediction Intervals
^^^^^^^^^^^^^^^^^^^^^

Generate prediction intervals at new x values:

.. code-block:: python

   x_fine = np.logspace(-2, 3, 500)
   pi_lower, pi_upper = result.get_prediction_interval(x_fine)

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8, 5))
   ax.errorbar(t, g2_noise, yerr=g2_err, fmt="o", ms=3, label="Data")
   ax.fill_between(x_fine, pi_lower, pi_upper, alpha=0.3, label="95% PI")
   ax.set_xscale("log")
   ax.set_xlabel("Delay time (s)")
   ax.set_ylabel(r"$g_2(\tau)$")
   ax.legend()
   plt.tight_layout()
   plt.show()

Part 2: Bayesian Fitting
--------------------------

The :func:`~xpcsviewer.fitting.fit_single_exp` function runs the full
Bayesian pipeline: NLSQ warm-start followed by NumPyro NUTS sampling.

Running the Sampler
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from xpcsviewer.fitting import fit_single_exp

   # Requires NumPyro and JAX
   result = fit_single_exp(
       x=t,
       y=g2_noise,
       yerr=g2_err,
       # SamplerConfig options
       num_warmup=500,
       num_samples=1000,
       num_chains=4,
       target_accept_prob=0.8,
       random_seed=42,
   )

   # Posterior summaries
   print(f"tau = {result.get_mean('tau'):.3f} +/- {result.get_std('tau'):.3f}")
   print(f"baseline = {result.get_mean('baseline'):.4f} +/- {result.get_std('baseline'):.4f}")
   print(f"contrast = {result.get_mean('contrast'):.4f} +/- {result.get_std('contrast'):.4f}")

The result includes the NLSQ warm-start values used for initialization:

.. code-block:: python

   print(f"NLSQ warm-start: {result.nlsq_init}")

Convergence Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^

Always check diagnostics before trusting posterior estimates:

.. code-block:: python

   diag = result.diagnostics

   print(f"Converged: {diag.converged}")
   print(f"R-hat: {diag.r_hat}")
   print(f"ESS (bulk): {diag.ess_bulk}")
   print(f"ESS (tail): {diag.ess_tail}")
   print(f"Divergences: {diag.divergences}")
   print(f"BFMI: {diag.bfmi}")

   # Rule of thumb:
   # - R-hat < 1.01 for all parameters
   # - ESS > 400 for reliable estimates
   # - Divergences = 0
   # - BFMI > 0.3

Highest Density Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 94% HDI (default)
   hdi_lo, hdi_hi = result.get_hdi("tau")
   print(f"tau 94% HDI: [{hdi_lo:.3f}, {hdi_hi:.3f}]")

   # Custom probability
   hdi_lo, hdi_hi = result.get_hdi("tau", prob=0.89)
   print(f"tau 89% HDI: [{hdi_lo:.3f}, {hdi_hi:.3f}]")

ArviZ Integration
^^^^^^^^^^^^^^^^^^

The result contains an xarray ``DataTree`` object (ArviZ 1.0+ format)
for advanced diagnostics:

.. code-block:: python

   import arviz as az

   # Trace plot
   az.plot_trace(result.arviz_data)

   # Forest plot
   az.plot_forest(result.arviz_data)

   # Generate full diagnostic suite
   diagnostics = result.generate_diagnostics(
       output_dir="./diagnostics",
       formats=("pdf", "png"),
   )

Posterior Predictive Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from xpcsviewer.fitting.models import single_exp_func

   fig = result.plot_posterior_predictive(
       model=single_exp_func,
       x_data=t,
       y_data=g2_noise,
   )
   plt.show()

Serialization
^^^^^^^^^^^^^^

Export the full result including reproducibility metadata:

.. code-block:: python

   result_dict = result.to_dict()

   # Includes:
   # - samples: posterior draws
   # - nlsq_init: warm-start values
   # - diagnostics: R-hat, ESS, BFMI
   # - versions: package versions (xpcsviewer, numpyro, jax, arviz)
   # - sampler_config: num_warmup, num_samples, etc.
   # - data_metadata: n_points, x_range

   import json
   with open("fit_result.json", "w") as f:
       json.dump(result_dict, f, indent=2)

Part 3: Fitting from XpcsFile
-------------------------------

The :meth:`~xpcsviewer.xpcs_file.XpcsFile.fit_g2` method integrates
fitting directly into the data loading workflow:

.. code-block:: python

   from xpcsviewer.xpcs_file import XpcsFile

   with XpcsFile("/path/to/data.hdf") as xf:
       fit_summary = xf.fit_g2(
           q_range=(0.01, 0.05),
           bounds=[[0.9, 1e-6, 0.0, 0.5], [1.1, 1e3, 0.5, 1.5]],
           fit_func="single",     # or "double"
           robust_fitting=False,   # Set True for enhanced reliability
           use_parallel=True,      # Parallel fitting across Q bins
       )

       # Access results
       print(f"Fit function: {fit_summary['fit_func']}")
       print(f"Q values: {fit_summary['q_val']}")
       print(f"Fit parameters: {fit_summary['fit_val'].shape}")

       # fit_summary['fit_line'] contains the smooth fit curves
       # fit_summary['fit_x'] contains the x values for the fit curves

Model Comparison
-----------------

Compare single vs. double exponential using AIC/BIC:

.. code-block:: python

   from xpcsviewer.fitting import nlsq_fit
   from xpcsviewer.fitting.models import single_exp_func, double_exp_func

   # Fit single exponential
   result_single = nlsq_fit(
       model_fn=single_exp_func,
       x=t, y=g2_noise, yerr=g2_err,
       p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
       bounds={"tau": (1e-6, 1e6), "baseline": (0.5, 2.0), "contrast": (0, 1)},
   )

   # Fit double exponential
   result_double = nlsq_fit(
       model_fn=double_exp_func,
       x=t, y=g2_noise, yerr=g2_err,
       p0={"tau1": 1.0, "tau2": 50.0, "baseline": 1.0,
           "contrast1": 0.2, "contrast2": 0.1},
       bounds={
           "tau1": (1e-6, 1e4), "tau2": (1e-6, 1e6),
           "baseline": (0.5, 2.0),
           "contrast1": (0, 1), "contrast2": (0, 1),
       },
   )

   print(f"Single exp -- AIC: {result_single.aic:.2f}, BIC: {result_single.bic:.2f}")
   print(f"Double exp -- AIC: {result_double.aic:.2f}, BIC: {result_double.bic:.2f}")
   print(f"Prefer: {'single' if result_single.bic < result_double.bic else 'double'}"
         f" (lower BIC)")

Next Steps
----------

- :doc:`backend_selection` -- Use JAX for GPU-accelerated fitting
- :doc:`cookbook` -- Batch fitting recipes across multiple files
- :doc:`getting_started` -- Loading data basics
