Fitting Module
==============

Bayesian fitting with NumPyro NUTS sampler and JAX-accelerated NLSQ warm-start.

.. currentmodule:: xpcsviewer.fitting

Overview
--------

The fitting module provides Bayesian parameter estimation using NumPyro's
NUTS sampler with JAX-accelerated nonlinear least squares (NLSQ) warm-start.

**Key Features (NLSQ 0.6.0):**

- Enhanced statistical metrics: R², adjusted R², RMSE, MAE, AIC, BIC
- Confidence intervals for parameters (95% level)
- Prediction intervals accounting for observation noise
- Automatic bounds inference
- Numerical stability checks and fixes
- Fallback strategies for difficult optimization problems
- Model health diagnostics

Quick Start
-----------

Basic NLSQ fitting:

.. code-block:: python

   from xpcsviewer.fitting import nlsq_fit
   import jax.numpy as jnp

   def model(x, tau, baseline, contrast):
       return baseline + contrast * jnp.exp(-2 * x / tau)

   result = nlsq_fit(
       model, x_data, y_data, y_errors,
       p0={'tau': 1.0, 'baseline': 1.0, 'contrast': 0.3},
       bounds={'tau': (0.01, 100), 'baseline': (0.9, 1.1), 'contrast': (0.1, 0.5)},
       preset='robust',
   )

   print(f"R² = {result.r_squared:.4f}")
   print(result.summary())

Bayesian fitting:

.. code-block:: python

   from xpcsviewer.fitting import fit_single_exp

   result = fit_single_exp(delay_times, g2_values, g2_errors)
   print(f"τ = {result.get_mean('tau'):.3f} ± {result.get_std('tau'):.3f}")

Public API
----------

Fitting Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: fit_single_exp

.. autofunction:: fit_double_exp

.. autofunction:: fit_stretched_exp

.. autofunction:: fit_power_law

.. autofunction:: nlsq_fit

Result Classes
~~~~~~~~~~~~~~

.. autoclass:: NLSQResult
   :members:
   :no-index:

.. autoclass:: FitResult
   :members:
   :no-index:

.. autoclass:: FitDiagnostics
   :members:
   :no-index:

.. autoclass:: SamplerConfig
   :members:
   :no-index:

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: compute_uncertainty_band

.. autofunction:: compute_prediction_interval

.. autofunction:: plot_nlsq_fit

.. autofunction:: plot_posterior_predictive

.. autofunction:: plot_comparison

.. autofunction:: generate_arviz_diagnostics

.. autofunction:: save_figure

.. autofunction:: validate_pcov

.. autofunction:: apply_publication_style

Gradient-Based Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: grad

.. autofunction:: value_and_grad

.. autofunction:: minimize_with_grad

Constants
~~~~~~~~~

.. py:data:: PUBLICATION_STYLE

   Dictionary of matplotlib rcParams for publication-quality figures.

NLSQ 0.6.0 Features
-------------------

Statistical Metrics
~~~~~~~~~~~~~~~~~~~

The ``NLSQResult`` class provides enhanced statistical metrics:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Description
     - Usage
   * - ``r_squared``
     - Coefficient of determination (R²)
     - Model explanatory power (0-1, higher is better)
   * - ``adj_r_squared``
     - Adjusted R² for model comparison
     - Penalizes additional parameters
   * - ``rmse``
     - Root mean squared error
     - Fit quality in data units
   * - ``mae``
     - Mean absolute error
     - Robust to outliers
   * - ``aic``
     - Akaike Information Criterion
     - Model selection (lower is better)
   * - ``bic``
     - Bayesian Information Criterion
     - Penalizes complexity more than AIC

.. note::

   **Weighted R² Behavior**: When measurement uncertainties (``yerr``) are provided,
   NLSQ 0.6.0 computes R² using weighted residuals. This can result in negative R²
   values if the weighted model fit is worse than the weighted mean. For weighted
   fits, use ``chi_squared`` (reduced chi-squared should be ~1 for a good fit)
   rather than R² to assess fit quality.

Confidence vs Prediction Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module provides two types of uncertainty bands:

**Confidence Intervals** (via ``compute_uncertainty_band``):

- Represent uncertainty in the fitted curve
- Account only for parameter covariance
- Show where the "true" regression line likely falls

**Prediction Intervals** (via ``compute_prediction_interval``):

- Represent uncertainty for new observations
- Account for parameter uncertainty AND observation noise
- Always wider than confidence intervals
- Show where new data points would likely fall

.. code-block:: python

   from xpcsviewer.fitting import compute_uncertainty_band, compute_prediction_interval

   # Confidence band (parameter uncertainty only)
   y_fit, ci_lower, ci_upper = compute_uncertainty_band(
       model, x, popt, pcov, confidence=0.95
   )

   # Prediction interval (includes observation noise)
   y_fit, pi_lower, pi_upper = compute_prediction_interval(
       model, x, popt, pcov, residuals, confidence=0.95
   )

Advanced Options
~~~~~~~~~~~~~~~~

The ``nlsq_fit`` function supports advanced NLSQ 0.6.0 options:

.. code-block:: python

   result = nlsq_fit(
       model_fn, x, y, yerr, p0, bounds,
       preset='robust',          # 'fast', 'robust', 'global', 'streaming', 'large'
       auto_bounds=True,         # Automatic bounds inference from data
       stability='auto',         # 'auto', 'check', or False
       fallback=True,            # Enable fallback strategies
       compute_diagnostics=True, # Model health diagnostics
       show_progress=True,       # Progress bar for long fits
   )

   # Access diagnostics
   if result.model_diagnostics:
       print(result.model_diagnostics)

Presets
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 60 25

   * - Preset
     - Description
     - Use Case
   * - ``fast``
     - Single-start optimization
     - Quick fits, good initial guess
   * - ``robust``
     - Multi-start with 5 starts (default)
     - General use
   * - ``global``
     - Thorough search with 20 starts
     - Complex optimization landscapes
   * - ``streaming``
     - Streaming for large datasets
     - N > 100,000 points
   * - ``large``
     - Auto-detect and use appropriate strategy
     - Unknown dataset size

Models
------

The module provides four fitting models:

Single Exponential
~~~~~~~~~~~~~~~~~~

.. math::

   G_2(\tau) = \text{baseline} + \text{contrast} \cdot \exp(-2\tau/\tau_c)

Parameters: ``tau``, ``baseline``, ``contrast``

Double Exponential
~~~~~~~~~~~~~~~~~~

.. math::

   G_2(\tau) = \text{baseline} + c_1 \exp(-2\tau/\tau_1) + c_2 \exp(-2\tau/\tau_2)

Parameters: ``tau1``, ``tau2``, ``baseline``, ``contrast1``, ``contrast2``

Stretched Exponential (KWW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   G_2(\tau) = \text{baseline} + \text{contrast} \cdot \exp(-(2\tau/\tau_c)^\beta)

Parameters: ``tau``, ``baseline``, ``contrast``, ``beta``

Power Law
~~~~~~~~~

.. math::

   \tau = \tau_0 \cdot q^{-\alpha}

Parameters: ``tau0``, ``alpha``

Legacy Functions
----------------

For backward compatibility with older code:

.. autofunction:: xpcsviewer.fitting.single_exp

.. autofunction:: xpcsviewer.fitting.double_exp

.. autofunction:: xpcsviewer.fitting.single_exp_all

.. autofunction:: xpcsviewer.fitting.double_exp_all

.. autofunction:: xpcsviewer.fitting.fit_with_fixed

.. autofunction:: xpcsviewer.fitting.robust_curve_fit

See Also
--------

- :doc:`plotting` - Additional visualization utilities
- :doc:`modules` - G2 correlation analysis module
