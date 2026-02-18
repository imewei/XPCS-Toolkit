G2 Correlation Function Theory
===============================

This page explains the theoretical background of the G2 autocorrelation
function, the primary observable in XPCS experiments, and how XPCS Viewer
models and fits it.

Physical Background
--------------------

In an XPCS experiment, coherent X-rays scattered from a sample produce a
speckle pattern on the detector. As the sample evolves (particles diffusing,
domains relaxing, etc.), the speckle pattern fluctuates. The **intensity
autocorrelation function** quantifies these temporal fluctuations:

.. math::

   g_2(q, \tau) = \frac{\langle I(q, t) \cdot I(q, t + \tau) \rangle_t}
                       {\langle I(q, t) \rangle_t^2}

where:

- :math:`I(q, t)` is the scattered intensity at wavevector :math:`q` and
  time :math:`t`
- :math:`\tau` is the delay time (lag)
- :math:`\langle \cdot \rangle_t` denotes a time average over frames

The :math:`g_2` function is computed per Q-bin from the detector pixel data
and the Q-map partition.

Siegert Relation
-----------------

The intensity autocorrelation :math:`g_2` is related to the **field
autocorrelation** :math:`g_1` via the Siegert relation:

.. math::

   g_2(q, \tau) = 1 + \beta \cdot |g_1(q, \tau)|^2

where :math:`\beta` is the **speckle contrast** (or coherence factor),
which depends on the optical setup (beam coherence, pixel-to-speckle size
ratio, etc.). In practice, :math:`\beta \in [0, 1]` and is treated as a
fit parameter.

Fitting Models
---------------

XPCS Viewer provides several functional forms for fitting :math:`g_2(\tau)`.
All models share the parameterization convention where ``baseline``
corresponds to the long-time limit (typically 1.0) and ``contrast``
corresponds to the Siegert :math:`\beta` factor.

Single Exponential
^^^^^^^^^^^^^^^^^^^

The simplest and most common model, appropriate for simple diffusive dynamics:

.. math::

   g_2(\tau) = \text{baseline} + \text{contrast} \cdot \exp\!\left(-\frac{2\tau}{\tau_c}\right)

where :math:`\tau_c` is the **characteristic relaxation time**. The factor
of 2 comes from the Siegert relation (:math:`|g_1|^2` squares the field
correlation, doubling the decay rate).

.. note::

   The implementation uses ``jnp.clip(tau, 1e-30)`` to prevent numerical
   instability when the optimizer samples near :math:`\tau_c = 0`.

**Physical interpretation**: :math:`\tau_c \propto 1 / (D q^2)` for
Brownian diffusion, where :math:`D` is the diffusion coefficient.
Plotting :math:`\tau_c` vs. :math:`q` (tau-Q analysis) extracts the
scaling exponent.

Double Exponential
^^^^^^^^^^^^^^^^^^^

For systems with two distinct relaxation processes:

.. math::

   g_2(\tau) = \text{baseline}
              + \text{contrast}_1 \cdot \exp\!\left(-\frac{2\tau}{\tau_1}\right)
              + \text{contrast}_2 \cdot \exp\!\left(-\frac{2\tau}{\tau_2}\right)

This model captures bimodal dynamics (e.g., fast cage rattling and slow
structural relaxation in glassy systems).

Stretched Exponential (KWW)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Kohlrausch-Williams-Watts (KWW) function models heterogeneous dynamics:

.. math::

   g_2(\tau) = \text{baseline}
              + \text{contrast} \cdot \exp\!\left[-\left(\frac{2\tau}{\tau_c}\right)^\beta\right]

where :math:`\beta \in (0, 1]` is the **stretching exponent**:

- :math:`\beta = 1`: simple exponential (monodisperse dynamics)
- :math:`\beta < 1`: stretched decay (broad distribution of relaxation times)
- :math:`\beta > 1`: compressed exponential (ballistic or stress-driven dynamics)

Two-Stage Fitting Pipeline
---------------------------

XPCS Viewer uses a two-stage approach for fitting :math:`g_2` data:

**Stage 1: Point Estimates (NLSQ)**

NLSQ 0.6.0 performs nonlinear least squares optimization to obtain fast
point estimates (:math:`\hat{\tau}_c`, :math:`\hat{\beta}`, etc.) along
with a covariance matrix. The native ``CurveFitResult`` provides R-squared,
AIC, BIC, prediction intervals, and model health diagnostics.

**Stage 2: Bayesian Inference (NumPyro NUTS)**

The NLSQ point estimates serve as a **warm start** for the No-U-Turn Sampler
(NUTS), a Hamiltonian Monte Carlo algorithm implemented in NumPyro. NUTS
explores the full posterior distribution over parameters, providing:

- Credible intervals (not just confidence intervals)
- Model comparison via Bayesian information criteria
- Diagnostics for convergence (R-hat, ESS, BFMI)

The Bayesian models use weakly informative priors:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Prior
     - Rationale
   * - :math:`\tau_c`
     - LogNormal(0, 2)
     - Positive, broad, spans many decades
   * - baseline
     - Normal(1.0, 0.1)
     - Close to 1 for normalized :math:`g_2`
   * - contrast
     - HalfNormal(1.0)
     - Positive, typically 0--0.5
   * - :math:`\sigma`
     - HalfNormal(0.1)
     - Observation noise (when errors not provided)

Convergence Diagnostics
------------------------

After NUTS sampling, XPCS Viewer checks five convergence diagnostics
via ArviZ:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Diagnostic
     - Threshold
     - Meaning
   * - R-hat
     - < 1.01
     - All chains converge to the same distribution
   * - ESS bulk
     - > 400
     - Sufficient effective samples for mean/variance
   * - ESS tail
     - > 400
     - Sufficient effective samples for quantiles
   * - Divergences
     - = 0
     - No divergent transitions (pathological geometry)
   * - BFMI
     - >= 0.2
     - Adequate energy exploration in Hamiltonian dynamics

A fit is considered **converged** only when all five criteria are satisfied.
Failed diagnostics are reported in the ``FitDiagnostics`` object attached
to each ``FitResult``.

Model Selection
----------------

When multiple models (single exponential, double exponential, KWW) are fit
to the same data, XPCS Viewer supports model comparison via:

- **AIC (Akaike Information Criterion)**: Balances goodness of fit against
  model complexity. Lower is better.
- **BIC (Bayesian Information Criterion)**: Similar to AIC but with a
  stronger penalty for extra parameters. Preferred for large datasets.
- **R-squared**: Coefficient of determination. Useful for quick assessment
  but does not penalize complexity.

These metrics are available directly from the ``NLSQResult`` object, which
delegates to NLSQ 0.6.0's native ``CurveFitResult`` properties.
