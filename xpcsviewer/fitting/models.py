"""NumPyro model definitions for G2 correlation fitting.

This module defines probabilistic models for Bayesian inference
of XPCS correlation function parameters using NumPyro.

Models:
    - single_exp_model: Single exponential decay
    - double_exp_model: Double exponential decay
    - stretched_exp_model: Stretched exponential (KWW)
    - power_law_model: Power law Q-dependence
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array

# Check if NumPyro is available
try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    NUMPYRO_AVAILABLE = True
    JAX_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    JAX_AVAILABLE = False


def check_numpyro() -> None:
    """Raise error if NumPyro is not available."""
    if not NUMPYRO_AVAILABLE:
        raise ImportError(
            "NumPyro is required for Bayesian fitting. "
            "Install with: pip install numpyro"
        )


def single_exp_model(
    x: Array,
    y: Array | None = None,
    yerr: Array | None = None,
) -> None:
    """Single exponential decay model for G2 correlation.

    Model: y = baseline + contrast * exp(-2 * x / tau)

    Parameters
    ----------
    x : Array
        Delay times
    y : Array, optional
        Observed G2 values (None for prior predictive)
    yerr : Array, optional
        Measurement uncertainties
    """
    check_numpyro()

    # Priors
    tau = numpyro.sample("tau", dist.LogNormal(0.0, 2.0))
    baseline = numpyro.sample("baseline", dist.Normal(1.0, 0.1))
    contrast = numpyro.sample("contrast", dist.HalfNormal(1.0))

    # Model prediction
    mu = baseline + contrast * jnp.exp(-2 * x / tau)

    # Observation noise
    if yerr is not None:
        sigma = yerr
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def double_exp_model(
    x: Array,
    y: Array | None = None,
    yerr: Array | None = None,
) -> None:
    """Double exponential decay model for G2 correlation.

    Model: y = baseline + contrast1 * exp(-2x/tau1) + contrast2 * exp(-2x/tau2)

    Parameters
    ----------
    x : Array
        Delay times
    y : Array, optional
        Observed G2 values (None for prior predictive)
    yerr : Array, optional
        Measurement uncertainties
    """
    check_numpyro()

    # Priors - order tau1 < tau2
    tau1 = numpyro.sample("tau1", dist.LogNormal(0.0, 2.0))
    tau2_factor = numpyro.sample("tau2_factor", dist.LogNormal(0.0, 1.0))
    tau2 = numpyro.deterministic("tau2", tau1 * (1 + tau2_factor))

    baseline = numpyro.sample("baseline", dist.Normal(1.0, 0.1))
    contrast1 = numpyro.sample("contrast1", dist.HalfNormal(0.5))
    contrast2 = numpyro.sample("contrast2", dist.HalfNormal(0.5))

    # Model prediction
    mu = (
        baseline
        + contrast1 * jnp.exp(-2 * x / tau1)
        + contrast2 * jnp.exp(-2 * x / tau2)
    )

    # Observation noise
    if yerr is not None:
        sigma = yerr
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def stretched_exp_model(
    x: Array,
    y: Array | None = None,
    yerr: Array | None = None,
) -> None:
    """Stretched exponential (KWW) model for G2 correlation.

    Model: y = baseline + contrast * exp(-(2 * x / tau)^beta)

    Parameters
    ----------
    x : Array
        Delay times
    y : Array, optional
        Observed G2 values (None for prior predictive)
    yerr : Array, optional
        Measurement uncertainties
    """
    check_numpyro()

    # Priors
    tau = numpyro.sample("tau", dist.LogNormal(0.0, 2.0))
    baseline = numpyro.sample("baseline", dist.Normal(1.0, 0.1))
    contrast = numpyro.sample("contrast", dist.HalfNormal(1.0))
    # Stretching exponent typically in (0, 1) for subdiffusion
    beta = numpyro.sample("beta", dist.Beta(2.0, 2.0))

    # Model prediction
    mu = baseline + contrast * jnp.exp(-jnp.power(2 * x / tau, beta))

    # Observation noise
    if yerr is not None:
        sigma = yerr
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def power_law_model(
    q: Array,
    tau: Array | None = None,
    tau_err: Array | None = None,
) -> None:
    """Power law Q-dependence model for relaxation time.

    Model: tau = tau0 * q^(-alpha)

    Parameters
    ----------
    q : Array
        Q values
    tau : Array, optional
        Observed relaxation times (None for prior predictive)
    tau_err : Array, optional
        Uncertainties on tau values
    """
    check_numpyro()

    # Priors
    tau0 = numpyro.sample("tau0", dist.LogNormal(0.0, 2.0))
    alpha = numpyro.sample("alpha", dist.Normal(2.0, 1.0))  # Expect ~2 for diffusion

    # Model prediction (in log space for stability)
    log_mu = jnp.log(tau0) - alpha * jnp.log(q)
    mu = jnp.exp(log_mu)

    # Observation noise
    if tau_err is not None:
        sigma = tau_err
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))

    # Likelihood
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=tau)


# Model function signatures for NLSQ fitting
# These use jax.numpy for JAX tracing compatibility
# JIT-compiled for 5-10x performance improvement (OPT-001)


@jax.jit
def single_exp_func(x, tau, baseline, contrast):
    """Single exponential function for NLSQ fitting.

    Uses jax.numpy for JAX compatibility during optimization.
    JIT-compiled for accelerated execution.
    """
    return baseline + contrast * jnp.exp(-2 * x / tau)


@jax.jit
def double_exp_func(x, tau1, tau2, baseline, contrast1, contrast2):
    """Double exponential function for NLSQ fitting.

    Uses jax.numpy for JAX compatibility during optimization.
    JIT-compiled for accelerated execution.
    """
    return (
        baseline
        + contrast1 * jnp.exp(-2 * x / tau1)
        + contrast2 * jnp.exp(-2 * x / tau2)
    )


@jax.jit
def stretched_exp_func(x, tau, baseline, contrast, beta):
    """Stretched exponential function for NLSQ fitting.

    Uses jax.numpy for JAX compatibility during optimization.
    JIT-compiled for accelerated execution.
    """
    return baseline + contrast * jnp.exp(-jnp.power(2 * x / tau, beta))


@jax.jit
def power_law_func(q, tau0, alpha):
    """Power law function for NLSQ fitting.

    Uses jax.numpy for JAX compatibility during optimization.
    JIT-compiled for accelerated execution.
    """
    return tau0 * jnp.power(q, -alpha)
