"""NumPyro NUTS sampler with NLSQ warm-start.

This module provides the MCMC sampling functionality using NumPyro's
NUTS sampler with JAX-accelerated NLSQ warm-start.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

from xpcsviewer.utils.log_utils import log_timing

from .models import (
    double_exp_func,
    double_exp_model,
    power_law_func,
    power_law_model,
    single_exp_func,
    single_exp_model,
    stretched_exp_func,
    stretched_exp_model,
)
from .nlsq import nlsq_optimize
from .results import FitDiagnostics, FitResult, SamplerConfig

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Check if NumPyro is available
try:
    import arviz as az
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


def check_numpyro() -> None:
    """Raise error if NumPyro is not available."""
    if not NUMPYRO_AVAILABLE:
        raise ImportError(
            "NumPyro is required for Bayesian fitting. "
            "Install with: pip install numpyro arviz"
        )


def _extract_config(kwargs: dict) -> SamplerConfig:
    """Extract SamplerConfig from kwargs or use defaults."""
    if "sampler_config" in kwargs:
        return kwargs["sampler_config"]

    return SamplerConfig(
        num_warmup=kwargs.get("num_warmup", 500),
        num_samples=kwargs.get("num_samples", 1000),
        num_chains=kwargs.get("num_chains", 4),
        target_accept_prob=kwargs.get("target_accept_prob", 0.8),
        max_tree_depth=kwargs.get("max_tree_depth", 10),
        random_seed=kwargs.get("random_seed"),
    )


def _run_mcmc(
    model,
    model_args: tuple,
    config: SamplerConfig,
    init_params: dict[str, float] | None = None,
) -> tuple[MCMC, dict]:
    """Run MCMC sampling with optional warm-start initialization."""
    check_numpyro()

    # Set random seed
    if config.random_seed is not None:
        rng_key = jax.random.PRNGKey(config.random_seed)
    else:
        rng_key = jax.random.PRNGKey(0)

    # Configure NUTS sampler
    kernel = NUTS(
        model,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
    )

    # Create MCMC instance
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
    )

    # Run sampling
    if init_params is not None:
        # Convert init_params to JAX arrays
        init_params_jax = {k: jnp.array(v) for k, v in init_params.items()}
        mcmc.run(rng_key, *model_args, init_params=init_params_jax)
    else:
        mcmc.run(rng_key, *model_args)

    return mcmc, mcmc.get_samples()


def _build_fit_result(
    mcmc: MCMC,
    samples: dict,
    nlsq_init: dict[str, float],
    param_names: list[str],
) -> FitResult:
    """Build FitResult from MCMC output."""
    # Convert samples to numpy
    samples_np = {k: np.asarray(v) for k, v in samples.items() if k in param_names}

    # Get diagnostics
    summary = az.summary(mcmc, var_names=param_names)

    # Extract diagnostics
    r_hat = {}
    ess_bulk = {}
    ess_tail = {}

    for name in param_names:
        if name in summary.index:
            r_hat[name] = float(summary.loc[name, "r_hat"])  # type: ignore
            ess_bulk[name] = int(summary.loc[name, "ess_bulk"])  # type: ignore
            ess_tail[name] = int(summary.loc[name, "ess_tail"])  # type: ignore

    # Count divergences
    num_divergent = int(np.sum(mcmc.get_extra_fields()["diverging"]))

    diagnostics = FitDiagnostics(
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        divergences=num_divergent,
        max_treedepth_reached=0,  # NumPyro doesn't track this directly
    )

    # Convert to ArviZ InferenceData
    arviz_data = az.from_numpyro(mcmc)

    return FitResult(
        samples=samples_np,
        summary=summary,
        diagnostics=diagnostics,
        nlsq_init=nlsq_init,
        arviz_data=arviz_data,
    )


@log_timing(threshold_ms=2000)
def run_single_exp_fit(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    stability: Literal["auto", "check", False] = "auto",
    auto_bounds: bool = False,
    **kwargs,
) -> FitResult:
    """Run single exponential fit with NLSQ warm-start.

    Parameters
    ----------
    x : array_like
        Delay times
    y : array_like
        G2 correlation values
    yerr : array_like, optional
        Measurement uncertainties
    stability : str, optional
        NLSQ stability mode: 'auto', 'check', or False (default: 'auto')
    auto_bounds : bool, optional
        Use NLSQ auto-bounds inference (default: False)
    **kwargs
        Sampler configuration

    Returns
    -------
    FitResult
        Posterior samples for tau, baseline, contrast
    """
    check_numpyro()

    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    config = _extract_config(kwargs)
    param_names = ["tau", "baseline", "contrast"]

    # NLSQ warm-start with NLSQ 0.6.0 features
    logger.info("Running NLSQ warm-start for single exponential fit")
    p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
    bounds = {
        "tau": (1e-6, 1e6),
        "baseline": (0.0, 2.0),
        "contrast": (0.0, 1.0),
    }

    nlsq_result = nlsq_optimize(
        single_exp_func,
        x,
        y,
        yerr,
        p0,
        bounds,
        stability=stability,
        auto_bounds=auto_bounds,
        compute_diagnostics=True,  # Enable for health checking
    )
    nlsq_init = nlsq_result.params

    # Log warning if NLSQ fit is unhealthy
    if hasattr(nlsq_result, "is_healthy") and not nlsq_result.is_healthy:
        health_score = getattr(nlsq_result, "health_score", "N/A")
        logger.warning(
            f"NLSQ warm-start may be unreliable: health_score={health_score}"
        )

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # Run MCMC with warm-start
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        single_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params=nlsq_init,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names)


@log_timing(threshold_ms=2000)
def run_double_exp_fit(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    stability: Literal["auto", "check", False] = "auto",
    auto_bounds: bool = False,
    **kwargs,
) -> FitResult:
    """Run double exponential fit with NLSQ warm-start.

    Parameters
    ----------
    x : array_like
        Delay times
    y : array_like
        G2 correlation values
    yerr : array_like, optional
        Measurement uncertainties
    stability : str, optional
        NLSQ stability mode: 'auto', 'check', or False (default: 'auto')
    auto_bounds : bool, optional
        Use NLSQ auto-bounds inference (default: False)
    **kwargs
        Sampler configuration

    Returns
    -------
    FitResult
        Posterior samples for tau1, tau2, baseline, contrast1, contrast2
    """
    check_numpyro()

    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    config = _extract_config(kwargs)
    param_names = ["tau1", "tau2", "baseline", "contrast1", "contrast2"]

    # NLSQ warm-start with NLSQ 0.6.0 features
    logger.info("Running NLSQ warm-start for double exponential fit")
    p0 = {
        "tau1": 0.1,
        "tau2": 10.0,
        "baseline": 1.0,
        "contrast1": 0.15,
        "contrast2": 0.15,
    }
    bounds = {
        "tau1": (1e-6, 1e6),
        "tau2": (1e-6, 1e6),
        "baseline": (0.0, 2.0),
        "contrast1": (0.0, 1.0),
        "contrast2": (0.0, 1.0),
    }

    nlsq_result = nlsq_optimize(
        double_exp_func,
        x,
        y,
        yerr,
        p0,
        bounds,
        stability=stability,
        auto_bounds=auto_bounds,
        compute_diagnostics=True,
    )
    nlsq_init = nlsq_result.params

    # Log warning if NLSQ fit is unhealthy
    if hasattr(nlsq_result, "is_healthy") and not nlsq_result.is_healthy:
        health_score = getattr(nlsq_result, "health_score", "N/A")
        logger.warning(
            f"NLSQ warm-start may be unreliable: health_score={health_score}"
        )

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # Run MCMC with warm-start
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        double_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params={
            "tau1": nlsq_init["tau1"],
            "tau2_factor": nlsq_init["tau2"] / nlsq_init["tau1"] - 1,
            "baseline": nlsq_init["baseline"],
            "contrast1": nlsq_init["contrast1"],
            "contrast2": nlsq_init["contrast2"],
        },
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names)


@log_timing(threshold_ms=2000)
def run_stretched_exp_fit(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    stability: Literal["auto", "check", False] = "auto",
    auto_bounds: bool = False,
    **kwargs,
) -> FitResult:
    """Run stretched exponential fit with NLSQ warm-start.

    Parameters
    ----------
    x : array_like
        Delay times
    y : array_like
        G2 correlation values
    yerr : array_like, optional
        Measurement uncertainties
    stability : str, optional
        NLSQ stability mode: 'auto', 'check', or False (default: 'auto')
    auto_bounds : bool, optional
        Use NLSQ auto-bounds inference (default: False)
    **kwargs
        Sampler configuration

    Returns
    -------
    FitResult
        Posterior samples for tau, baseline, contrast, beta
    """
    check_numpyro()

    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    config = _extract_config(kwargs)
    param_names = ["tau", "baseline", "contrast", "beta"]

    # NLSQ warm-start with NLSQ 0.6.0 features
    logger.info("Running NLSQ warm-start for stretched exponential fit")
    p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3, "beta": 0.8}
    bounds = {
        "tau": (1e-6, 1e6),
        "baseline": (0.0, 2.0),
        "contrast": (0.0, 1.0),
        "beta": (0.01, 0.99),
    }

    nlsq_result = nlsq_optimize(
        stretched_exp_func,
        x,
        y,
        yerr,
        p0,
        bounds,
        stability=stability,
        auto_bounds=auto_bounds,
        compute_diagnostics=True,
    )
    nlsq_init = nlsq_result.params

    # Log warning if NLSQ fit is unhealthy
    if hasattr(nlsq_result, "is_healthy") and not nlsq_result.is_healthy:
        health_score = getattr(nlsq_result, "health_score", "N/A")
        logger.warning(
            f"NLSQ warm-start may be unreliable: health_score={health_score}"
        )

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # Run MCMC with warm-start
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        stretched_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params=nlsq_init,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names)


@log_timing(threshold_ms=2000)
def run_power_law_fit(
    q: ArrayLike,
    tau: ArrayLike | FitResult,
    stability: Literal["auto", "check", False] = "auto",
    auto_bounds: bool = False,
    **kwargs,
) -> FitResult:
    """Run power law fit with NLSQ warm-start.

    Parameters
    ----------
    q : array_like
        Q values
    tau : array_like or FitResult
        Relaxation times (or FitResult with tau samples)
    stability : str, optional
        NLSQ stability mode: 'auto', 'check', or False (default: 'auto')
    auto_bounds : bool, optional
        Use NLSQ auto-bounds inference (default: False)
    **kwargs
        Sampler configuration

    Returns
    -------
    FitResult
        Posterior samples for tau0, alpha
    """
    check_numpyro()

    q = np.asarray(q)

    # Handle FitResult input
    if isinstance(tau, FitResult):
        tau_values = tau.get_mean("tau")
        tau_err = tau.get_std("tau")
        tau = np.full(len(q), tau_values)  # Broadcast to Q array size
    else:
        tau = np.asarray(tau)
        tau_err = None

    config = _extract_config(kwargs)
    param_names = ["tau0", "alpha"]

    # NLSQ warm-start with NLSQ 0.6.0 features
    logger.info("Running NLSQ warm-start for power law fit")
    p0 = {"tau0": 1.0, "alpha": 2.0}
    bounds = {
        "tau0": (1e-6, 1e6),
        "alpha": (0.0, 10.0),
    }

    nlsq_result = nlsq_optimize(
        power_law_func,
        q,
        tau,
        tau_err,
        p0,
        bounds,
        stability=stability,
        auto_bounds=auto_bounds,
        compute_diagnostics=True,
    )
    nlsq_init = nlsq_result.params

    # Log warning if NLSQ fit is unhealthy
    if hasattr(nlsq_result, "is_healthy") and not nlsq_result.is_healthy:
        health_score = getattr(nlsq_result, "health_score", "N/A")
        logger.warning(
            f"NLSQ warm-start may be unreliable: health_score={health_score}"
        )

    # Convert to JAX arrays
    q_jax = jnp.asarray(q)
    tau_jax = jnp.asarray(tau)
    tau_err_jax = jnp.asarray(tau_err) if tau_err is not None else None

    # Run MCMC with warm-start
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        power_law_model,
        (q_jax, tau_jax, tau_err_jax),
        config,
        init_params=nlsq_init,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names)
