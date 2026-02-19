"""NumPyro NUTS sampler with NLSQ warm-start.

This module provides the MCMC sampling functionality using NumPyro's
NUTS sampler with JAX-accelerated NLSQ warm-start.
"""

from __future__ import annotations

import logging
import time
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

# Check availability of optional dependencies independently (T-10).
# Splitting the monolithic try/except allows partial functionality when only
# some packages are missing (e.g., arviz absent but jax+numpyro present).
JAX_AVAILABLE = False
NUMPYRO_AVAILABLE = False
ARVIZ_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    pass
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    NUMPYRO_AVAILABLE = True
except ImportError:
    pass
try:
    import arviz as az

    ARVIZ_AVAILABLE = True
except ImportError:
    pass


def check_numpyro() -> None:
    """Raise error if JAX, NumPyro, or ArviZ are not available."""
    missing = []
    if not JAX_AVAILABLE:
        missing.append("jax")
    if not NUMPYRO_AVAILABLE:
        missing.append("numpyro")
    if not ARVIZ_AVAILABLE:
        missing.append("arviz")
    if missing:
        raise ImportError(
            f"Bayesian fitting requires JAX, NumPyro, and ArviZ. "
            f"Missing: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
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
        seed = config.random_seed
        rng_key = jax.random.PRNGKey(seed)
        logger.info(f"MCMC PRNG seed (user-specified): {seed}")
    else:
        # Use time-based seed for non-deterministic runs (BUG-020).
        # Log the seed so results can be reproduced if needed.
        seed = int(time.time_ns() % 2**31)
        rng_key = jax.random.PRNGKey(seed)
        logger.info(f"MCMC PRNG seed (time-based): {seed}")

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
        # Convert init_params to JAX arrays with per-chain jitter (BUG-021).
        # Broadcasting identical values to all chains makes R-hat diagnostics
        # meaningless: chains start at the same point so R-hat is trivially 1.0.
        # Instead, add small Gaussian jitter (0.01 * normal) so each chain
        # starts from a distinct point while staying close to the warm-start.
        init_params_jax = {}
        # Split the rng_key into per-parameter subkeys for reproducible jitter
        param_keys = list(init_params.keys())
        subkeys = jax.random.split(rng_key, num=len(param_keys))
        for subkey, k in zip(subkeys, param_keys):
            v = init_params[k]
            val = jnp.array(v)
            if config.num_chains > 1:
                shape = (config.num_chains,) + val.shape
                # Add small Gaussian jitter per chain instead of broadcasting
                jitter = 0.01 * jax.random.normal(subkey, shape=shape)
                val = val + jitter
            init_params_jax[k] = val
        mcmc.run(rng_key, *model_args, init_params=init_params_jax)
    else:
        mcmc.run(rng_key, *model_args)

    return mcmc, mcmc.get_samples()


def compute_bfmi(arviz_data) -> float | None:
    """Compute BFMI from ArviZ InferenceData.

    Parameters
    ----------
    arviz_data : az.InferenceData
        InferenceData object from MCMC sampling

    Returns
    -------
    float | None
        Mean BFMI across chains, or None if computation fails

    Notes
    -----
    Uses az.bfmi() which returns per-chain values.
    Returns mean across all chains.
    Logs warning if BFMI < 0.2 per Technical Guidelines.
    """
    try:
        bfmi_values = az.bfmi(arviz_data)
        bfmi_mean = float(np.mean(bfmi_values))
        if bfmi_mean < 0.2:
            logger.warning(
                f"Low BFMI ({bfmi_mean:.3f}) indicates poor posterior exploration. "
                f"Consider reparameterization or increasing warmup."
            )
        return bfmi_mean
    except Exception as e:
        logger.warning(f"Failed to compute BFMI: {e}")
        return None


def _build_fit_result(
    mcmc: MCMC,
    samples: dict,
    nlsq_init: dict[str, float],
    param_names: list[str],
    config: SamplerConfig | None = None,
    x: np.ndarray | None = None,
) -> FitResult:
    """Build FitResult from MCMC output."""
    # Convert samples to numpy
    samples_np = {k: np.asarray(v) for k, v in samples.items() if k in param_names}

    # Convert to ArviZ InferenceData first (needed for summary and BFMI)
    arviz_data = az.from_numpyro(mcmc)

    # Get diagnostics
    summary = az.summary(arviz_data, var_names=param_names)

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

    # Compute BFMI per Technical Guidelines
    bfmi = compute_bfmi(arviz_data)

    diagnostics = FitDiagnostics(
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        divergences=num_divergent,
        max_treedepth_reached=0,  # NumPyro doesn't track this directly
        bfmi=bfmi,
    )

    return FitResult(
        samples=samples_np,
        summary=summary,
        diagnostics=diagnostics,
        nlsq_init=nlsq_init,
        arviz_data=arviz_data,
        config=config,
        x=x,
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
    if not nlsq_result.is_healthy:
        health_score = nlsq_result.health_score
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

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=x)


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
    if not nlsq_result.is_healthy:
        health_score = nlsq_result.health_score
        logger.warning(
            f"NLSQ warm-start may be unreliable: health_score={health_score}"
        )

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # Run MCMC with warm-start
    logger.info("Running NUTS sampling")
    # BUG-022: Sort tau1/tau2 before computing tau2_factor.
    # NLSQ may return tau1 > tau2 which would make tau2_factor negative,
    # causing invalid init params for the double_exp_model parameterization
    # (which enforces tau2 = tau1 * (1 + tau2_factor) with tau2_factor > 0).
    tau_vals = sorted([nlsq_init["tau1"], nlsq_init["tau2"]])
    tau1_sorted = tau_vals[0]
    tau2_sorted = tau_vals[1]
    # Clamp tau2_factor to avoid extreme values from NLSQ warm-start
    tau2_factor = max(0.01, min(tau2_sorted / tau1_sorted - 1, 1000.0))

    mcmc, samples = _run_mcmc(
        double_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params={
            "tau1": tau1_sorted,  # BUG-022: use sorted tau1 (always the smaller value)
            "tau2_factor": tau2_factor,
            "baseline": nlsq_init["baseline"],
            "contrast1": nlsq_init["contrast1"],
            "contrast2": nlsq_init["contrast2"],
        },
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=x)


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
    if not nlsq_result.is_healthy:
        health_score = nlsq_result.health_score
        logger.warning(
            f"NLSQ warm-start may be unreliable: health_score={health_score}"
        )

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # JAX-N-06: Clamp beta from NLSQ init to avoid boundary issues in NUTS.
    # Beta near 0 or 1 causes numerical instability in the stretched exp model.
    if "beta" in nlsq_init:
        nlsq_init["beta"] = max(0.05, min(0.95, nlsq_init["beta"]))

    # Run MCMC with warm-start
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        stretched_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params=nlsq_init,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=x)


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
        raise TypeError(
            "run_power_law_fit requires per-Q tau values as an array, "
            "not a single FitResult. Pass an array of tau values from "
            "individual per-Q fits instead."
        )

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
    if not nlsq_result.is_healthy:
        health_score = nlsq_result.health_score
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

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=q)
