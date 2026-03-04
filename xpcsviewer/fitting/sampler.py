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
    from numpyro.infer.initialization import init_to_value

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
    """Run MCMC sampling with optional warm-start initialization.

    Parameters
    ----------
    model : callable
        NumPyro model function
    model_args : tuple
        Positional arguments for the model
    config : SamplerConfig
        Sampler configuration
    init_params : dict, optional
        Initial parameter values in **constrained** (natural) space,
        e.g. from NLSQ warm-start. Passed to NumPyro's
        ``init_to_value`` which handles the constrained→unconstrained
        transform internally and falls back to ``init_to_uniform``
        for any sample sites not in ``init_params``.
    """
    check_numpyro()

    # Set random seed
    if config.random_seed is not None:
        seed = config.random_seed
        rng_key = jax.random.PRNGKey(seed)
        logger.info(f"MCMC PRNG seed (user-specified): {seed}")
    else:
        # Use time-based seed for non-deterministic runs (BUG-020).
        # BUG-E fix: Store actual seed in config for reproducibility.
        # Previously, the time-based seed was only logged but not stored,
        # so FitResult.to_dict() exported random_seed=null.
        seed = int(time.time_ns() % 2**31)
        rng_key = jax.random.PRNGKey(seed)
        logger.info(f"MCMC PRNG seed (time-based): {seed}")
        config.random_seed = seed

    # BUG-A fix: Use init_to_value() for warm-start initialization.
    # This accepts constrained (natural) values from NLSQ and handles
    # the transform to unconstrained space internally. It also falls
    # back to init_to_uniform for any sample sites not in init_params
    # (e.g. 'sigma' when yerr is None).
    #
    # Previously, constrained NLSQ values were passed directly as
    # init_params to mcmc.run(), but NUTS expects unconstrained values.
    # This caused NUTS to start at exp(value) instead of value for
    # LogNormal priors, making the warm-start ineffective.
    if init_params is not None:
        init_strategy = init_to_value(values=init_params)
    else:
        init_strategy = None  # Use NUTS default (init_to_uniform)

    # Configure NUTS sampler
    kernel_kwargs = {
        "target_accept_prob": config.target_accept_prob,
        "max_tree_depth": config.max_tree_depth,
    }
    if init_strategy is not None:
        kernel_kwargs["init_strategy"] = init_strategy
    kernel = NUTS(model, **kernel_kwargs)

    # Create MCMC instance — use vectorized (vmap) on single-device systems
    # to run chains in parallel, or pmap when multiple devices are available.
    chain_method = (
        "vectorized" if jax.local_device_count() < config.num_chains else "parallel"
    )
    logger.info(
        "MCMC config: warmup=%d, samples=%d, chains=%d, chain_method=%s",
        config.num_warmup,
        config.num_samples,
        config.num_chains,
        chain_method,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method=chain_method,
    )

    # BUG-D fix: Request num_steps to compute max_treedepth_reached.
    # Previously hardcoded to 0 with the comment "NumPyro doesn't track
    # this directly", but num_steps IS available via extra_fields.
    mcmc.run(rng_key, *model_args, extra_fields=("num_steps", "energy"))

    samples = mcmc.get_samples()
    first_key = next(iter(samples))
    n_draws = len(samples[first_key])
    logger.info(
        "MCMC completed: %d total draws (%d chains x %d samples), chain_method=%s",
        n_draws,
        config.num_chains,
        config.num_samples,
        chain_method,
    )
    return mcmc, samples


def compute_bfmi(arviz_data) -> float | None:
    """Compute BFMI from ArviZ DataTree.

    Parameters
    ----------
    arviz_data : DataTree
        ArviZ DataTree object from MCMC sampling

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
    # Convert samples to numpy, preserving param_names order so that
    # dict key iteration matches the model function's signature.
    samples_np = {k: np.asarray(samples[k]) for k in param_names if k in samples}

    # Convert to ArviZ DataTree first (needed for summary and BFMI)
    arviz_data = az.from_numpyro(mcmc)

    # Get diagnostics
    summary = az.summary(arviz_data, var_names=param_names, round_to="none")

    # Extract diagnostics
    r_hat = {}
    ess_bulk = {}
    ess_tail = {}

    for name in param_names:
        if name in summary.index:
            r_hat[name] = float(summary.loc[name, "r_hat"])
            ess_bulk[name] = int(summary.loc[name, "ess_bulk"])
            ess_tail[name] = int(summary.loc[name, "ess_tail"])

    # Count divergences
    extra = mcmc.get_extra_fields()
    num_divergent = int(np.sum(extra["diverging"]))

    # BUG-D fix: Compute max_treedepth_reached from num_steps.
    # Previously hardcoded to 0. NumPyro provides num_steps via
    # extra_fields — steps hitting 2^max_tree_depth - 1 indicate
    # the sampler exhausted its tree depth budget.
    max_treedepth_reached = 0
    if "num_steps" in extra and config is not None:
        max_steps = 2**config.max_tree_depth - 1
        max_treedepth_reached = int(np.sum(extra["num_steps"] >= max_steps))
        if max_treedepth_reached > 0:
            total_samples = len(extra["num_steps"])
            pct = 100.0 * max_treedepth_reached / max(total_samples, 1)
            msg = (
                f"{max_treedepth_reached}/{total_samples} samples "
                f"({pct:.1f}%) hit max_tree_depth={config.max_tree_depth}"
            )
            if pct > 1.0:
                logger.warning(f"{msg}. Consider increasing max_tree_depth.")
            else:
                logger.debug(msg)

    # Compute BFMI per Technical Guidelines
    bfmi = compute_bfmi(arviz_data)

    diagnostics = FitDiagnostics(
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        divergences=num_divergent,
        max_treedepth_reached=max_treedepth_reached,
        bfmi=bfmi,
    )

    return FitResult(
        samples=samples_np,
        param_names=param_names,
        summary=summary,
        diagnostics=diagnostics,
        nlsq_init=nlsq_init,
        arviz_data=arviz_data,
        config=config,
        x=x,
    )


@log_timing(threshold_ms=60_000)
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

    # Determine warm-start viability
    use_warm_start = nlsq_result.is_healthy
    if not use_warm_start:
        health_score = nlsq_result.health_score
        logger.warning(
            f"NLSQ warm-start unreliable (health_score={health_score}); "
            "falling back to init_to_uniform with boosted warmup"
        )
        config.num_warmup = int(config.num_warmup * 1.5)

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # Run MCMC with warm-start (or uniform init if NLSQ failed)
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        single_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params=nlsq_init if use_warm_start else None,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=x)


@log_timing(threshold_ms=60_000)
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

    # Determine warm-start viability
    use_warm_start = nlsq_result.is_healthy
    if not use_warm_start:
        health_score = nlsq_result.health_score
        logger.warning(
            f"NLSQ warm-start unreliable (health_score={health_score}); "
            "falling back to init_to_uniform with boosted warmup"
        )
        config.num_warmup = int(config.num_warmup * 1.5)

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    # Run MCMC with warm-start (or uniform init if NLSQ failed)
    logger.info("Running NUTS sampling")
    init_params = None
    if use_warm_start:
        # BUG-022: Sort tau1/tau2 before computing tau2_factor.
        # NLSQ may return tau1 > tau2 which would make tau2_factor negative,
        # causing invalid init params for the double_exp_model parameterization
        # (which enforces tau2 = tau1 * (1 + tau2_factor) with tau2_factor > 0).
        tau_vals = sorted([nlsq_init["tau1"], nlsq_init["tau2"]])
        tau1_sorted = tau_vals[0]
        tau2_sorted = tau_vals[1]
        # Clamp tau2_factor to avoid extreme values from NLSQ warm-start
        tau2_factor = max(0.01, min(tau2_sorted / tau1_sorted - 1, 1000.0))
        init_params = {
            "tau1": tau1_sorted,  # BUG-022: use sorted tau1 (always the smaller value)
            "tau2_factor": tau2_factor,
            "baseline": nlsq_init["baseline"],
            "contrast1": nlsq_init["contrast1"],
            "contrast2": nlsq_init["contrast2"],
        }

    mcmc, samples = _run_mcmc(
        double_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params=init_params,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=x)


@log_timing(threshold_ms=60_000)
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

    # Determine warm-start viability
    use_warm_start = nlsq_result.is_healthy
    if not use_warm_start:
        health_score = nlsq_result.health_score
        logger.warning(
            f"NLSQ warm-start unreliable (health_score={health_score}); "
            "falling back to init_to_uniform with boosted warmup"
        )
        config.num_warmup = int(config.num_warmup * 1.5)

    # Convert to JAX arrays
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    yerr_jax = jnp.asarray(yerr) if yerr is not None else None

    init_params = None
    if use_warm_start:
        # JAX-N-06: Clamp beta from NLSQ init to avoid boundary issues in NUTS.
        # Beta near 0 or 1 causes numerical instability in the stretched exp model.
        if "beta" in nlsq_init:
            nlsq_init["beta"] = max(0.05, min(0.95, nlsq_init["beta"]))
        init_params = nlsq_init

    # Run MCMC (uniform init if NLSQ failed, warm-start otherwise)
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        stretched_exp_model,
        (x_jax, y_jax, yerr_jax),
        config,
        init_params=init_params,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=x)


@log_timing(threshold_ms=60_000)
def run_power_law_fit(
    q: ArrayLike,
    tau: ArrayLike | FitResult,
    tau_err: ArrayLike | None = None,
    stability: Literal["auto", "check", False] = "auto",
    auto_bounds: bool = False,
    bounds: dict[str, tuple[float, float]] | None = None,
    **kwargs,
) -> FitResult:
    """Run power law fit with NLSQ warm-start.

    Parameters
    ----------
    q : array_like
        Q values
    tau : array_like or FitResult
        Relaxation times (or FitResult with tau samples)
    tau_err : array_like, optional
        Measurement uncertainties on tau values from G2 fitting
    stability : str, optional
        NLSQ stability mode: 'auto', 'check', or False (default: 'auto')
    auto_bounds : bool, optional
        Use NLSQ auto-bounds inference (default: False)
    bounds : dict, optional
        Override NLSQ warm-start bounds. Keys are parameter names
        (``"tau0"``, ``"alpha"``), values are ``(min, max)`` tuples.
        If ``None``, uses defaults: ``tau0=(1e-6, 1e6)``,
        ``alpha=(0.0, 10.0)``.
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
    if tau_err is not None:
        tau_err = np.asarray(tau_err)

    config = _extract_config(kwargs)
    param_names = ["tau0", "alpha"]

    # NLSQ warm-start with NLSQ 0.6.0 features
    logger.info("Running NLSQ warm-start for power law fit")
    default_bounds = {"tau0": (1e-6, 1e6), "alpha": (0.0, 10.0)}
    if bounds is not None:
        default_bounds.update(bounds)
    bounds = default_bounds

    # Data-driven initial guess via log-log linear regression.
    # power_law: tau = tau0 * q^(-alpha) → log(tau) = log(tau0) - alpha*log(q)
    pos = (q > 0) & (tau > 0)
    if np.sum(pos) >= 2:
        log_q = np.log(q[pos])
        log_tau = np.log(tau[pos])
        slope, intercept = np.polyfit(log_q, log_tau, 1)
        p0_alpha = float(np.clip(-slope, bounds["alpha"][0], bounds["alpha"][1]))
        p0_tau0 = float(
            np.clip(np.exp(intercept), bounds["tau0"][0], bounds["tau0"][1])
        )
        logger.info(
            "Power law p0 from log-log regression: tau0=%.3g, alpha=%.3f",
            p0_tau0,
            p0_alpha,
        )
    else:
        p0_tau0 = 1.0
        p0_alpha = float(np.clip(2.0, bounds["alpha"][0], bounds["alpha"][1]))

    p0 = {"tau0": p0_tau0, "alpha": p0_alpha}

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

    # Determine warm-start viability.
    # For sparse data (typical of power-law / diffusion fits), NLSQ
    # diagnostics are often unavailable (health_score=0), even when
    # the optimizer converged.  Accept warm-start when NLSQ converged
    # and chi² is finite, falling back only on true failure.
    use_warm_start = nlsq_result.is_healthy
    if (
        not use_warm_start
        and nlsq_result.converged
        and np.isfinite(nlsq_result.chi_squared)
    ):
        logger.info(
            "NLSQ diagnostics unavailable (health_score=%d) but fit converged "
            "(chi2=%.3g); accepting warm-start",
            nlsq_result.health_score,
            nlsq_result.chi_squared,
        )
        use_warm_start = True
    if not use_warm_start:
        health_score = nlsq_result.health_score
        logger.warning(
            f"NLSQ warm-start unreliable (health_score={health_score}, "
            f"converged={nlsq_result.converged}); "
            "falling back to init_to_uniform with boosted warmup"
        )
        config.num_warmup = int(config.num_warmup * 1.5)

    # Convert to JAX arrays
    q_jax = jnp.asarray(q)
    tau_jax = jnp.asarray(tau)
    tau_err_jax = jnp.asarray(tau_err) if tau_err is not None else None

    # Run MCMC (uniform init if NLSQ failed, warm-start otherwise)
    logger.info("Running NUTS sampling")
    mcmc, samples = _run_mcmc(
        power_law_model,
        (q_jax, tau_jax, tau_err_jax),
        config,
        init_params=nlsq_init if use_warm_start else None,
    )

    return _build_fit_result(mcmc, samples, nlsq_init, param_names, config=config, x=q)
