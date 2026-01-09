"""JAX-accelerated NLSQ solver using the nlsq library.

This module provides nonlinear least squares optimization for warm-starting
the Bayesian MCMC sampler, using the nlsq library for JAX-accelerated fitting
with support for large datasets.

NLSQ 0.6.0 Features:
- Automatic bounds inference (auto_bounds)
- Numerical stability checks and fixes (stability)
- Fallback strategies for difficult problems (fallback)
- Model health diagnostics (compute_diagnostics)
- Enhanced statistical metrics (R², AIC, BIC, RMSE, MAE)
- Confidence intervals for parameters
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

from .results import NLSQResult
from .visualization import validate_pcov

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def nlsq_optimize(
    model_fn: Callable,
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None,
    p0: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    preset: Literal["fast", "robust", "global", "streaming", "large"] = "robust",
    *,
    auto_bounds: bool = False,
    stability: Literal["auto", "check", False] = False,
    fallback: bool = False,
    compute_diagnostics: bool = False,
    show_progress: bool = False,
) -> NLSQResult:
    """JAX-accelerated nonlinear least squares optimization.

    Uses the nlsq library for JAX-accelerated NLSQ fitting with
    automatic handling of large datasets and enhanced NLSQ 0.6.0 features.

    Parameters
    ----------
    model_fn : callable
        Model function: y = model_fn(x, *params). Must use jax.numpy operations.
    x : array_like
        Independent variable
    y : array_like
        Dependent variable
    yerr : array_like or None
        Measurement uncertainties
    p0 : dict
        Initial parameter guess {name: value}
    bounds : dict
        Parameter bounds {name: (min, max)}
    preset : {'fast', 'robust', 'global', 'streaming', 'large'}, optional
        NLSQ preset configuration:

        - 'fast': Single-start optimization for maximum speed
        - 'robust': Multi-start with 5 starts for robustness (default)
        - 'global': Thorough global search with 20 starts
        - 'streaming': Streaming optimization for large datasets
        - 'large': Auto-detect dataset size and use appropriate strategy

    auto_bounds : bool, optional
        Enable automatic parameter bounds inference from data characteristics.
        Inferred bounds are merged with user-provided bounds. Default: False.
    stability : {'auto', 'check', False}, optional
        Control numerical stability checks:

        - 'auto': Check and automatically apply fixes (rescale, normalize)
        - 'check': Check and warn, but don't apply fixes
        - False: Skip stability checks (default)

    fallback : bool, optional
        Enable automatic fallback strategies for difficult problems.
        Tries alternative methods, perturbed guesses, relaxed tolerances.
        Default: False.
    compute_diagnostics : bool, optional
        Compute model health diagnostics including identifiability analysis,
        parameter sensitivity, and gradient health. Default: False.
    show_progress : bool, optional
        Display progress bar for long operations. Default: False.

    Returns
    -------
    NLSQResult
        Fitted parameters, covariance, and enhanced statistical metrics including:

        - r_squared: Coefficient of determination (see note below)
        - adj_r_squared: Adjusted R² for model comparison
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
        - confidence_intervals: 95% CI for each parameter
        - predictions: Model predictions at input x values
        - model_diagnostics: Health diagnostics (if compute_diagnostics=True)

    Notes
    -----
    **R² with measurement uncertainties**: When ``yerr`` is provided, NLSQ 0.6.0
    computes R² using weighted residuals. This can result in negative R² values
    if the weighted fit is worse than the weighted mean. For weighted fits, use
    ``chi_squared`` (reduced chi-squared should be ~1 for a good fit) rather
    than R² to assess fit quality.

    Examples
    --------
    Basic usage with default settings:

    >>> result = nlsq_optimize(model_fn, x, y, yerr, p0, bounds)
    >>> print(f"R² = {result.r_squared:.4f}")
    >>> print(result.summary())

    Robust fitting with stability checks and fallbacks:

    >>> result = nlsq_optimize(
    ...     model_fn, x, y, yerr, p0, bounds,
    ...     preset='robust',
    ...     stability='auto',
    ...     fallback=True,
    ...     compute_diagnostics=True,
    ... )

    Large dataset with progress bar:

    >>> result = nlsq_optimize(
    ...     model_fn, big_x, big_y, big_yerr, p0, bounds,
    ...     preset='large',
    ...     show_progress=True,
    ... )
    """
    import nlsq

    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    param_names = list(p0.keys())
    p0_array = np.array([p0[name] for name in param_names])
    n_params = len(param_names)
    n_data = len(y)

    # Convert bounds to nlsq format: (lower_array, upper_array)
    lower = np.array([bounds.get(n, (-np.inf, np.inf))[0] for n in param_names])
    upper = np.array([bounds.get(n, (-np.inf, np.inf))[1] for n in param_names])
    nlsq_bounds = (lower, upper)

    # Initialize default values for error case
    r_squared = 0.0
    adj_r_squared = 0.0
    rmse = 0.0
    mae = 0.0
    aic = 0.0
    bic = 0.0
    confidence_intervals: dict[str, tuple[float, float]] = {}
    predictions = None
    model_diagnostics = None

    try:
        # Use nlsq.fit with the specified preset and new 0.6.0 features
        result = nlsq.fit(
            model_fn,
            x,
            y,
            p0=p0_array,
            sigma=yerr,
            absolute_sigma=yerr is not None,
            bounds=nlsq_bounds,
            preset=preset,
            auto_bounds=auto_bounds,
            stability=stability,
            fallback=fallback,
            compute_diagnostics=compute_diagnostics,
            show_progress=show_progress,
        )

        # Extract results from CurveFitResult object
        if hasattr(result, "popt"):
            # CurveFitResult object (NLSQ 0.6.0)
            popt = np.asarray(result.popt)
            pcov = np.asarray(result.pcov)
            converged = result.success if hasattr(result, "success") else True

            # Extract enhanced metrics from CurveFitResult using getattr with defaults
            r_squared = float(getattr(result, "r_squared", 0.0))
            adj_r_squared = float(getattr(result, "adj_r_squared", 0.0))
            rmse = float(getattr(result, "rmse", 0.0))
            mae = float(getattr(result, "mae", 0.0))
            aic = float(getattr(result, "aic", 0.0))
            bic = float(getattr(result, "bic", 0.0))

            # Extract residuals from result (NLSQ 0.6.0 built-in)
            if hasattr(result, "residuals"):
                residuals = np.asarray(result.residuals)
            else:
                # Fallback to manual computation
                y_fit = model_fn(x, *popt)
                residuals = y - np.asarray(y_fit)

            # Extract predictions
            if hasattr(result, "predictions"):
                predictions = np.asarray(result.predictions)
            else:
                predictions = np.asarray(model_fn(x, *popt))

            # Extract confidence intervals (95% level)
            if hasattr(result, "confidence_intervals"):
                try:
                    ci_result = result.confidence_intervals(alpha=0.95)
                    if ci_result is not None:
                        ci_array = np.asarray(ci_result)
                        for i, name in enumerate(param_names):
                            if i < len(ci_array):
                                confidence_intervals[name] = (
                                    float(ci_array[i, 0]),
                                    float(ci_array[i, 1]),
                                )
                except Exception as ci_err:
                    logger.debug(f"Could not extract confidence intervals: {ci_err}")

            # Extract model diagnostics
            if compute_diagnostics and hasattr(result, "diagnostics"):
                diagnostics_obj = result.diagnostics
                if diagnostics_obj is not None:
                    model_diagnostics = {
                        "available": True,
                        "diagnostics": str(diagnostics_obj),
                    }
                    # Try to extract specific diagnostic fields
                    for attr in [
                        "condition_number",
                        "parameter_sensitivity",
                        "gradient_health",
                        "identifiability",
                    ]:
                        if hasattr(diagnostics_obj, attr):
                            val = getattr(diagnostics_obj, attr)
                            if val is not None:
                                model_diagnostics[attr] = (
                                    float(val)
                                    if isinstance(val, (int, float))
                                    else str(val)
                                )

        else:
            # Tuple (popt, pcov) - legacy return type
            popt, pcov = result
            popt = np.asarray(popt)
            pcov = np.asarray(pcov)
            converged = True

            # Manual computation for legacy return type
            y_fit = model_fn(x, *popt)
            predictions = np.asarray(y_fit)
            residuals = y - predictions

            # Compute R² manually
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot > 0:
                r_squared = 1.0 - ss_res / ss_tot
                adj_r_squared = 1.0 - (1 - r_squared) * (n_data - 1) / max(
                    1, n_data - n_params - 1
                )

            # Compute RMSE and MAE
            rmse = float(np.sqrt(np.mean(residuals**2)))
            mae = float(np.mean(np.abs(residuals)))

            # Compute AIC and BIC (assuming Gaussian errors)
            rss = float(ss_res)
            if rss > 0:
                log_likelihood = -n_data / 2 * (np.log(2 * np.pi * rss / n_data) + 1)
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(n_data) - 2 * log_likelihood

    except Exception as e:
        logger.warning(f"nlsq fitting failed: {e}, using initial guess")
        popt = p0_array
        pcov = np.full((n_params, n_params), np.inf)
        converged = False

        # Compute residuals with initial guess
        y_fit = model_fn(x, *popt)
        predictions = np.asarray(y_fit)
        residuals = y - predictions

    # Compute reduced chi-squared
    dof = max(1, n_data - n_params)
    if yerr is not None:
        chi2 = float(np.sum((residuals / yerr) ** 2) / dof)
    else:
        chi2 = float(np.sum(residuals**2) / dof)

    # Validate covariance
    pcov_valid, pcov_message = validate_pcov(pcov, param_names)

    return NLSQResult(
        params=dict(zip(param_names, popt)),
        covariance=pcov,
        residuals=residuals,
        chi_squared=chi2,
        converged=converged,
        pcov_valid=pcov_valid,
        pcov_message=pcov_message,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        rmse=rmse,
        mae=mae,
        aic=aic,
        bic=bic,
        confidence_intervals=confidence_intervals,
        predictions=predictions,
        model_diagnostics=model_diagnostics,
    )
