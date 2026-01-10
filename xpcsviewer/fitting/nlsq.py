"""JAX-accelerated NLSQ solver using the nlsq library.

This module provides nonlinear least squares optimization for warm-starting
the Bayesian MCMC sampler, using NLSQ 0.6.0 with property delegation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

from xpcsviewer.utils.log_utils import log_timing

from .results import NLSQResult
from .visualization import validate_pcov

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


@log_timing(threshold_ms=500)
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
    """NLSQ 0.6.0 curve fitting with native result delegation."""
    import nlsq

    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr) if yerr is not None else None

    param_names = list(p0.keys())
    p0_array = np.array([p0[name] for name in param_names])
    n_params = len(param_names)
    n_data = len(y)

    # Convert bounds to nlsq format
    lower = np.array([bounds.get(n, (-np.inf, np.inf))[0] for n in param_names])
    upper = np.array([bounds.get(n, (-np.inf, np.inf))[1] for n in param_names])

    try:
        native_result = nlsq.fit(
            model_fn,
            x,
            y,
            p0=p0_array,
            sigma=yerr,
            absolute_sigma=yerr is not None,
            bounds=(lower, upper),
            preset=preset,
            auto_bounds=auto_bounds,
            stability=stability,
            fallback=fallback,
            compute_diagnostics=compute_diagnostics,
            show_progress=show_progress,
        )
        popt = np.asarray(native_result.popt)
        pcov = np.asarray(native_result.pcov)
        converged = getattr(native_result, "success", True)
    except Exception as e:
        logger.warning(f"nlsq fitting failed: {e}, using initial guess")
        native_result = None
        popt = p0_array
        pcov = np.full((n_params, n_params), np.inf)
        converged = False

    # Compute chi-squared
    residuals = y - model_fn(x, *popt)
    dof = max(1, n_data - n_params)
    chi2 = (
        float(np.sum((residuals / yerr) ** 2) / dof)
        if yerr is not None
        else float(np.sum(residuals**2) / dof)
    )

    # Validate covariance
    pcov_valid, pcov_message = validate_pcov(pcov, param_names)

    return NLSQResult(
        params=dict(zip(param_names, popt)),
        chi_squared=chi2,
        converged=converged,
        pcov_valid=pcov_valid,
        pcov_message=pcov_message,
        native_result=native_result,
        _param_names=param_names,
    )
