"""JAX-accelerated NLSQ solver using the nlsq library.

This module provides nonlinear least squares optimization for warm-starting
the Bayesian MCMC sampler, using the nlsq library for JAX-accelerated fitting
with support for large datasets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    preset: str = "robust",
) -> NLSQResult:
    """JAX-accelerated nonlinear least squares optimization.

    Uses the nlsq library for JAX-accelerated NLSQ fitting with
    automatic handling of large datasets.

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
    preset : str, optional
        NLSQ preset configuration. Options:
        - 'fast': Single-start optimization for maximum speed
        - 'robust': Multi-start with 5 starts for robustness (default)
        - 'global': Thorough global search with 20 starts
        - 'streaming': Streaming optimization for large datasets
        - 'large': Auto-detect dataset size and use appropriate strategy

    Returns
    -------
    NLSQResult
        Fitted parameters and covariance
    """
    import nlsq

    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    param_names = list(p0.keys())
    p0_array = np.array([p0[name] for name in param_names])

    # Convert bounds to nlsq format: (lower_array, upper_array)
    lower = np.array([bounds.get(n, (-np.inf, np.inf))[0] for n in param_names])
    upper = np.array([bounds.get(n, (-np.inf, np.inf))[1] for n in param_names])
    nlsq_bounds = (lower, upper)

    try:
        # Use nlsq.fit with the specified preset
        result = nlsq.fit(
            model_fn,
            x,
            y,
            p0=p0_array,
            sigma=yerr,
            absolute_sigma=True if yerr is not None else False,
            bounds=nlsq_bounds,
            preset=preset,
        )

        # Extract results from OptimizeResult
        if hasattr(result, "popt"):
            # OptimizeResult object
            popt = np.asarray(result.popt)
            pcov = np.asarray(result.pcov)
            converged = result.success if hasattr(result, "success") else True
        else:
            # Tuple (popt, pcov)
            popt, pcov = result
            popt = np.asarray(popt)
            pcov = np.asarray(pcov)
            converged = True

    except Exception as e:
        logger.warning(f"nlsq fitting failed: {e}, using initial guess")
        popt = p0_array
        pcov = np.full((len(p0_array), len(p0_array)), np.inf)
        converged = False

    # Compute residuals
    y_fit = model_fn(x, *popt)
    residuals = y - np.asarray(y_fit)

    # Compute chi-squared
    dof = max(1, len(y) - len(p0_array))
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
    )
