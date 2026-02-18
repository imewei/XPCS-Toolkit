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
    """Perform non-linear least-squares curve fitting via NLSQ 0.6.0.

    Wraps ``nlsq.fit`` with automatic array conversion, covariance
    validation, and chi-squared computation. On failure, returns a
    sentinel ``NLSQResult`` with ``converged=False`` instead of
    raising.

    Args:
        model_fn: Model callable ``f(x, *params) -> y``.
        x: Independent variable array.
        y: Observed data array.
        yerr: Per-point standard errors (optional). When provided,
            ``absolute_sigma=True`` is passed to the solver.
        p0: Initial parameter guesses as ``{name: value}``.
        bounds: Parameter bounds as ``{name: (lower, upper)}``.
        preset: NLSQ solver preset. One of ``"fast"``,
            ``"robust"``, ``"global"``, ``"streaming"``, ``"large"``.
        auto_bounds: If True, let NLSQ infer bounds from data.
        stability: Numerical stability mode. ``"auto"`` applies
            automatic fixes, ``"check"`` only diagnoses, ``False``
            disables.
        fallback: Enable automatic fallback strategies on failure.
        compute_diagnostics: Populate ``ModelHealthReport``
            diagnostics in the native result.
        show_progress: Display a progress bar during fitting.

    Returns:
        NLSQResult: Fitting result containing parameter estimates,
        chi-squared, covariance status, and the native NLSQ result
        object.

    Example:
        >>> result = nlsq_optimize(
        ...     model_fn=lambda x, a, b: a * np.exp(-x / b),
        ...     x=delay, y=g2, yerr=g2_err,
        ...     p0={"a": 0.3, "b": 1.0},
        ...     bounds={"a": (0, 1), "b": (1e-6, 1e6)},
        ...     preset="robust",
        ... )
        >>> print(f"converged={result.converged}, chi2={result.chi_squared:.3f}")
    """
    import nlsq

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yerr = np.asarray(yerr, dtype=np.float64) if yerr is not None else None

    param_names = list(p0.keys())
    p0_array = np.array([p0[name] for name in param_names])
    n_params = len(param_names)
    n_data = len(y)

    # Convert bounds to nlsq format
    lower = np.array([bounds.get(n, (-np.inf, np.inf))[0] for n in param_names])
    upper = np.array([bounds.get(n, (-np.inf, np.inf))[1] for n in param_names])

    try:
        from typing import Any

        native_result: Any = nlsq.fit(
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
        converged = getattr(native_result, "success", False)  # BUG-017: default False
    except Exception as e:
        # Return a sentinel NLSQResult so callers can detect fitting failure.
        # Do NOT silently return p0 as though it were a real result (BUG-003).
        logger.warning(f"nlsq fitting failed: {e}; returning fallback sentinel")
        return NLSQResult(
            params=dict(zip(param_names, p0_array.tolist())),
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            pcov_valid=False,
            pcov_message=f"Fitting failed: {e}",
            native_result=None,
            _param_names=param_names,
            _r_squared=float("nan"),
            _adj_r_squared=float("nan"),
            _rmse=float("nan"),
            _mae=float("nan"),
            _aic=float("nan"),
            _bic=float("nan"),
        )

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
