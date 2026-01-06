"""JAX-accelerated NLSQ solver using optimistix.

This module provides nonlinear least squares optimization for warm-starting
the Bayesian MCMC sampler, using optimistix.LevenbergMarquardt for
JAX-accelerated fitting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .results import NLSQResult
from .visualization import validate_pcov

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


# Check if JAX/optimistix is available
try:
    import jax
    import jax.numpy as jnp
    import optimistix as optx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def nlsq_optimize(
    model_fn: Callable,
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None,
    p0: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> NLSQResult:
    """JAX-accelerated nonlinear least squares optimization.

    Uses optimistix.LevenbergMarquardt for NLSQ fitting.
    Falls back to scipy.optimize.curve_fit if JAX is unavailable.

    Parameters
    ----------
    model_fn : callable
        Model function: y = model_fn(x, *params)
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

    Returns
    -------
    NLSQResult
        Fitted parameters and covariance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)

    param_names = list(p0.keys())
    p0_array = np.array([p0[name] for name in param_names])

    if JAX_AVAILABLE:
        return _nlsq_jax(model_fn, x, y, yerr, param_names, p0_array, bounds)
    return _nlsq_scipy(model_fn, x, y, yerr, param_names, p0_array, bounds)


def _nlsq_jax(
    model_fn: Callable,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    param_names: list[str],
    p0: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> NLSQResult:
    """NLSQ fitting using optimistix.LevenbergMarquardt."""
    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    weights = 1.0 / jnp.asarray(yerr) if yerr is not None else jnp.ones_like(y_jax)

    # Extract bounds as numpy arrays (to avoid tracing issues)
    lower_bounds_np = np.array(
        [bounds.get(n, (-np.inf, np.inf))[0] for n in param_names]
    )
    upper_bounds_np = np.array(
        [bounds.get(n, (-np.inf, np.inf))[1] for n in param_names]
    )
    lower_bounds = jnp.asarray(lower_bounds_np)
    upper_bounds = jnp.asarray(upper_bounds_np)

    # Pre-compute bound flags OUTSIDE of traced function (static Python bools)
    lb_finite = np.isfinite(lower_bounds_np)
    ub_finite = np.isfinite(upper_bounds_np)
    both_finite = lb_finite & ub_finite
    only_lb_finite = lb_finite & ~ub_finite
    only_ub_finite = ~lb_finite & ub_finite

    # Transform parameters to unbounded space for optimization
    def to_unconstrained(params):
        """Transform bounded params to unbounded space."""
        # Use sigmoid-like transform for bounded params
        result = []
        for i in range(len(param_names)):
            p = params[i]
            lb, ub = lower_bounds[i], upper_bounds[i]
            if both_finite[i]:
                # Logit transform
                scaled = (p - lb) / (ub - lb)
                scaled = jnp.clip(scaled, 1e-6, 1 - 1e-6)
                result.append(jnp.log(scaled / (1 - scaled)))
            elif only_lb_finite[i]:
                # Log transform
                result.append(jnp.log(p - lb + 1e-6))
            elif only_ub_finite[i]:
                # Negative log transform
                result.append(-jnp.log(ub - p + 1e-6))
            else:
                result.append(p)
        return jnp.array(result)

    def from_unconstrained(unconstrained):
        """Transform back from unbounded space."""
        result = []
        for i in range(len(param_names)):
            u = unconstrained[i]
            lb, ub = lower_bounds[i], upper_bounds[i]
            if both_finite[i]:
                # Inverse logit
                sigmoid = 1 / (1 + jnp.exp(-u))
                result.append(lb + sigmoid * (ub - lb))
            elif only_lb_finite[i]:
                # Inverse log
                result.append(lb + jnp.exp(u))
            elif only_ub_finite[i]:
                # Inverse negative log
                result.append(ub - jnp.exp(-u))
            else:
                result.append(u)
        return jnp.array(result)

    # Define residual function for NLSQ
    def residual_fn(params_unconstrained, args):
        params = from_unconstrained(params_unconstrained)
        pred = model_fn(x_jax, *params)
        return (y_jax - pred) * weights

    # Initial guess in unconstrained space
    p0_unconstrained = to_unconstrained(jnp.asarray(p0))

    # Run Levenberg-Marquardt optimization
    try:
        solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        solution = optx.least_squares(
            residual_fn,
            solver,
            p0_unconstrained,
            args=None,
            max_steps=100,
        )

        popt_unconstrained = solution.value
        popt = from_unconstrained(popt_unconstrained)
        converged = True

    except Exception as e:
        logger.warning(f"optimistix NLSQ failed: {e}, using initial guess")
        popt = jnp.asarray(p0)
        converged = False

    # Convert back to numpy
    popt_np = np.asarray(popt)

    # Compute residuals
    y_fit = model_fn(x_jax, *popt)
    residuals = np.asarray(y_jax - y_fit)

    # Compute chi-squared
    if yerr is not None:
        chi2 = float(np.sum((residuals / yerr) ** 2) / (len(y) - len(p0)))
    else:
        chi2 = float(np.sum(residuals**2) / (len(y) - len(p0)))

    # Compute covariance via Jacobian
    pcov = _compute_covariance_jax(model_fn, x_jax, popt, yerr)
    pcov_np = np.asarray(pcov)

    # Validate covariance
    pcov_valid, pcov_message = validate_pcov(pcov_np, param_names)

    return NLSQResult(
        params=dict(zip(param_names, popt_np)),
        covariance=pcov_np,
        residuals=residuals,
        chi_squared=chi2,
        converged=converged,
        pcov_valid=pcov_valid,
        pcov_message=pcov_message,
    )


def _compute_covariance_jax(
    model_fn: Callable,
    x: Any,
    popt: Any,
    yerr: np.ndarray | None,
) -> Any:
    """Compute parameter covariance matrix via Jacobian.

    Uses JAX automatic differentiation for efficient Jacobian computation.
    """

    # Compute Jacobian using JAX
    def model_at_params(params):
        return model_fn(x, *params)

    jacobian = jax.jacfwd(model_at_params)(popt)

    # J is (n_points, n_params)
    # pcov = inv(J.T @ W @ J) where W is weight matrix
    if yerr is not None:
        weights = 1.0 / jnp.asarray(yerr) ** 2
        W = jnp.diag(weights)
        JTW = jacobian.T @ W
        hessian = JTW @ jacobian
    else:
        hessian = jacobian.T @ jacobian

    # Add regularization for numerical stability
    reg = 1e-10 * jnp.eye(hessian.shape[0])
    hessian_reg = hessian + reg

    try:
        pcov = jnp.linalg.inv(hessian_reg)
    except Exception:
        # If inversion fails, return inf covariance
        pcov = jnp.full_like(hessian_reg, jnp.inf)

    return pcov


def _nlsq_scipy(
    model_fn: Callable,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    param_names: list[str],
    p0: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> NLSQResult:
    """Fallback NLSQ fitting using scipy.optimize.curve_fit."""
    from scipy import optimize

    # Convert bounds to scipy format
    lower = [bounds.get(n, (-np.inf, np.inf))[0] for n in param_names]
    upper = [bounds.get(n, (-np.inf, np.inf))[1] for n in param_names]
    scipy_bounds = (lower, upper)

    try:
        popt, pcov = optimize.curve_fit(
            model_fn,
            x,
            y,
            p0=p0,
            sigma=yerr,
            bounds=scipy_bounds,
            absolute_sigma=True if yerr is not None else False,
            maxfev=1000,
        )
        converged = True
    except Exception as e:
        logger.warning(f"scipy curve_fit failed: {e}, using initial guess")
        popt = p0
        pcov = np.full((len(p0), len(p0)), np.inf)
        converged = False

    # Compute residuals
    y_fit = model_fn(x, *popt)
    residuals = y - y_fit

    # Compute chi-squared
    if yerr is not None:
        chi2 = float(np.sum((residuals / yerr) ** 2) / (len(y) - len(p0)))
    else:
        chi2 = float(np.sum(residuals**2) / (len(y) - len(p0)))

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
