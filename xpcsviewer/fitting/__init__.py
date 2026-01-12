"""Bayesian fitting module for XPCS correlation analysis.

This module provides Bayesian parameter estimation using NumPyro NUTS
sampler with JAX-accelerated NLSQ warm-start.

NLSQ 0.6.0 Enhanced Features:
    - R², adjusted R², RMSE, MAE, AIC, BIC metrics on NLSQResult
    - Confidence intervals for parameters
    - Prediction intervals (accounting for observation noise)
    - Automatic bounds inference (auto_bounds)
    - Numerical stability checks (stability)
    - Fallback strategies for difficult problems (fallback)
    - Model health diagnostics (compute_diagnostics)

Public API:
    fit_single_exp(x, y, yerr=None, **kwargs) -> FitResult
    fit_double_exp(x, y, yerr=None, **kwargs) -> FitResult
    fit_stretched_exp(x, y, yerr=None, **kwargs) -> FitResult
    fit_power_law(q, tau, **kwargs) -> FitResult
    nlsq_fit(model_fn, x, y, yerr, p0, bounds, **kwargs) -> NLSQResult
    SamplerConfig
    FitResult
    NLSQResult
    FitDiagnostics

Models:
    Single exponential: y = baseline + contrast * exp(-2 * x / tau)
    Double exponential: y = baseline + c1*exp(-2x/tau1) + c2*exp(-2x/tau2)
    Stretched exponential: y = baseline + contrast * exp(-(2x/tau)^beta)
    Power law: tau = tau0 * q^(-alpha)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .results import FitResult, NLSQResult


def fit_single_exp(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    **kwargs,
) -> FitResult:
    """Fit single exponential decay model with Bayesian inference.

    Model: y = baseline + contrast * exp(-2 * x / tau)

    Parameters
    ----------
    x : array_like
        Delay times
    y : array_like
        G2 correlation values
    yerr : array_like, optional
        Measurement uncertainties
    **kwargs
        Sampler configuration (see SamplerConfig)

    Returns
    -------
    FitResult
        Posterior samples for tau, baseline, contrast
    """
    from .sampler import run_single_exp_fit

    return run_single_exp_fit(x, y, yerr, **kwargs)


def fit_double_exp(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    **kwargs,
) -> FitResult:
    """Fit double exponential decay model with Bayesian inference.

    Model: y = baseline + c1*exp(-2x/tau1) + c2*exp(-2x/tau2)

    Parameters
    ----------
    x : array_like
        Delay times
    y : array_like
        G2 correlation values
    yerr : array_like, optional
        Measurement uncertainties
    **kwargs
        Sampler configuration (see SamplerConfig)

    Returns
    -------
    FitResult
        Posterior samples for tau1, tau2, baseline, contrast1, contrast2
    """
    from .sampler import run_double_exp_fit

    return run_double_exp_fit(x, y, yerr, **kwargs)


def fit_stretched_exp(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None = None,
    **kwargs,
) -> FitResult:
    """Fit stretched exponential (KWW) model with Bayesian inference.

    Model: y = baseline + contrast * exp(-(2 * x / tau)^beta)

    Parameters
    ----------
    x : array_like
        Delay times
    y : array_like
        G2 correlation values
    yerr : array_like, optional
        Measurement uncertainties
    **kwargs
        Sampler configuration (see SamplerConfig)

    Returns
    -------
    FitResult
        Posterior samples for tau, baseline, contrast, beta
    """
    from .sampler import run_stretched_exp_fit

    return run_stretched_exp_fit(x, y, yerr, **kwargs)


def fit_power_law(
    q: ArrayLike,
    tau: ArrayLike | FitResult,
    **kwargs,
) -> FitResult:
    """Fit power law Q-dependence of relaxation time.

    Model: tau = tau0 * q^(-alpha)

    Parameters
    ----------
    q : array_like
        Q values
    tau : array_like or FitResult
        Relaxation times (or FitResult with tau samples)
    **kwargs
        Sampler configuration (see SamplerConfig)

    Returns
    -------
    FitResult
        Posterior samples for tau0, alpha
    """
    from .sampler import run_power_law_fit

    return run_power_law_fit(q, tau, **kwargs)


def nlsq_fit(
    model_fn,
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None,
    p0: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    preset: str = "robust",
    *,
    auto_bounds: bool = False,
    stability: str | bool = False,
    fallback: bool = False,
    compute_diagnostics: bool = False,
    show_progress: bool = False,
) -> NLSQResult:
    """JAX-accelerated nonlinear least squares with NLSQ 0.6.0 features.

    Parameters
    ----------
    model_fn : callable
        Model function taking x and parameter values. Uses JAX operations.
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
        NLSQ preset configuration (default: 'robust'):

        - 'fast': Single-start for speed
        - 'robust': Multi-start with 5 starts
        - 'global': Thorough search with 20 starts
        - 'streaming': For large datasets
        - 'large': Auto-detect dataset size

    auto_bounds : bool, optional
        Enable automatic bounds inference (default: False)
    stability : {'auto', 'check', False}, optional
        Numerical stability checks (default: False):

        - 'auto': Check and apply fixes
        - 'check': Check and warn only
        - False: Skip checks

    fallback : bool, optional
        Enable fallback strategies for difficult problems (default: False)
    compute_diagnostics : bool, optional
        Compute model health diagnostics (default: False)
    show_progress : bool, optional
        Display progress bar (default: False)

    Returns
    -------
    NLSQResult
        Enhanced result with R², RMSE, AIC, BIC, confidence intervals,
        predictions, and optional model diagnostics.

    Examples
    --------
    Basic usage:

    >>> result = nlsq_fit(model_fn, x, y, yerr, p0, bounds)
    >>> print(f"R² = {result.r_squared:.4f}")
    >>> print(result.summary())

    Robust fitting with diagnostics:

    >>> result = nlsq_fit(
    ...     model_fn, x, y, yerr, p0, bounds,
    ...     preset='robust',
    ...     stability='auto',
    ...     fallback=True,
    ...     compute_diagnostics=True,
    ... )
    """
    from typing import Any, cast

    from .nlsq import nlsq_optimize

    return nlsq_optimize(
        model_fn,
        x,
        y,
        yerr,
        p0,
        bounds,
        preset=cast(Any, preset),
        auto_bounds=auto_bounds,
        stability=cast(Any, stability),
        fallback=fallback,
        compute_diagnostics=compute_diagnostics,
        show_progress=show_progress,
    )


# =============================================================================
# Gradient-Based Optimization API (FR-011: Auto-differentiation)
# =============================================================================


def value_and_grad(func, argnums=0):
    """Create a function that computes both value and gradient.

    Wraps JAX's value_and_grad for user-defined objective functions.
    The wrapped function returns (value, gradient) tuple.

    Parameters
    ----------
    func : callable
        Differentiable function that returns a scalar
    argnums : int or tuple of int, optional
        Arguments to differentiate with respect to (default: 0)

    Returns
    -------
    callable
        Function returning (value, gradient) tuple

    Raises
    ------
    RuntimeError
        If JAX backend is not available

    Examples
    --------
    >>> def loss(params, x, y):
    ...     pred = params[0] * x + params[1]
    ...     return jnp.sum((pred - y) ** 2)
    >>> val_grad = value_and_grad(loss)
    >>> value, gradient = val_grad(params, x, y)
    """
    from xpcsviewer.backends import get_backend

    backend = get_backend()
    if backend.name != "jax":
        raise RuntimeError(
            "value_and_grad requires JAX backend. Set XPCS_USE_JAX=1 to enable."
        )

    return backend.value_and_grad(func, argnums=argnums)


def grad(func, argnums=0):
    """Create a gradient function for a scalar-valued function.

    Wraps JAX's grad for user-defined objective functions.

    Parameters
    ----------
    func : callable
        Differentiable function that returns a scalar
    argnums : int or tuple of int, optional
        Arguments to differentiate with respect to (default: 0)

    Returns
    -------
    callable
        Function that computes gradients

    Raises
    ------
    RuntimeError
        If JAX backend is not available

    Examples
    --------
    >>> def loss(params, x, y):
    ...     pred = params[0] * x + params[1]
    ...     return jnp.sum((pred - y) ** 2)
    >>> gradient_fn = grad(loss)
    >>> gradient = gradient_fn(params, x, y)
    """
    from xpcsviewer.backends import get_backend

    backend = get_backend()
    if backend.name != "jax":
        raise RuntimeError("grad requires JAX backend. Set XPCS_USE_JAX=1 to enable.")

    return backend.grad(func, argnums=argnums)


def minimize_with_grad(
    objective,
    initial_params,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    learning_rate: float = 0.01,
):
    """Minimize objective function using gradient descent.

    Simple gradient descent optimizer for user-defined differentiable
    objective functions. For more sophisticated optimization, consider
    using optimistix or scipy.optimize.

    Parameters
    ----------
    objective : callable
        Differentiable objective function: f(params) -> scalar
    initial_params : array_like
        Initial parameter values
    max_iterations : int, optional
        Maximum iterations (default: 500)
    tolerance : float, optional
        Convergence tolerance for loss change (default: 1e-8)
    learning_rate : float, optional
        Learning rate / step size (default: 0.01)

    Returns
    -------
    tuple
        (optimal_params, diagnostics_dict) where diagnostics contains:
        - iterations: Number of iterations performed
        - losses: Array of loss values at each iteration
        - converged: Whether optimization converged
        - final_loss: Final loss value

    Raises
    ------
    RuntimeError
        If JAX backend is not available

    Examples
    --------
    >>> def loss(params):
    ...     return jnp.sum((params - target) ** 2)
    >>> optimal, diag = minimize_with_grad(loss, initial_guess)
    >>> print(f"Converged: {diag['converged']}, Loss: {diag['final_loss']}")
    """
    from xpcsviewer.simplemask.calibration import minimize_with_grad as _minimize

    return _minimize(
        objective,
        initial_params,
        max_iterations=max_iterations,
        tolerance=tolerance,
        learning_rate=learning_rate,
    )


# Re-export legacy fitting functions for xpcs_file.py compatibility
from .legacy import (
    double_exp,
    double_exp_all,
    fit_with_fixed,
    fit_with_fixed_parallel,
    fit_with_fixed_sequential,
    robust_curve_fit,
    sequential_fitting,
    single_exp,
    single_exp_all,
    vectorized_parameter_estimation,
    vectorized_residual_analysis,
)

# Re-export public classes
from .results import FitDiagnostics, FitResult, NLSQResult, SamplerConfig

# Re-export visualization functions (FR-013 to FR-021, NLSQ 0.6.0)
from .visualization import (
    PUBLICATION_STYLE,
    apply_publication_style,
    compute_prediction_interval,
    compute_uncertainty_band,
    generate_arviz_diagnostics,
    plot_comparison,
    plot_nlsq_fit,
    plot_posterior_predictive,
    save_figure,
    validate_pcov,
)

__all__ = [
    # Fitting functions
    "fit_single_exp",
    "fit_double_exp",
    "fit_stretched_exp",
    "fit_power_law",
    "nlsq_fit",
    # Result classes
    "SamplerConfig",
    "FitResult",
    "NLSQResult",
    "FitDiagnostics",
    # Gradient-based optimization API (FR-011)
    "grad",
    "value_and_grad",
    "minimize_with_grad",
    # Visualization functions (FR-013 to FR-021, NLSQ 0.6.0)
    "PUBLICATION_STYLE",
    "apply_publication_style",
    "validate_pcov",
    "compute_uncertainty_band",
    "compute_prediction_interval",
    "plot_nlsq_fit",
    "generate_arviz_diagnostics",
    "plot_posterior_predictive",
    "plot_comparison",
    "save_figure",
    # Legacy fitting functions
    "single_exp",
    "double_exp",
    "single_exp_all",
    "double_exp_all",
    "fit_with_fixed",
    "fit_with_fixed_parallel",
    "fit_with_fixed_sequential",
    "robust_curve_fit",
    "sequential_fitting",
    "vectorized_parameter_estimation",
    "vectorized_residual_analysis",
]
