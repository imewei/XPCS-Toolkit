"""JAX replacements for scipy.optimize functions using optimistix and optax.

This module provides JAX-compatible implementations of scipy.optimize
functions, using optimistix for root-finding and minimization, and
optax for gradient-based optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class OptimizeResult:
    """Container for optimization result.

    Compatible with scipy.optimize.OptimizeResult.

    Attributes
    ----------
    x : ndarray
        Solution vector
    fun : float
        Function value at solution
    success : bool
        Whether optimization converged
    message : str
        Description of termination
    nfev : int
        Number of function evaluations
    nit : int
        Number of iterations
    jac : ndarray, optional
        Jacobian at solution (if available)
    """

    x: np.ndarray
    fun: float
    success: bool
    message: str
    nfev: int = 0
    nit: int = 0
    jac: np.ndarray | None = None


def minimize(
    fun: Callable,
    x0: ArrayLike,
    args: tuple = (),
    method: str | None = None,
    jac: Callable | bool | None = None,
    bounds: list[tuple] | None = None,
    tol: float | None = None,
    options: dict | None = None,
) -> OptimizeResult:
    """Minimize a scalar function using optimistix/optax.

    Parameters
    ----------
    fun : callable
        Objective function to minimize
    x0 : array-like
        Initial guess
    args : tuple
        Extra arguments passed to fun
    method : str, optional
        Optimization method: 'BFGS', 'L-BFGS-B', 'CG', 'Adam', 'SGD'
    jac : callable or bool, optional
        Jacobian function or True for auto-diff
    bounds : list of tuple, optional
        Bounds for each variable
    tol : float, optional
        Tolerance for termination
    options : dict, optional
        Solver-specific options

    Returns
    -------
    OptimizeResult
        Optimization result container
    """
    options = options or {}
    tol = tol or 1e-8
    maxiter = options.get("maxiter", 1000)

    # Try optimistix first, fall back to scipy
    try:
        return _minimize_optimistix(fun, x0, args, method, jac, bounds, tol, maxiter)
    except ImportError:
        return _minimize_scipy(fun, x0, args, method, jac, bounds, tol, options)


def _minimize_optimistix(
    fun: Callable,
    x0: ArrayLike,
    args: tuple,
    method: str | None,
    jac: Callable | bool | None,
    bounds: list[tuple] | None,
    tol: float,
    maxiter: int,
) -> OptimizeResult:
    """Minimization using optimistix."""
    import jax
    import jax.numpy as jnp
    import optimistix as optx

    x0 = jnp.asarray(x0)

    # Wrap function with args
    def objective(x, _):
        if args:
            return fun(x, *args)
        return fun(x)

    # Select solver based on method
    if method is None or method.upper() in ("BFGS", "L-BFGS-B"):
        solver = optx.BFGS(rtol=tol, atol=tol)
    elif method.upper() == "CG":
        solver = optx.NonlinearCG(rtol=tol, atol=tol)
    elif method.upper() in ("ADAM", "SGD"):
        # Use gradient descent with optax
        return _minimize_optax(fun, x0, args, method, tol, maxiter)
    else:
        solver = optx.BFGS(rtol=tol, atol=tol)

    # Run optimization
    try:
        sol = optx.minimise(
            objective,
            solver,
            x0,
            max_steps=maxiter,
        )
        return OptimizeResult(
            x=np.asarray(sol.value),
            fun=float(objective(sol.value, None)),
            success=True,
            message="Optimization converged",
            nit=int(sol.stats.get("num_steps", 0)) if hasattr(sol, "stats") else 0,
        )
    except Exception as e:
        return OptimizeResult(
            x=np.asarray(x0),
            fun=float(objective(x0, None)),
            success=False,
            message=str(e),
        )


def _minimize_optax(
    fun: Callable,
    x0: ArrayLike,
    args: tuple,
    method: str,
    tol: float,
    maxiter: int,
) -> OptimizeResult:
    """Minimization using optax gradient descent."""
    import jax
    import jax.numpy as jnp
    import optax

    x0 = jnp.asarray(x0)

    # Wrap function with args
    def objective(x):
        if args:
            return fun(x, *args)
        return fun(x)

    # Select optimizer
    if method.upper() == "ADAM":
        optimizer = optax.adam(learning_rate=0.01)
    elif method.upper() == "SGD":
        optimizer = optax.sgd(learning_rate=0.01)
    else:
        optimizer = optax.adam(learning_rate=0.01)

    # Initialize optimizer state
    opt_state = optimizer.init(x0)
    x = x0

    # Optimization loop
    grad_fn = jax.grad(objective)

    for i in range(maxiter):
        grads = grad_fn(x)
        updates, opt_state = optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)

        # Check convergence
        grad_norm = float(jnp.linalg.norm(grads))
        if grad_norm < tol:
            return OptimizeResult(
                x=np.asarray(x),
                fun=float(objective(x)),
                success=True,
                message=f"Converged after {i + 1} iterations",
                nit=i + 1,
            )

    return OptimizeResult(
        x=np.asarray(x),
        fun=float(objective(x)),
        success=False,
        message=f"Maximum iterations ({maxiter}) reached",
        nit=maxiter,
    )


def _minimize_scipy(
    fun: Callable,
    x0: ArrayLike,
    args: tuple,
    method: str | None,
    jac: Callable | bool | None,
    bounds: list[tuple] | None,
    tol: float,
    options: dict,
) -> OptimizeResult:
    """Fallback to scipy.optimize.minimize."""
    from scipy.optimize import minimize as scipy_minimize

    result = scipy_minimize(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        bounds=bounds,
        tol=tol,
        options=options,
    )
    return OptimizeResult(
        x=result.x,
        fun=result.fun,
        success=result.success,
        message=result.message,
        nfev=result.nfev if hasattr(result, "nfev") else 0,
        nit=result.nit if hasattr(result, "nit") else 0,
        jac=result.jac if hasattr(result, "jac") else None,
    )


def curve_fit(
    f: Callable,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    bounds: tuple | None = None,
    maxfev: int = 1000,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear least squares curve fitting using optimistix.

    Parameters
    ----------
    f : callable
        Model function: f(x, *params)
    xdata : array-like
        Independent variable
    ydata : array-like
        Dependent variable (data to fit)
    p0 : array-like, optional
        Initial guess for parameters
    sigma : array-like, optional
        Uncertainties in ydata
    absolute_sigma : bool
        If True, sigma is used in absolute sense
    bounds : tuple, optional
        Bounds (lower, upper) for parameters
    maxfev : int
        Maximum function evaluations

    Returns
    -------
    popt : ndarray
        Optimal parameters
    pcov : ndarray
        Covariance matrix of parameters
    """
    try:
        return _curve_fit_optimistix(
            f, xdata, ydata, p0, sigma, absolute_sigma, bounds, maxfev
        )
    except ImportError:
        return _curve_fit_scipy(
            f, xdata, ydata, p0, sigma, absolute_sigma, bounds, maxfev, **kwargs
        )


def _curve_fit_optimistix(
    f: Callable,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    bounds: tuple | None,
    maxfev: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Curve fitting using optimistix least squares."""
    import jax
    import jax.numpy as jnp
    import optimistix as optx

    xdata = jnp.asarray(xdata)
    ydata = jnp.asarray(ydata)

    if p0 is None:
        # Estimate number of parameters from function signature
        import inspect

        sig = inspect.signature(f)
        n_params = len(sig.parameters) - 1  # Exclude x
        p0 = jnp.ones(n_params)
    else:
        p0 = jnp.asarray(p0)

    if sigma is not None:
        sigma = jnp.asarray(sigma)
        weights = 1.0 / sigma
    else:
        weights = None

    # Define residual function for least squares
    def residual_fn(params, _):
        model = f(xdata, *params)
        residual = model - ydata
        if weights is not None:
            residual = residual * weights
        return residual

    # Use Levenberg-Marquardt for least squares
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)

    try:
        sol = optx.least_squares(
            residual_fn,
            solver,
            p0,
            max_steps=maxfev,
        )
        popt = np.asarray(sol.value)

        # Compute covariance matrix via Jacobian
        # pcov = (J^T @ J)^-1 * s^2
        # where s^2 is the residual variance
        jacobian = jax.jacobian(lambda p: residual_fn(p, None))(sol.value)
        residuals = residual_fn(sol.value, None)

        n_data = len(ydata)
        n_params = len(popt)
        dof = max(0, n_data - n_params)

        if dof > 0:
            chi_sq = float(jnp.sum(residuals**2))
            s_sq = chi_sq / dof

            try:
                jtj = jacobian.T @ jacobian
                pcov = np.asarray(jnp.linalg.inv(jtj) * s_sq)
            except Exception:
                pcov = np.full((n_params, n_params), np.inf)
        else:
            pcov = np.full((n_params, n_params), np.inf)

        if not absolute_sigma and sigma is not None:
            pcov = pcov * s_sq

        return popt, pcov

    except Exception:
        # Fall back to scipy
        return _curve_fit_scipy(
            f,
            np.asarray(xdata),
            np.asarray(ydata),
            np.asarray(p0) if p0 is not None else None,
            np.asarray(sigma) if sigma is not None else None,
            absolute_sigma,
            bounds,
            maxfev,
        )


def _curve_fit_scipy(
    f: Callable,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    bounds: tuple | None,
    maxfev: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback to scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit as scipy_curve_fit

    bounds_scipy = bounds if bounds is not None else (-np.inf, np.inf)

    return scipy_curve_fit(
        f,
        xdata,
        ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        bounds=bounds_scipy,
        maxfev=maxfev,
        **kwargs,
    )


def least_squares(
    fun: Callable,
    x0: ArrayLike,
    bounds: tuple = (-np.inf, np.inf),
    method: str = "trf",
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    **kwargs,
) -> OptimizeResult:
    """Solve nonlinear least squares using optimistix.

    Parameters
    ----------
    fun : callable
        Function returning residuals: fun(x) -> residuals
    x0 : array-like
        Initial guess
    bounds : tuple
        Lower and upper bounds
    method : str
        Method: 'trf', 'dogbox', 'lm'
    ftol, xtol, gtol : float
        Tolerances
    max_nfev : int, optional
        Maximum function evaluations

    Returns
    -------
    OptimizeResult
        Optimization result
    """
    max_nfev = max_nfev or 1000

    try:
        return _least_squares_optimistix(fun, x0, bounds, ftol, xtol, max_nfev)
    except ImportError:
        return _least_squares_scipy(
            fun, x0, bounds, method, ftol, xtol, gtol, max_nfev, **kwargs
        )


def _least_squares_optimistix(
    fun: Callable,
    x0: ArrayLike,
    bounds: tuple,
    ftol: float,
    xtol: float,
    max_nfev: int,
) -> OptimizeResult:
    """Least squares using optimistix."""
    import jax.numpy as jnp
    import optimistix as optx

    x0 = jnp.asarray(x0)

    def residual_fn(x, _):
        return fun(x)

    solver = optx.LevenbergMarquardt(rtol=xtol, atol=ftol)

    try:
        sol = optx.least_squares(
            residual_fn,
            solver,
            x0,
            max_steps=max_nfev,
        )
        residuals = fun(sol.value)
        cost = float(0.5 * jnp.sum(residuals**2))

        return OptimizeResult(
            x=np.asarray(sol.value),
            fun=cost,
            success=True,
            message="Optimization converged",
            nit=int(sol.stats.get("num_steps", 0)) if hasattr(sol, "stats") else 0,
        )
    except Exception as e:
        residuals = fun(x0)
        cost = float(0.5 * np.sum(np.asarray(residuals) ** 2))
        return OptimizeResult(
            x=np.asarray(x0),
            fun=cost,
            success=False,
            message=str(e),
        )


def _least_squares_scipy(
    fun: Callable,
    x0: ArrayLike,
    bounds: tuple,
    method: str,
    ftol: float,
    xtol: float,
    gtol: float,
    max_nfev: int,
    **kwargs,
) -> OptimizeResult:
    """Fallback to scipy.optimize.least_squares."""
    from scipy.optimize import least_squares as scipy_least_squares

    result = scipy_least_squares(
        fun,
        x0,
        bounds=bounds,
        method=method,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        **kwargs,
    )
    return OptimizeResult(
        x=result.x,
        fun=result.cost,
        success=result.success,
        message=result.message,
        nfev=result.nfev,
        nit=result.njev if hasattr(result, "njev") else 0,
        jac=result.jac if hasattr(result, "jac") else None,
    )


def root(
    fun: Callable,
    x0: ArrayLike,
    args: tuple = (),
    method: str = "hybr",
    jac: Callable | bool | None = None,
    tol: float | None = None,
    options: dict | None = None,
) -> OptimizeResult:
    """Find roots of a function using optimistix.

    Parameters
    ----------
    fun : callable
        Function returning residuals
    x0 : array-like
        Initial guess
    args : tuple
        Extra arguments
    method : str
        Method (ignored, always uses Newton)
    jac : callable, optional
        Jacobian function
    tol : float, optional
        Tolerance
    options : dict, optional
        Solver options

    Returns
    -------
    OptimizeResult
        Root finding result
    """
    tol = tol or 1e-8
    options = options or {}
    maxiter = options.get("maxiter", 1000)

    try:
        return _root_optimistix(fun, x0, args, tol, maxiter)
    except ImportError:
        return _root_scipy(fun, x0, args, method, jac, tol, options)


def _root_optimistix(
    fun: Callable,
    x0: ArrayLike,
    args: tuple,
    tol: float,
    maxiter: int,
) -> OptimizeResult:
    """Root finding using optimistix."""
    import jax.numpy as jnp
    import optimistix as optx

    x0 = jnp.asarray(x0)

    def residual_fn(x, _):
        if args:
            return fun(x, *args)
        return fun(x)

    solver = optx.Newton(rtol=tol, atol=tol)

    try:
        sol = optx.root_find(
            residual_fn,
            solver,
            x0,
            max_steps=maxiter,
        )
        return OptimizeResult(
            x=np.asarray(sol.value),
            fun=float(jnp.sum(residual_fn(sol.value, None) ** 2)),
            success=True,
            message="Root finding converged",
            nit=int(sol.stats.get("num_steps", 0)) if hasattr(sol, "stats") else 0,
        )
    except Exception as e:
        return OptimizeResult(
            x=np.asarray(x0),
            fun=float(np.sum(np.asarray(fun(x0, *args) if args else fun(x0)) ** 2)),
            success=False,
            message=str(e),
        )


def _root_scipy(
    fun: Callable,
    x0: ArrayLike,
    args: tuple,
    method: str,
    jac: Callable | bool | None,
    tol: float,
    options: dict,
) -> OptimizeResult:
    """Fallback to scipy.optimize.root."""
    from scipy.optimize import root as scipy_root

    result = scipy_root(
        fun,
        x0,
        args=args,
        method=method,
        jac=jac,
        tol=tol,
        options=options,
    )
    return OptimizeResult(
        x=result.x,
        fun=float(np.sum(result.fun**2)) if result.fun is not None else 0.0,
        success=result.success,
        message=result.message,
        nfev=result.nfev if hasattr(result, "nfev") else 0,
    )
