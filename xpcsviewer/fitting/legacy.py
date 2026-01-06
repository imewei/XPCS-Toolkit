"""Legacy fitting utilities for backward compatibility.

Provides the same API as the old xpcsviewer.helper.fitting module,
migrated to use the new JAX-accelerated backend where possible.

These functions maintain the same interface for xpcs_file.py compatibility
while leveraging the new fitting infrastructure internally.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from xpcsviewer.backends import get_backend
from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


def single_exp(
    x: NDArray[np.floating[Any]], tau: float, bkg: float, cts: float
) -> NDArray[np.floating[Any]]:
    """Single exponential model for G2 correlation function."""
    backend = get_backend()
    x_arr = backend.array(x)
    result = cts * backend.exp(-2 * x_arr / tau) + bkg
    return ensure_numpy(result)


def double_exp(
    x: NDArray[np.floating[Any]],
    tau1: float,
    bkg: float,
    cts1: float,
    tau2: float,
    cts2: float,
) -> NDArray[np.floating[Any]]:
    """Double exponential model for G2 correlation function."""
    backend = get_backend()
    x_arr = backend.array(x)
    result = (
        cts1 * backend.exp(-2 * x_arr / tau1)
        + cts2 * backend.exp(-2 * x_arr / tau2)
        + bkg
    )
    return ensure_numpy(result)


def single_exp_all(
    x: NDArray[np.floating[Any]], a: float, b: float, c: float, d: float
) -> NDArray[np.floating[Any]]:
    """Single exponential with all parameters."""
    backend = get_backend()
    x_arr = backend.array(x)
    result = a * backend.exp(-2 * x_arr / b) + c + d
    return ensure_numpy(result)


def double_exp_all(
    x: NDArray[np.floating[Any]],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
) -> NDArray[np.floating[Any]]:
    """Double exponential with all parameters."""
    backend = get_backend()
    x_arr = backend.array(x)
    result = a * backend.exp(-2 * x_arr / b) + c * backend.exp(-2 * x_arr / d) + e + f
    return ensure_numpy(result)


def fit_with_fixed(
    base_func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
    fit_x: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]] | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Fitting with fixed parameters using scipy.optimize.curve_fit.

    Parameters
    ----------
    base_func : callable
        Function to fit
    x : array
        Input data
    y : array
        Output data
    sigma : array
        Error bars
    bounds : tuple
        (lower_bounds, upper_bounds)
    fit_flag : array
        Boolean array indicating which parameters to fit
    fit_x : array
        X values for output curve
    p0 : array, optional
        Initial parameter values

    Returns
    -------
    tuple
        (fit_line, fit_params)
    """
    # Ensure numpy arrays at scipy boundary
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma)
    fit_x = ensure_numpy(fit_x)

    if not isinstance(fit_flag, np.ndarray):
        fit_flag = np.array(fit_flag)

    fix_flag = np.logical_not(fit_flag)

    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    num_args = len(fit_flag)

    # Process boundaries for fitting parameters only
    bounds_fit = bounds[:, fit_flag]

    # Initial guess for fitting parameters
    p0 = np.mean(bounds_fit, axis=0) if p0 is None else np.array(p0)[fit_flag]

    fit_val = np.zeros((y.shape[1], 2, num_args))

    # Create wrapper function for fixed parameters
    def wrapper_func(x_data, *fit_params):
        full_params = np.zeros(num_args)
        full_params[fit_flag] = fit_params
        full_params[fix_flag] = bounds[1, fix_flag]  # Use upper bound as fixed value
        return base_func(x_data, *full_params)

    # Fit each column
    for n in range(y.shape[1]):
        try:
            sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma
            popt, pcov = curve_fit(
                wrapper_func,
                x,
                y[:, n],
                sigma=sigma_col,
                p0=p0,
                bounds=(bounds_fit[0], bounds_fit[1]),
                method="trf",
                max_nfev=5000,
            )

            fit_val[n, 0, fit_flag] = popt
            pcov_diag = np.diag(pcov)
            errors = np.sqrt(pcov_diag)

            if np.any(pcov_diag < 0):
                logger.warning(
                    f"Column {n}: Negative diagonal elements in covariance matrix"
                )
            if np.any(~np.isfinite(errors)):
                errors = np.where(np.isfinite(errors), errors, np.abs(popt) * 0.1)

            fit_val[n, 1, fit_flag] = errors
            fit_val[n, 0, fix_flag] = bounds[1, fix_flag]
            fit_val[n, 1, fix_flag] = 0

        except Exception as e:
            logger.warning(f"Fitting failed for column {n}: {e}")
            fit_val[n, 0, :] = np.mean(bounds, axis=0)
            fit_val[n, 1, :] = 0

    # Generate fit lines
    fit_line = np.zeros((y.shape[1], len(fit_x)))
    for n in range(y.shape[1]):
        fit_line[n] = ensure_numpy(base_func(fit_x, *fit_val[n, 0, :]))

    return fit_line, fit_val


def _fit_single_qvalue(
    args: tuple[Any, ...],
) -> tuple[int, NDArray[np.floating[Any]], NDArray[np.floating[Any]], bool]:
    """Worker function for parallel fitting of a single q-value."""
    col_idx, x, y_col, sigma_col, wrapper_func, p0, bounds_fit = args

    try:
        popt, pcov = curve_fit(
            wrapper_func,
            x,
            y_col,
            sigma=sigma_col,
            p0=p0,
            bounds=(bounds_fit[0], bounds_fit[1]),
            method="trf",
            max_nfev=5000,
        )

        pcov_diag = np.diag(pcov)
        errors = np.sqrt(np.abs(pcov_diag))

        if np.any(~np.isfinite(errors)):
            errors = np.where(np.isfinite(errors), errors, np.abs(popt) * 0.1)

        return col_idx, popt, errors, True

    except Exception as e:
        logger.warning(f"Fitting failed for q-value {col_idx}: {e}")
        return col_idx, None, None, False


def fit_with_fixed_parallel(
    base_func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
    fit_x: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]] | None = None,
    max_workers: int | None = None,
    use_threads: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Parallel version of fit_with_fixed for processing multiple q-values simultaneously."""
    # Ensure numpy arrays at scipy boundary
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma)
    fit_x = ensure_numpy(fit_x)

    if not isinstance(fit_flag, np.ndarray):
        fit_flag = np.array(fit_flag)

    fix_flag = np.logical_not(fit_flag)

    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    num_args = len(fit_flag)
    num_qvals = y.shape[1]

    bounds_fit = bounds[:, fit_flag]
    p0 = np.mean(bounds_fit, axis=0) if p0 is None else np.array(p0)[fit_flag]

    fit_val = np.zeros((num_qvals, 2, num_args))

    def wrapper_func(x_data, *fit_params):
        full_params = np.zeros(num_args)
        full_params[fit_flag] = fit_params
        full_params[fix_flag] = bounds[1, fix_flag]
        return ensure_numpy(base_func(x_data, *full_params))

    fit_args = []
    for n in range(num_qvals):
        sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma
        fit_args.append((n, x, y[:, n], sigma_col, wrapper_func, p0, bounds_fit))

    if max_workers is None:
        import os

        max_workers = min(num_qvals, os.cpu_count() or 1)

    logger.info(
        f"Starting parallel G2 fitting for {num_qvals} q-values using {max_workers} workers"
    )

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        future_to_col = {
            executor.submit(_fit_single_qvalue, args): args[0] for args in fit_args
        }

        for completed_fits, future in enumerate(as_completed(future_to_col), start=1):
            col_idx, popt, errors, success = future.result()

            if success:
                fit_val[col_idx, 0, fit_flag] = popt
                fit_val[col_idx, 1, fit_flag] = errors
                fit_val[col_idx, 0, fix_flag] = bounds[1, fix_flag]
                fit_val[col_idx, 1, fix_flag] = 0
            else:
                fit_val[col_idx, 0, :] = np.mean(bounds, axis=0)
                fit_val[col_idx, 1, :] = 0

            if completed_fits % max(1, num_qvals // 10) == 0:
                progress = (completed_fits / num_qvals) * 100
                logger.debug(
                    f"Parallel fitting progress: {progress:.1f}% ({completed_fits}/{num_qvals})"
                )

    def generate_fit_line(n):
        return n, ensure_numpy(base_func(fit_x, *fit_val[n, 0, :]))

    fit_line = np.zeros((num_qvals, len(fit_x)))

    with executor_class(max_workers=max_workers) as executor:
        line_futures = {
            executor.submit(generate_fit_line, n): n for n in range(num_qvals)
        }

        for future in as_completed(line_futures):
            n, line_data = future.result()
            fit_line[n] = line_data

    logger.info(f"Parallel G2 fitting completed for {num_qvals} q-values")
    return fit_line, fit_val


def sequential_fitting(
    func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]] | None = None,
    p0: NDArray[np.floating[Any]] | None = None,
    bounds: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], str]:
    """Sequential fitting approach: robust → least squares → differential evolution."""
    import warnings

    from scipy.optimize import differential_evolution

    # Ensure numpy arrays
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    if sigma is not None:
        sigma = ensure_numpy(sigma)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Method 1: Robust fitting with TRF
        try:
            logger.debug("Attempting robust fitting")
            try:
                is_bounded = not (
                    isinstance(bounds, tuple)
                    and len(bounds) == 2
                    and np.all(bounds[0] == -np.inf)
                    and np.all(bounds[1] == np.inf)
                )
            except (ValueError, TypeError):
                is_bounded = True

            safe_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["max_nfev", "maxfev"]
            }

            if is_bounded:
                popt, pcov = curve_fit(
                    func,
                    x,
                    y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    method="trf",
                    max_nfev=5000,
                    **safe_kwargs,
                )
            else:
                popt, pcov = curve_fit(
                    func,
                    x,
                    y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    method="lm",
                    maxfev=5000,
                    **safe_kwargs,
                )
            if np.all(np.isfinite(popt)) and np.all(np.isfinite(pcov)):
                logger.debug("Robust fitting succeeded")
                return popt, pcov, "robust"
        except Exception as e:
            logger.debug(f"Robust fitting failed: {e}")

        # Method 2: Standard least squares
        try:
            logger.debug("Attempting standard least squares fitting")
            try:
                is_bounded = not (
                    isinstance(bounds, tuple)
                    and len(bounds) == 2
                    and np.all(bounds[0] == -np.inf)
                    and np.all(bounds[1] == np.inf)
                )
            except (ValueError, TypeError):
                is_bounded = True

            safe_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["max_nfev", "maxfev"]
            }

            if is_bounded:
                popt, pcov = curve_fit(
                    func,
                    x,
                    y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    method="trf",
                    max_nfev=5000,
                    **safe_kwargs,
                )
            else:
                popt, pcov = curve_fit(
                    func,
                    x,
                    y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000,
                    **safe_kwargs,
                )
            if np.all(np.isfinite(popt)) and np.all(np.isfinite(pcov)):
                logger.debug("Standard fitting succeeded")
                return popt, pcov, "least_squares"
        except Exception as e:
            logger.debug(f"Standard fitting failed: {e}")

        # Method 3: Differential Evolution
        try:
            logger.debug("Attempting differential evolution fitting")
            if bounds is None:
                raise ValueError("Bounds required for differential evolution")

            def objective(params):
                try:
                    y_pred = func(x, *params)
                    if sigma is not None:
                        residuals = (y - y_pred) / sigma
                    else:
                        residuals = y - y_pred
                    return np.sum(residuals**2)
                except (ValueError, FloatingPointError, OverflowError):
                    return np.inf

            result = differential_evolution(
                objective,
                bounds=list(zip(bounds[0], bounds[1], strict=False)),
                seed=42,
                maxiter=1000,
                popsize=15,
                tol=1e-6,
            )

            if result.success:
                popt = result.x
                try:
                    eps = np.sqrt(np.finfo(float).eps)
                    jac = np.zeros((len(y), len(popt)))
                    y0 = func(x, *popt)
                    for i in range(len(popt)):
                        params_plus = popt.copy()
                        params_plus[i] += eps
                        y_plus = func(x, *params_plus)
                        jac[:, i] = (y_plus - y0) / eps

                    if sigma is not None:
                        jac_weighted = jac / sigma[:, np.newaxis]
                    else:
                        jac_weighted = jac

                    pcov = np.linalg.inv(jac_weighted.T @ jac_weighted)

                except (np.linalg.LinAlgError, ValueError):
                    pcov = np.eye(len(popt)) * (np.abs(popt) + 1e-10)

                logger.debug("Differential evolution succeeded")
                return popt, pcov, "differential_evolution"
            logger.debug("Differential evolution failed to converge")

        except Exception as e:
            logger.debug(f"Differential evolution failed: {e}")

    # All methods failed
    logger.warning("All fitting methods failed, using fallback parameters")
    n_params = func.__code__.co_argcount - 1
    if p0 is not None:
        popt = np.array(p0)
    elif bounds is not None:
        popt = np.mean(bounds, axis=0)
    else:
        popt = np.ones(n_params)

    pcov = np.eye(n_params) * 1e6
    return popt, pcov, "fallback"


def fit_with_fixed_sequential(
    base_func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    bounds: NDArray[np.floating[Any]],
    fit_flag: NDArray[np.bool_],
    fit_x: NDArray[np.floating[Any]],
    p0: NDArray[np.floating[Any]] | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[str]]:
    """Enhanced fitting with sequential method approach."""
    # Ensure numpy arrays
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    sigma = ensure_numpy(sigma)
    fit_x = ensure_numpy(fit_x)

    if not isinstance(fit_flag, np.ndarray):
        fit_flag = np.array(fit_flag)

    fix_flag = np.logical_not(fit_flag)

    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    num_args = len(fit_flag)
    bounds_fit = bounds[:, fit_flag]
    p0 = np.mean(bounds_fit, axis=0) if p0 is None else np.array(p0)[fit_flag]

    fit_val = np.zeros((y.shape[1], 2, num_args))
    fit_methods = []

    def wrapper_func(x_data, *fit_params):
        full_params = np.zeros(num_args)
        full_params[fit_flag] = fit_params
        full_params[fix_flag] = bounds[1, fix_flag]
        return ensure_numpy(base_func(x_data, *full_params))

    for n in range(y.shape[1]):
        try:
            sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma

            popt, pcov, method_used = sequential_fitting(
                wrapper_func,
                x,
                y[:, n],
                sigma=sigma_col,
                p0=p0,
                bounds=(bounds_fit[0], bounds_fit[1]),
                max_nfev=5000,
            )

            fit_methods.append(method_used)
            logger.debug(f"Column {n}: fitted using {method_used}")

            fit_val[n, 0, fit_flag] = popt
            fit_val[n, 1, fit_flag] = np.sqrt(np.diag(pcov))
            fit_val[n, 0, fix_flag] = bounds[1, fix_flag]
            fit_val[n, 1, fix_flag] = 0

        except Exception as e:
            logger.warning(f"Sequential fitting failed for column {n}: {e}")
            fit_methods.append("fallback_error")
            fit_val[n, 0, :] = np.mean(bounds, axis=0)
            fit_val[n, 1, :] = 0

    fit_line = np.zeros((y.shape[1], len(fit_x)))
    for n in range(y.shape[1]):
        fit_line[n] = ensure_numpy(base_func(fit_x, *fit_val[n, 0, :]))

    method_counts = {}
    for method in fit_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    logger.info(f"Fitting methods used: {method_counts}")

    return fit_line, fit_val, fit_methods


def robust_curve_fit(
    func: Callable[..., NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Simple wrapper around scipy.optimize.curve_fit with error handling."""
    x = ensure_numpy(x)
    y = ensure_numpy(y)
    try:
        return curve_fit(func, x, y, **kwargs)
    except Exception as e:
        logger.warning(f"Curve fitting failed: {e}")
        n_params = func.__code__.co_argcount - 1
        return np.ones(n_params), np.eye(n_params)


def vectorized_parameter_estimation(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    model_type: str = "exponential",
) -> tuple | None:
    """Vectorized parameter estimation."""
    x = ensure_numpy(x)
    y = ensure_numpy(y)

    try:
        if model_type == "exponential":
            y_min = np.min(y)
            y_max = np.max(y)
            baseline_guess = y_min
            amplitude_guess = y_max - y_min

            half_amp = y_min + (y_max - y_min) / np.e
            idx_half = np.argmin(np.abs(y - half_amp))
            tau_guess = x[idx_half] if idx_half > 0 else x[len(x) // 2]

            p0 = [tau_guess, baseline_guess, amplitude_guess]

            bounds = (
                [x[1] * 0.1, -np.abs(y_max), amplitude_guess * 0.1],
                [x[-1] * 10, y_max * 1.1, amplitude_guess * 10],
            )

            popt, _ = curve_fit(
                single_exp, x, y, p0=p0, bounds=bounds, method="trf", maxfev=5000
            )
            return tuple(popt)

    except Exception:
        return None

    return None


def vectorized_residual_analysis(
    x: NDArray[np.floating[Any]],
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> dict[str, float | NDArray[np.floating[Any]]]:
    """Vectorized residual analysis."""
    y_true = ensure_numpy(y_true)
    y_pred = ensure_numpy(y_pred)
    residuals = y_true - y_pred

    return {
        "mean_residual": np.mean(residuals),
        "std_residual": np.std(residuals),
        "rmse": np.sqrt(np.mean(residuals**2)),
        "mae": np.mean(np.abs(residuals)),
    }
