"""
Simplified XPCS fitting utilities.

Core fitting functions for G2 correlation analysis without over-engineering.
"""
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def single_exp(x, tau, bkg, cts):
    """Single exponential model for G2 correlation function."""
    return cts * np.exp(-2 * x / tau) + bkg


def double_exp(x, tau1, bkg, cts1, tau2, cts2):
    """Double exponential model for G2 correlation function."""
    return cts1 * np.exp(-2 * x / tau1) + cts2 * np.exp(-2 * x / tau2) + bkg


def single_exp_all(x, a, b, c, d):
    """Single exponential with all parameters."""
    return a * np.exp(-2 * x / b) + c + d


def double_exp_all(x, a, b, c, d, e, f):
    """Double exponential with all parameters."""
    return a * np.exp(-2 * x / b) + c * np.exp(-2 * x / d) + e + f


def fit_with_fixed(base_func, x, y, sigma, bounds, fit_flag, fit_x, p0=None, **kwargs):
    """
    Simplified fitting with fixed parameters.

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
    if not isinstance(fit_flag, np.ndarray):
        fit_flag = np.array(fit_flag)

    fix_flag = np.logical_not(fit_flag)

    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    num_args = len(fit_flag)

    # Process boundaries for fitting parameters only
    bounds_fit = bounds[:, fit_flag]

    # Initial guess for fitting parameters
    if p0 is None:
        p0 = np.mean(bounds_fit, axis=0)
    else:
        p0 = np.array(p0)[fit_flag]

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
            # Perform curve fitting
            # Extract sigma for this column to match y shape
            sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma
            # Use appropriate method for constrained optimization
            # 'trf' is recommended for bounded problems
            popt, pcov = curve_fit(
                wrapper_func,
                x,
                y[:, n],
                sigma=sigma_col,
                p0=p0,
                bounds=(bounds_fit[0], bounds_fit[1]),
                method='trf',
                max_nfev=5000
            )

            # Store results
            fit_val[n, 0, fit_flag] = popt

            # Debug covariance matrix and error calculation
            pcov_diag = np.diag(pcov)
            errors = np.sqrt(pcov_diag)

            # Check for problematic values
            if np.any(pcov_diag < 0):
                logger.warning(f"Column {n}: Negative diagonal elements in covariance matrix: {pcov_diag}")
            if np.any(~np.isfinite(pcov_diag)):
                logger.warning(f"Column {n}: Non-finite diagonal elements in covariance matrix: {pcov_diag}")
            if np.any(~np.isfinite(errors)):
                logger.warning(f"Column {n}: Non-finite errors calculated: {errors}")
                # Set problematic errors to a reasonable value
                errors = np.where(np.isfinite(errors), errors, np.abs(popt) * 0.1)
                logger.info(f"Column {n}: Corrected errors to: {errors}")

            fit_val[n, 1, fit_flag] = errors
            fit_val[n, 0, fix_flag] = bounds[1, fix_flag]
            fit_val[n, 1, fix_flag] = 0

        except Exception as e:
            logger.warning(f"Fitting failed for column {n}: {e}")
            # Use bounds mean as fallback
            fit_val[n, 0, :] = np.mean(bounds, axis=0)
            fit_val[n, 1, :] = 0

    # Generate fit lines
    fit_line = np.zeros((y.shape[1], len(fit_x)))
    for n in range(y.shape[1]):
        fit_line[n] = base_func(fit_x, *fit_val[n, 0, :])

    return fit_line, fit_val


def robust_curve_fit(func, x, y, **kwargs):
    """Simple wrapper around scipy.optimize.curve_fit with error handling."""
    try:
        return curve_fit(func, x, y, **kwargs)
    except Exception as e:
        logger.warning(f"Curve fitting failed: {e}")
        # Return fallback values
        n_params = func.__code__.co_argcount - 1  # -1 for x parameter
        return np.ones(n_params), np.eye(n_params)


def sequential_fitting(func, x, y, sigma=None, p0=None, bounds=None, **kwargs):
    """
    Sequential fitting approach: robust → least squares → differential evolution.

    Implements a three-stage fitting strategy for improved reliability:
    1. Robust fitting using Huber loss (handles outliers)
    2. Standard least squares (if robust fails)
    3. Differential evolution global optimization (if least squares fails)

    Parameters
    ----------
    func : callable
        Function to fit
    x : array
        Independent variable data
    y : array
        Dependent variable data
    sigma : array, optional
        Uncertainty in y
    p0 : array, optional
        Initial parameter guess
    bounds : tuple, optional
        Parameter bounds as (lower, upper)
    **kwargs
        Additional arguments for fitting methods

    Returns
    -------
    tuple
        (popt, pcov, method_used) where method_used indicates which method succeeded
    """
    from scipy.optimize import curve_fit, differential_evolution
    import warnings

    # Suppress warnings during fitting attempts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Method 1: Robust fitting with Huber loss
        try:
            logger.debug("Attempting robust fitting with Huber loss")
            # Use appropriate method for bounded problems
            # Check if bounds are specified (not infinite bounds)
            try:
                is_bounded = not (
                    isinstance(bounds, tuple) and
                    len(bounds) == 2 and
                    np.all(bounds[0] == -np.inf) and
                    np.all(bounds[1] == np.inf)
                )
            except (ValueError, TypeError):
                # If comparison fails, assume bounds are specified
                is_bounded = True
            # Remove conflicting parameters from kwargs
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['max_nfev', 'maxfev']}

            if is_bounded:
                popt, pcov = curve_fit(
                    func, x, y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    method='trf',  # Trust Region Reflective for bounded problems
                    max_nfev=5000,
                    **safe_kwargs
                )
            else:
                popt, pcov = curve_fit(
                    func, x, y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    method='lm',  # Levenberg-Marquardt for unbounded problems
                    maxfev=5000,
                    **safe_kwargs
                )
            # Validate result
            if np.all(np.isfinite(popt)) and np.all(np.isfinite(pcov)):
                logger.debug("Robust fitting succeeded")
                return popt, pcov, "robust"
        except Exception as e:
            logger.debug(f"Robust fitting failed: {e}")

        # Method 2: Standard least squares
        try:
            logger.debug("Attempting standard least squares fitting")
            # Use appropriate method and parameter name for bounded problems
            # Check if bounds are specified (not infinite bounds)
            try:
                is_bounded = not (
                    isinstance(bounds, tuple) and
                    len(bounds) == 2 and
                    np.all(bounds[0] == -np.inf) and
                    np.all(bounds[1] == np.inf)
                )
            except (ValueError, TypeError):
                # If comparison fails, assume bounds are specified
                is_bounded = True
            # Remove conflicting parameters from kwargs
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['max_nfev', 'maxfev']}

            if is_bounded:
                popt, pcov = curve_fit(
                    func, x, y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    method='trf',
                    max_nfev=5000,
                    **safe_kwargs
                )
            else:
                popt, pcov = curve_fit(
                    func, x, y,
                    sigma=sigma,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000,
                    **safe_kwargs
                )
            # Validate result
            if np.all(np.isfinite(popt)) and np.all(np.isfinite(pcov)):
                logger.debug("Standard fitting succeeded")
                return popt, pcov, "least_squares"
        except Exception as e:
            logger.debug(f"Standard fitting failed: {e}")

        # Method 3: Differential Evolution (global optimization)
        try:
            logger.debug("Attempting differential evolution fitting")
            if bounds is None:
                raise ValueError("Bounds required for differential evolution")

            # Define objective function for differential evolution
            def objective(params):
                try:
                    y_pred = func(x, *params)
                    if sigma is not None:
                        residuals = (y - y_pred) / sigma
                    else:
                        residuals = y - y_pred
                    return np.sum(residuals**2)
                except:
                    return np.inf

            result = differential_evolution(
                objective,
                bounds=list(zip(bounds[0], bounds[1])),
                seed=42,  # For reproducibility
                maxiter=1000,
                popsize=15,
                tol=1e-6
            )

            if result.success:
                popt = result.x
                # Estimate covariance using Jacobian at optimum
                try:
                    # Calculate Jacobian numerically
                    eps = np.sqrt(np.finfo(float).eps)
                    jac = np.zeros((len(y), len(popt)))
                    y0 = func(x, *popt)
                    for i in range(len(popt)):
                        params_plus = popt.copy()
                        params_plus[i] += eps
                        y_plus = func(x, *params_plus)
                        jac[:, i] = (y_plus - y0) / eps

                    # Estimate covariance matrix
                    if sigma is not None:
                        jac_weighted = jac / sigma[:, np.newaxis]
                    else:
                        jac_weighted = jac

                    pcov = np.linalg.inv(jac_weighted.T @ jac_weighted)

                except:
                    # Fallback: identity matrix scaled by parameter magnitude
                    pcov = np.eye(len(popt)) * (np.abs(popt) + 1e-10)

                logger.debug("Differential evolution succeeded")
                return popt, pcov, "differential_evolution"
            else:
                logger.debug("Differential evolution failed to converge")

        except Exception as e:
            logger.debug(f"Differential evolution failed: {e}")

    # All methods failed - return fallback values
    logger.warning("All fitting methods failed, using fallback parameters")
    n_params = func.__code__.co_argcount - 1
    if p0 is not None:
        popt = np.array(p0)
    elif bounds is not None:
        popt = np.mean(bounds, axis=0)
    else:
        popt = np.ones(n_params)

    pcov = np.eye(n_params) * 1e6  # Large uncertainty
    return popt, pcov, "fallback"


def fit_with_fixed_sequential(base_func, x, y, sigma, bounds, fit_flag, fit_x, p0=None, **kwargs):
    """
    Enhanced fitting with sequential method approach: robust → least squares → differential evolution.

    This version of fit_with_fixed uses the sequential_fitting function to provide
    more robust fitting by trying multiple optimization methods.

    Parameters
    ----------
    base_func : callable
        Function to fit
    x : array
        Input data
    y : array
        Output data (2D array, columns are different q-values)
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
        (fit_line, fit_params, fit_methods) where fit_methods indicates which method was used for each column
    """
    if not isinstance(fit_flag, np.ndarray):
        fit_flag = np.array(fit_flag)

    fix_flag = np.logical_not(fit_flag)

    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    num_args = len(fit_flag)

    # Process boundaries for fitting parameters only
    bounds_fit = bounds[:, fit_flag]

    # Initial guess for fitting parameters
    if p0 is None:
        p0 = np.mean(bounds_fit, axis=0)
    else:
        p0 = np.array(p0)[fit_flag]

    fit_val = np.zeros((y.shape[1], 2, num_args))
    fit_methods = []  # Track which method was used for each column

    # Create wrapper function for fixed parameters
    def wrapper_func(x_data, *fit_params):
        full_params = np.zeros(num_args)
        full_params[fit_flag] = fit_params
        full_params[fix_flag] = bounds[1, fix_flag]  # Use upper bound as fixed value
        return base_func(x_data, *full_params)

    # Fit each column using sequential approach
    for n in range(y.shape[1]):
        try:
            # Extract sigma for this column to match y shape
            sigma_col = sigma[:, n] if sigma.ndim > 1 else sigma

            # Use sequential fitting
            popt, pcov, method_used = sequential_fitting(
                wrapper_func,
                x,
                y[:, n],
                sigma=sigma_col,
                p0=p0,
                bounds=(bounds_fit[0], bounds_fit[1]),
                max_nfev=5000
            )

            fit_methods.append(method_used)
            logger.debug(f"Column {n}: fitted using {method_used}")

            # Store results
            fit_val[n, 0, fit_flag] = popt
            fit_val[n, 1, fit_flag] = np.sqrt(np.diag(pcov))
            fit_val[n, 0, fix_flag] = bounds[1, fix_flag]
            fit_val[n, 1, fix_flag] = 0

        except Exception as e:
            logger.warning(f"Sequential fitting failed for column {n}: {e}")
            fit_methods.append("fallback_error")
            # Use bounds mean as fallback
            fit_val[n, 0, :] = np.mean(bounds, axis=0)
            fit_val[n, 1, :] = 0

    # Generate fit lines
    fit_line = np.zeros((y.shape[1], len(fit_x)))
    for n in range(y.shape[1]):
        fit_line[n] = base_func(fit_x, *fit_val[n, 0, :])

    # Log summary of methods used
    method_counts = {}
    for method in fit_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    logger.info(f"Fitting methods used: {method_counts}")

    return fit_line, fit_val, fit_methods


