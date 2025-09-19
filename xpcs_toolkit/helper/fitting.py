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
            popt, pcov = curve_fit(
                wrapper_func,
                x,
                y[:, n],
                sigma=sigma,
                p0=p0,
                bounds=(bounds_fit[0], bounds_fit[1]),
                maxfev=5000
            )

            # Store results
            fit_val[n, 0, fit_flag] = popt
            fit_val[n, 1, fit_flag] = np.sqrt(np.diag(pcov))
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


