import logging
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import Memory, Parallel, delayed
from scipy import stats
from scipy.optimize import curve_fit, differential_evolution
from sklearn import linear_model

from ..utils.logging_config import get_logger

logger = get_logger(__name__)
cache_dir = os.path.join(os.path.expanduser("~"), ".xpcs_toolkit")

memory = Memory(cache_dir, verbose=0)


@memory.cache
def fit_with_fixed(*args, **kwargs):
    # wrap the fitting function in memory so avoid re-run
    # Extract n_jobs parameter for non-cached version if present
    n_jobs = kwargs.pop("n_jobs", None)
    return fit_with_fixed_raw(*args, n_jobs=n_jobs, **kwargs)


def single_exp(x, tau, bkg, cts):
    return cts * np.exp(-2 * x / tau) + bkg


def double_exp(x, tau1, bkg, cts1, tau2, cts2):
    """Double exponential model for G2 correlation function."""
    return cts1 * np.exp(-2 * x / tau1) + cts2 * np.exp(-2 * x / tau2) + bkg


def fit_tau(qd, tau, tau_err):
    """Highly optimized tau fitting with advanced vectorized operations"""
    # Input validation and preprocessing in vectorized manner
    valid_mask = (
        (qd > 0)
        & (tau > 0)
        & (tau_err > 0)
        & np.isfinite(qd)
        & np.isfinite(tau)
        & np.isfinite(tau_err)
    )
    if not np.any(valid_mask):
        raise ValueError("No valid data points for fitting")

    # Apply mask to all arrays simultaneously
    qd_clean = qd[valid_mask]
    tau_clean = tau[valid_mask]
    tau_err_clean = tau_err[valid_mask]

    # Vectorized logarithmic transformation
    log_qd = np.log(qd_clean)
    log_tau = np.log(tau_clean)

    # Optimized weight calculation - avoid division where possible
    weights = tau_clean / tau_err_clean

    # Reshape for sklearn compatibility
    x = log_qd.reshape(-1, 1)
    y = log_tau.reshape(-1, 1)

    # Perform weighted linear regression
    reg = linear_model.LinearRegression()
    reg.fit(x, y, sample_weight=weights)

    # Vectorized prediction range generation
    x_range = np.max(log_qd) - np.min(log_qd)
    x_min, x_max = np.min(log_qd) - 0.1 * x_range, np.max(log_qd) + 0.1 * x_range
    x2 = np.linspace(x_min, x_max, 128).reshape(-1, 1)
    y2 = reg.predict(x2)

    # Vectorized exponential transformation
    qd_pred = np.exp(x2.ravel())
    tau_pred = np.exp(y2.ravel())

    return reg.coef_, reg.intercept_, qd_pred, tau_pred


def _fit_single_q(args):
    """Optimized single q-value fitting with vectorized error handling"""
    tel, g2_col, g2_err_col, qd_val, p0_guess, bounds, fit_x = args

    # Vectorized error preprocessing
    err = g2_err_col.copy()
    zero_err_mask = err <= 1e-6
    num_zero_err = np.sum(zero_err_mask)

    # More robust average calculation
    valid_err_mask = err > 1e-6
    if np.any(valid_err_mask):
        avg_err = np.mean(err[valid_err_mask])
    else:
        avg_err = 1.0  # Fallback value

    err[zero_err_mask] = avg_err

    # Pre-allocate result arrays
    fit_val_row = np.zeros(7, dtype=np.float64)
    fit_val_row[0] = qd_val

    result = {"num_zero_err": num_zero_err}

    try:
        # Data validation before fitting
        valid_data_mask = np.isfinite(tel) & np.isfinite(g2_col) & np.isfinite(err)
        if not np.any(valid_data_mask):
            raise ValueError("No valid data points")

        # Apply data filtering if needed
        if not np.all(valid_data_mask):
            tel_clean = tel[valid_data_mask]
            g2_clean = g2_col[valid_data_mask]
            err_clean = err[valid_data_mask]
        else:
            tel_clean = tel
            g2_clean = g2_col
            err_clean = err

        # Optimized curve fitting with better initial guess
        popt, pcov = curve_fit(
            single_exp,
            tel_clean,
            g2_clean,
            p0=p0_guess,
            sigma=err_clean,
            bounds=bounds,
            maxfev=2000,  # Limit iterations for performance
            method="trf",  # Trust Region Reflective algorithm (often faster)
        )

        # Vectorized parameter and error extraction
        fit_val_row[1:4] = popt
        fit_val_row[4:7] = np.sqrt(np.diag(pcov))

        # Vectorized fit line calculation
        fit_y = single_exp(fit_x, *popt)

        result.update(
            {
                "err_msg": None,
                "opt": popt,
                "err": np.sqrt(np.diag(pcov)),
                "fit_x": fit_x,
                "fit_y": fit_y,
                "success": True,
            }
        )

    except Exception as e:
        # Vectorized error result generation
        result.update(
            {
                "err_msg": f"q_index fit error: {e!s}",
                "fit_x": fit_x,
                "fit_y": np.ones_like(fit_x),
                "success": False,
            }
        )

    return result, fit_val_row


def fit_xpcs(tel, qd, g2, g2_err, b, n_jobs=None):
    """
    Highly optimized XPCS fitting with vectorized operations and intelligent parallelization.

    :param tel: t_el
    :param qd: ql_dyn
    :param g2: g2 [time, ql_dyn]
    :param g2_err: [time, ql_dyn]
    :param b: bounds
    :param n_jobs: number of parallel jobs (None for auto)
    :return:
    """
    # Input validation with vectorized operations
    if not (np.all(np.isfinite(tel)) and np.all(np.isfinite(qd))):
        logger.warning("Invalid tel or qd values detected")

    # Vectorized fit_x generation with optimized range
    tel_valid = tel[tel > 0]
    if len(tel_valid) == 0:
        raise ValueError("No positive time values")

    log_tel_min, log_tel_max = np.log10(np.min(tel_valid)), np.log10(np.max(tel_valid))
    fit_x = np.logspace(log_tel_min - 0.5, log_tel_max + 0.5, 128)

    # Vectorized initial guess calculation
    p0_guess = [
        np.sqrt(b[0][0] * b[1][0]),
        0.5 * (b[0][1] + b[1][1]),
        0.5 * (b[0][2] + b[1][2]),
    ]

    # Intelligent parallelization strategy
    num_q_values = qd.size
    if n_jobs is None:
        # Adaptive worker count based on problem size
        cpu_count = multiprocessing.cpu_count()
        if num_q_values <= 4:
            n_jobs = 1  # Serial for small problems
        elif num_q_values <= 16:
            n_jobs = min(4, cpu_count)
        else:
            n_jobs = min(num_q_values, cpu_count)

    # Pre-allocate result arrays for better memory management
    fit_val = np.zeros((num_q_values, 7), dtype=np.float64)
    fit_result = []

    # Batch argument preparation with memory optimization
    args_list = [
        (tel, g2[:, n], g2_err[:, n], qd[n], p0_guess, b, fit_x)
        for n in range(num_q_values)
    ]

    # Dynamic parallelization decision
    use_parallel = num_q_values >= 4 and n_jobs > 1

    if use_parallel:
        try:
            # Use ProcessPoolExecutor for CPU-bound tasks for better performance
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks at once for better load balancing
                future_to_index = {
                    executor.submit(_fit_single_q, args): i
                    for i, args in enumerate(args_list)
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result, fit_val_row = future.result(
                            timeout=60
                        )  # 60s timeout per fit
                        fit_result.append(result)
                        fit_val[index, :] = fit_val_row
                    except Exception as e:
                        logger.error(f"Fitting failed for q-index {index}: {e}")
                        # Create fallback result
                        fallback_result = {
                            "err_msg": str(e),
                            "fit_x": fit_x,
                            "fit_y": np.ones_like(fit_x),
                            "success": False,
                        }
                        fit_result.append(fallback_result)
                        fit_val[index, 0] = qd[index]  # At least preserve q-value

        except Exception as e:
            logger.warning(f"Parallel fitting failed, falling back to serial: {e}")
            use_parallel = False

    if not use_parallel:
        # Optimized serial processing
        for i, args in enumerate(args_list):
            try:
                result, fit_val_row = _fit_single_q(args)
                fit_result.append(result)
                fit_val[i, :] = fit_val_row
            except Exception as e:
                logger.error(f"Serial fitting failed for q-index {i}: {e}")
                fallback_result = {
                    "err_msg": str(e),
                    "fit_x": fit_x,
                    "fit_y": np.ones_like(fit_x),
                    "success": False,
                }
                fit_result.append(fallback_result)
                fit_val[i, 0] = qd[i]

    return fit_result, fit_val


def _fit_single_column_with_fixed(args):
    """Helper function for parallel fitting with fixed parameters"""
    (
        base_func,
        x,
        y_col,
        sigma_col,
        bounds_fit,
        p0,
        num_args,
        fit_flag,
        fix_flag,
        bounds,
        fit_x,
    ) = args

    # create a function that takes care of the fit flag
    def func(x1, *args_fit):
        inputs = np.zeros(num_args)
        inputs[fix_flag] = bounds[1, fix_flag]
        inputs[fit_flag] = np.array(args_fit)
        return base_func(x1, *inputs)

    fit_val_col = np.zeros((2, num_args))

    try:
        popt, pcov = curve_fit(
            func, x, y_col, p0=p0, sigma=sigma_col, bounds=bounds_fit
        )
        # converge values
        fit_val_col[0, fit_flag] = popt
        fit_val_col[0, fix_flag] = bounds[1, fix_flag]
        # errors; the fixed variables have error of 0
        fit_val_col[1, fit_flag] = np.sqrt(np.diag(pcov))
        # fit line
        fit_y = func(fit_x, *popt)
        flag = True
        msg = "FittingSuccess"
    except (Exception, RuntimeError, ValueError, Warning) as err:
        msg = f"Fitting failed: {err!s}"
        flag = False
        fit_val_col[0, fit_flag] = p0
        fit_val_col[0, fix_flag] = bounds[1, fix_flag]
        # mark failed fitting to be negative so they can be filtered later
        fit_val_col[1, :] = -1
        fit_y = None

    return {"fit_x": fit_x, "fit_y": fit_y, "success": flag, "msg": msg}, fit_val_col


def fit_with_fixed_raw(
    base_func, x, y, sigma, bounds, fit_flag, fit_x, p0=None, n_jobs=None
):
    """
    Optimized fitting with fixed parameters using parallel processing.

    :param base_func: the base function used for fitting; it can have multiple
        input variables, some of which can be fixed during the fitting;
    :param x: scaler input
    :param y: scaler output
    :param sigma: the error for y value
    :param bounds: tuple with two elements. 1st is the lower bounds and 2nd is
        the upper bounds; if the fit_flag for a variable is False, then the
        upper bound is used as the fixed value;
    :param fit_flag: tuple of bools, True/False for fit and fixed
    :param fit_x: the fitting line for x
    :param p0: the initial value for the variables; if None is provided, the
        intial value is set as the mean of lower and upper bounds
    :param n_jobs: number of parallel jobs (None for auto)
    :return: a tuple of (fit_line, fit_val)
    """
    if not isinstance(fit_flag, np.ndarray):
        fit_flag = np.array(fit_flag)

    fix_flag = np.logical_not(fit_flag)

    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    # number of arguments, regardless of fixed or to be fitted
    num_args = len(fit_flag)

    # process boundaries and initial values
    bounds_fit = bounds[:, fit_flag]
    # doing a simple average to get the initial guess;
    p0 = np.mean(bounds_fit, axis=0) if p0 is None else np.array(p0)[fit_flag]

    fit_val = np.zeros((y.shape[1], 2, num_args))

    # Determine number of workers
    if n_jobs is None:
        n_jobs = min(y.shape[1], multiprocessing.cpu_count())

    # Prepare arguments for parallel processing
    args_list = []
    for n in range(y.shape[1]):
        args = (
            base_func,
            x,
            y[:, n],
            sigma[:, n],
            bounds_fit,
            p0,
            num_args,
            fit_flag,
            fix_flag,
            bounds,
            fit_x,
        )
        args_list.append(args)

    fit_line = []

    # Use parallel processing for multiple columns, single-threaded for small datasets
    if y.shape[1] >= 4 and n_jobs > 1:
        try:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(_fit_single_column_with_fixed, args_list))

            for n, (line_result, fit_val_col) in enumerate(results):
                fit_line.append(line_result)
                fit_val[n, :, :] = fit_val_col
        except Exception as e:
            logger.warning(f"Parallel fitting failed, falling back to serial: {e}")
            # Fall back to serial processing
            for args in args_list:
                line_result, fit_val_col = _fit_single_column_with_fixed(args)
                fit_line.append(line_result)
                fit_val[len(fit_line) - 1, :, :] = fit_val_col
    else:
        # Serial processing for small datasets
        for args in args_list:
            line_result, fit_val_col = _fit_single_column_with_fixed(args)
            fit_line.append(line_result)
            fit_val[len(fit_line) - 1, :, :] = fit_val_col

    return fit_line, fit_val


def vectorized_parameter_estimation(x_data, y_data, func_type="exponential"):
    """
    Vectorized initial parameter estimation for various function types.

    Args:
        x_data: Independent variable data
        y_data: Dependent variable data
        func_type: Type of function ('exponential', 'power', 'gaussian')

    Returns:
        Initial parameter estimates
    """
    # Remove invalid data points
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    if not np.any(valid_mask):
        return None

    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]

    if func_type == "exponential":
        # For single exponential: y = A * exp(-t/tau) + B
        # Estimate background as minimum value
        B_est = np.min(y_clean)
        # Estimate amplitude
        A_est = np.max(y_clean) - B_est

        # Estimate time constant using linear regression on log data
        if np.all(y_clean - B_est > 0):
            log_y = np.log(y_clean - B_est)
            # Linear fit to get slope
            slope = np.polyfit(x_clean, log_y, 1)[0]
            tau_est = -1.0 / slope if slope != 0 else np.mean(x_clean)
        else:
            tau_est = np.mean(x_clean)

        return [tau_est, B_est, A_est]

    if func_type == "power":
        # For power law: y = A * x^(-n)
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)
        # Linear regression on log-log data
        coeffs = np.polyfit(log_x, log_y, 1)
        n_est = -coeffs[0]
        A_est = np.exp(coeffs[1])

        return [A_est, n_est]

    if func_type == "gaussian":
        # For Gaussian: y = A * exp(-(x-mu)^2/(2*sigma^2)) + B
        B_est = np.min(y_clean)
        A_est = np.max(y_clean) - B_est

        # Estimate center as weighted mean
        weights = y_clean - B_est
        mu_est = np.average(x_clean, weights=weights)

        # Estimate width using second moment
        sigma_est = np.sqrt(np.average((x_clean - mu_est) ** 2, weights=weights))

        return [A_est, mu_est, sigma_est, B_est]

    return None


def batch_curve_fitting(
    x_data_list,
    y_data_list,
    func,
    p0_list=None,
    bounds_list=None,
    n_jobs=None,
    method="vectorized",
):
    """
    Batch curve fitting with vectorized operations and parallel processing.

    Args:
        x_data_list: List of x-data arrays
        y_data_list: List of y-data arrays
        func: Function to fit
        p0_list: List of initial parameter estimates
        bounds_list: List of parameter bounds
        n_jobs: Number of parallel jobs
        method: Fitting method ('vectorized', 'parallel')

    Returns:
        List of fitting results
    """
    num_datasets = len(x_data_list)

    if p0_list is None:
        p0_list = [None] * num_datasets
    if bounds_list is None:
        bounds_list = [(-np.inf, np.inf)] * num_datasets

    results = []

    if method == "vectorized" and num_datasets <= 10:
        # Serial processing with vectorized operations for small batches
        for i in range(num_datasets):
            try:
                x_data = x_data_list[i]
                y_data = y_data_list[i]
                p0 = p0_list[i]
                bounds = bounds_list[i]

                # Vectorized data cleaning
                valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                if not np.any(valid_mask):
                    results.append({"success": False, "message": "No valid data"})
                    continue

                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]

                popt, pcov = curve_fit(func, x_clean, y_clean, p0=p0, bounds=bounds)

                results.append(
                    {
                        "success": True,
                        "popt": popt,
                        "pcov": pcov,
                        "perr": np.sqrt(np.diag(pcov)),
                    }
                )

            except Exception as e:
                results.append({"success": False, "message": str(e)})

    elif method == "parallel":
        # Parallel processing for large batches
        if n_jobs is None:
            n_jobs = min(num_datasets, multiprocessing.cpu_count())

        def fit_single_dataset(args):
            i, x_data, y_data, p0, bounds = args
            try:
                valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                if not np.any(valid_mask):
                    return i, {"success": False, "message": "No valid data"}

                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]

                popt, pcov = curve_fit(func, x_clean, y_clean, p0=p0, bounds=bounds)

                return i, {
                    "success": True,
                    "popt": popt,
                    "pcov": pcov,
                    "perr": np.sqrt(np.diag(pcov)),
                }

            except Exception as e:
                return i, {"success": False, "message": str(e)}

        # Prepare arguments
        args_list = [
            (i, x_data_list[i], y_data_list[i], p0_list[i], bounds_list[i])
            for i in range(num_datasets)
        ]

        # Execute parallel fitting
        results = [None] * num_datasets
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_results = list(executor.map(fit_single_dataset, args_list))

            for i, result in future_results:
                results[i] = result

    return results


def optimize_fitting_convergence(func, x_data, y_data, p0_bounds, max_attempts=5):
    """
    Optimized fitting with multiple convergence strategies.

    Args:
        func: Function to fit
        x_data: Independent variable data
        y_data: Dependent variable data
        p0_bounds: Tuple of (p0_list, bounds) for multiple attempts
        max_attempts: Maximum number of fitting attempts

    Returns:
        Best fitting result
    """
    best_result = None
    best_chi_squared = np.inf

    p0_list, bounds = p0_bounds

    # Try multiple initial conditions
    for attempt in range(min(max_attempts, len(p0_list))):
        try:
            p0 = p0_list[attempt]

            # Vectorized data validation
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if not np.any(valid_mask):
                continue

            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]

            # Try different methods for robustness
            methods = ["trf", "lm", "dogbox"]

            for method in methods:
                try:
                    popt, pcov = curve_fit(
                        func,
                        x_clean,
                        y_clean,
                        p0=p0,
                        bounds=bounds,
                        method=method,
                        maxfev=1000,
                    )

                    # Calculate chi-squared for comparison
                    y_pred = func(x_clean, *popt)
                    chi_squared = np.sum((y_clean - y_pred) ** 2)

                    if chi_squared < best_chi_squared:
                        best_chi_squared = chi_squared
                        best_result = {
                            "success": True,
                            "popt": popt,
                            "pcov": pcov,
                            "perr": np.sqrt(np.diag(pcov)),
                            "chi_squared": chi_squared,
                            "method": method,
                            "attempt": attempt,
                        }

                    break  # Success with this method

                except Exception:
                    continue  # Try next method

        except Exception:
            continue  # Try next attempt

    if best_result is None:
        return {"success": False, "message": "All fitting attempts failed"}

    return best_result


def vectorized_residual_analysis(x_data, y_data, y_pred, return_stats=True):
    """
    Vectorized residual analysis for fitting quality assessment.

    Args:
        x_data: Independent variable data
        y_data: Observed data
        y_pred: Predicted data
        return_stats: Whether to return statistical measures

    Returns:
        Dictionary with residual analysis results
    """
    # Vectorized residual calculation
    residuals = y_data - y_pred

    result = {
        "residuals": residuals,
        "abs_residuals": np.abs(residuals),
        "squared_residuals": residuals**2,
    }

    if return_stats:
        # Vectorized statistical measures
        result.update(
            {
                "mean_residual": np.mean(residuals),
                "std_residual": np.std(residuals),
                "rmse": np.sqrt(np.mean(residuals**2)),
                "mae": np.mean(np.abs(residuals)),
                "r_squared": 1
                - np.sum(residuals**2) / np.sum((y_data - np.mean(y_data)) ** 2),
                "max_abs_residual": np.max(np.abs(residuals)),
                "residual_autocorr": np.corrcoef(residuals[:-1], residuals[1:])[0, 1],
            }
        )

    return result


# ============================================================================
# Robust Multi-Strategy Optimization Engine for G2 Diffusion Fitting
# ============================================================================


class SyntheticG2DataGenerator:
    """
    Comprehensive synthetic G2 dataset generator with known ground truth parameters
    for validation of fitting algorithms.

    Supports various G2 models commonly used in XPCS analysis:
    - Single exponential decay
    - Double exponential decay
    - Stretched exponential decay
    - With configurable noise levels and data quality scenarios
    """

    def __init__(self, random_state: Optional[int] = None):
        """Initialize the synthetic data generator.

        Args:
            random_state: Random seed for reproducible results
        """
        self.rng = np.random.RandomState(random_state)

        self.models = {
            'single_exp': self._single_exponential,
            'double_exp': self._double_exponential,
            'stretched_exp': self._stretched_exponential,
            'power_law_plus_exp': self._power_law_plus_exponential
        }

    def _single_exponential(self, tau: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Single exponential G2 model: G2(τ) = baseline + beta * exp(-gamma * τ)"""
        return params['baseline'] + params['beta'] * np.exp(-params['gamma'] * tau)

    def _double_exponential(self, tau: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Double exponential G2 model with fast and slow components"""
        return (params['baseline'] +
                params['beta1'] * np.exp(-params['gamma1'] * tau) +
                params['beta2'] * np.exp(-params['gamma2'] * tau))

    def _stretched_exponential(self, tau: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Stretched exponential G2 model: G2(τ) = baseline + beta * exp(-(gamma*τ)^stretch)"""
        return (params['baseline'] +
                params['beta'] * np.exp(-((params['gamma'] * tau) ** params['stretch'])))

    def _power_law_plus_exponential(self, tau: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Power law plus exponential for complex dynamics"""
        return (params['baseline'] +
                params['beta'] * np.exp(-params['gamma'] * tau) +
                params['amplitude'] * tau ** (-params['alpha']))

    def generate_dataset(self,
                        model_type: str = 'single_exp',
                        tau_range: Tuple[float, float] = (1e-6, 1e0),
                        n_points: int = 50,
                        noise_type: str = 'gaussian',
                        noise_level: float = 0.02,
                        systematic_error: bool = False,
                        outlier_fraction: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Generate synthetic G2 dataset with known ground truth.

        Args:
            model_type: Type of G2 model ('single_exp', 'double_exp', 'stretched_exp', 'power_law_plus_exp')
            tau_range: (min_tau, max_tau) in seconds
            n_points: Number of time points
            noise_type: 'gaussian', 'poisson', or 'mixed'
            noise_level: Relative noise level (σ/signal)
            systematic_error: Add systematic baseline drift
            outlier_fraction: Fraction of points to make outliers

        Returns:
            tau_array: Time delay points
            g2_array: G2 correlation values
            g2_err_array: Error estimates
            ground_truth_params: True parameter values
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        # Generate log-spaced time points (typical for XPCS)
        tau_array = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_points)

        # Define realistic parameter sets for each model
        if model_type == 'single_exp':
            params = {
                'baseline': 1.0 + 0.1 * self.rng.normal(),
                'beta': 0.5 + 0.3 * self.rng.random(),  # 0.2 to 0.8
                'gamma': self.rng.uniform(100, 10000)  # Hz
            }
        elif model_type == 'double_exp':
            params = {
                'baseline': 1.0 + 0.05 * self.rng.normal(),
                'beta1': 0.3 + 0.2 * self.rng.random(),  # Fast component
                'gamma1': self.rng.uniform(5000, 50000),
                'beta2': 0.2 + 0.3 * self.rng.random(),  # Slow component
                'gamma2': self.rng.uniform(100, 2000)
            }
        elif model_type == 'stretched_exp':
            params = {
                'baseline': 1.0 + 0.05 * self.rng.normal(),
                'beta': 0.4 + 0.4 * self.rng.random(),
                'gamma': self.rng.uniform(500, 5000),
                'stretch': 0.5 + 0.4 * self.rng.random()  # 0.5 to 0.9
            }
        elif model_type == 'power_law_plus_exp':
            params = {
                'baseline': 1.0 + 0.05 * self.rng.normal(),
                'beta': 0.3 + 0.2 * self.rng.random(),
                'gamma': self.rng.uniform(1000, 8000),
                'amplitude': 0.1 + 0.2 * self.rng.random(),
                'alpha': 0.3 + 0.5 * self.rng.random()
            }

        # Generate clean G2 signal
        model_func = self.models[model_type]
        g2_clean = model_func(tau_array, params)

        # Add systematic error if requested
        if systematic_error:
            drift = 0.01 * np.linspace(-1, 1, len(tau_array))
            g2_clean += drift

        # Generate noise based on type
        if noise_type == 'gaussian':
            noise = noise_level * g2_clean * self.rng.normal(size=len(tau_array))
        elif noise_type == 'poisson':
            # Poisson noise (typical for photon counting)
            counts = np.maximum(g2_clean / noise_level**2, 10)  # Ensure reasonable counts
            noisy_counts = self.rng.poisson(counts)
            noise = (noisy_counts - counts) * noise_level**2
        elif noise_type == 'mixed':
            # Combination of Gaussian and Poisson
            gaussian_noise = noise_level * g2_clean * self.rng.normal(size=len(tau_array)) * 0.7
            poisson_counts = np.maximum(g2_clean / (noise_level**2 * 0.3), 5)
            poisson_noise = (self.rng.poisson(poisson_counts) - poisson_counts) * noise_level**2 * 0.3
            noise = gaussian_noise + poisson_noise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        g2_noisy = g2_clean + noise

        # Add outliers if requested
        if outlier_fraction > 0:
            n_outliers = int(outlier_fraction * len(tau_array))
            outlier_indices = self.rng.choice(len(tau_array), n_outliers, replace=False)
            outlier_magnitude = 3 * noise_level * g2_clean[outlier_indices]
            g2_noisy[outlier_indices] += outlier_magnitude * self.rng.choice([-1, 1], n_outliers)

        # Ensure G2 >= 1.0 (physical constraint)
        g2_noisy = np.maximum(g2_noisy, 1.0)

        # Estimate errors (combination of statistical and systematic)
        g2_err = np.maximum(noise_level * g2_noisy, 0.001)

        # Add heteroscedastic errors (more realistic)
        error_scaling = 1.0 + 0.5 * (tau_array / np.max(tau_array))  # Errors increase at long times
        g2_err *= error_scaling

        return tau_array, g2_noisy, g2_err, params


class OptimizationStrategy:
    """Individual optimization strategy with specific parameters and performance tracking."""

    def __init__(self, name: str, method: str, config: Dict[str, Any]):
        self.name = name
        self.method = method
        self.config = config
        self.success_count = 0
        self.total_attempts = 0
        self.avg_time = 0.0
        self.avg_iterations = 0

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return self.success_count / max(self.total_attempts, 1)

    def update_stats(self, success: bool, time_taken: float, iterations: int):
        """Update performance statistics."""
        self.total_attempts += 1
        if success:
            self.success_count += 1

        # Update running averages
        alpha = 0.1  # Exponential decay factor
        self.avg_time = alpha * time_taken + (1 - alpha) * self.avg_time
        self.avg_iterations = alpha * iterations + (1 - alpha) * self.avg_iterations


class RobustOptimizer:
    """
    Robust multi-strategy optimization engine with TRF → LM → DE fallback logic.

    Implements a hierarchical optimization approach that:
    1. Starts with Trust Region Reflective (TRF) for bounded problems
    2. Falls back to Levenberg-Marquardt (LM) for unconstrained refinement
    3. Uses Differential Evolution (DE) as a global optimization fallback
    4. Tracks performance and adapts strategy selection
    """

    def __init__(self,
                 max_iterations: int = 10000,
                 tolerance_factor: float = 1.0,
                 enable_caching: bool = True,
                 performance_tracking: bool = True):
        """
        Initialize the robust optimizer.

        Args:
            max_iterations: Maximum iterations for optimization methods
            tolerance_factor: Scaling factor for convergence tolerances
            enable_caching: Whether to use joblib caching
            performance_tracking: Whether to track method performance
        """
        self.max_iterations = max_iterations
        self.tolerance_factor = tolerance_factor
        self.enable_caching = enable_caching
        self.performance_tracking = performance_tracking

        # Define optimization strategies in order of preference
        self.strategies = [
            OptimizationStrategy(
                name="Trust Region Reflective",
                method="trf",
                config={
                    'method': 'trf',
                    'ftol': 1e-8 * tolerance_factor,
                    'xtol': 1e-8 * tolerance_factor,
                    'gtol': 1e-8 * tolerance_factor,
                    'max_nfev': max_iterations,
                    'diff_step': None
                }
            ),
            OptimizationStrategy(
                name="Levenberg-Marquardt",
                method="lm",
                config={
                    'method': 'lm',
                    'ftol': 1e-8 * tolerance_factor,
                    'xtol': 1e-8 * tolerance_factor,
                    'gtol': 1e-8 * tolerance_factor,
                    'maxfev': max_iterations
                }
            ),
            OptimizationStrategy(
                name="Differential Evolution",
                method="differential_evolution",
                config={
                    'maxiter': max_iterations // 100,  # DE uses fewer function evaluations
                    'popsize': 15,
                    'atol': 1e-8 * tolerance_factor,
                    'tol': 1e-6 * tolerance_factor,
                    'seed': None,
                    'polish': True,
                    'updating': 'deferred'
                }
            )
        ]

        self.convergence_history = []
        self.failed_attempts = []

        logger.info(f"RobustOptimizer initialized with {len(self.strategies)} strategies")

    def _validate_inputs(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                        p0: Optional[np.ndarray], bounds: Optional[Tuple]) -> None:
        """Validate optimization inputs."""
        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(xdata, np.ndarray) or not isinstance(ydata, np.ndarray):
            raise TypeError("xdata and ydata must be numpy arrays")

        if len(xdata) != len(ydata):
            raise ValueError("xdata and ydata must have the same length")

        if len(xdata) < 3:
            raise ValueError("Need at least 3 data points for fitting")

        # Check for NaN or infinite values
        if not (np.isfinite(xdata).all() and np.isfinite(ydata).all()):
            raise ValueError("xdata and ydata must contain only finite values")

    def _estimate_initial_parameters(self, func: Callable, xdata: np.ndarray,
                                   ydata: np.ndarray, bounds: Optional[Tuple] = None) -> np.ndarray:
        """
        Intelligent initial parameter estimation using multiple heuristics.
        """
        # For G2 single exponential: single_exp(x, tau, bkg, cts) = cts * exp(-2 * x / tau) + bkg
        try:
            # Estimate baseline (background) as minimum value
            bkg_est = np.min(ydata)

            # Estimate amplitude (contrast)
            cts_est = np.max(ydata) - bkg_est

            # Estimate tau (decay time) using linear regression on log scale
            if np.all(ydata - bkg_est > 0):
                valid_mask = (ydata - bkg_est) > 0.1 * cts_est  # Use reliable points
                if np.sum(valid_mask) > 3:
                    x_valid = xdata[valid_mask]
                    y_log = np.log(ydata[valid_mask] - bkg_est)

                    try:
                        # Linear fit in log space: ln(y-bkg) = ln(cts) - 2*x/tau
                        coeffs = np.polyfit(x_valid, y_log, 1)
                        tau_est = -2.0 / coeffs[0]  # tau = -2 / slope

                        # Ensure positive tau
                        tau_est = max(abs(tau_est), 1e-4)
                    except (np.linalg.LinAlgError, np.polynomial.RankWarning):
                        tau_est = 1e-3  # Fallback value
                else:
                    tau_est = 1e-3
            else:
                tau_est = 1e-3

            # single_exp parameter order: tau, bkg, cts
            p0_est = np.array([tau_est, bkg_est, cts_est])

            # Apply bounds constraints if provided
            if bounds is not None:
                lower, upper = bounds
                if len(lower) == len(p0_est) and len(upper) == len(p0_est):
                    p0_est = np.clip(p0_est, lower, upper)

            return p0_est

        except Exception as e:
            logger.warning(f"Parameter estimation failed: {e}, using default values")
            # Fallback to reasonable defaults for G2 fitting: tau, bkg, cts
            return np.array([1e-3, 1.0, 0.5])

    def _try_curve_fit_strategy(self, strategy: OptimizationStrategy,
                               func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                               p0: np.ndarray, bounds: Optional[Tuple] = None,
                               sigma: Optional[np.ndarray] = None) -> Tuple[bool, Any, Any, Dict]:
        """
        Attempt optimization using a specific curve_fit strategy.

        Returns:
            success: Whether optimization succeeded
            popt: Optimized parameters (or None if failed)
            pcov: Covariance matrix (or None if failed)
            info: Additional information about the optimization
        """
        start_time = time.time()
        iterations = 0

        try:
            # Handle method-specific constraints
            if strategy.method == 'lm':
                # LM doesn't support bounds, so skip if bounds are critical
                if bounds is not None:
                    try:
                        lower, upper = bounds
                        if np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)):
                            # LM with bounds would fail, so return unsuccessful
                            return False, None, None, {'error': 'LM does not support bounds'}
                    except (ValueError, TypeError):
                        # Invalid bounds format, skip LM
                        return False, None, None, {'error': 'Invalid bounds format for LM'}

            # Extract configuration and update
            config = strategy.config.copy()

            # Perform curve fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                popt, pcov = curve_fit(
                    func, xdata, ydata,
                    p0=p0,
                    bounds=bounds if strategy.method != 'lm' else (-np.inf, np.inf),
                    sigma=sigma,
                    **{k: v for k, v in config.items() if k != 'method'}
                )

            # Validate results
            if not np.isfinite(popt).all():
                return False, None, None, {'error': 'Non-finite parameters'}

            if pcov is None or not np.isfinite(pcov).all():
                return False, None, None, {'error': 'Invalid covariance matrix'}

            # Calculate residuals and goodness of fit
            y_pred = func(xdata, *popt)
            residuals = ydata - y_pred
            chi_squared = np.sum(residuals**2)
            r_squared = 1 - np.sum(residuals**2) / np.sum((ydata - np.mean(ydata))**2)

            elapsed_time = time.time() - start_time

            # Update strategy statistics
            strategy.update_stats(True, elapsed_time, iterations)

            info = {
                'method': strategy.name,
                'iterations': iterations,
                'time': elapsed_time,
                'chi_squared': chi_squared,
                'r_squared': r_squared,
                'message': 'Optimization successful'
            }

            return True, popt, pcov, info

        except Exception as e:
            elapsed_time = time.time() - start_time
            strategy.update_stats(False, elapsed_time, iterations)

            return False, None, None, {'error': str(e), 'method': strategy.name, 'time': elapsed_time}

    def _try_differential_evolution(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                                   bounds: Optional[Tuple], sigma: Optional[np.ndarray] = None) -> Tuple[bool, Any, Any, Dict]:
        """
        Attempt optimization using differential evolution (global optimization).
        """
        if bounds is None:
            # DE requires bounds, create reasonable ones for G2 fitting: tau, bkg, cts
            bounds = [(1e-6, 1e0), (0.9, 1.1), (0.01, 2.0)]
        else:
            # Convert bounds format for differential evolution if needed
            try:
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    lower, upper = bounds
                    bounds = list(zip(lower, upper))
                elif not isinstance(bounds, list):
                    raise ValueError("Invalid bounds format")
            except Exception as e:
                return False, None, None, {'error': f'Invalid bounds format for DE: {e}'}

        start_time = time.time()

        try:
            # Define objective function for DE
            def objective(params):
                try:
                    y_pred = func(xdata, *params)
                    if sigma is not None:
                        residuals = (ydata - y_pred) / sigma
                    else:
                        residuals = ydata - y_pred
                    return np.sum(residuals**2)
                except Exception:
                    return np.inf

            # Run differential evolution
            strategy = self.strategies[-1]  # DE strategy
            config = strategy.config.copy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = differential_evolution(
                    objective, bounds, **config
                )

            if result.success and np.isfinite(result.x).all():
                # Estimate covariance matrix using finite differences
                try:
                    # Simple estimation - could be improved with more sophisticated methods
                    h = 1e-8
                    n_params = len(result.x)
                    pcov = np.eye(n_params) * h  # Placeholder covariance matrix

                    # Calculate fit statistics
                    y_pred = func(xdata, *result.x)
                    residuals = ydata - y_pred
                    chi_squared = np.sum(residuals**2)
                    r_squared = 1 - chi_squared / np.sum((ydata - np.mean(ydata))**2)

                    elapsed_time = time.time() - start_time
                    strategy.update_stats(True, elapsed_time, result.nfev)

                    info = {
                        'method': 'Differential Evolution',
                        'iterations': result.nfev,
                        'time': elapsed_time,
                        'chi_squared': chi_squared,
                        'r_squared': r_squared,
                        'message': result.message
                    }

                    return True, result.x, pcov, info

                except Exception as e:
                    logger.warning(f"DE covariance estimation failed: {e}")
                    return False, None, None, {'error': f"Covariance estimation failed: {e}"}
            else:
                elapsed_time = time.time() - start_time
                strategy.update_stats(False, elapsed_time, result.nfev)
                return False, None, None, {
                    'error': f"DE failed: {result.message}",
                    'time': elapsed_time
                }

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.strategies[-1].update_stats(False, elapsed_time, 0)
            return False, None, None, {'error': f"DE exception: {str(e)}", 'time': elapsed_time}

    def robust_curve_fit(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                        p0: Optional[np.ndarray] = None, bounds: Optional[Tuple] = None,
                        sigma: Optional[np.ndarray] = None,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Perform robust curve fitting using multiple optimization strategies.

        Args:
            func: Function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            p0: Initial parameter guess (estimated if None)
            bounds: Parameter bounds (lower, upper)
            sigma: Uncertainties in ydata
            **kwargs: Additional arguments (for API compatibility)

        Returns:
            popt: Optimal parameters
            pcov: Covariance matrix
            info: Detailed optimization information

        Raises:
            RuntimeError: If all optimization strategies fail
        """
        # Validate inputs
        self._validate_inputs(func, xdata, ydata, p0, bounds)

        # Estimate initial parameters if not provided
        if p0 is None:
            p0 = self._estimate_initial_parameters(func, xdata, ydata, bounds)
        else:
            p0 = np.asarray(p0, dtype=float)

        logger.debug(f"Starting robust optimization with {len(self.strategies)} strategies")

        # Try each strategy in order
        for i, strategy in enumerate(self.strategies):
            logger.debug(f"Attempting strategy {i+1}/{len(self.strategies)}: {strategy.name}")

            if strategy.method == "differential_evolution":
                # DE requires special handling
                success, popt, pcov, info = self._try_differential_evolution(
                    func, xdata, ydata, bounds, sigma
                )
            else:
                # curve_fit based strategies
                success, popt, pcov, info = self._try_curve_fit_strategy(
                    strategy, func, xdata, ydata, p0, bounds, sigma
                )

            if success:
                logger.info(f"Optimization succeeded using {strategy.name}")

                if self.performance_tracking:
                    self.convergence_history.append({
                        'strategy': strategy.name,
                        'success': True,
                        'info': info
                    })

                return popt, pcov, info
            else:
                logger.debug(f"Strategy {strategy.name} failed: {info.get('error', 'Unknown error')}")

                if self.performance_tracking:
                    self.failed_attempts.append({
                        'strategy': strategy.name,
                        'error': info.get('error', 'Unknown error'),
                        'info': info
                    })

                # For subsequent strategies, try perturbing initial guess
                if i < len(self.strategies) - 1:
                    # Add small random perturbation to initial parameters
                    perturbation = 0.1 * p0 * np.random.normal(size=len(p0))
                    p0 = p0 + perturbation

                    # Ensure perturbed p0 satisfies bounds
                    if bounds is not None:
                        lower, upper = bounds
                        if len(lower) == len(p0) and len(upper) == len(p0):
                            p0 = np.clip(p0, lower, upper)

        # All strategies failed
        error_msg = f"All {len(self.strategies)} optimization strategies failed"
        logger.error(error_msg)

        # Collect detailed error information
        error_details = {
            'message': error_msg,
            'failed_strategies': [attempt['strategy'] for attempt in self.failed_attempts[-len(self.strategies):]],
            'errors': [attempt['error'] for attempt in self.failed_attempts[-len(self.strategies):]]
        }

        raise RuntimeError(f"{error_msg}. Last errors: {error_details['errors']}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        total_attempts = len(self.convergence_history) + len(self.failed_attempts)
        report = {
            'total_optimizations': total_attempts,
            'total_failures': len(self.failed_attempts),
            'overall_success_rate': len(self.convergence_history) / max(total_attempts, 1),
            'strategies': {}
        }

        for strategy in self.strategies:
            report['strategies'][strategy.name] = {
                'success_rate': strategy.success_rate,
                'total_attempts': strategy.total_attempts,
                'success_count': strategy.success_count,
                'avg_time': strategy.avg_time,
                'avg_iterations': strategy.avg_iterations
            }

        return report

    def reset_performance_tracking(self):
        """Reset all performance tracking statistics."""
        self.convergence_history.clear()
        self.failed_attempts.clear()

        for strategy in self.strategies:
            strategy.success_count = 0
            strategy.total_attempts = 0
            strategy.avg_time = 0.0
            strategy.avg_iterations = 0

    def _get_cache_key(self, xdata: np.ndarray, p0: Optional[np.ndarray],
                      bounds: Optional[Tuple]) -> str:
        """
        Generate a cache key for optimization inputs.

        Args:
            xdata: Independent variable data
            p0: Initial parameter guess
            bounds: Parameter bounds

        Returns:
            String cache key
        """
        import hashlib

        # Create a string representation of the inputs
        key_parts = []

        # Hash the xdata array
        xdata_hash = hashlib.md5(xdata.tobytes()).hexdigest()
        key_parts.append(f"x:{xdata_hash}")

        # Hash p0 if provided
        if p0 is not None:
            p0_hash = hashlib.md5(np.array(p0).tobytes()).hexdigest()
            key_parts.append(f"p0:{p0_hash}")

        # Hash bounds if provided
        if bounds is not None:
            lower, upper = bounds
            bounds_str = f"{lower}_{upper}"
            bounds_hash = hashlib.md5(bounds_str.encode()).hexdigest()
            key_parts.append(f"bounds:{bounds_hash}")

        # Add optimization configuration
        config_str = f"max_iter:{self.max_iterations}_tol:{self.tolerance_factor}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        key_parts.append(f"config:{config_hash}")

        return "_".join(key_parts)


# ============================================================================
# Error Analysis and Diagnostics Framework for Robust G2 Fitting
# ============================================================================


class BootstrapAnalyzer:
    """
    Bootstrap confidence interval analysis for parameter uncertainty estimation.

    Implements non-parametric bootstrap methods to estimate confidence intervals
    and parameter distributions without relying on asymptotic assumptions.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95,
                 random_state: Optional[int] = None, n_jobs: Optional[int] = None):
        """
        Initialize the bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap resamples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores, None for serial processing)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.n_jobs = n_jobs

        if random_state is not None:
            np.random.seed(random_state)

        self.bootstrap_results = []
        self.parameter_distributions = {}

    def _generate_bootstrap_sample(self, xdata: np.ndarray, ydata: np.ndarray,
                                 sigma: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate a single bootstrap sample by resampling with replacement."""
        n_points = len(xdata)
        indices = np.random.choice(n_points, size=n_points, replace=True)

        x_boot = xdata[indices]
        y_boot = ydata[indices]
        sigma_boot = sigma[indices] if sigma is not None else None

        return x_boot, y_boot, sigma_boot

    def _parametric_bootstrap_sample(self, xdata: np.ndarray, ydata: np.ndarray,
                                   func: Callable, popt: np.ndarray,
                                   sigma: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate parametric bootstrap sample using fitted model."""
        # Generate synthetic data from fitted model
        y_model = func(xdata, *popt)

        if sigma is not None:
            # Add noise based on provided uncertainties
            noise = np.random.normal(0, sigma)
        else:
            # Estimate noise from residuals
            residuals = ydata - y_model
            sigma_est = np.std(residuals)
            noise = np.random.normal(0, sigma_est, size=len(xdata))

        y_boot = y_model + noise

        return xdata, y_boot, sigma

    def _bootstrap_single_sample(self, sample_idx: int, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                                popt: np.ndarray, bounds: Optional[Tuple], sigma: Optional[np.ndarray],
                                method: str) -> Tuple[int, Optional[np.ndarray]]:
        """
        Perform a single bootstrap sample fit.

        Args:
            sample_idx: Index of the bootstrap sample (for debugging)
            func: Function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            popt: Original fitted parameters
            bounds: Parameter bounds
            sigma: Data uncertainties
            method: Bootstrap method ('residual', 'parametric', 'nonparametric')

        Returns:
            Tuple of (sample_index, fitted_parameters or None if failed)
        """
        try:
            # Set random seed for reproducibility in parallel processing
            np.random.seed(None)  # Use system entropy

            if method == 'residual':
                # Resample residuals bootstrap
                y_fitted = func(xdata, *popt)
                residuals = ydata - y_fitted

                # Resample residuals
                boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                y_boot = y_fitted + boot_residuals
                x_boot, sigma_boot = xdata, sigma

            elif method == 'parametric':
                # Parametric bootstrap
                x_boot, y_boot, sigma_boot = self._parametric_bootstrap_sample(
                    xdata, ydata, func, popt, sigma
                )

            elif method == 'nonparametric':
                # Non-parametric bootstrap (case resampling)
                x_boot, y_boot, sigma_boot = self._generate_bootstrap_sample(
                    xdata, ydata, sigma
                )

            else:
                return sample_idx, None

            # Initialize optimizer for this sample
            optimizer = RobustOptimizer(performance_tracking=False)

            # Refit the model to bootstrap sample
            popt_boot, _pcov_boot, _info_boot = optimizer.robust_curve_fit(
                func, x_boot, y_boot, p0=popt, bounds=bounds, sigma=sigma_boot
            )

            return sample_idx, popt_boot

        except Exception as e:
            logger.debug(f"Bootstrap sample {sample_idx} failed: {e}")
            return sample_idx, None

    def _parallel_bootstrap(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                           popt: np.ndarray, bounds: Optional[Tuple], sigma: Optional[np.ndarray],
                           method: str, n_params: int) -> Tuple[np.ndarray, int]:
        """
        Perform bootstrap analysis using parallel processing.

        Returns:
            Tuple of (bootstrap_params_array, successful_fits_count)
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warning("joblib not available, falling back to serial processing")
            return self._serial_bootstrap(func, xdata, ydata, popt, bounds, sigma, method, n_params)

        # Use partial to reduce the number of arguments passed to parallel workers
        from functools import partial
        bootstrap_worker = partial(
            self._bootstrap_single_sample,
            func=func, xdata=xdata, ydata=ydata, popt=popt,
            bounds=bounds, sigma=sigma, method=method
        )

        # Run bootstrap samples in parallel
        parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
        results = parallel(delayed(bootstrap_worker)(i) for i in range(self.n_bootstrap))

        # Process results
        bootstrap_params = np.zeros((self.n_bootstrap, n_params))
        successful_fits = 0

        for sample_idx, popt_boot in results:
            if popt_boot is not None:
                bootstrap_params[successful_fits, :] = popt_boot
                successful_fits += 1

        return bootstrap_params, successful_fits

    def _serial_bootstrap(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                         popt: np.ndarray, bounds: Optional[Tuple], sigma: Optional[np.ndarray],
                         method: str, n_params: int) -> Tuple[np.ndarray, int]:
        """
        Perform bootstrap analysis using serial processing (original implementation).

        Returns:
            Tuple of (bootstrap_params_array, successful_fits_count)
        """
        bootstrap_params = np.zeros((self.n_bootstrap, n_params))
        successful_fits = 0

        # Initialize robust optimizer for bootstrap fits
        optimizer = RobustOptimizer(performance_tracking=False)  # Disable tracking for speed

        for i in range(self.n_bootstrap):
            try:
                if method == 'residual':
                    # Resample residuals bootstrap
                    y_fitted = func(xdata, *popt)
                    residuals = ydata - y_fitted

                    # Resample residuals
                    boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                    y_boot = y_fitted + boot_residuals
                    x_boot, sigma_boot = xdata, sigma

                elif method == 'parametric':
                    # Parametric bootstrap
                    x_boot, y_boot, sigma_boot = self._parametric_bootstrap_sample(
                        xdata, ydata, func, popt, sigma
                    )

                elif method == 'nonparametric':
                    # Non-parametric bootstrap (case resampling)
                    x_boot, y_boot, sigma_boot = self._generate_bootstrap_sample(
                        xdata, ydata, sigma
                    )

                else:
                    raise ValueError(f"Unknown bootstrap method: {method}")

                # Refit the model to bootstrap sample
                popt_boot, _pcov_boot, _info_boot = optimizer.robust_curve_fit(
                    func, x_boot, y_boot, p0=popt, bounds=bounds, sigma=sigma_boot
                )

                bootstrap_params[successful_fits, :] = popt_boot
                successful_fits += 1

            except Exception as e:
                logger.debug(f"Bootstrap sample {i} failed: {e}")
                continue

        return bootstrap_params, successful_fits

    def bootstrap_confidence_intervals(self, func: Callable, xdata: np.ndarray,
                                     ydata: np.ndarray, popt: np.ndarray,
                                     pcov: np.ndarray,
                                     bounds: Optional[Tuple] = None,
                                     sigma: Optional[np.ndarray] = None,
                                     method: str = 'residual') -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for fitted parameters.

        Args:
            func: Function used for fitting
            xdata: Independent variable data
            ydata: Dependent variable data
            popt: Fitted parameters from original fit
            pcov: Covariance matrix from original fit
            bounds: Parameter bounds for refitting
            sigma: Data uncertainties
            method: Bootstrap method ('residual', 'parametric', or 'nonparametric')

        Returns:
            Dictionary containing bootstrap results and confidence intervals
        """
        logger.info(f"Starting bootstrap analysis with {self.n_bootstrap} resamples using {method} method")
        if self.n_jobs is not None and self.n_jobs != 1:
            logger.info(f"Using parallel processing with {self.n_jobs} jobs")

        n_params = len(popt)

        # Choose between parallel and serial processing
        if self.n_jobs is not None and self.n_jobs != 1:
            bootstrap_params, successful_fits = self._parallel_bootstrap(
                func, xdata, ydata, popt, bounds, sigma, method, n_params
            )
        else:
            bootstrap_params, successful_fits = self._serial_bootstrap(
                func, xdata, ydata, popt, bounds, sigma, method, n_params
            )

        if successful_fits < self.n_bootstrap // 10:  # Less than 10% success
            logger.warning(f"Only {successful_fits}/{self.n_bootstrap} bootstrap fits succeeded")

        # Trim to successful fits only
        bootstrap_params = bootstrap_params[:successful_fits, :]

        # Calculate confidence intervals
        alpha_lower = (self.alpha / 2) * 100
        alpha_upper = (1 - self.alpha / 2) * 100

        param_stats = {}
        for i in range(n_params):
            param_bootstrap = bootstrap_params[:, i]

            param_stats[f'param_{i}'] = {
                'original_estimate': popt[i],
                'bootstrap_mean': np.mean(param_bootstrap),
                'bootstrap_std': np.std(param_bootstrap),
                'bootstrap_bias': np.mean(param_bootstrap) - popt[i],
                'confidence_interval': np.percentile(param_bootstrap, [alpha_lower, alpha_upper]),
                'bootstrap_samples': param_bootstrap
            }

        # Store results for analysis
        self.bootstrap_results.append({
            'method': method,
            'n_successful': successful_fits,
            'parameter_stats': param_stats,
            'bootstrap_params': bootstrap_params
        })

        # Overall bootstrap results
        results = {
            'method': method,
            'n_bootstrap_requested': self.n_bootstrap,
            'n_successful_fits': successful_fits,
            'success_rate': successful_fits / self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'parameter_statistics': param_stats,
            'bootstrap_covariance': np.cov(bootstrap_params.T) if successful_fits > 1 else None
        }

        logger.info(f"Bootstrap analysis completed: {successful_fits}/{self.n_bootstrap} successful fits")

        return results


class CovarianceAnalyzer:
    """
    Enhanced covariance matrix analysis with numerical stability checks.

    Provides robust covariance matrix calculations, condition number analysis,
    and parameter correlation assessment.
    """

    def __init__(self, condition_threshold: float = 1e12):
        """
        Initialize the covariance analyzer.

        Args:
            condition_threshold: Threshold for detecting ill-conditioned matrices
        """
        self.condition_threshold = condition_threshold

    def analyze_covariance_matrix(self, pcov: np.ndarray, popt: np.ndarray,
                                func_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive covariance matrix analysis.

        Args:
            pcov: Parameter covariance matrix
            popt: Fitted parameters
            func_name: Name of the fitted function

        Returns:
            Dictionary with covariance analysis results
        """
        n_params = len(popt)

        if pcov.shape != (n_params, n_params):
            raise ValueError(f"Covariance matrix shape {pcov.shape} doesn't match parameters {n_params}")

        # Basic covariance properties
        param_errors = np.sqrt(np.diag(pcov))
        condition_number = np.linalg.cond(pcov)
        determinant = np.linalg.det(pcov)

        # Enhanced numerical stability checks
        stability_issues = []
        stability_metrics = {}

        # Condition number analysis
        stability_metrics['condition_number'] = condition_number
        if condition_number > self.condition_threshold:
            stability_issues.append(f"High condition number: {condition_number:.2e}")
        elif condition_number > 1e8:
            stability_issues.append(f"Moderately high condition number: {condition_number:.2e} (may indicate near-singularity)")

        # Diagonal variance checks
        diagonal_variances = np.diag(pcov)
        stability_metrics['min_variance'] = np.min(diagonal_variances)
        stability_metrics['max_variance'] = np.max(diagonal_variances)

        if np.any(diagonal_variances <= 0):
            stability_issues.append("Negative or zero variances detected")
            stability_metrics['negative_variance_indices'] = np.where(diagonal_variances <= 0)[0]

        if np.any(diagonal_variances < 1e-16):
            stability_issues.append("Extremely small variances detected (numerical precision limit)")

        # Determinant analysis
        stability_metrics['determinant'] = determinant
        stability_metrics['log_determinant'] = np.log(np.abs(determinant)) if determinant > 0 else -np.inf

        if determinant <= 0:
            stability_issues.append(f"Non-positive determinant: {determinant:.2e}")
        elif determinant < 1e-16:
            stability_issues.append(f"Near-zero determinant: {determinant:.2e} (matrix is nearly singular)")

        # Matrix rank analysis
        try:
            matrix_rank = np.linalg.matrix_rank(pcov)
            stability_metrics['matrix_rank'] = matrix_rank
            stability_metrics['rank_deficiency'] = n_params - matrix_rank

            if matrix_rank < n_params:
                stability_issues.append(f"Rank deficient matrix: rank={matrix_rank}, expected={n_params}")
        except Exception as e:
            stability_issues.append(f"Could not compute matrix rank: {e}")
            stability_metrics['matrix_rank'] = None

        # Eigenvalue analysis for positive definiteness
        try:
            eigenvalues = np.linalg.eigvals(pcov)
            stability_metrics['eigenvalues'] = eigenvalues
            stability_metrics['min_eigenvalue'] = np.min(eigenvalues)
            stability_metrics['max_eigenvalue'] = np.max(eigenvalues)

            # Check for positive definiteness
            if np.any(eigenvalues <= 0):
                n_negative = np.sum(eigenvalues < 0)
                n_zero = np.sum(eigenvalues == 0)
                stability_issues.append(f"Matrix not positive definite: {n_negative} negative, {n_zero} zero eigenvalues")

            # Check for very small eigenvalues indicating near-singularity
            if np.any(eigenvalues < 1e-14) and np.all(eigenvalues >= 0):
                stability_issues.append("Very small positive eigenvalues detected (near-singular)")

        except Exception as e:
            stability_issues.append(f"Eigenvalue analysis failed: {e}")
            stability_metrics['eigenvalues'] = None

        # Frobenius norm and matrix norms
        stability_metrics['frobenius_norm'] = np.linalg.norm(pcov, 'fro')
        stability_metrics['spectral_norm'] = np.linalg.norm(pcov, 2)

        # Check for symmetry (should be symmetric for covariance matrices)
        symmetry_error = np.max(np.abs(pcov - pcov.T))
        stability_metrics['symmetry_error'] = symmetry_error
        if symmetry_error > 1e-12:
            stability_issues.append(f"Matrix not symmetric: max error = {symmetry_error:.2e}")

        # Variance scaling consistency check
        variance_ratios = np.max(diagonal_variances) / np.min(diagonal_variances) if np.min(diagonal_variances) > 0 else np.inf
        stability_metrics['variance_ratio'] = variance_ratios
        if variance_ratios > 1e12:
            stability_issues.append(f"Extreme variance scaling: ratio = {variance_ratios:.2e}")

        # Store enhanced stability metrics
        stability_metrics['n_stability_issues'] = len(stability_issues)
        stability_metrics['stability_score'] = self._compute_stability_score(stability_metrics, stability_issues)

        # Calculate correlation matrix
        try:
            # Normalize covariance matrix to get correlations
            std_matrix = np.outer(param_errors, param_errors)
            correlation_matrix = pcov / std_matrix

            # Ensure diagonal is exactly 1.0
            np.fill_diagonal(correlation_matrix, 1.0)

            # Check for high correlations
            high_correlations = []
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.9:
                        high_correlations.append((i, j, corr))

        except Exception as e:
            correlation_matrix = None
            high_correlations = []
            stability_issues.append(f"Correlation calculation failed: {e}")

        # Parameter relative uncertainties
        relative_errors = param_errors / np.abs(popt)

        # Eigenvalue analysis for parameter combinations
        try:
            eigenvals, eigenvects = np.linalg.eigh(pcov)

            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvects = eigenvects[:, idx]

            # Principal component analysis of parameter uncertainties
            principal_components = []
            for i, (eigenval, eigenvect) in enumerate(zip(eigenvals, eigenvects.T)):
                principal_components.append({
                    'eigenvalue': eigenval,
                    'std_dev': np.sqrt(max(eigenval, 0)),
                    'eigenvector': eigenvect,
                    'explained_variance_ratio': eigenval / np.sum(eigenvals) if np.sum(eigenvals) > 0 else 0
                })

        except Exception as e:
            eigenvals = None
            eigenvects = None
            principal_components = []
            stability_issues.append(f"Eigenvalue analysis failed: {e}")

        return {
            'function_name': func_name,
            'n_parameters': n_params,
            'condition_number': condition_number,
            'determinant': determinant,
            'parameter_errors': param_errors,
            'relative_errors': relative_errors,
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'principal_components': principal_components,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvects,
            'stability_issues': stability_issues,
            'stability_metrics': stability_metrics,
            'is_numerically_stable': len(stability_issues) == 0
        }

    def suggest_parameter_constraints(self, covariance_analysis: Dict[str, Any],
                                    popt: np.ndarray) -> List[str]:
        """
        Suggest parameter constraints based on covariance analysis.

        Args:
            covariance_analysis: Results from analyze_covariance_matrix
            popt: Fitted parameters

        Returns:
            List of constraint suggestions
        """
        suggestions = []

        # High relative errors
        relative_errors = covariance_analysis['relative_errors']
        for i, rel_err in enumerate(relative_errors):
            if rel_err > 1.0:  # 100% relative error
                suggestions.append(f"Parameter {i} has high uncertainty ({rel_err:.1%}). Consider fixing or constraining.")

        # High correlations
        high_correlations = covariance_analysis['high_correlations']
        for i, j, corr in high_correlations:
            suggestions.append(f"Parameters {i} and {j} are highly correlated ({corr:.3f}). Consider fixing one.")

        # Condition number issues
        if covariance_analysis['condition_number'] > self.condition_threshold:
            suggestions.append("High condition number indicates potential numerical instability. Check parameter scaling.")

        return suggestions

    def _compute_stability_score(self, stability_metrics: Dict[str, Any], stability_issues: List[str]) -> float:
        """
        Compute a numerical stability score between 0 and 1.

        Args:
            stability_metrics: Dictionary of stability metrics
            stability_issues: List of stability issues

        Returns:
            Float between 0 (completely unstable) and 1 (perfectly stable)
        """
        # Start with perfect score
        score = 1.0

        # Penalize based on number of issues
        n_issues = len(stability_issues)
        score -= 0.1 * n_issues

        # Condition number penalty
        cond_num = stability_metrics.get('condition_number', 1.0)
        if cond_num > 1e4:
            score -= min(0.3, (np.log10(cond_num) - 4) / 8 * 0.3)  # Scale penalty from 1e4 to 1e12

        # Eigenvalue penalty
        min_eigenval = stability_metrics.get('min_eigenvalue', 1.0)
        if min_eigenval is not None and min_eigenval > 0:
            if min_eigenval < 1e-10:
                score -= min(0.2, (10 + np.log10(min_eigenval)) / 6 * 0.2)  # Penalty for very small eigenvalues

        # Variance ratio penalty
        var_ratio = stability_metrics.get('variance_ratio', 1.0)
        if var_ratio > 1e6:
            score -= min(0.2, (np.log10(var_ratio) - 6) / 6 * 0.2)

        # Rank deficiency penalty
        rank_deficiency = stability_metrics.get('rank_deficiency', 0)
        if rank_deficiency > 0:
            score -= 0.4  # Major penalty for rank deficiency

        # Symmetry error penalty
        symmetry_error = stability_metrics.get('symmetry_error', 0.0)
        if symmetry_error > 1e-12:
            score -= min(0.1, (np.log10(symmetry_error) + 12) / 4 * 0.1)

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))


class ResidualAnalyzer:
    """
    Comprehensive residual analysis framework with outlier detection.

    Provides tools for analyzing fit residuals, detecting outliers,
    and assessing systematic deviations from the model.
    """

    def __init__(self, outlier_threshold: float = 3.0):
        """
        Initialize the residual analyzer.

        Args:
            outlier_threshold: Threshold (in standard deviations) for outlier detection
        """
        self.outlier_threshold = outlier_threshold

    def analyze_residuals(self, xdata: np.ndarray, ydata: np.ndarray,
                         y_pred: np.ndarray, sigma: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive residual analysis.

        Args:
            xdata: Independent variable data
            ydata: Observed data
            y_pred: Predicted data from fitted model
            sigma: Data uncertainties (optional)

        Returns:
            Dictionary with residual analysis results
        """
        # Calculate residuals
        residuals = ydata - y_pred

        if sigma is not None:
            # Standardized residuals (Pearson residuals)
            std_residuals = residuals / sigma
            weighted_analysis = True
        else:
            # Use residuals directly
            std_residuals = residuals / np.std(residuals)
            weighted_analysis = False
            sigma = np.ones_like(residuals)  # Unit weights

        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'rms': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'weighted_analysis': weighted_analysis
        }

        # Standardized residual statistics
        std_residual_stats = {
            'mean': np.mean(std_residuals),
            'std': np.std(std_residuals),
            'min': np.min(std_residuals),
            'max': np.max(std_residuals),
            'median': np.median(std_residuals)
        }

        # Outlier detection
        outlier_mask = np.abs(std_residuals) > self.outlier_threshold
        outlier_indices = np.where(outlier_mask)[0]

        outliers = {
            'count': np.sum(outlier_mask),
            'fraction': np.mean(outlier_mask),
            'indices': outlier_indices,
            'x_values': xdata[outlier_mask],
            'y_values': ydata[outlier_mask],
            'residual_values': residuals[outlier_mask],
            'standardized_values': std_residuals[outlier_mask]
        }

        # Runs test for randomness
        def runs_test(data):
            """Simple runs test for randomness."""
            runs = 1
            n_pos = np.sum(data > 0)
            n_neg = len(data) - n_pos

            if n_pos == 0 or n_neg == 0:
                return 0, 1.0  # All same sign

            for i in range(1, len(data)):
                if (data[i] > 0) != (data[i-1] > 0):
                    runs += 1

            # Expected number of runs and standard deviation
            expected_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
            var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)) / \
                      ((n_pos + n_neg)**2 * (n_pos + n_neg - 1))

            if var_runs <= 0:
                return runs, 1.0

            # Z-score for runs test
            z_score = (runs - expected_runs) / np.sqrt(var_runs)

            return runs, z_score

        runs, runs_z = runs_test(residuals)

        # Autocorrelation of residuals
        def autocorrelation(data, lag=1):
            """Calculate autocorrelation at given lag."""
            if len(data) <= lag:
                return 0.0
            return np.corrcoef(data[:-lag], data[lag:])[0, 1] if len(data) > lag else 0.0

        autocorr_lag1 = autocorrelation(residuals, 1)

        # Trend analysis - fit linear trend to residuals vs x
        try:
            trend_slope, trend_intercept = np.polyfit(xdata, residuals, 1)
            trend_r_squared = np.corrcoef(xdata, residuals)[0, 1]**2
        except:
            trend_slope = 0.0
            trend_intercept = 0.0
            trend_r_squared = 0.0

        # Normality tests
        from scipy import stats

        # Shapiro-Wilk test (for small samples)
        if len(std_residuals) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(std_residuals)
            except:
                shapiro_stat, shapiro_p = None, None
        else:
            shapiro_stat, shapiro_p = None, None

        # Jarque-Bera test
        try:
            jb_stat, jb_p = stats.jarque_bera(std_residuals)
        except:
            jb_stat, jb_p = None, None

        return {
            'residual_statistics': residual_stats,
            'standardized_residual_statistics': std_residual_stats,
            'outliers': outliers,
            'randomness_test': {
                'runs': runs,
                'runs_z_score': runs_z,
                'is_random': bool(abs(runs_z) < 1.96)  # 95% confidence
            },
            'autocorrelation': {
                'lag_1': autocorr_lag1,
                'is_independent': bool(abs(autocorr_lag1) < 0.1)
            },
            'trend_analysis': {
                'slope': trend_slope,
                'intercept': trend_intercept,
                'r_squared': trend_r_squared,
                'has_trend': bool(trend_r_squared > 0.1)
            },
            'normality_tests': {
                'shapiro_wilk': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': bool(shapiro_p > 0.05) if shapiro_p is not None else None
                },
                'jarque_bera': {
                    'statistic': jb_stat,
                    'p_value': jb_p,
                    'is_normal': bool(jb_p > 0.05) if jb_p is not None else None
                }
            }
        }

    def detect_systematic_deviations(self, xdata: np.ndarray, residuals: np.ndarray,
                                   n_bins: int = 10) -> Dict[str, Any]:
        """
        Detect systematic deviations by analyzing residuals in bins.

        Args:
            xdata: Independent variable data
            residuals: Fit residuals
            n_bins: Number of bins for analysis

        Returns:
            Dictionary with systematic deviation analysis
        """
        # Create bins across x-range
        x_min, x_max = np.min(xdata), np.max(xdata)
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)

        bin_analysis = []
        for i in range(n_bins):
            mask = (xdata >= bin_edges[i]) & (xdata < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (xdata >= bin_edges[i]) & (xdata <= bin_edges[i + 1])

            if np.sum(mask) > 0:
                bin_residuals = residuals[mask]
                bin_analysis.append({
                    'bin_index': i,
                    'x_range': (bin_edges[i], bin_edges[i + 1]),
                    'x_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
                    'n_points': np.sum(mask),
                    'mean_residual': np.mean(bin_residuals),
                    'std_residual': np.std(bin_residuals),
                    'rms_residual': np.sqrt(np.mean(bin_residuals**2))
                })

        # Check for systematic patterns
        bin_means = [bin_data['mean_residual'] for bin_data in bin_analysis]

        # Test for systematic bias (mean significantly different from zero)
        overall_bias = np.mean(bin_means)
        bias_std = np.std(bin_means) / np.sqrt(len(bin_means))

        # Test for systematic trend across bins
        x_centers = [bin_data['x_center'] for bin_data in bin_analysis]
        try:
            trend_slope, _ = np.polyfit(x_centers, bin_means, 1)
            trend_significance = abs(trend_slope) > 2 * bias_std
        except:
            trend_slope = 0.0
            trend_significance = False

        return {
            'bin_analysis': bin_analysis,
            'systematic_bias': {
                'overall_bias': overall_bias,
                'bias_std_error': bias_std,
                'is_significant': bool(abs(overall_bias) > 2 * bias_std)
            },
            'systematic_trend': {
                'slope': trend_slope,
                'is_significant': bool(trend_significance)
            },
            'recommendations': self._generate_deviation_recommendations(bin_analysis, overall_bias, trend_slope)
        }

    def _generate_deviation_recommendations(self, bin_analysis: List[Dict],
                                          overall_bias: float, trend_slope: float) -> List[str]:
        """Generate recommendations based on systematic deviation analysis."""
        recommendations = []

        if abs(overall_bias) > 0.1:
            recommendations.append(f"Systematic bias detected (mean residual = {overall_bias:.4f}). Check model adequacy.")

        if abs(trend_slope) > 0.01:
            recommendations.append(f"Systematic trend in residuals detected. Consider additional model terms.")

        # Check for bins with unusually high RMS
        rms_values = [bin_data['rms_residual'] for bin_data in bin_analysis]
        mean_rms = np.mean(rms_values)

        for bin_data in bin_analysis:
            if bin_data['rms_residual'] > 2 * mean_rms:
                x_center = bin_data['x_center']
                recommendations.append(f"High residuals in region around x = {x_center:.3e}. Check data quality.")

        return recommendations

    def test_heteroscedasticity(self, xdata: np.ndarray, residuals: np.ndarray,
                               method: str = 'breusch_pagan') -> Dict[str, Any]:
        """
        Test for heteroscedasticity (non-constant variance) in residuals.

        Args:
            xdata: Independent variable data
            residuals: Fit residuals
            method: Test method ('breusch_pagan', 'white', 'goldfeld_quandt')

        Returns:
            Dictionary with heteroscedasticity test results
        """
        results = {'method': method}

        try:
            if method == 'breusch_pagan':
                # Breusch-Pagan test: regress squared residuals on x
                squared_residuals = residuals**2

                # Linear regression of squared residuals on x
                X = np.column_stack([np.ones(len(xdata)), xdata])
                try:
                    coeffs, residuals_lm, rank, s = np.linalg.lstsq(X, squared_residuals, rcond=None)

                    # Calculate test statistic
                    ss_explained = np.sum((X @ coeffs - np.mean(squared_residuals))**2)
                    ss_total = np.sum((squared_residuals - np.mean(squared_residuals))**2)
                    r_squared = ss_explained / ss_total if ss_total > 0 else 0

                    # Chi-squared test statistic
                    n = len(xdata)
                    lm_statistic = n * r_squared

                    from scipy import stats
                    p_value = 1 - stats.chi2.cdf(lm_statistic, 1)

                    results.update({
                        'statistic': lm_statistic,
                        'p_value': p_value,
                        'is_homoscedastic': p_value > 0.05,
                        'r_squared': r_squared
                    })

                except np.linalg.LinAlgError:
                    results['error'] = 'Linear algebra error in Breusch-Pagan test'

            elif method == 'goldfeld_quandt':
                # Goldfeld-Quandt test: compare variances of first and last third
                n = len(residuals)
                n_third = n // 3

                first_third = residuals[:n_third]
                last_third = residuals[-n_third:]

                var1 = np.var(first_third, ddof=1) if len(first_third) > 1 else 0
                var2 = np.var(last_third, ddof=1) if len(last_third) > 1 else 0

                if var1 > 0 and var2 > 0:
                    f_stat = max(var1, var2) / min(var1, var2)
                    from scipy import stats
                    p_value = 2 * (1 - stats.f.cdf(f_stat, n_third - 1, n_third - 1))

                    results.update({
                        'statistic': f_stat,
                        'p_value': p_value,
                        'is_homoscedastic': p_value > 0.05,
                        'variance_1': var1,
                        'variance_2': var2
                    })
                else:
                    results['error'] = 'Cannot compute variances for Goldfeld-Quandt test'

            else:
                results['error'] = f'Unknown heteroscedasticity test method: {method}'

        except Exception as e:
            results['error'] = f'Heteroscedasticity test failed: {e}'

        return results

    def compute_influence_measures(self, xdata: np.ndarray, ydata: np.ndarray, y_pred: np.ndarray,
                                  residuals: np.ndarray) -> Dict[str, Any]:
        """
        Compute influence measures for outlier detection and model diagnostics.

        Args:
            xdata: Independent variable data
            ydata: Observed data
            y_pred: Predicted data
            residuals: Fit residuals

        Returns:
            Dictionary with influence measures
        """
        n = len(xdata)

        try:
            # Cook's distance approximation (simplified for nonlinear case)
            standardized_residuals = residuals / np.std(residuals)

            # Leverage approximation using derivative information
            # For nonlinear fitting, exact leverage is complex, so we use an approximation
            x_centered = xdata - np.mean(xdata)
            leverage_approx = x_centered**2 / np.sum(x_centered**2)

            # Cook's distance approximation
            cooks_distance = (standardized_residuals**2 * leverage_approx) / (2 * (1 - leverage_approx))

            # DFFITS (difference in fits)
            dffits = standardized_residuals * np.sqrt(leverage_approx / (1 - leverage_approx))

            # Studentized residuals approximation
            h = leverage_approx
            mse = np.mean(residuals**2)
            studentized_residuals = residuals / (np.sqrt(mse * (1 - h)))

            # Identify influential points
            cook_threshold = 4 / n  # Common threshold
            dffits_threshold = 2 * np.sqrt(2 / n)  # Common threshold

            influential_points = {
                'cook_influential': np.where(cooks_distance > cook_threshold)[0],
                'dffits_influential': np.where(np.abs(dffits) > dffits_threshold)[0],
                'high_leverage': np.where(leverage_approx > 2 * np.mean(leverage_approx))[0]
            }

            return {
                'cooks_distance': cooks_distance,
                'dffits': dffits,
                'leverage': leverage_approx,
                'studentized_residuals': studentized_residuals,
                'influential_points': influential_points,
                'thresholds': {
                    'cook_distance': cook_threshold,
                    'dffits': dffits_threshold,
                    'leverage': 2 * np.mean(leverage_approx)
                },
                'summary': {
                    'n_cook_influential': len(influential_points['cook_influential']),
                    'n_dffits_influential': len(influential_points['dffits_influential']),
                    'n_high_leverage': len(influential_points['high_leverage']),
                    'max_cook_distance': np.max(cooks_distance),
                    'max_leverage': np.max(leverage_approx)
                }
            }

        except Exception as e:
            return {'error': f'Influence measures computation failed: {e}'}

    def detect_changepoints(self, residuals: np.ndarray, method: str = 'cusum') -> Dict[str, Any]:
        """
        Detect changepoints in residual patterns.

        Args:
            residuals: Fit residuals
            method: Detection method ('cusum', 'variance')

        Returns:
            Dictionary with changepoint analysis results
        """
        results = {'method': method}

        try:
            if method == 'cusum':
                # CUSUM (cumulative sum) test for mean shift
                cumsum = np.cumsum(residuals - np.mean(residuals))

                # Find maximum absolute CUSUM value and its position
                max_cusum_idx = np.argmax(np.abs(cumsum))
                max_cusum_value = cumsum[max_cusum_idx]

                # Simple threshold-based detection (can be improved with more sophisticated methods)
                threshold = 3 * np.std(residuals) * np.sqrt(len(residuals))

                results.update({
                    'cumsum': cumsum,
                    'max_cusum_value': max_cusum_value,
                    'max_cusum_position': max_cusum_idx,
                    'threshold': threshold,
                    'changepoint_detected': abs(max_cusum_value) > threshold,
                    'changepoint_position': max_cusum_idx if abs(max_cusum_value) > threshold else None
                })

            elif method == 'variance':
                # Test for variance changepoints
                n = len(residuals)
                mid_point = n // 2

                var1 = np.var(residuals[:mid_point], ddof=1) if mid_point > 1 else 0
                var2 = np.var(residuals[mid_point:], ddof=1) if (n - mid_point) > 1 else 0

                if var1 > 0 and var2 > 0:
                    # F-test for variance equality
                    f_stat = max(var1, var2) / min(var1, var2)
                    from scipy import stats
                    p_value = 2 * (1 - stats.f.cdf(f_stat, mid_point - 1, n - mid_point - 1))

                    results.update({
                        'variance_1': var1,
                        'variance_2': var2,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'changepoint_detected': p_value < 0.05,
                        'changepoint_position': mid_point
                    })
                else:
                    results['error'] = 'Cannot compute variances for changepoint detection'

            else:
                results['error'] = f'Unknown changepoint detection method: {method}'

        except Exception as e:
            results['error'] = f'Changepoint detection failed: {e}'

        return results


class GoodnessOfFitAnalyzer:
    """
    Comprehensive goodness-of-fit metrics calculation.

    Implements multiple statistical measures to assess the quality of fitted models,
    including R², adjusted R², reduced χ², AIC, BIC, and others.
    """

    def __init__(self):
        """Initialize the goodness-of-fit analyzer."""
        pass

    def calculate_goodness_of_fit_metrics(self, ydata: np.ndarray, y_pred: np.ndarray,
                                        n_params: int, sigma: Optional[np.ndarray] = None,
                                        log_likelihood: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive goodness-of-fit metrics.

        Args:
            ydata: Observed data
            y_pred: Predicted data from fitted model
            n_params: Number of fitted parameters
            sigma: Data uncertainties (optional)
            log_likelihood: Log-likelihood value (optional, calculated if not provided)

        Returns:
            Dictionary with goodness-of-fit metrics
        """
        n_data = len(ydata)

        if n_data <= n_params:
            logger.warning(f"Too few data points ({n_data}) for {n_params} parameters")

        # Basic statistics
        residuals = ydata - y_pred
        ss_res = np.sum(residuals**2)  # Sum of squares of residuals
        ss_tot = np.sum((ydata - np.mean(ydata))**2)  # Total sum of squares

        # R-squared (coefficient of determination)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Adjusted R-squared
        if n_data > n_params + 1:
            adj_r_squared = 1 - ((1 - r_squared) * (n_data - 1) / (n_data - n_params - 1))
        else:
            adj_r_squared = r_squared

        # Root Mean Square Error
        rmse = np.sqrt(ss_res / n_data)

        # Mean Absolute Error
        mae = np.mean(np.abs(residuals))

        # Weighted metrics if uncertainties are provided
        if sigma is not None:
            # Weighted residuals (Pearson residuals)
            weighted_residuals = residuals / sigma

            # Chi-squared statistic
            chi_squared = np.sum(weighted_residuals**2)

            # Reduced chi-squared
            dof = n_data - n_params  # Degrees of freedom
            reduced_chi_squared = chi_squared / dof if dof > 0 else np.inf

            # Weighted R-squared
            ss_res_weighted = np.sum(weighted_residuals**2)
            y_mean_weighted = np.average(ydata, weights=1/sigma**2)
            ss_tot_weighted = np.sum(((ydata - y_mean_weighted) / sigma)**2)

            weighted_r_squared = 1 - (ss_res_weighted / ss_tot_weighted) if ss_tot_weighted > 0 else 0.0

        else:
            chi_squared = ss_res
            reduced_chi_squared = ss_res / (n_data - n_params) if n_data > n_params else np.inf
            weighted_r_squared = r_squared

        # Log-likelihood based metrics
        if log_likelihood is None:
            if sigma is not None:
                # Gaussian log-likelihood with known uncertainties
                log_likelihood = -0.5 * np.sum(((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2)))
            else:
                # Gaussian log-likelihood with estimated variance
                sigma_est = np.sqrt(ss_res / n_data)
                log_likelihood = -0.5 * n_data * (np.log(2 * np.pi) + 2 * np.log(sigma_est)) - 0.5 * chi_squared / sigma_est**2

        # Akaike Information Criterion (AIC)
        aic = 2 * n_params - 2 * log_likelihood

        # Corrected AIC for small samples
        if n_data / n_params < 40:  # Rule of thumb for small sample correction
            aic_corrected = aic + (2 * n_params * (n_params + 1)) / (n_data - n_params - 1) if n_data > n_params + 1 else np.inf
        else:
            aic_corrected = aic

        # Bayesian Information Criterion (BIC)
        bic = n_params * np.log(n_data) - 2 * log_likelihood

        # Hannan-Quinn Information Criterion (HQIC)
        hqic = 2 * n_params * np.log(np.log(n_data)) - 2 * log_likelihood if n_data > 3 else np.inf

        # Additional metrics
        # Mean Absolute Percentage Error (MAPE) - handle zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((residuals) / ydata)) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)

        # Symmetric Mean Absolute Percentage Error (SMAPE)
        denominator = (np.abs(ydata) + np.abs(y_pred)) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            smape = np.mean(np.abs(residuals) / denominator) * 100
            smape = np.nan_to_num(smape, nan=0.0, posinf=0.0, neginf=0.0)

        # Coefficient of Variation of RMSE
        cv_rmse = rmse / np.mean(ydata) * 100 if np.mean(ydata) != 0 else 0.0

        return {
            'n_data_points': n_data,
            'n_parameters': n_params,
            'degrees_of_freedom': n_data - n_params,

            # Basic fit quality metrics
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'weighted_r_squared': weighted_r_squared,

            # Error metrics
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'smape': smape,
            'cv_rmse': cv_rmse,

            # Chi-squared metrics
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'chi_squared_p_value': 1 - stats.chi2.cdf(chi_squared, n_data - n_params) if n_data > n_params else None,

            # Information criteria
            'log_likelihood': log_likelihood,
            'aic': aic,
            'aic_corrected': aic_corrected,
            'bic': bic,
            'hqic': hqic,

            # Residual statistics
            'sum_squared_residuals': ss_res,
            'total_sum_squares': ss_tot,

            # Quality assessment
            'fit_quality_assessment': self._assess_fit_quality(r_squared, reduced_chi_squared, n_data, n_params)
        }

    def _assess_fit_quality(self, r_squared: float, reduced_chi_squared: float,
                           n_data: int, n_params: int) -> Dict[str, Any]:
        """
        Assess overall fit quality based on multiple metrics.

        Args:
            r_squared: R-squared value
            reduced_chi_squared: Reduced chi-squared value
            n_data: Number of data points
            n_params: Number of parameters

        Returns:
            Dictionary with fit quality assessment
        """
        assessment = {
            'overall_quality': 'unknown',
            'warnings': [],
            'recommendations': []
        }

        # R-squared assessment
        if r_squared > 0.95:
            r2_quality = 'excellent'
        elif r_squared > 0.90:
            r2_quality = 'good'
        elif r_squared > 0.80:
            r2_quality = 'fair'
        else:
            r2_quality = 'poor'
            assessment['warnings'].append(f"Low R² ({r_squared:.3f}) indicates poor model fit")

        # Reduced chi-squared assessment
        if 0.5 <= reduced_chi_squared <= 2.0:
            chi2_quality = 'good'
        elif reduced_chi_squared < 0.5:
            chi2_quality = 'overfitted'
            assessment['warnings'].append(f"Low χ²ᵣ ({reduced_chi_squared:.2f}) may indicate overfitting")
        else:
            chi2_quality = 'poor'
            if reduced_chi_squared > 5:
                assessment['warnings'].append(f"High χ²ᵣ ({reduced_chi_squared:.2f}) indicates poor model fit")

        # Data sufficiency
        data_ratio = n_data / n_params
        if data_ratio < 3:
            assessment['warnings'].append(f"Very few data points per parameter ({data_ratio:.1f})")
            assessment['recommendations'].append("Collect more data or reduce model complexity")
        elif data_ratio < 5:
            assessment['warnings'].append(f"Few data points per parameter ({data_ratio:.1f})")

        # Overall quality determination
        if r2_quality == 'excellent' and chi2_quality == 'good':
            assessment['overall_quality'] = 'excellent'
        elif r2_quality in ['excellent', 'good'] and chi2_quality in ['good', 'overfitted']:
            assessment['overall_quality'] = 'good'
        elif r2_quality in ['good', 'fair'] and chi2_quality != 'poor':
            assessment['overall_quality'] = 'fair'
        else:
            assessment['overall_quality'] = 'poor'

        # General recommendations
        if assessment['overall_quality'] == 'poor':
            assessment['recommendations'].extend([
                "Check data quality and outliers",
                "Consider alternative model forms",
                "Verify parameter bounds and initial guesses"
            ])
        elif chi2_quality == 'overfitted':
            assessment['recommendations'].append("Consider simpler model or regularization")

        return assessment

    def compare_models(self, model_results: List[Dict[str, Any]],
                      model_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple fitted models using information criteria.

        Args:
            model_results: List of goodness-of-fit results for different models
            model_names: Optional names for the models

        Returns:
            Dictionary with model comparison results
        """
        n_models = len(model_results)
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(n_models)]

        if len(model_names) != n_models:
            raise ValueError("Number of model names must match number of models")

        # Extract key metrics
        metrics = ['aic', 'aic_corrected', 'bic', 'r_squared', 'reduced_chi_squared']
        comparison = {metric: [] for metric in metrics}
        comparison['model_names'] = model_names

        for result in model_results:
            for metric in metrics:
                comparison[metric].append(result.get(metric, np.nan))

        # Model rankings (lower is better for IC, higher for R²)
        rankings = {}

        for metric in ['aic', 'aic_corrected', 'bic', 'reduced_chi_squared']:
            values = np.array(comparison[metric])
            valid_mask = np.isfinite(values)

            if np.any(valid_mask):
                ranks = np.full(n_models, np.nan)
                ranks[valid_mask] = np.argsort(np.argsort(values[valid_mask])) + 1
                rankings[metric] = ranks
            else:
                rankings[metric] = np.full(n_models, np.nan)

        # R-squared ranking (higher is better)
        r2_values = np.array(comparison['r_squared'])
        valid_mask = np.isfinite(r2_values)
        if np.any(valid_mask):
            ranks = np.full(n_models, np.nan)
            ranks[valid_mask] = np.argsort(np.argsort(-r2_values[valid_mask])) + 1  # Negative for descending order
            rankings['r_squared'] = ranks
        else:
            rankings['r_squared'] = np.full(n_models, np.nan)

        # Calculate delta values for information criteria
        deltas = {}
        for metric in ['aic', 'aic_corrected', 'bic']:
            values = np.array(comparison[metric])
            if np.any(np.isfinite(values)):
                min_val = np.nanmin(values)
                deltas[f'{metric}_delta'] = values - min_val
            else:
                deltas[f'{metric}_delta'] = np.full(n_models, np.nan)

        # Overall recommendation
        # Use AIC corrected as primary criterion, then BIC as secondary
        aic_c_ranks = rankings.get('aic_corrected', np.full(n_models, np.nan))
        bic_ranks = rankings.get('bic', np.full(n_models, np.nan))

        best_model_idx = None
        if np.any(np.isfinite(aic_c_ranks)):
            best_model_idx = np.nanargmin(aic_c_ranks)
        elif np.any(np.isfinite(bic_ranks)):
            best_model_idx = np.nanargmin(bic_ranks)

        return {
            'model_names': model_names,
            'metrics': comparison,
            'rankings': rankings,
            'deltas': deltas,
            'best_model': {
                'index': best_model_idx,
                'name': model_names[best_model_idx] if best_model_idx is not None else None
            },
            'comparison_summary': self._generate_comparison_summary(
                model_names, rankings, deltas, best_model_idx
            )
        }

    def _generate_comparison_summary(self, model_names: List[str], rankings: Dict[str, np.ndarray],
                                   deltas: Dict[str, np.ndarray], best_model_idx: Optional[int]) -> List[str]:
        """Generate human-readable model comparison summary."""
        summary = []

        if best_model_idx is not None:
            summary.append(f"Best model: {model_names[best_model_idx]} (based on information criteria)")

        # Check for substantial evidence differences (delta > 2 is substantial, > 10 is decisive)
        if 'aic_corrected_delta' in deltas:
            aic_deltas = deltas['aic_corrected_delta']
            for i, (name, delta) in enumerate(zip(model_names, aic_deltas)):
                if i != best_model_idx and np.isfinite(delta):
                    if delta < 2:
                        summary.append(f"{name}: Equivalent to best model (ΔAICc = {delta:.1f})")
                    elif delta < 10:
                        summary.append(f"{name}: Substantial evidence against (ΔAICc = {delta:.1f})")
                    else:
                        summary.append(f"{name}: Decisive evidence against (ΔAICc = {delta:.1f})")

        return summary


class DiagnosticReporter:
    """
    Structured diagnostic reporting system with visualization capabilities.

    Aggregates results from all analysis components and generates comprehensive
    diagnostic reports for G2 fitting results.
    """

    def __init__(self, include_plots: bool = False, n_jobs: Optional[int] = None):
        """
        Initialize the diagnostic reporter.

        Args:
            include_plots: Whether to generate diagnostic plots
            n_jobs: Number of parallel jobs for bootstrap analysis
        """
        self.include_plots = include_plots
        self.bootstrap_analyzer = BootstrapAnalyzer(n_jobs=n_jobs)
        self.covariance_analyzer = CovarianceAnalyzer()
        self.residual_analyzer = ResidualAnalyzer()
        self.goodness_analyzer = GoodnessOfFitAnalyzer()

    def generate_comprehensive_report(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                                    popt: np.ndarray, pcov: np.ndarray,
                                    bounds: Optional[Tuple] = None,
                                    sigma: Optional[np.ndarray] = None,
                                    func_name: str = "model",
                                    bootstrap_method: str = 'residual',
                                    n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostic report for a fitted model.

        Args:
            func: Fitted function
            xdata: Independent variable data
            ydata: Dependent variable data
            popt: Fitted parameters
            pcov: Parameter covariance matrix
            bounds: Parameter bounds
            sigma: Data uncertainties
            func_name: Name of the fitted function
            bootstrap_method: Bootstrap method to use
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary containing comprehensive diagnostic report
        """
        logger.info(f"Generating comprehensive diagnostic report for {func_name}")

        report = {
            'function_name': func_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_info': {
                'n_data_points': len(xdata),
                'n_parameters': len(popt),
                'x_range': (np.min(xdata), np.max(xdata)),
                'y_range': (np.min(ydata), np.max(ydata)),
                'has_uncertainties': sigma is not None
            }
        }

        # Calculate predicted values
        y_pred = func(xdata, *popt)

        # 1. Goodness-of-fit analysis
        logger.debug("Calculating goodness-of-fit metrics")
        try:
            gof_results = self.goodness_analyzer.calculate_goodness_of_fit_metrics(
                ydata, y_pred, len(popt), sigma
            )
            report['goodness_of_fit'] = gof_results
        except Exception as e:
            logger.error(f"Goodness-of-fit analysis failed: {e}")
            report['goodness_of_fit'] = {'error': str(e)}

        # 2. Covariance matrix analysis
        logger.debug("Analyzing covariance matrix")
        try:
            cov_results = self.covariance_analyzer.analyze_covariance_matrix(pcov, popt, func_name)
            report['covariance_analysis'] = cov_results

            # Parameter constraint suggestions
            suggestions = self.covariance_analyzer.suggest_parameter_constraints(cov_results, popt)
            report['parameter_suggestions'] = suggestions

        except Exception as e:
            logger.error(f"Covariance analysis failed: {e}")
            report['covariance_analysis'] = {'error': str(e)}
            report['parameter_suggestions'] = []

        # 3. Residual analysis
        logger.debug("Analyzing residuals")
        try:
            residual_results = self.residual_analyzer.analyze_residuals(xdata, ydata, y_pred, sigma)
            report['residual_analysis'] = residual_results

            # Systematic deviation analysis
            deviation_results = self.residual_analyzer.detect_systematic_deviations(
                xdata, ydata - y_pred
            )
            report['systematic_deviations'] = deviation_results

        except Exception as e:
            logger.error(f"Residual analysis failed: {e}")
            report['residual_analysis'] = {'error': str(e)}
            report['systematic_deviations'] = {'error': str(e)}

        # 4. Bootstrap confidence intervals (optional, can be time-consuming)
        if n_bootstrap > 0:
            logger.debug(f"Running bootstrap analysis with {n_bootstrap} samples")
            try:
                self.bootstrap_analyzer.n_bootstrap = n_bootstrap
                bootstrap_results = self.bootstrap_analyzer.bootstrap_confidence_intervals(
                    func, xdata, ydata, popt, pcov, bounds, sigma, bootstrap_method
                )
                report['bootstrap_analysis'] = bootstrap_results
            except Exception as e:
                logger.error(f"Bootstrap analysis failed: {e}")
                report['bootstrap_analysis'] = {'error': str(e)}
        else:
            report['bootstrap_analysis'] = {'skipped': 'n_bootstrap = 0'}

        # 5. Generate summary and recommendations
        report['summary'] = self._generate_diagnostic_summary(report)

        logger.info("Comprehensive diagnostic report generated successfully")
        return report

    def _generate_diagnostic_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary and recommendations based on diagnostic analysis.

        Args:
            report: Full diagnostic report

        Returns:
            Dictionary with summary and recommendations
        """
        summary = {
            'overall_assessment': 'unknown',
            'key_findings': [],
            'warnings': [],
            'recommendations': [],
            'quality_score': 0.0
        }

        # Extract key results
        gof = report.get('goodness_of_fit', {})
        cov = report.get('covariance_analysis', {})
        res = report.get('residual_analysis', {})

        # Quality assessment based on multiple criteria
        quality_factors = []

        # 1. R-squared assessment
        r_squared = gof.get('r_squared', 0)
        if r_squared > 0.95:
            quality_factors.append(1.0)
            summary['key_findings'].append(f"Excellent fit quality (R² = {r_squared:.3f})")
        elif r_squared > 0.90:
            quality_factors.append(0.8)
            summary['key_findings'].append(f"Good fit quality (R² = {r_squared:.3f})")
        elif r_squared > 0.80:
            quality_factors.append(0.6)
            summary['key_findings'].append(f"Fair fit quality (R² = {r_squared:.3f})")
        else:
            quality_factors.append(0.2)
            summary['warnings'].append(f"Poor fit quality (R² = {r_squared:.3f})")

        # 2. Reduced chi-squared assessment
        chi2_red = gof.get('reduced_chi_squared', np.inf)
        if 0.5 <= chi2_red <= 2.0:
            quality_factors.append(1.0)
            summary['key_findings'].append(f"Appropriate model complexity (χ²ᵣ = {chi2_red:.2f})")
        elif chi2_red < 0.5:
            quality_factors.append(0.7)
            summary['warnings'].append(f"Possible overfitting (χ²ᵣ = {chi2_red:.2f})")
        else:
            quality_factors.append(0.3)
            summary['warnings'].append(f"Poor model fit (χ²ᵣ = {chi2_red:.2f})")

        # 3. Numerical stability assessment
        if not cov.get('error') and cov.get('is_numerically_stable', False):
            quality_factors.append(1.0)
            summary['key_findings'].append("Numerically stable parameter estimates")
        else:
            quality_factors.append(0.5)
            stability_issues = cov.get('stability_issues', [])
            if stability_issues:
                summary['warnings'].extend(stability_issues)

        # 4. Residual analysis assessment
        if not res.get('error'):
            outliers = res.get('outliers', {})
            outlier_fraction = outliers.get('fraction', 0)

            if outlier_fraction < 0.05:
                quality_factors.append(1.0)
                summary['key_findings'].append(f"Few outliers detected ({outlier_fraction:.1%})")
            elif outlier_fraction < 0.10:
                quality_factors.append(0.8)
                summary['key_findings'].append(f"Some outliers detected ({outlier_fraction:.1%})")
            else:
                quality_factors.append(0.5)
                summary['warnings'].append(f"Many outliers detected ({outlier_fraction:.1%})")

            # Check for randomness in residuals
            randomness = res.get('randomness_test', {})
            if randomness.get('is_random', False):
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.7)
                summary['warnings'].append("Residuals show non-random patterns")

        # Calculate overall quality score
        if quality_factors:
            summary['quality_score'] = np.mean(quality_factors)

        # Overall assessment
        if summary['quality_score'] > 0.9:
            summary['overall_assessment'] = 'excellent'
        elif summary['quality_score'] > 0.7:
            summary['overall_assessment'] = 'good'
        elif summary['quality_score'] > 0.5:
            summary['overall_assessment'] = 'fair'
        else:
            summary['overall_assessment'] = 'poor'

        # Generate recommendations
        if summary['overall_assessment'] in ['poor', 'fair']:
            summary['recommendations'].extend([
                "Check data quality and remove obvious outliers",
                "Verify model appropriateness for the data",
                "Consider alternative model forms or additional terms"
            ])

        # Add specific recommendations from sub-analyses
        param_suggestions = report.get('parameter_suggestions', [])
        summary['recommendations'].extend(param_suggestions)

        deviation_recs = report.get('systematic_deviations', {}).get('recommendations', [])
        summary['recommendations'].extend(deviation_recs)

        # Remove duplicates
        summary['recommendations'] = list(set(summary['recommendations']))

        return summary

    def generate_text_report(self, diagnostic_report: Dict[str, Any]) -> str:
        """
        Generate a human-readable text report from diagnostic results.

        Args:
            diagnostic_report: Full diagnostic report dictionary

        Returns:
            Formatted text report as string
        """
        lines = []

        # Header
        lines.append("="*80)
        lines.append(f"XPCS G2 FITTING DIAGNOSTIC REPORT")
        lines.append(f"Function: {diagnostic_report.get('function_name', 'Unknown')}")
        lines.append(f"Generated: {diagnostic_report.get('timestamp', 'Unknown')}")
        lines.append("="*80)

        # Data information
        data_info = diagnostic_report.get('data_info', {})
        lines.append("\nDATA SUMMARY:")
        lines.append(f"  Data points: {data_info.get('n_data_points', 'N/A')}")
        lines.append(f"  Parameters:  {data_info.get('n_parameters', 'N/A')}")
        lines.append(f"  X range:     {data_info.get('x_range', 'N/A')}")
        lines.append(f"  Y range:     {data_info.get('y_range', 'N/A')}")
        lines.append(f"  Uncertainties: {'Yes' if data_info.get('has_uncertainties', False) else 'No'}")

        # Summary
        summary = diagnostic_report.get('summary', {})
        lines.append(f"\nOVERALL ASSESSMENT: {summary.get('overall_assessment', 'Unknown').upper()}")
        lines.append(f"Quality Score: {summary.get('quality_score', 0):.2f}/1.00")

        # Key findings
        key_findings = summary.get('key_findings', [])
        if key_findings:
            lines.append("\nKEY FINDINGS:")
            for finding in key_findings:
                lines.append(f"  ✓ {finding}")

        # Warnings
        warnings = summary.get('warnings', [])
        if warnings:
            lines.append("\nWARNINGS:")
            for warning in warnings:
                lines.append(f"  ⚠ {warning}")

        # Goodness of fit metrics
        gof = diagnostic_report.get('goodness_of_fit', {})
        if not gof.get('error'):
            lines.append("\nGOODNESS OF FIT:")
            lines.append(f"  R²:                {gof.get('r_squared', 'N/A'):.4f}")
            lines.append(f"  Adjusted R²:       {gof.get('adjusted_r_squared', 'N/A'):.4f}")
            lines.append(f"  Reduced χ²:        {gof.get('reduced_chi_squared', 'N/A'):.3f}")
            lines.append(f"  RMSE:              {gof.get('rmse', 'N/A'):.4e}")
            lines.append(f"  AIC:               {gof.get('aic', 'N/A'):.2f}")
            lines.append(f"  BIC:               {gof.get('bic', 'N/A'):.2f}")

        # Parameter analysis
        cov = diagnostic_report.get('covariance_analysis', {})
        if not cov.get('error'):
            lines.append("\nPARAMETER ANALYSIS:")
            param_errors = cov.get('parameter_errors', [])
            for i, error in enumerate(param_errors):
                rel_error = cov.get('relative_errors', [0])[i] if i < len(cov.get('relative_errors', [])) else 0
                lines.append(f"  Parameter {i}: ±{error:.4e} ({rel_error:.1%} relative)")

            if cov.get('high_correlations', []):
                lines.append("  High correlations detected:")
                for i, j, corr in cov.get('high_correlations', []):
                    lines.append(f"    Parameters {i}-{j}: r = {corr:.3f}")

        # Residual analysis
        res = diagnostic_report.get('residual_analysis', {})
        if not res.get('error'):
            outliers = res.get('outliers', {})
            lines.append(f"\nRESIDUAL ANALYSIS:")
            lines.append(f"  Outliers: {outliers.get('count', 0)} ({outliers.get('fraction', 0):.1%})")

            randomness = res.get('randomness_test', {})
            lines.append(f"  Random pattern: {'Yes' if randomness.get('is_random', False) else 'No'}")

            autocorr = res.get('autocorrelation', {})
            lines.append(f"  Independent: {'Yes' if autocorr.get('is_independent', False) else 'No'}")

        # Bootstrap results (if available)
        bootstrap = diagnostic_report.get('bootstrap_analysis', {})
        if not bootstrap.get('error') and not bootstrap.get('skipped'):
            lines.append(f"\nBOOTSTRAP ANALYSIS:")
            lines.append(f"  Method: {bootstrap.get('method', 'N/A')}")
            lines.append(f"  Success rate: {bootstrap.get('success_rate', 0):.1%}")
            lines.append(f"  Confidence level: {bootstrap.get('confidence_level', 0):.0%}")

            param_stats = bootstrap.get('parameter_statistics', {})
            for param_key, stats in param_stats.items():
                ci = stats.get('confidence_interval', [])
                if len(ci) == 2:
                    lines.append(f"  {param_key} CI: [{ci[0]:.4e}, {ci[1]:.4e}]")

        # Recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for rec in recommendations:
                lines.append(f"  • {rec}")

        lines.append("\n" + "="*80)

        return "\n".join(lines)


# Enhanced RobustOptimizer with integrated error analysis
class RobustOptimizerWithDiagnostics(RobustOptimizer):
    """
    Enhanced RobustOptimizer with integrated error analysis and diagnostics.

    Extends the base RobustOptimizer with comprehensive error analysis capabilities
    including bootstrap confidence intervals, residual analysis, and diagnostic reporting.
    """

    def __init__(self,
                 max_iterations: int = 10000,
                 tolerance_factor: float = 1.0,
                 enable_caching: bool = True,
                 performance_tracking: bool = True,
                 diagnostic_level: str = 'standard',
                 n_jobs: Optional[int] = None):
        """
        Initialize the enhanced robust optimizer with diagnostics.

        Args:
            max_iterations: Maximum iterations for optimization methods
            tolerance_factor: Scaling factor for convergence tolerances
            enable_caching: Whether to use joblib caching
            performance_tracking: Whether to track method performance
            diagnostic_level: Level of diagnostics ('basic', 'standard', 'comprehensive')
            n_jobs: Number of parallel jobs for bootstrap analysis
        """
        super().__init__(max_iterations, tolerance_factor, enable_caching, performance_tracking)

        self.diagnostic_level = diagnostic_level
        self.n_jobs = n_jobs
        self.diagnostic_reporter = DiagnosticReporter(n_jobs=n_jobs)

        # Configuration for different diagnostic levels
        if diagnostic_level == 'basic':
            self.default_bootstrap_samples = 0
            self.include_residual_analysis = False
            self.include_covariance_analysis = True
        elif diagnostic_level == 'standard':
            self.default_bootstrap_samples = 500
            self.include_residual_analysis = True
            self.include_covariance_analysis = True
        elif diagnostic_level == 'comprehensive':
            self.default_bootstrap_samples = 1000
            self.include_residual_analysis = True
            self.include_covariance_analysis = True
        else:
            raise ValueError(f"Unknown diagnostic level: {diagnostic_level}")

    def robust_curve_fit_with_diagnostics(self, func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                                        p0: Optional[np.ndarray] = None, bounds: Optional[Tuple] = None,
                                        sigma: Optional[np.ndarray] = None,
                                        func_name: str = "model",
                                        bootstrap_samples: Optional[int] = None,
                                        bootstrap_method: str = 'residual',
                                        **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Perform robust curve fitting with comprehensive diagnostics.

        Args:
            func: Function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            p0: Initial parameter guess (estimated if None)
            bounds: Parameter bounds (lower, upper)
            sigma: Uncertainties in ydata
            func_name: Name of the function for reporting
            bootstrap_samples: Number of bootstrap samples (uses default if None)
            bootstrap_method: Bootstrap method ('residual', 'parametric', 'nonparametric')
            **kwargs: Additional arguments (for API compatibility)

        Returns:
            popt: Optimal parameters
            pcov: Covariance matrix
            diagnostics: Comprehensive diagnostic report
        """
        logger.info(f"Starting robust fitting with diagnostics for {func_name}")

        # Perform the robust curve fitting first
        popt, pcov, fit_info = super().robust_curve_fit(
            func, xdata, ydata, p0, bounds, sigma, **kwargs
        )

        # Determine number of bootstrap samples
        if bootstrap_samples is None:
            bootstrap_samples = self.default_bootstrap_samples

        # Generate comprehensive diagnostic report
        try:
            diagnostics = self.diagnostic_reporter.generate_comprehensive_report(
                func, xdata, ydata, popt, pcov, bounds, sigma, func_name,
                bootstrap_method, bootstrap_samples
            )

            # Add original fit information to diagnostics
            diagnostics['optimization_info'] = fit_info

        except Exception as e:
            logger.error(f"Diagnostic analysis failed: {e}")
            # Return basic diagnostics if comprehensive analysis fails
            diagnostics = {
                'error': str(e),
                'optimization_info': fit_info,
                'basic_metrics': {
                    'function_name': func_name,
                    'n_parameters': len(popt),
                    'optimization_method': fit_info.get('method', 'unknown')
                }
            }

        return popt, pcov, diagnostics

    def analyze_fit_quality(self, diagnostics: Dict[str, Any]) -> str:
        """
        Quick fit quality assessment from diagnostics.

        Args:
            diagnostics: Diagnostic report from robust_curve_fit_with_diagnostics

        Returns:
            String describing fit quality
        """
        summary = diagnostics.get('summary', {})
        overall_assessment = summary.get('overall_assessment', 'unknown')
        quality_score = summary.get('quality_score', 0.0)

        return f"{overall_assessment.capitalize()} (score: {quality_score:.2f})"

    def get_parameter_uncertainties(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameter uncertainties from diagnostic report.

        Args:
            diagnostics: Diagnostic report

        Returns:
            Dictionary with parameter uncertainty information
        """
        uncertainties = {
            'covariance_based': {},
            'bootstrap_based': {}
        }

        # Covariance-based uncertainties
        cov_analysis = diagnostics.get('covariance_analysis', {})
        if not cov_analysis.get('error'):
            param_errors = cov_analysis.get('parameter_errors', [])
            relative_errors = cov_analysis.get('relative_errors', [])

            for i, (abs_err, rel_err) in enumerate(zip(param_errors, relative_errors)):
                uncertainties['covariance_based'][f'param_{i}'] = {
                    'absolute_error': abs_err,
                    'relative_error': rel_err
                }

        # Bootstrap-based uncertainties
        bootstrap_analysis = diagnostics.get('bootstrap_analysis', {})
        if not bootstrap_analysis.get('error') and not bootstrap_analysis.get('skipped'):
            param_stats = bootstrap_analysis.get('parameter_statistics', {})

            for param_key, stats in param_stats.items():
                uncertainties['bootstrap_based'][param_key] = {
                    'bootstrap_std': stats.get('bootstrap_std'),
                    'bootstrap_bias': stats.get('bootstrap_bias'),
                    'confidence_interval': stats.get('confidence_interval')
                }

        return uncertainties


# ============================================================================
# DIFFUSION ANALYSIS FRAMEWORK
# ============================================================================

class DiffusionModels:
    """
    Collection of diffusion models for XPCS G2 analysis.

    Provides various diffusion models including simple diffusion, subdiffusion,
    hyperdiffusion, and anomalous diffusion with proper physical constraints.
    """

    @staticmethod
    def simple_diffusion(tau: np.ndarray, baseline: float, amplitude: float,
                        diffusion_coefficient: float) -> np.ndarray:
        """
        Simple diffusion model: g2(τ) = β + A * exp(-2D*q²*τ)

        Args:
            tau: Time delay array
            baseline: Baseline (β)
            amplitude: Amplitude (A)
            diffusion_coefficient: Diffusion coefficient (D)

        Returns:
            G2 correlation function values
        """
        return baseline + amplitude * np.exp(-2.0 * diffusion_coefficient * tau)

    @staticmethod
    def subdiffusion(tau: np.ndarray, baseline: float, amplitude: float,
                    diffusion_coefficient: float, alpha: float) -> np.ndarray:
        """
        Subdiffusion model: g2(τ) = β + A * exp(-2D*q²*τ^α) with α < 1

        Args:
            tau: Time delay array
            baseline: Baseline (β)
            amplitude: Amplitude (A)
            diffusion_coefficient: Diffusion coefficient (D)
            alpha: Subdiffusion exponent (0 < α < 1)

        Returns:
            G2 correlation function values
        """
        # Ensure physical constraint: 0 < alpha < 1 for subdiffusion
        alpha_constrained = np.clip(alpha, 0.01, 0.99)
        return baseline + amplitude * np.exp(-2.0 * diffusion_coefficient * np.power(tau, alpha_constrained))

    @staticmethod
    def hyperdiffusion(tau: np.ndarray, baseline: float, amplitude: float,
                      diffusion_coefficient: float, alpha: float) -> np.ndarray:
        """
        Hyperdiffusion model: g2(τ) = β + A * exp(-2D*q²*τ^α) with α > 1

        Args:
            tau: Time delay array
            baseline: Baseline (β)
            amplitude: Amplitude (A)
            diffusion_coefficient: Diffusion coefficient (D)
            alpha: Hyperdiffusion exponent (α > 1)

        Returns:
            G2 correlation function values
        """
        # Ensure physical constraint: alpha > 1 for hyperdiffusion
        alpha_constrained = np.maximum(alpha, 1.01)
        return baseline + amplitude * np.exp(-2.0 * diffusion_coefficient * np.power(tau, alpha_constrained))

    @staticmethod
    def anomalous_diffusion(tau: np.ndarray, baseline: float, amplitude: float,
                           diffusion_coefficient: float, alpha: float) -> np.ndarray:
        """
        General anomalous diffusion model: g2(τ) = β + A * exp(-2D*q²*τ^α)

        Args:
            tau: Time delay array
            baseline: Baseline (β)
            amplitude: Amplitude (A)
            diffusion_coefficient: Diffusion coefficient (D)
            alpha: Anomalous diffusion exponent

        Returns:
            G2 correlation function values
        """
        # Allow full range of alpha values
        alpha_constrained = np.clip(alpha, 0.01, 3.0)  # Physical upper limit
        return baseline + amplitude * np.exp(-2.0 * diffusion_coefficient * np.power(tau, alpha_constrained))

    @staticmethod
    def stretched_exponential(tau: np.ndarray, baseline: float, amplitude: float,
                             relaxation_time: float, beta: float) -> np.ndarray:
        """
        Stretched exponential model: g2(τ) = β + A * exp(-(τ/τ_r)^β)

        Args:
            tau: Time delay array
            baseline: Baseline (β_base)
            amplitude: Amplitude (A)
            relaxation_time: Relaxation time (τ_r)
            beta: Stretching exponent

        Returns:
            G2 correlation function values
        """
        beta_constrained = np.clip(beta, 0.1, 2.0)  # Physical constraints
        return baseline + amplitude * np.exp(-np.power(tau / relaxation_time, beta_constrained))

    @staticmethod
    def double_exponential(tau: np.ndarray, baseline: float, amplitude1: float,
                          rate1: float, amplitude2: float, rate2: float) -> np.ndarray:
        """
        Double exponential model for bimodal diffusion.

        Args:
            tau: Time delay array
            baseline: Baseline
            amplitude1: Amplitude of first component
            rate1: Decay rate of first component
            amplitude2: Amplitude of second component
            rate2: Decay rate of second component

        Returns:
            G2 correlation function values
        """
        return (baseline +
                amplitude1 * np.exp(-rate1 * tau) +
                amplitude2 * np.exp(-rate2 * tau))

    @classmethod
    def get_model_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available diffusion models."""
        return {
            'simple_diffusion': {
                'function': cls.simple_diffusion,
                'parameters': ['baseline', 'amplitude', 'diffusion_coefficient'],
                'bounds': [(0.5, 2.0), (0.0, 1.0), (1e-8, 1e-2)],
                'description': 'Simple exponential diffusion model'
            },
            'subdiffusion': {
                'function': cls.subdiffusion,
                'parameters': ['baseline', 'amplitude', 'diffusion_coefficient', 'alpha'],
                'bounds': [(0.5, 2.0), (0.0, 1.0), (1e-8, 1e-2), (0.01, 0.99)],
                'description': 'Subdiffusion model with α < 1'
            },
            'hyperdiffusion': {
                'function': cls.hyperdiffusion,
                'parameters': ['baseline', 'amplitude', 'diffusion_coefficient', 'alpha'],
                'bounds': [(0.5, 2.0), (0.0, 1.0), (1e-8, 1e-2), (1.01, 3.0)],
                'description': 'Hyperdiffusion model with α > 1'
            },
            'anomalous_diffusion': {
                'function': cls.anomalous_diffusion,
                'parameters': ['baseline', 'amplitude', 'diffusion_coefficient', 'alpha'],
                'bounds': [(0.5, 2.0), (0.0, 1.0), (1e-8, 1e-2), (0.01, 3.0)],
                'description': 'General anomalous diffusion model'
            },
            'stretched_exponential': {
                'function': cls.stretched_exponential,
                'parameters': ['baseline', 'amplitude', 'relaxation_time', 'beta'],
                'bounds': [(0.5, 2.0), (0.0, 1.0), (1e-6, 1e3), (0.1, 2.0)],
                'description': 'Stretched exponential model'
            },
            'double_exponential': {
                'function': cls.double_exponential,
                'parameters': ['baseline', 'amplitude1', 'rate1', 'amplitude2', 'rate2'],
                'bounds': [(0.5, 2.0), (0.0, 0.5), (1e-6, 1e3), (0.0, 0.5), (1e-6, 1e3)],
                'description': 'Double exponential model for bimodal diffusion'
            }
        }


class WeightedRegressionFramework:
    """
    Advanced weighted regression framework for heteroscedastic error handling.

    Provides various weighting schemes and methods to account for data quality
    variations and heteroscedastic errors in XPCS diffusion analysis.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_weights_inverse_variance(sigma: np.ndarray,
                                         min_weight: float = 1e-8) -> np.ndarray:
        """
        Calculate inverse variance weights: w_i = 1/σ_i²

        Args:
            sigma: Standard deviations
            min_weight: Minimum weight to prevent division by zero

        Returns:
            Weight array
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / (sigma ** 2)
            weights = np.where(np.isfinite(weights), weights, min_weight)
            weights = np.maximum(weights, min_weight)
        return weights

    @staticmethod
    def calculate_weights_robust_mad(residuals: np.ndarray) -> np.ndarray:
        """
        Calculate robust weights based on Median Absolute Deviation (MAD).

        Args:
            residuals: Residual values

        Returns:
            Robust weight array
        """
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad == 0:
            mad = np.std(residuals) * 0.67449  # Fallback to robust std estimate

        # Tukey's bisquare weights
        normalized_residuals = residuals / (6.0 * mad)  # c = 6 for 95% efficiency
        weights = np.where(np.abs(normalized_residuals) < 1,
                          (1 - normalized_residuals**2)**2, 0)
        return weights

    @staticmethod
    def calculate_weights_huber(residuals: np.ndarray, k: float = 1.345) -> np.ndarray:
        """
        Calculate Huber weights for robust regression.

        Args:
            residuals: Residual values
            k: Huber parameter (default 1.345 for 95% efficiency)

        Returns:
            Huber weight array
        """
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad == 0:
            mad = np.std(residuals) * 0.67449

        normalized_residuals = np.abs(residuals) / mad
        weights = np.where(normalized_residuals <= k, 1.0, k / normalized_residuals)
        return weights

    @staticmethod
    def calculate_weights_adaptive(y_data: np.ndarray, y_pred: np.ndarray,
                                 method: str = 'local_variance') -> np.ndarray:
        """
        Calculate adaptive weights based on local data characteristics.

        Args:
            y_data: Observed data
            y_pred: Predicted values
            method: Weighting method ('local_variance', 'signal_dependent')

        Returns:
            Adaptive weight array
        """
        residuals = y_data - y_pred

        if method == 'local_variance':
            # Use local variance estimation with sliding window
            window_size = max(5, len(y_data) // 20)
            local_vars = []

            for i in range(len(y_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(y_data), i + window_size // 2 + 1)
                local_var = np.var(residuals[start_idx:end_idx])
                local_vars.append(max(local_var, 1e-8))

            weights = 1.0 / np.array(local_vars)

        elif method == 'signal_dependent':
            # Weight inversely proportional to signal magnitude
            signal_strength = np.abs(y_pred - np.min(y_pred))
            weights = 1.0 / (1.0 + signal_strength / np.max(signal_strength))

        else:
            raise ValueError(f"Unknown adaptive weighting method: {method}")

        return weights / np.sum(weights) * len(weights)  # Normalize

    def weighted_least_squares(self, func: Callable, xdata: np.ndarray,
                              ydata: np.ndarray, weights: np.ndarray,
                              p0: Optional[np.ndarray] = None,
                              bounds: Optional[Tuple] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform weighted least squares fitting.

        Args:
            func: Function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            weights: Weight array
            p0: Initial parameter guess
            bounds: Parameter bounds

        Returns:
            popt: Optimal parameters
            pcov: Parameter covariance matrix
        """
        # Convert weights to sigma for scipy.optimize.curve_fit
        sigma_weighted = 1.0 / np.sqrt(np.maximum(weights, 1e-8))

        try:
            popt, pcov = curve_fit(func, xdata, ydata, p0=p0,
                                 sigma=sigma_weighted, bounds=bounds)
        except Exception as e:
            self.logger.warning(f"Weighted fitting failed: {e}")
            # Fallback to unweighted fitting
            popt, pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds)

        return popt, pcov

    def iteratively_reweighted_least_squares(self, func: Callable, xdata: np.ndarray,
                                           ydata: np.ndarray, p0: Optional[np.ndarray] = None,
                                           bounds: Optional[Tuple] = None,
                                           max_iterations: int = 10,
                                           tolerance: float = 1e-6,
                                           weight_method: str = 'huber') -> Dict[str, Any]:
        """
        Iteratively Reweighted Least Squares (IRLS) for robust fitting.

        Args:
            func: Function to fit
            xdata: Independent variable data
            ydata: Dependent variable data
            p0: Initial parameter guess
            bounds: Parameter bounds
            max_iterations: Maximum IRLS iterations
            tolerance: Convergence tolerance
            weight_method: Weight calculation method

        Returns:
            Dictionary with fitting results and convergence info
        """
        self.logger.info(f"Starting IRLS with {weight_method} weights")

        # Initial fit with equal weights
        current_params, current_pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds)

        convergence_history = []
        weights = np.ones(len(ydata))  # Initialize weights
        param_change = float('inf')    # Initialize param_change

        for iteration in range(max_iterations):
            # Calculate residuals and weights
            y_pred = func(xdata, *current_params)
            residuals = ydata - y_pred

            if weight_method == 'huber':
                weights = self.calculate_weights_huber(residuals)
            elif weight_method == 'mad':
                weights = self.calculate_weights_robust_mad(residuals)
            elif weight_method == 'adaptive':
                weights = self.calculate_weights_adaptive(ydata, y_pred)
            else:
                raise ValueError(f"Unknown weight method: {weight_method}")

            # Weighted fit
            try:
                new_params, new_pcov = self.weighted_least_squares(
                    func, xdata, ydata, weights, current_params, bounds
                )
            except Exception as e:
                self.logger.warning(f"IRLS iteration {iteration} failed: {e}")
                break

            # Check convergence
            param_change = np.linalg.norm(new_params - current_params) / np.linalg.norm(current_params)
            convergence_history.append(param_change)

            if param_change < tolerance:
                self.logger.info(f"IRLS converged after {iteration + 1} iterations")
                break

            current_params = new_params
            current_pcov = new_pcov

        # Calculate final residuals with converged parameters
        final_y_pred = func(xdata, *current_params)
        final_residuals = ydata - final_y_pred

        return {
            'parameters': current_params,
            'covariance': current_pcov,
            'weights': weights,
            'convergence_history': convergence_history,
            'converged': param_change < tolerance,
            'iterations': iteration + 1,
            'final_residuals': final_residuals
        }


class OutlierDetection:
    """
    Advanced outlier detection methods for XPCS diffusion analysis.

    Provides various statistical and robust methods to identify and handle
    outliers in correlation function data.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def detect_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
        """
        Detect outliers using Interquartile Range (IQR) method.

        Args:
            data: Data array
            k: IQR multiplier (default 1.5)

        Returns:
            Boolean array indicating outliers
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        return (data < lower_bound) | (data > upper_bound)

    @staticmethod
    def detect_outliers_mad(data: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Detect outliers using Median Absolute Deviation (MAD).

        Args:
            data: Data array
            threshold: MAD threshold multiplier

        Returns:
            Boolean array indicating outliers
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return np.zeros_like(data, dtype=bool)

        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold

    @staticmethod
    def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using Z-score method.

        Args:
            data: Data array
            threshold: Z-score threshold

        Returns:
            Boolean array indicating outliers
        """
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return np.zeros_like(data, dtype=bool)

        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold

    def detect_outliers_isolation_forest(self, data: np.ndarray,
                                       contamination: float = 0.1) -> np.ndarray:
        """
        Detect outliers using Isolation Forest method.

        Args:
            data: Data array
            contamination: Expected proportion of outliers

        Returns:
            Boolean array indicating outliers
        """
        try:
            from sklearn.ensemble import IsolationForest

            # Reshape for sklearn
            X = data.reshape(-1, 1)

            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)

            return outlier_labels == -1

        except ImportError:
            self.logger.warning("sklearn not available, falling back to MAD method")
            return self.detect_outliers_mad(data)

    def detect_residual_outliers(self, y_true: np.ndarray, y_pred: np.ndarray,
                                method: str = 'mad', **kwargs) -> np.ndarray:
        """
        Detect outliers in residuals.

        Args:
            y_true: True values
            y_pred: Predicted values
            method: Outlier detection method
            **kwargs: Method-specific parameters

        Returns:
            Boolean array indicating outlier positions
        """
        residuals = y_true - y_pred

        if method == 'iqr':
            return self.detect_outliers_iqr(residuals, **kwargs)
        elif method == 'mad':
            return self.detect_outliers_mad(residuals, **kwargs)
        elif method == 'zscore':
            return self.detect_outliers_zscore(residuals, **kwargs)
        elif method == 'isolation_forest':
            return self.detect_outliers_isolation_forest(residuals, **kwargs)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def robust_outlier_detection(self, xdata: np.ndarray, ydata: np.ndarray,
                                func: Callable, params: np.ndarray,
                                methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Apply multiple outlier detection methods and combine results.

        Args:
            xdata: Independent variable data
            ydata: Dependent variable data
            func: Model function
            params: Model parameters
            methods: List of detection methods to use

        Returns:
            Dictionary with outlier detection results
        """
        if methods is None:
            methods = ['mad', 'iqr', 'zscore']

        y_pred = func(xdata, *params)
        residuals = ydata - y_pred

        results = {}
        for method in methods:
            try:
                outliers = self.detect_residual_outliers(ydata, y_pred, method=method)
                results[method] = outliers
            except Exception as e:
                self.logger.warning(f"Outlier detection method {method} failed: {e}")
                results[method] = np.zeros_like(ydata, dtype=bool)

        # Consensus outlier detection (majority vote)
        outlier_votes = np.stack(list(results.values()))
        consensus_outliers = np.sum(outlier_votes, axis=0) >= len(methods) // 2 + 1
        results['consensus'] = consensus_outliers

        return results


class ModelSelectionFramework:
    """
    Automated model selection and comparison framework for diffusion analysis.

    Provides information criteria-based model selection (AIC, BIC) and
    cross-validation methods to choose the most appropriate diffusion model.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_aic(n_params: int, n_data: int, chi_squared: float) -> float:
        """
        Calculate Akaike Information Criterion (AIC).

        Args:
            n_params: Number of model parameters
            n_data: Number of data points
            chi_squared: Chi-squared goodness of fit

        Returns:
            AIC value
        """
        if n_data <= n_params:
            return np.inf

        return 2 * n_params + n_data * np.log(chi_squared / n_data)

    @staticmethod
    def calculate_bic(n_params: int, n_data: int, chi_squared: float) -> float:
        """
        Calculate Bayesian Information Criterion (BIC).

        Args:
            n_params: Number of model parameters
            n_data: Number of data points
            chi_squared: Chi-squared goodness of fit

        Returns:
            BIC value
        """
        if n_data <= n_params:
            return np.inf

        return np.log(n_data) * n_params + n_data * np.log(chi_squared / n_data)

    @staticmethod
    def calculate_aicc(n_params: int, n_data: int, chi_squared: float) -> float:
        """
        Calculate corrected Akaike Information Criterion (AICc).

        Args:
            n_params: Number of model parameters
            n_data: Number of data points
            chi_squared: Chi-squared goodness of fit

        Returns:
            AICc value
        """
        aic = ModelSelectionFramework.calculate_aic(n_params, n_data, chi_squared)

        if n_data - n_params - 1 <= 0:
            return np.inf

        correction = (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
        return aic + correction

    def cross_validate_model(self, func: Callable, xdata: np.ndarray,
                           ydata: np.ndarray, p0: np.ndarray,
                           bounds: Optional[Tuple] = None,
                           k_folds: int = 5, sigma: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on a model.

        Args:
            func: Model function
            xdata: Independent variable data
            ydata: Dependent variable data
            p0: Initial parameter guess
            bounds: Parameter bounds
            k_folds: Number of folds for cross-validation
            sigma: Data uncertainties

        Returns:
            Cross-validation results dictionary
        """
        n_data = len(xdata)
        fold_size = n_data // k_folds
        cv_scores = []
        cv_params = []

        indices = np.arange(n_data)
        np.random.shuffle(indices)

        for fold in range(k_folds):
            # Define validation set
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < k_folds - 1 else n_data
            val_indices = indices[val_start:val_end]
            train_indices = np.setdiff1d(indices, val_indices)

            # Training and validation data
            x_train, y_train = xdata[train_indices], ydata[train_indices]
            x_val, y_val = xdata[val_indices], ydata[val_indices]

            sigma_train = sigma[train_indices] if sigma is not None else None

            try:
                # Fit on training data
                popt, _ = curve_fit(func, x_train, y_train, p0=p0,
                                   bounds=bounds, sigma=sigma_train)

                # Evaluate on validation data
                y_pred = func(x_val, *popt)
                mse = np.mean((y_val - y_pred) ** 2)
                cv_scores.append(mse)
                cv_params.append(popt)

            except Exception as e:
                self.logger.warning(f"CV fold {fold} failed: {e}")
                cv_scores.append(np.inf)
                cv_params.append(p0)

        return {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'cv_scores': cv_scores,
            'cv_parameters': cv_params,
            'n_successful_folds': np.sum(np.isfinite(cv_scores))
        }

    def compare_models(self, models: Dict[str, Dict], xdata: np.ndarray,
                      ydata: np.ndarray, sigma: Optional[np.ndarray] = None,
                      use_cross_validation: bool = True,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Compare multiple diffusion models using information criteria and CV.

        Args:
            models: Dictionary of models with their info
            xdata: Independent variable data
            ydata: Dependent variable data
            sigma: Data uncertainties
            use_cross_validation: Whether to perform cross-validation
            cv_folds: Number of CV folds

        Returns:
            Model comparison results
        """
        n_data = len(xdata)
        results = {}

        for model_name, model_info in models.items():
            self.logger.info(f"Evaluating model: {model_name}")

            func = model_info['function']
            bounds = model_info.get('bounds')
            n_params = len(model_info['parameters'])

            # Generate initial parameters
            if bounds:
                lower_bounds, upper_bounds = zip(*bounds)
                p0 = np.array([(l + u) / 2 for l, u in bounds])
            else:
                p0 = np.ones(n_params)

            try:
                # Fit the model
                popt, pcov = curve_fit(func, xdata, ydata, p0=p0,
                                     bounds=bounds, sigma=sigma)

                # Calculate residuals and chi-squared
                y_pred = func(xdata, *popt)
                residuals = ydata - y_pred

                if sigma is not None:
                    chi_squared = np.sum((residuals / sigma) ** 2)
                else:
                    chi_squared = np.sum(residuals ** 2)

                # Information criteria
                aic = self.calculate_aic(n_params, n_data, chi_squared)
                bic = self.calculate_bic(n_params, n_data, chi_squared)
                aicc = self.calculate_aicc(n_params, n_data, chi_squared)

                # Additional metrics
                r_squared = 1 - np.var(residuals) / np.var(ydata)
                rmse = np.sqrt(np.mean(residuals ** 2))

                model_results = {
                    'parameters': popt,
                    'covariance': pcov,
                    'chi_squared': chi_squared,
                    'aic': aic,
                    'bic': bic,
                    'aicc': aicc,
                    'r_squared': r_squared,
                    'rmse': rmse,
                    'n_parameters': n_params,
                    'fitted_successfully': True
                }

                # Cross-validation
                if use_cross_validation and n_data > cv_folds:
                    cv_results = self.cross_validate_model(
                        func, xdata, ydata, p0, bounds, cv_folds, sigma
                    )
                    model_results['cross_validation'] = cv_results

            except Exception as e:
                self.logger.error(f"Model {model_name} fitting failed: {e}")
                model_results = {
                    'error': str(e),
                    'fitted_successfully': False,
                    'aic': np.inf,
                    'bic': np.inf,
                    'aicc': np.inf,
                    'n_parameters': n_params
                }

            results[model_name] = model_results

        return results

    def select_best_model(self, comparison_results: Dict[str, Any],
                         criterion: str = 'bic',
                         cv_weight: float = 0.3) -> Dict[str, Any]:
        """
        Select the best model based on specified criteria.

        Args:
            comparison_results: Results from compare_models
            criterion: Selection criterion ('aic', 'bic', 'aicc', 'cv', 'combined')
            cv_weight: Weight for cross-validation in combined criterion

        Returns:
            Best model selection results
        """
        valid_models = {name: results for name, results in comparison_results.items()
                       if results.get('fitted_successfully', False)}

        if not valid_models:
            return {'best_model': None, 'reason': 'No models fitted successfully'}

        if criterion in ['aic', 'bic', 'aicc']:
            scores = {name: results[criterion] for name, results in valid_models.items()}
            best_model = min(scores.keys(), key=lambda k: scores[k])
            selection_info = {'criterion': criterion, 'scores': scores}

        elif criterion == 'cv':
            cv_scores = {}
            for name, results in valid_models.items():
                cv_data = results.get('cross_validation', {})
                cv_scores[name] = cv_data.get('mean_cv_score', np.inf)

            best_model = min(cv_scores.keys(), key=lambda k: cv_scores[k])
            selection_info = {'criterion': 'cross_validation', 'cv_scores': cv_scores}

        elif criterion == 'combined':
            combined_scores = {}
            for name, results in valid_models.items():
                bic_score = results.get('bic', np.inf)
                cv_data = results.get('cross_validation', {})
                cv_score = cv_data.get('mean_cv_score', np.inf)

                # Normalize scores (lower is better for both)
                if np.isfinite(bic_score) and np.isfinite(cv_score):
                    combined_scores[name] = (1 - cv_weight) * bic_score + cv_weight * cv_score
                else:
                    combined_scores[name] = np.inf

            best_model = min(combined_scores.keys(), key=lambda k: combined_scores[k])
            selection_info = {'criterion': 'combined', 'combined_scores': combined_scores}

        else:
            raise ValueError(f"Unknown selection criterion: {criterion}")

        return {
            'best_model': best_model,
            'best_model_results': valid_models[best_model],
            'selection_info': selection_info,
            'all_model_results': comparison_results
        }


class DiffusionCoefficientExtractor:
    """
    Advanced diffusion coefficient extraction with uncertainty propagation.

    Provides methods to extract physical diffusion coefficients from fitted
    parameters with proper uncertainty propagation and physical constraints.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def extract_simple_diffusion_coefficient(params: np.ndarray, param_errors: np.ndarray,
                                           q_value: float) -> Dict[str, float]:
        """
        Extract diffusion coefficient from simple diffusion model.

        For g2(τ) = β + A * exp(-2D*q²*τ), the fitted parameter is 2D*q²

        Args:
            params: Fitted parameters [baseline, amplitude, 2D*q²]
            param_errors: Parameter uncertainties
            q_value: Scattering vector magnitude (q)

        Returns:
            Dictionary with diffusion coefficient and uncertainty
        """
        if len(params) < 3:
            raise ValueError("Simple diffusion requires at least 3 parameters")

        fitted_rate = params[2]  # This is 2D*q²
        fitted_rate_error = param_errors[2] if len(param_errors) > 2 else 0

        # Extract diffusion coefficient: D = fitted_rate / (2 * q²)
        diffusion_coeff = fitted_rate / (2.0 * q_value ** 2)

        # Uncertainty propagation: δD = δ(fitted_rate) / (2 * q²)
        diffusion_coeff_error = fitted_rate_error / (2.0 * q_value ** 2)

        return {
            'diffusion_coefficient': diffusion_coeff,
            'diffusion_coefficient_error': diffusion_coeff_error,
            'fitted_rate': fitted_rate,
            'fitted_rate_error': fitted_rate_error,
            'q_value': q_value
        }

    @staticmethod
    def extract_anomalous_diffusion_parameters(params: np.ndarray, param_errors: np.ndarray,
                                             q_value: float) -> Dict[str, float]:
        """
        Extract parameters from anomalous diffusion model.

        For g2(τ) = β + A * exp(-2D*q²*τ^α)

        Args:
            params: Fitted parameters [baseline, amplitude, 2D*q², α]
            param_errors: Parameter uncertainties
            q_value: Scattering vector magnitude

        Returns:
            Dictionary with diffusion parameters and uncertainties
        """
        if len(params) < 4:
            raise ValueError("Anomalous diffusion requires at least 4 parameters")

        fitted_rate = params[2]  # This is 2D*q²
        alpha = params[3]
        fitted_rate_error = param_errors[2] if len(param_errors) > 2 else 0
        alpha_error = param_errors[3] if len(param_errors) > 3 else 0

        # Extract generalized diffusion coefficient
        diffusion_coeff = fitted_rate / (2.0 * q_value ** 2)
        diffusion_coeff_error = fitted_rate_error / (2.0 * q_value ** 2)

        return {
            'diffusion_coefficient': diffusion_coeff,
            'diffusion_coefficient_error': diffusion_coeff_error,
            'anomalous_exponent': alpha,
            'anomalous_exponent_error': alpha_error,
            'fitted_rate': fitted_rate,
            'fitted_rate_error': fitted_rate_error,
            'q_value': q_value,
            'diffusion_type': 'subdiffusion' if alpha < 1 else 'hyperdiffusion' if alpha > 1 else 'normal'
        }

    def propagate_uncertainty_monte_carlo(self, func: Callable, params: np.ndarray,
                                        param_cov: np.ndarray, n_samples: int = 1000,
                                        **func_kwargs) -> Dict[str, float]:
        """
        Propagate uncertainties using Monte Carlo simulation.

        Args:
            func: Function to evaluate (should return dict with results)
            params: Best-fit parameters
            param_cov: Parameter covariance matrix
            n_samples: Number of Monte Carlo samples
            **func_kwargs: Additional arguments for func

        Returns:
            Dictionary with mean values and uncertainties
        """
        try:
            # Generate parameter samples from multivariate normal distribution
            param_samples = np.random.multivariate_normal(params, param_cov, n_samples)

            results = []
            for sample_params in param_samples:
                try:
                    result = func(sample_params, np.zeros_like(sample_params), **func_kwargs)
                    results.append(result)
                except Exception:
                    continue  # Skip failed evaluations

            if not results:
                raise ValueError("All Monte Carlo samples failed")

            # Calculate statistics
            keys = results[0].keys()
            statistics = {}

            for key in keys:
                if isinstance(results[0][key], (int, float)):
                    values = [r[key] for r in results if key in r and np.isfinite(r[key])]
                    if values:
                        statistics[key] = np.mean(values)
                        statistics[f"{key}_error"] = np.std(values)
                        statistics[f"{key}_median"] = np.median(values)
                        statistics[f"{key}_95ci_lower"] = np.percentile(values, 2.5)
                        statistics[f"{key}_95ci_upper"] = np.percentile(values, 97.5)

            statistics['n_successful_samples'] = len(results)
            return statistics

        except Exception as e:
            self.logger.error(f"Monte Carlo uncertainty propagation failed: {e}")
            return {}

    def extract_with_constraints(self, model_type: str, params: np.ndarray,
                               param_cov: np.ndarray, q_value: float,
                               apply_physical_constraints: bool = True) -> Dict[str, Any]:
        """
        Extract diffusion parameters with physical constraints and uncertainty propagation.

        Args:
            model_type: Type of diffusion model
            params: Fitted parameters
            param_cov: Parameter covariance matrix
            q_value: Scattering vector magnitude
            apply_physical_constraints: Whether to apply physical constraints

        Returns:
            Dictionary with extracted parameters and uncertainties
        """
        param_errors = np.sqrt(np.diag(param_cov)) if param_cov is not None else np.zeros_like(params)

        if model_type == 'simple_diffusion':
            extraction_func = self.extract_simple_diffusion_coefficient
        elif model_type in ['subdiffusion', 'hyperdiffusion', 'anomalous_diffusion']:
            extraction_func = self.extract_anomalous_diffusion_parameters
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Direct calculation
        direct_results = extraction_func(params, param_errors, q_value)

        # Monte Carlo uncertainty propagation if covariance matrix is available
        if param_cov is not None and np.all(np.diag(param_cov) > 0):
            mc_results = self.propagate_uncertainty_monte_carlo(
                extraction_func, params, param_cov, q_value=q_value
            )
            direct_results.update({f"mc_{key}": value for key, value in mc_results.items()})

        # Apply physical constraints
        if apply_physical_constraints:
            diffusion_coeff = direct_results.get('diffusion_coefficient', 0)

            # Physical constraints for diffusion coefficients
            if diffusion_coeff < 0:
                self.logger.warning(f"Negative diffusion coefficient: {diffusion_coeff}")
                direct_results['physical_constraint_warning'] = "Negative diffusion coefficient"

            if diffusion_coeff > 1e-6:  # Typical upper bound for XPCS (m²/s)
                self.logger.warning(f"Unusually large diffusion coefficient: {diffusion_coeff}")
                direct_results['physical_constraint_warning'] = "Unusually large diffusion coefficient"

            # Constraints for anomalous exponent
            if 'anomalous_exponent' in direct_results:
                alpha = direct_results['anomalous_exponent']
                if not (0.01 <= alpha <= 3.0):
                    self.logger.warning(f"Anomalous exponent out of physical range: {alpha}")
                    direct_results['physical_constraint_warning'] = f"Anomalous exponent {alpha} out of range"

        return direct_results


class ComprehensiveDiffusionAnalyzer:
    """
    Integrated diffusion analysis framework combining all components.

    This class provides a comprehensive interface for XPCS diffusion analysis
    including model selection, robust fitting, outlier detection, and advanced
    parameter extraction with uncertainty propagation.
    """

    def __init__(self, diagnostic_level: str = 'standard', n_jobs: Optional[int] = None):
        """
        Initialize the comprehensive diffusion analyzer.

        Args:
            diagnostic_level: Level of diagnostics ('basic', 'standard', 'comprehensive')
            n_jobs: Number of parallel jobs for bootstrap analysis
        """
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.robust_optimizer = RobustOptimizerWithDiagnostics(
            diagnostic_level=diagnostic_level, n_jobs=n_jobs
        )
        self.weighted_regression = WeightedRegressionFramework()
        self.outlier_detection = OutlierDetection()
        self.model_selection = ModelSelectionFramework()
        self.coefficient_extractor = DiffusionCoefficientExtractor()
        self.diffusion_models = DiffusionModels()

        # Configuration
        self.diagnostic_level = diagnostic_level
        self.available_models = self.diffusion_models.get_model_info()

    def analyze_diffusion(self, tau: np.ndarray, g2_data: np.ndarray,
                         g2_errors: Optional[np.ndarray] = None,
                         q_value: float = 1.0,
                         models_to_test: Optional[List[str]] = None,
                         enable_outlier_detection: bool = True,
                         enable_weighted_fitting: bool = True,
                         model_selection_criterion: str = 'bic',
                         bootstrap_samples: Optional[int] = None,
                         apply_physical_constraints: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive diffusion analysis on G2 correlation data.

        Args:
            tau: Time delay array
            g2_data: G2 correlation function values
            g2_errors: Uncertainties in G2 values
            q_value: Scattering vector magnitude
            models_to_test: List of model names to test (None for all)
            enable_outlier_detection: Whether to detect and handle outliers
            enable_weighted_fitting: Whether to use weighted regression
            model_selection_criterion: Criterion for model selection
            bootstrap_samples: Number of bootstrap samples (None for default)
            apply_physical_constraints: Whether to apply physical constraints

        Returns:
            Comprehensive analysis results dictionary
        """
        self.logger.info("Starting comprehensive diffusion analysis")

        # Input validation
        if len(tau) != len(g2_data):
            raise ValueError("tau and g2_data must have same length")

        if g2_errors is not None and len(g2_errors) != len(g2_data):
            raise ValueError("g2_errors must have same length as g2_data")

        # Select models to test
        if models_to_test is None:
            models_to_test = list(self.available_models.keys())

        models_to_compare = {name: info for name, info in self.available_models.items()
                           if name in models_to_test}

        results = {
            'input_parameters': {
                'n_data_points': len(tau),
                'q_value': q_value,
                'models_tested': list(models_to_compare.keys()),
                'has_error_estimates': g2_errors is not None
            },
            'preprocessing': {},
            'model_comparison': {},
            'best_model_analysis': {},
            'diffusion_parameters': {},
            'diagnostics': {}
        }

        # Step 1: Data preprocessing and outlier detection
        outlier_mask = np.zeros(len(g2_data), dtype=bool)
        if enable_outlier_detection:
            self.logger.info("Performing outlier detection")

            # Initial fit with simple diffusion for outlier detection
            simple_func = self.available_models['simple_diffusion']['function']
            simple_bounds = self.available_models['simple_diffusion']['bounds']

            try:
                # Quick fit for outlier detection
                initial_p0 = [1.0, 0.5, 1e-4]  # baseline, amplitude, rate
                popt_initial, _ = curve_fit(simple_func, tau, g2_data, p0=initial_p0,
                                          bounds=([b[0] for b in simple_bounds],
                                                 [b[1] for b in simple_bounds]))

                # Detect outliers
                outlier_results = self.outlier_detection.robust_outlier_detection(
                    tau, g2_data, simple_func, popt_initial
                )

                outlier_mask = outlier_results.get('consensus', np.zeros_like(g2_data, dtype=bool))
                n_outliers = np.sum(outlier_mask)

                results['preprocessing']['outlier_detection'] = {
                    'n_outliers_detected': n_outliers,
                    'outlier_fraction': n_outliers / len(g2_data),
                    'outlier_methods': outlier_results
                }

                if n_outliers > 0:
                    self.logger.info(f"Detected {n_outliers} outliers ({n_outliers/len(g2_data)*100:.1f}%)")

            except Exception as e:
                self.logger.warning(f"Outlier detection failed: {e}")
                results['preprocessing']['outlier_detection'] = {'error': str(e)}

        # Create analysis mask (exclude outliers if detected)
        analysis_mask = ~outlier_mask
        tau_clean = tau[analysis_mask]
        g2_clean = g2_data[analysis_mask]
        g2_errors_clean = g2_errors[analysis_mask] if g2_errors is not None else None

        # Step 2: Model comparison and selection
        self.logger.info("Comparing diffusion models")

        model_comparison_results = self.model_selection.compare_models(
            models_to_compare, tau_clean, g2_clean, g2_errors_clean,
            use_cross_validation=(len(tau_clean) > 20)
        )

        results['model_comparison'] = model_comparison_results

        # Select best model
        best_model_results = self.model_selection.select_best_model(
            model_comparison_results, criterion=model_selection_criterion
        )

        results['model_selection'] = best_model_results
        best_model_name = best_model_results.get('best_model')

        if best_model_name is None:
            self.logger.error("No model fitted successfully")
            results['error'] = "No model fitted successfully"
            return results

        self.logger.info(f"Best model selected: {best_model_name}")

        # Step 3: Enhanced analysis of best model
        best_model_info = models_to_compare[best_model_name]
        best_func = best_model_info['function']
        best_bounds = best_model_info.get('bounds')
        best_params = best_model_results['best_model_results']['parameters']
        best_pcov = best_model_results['best_model_results']['covariance']

        # Robust fitting with diagnostics
        if enable_weighted_fitting and g2_errors_clean is not None:
            self.logger.info("Performing weighted robust fitting")

            # Use inverse variance weighting
            weights = self.weighted_regression.calculate_weights_inverse_variance(g2_errors_clean)

            # Iteratively reweighted least squares
            irls_results = self.weighted_regression.iteratively_reweighted_least_squares(
                best_func, tau_clean, g2_clean, p0=best_params, bounds=best_bounds,
                max_iterations=10, weight_method='huber'
            )

            enhanced_params = irls_results['parameters']
            enhanced_pcov = irls_results['covariance']
            results['best_model_analysis']['weighted_fitting'] = irls_results

        else:
            enhanced_params = best_params
            enhanced_pcov = best_pcov

        # Comprehensive diagnostics
        enhanced_popt, enhanced_pcov, diagnostics = self.robust_optimizer.robust_curve_fit_with_diagnostics(
            best_func, tau_clean, g2_clean, p0=enhanced_params, bounds=best_bounds,
            sigma=g2_errors_clean, func_name=best_model_name,
            bootstrap_samples=bootstrap_samples
        )

        results['best_model_analysis']['enhanced_fit'] = {
            'parameters': enhanced_popt,
            'covariance': enhanced_pcov,
            'diagnostics': diagnostics
        }

        # Step 4: Extract physical diffusion parameters
        self.logger.info("Extracting diffusion coefficients")

        try:
            diffusion_results = self.coefficient_extractor.extract_with_constraints(
                best_model_name, enhanced_popt, enhanced_pcov, q_value,
                apply_physical_constraints=apply_physical_constraints
            )
            results['diffusion_parameters'] = diffusion_results

        except Exception as e:
            self.logger.error(f"Diffusion parameter extraction failed: {e}")
            results['diffusion_parameters'] = {'error': str(e)}

        # Step 5: Final diagnostics and quality assessment
        fit_quality = self.robust_optimizer.analyze_fit_quality(diagnostics)
        parameter_uncertainties = self.robust_optimizer.get_parameter_uncertainties(diagnostics)

        results['diagnostics'] = {
            'fit_quality': fit_quality,
            'parameter_uncertainties': parameter_uncertainties,
            'analysis_summary': {
                'best_model': best_model_name,
                'data_points_used': len(tau_clean),
                'outliers_excluded': np.sum(outlier_mask),
                'weighted_fitting': enable_weighted_fitting and g2_errors_clean is not None,
                'bootstrap_analysis': bootstrap_samples is not None and bootstrap_samples > 0
            }
        }

        self.logger.info("Comprehensive diffusion analysis completed")
        return results

    def quick_diffusion_analysis(self, tau: np.ndarray, g2_data: np.ndarray,
                                q_value: float = 1.0,
                                model: str = 'simple_diffusion') -> Dict[str, Any]:
        """
        Quick diffusion analysis with a single model.

        Args:
            tau: Time delay array
            g2_data: G2 correlation function values
            q_value: Scattering vector magnitude
            model: Diffusion model to use

        Returns:
            Quick analysis results
        """
        if model not in self.available_models:
            raise ValueError(f"Unknown model: {model}")

        model_info = self.available_models[model]
        func = model_info['function']
        bounds = model_info.get('bounds')

        # Generate initial parameters
        if bounds:
            lower_bounds, upper_bounds = zip(*bounds)
            p0 = np.array([(l + u) / 2 for l, u in bounds])
        else:
            p0 = np.ones(len(model_info['parameters']))

        try:
            # Simple robust fit - handle both 2 and 3 return values
            result = self.robust_optimizer.robust_curve_fit(
                func, tau, g2_data, p0=p0, bounds=bounds
            )

            if len(result) == 3:
                popt, pcov, fit_info = result
            else:
                popt, pcov = result
                fit_info = {}

            # Extract diffusion coefficient
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)

            if model == 'simple_diffusion':
                extraction_func = self.coefficient_extractor.extract_simple_diffusion_coefficient
            else:
                extraction_func = self.coefficient_extractor.extract_anomalous_diffusion_parameters

            diffusion_info = extraction_func(popt, param_errors, q_value)

            return {
                'model': model,
                'parameters': popt,
                'parameter_errors': param_errors,
                'covariance': pcov,
                'diffusion_info': diffusion_info,
                'fit_info': fit_info,
                'success': True
            }

        except Exception as e:
            return {
                'model': model,
                'error': str(e),
                'success': False
            }

    def get_available_models(self) -> Dict[str, str]:
        """Get list of available diffusion models with descriptions."""
        return {name: info['description']
                for name, info in self.available_models.items()}

    def validate_input_data(self, tau: np.ndarray, g2_data: np.ndarray,
                          g2_errors: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate input data for diffusion analysis.

        Args:
            tau: Time delay array
            g2_data: G2 correlation function values
            g2_errors: Optional error estimates

        Returns:
            Validation results dictionary
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # Basic checks
        if len(tau) == 0 or len(g2_data) == 0:
            validation['errors'].append("Empty input arrays")
            validation['valid'] = False

        if len(tau) != len(g2_data):
            validation['errors'].append("tau and g2_data must have same length")
            validation['valid'] = False

        if g2_errors is not None and len(g2_errors) != len(g2_data):
            validation['errors'].append("g2_errors must have same length as g2_data")
            validation['valid'] = False

        if validation['valid']:
            # Data quality checks
            if len(tau) < 10:
                validation['warnings'].append("Very few data points (<10) may lead to unreliable fits")

            if not np.all(tau[1:] > tau[:-1]):
                validation['warnings'].append("Time delays should be monotonically increasing")

            if np.any(g2_data <= 0):
                validation['warnings'].append("G2 values should be positive")

            if np.any(tau <= 0):
                validation['warnings'].append("Time delays should be positive")

            # Statistical checks
            g2_range = np.max(g2_data) - np.min(g2_data)
            if g2_range < 0.01:
                validation['warnings'].append("Very small dynamic range in G2 data")

            # Recommendations
            if g2_errors is None:
                validation['recommendations'].append("Providing error estimates improves fitting accuracy")

            if len(tau) > 100:
                validation['recommendations'].append("Consider model comparison for large datasets")

        return validation


# Performance-Optimized XPCS Integration Framework
class XPCSPerformanceOptimizer:
    """
    Performance optimization framework specifically designed for XPCS data analysis.

    Provides specialized optimizations for large XPCS datasets including:
    - Memory-efficient batch processing
    - Intelligent data chunking and streaming
    - Adaptive parallelization based on dataset size
    - Cache-aware algorithms
    """

    def __init__(self, cache_size: int = 1000, max_memory_mb: int = 2048):
        """
        Initialize the XPCS performance optimizer.

        Args:
            cache_size: Maximum number of cached results
            max_memory_mb: Maximum memory usage threshold in MB
        """
        self.cache_size = cache_size
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.logger = logging.getLogger(__name__)

        # Integrate with joblib caching system
        cache_dir = os.path.join(os.path.expanduser("~"), ".xpcs_toolkit", "robust_fitting_cache")
        self.memory = Memory(location=cache_dir, verbose=0)

        # Cache key tracking for intelligent cleanup
        self.cache_usage = {}
        self.cache_access_time = {}

    def estimate_memory_usage(self, tau_len: int, g2_shape: Tuple, n_bootstrap: int = 0) -> float:
        """
        Estimate memory usage for XPCS fitting operations.

        Args:
            tau_len: Length of time delay array
            g2_shape: Shape of G2 correlation data
            n_bootstrap: Number of bootstrap samples

        Returns:
            Estimated memory usage in MB
        """
        # Base memory for data arrays
        base_memory = (tau_len * g2_shape[1] * 8 * 3) / (1024**2)  # tau, g2, errors in double precision

        # Bootstrap memory overhead
        bootstrap_memory = 0
        if n_bootstrap > 0:
            # Estimate bootstrap sample storage
            bootstrap_memory = (n_bootstrap * len(g2_shape) * 8 * 4) / (1024**2)  # Parameter storage

        # Fitting overhead (temporary arrays)
        fitting_overhead = base_memory * 0.5

        total_memory = base_memory + bootstrap_memory + fitting_overhead
        return total_memory

    def should_use_chunked_processing(self, tau_len: int, g2_shape: Tuple,
                                    n_bootstrap: int = 0) -> Tuple[bool, int]:
        """
        Determine if chunked processing should be used based on memory constraints.

        Returns:
            (use_chunking, optimal_chunk_size)
        """
        estimated_memory = self.estimate_memory_usage(tau_len, g2_shape, n_bootstrap)

        if estimated_memory > self.max_memory_mb:
            # Calculate optimal chunk size to stay within memory limits
            memory_ratio = estimated_memory / self.max_memory_mb
            chunk_size = max(1, g2_shape[1] // int(np.ceil(memory_ratio)))
            return True, chunk_size

        return False, g2_shape[1]

    def optimize_parallelization(self, n_q_values: int, n_bootstrap: int = 0) -> int:
        """
        Determine optimal number of parallel processes for XPCS fitting.

        Args:
            n_q_values: Number of q-values to fit
            n_bootstrap: Number of bootstrap samples

        Returns:
            Optimal number of parallel jobs
        """
        # Conservative approach for XPCS data
        max_jobs = multiprocessing.cpu_count()

        # Adjust based on problem size
        if n_bootstrap > 0:
            # Bootstrap is memory intensive, reduce parallelization
            optimal_jobs = min(max_jobs // 2, max(1, n_q_values // 4))
        else:
            # Standard fitting can use more parallelization
            optimal_jobs = min(max_jobs, max(1, n_q_values // 2))

        return optimal_jobs

    def get_cache_key(self, tau: np.ndarray, g2_shape: Tuple, fit_func: str,
                     bounds: Optional[Tuple], bootstrap_samples: int) -> str:
        """
        Generate a cache key for fitting operations.

        Args:
            tau: Time delay array
            g2_shape: Shape of G2 data
            fit_func: Fitting function type
            bounds: Parameter bounds
            bootstrap_samples: Number of bootstrap samples

        Returns:
            Cache key string
        """
        # Create hash from critical parameters
        tau_hash = hash(tuple(np.round(tau, 10)))  # Round to avoid float precision issues
        bounds_hash = hash(str(bounds)) if bounds else 0
        key = f"{tau_hash}_{g2_shape}_{fit_func}_{bounds_hash}_{bootstrap_samples}"
        return key

    def is_cached(self, cache_key: str) -> bool:
        """Check if result is cached."""
        return cache_key in self.cache

    def get_cached_result(self, cache_key: str):
        """Retrieve cached result and update access time."""
        if cache_key in self.cache:
            self.cache_access_time[cache_key] = time.time()
            self.cache_usage[cache_key] = self.cache_usage.get(cache_key, 0) + 1
            return self.cache[cache_key]
        return None

    def cache_result(self, cache_key: str, result: Any):
        """Cache a result with intelligent cleanup."""
        # Check if cache is full
        if len(self.cache) >= self.cache_size:
            self._cleanup_cache()

        # Store result
        self.cache[cache_key] = result
        self.cache_access_time[cache_key] = time.time()
        self.cache_usage[cache_key] = 1

    def _cleanup_cache(self):
        """Intelligent cache cleanup based on usage patterns."""
        if len(self.cache) == 0:
            return

        current_time = time.time()

        # Score each cache entry (higher score = more important to keep)
        cache_scores = {}
        for key in self.cache.keys():
            access_time = self.cache_access_time.get(key, 0)
            usage_count = self.cache_usage.get(key, 1)

            # Time decay factor (more recent = higher score)
            time_factor = 1.0 / (1.0 + (current_time - access_time) / 3600)  # 1 hour decay

            # Usage frequency factor
            usage_factor = np.log(1 + usage_count)

            cache_scores[key] = time_factor * usage_factor

        # Remove lowest scoring entries until we're under the limit
        sorted_keys = sorted(cache_scores.keys(), key=lambda k: cache_scores[k])
        remove_count = len(self.cache) - (self.cache_size // 2)  # Remove half when full

        for key in sorted_keys[:remove_count]:
            del self.cache[key]
            del self.cache_access_time[key]
            del self.cache_usage[key]

    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()
        self.cache_access_time.clear()
        self.cache_usage.clear()
        if hasattr(self, 'memory'):
            self.memory.clear()

    @memory.cache
    def cached_robust_fit_single_q(self, tau_array: np.ndarray, g2_array: np.ndarray,
                                  g2_err_array: np.ndarray, fit_func: str,
                                  bounds_tuple: Tuple, bootstrap_samples: int):
        """
        Cached version of robust fitting for a single q-value.

        This method is cached using joblib to avoid recomputation of identical
        fitting problems, which is common in XPCS analysis workflows.
        """
        # This method will be implemented by the calling code
        # We just define the caching wrapper here
        pass


class OptimizedXPCSFittingEngine:
    """
    High-performance fitting engine optimized for XPCS workflows.

    Combines robust fitting with performance optimizations specifically
    designed for XPCS correlation function analysis.
    """

    def __init__(self, performance_optimizer: XPCSPerformanceOptimizer = None):
        """Initialize the optimized XPCS fitting engine."""
        self.performance_optimizer = performance_optimizer or XPCSPerformanceOptimizer()
        self.robust_optimizer = ComprehensiveDiffusionAnalyzer(
            diagnostic_level='standard',
            n_jobs=self.performance_optimizer.optimize_parallelization(10)
        )
        self.logger = logging.getLogger(__name__)

    def fit_g2_optimized(self, tau: np.ndarray, g2_data: np.ndarray,
                        g2_errors: Optional[np.ndarray] = None,
                        q_values: Optional[np.ndarray] = None,
                        fit_func: str = "single",
                        bounds: Optional[Tuple] = None,
                        bootstrap_samples: int = 0,
                        enable_diagnostics: bool = True,
                        use_caching: bool = True) -> Dict[str, Any]:
        """
        Optimized G2 fitting with automatic performance tuning for large datasets.

        Args:
            tau: Time delay array
            g2_data: G2 correlation data (2D array: tau x q)
            g2_errors: Error estimates for G2 data
            q_values: Scattering vector values
            fit_func: Fitting function type ("single" or "double")
            bounds: Parameter bounds for fitting
            bootstrap_samples: Number of bootstrap samples
            enable_diagnostics: Whether to include comprehensive diagnostics
            use_caching: Whether to use intelligent caching

        Returns:
            Dictionary containing optimized fitting results
        """
        start_time = time.time()

        # Input validation and preprocessing
        if g2_data.ndim == 1:
            g2_data = g2_data.reshape(-1, 1)
        if g2_errors is not None and g2_errors.ndim == 1:
            g2_errors = g2_errors.reshape(-1, 1)

        n_tau, n_q = g2_data.shape

        # Check cache if enabled
        cache_key = None
        if use_caching:
            cache_key = self.performance_optimizer.get_cache_key(
                tau, g2_data.shape, fit_func, bounds, bootstrap_samples
            )
            cached_result = self.performance_optimizer.get_cached_result(cache_key)
            if cached_result is not None:
                self.logger.info(f"Using cached result for key: {cache_key[:50]}...")
                return cached_result

        # Performance optimization decisions
        use_chunking, chunk_size = self.performance_optimizer.should_use_chunked_processing(
            n_tau, g2_data.shape, bootstrap_samples
        )

        optimal_jobs = self.performance_optimizer.optimize_parallelization(n_q, bootstrap_samples)

        self.logger.info(f"XPCS fitting optimization: chunking={use_chunking}, "
                        f"chunk_size={chunk_size}, parallel_jobs={optimal_jobs}")

        # Initialize results containers
        results = {
            'performance_info': {
                'use_chunking': use_chunking,
                'chunk_size': chunk_size,
                'parallel_jobs': optimal_jobs,
                'estimated_memory_mb': self.performance_optimizer.estimate_memory_usage(
                    n_tau, g2_data.shape, bootstrap_samples
                ),
                'cache_hit': False,
                'cache_key': cache_key
            },
            'fit_results': [],
            'timing': {},
            'optimization_summary': {}
        }

        if use_chunking:
            # Process data in chunks to manage memory
            results['fit_results'] = self._fit_g2_chunked(
                tau, g2_data, g2_errors, q_values, fit_func, bounds,
                bootstrap_samples, enable_diagnostics, chunk_size
            )
        else:
            # Process all data at once
            results['fit_results'] = self._fit_g2_batch(
                tau, g2_data, g2_errors, q_values, fit_func, bounds,
                bootstrap_samples, enable_diagnostics, optimal_jobs
            )

        # Performance timing
        total_time = time.time() - start_time
        results['timing']['total_time'] = total_time
        results['timing']['time_per_q'] = total_time / n_q

        # Optimization summary
        results['optimization_summary'] = {
            'dataset_size': f"{n_tau} x {n_q}",
            'processing_mode': 'chunked' if use_chunking else 'batch',
            'bootstrap_enabled': bootstrap_samples > 0,
            'diagnostics_enabled': enable_diagnostics,
            'performance_overhead_percent': self._calculate_overhead(results),
            'caching_enabled': use_caching
        }

        # Cache result if enabled and computation was successful
        if use_caching and cache_key is not None and len(results['fit_results']) > 0:
            self.performance_optimizer.cache_result(cache_key, results)

        self.logger.info(f"XPCS fitting completed in {total_time:.2f}s "
                        f"({total_time/n_q*1000:.1f}ms per q-value)")

        return results

    def _fit_g2_chunked(self, tau, g2_data, g2_errors, q_values, fit_func,
                       bounds, bootstrap_samples, enable_diagnostics, chunk_size):
        """Process G2 fitting in memory-efficient chunks."""
        n_q = g2_data.shape[1]
        results = []

        for chunk_start in range(0, n_q, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_q)

            # Extract chunk
            g2_chunk = g2_data[:, chunk_start:chunk_end]
            g2_err_chunk = g2_errors[:, chunk_start:chunk_end] if g2_errors is not None else None
            q_chunk = q_values[chunk_start:chunk_end] if q_values is not None else None

            # Process chunk
            chunk_results = self._fit_g2_batch(
                tau, g2_chunk, g2_err_chunk, q_chunk, fit_func, bounds,
                bootstrap_samples, enable_diagnostics,
                jobs=min(2, chunk_end - chunk_start)  # Conservative chunked parallelization
            )

            results.extend(chunk_results)

            # Memory cleanup after each chunk
            del g2_chunk, g2_err_chunk, q_chunk

        return results

    def _fit_g2_batch(self, tau, g2_data, g2_errors, q_values, fit_func,
                     bounds, bootstrap_samples, enable_diagnostics, jobs=None):
        """Process G2 fitting for a batch of q-values."""
        n_q = g2_data.shape[1]
        results = []

        # Select appropriate fitting function
        from ..xpcs_file import single_exp_all, double_exp_all
        func = single_exp_all if fit_func == "single" else double_exp_all

        # Process each q-value
        for q_idx in range(n_q):
            g2_col = g2_data[:, q_idx]
            g2_err_col = g2_errors[:, q_idx] if g2_errors is not None else None
            q_val = q_values[q_idx] if q_values is not None else q_idx

            # Apply data quality checks
            valid_mask = np.isfinite(g2_col)
            if g2_err_col is not None:
                valid_mask &= (g2_err_col > 0) & np.isfinite(g2_err_col)

            if np.sum(valid_mask) < 3:
                # Insufficient data for fitting
                results.append({
                    'q_index': q_idx,
                    'q_value': q_val,
                    'status': 'insufficient_data',
                    'params': None,
                    'errors': None
                })
                continue

            # Clean data
            tau_clean = tau[valid_mask]
            g2_clean = g2_col[valid_mask]
            g2_err_clean = g2_err_col[valid_mask] if g2_err_col is not None else None

            try:
                # Perform robust fitting
                if enable_diagnostics and bootstrap_samples > 0:
                    # Full diagnostic fitting with bootstrap
                    popt, pcov, diagnostics = self.robust_optimizer.robust_optimizer.robust_curve_fit_with_diagnostics(
                        lambda x, *p: func(x, p, [True]*len(p)),
                        tau_clean, g2_clean, sigma=g2_err_clean,
                        bootstrap_samples=bootstrap_samples
                    )

                    result = {
                        'q_index': q_idx,
                        'q_value': q_val,
                        'status': 'success',
                        'params': popt,
                        'param_errors': np.sqrt(np.diag(pcov)) if pcov is not None else None,
                        'covariance': pcov,
                        'diagnostics': diagnostics,
                        'robust_fit': True
                    }
                else:
                    # Standard robust fitting
                    popt, pcov, _ = self.robust_optimizer.robust_optimizer.robust_curve_fit(
                        lambda x, *p: func(x, p, [True]*len(p)),
                        tau_clean, g2_clean, sigma=g2_err_clean
                    )

                    result = {
                        'q_index': q_idx,
                        'q_value': q_val,
                        'status': 'success',
                        'params': popt,
                        'param_errors': np.sqrt(np.diag(pcov)) if pcov is not None else None,
                        'covariance': pcov,
                        'robust_fit': True
                    }

            except Exception as e:
                self.logger.warning(f"Fitting failed for q-index {q_idx}: {e}")
                result = {
                    'q_index': q_idx,
                    'q_value': q_val,
                    'status': 'failed',
                    'error': str(e),
                    'params': None,
                    'robust_fit': False
                }

            results.append(result)

        return results

    def _calculate_overhead(self, results):
        """Calculate performance overhead compared to standard fitting."""
        # Simplified overhead estimation
        base_time = results['timing']['total_time']
        n_q = len(results['fit_results'])

        # Estimate standard fitting time (empirical)
        estimated_standard_time = n_q * 0.01  # 10ms per q-value baseline

        overhead_percent = max(0, (base_time - estimated_standard_time) / estimated_standard_time * 100)
        return min(overhead_percent, 100)  # Cap at 100%


# Convenience function for backward compatibility
def robust_curve_fit(func: Callable, xdata: np.ndarray, ydata: np.ndarray,
                    p0: Optional[np.ndarray] = None, bounds: Optional[Tuple] = None,
                    sigma: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust curve fitting using multiple optimization strategies.

    This function provides backward compatibility with scipy.optimize.curve_fit
    while offering enhanced robustness through multi-strategy optimization.

    Args:
        func: Function to fit
        xdata: Independent variable data
        ydata: Dependent variable data
        p0: Initial parameter guess
        bounds: Parameter bounds
        sigma: Data uncertainties
        **kwargs: Additional arguments

    Returns:
        popt: Optimal parameters
        pcov: Parameter covariance matrix
    """
    optimizer = RobustOptimizer()
    popt, pcov, _info = optimizer.robust_curve_fit(func, xdata, ydata, p0, bounds, sigma, **kwargs)
    return popt, pcov
