import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from joblib import Memory
from scipy.optimize import curve_fit
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
                "err_msg": f"q_index fit error: {str(e)}",
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
        msg = f"Fitting failed: {str(err)}"
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
    if p0 is None:
        p0 = np.mean(bounds_fit, axis=0)
    else:
        p0 = np.array(p0)[fit_flag]

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

    elif func_type == "power":
        # For power law: y = A * x^(-n)
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)
        # Linear regression on log-log data
        coeffs = np.polyfit(log_x, log_y, 1)
        n_est = -coeffs[0]
        A_est = np.exp(coeffs[1])

        return [A_est, n_est]

    elif func_type == "gaussian":
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
