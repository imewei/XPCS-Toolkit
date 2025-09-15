"""
Statistical Properties and Constraints

This module defines statistical properties that XPCS algorithms must satisfy,
including parameter estimation properties, uncertainty quantification,
and statistical test validation.
"""

import numpy as np
from scipy import stats


def verify_parameter_uncertainty_scaling(
    parameter_estimates: np.ndarray, parameter_errors: np.ndarray, n_samples: np.ndarray
) -> tuple[bool, str]:
    """
    Verify that parameter uncertainties scale properly with sample size

    For most estimators: σ ∝ 1/√N

    Args:
        parameter_estimates: Parameter estimates for different sample sizes
        parameter_errors: Corresponding parameter uncertainties
        n_samples: Sample sizes used

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(parameter_estimates) != len(parameter_errors) or len(
        parameter_estimates
    ) != len(n_samples):
        return False, "Arrays must have same length"

    if len(n_samples) < 3:
        return False, "Need at least 3 different sample sizes"

    # Expected scaling: error ∝ N^(-0.5)
    expected_scaling = -0.5

    # Fit log(error) vs log(N)
    log_n = np.log10(n_samples)
    log_error = np.log10(parameter_errors)

    try:
        fit_coeff = np.polyfit(log_n, log_error, 1)
        measured_scaling = fit_coeff[0]

        r_squared = np.corrcoef(log_n, log_error)[0, 1] ** 2

        # Check scaling exponent
        scaling_error = abs(measured_scaling - expected_scaling)
        if scaling_error > 0.3:  # Allow some deviation
            return (
                False,
                f"Scaling exponent {measured_scaling:.3f} != {expected_scaling:.3f}",
            )

        # Check fit quality
        if r_squared < 0.7:
            return False, f"Poor scaling fit, R² = {r_squared:.3f}"

        return True, f"Parameter uncertainty scaling valid: {measured_scaling:.3f}"

    except Exception as e:
        return False, f"Failed to analyze uncertainty scaling: {e}"


def verify_chi_squared_distribution(
    residuals: np.ndarray,
    errors: np.ndarray,
    n_params: int,
    significance_level: float = 0.05,
) -> tuple[bool, str]:
    """
    Verify that normalized residuals follow chi-squared distribution

    Args:
        residuals: Fitting residuals
        errors: Expected errors for each data point
        n_params: Number of fitted parameters
        significance_level: Significance level for chi-squared test

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(residuals) != len(errors):
        return False, "Residuals and errors must have same length"

    if len(residuals) <= n_params:
        return False, "Need more data points than parameters"

    # Calculate chi-squared statistic
    chi_squared_values = (residuals / errors) ** 2
    chi_squared_total = np.sum(chi_squared_values)

    degrees_of_freedom = len(residuals) - n_params
    reduced_chi_squared = chi_squared_total / degrees_of_freedom

    # Test against chi-squared distribution
    p_value = 1 - stats.chi2.cdf(chi_squared_total, degrees_of_freedom)

    # Check if chi-squared is reasonable
    if p_value < significance_level:
        return False, f"Chi-squared test failed, p-value = {p_value:.4f}"

    # Check reduced chi-squared is near 1
    if reduced_chi_squared < 0.5 or reduced_chi_squared > 2.0:
        return False, f"Reduced chi-squared = {reduced_chi_squared:.3f} not near 1"

    return (
        True,
        f"Chi-squared distribution valid, reduced χ² = {reduced_chi_squared:.3f}",
    )


def verify_residual_normality(
    residuals: np.ndarray, significance_level: float = 0.01
) -> tuple[bool, str]:
    """
    Verify that fitting residuals are normally distributed

    Args:
        residuals: Fitting residuals
        significance_level: Significance level for normality tests

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(residuals) < 8:
        return False, "Need at least 8 residuals for normality test"

    # Remove outliers (beyond 3 sigma)
    z_scores = np.abs(stats.zscore(residuals))
    clean_residuals = residuals[z_scores < 3]

    if len(clean_residuals) < 8:
        return False, "Too many outliers, cannot test normality"

    # Shapiro-Wilk test for normality
    try:
        _shapiro_stat, shapiro_p = stats.shapiro(clean_residuals)

        if shapiro_p < significance_level:
            return False, f"Residuals not normal (Shapiro-Wilk p = {shapiro_p:.4f})"

        # Additional check: Anderson-Darling test
        anderson_result = stats.anderson(clean_residuals, dist="norm")
        critical_value = anderson_result.critical_values[2]  # 5% significance level

        if anderson_result.statistic > critical_value:
            return (
                False,
                f"Residuals not normal (Anderson-Darling = {anderson_result.statistic:.4f})",
            )

        return True, f"Residuals normally distributed (p = {shapiro_p:.4f})"

    except Exception as e:
        return False, f"Failed to test residual normality: {e}"


def verify_bootstrap_consistency(
    original_estimate: float,
    bootstrap_estimates: np.ndarray,
    bootstrap_error: float,
    confidence_level: float = 0.68,
) -> tuple[bool, str]:
    """
    Verify bootstrap parameter estimation consistency

    Args:
        original_estimate: Parameter estimate from original data
        bootstrap_estimates: Parameter estimates from bootstrap samples
        bootstrap_error: Bootstrap error estimate
        confidence_level: Confidence level for consistency check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(bootstrap_estimates) < 50:
        return False, "Need at least 50 bootstrap samples"

    # Remove NaN values
    valid_estimates = bootstrap_estimates[np.isfinite(bootstrap_estimates)]
    if len(valid_estimates) < 50:
        return False, "Too many invalid bootstrap estimates"

    # Bootstrap statistics
    bootstrap_mean = np.mean(valid_estimates)
    bootstrap_std = np.std(valid_estimates, ddof=1)

    # Check bias
    bias = bootstrap_mean - original_estimate
    relative_bias = (
        abs(bias) / abs(original_estimate) if original_estimate != 0 else abs(bias)
    )

    if relative_bias > 0.1:  # 10% bias threshold
        return False, f"Large bootstrap bias: {relative_bias:.3f}"

    # Check error estimate consistency
    error_ratio = bootstrap_std / bootstrap_error if bootstrap_error > 0 else np.inf

    if error_ratio < 0.7 or error_ratio > 1.5:
        return False, f"Bootstrap error inconsistent: ratio = {error_ratio:.3f}"

    # Check confidence interval coverage
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    ci_lower = np.percentile(valid_estimates, lower_percentile)
    ci_upper = np.percentile(valid_estimates, upper_percentile)

    # Original estimate should be within confidence interval most of the time
    within_ci = ci_lower <= original_estimate <= ci_upper

    if not within_ci:
        # Check if it's close to the boundary
        distance_to_ci = min(
            abs(original_estimate - ci_lower), abs(original_estimate - ci_upper)
        )
        relative_distance = distance_to_ci / bootstrap_std

        if relative_distance > 1.0:  # More than 1 standard deviation away
            return False, "Original estimate outside confidence interval"

    return True, f"Bootstrap consistency valid, bias = {relative_bias:.4f}"


def verify_correlation_structure(
    correlation_matrix: np.ndarray,
    expected_correlations: dict | None = None,
    tolerance: float = 0.2,
) -> tuple[bool, str]:
    """
    Verify parameter correlation structure in fitting

    Args:
        correlation_matrix: Parameter correlation matrix
        expected_correlations: Expected correlations between specific parameter pairs
        tolerance: Tolerance for correlation comparisons

    Returns:
        Tuple of (is_valid, error_message)
    """
    n_params = correlation_matrix.shape[0]

    # Check matrix properties
    if not np.allclose(correlation_matrix, correlation_matrix.T):
        return False, "Correlation matrix not symmetric"

    if not np.allclose(np.diag(correlation_matrix), 1.0):
        return False, "Correlation matrix diagonal not equal to 1"

    # Check eigenvalues (positive semi-definite)
    try:
        eigenvals = np.linalg.eigvals(correlation_matrix)
        if np.any(eigenvals < -1e-10):
            return False, "Correlation matrix not positive semi-definite"
    except Exception:
        return False, "Failed to compute correlation matrix eigenvalues"

    # Check off-diagonal elements are in [-1, 1]
    off_diag_mask = ~np.eye(n_params, dtype=bool)
    off_diag_values = correlation_matrix[off_diag_mask]

    if np.any(off_diag_values < -1) or np.any(off_diag_values > 1):
        return False, "Correlation coefficients outside [-1, 1] range"

    # Check expected correlations if provided
    if expected_correlations is not None:
        for (i, j), expected_corr in expected_correlations.items():
            if i < n_params and j < n_params:
                actual_corr = correlation_matrix[i, j]
                error = abs(actual_corr - expected_corr)

                if error > tolerance:
                    return (
                        False,
                        f"Correlation ({i},{j}) = {actual_corr:.3f}, "
                        f"expected {expected_corr:.3f}",
                    )

    return True, "Correlation structure valid"


def verify_monte_carlo_convergence(
    estimates: np.ndarray, true_value: float, convergence_rate: float = -0.5
) -> tuple[bool, str]:
    """
    Verify Monte Carlo estimation convergence

    Args:
        estimates: Cumulative estimates vs sample size
        true_value: Known true value
        convergence_rate: Expected convergence rate (typically -0.5 for MC)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(estimates) < 10:
        return False, "Need at least 10 estimates for convergence analysis"

    # Calculate errors vs sample size
    sample_sizes = np.arange(1, len(estimates) + 1)
    errors = np.abs(estimates - true_value)

    # Remove zeros to avoid log issues
    nonzero_mask = errors > 1e-15
    if np.sum(nonzero_mask) < 5:
        return True, "Estimates converged to machine precision"

    log_n = np.log10(sample_sizes[nonzero_mask])
    log_error = np.log10(errors[nonzero_mask])

    # Fit convergence rate
    try:
        fit_coeff = np.polyfit(log_n, log_error, 1)
        measured_rate = fit_coeff[0]

        r_squared = np.corrcoef(log_n, log_error)[0, 1] ** 2

        # Check convergence rate
        rate_error = abs(measured_rate - convergence_rate)
        if rate_error > 0.3:
            return (
                False,
                f"Convergence rate {measured_rate:.3f} != {convergence_rate:.3f}",
            )

        # Check if actually converging
        if measured_rate > 0:
            return False, f"Estimates diverging, rate = {measured_rate:.3f}"

        # Check fit quality
        if r_squared < 0.6:
            return False, f"Poor convergence fit, R² = {r_squared:.3f}"

        return True, f"Monte Carlo convergence valid, rate = {measured_rate:.3f}"

    except Exception as e:
        return False, f"Failed to analyze convergence: {e}"


def verify_statistical_power(
    test_statistics: np.ndarray,
    null_hypothesis_stats: np.ndarray,
    significance_level: float = 0.05,
    expected_power: float = 0.8,
) -> tuple[bool, str]:
    """
    Verify statistical power of hypothesis tests

    Args:
        test_statistics: Test statistics under alternative hypothesis
        null_hypothesis_stats: Test statistics under null hypothesis
        significance_level: Type I error rate
        expected_power: Expected statistical power

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(test_statistics) < 50 or len(null_hypothesis_stats) < 50:
        return False, "Need at least 50 samples for power analysis"

    # Determine critical value from null distribution
    critical_value = np.percentile(
        null_hypothesis_stats, 100 * (1 - significance_level)
    )

    # Calculate actual Type I error (false positive rate)
    false_positives = np.sum(null_hypothesis_stats > critical_value)
    type_i_error = false_positives / len(null_hypothesis_stats)

    # Calculate statistical power (true positive rate)
    true_positives = np.sum(test_statistics > critical_value)
    power = true_positives / len(test_statistics)

    # Check Type I error
    if abs(type_i_error - significance_level) > significance_level * 0.5:
        return False, f"Type I error {type_i_error:.3f} != {significance_level:.3f}"

    # Check statistical power
    if power < expected_power * 0.8:  # Allow 20% deviation
        return False, f"Statistical power {power:.3f} < {expected_power:.3f}"

    return True, f"Statistical power valid: {power:.3f}"


def verify_information_criteria(
    log_likelihood: float,
    n_params: int,
    n_data: int,
    reference_aic: float | None = None,
    reference_bic: float | None = None,
) -> tuple[bool, str]:
    """
    Verify information criteria calculations (AIC, BIC)

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of model parameters
        n_data: Number of data points
        reference_aic: Reference AIC value for comparison
        reference_bic: Reference BIC value for comparison

    Returns:
        Tuple of (is_valid, error_message)
    """
    if n_params <= 0 or n_data <= 0:
        return False, "Number of parameters and data points must be positive"

    if n_params >= n_data:
        return False, "Cannot have more parameters than data points"

    # Calculate information criteria
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_data) * n_params - 2 * log_likelihood

    # AIC should be finite
    if not np.isfinite(aic):
        return False, f"AIC is not finite: {aic}"

    # BIC should be finite
    if not np.isfinite(bic):
        return False, f"BIC is not finite: {bic}"

    # BIC should be larger than AIC for n_data > e^2 ≈ 7.39
    if n_data > 8 and bic <= aic:
        return (
            False,
            f"BIC ({bic:.2f}) should be larger than AIC ({aic:.2f}) for large n",
        )

    # Compare with reference values if provided
    if reference_aic is not None:
        aic_diff = aic - reference_aic
        if abs(aic_diff) > 1e-6:
            return False, f"AIC differs from reference by {aic_diff:.2e}"

    if reference_bic is not None:
        bic_diff = bic - reference_bic
        if abs(bic_diff) > 1e-6:
            return False, f"BIC differs from reference by {bic_diff:.2e}"

    return True, f"Information criteria valid: AIC = {aic:.2f}, BIC = {bic:.2f}"
