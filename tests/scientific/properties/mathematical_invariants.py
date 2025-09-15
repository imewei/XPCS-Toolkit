"""
Mathematical Invariants for XPCS Analysis

This module defines mathematical properties and invariants that XPCS algorithms
must satisfy. These functions are used by property-based tests to validate
algorithm correctness across parameter ranges.

The invariants cover:
- G2 correlation function properties
- SAXS scattering properties
- Two-time correlation properties
- Diffusion analysis properties
- General mathematical constraints
"""

import numpy as np


def verify_g2_normalization(
    g2_data: np.ndarray, tau: np.ndarray, tolerance: float = 1e-10
) -> tuple[bool, str]:
    """
    Verify G2 normalization property: G2(τ=0) ≥ 1

    Args:
        g2_data: G2 correlation data, shape (n_times,) or (n_times, n_q)
        tau: Time delay values
        tolerance: Numerical tolerance for comparison

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(tau) == 0 or len(g2_data) == 0:
        return False, "Empty data provided"

    # Find τ=0 or closest to zero
    zero_idx = np.argmin(np.abs(tau))

    if g2_data.ndim == 1:
        g2_zero = g2_data[zero_idx]
        if g2_zero < 1.0 - tolerance:
            return False, f"G2(τ=0) = {g2_zero:.6f} < 1"
    else:
        # Multiple q-values
        g2_zero = g2_data[zero_idx, :]
        invalid_count = np.sum(g2_zero < 1.0 - tolerance)
        if invalid_count > 0:
            min_g2 = np.min(g2_zero)
            return (
                False,
                f"{invalid_count} q-values have G2(τ=0) < 1, minimum: {min_g2:.6f}",
            )

    return True, "G2 normalization valid"


def verify_g2_monotonicity(
    g2_data: np.ndarray, tau: np.ndarray, allow_exceptions: float = 0.05
) -> tuple[bool, str]:
    """
    Verify G2 monotonicity for simple exponential decay systems

    Args:
        g2_data: G2 correlation data
        tau: Time delay values (should be sorted)
        allow_exceptions: Fraction of points allowed to violate monotonicity (for noise)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(tau) < 2:
        return False, "Need at least 2 time points"

    if g2_data.ndim == 1:
        diff = np.diff(g2_data)
        violations = np.sum(diff > 0)  # Positive differences (increasing)
        violation_fraction = violations / len(diff)

        if violation_fraction > allow_exceptions:
            return False, f"Monotonicity violated in {violation_fraction:.3f} of points"

    else:
        # Check each q-value
        violation_fractions = []
        for q_idx in range(g2_data.shape[1]):
            diff = np.diff(g2_data[:, q_idx])
            violations = np.sum(diff > 0)
            violation_fractions.append(violations / len(diff))

        max_violations = np.max(violation_fractions)
        if max_violations > allow_exceptions:
            return False, f"Maximum monotonicity violation: {max_violations:.3f}"

    return True, "G2 monotonicity valid"


def verify_g2_asymptotic_behavior(
    g2_data: np.ndarray, tau: np.ndarray, baseline: float = 1.0, tolerance: float = 0.1
) -> tuple[bool, str]:
    """
    Verify G2 asymptotic behavior: G2(τ→∞) → baseline

    Args:
        g2_data: G2 correlation data
        tau: Time delay values
        baseline: Expected asymptotic value
        tolerance: Tolerance for asymptotic comparison

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(tau) < 10:
        return False, "Need at least 10 time points for asymptotic analysis"

    # Use last 10% of data points for asymptotic value
    asymptotic_region = int(0.9 * len(tau))

    if g2_data.ndim == 1:
        g2_asymptotic = np.mean(g2_data[asymptotic_region:])
        error = abs(g2_asymptotic - baseline)

        if error > tolerance:
            return False, f"G2(τ→∞) = {g2_asymptotic:.4f}, expected {baseline:.4f}"

    else:
        # Check each q-value
        for q_idx in range(g2_data.shape[1]):
            g2_asymptotic = np.mean(g2_data[asymptotic_region:, q_idx])
            error = abs(g2_asymptotic - baseline)

            if error > tolerance:
                return (
                    False,
                    f"G2(τ→∞, q={q_idx}) = {g2_asymptotic:.4f}, expected {baseline:.4f}",
                )

    return True, "G2 asymptotic behavior valid"


def verify_g2_causality(
    g2_matrix: np.ndarray, tolerance: float = 1e-12
) -> tuple[bool, str]:
    """
    Verify G2 causality principle through time-reversal symmetry

    Args:
        g2_matrix: Two-time correlation matrix G2(t1, t2)
        tolerance: Numerical tolerance for symmetry check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if g2_matrix.ndim != 2:
        return False, "G2 matrix must be 2D"

    if g2_matrix.shape[0] != g2_matrix.shape[1]:
        return False, "G2 matrix must be square"

    # Check symmetry: G2(t1, t2) = G2(t2, t1) for stationary processes
    symmetry_error = np.max(np.abs(g2_matrix - g2_matrix.T))

    if symmetry_error > tolerance:
        return False, f"G2 matrix not symmetric, max error: {symmetry_error:.2e}"

    return True, "G2 causality valid"


def verify_intensity_positivity(
    intensity: np.ndarray, min_value: float = 0.0
) -> tuple[bool, str]:
    """
    Verify that scattering intensities are non-negative

    Args:
        intensity: Intensity array
        min_value: Minimum allowed intensity value

    Returns:
        Tuple of (is_valid, error_message)
    """
    negative_count = np.sum(intensity < min_value)

    if negative_count > 0:
        min_intensity = np.min(intensity)
        negative_fraction = negative_count / len(intensity)
        return (
            False,
            f"{negative_count} negative intensities ({negative_fraction:.3f}), "
            f"minimum: {min_intensity:.2e}",
        )

    return True, "Intensity positivity valid"


def verify_form_factor_properties(
    q_values: np.ndarray, form_factor: np.ndarray, particle_type: str = "sphere"
) -> tuple[bool, str]:
    """
    Verify form factor mathematical properties

    Args:
        q_values: Scattering vector values
        form_factor: Form factor values
        particle_type: Type of particle ("sphere", "cylinder", etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(q_values) != len(form_factor):
        return False, "Q-values and form factor arrays must have same length"

    # Check positivity
    if not np.all(form_factor >= 0):
        return False, "Form factor must be non-negative"

    # Check normalization at q=0
    if len(q_values) > 0 and q_values[0] <= 1e-6:
        if abs(form_factor[0] - 1.0) > 1e-6:
            return False, f"Form factor at q=0 should be 1, got {form_factor[0]:.6f}"

    # Particle-specific checks
    if particle_type == "sphere":
        # For spheres, form factor should oscillate and decay
        if len(q_values) > 10:
            # Check that form factor generally decreases
            correlation = np.corrcoef(q_values[1:], form_factor[1:])[0, 1]
            if correlation > -0.3:  # Should be negatively correlated
                return False, "Sphere form factor should generally decrease with q"

    return True, "Form factor properties valid"


def verify_correlation_matrix_properties(
    correlation_matrix: np.ndarray, tolerance: float = 1e-10
) -> tuple[bool, str]:
    """
    Verify mathematical properties of correlation matrices

    Args:
        correlation_matrix: Correlation matrix to verify
        tolerance: Numerical tolerance for checks

    Returns:
        Tuple of (is_valid, error_message)
    """
    if correlation_matrix.ndim != 2:
        return False, "Correlation matrix must be 2D"

    n, m = correlation_matrix.shape
    if n != m:
        return False, "Correlation matrix must be square"

    # Check symmetry
    symmetry_error = np.max(np.abs(correlation_matrix - correlation_matrix.T))
    if symmetry_error > tolerance:
        return False, f"Correlation matrix not symmetric, error: {symmetry_error:.2e}"

    # Check positive semi-definiteness
    try:
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue < -tolerance:
            return (
                False,
                f"Correlation matrix not positive semi-definite, "
                f"min eigenvalue: {min_eigenvalue:.2e}",
            )
    except np.linalg.LinAlgError:
        return False, "Failed to compute eigenvalues of correlation matrix"

    return True, "Correlation matrix properties valid"


def verify_power_law_scaling(
    x_values: np.ndarray,
    y_values: np.ndarray,
    expected_exponent: float | None = None,
    tolerance: float = 0.2,
) -> tuple[bool, str]:
    """
    Verify power law scaling relationships: y ∝ x^α

    Args:
        x_values: Independent variable values (must be positive)
        y_values: Dependent variable values (must be positive)
        expected_exponent: Expected power law exponent (optional)
        tolerance: Tolerance for exponent comparison

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(x_values) != len(y_values):
        return False, "X and Y arrays must have same length"

    if len(x_values) < 5:
        return False, "Need at least 5 points for power law analysis"

    # Check positivity (required for log-log analysis)
    if not np.all(x_values > 0):
        return False, "X values must be positive for power law analysis"

    if not np.all(y_values > 0):
        return False, "Y values must be positive for power law analysis"

    # Perform log-log fit
    log_x = np.log10(x_values)
    log_y = np.log10(y_values)

    # Linear fit in log space
    try:
        coeffs = np.polyfit(log_x, log_y, 1)
        fitted_exponent = coeffs[0]

        # Calculate R-squared
        log_y_fit = coeffs[1] + fitted_exponent * log_x
        ss_res = np.sum((log_y - log_y_fit) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Check fit quality
        if r_squared < 0.8:
            return False, f"Poor power law fit, R² = {r_squared:.3f}"

        # Check expected exponent if provided
        if expected_exponent is not None:
            exponent_error = abs(fitted_exponent - expected_exponent)
            if exponent_error > tolerance:
                return (
                    False,
                    f"Power law exponent {fitted_exponent:.3f} != "
                    f"expected {expected_exponent:.3f}",
                )

        return (
            True,
            f"Power law scaling valid, exponent = {fitted_exponent:.3f}, R² = {r_squared:.3f}",
        )

    except np.linalg.LinAlgError:
        return False, "Failed to fit power law"


def verify_diffusion_constraints(
    tau_values: np.ndarray, q_values: np.ndarray, diffusion_type: str = "brownian"
) -> tuple[bool, str]:
    """
    Verify physical constraints for diffusion analysis

    Args:
        tau_values: Relaxation time values
        q_values: Scattering vector values
        diffusion_type: Type of diffusion ("brownian", "subdiffusion", etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(tau_values) != len(q_values):
        return False, "Tau and Q arrays must have same length"

    # Check positivity
    if not np.all(tau_values > 0):
        negative_count = np.sum(tau_values <= 0)
        return False, f"{negative_count} non-positive relaxation times"

    if not np.all(q_values > 0):
        return False, "Q values must be positive"

    # Diffusion-specific checks
    if diffusion_type == "brownian":
        # For Brownian motion: τ ∝ q^(-2)
        valid, message = verify_power_law_scaling(
            q_values, tau_values, expected_exponent=-2.0, tolerance=0.3
        )
        if not valid:
            return False, f"Brownian diffusion scaling invalid: {message}"

    elif diffusion_type == "subdiffusion":
        # For subdiffusion: τ ∝ q^(-α) where 0 < α < 2
        log_q = np.log10(q_values)
        log_tau = np.log10(tau_values)

        try:
            coeffs = np.polyfit(log_q, log_tau, 1)
            exponent = coeffs[0]

            if exponent >= 0:
                return (
                    False,
                    f"Subdiffusion should have negative exponent, got {exponent:.3f}",
                )

            if -exponent >= 2.0:
                return (
                    False,
                    f"Subdiffusion exponent should be < 2, got {-exponent:.3f}",
                )

        except np.linalg.LinAlgError:
            return False, "Failed to analyze subdiffusion scaling"

    return True, f"{diffusion_type.capitalize()} diffusion constraints valid"


def verify_statistical_moments(
    data: np.ndarray,
    expected_mean: float | None = None,
    expected_var: float | None = None,
    tolerance: float = 0.1,
) -> tuple[bool, str]:
    """
    Verify statistical moments of data distributions

    Args:
        data: Data array to analyze
        expected_mean: Expected mean value (optional)
        expected_var: Expected variance (optional)
        tolerance: Relative tolerance for comparisons

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(data) == 0:
        return False, "Empty data array"

    # Remove NaN and inf values
    clean_data = data[np.isfinite(data)]
    if len(clean_data) == 0:
        return False, "No finite data values"

    # Calculate moments
    data_mean = np.mean(clean_data)
    data_var = np.var(clean_data)

    # Check expected mean
    if expected_mean is not None:
        mean_error = abs(data_mean - expected_mean) / abs(expected_mean)
        if mean_error > tolerance:
            return False, f"Mean error {mean_error:.3f} > tolerance {tolerance:.3f}"

    # Check expected variance
    if expected_var is not None:
        var_error = abs(data_var - expected_var) / abs(expected_var)
        if var_error > tolerance:
            return False, f"Variance error {var_error:.3f} > tolerance {tolerance:.3f}"

    return True, f"Statistical moments valid: mean={data_mean:.4f}, var={data_var:.4f}"


def verify_conservation_laws(
    input_data: np.ndarray,
    output_data: np.ndarray,
    conservation_type: str = "intensity",
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """
    Verify conservation laws in data processing

    Args:
        input_data: Input data array
        output_data: Output data array after processing
        conservation_type: Type of conservation ("intensity", "area", "mass")
        tolerance: Relative tolerance for conservation check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if conservation_type == "intensity":
        input_total = np.sum(input_data)
        output_total = np.sum(output_data)

    elif conservation_type == "area":
        # Assume equal spacing for area calculation
        input_total = np.trapz(input_data)
        output_total = np.trapz(output_data)

    else:
        return False, f"Unknown conservation type: {conservation_type}"

    if abs(input_total) < 1e-15:
        # Special case: zero input
        if abs(output_total) > tolerance:
            return False, f"Zero input should give zero output, got {output_total:.2e}"
    else:
        rel_error = abs(output_total - input_total) / abs(input_total)
        if rel_error > tolerance:
            return (
                False,
                f"{conservation_type.capitalize()} not conserved, "
                f"relative error: {rel_error:.2e}",
            )

    return True, f"{conservation_type.capitalize()} conservation valid"


# Additional utility functions for property-based testing


def generate_valid_g2_data(
    tau: np.ndarray, decay_type: str = "single_exponential", **kwargs
) -> np.ndarray:
    """
    Generate valid G2 data satisfying mathematical properties

    Args:
        tau: Time delay values
        decay_type: Type of decay ("single_exponential", "double_exponential", "stretched")
        **kwargs: Parameters for the specific decay model

    Returns:
        Valid G2 data array
    """
    if decay_type == "single_exponential":
        beta = kwargs.get("beta", 0.8)
        gamma = kwargs.get("gamma", 1000.0)
        baseline = kwargs.get("baseline", 1.0)

        return baseline + beta * np.exp(-gamma * tau)

    if decay_type == "double_exponential":
        beta1 = kwargs.get("beta1", 0.5)
        gamma1 = kwargs.get("gamma1", 5000.0)
        beta2 = kwargs.get("beta2", 0.3)
        gamma2 = kwargs.get("gamma2", 500.0)
        baseline = kwargs.get("baseline", 1.0)

        return baseline + beta1 * np.exp(-gamma1 * tau) + beta2 * np.exp(-gamma2 * tau)

    if decay_type == "stretched":
        beta = kwargs.get("beta", 0.7)
        tau_char = kwargs.get("tau_char", 0.001)
        stretch_exp = kwargs.get("stretch_exp", 0.8)
        baseline = kwargs.get("baseline", 1.0)

        return baseline + beta * np.exp(-((tau / tau_char) ** stretch_exp))

    raise ValueError(f"Unknown decay type: {decay_type}")


def generate_valid_intensity_data(
    q_values: np.ndarray, scattering_type: str = "sphere", **kwargs
) -> np.ndarray:
    """
    Generate valid scattering intensity data

    Args:
        q_values: Scattering vector values
        scattering_type: Type of scattering ("sphere", "power_law", "guinier")
        **kwargs: Parameters for the specific model

    Returns:
        Valid intensity data array
    """
    if scattering_type == "sphere":
        radius = kwargs.get("radius", 25.0)  # Angstroms
        intensity_0 = kwargs.get("intensity_0", 1000.0)

        qR = q_values * radius
        qR = np.where(qR == 0, 1e-10, qR)

        form_factor = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR**3)
        return intensity_0 * form_factor**2

    if scattering_type == "power_law":
        amplitude = kwargs.get("amplitude", 100.0)
        exponent = kwargs.get("exponent", 2.5)
        background = kwargs.get("background", 1.0)

        return amplitude * q_values ** (-exponent) + background

    if scattering_type == "guinier":
        intensity_0 = kwargs.get("intensity_0", 1000.0)
        rg = kwargs.get("rg", 20.0)  # Radius of gyration

        return intensity_0 * np.exp(-(q_values**2) * rg**2 / 3)

    raise ValueError(f"Unknown scattering type: {scattering_type}")
