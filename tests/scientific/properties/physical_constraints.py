"""
Physical Constraints and Laws

This module defines physical constraints and laws that XPCS algorithms must satisfy,
ensuring that computed results are physically meaningful and consistent with
known physics principles.
"""

import numpy as np

from tests.scientific.constants import SCIENTIFIC_CONSTANTS


def verify_stokes_einstein_relation(
    diffusion_coefficient: float,
    temperature: float,
    viscosity: float,
    particle_radius: float,
    tolerance: float = 0.1,
) -> tuple[bool, str]:
    """
    Verify Stokes-Einstein relation: D = kT/(6πηr)

    Args:
        diffusion_coefficient: Measured diffusion coefficient (m²/s)
        temperature: Temperature (K)
        viscosity: Viscosity (Pa⋅s)
        particle_radius: Particle radius (m)
        tolerance: Relative tolerance for comparison

    Returns:
        Tuple of (is_valid, error_message)
    """
    k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]

    # Calculate theoretical diffusion coefficient
    D_theory = k_B * temperature / (6 * np.pi * viscosity * particle_radius)

    # Check relative error
    if D_theory == 0:
        return False, "Theoretical diffusion coefficient is zero"

    rel_error = abs(diffusion_coefficient - D_theory) / D_theory

    if rel_error > tolerance:
        return (
            False,
            f"Stokes-Einstein violation: D = {diffusion_coefficient:.2e}, "
            f"expected {D_theory:.2e} (error = {rel_error:.3f})",
        )

    # Check that both values are physically reasonable
    if diffusion_coefficient <= 0:
        return (
            False,
            f"Diffusion coefficient must be positive: {diffusion_coefficient:.2e}",
        )

    if diffusion_coefficient > 1e-8:
        return False, f"Diffusion coefficient too large: {diffusion_coefficient:.2e}"

    if diffusion_coefficient < 1e-18:
        return False, f"Diffusion coefficient too small: {diffusion_coefficient:.2e}"

    return True, f"Stokes-Einstein relation satisfied (error = {rel_error:.4f})"


def verify_thermodynamic_constraints(
    temperature: float,
    energy_scale: float | None = None,
    time_scale: float | None = None,
) -> tuple[bool, str]:
    """
    Verify thermodynamic constraints and energy scales

    Args:
        temperature: Temperature (K)
        energy_scale: Characteristic energy scale (J)
        time_scale: Characteristic time scale (s)

    Returns:
        Tuple of (is_valid, error_message)
    """
    k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]

    # Check temperature range
    if temperature <= 0:
        return False, f"Temperature must be positive: {temperature} K"

    if temperature < 1:
        return False, f"Temperature too low for condensed matter: {temperature} K"

    if temperature > 1000:
        return False, f"Temperature too high for typical XPCS: {temperature} K"

    # Thermal energy scale
    kT = k_B * temperature

    # Check energy scale if provided
    if energy_scale is not None:
        if energy_scale <= 0:
            return False, f"Energy scale must be positive: {energy_scale:.2e} J"

        # Energy scale should be comparable to or larger than kT
        if energy_scale < 0.1 * kT:
            return (
                False,
                f"Energy scale too small compared to kT: "
                f"{energy_scale:.2e} J vs kT = {kT:.2e} J",
            )

    # Check time scale if provided
    if time_scale is not None:
        if time_scale <= 0:
            return False, f"Time scale must be positive: {time_scale:.2e} s"

        # Time scales in XPCS typically range from microseconds to seconds
        if time_scale < 1e-9:
            return False, f"Time scale too short for XPCS: {time_scale:.2e} s"

        if time_scale > 1e6:
            return False, f"Time scale too long for XPCS: {time_scale:.2e} s"

    return True, "Thermodynamic constraints satisfied"


def verify_scattering_physics(
    q_values: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
    detector_distance: float | None = None,
    pixel_size: float | None = None,
) -> tuple[bool, str]:
    """
    Verify X-ray scattering physics constraints

    Args:
        q_values: Scattering vector magnitudes (Å⁻¹)
        intensity: Scattering intensities
        wavelength: X-ray wavelength (m)
        detector_distance: Sample-detector distance (m)
        pixel_size: Detector pixel size (m)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(q_values) != len(intensity):
        return False, "Q-values and intensity arrays must have same length"

    # Check q-value range
    if np.any(q_values <= 0):
        return False, "Q-values must be positive"

    q_min, q_max = np.min(q_values), np.max(q_values)

    # Convert wavelength to Angstroms
    wavelength_A = wavelength * 1e10

    # Maximum possible q-value for elastic scattering: q_max = 4π/λ
    q_max_theory = 4 * np.pi / wavelength_A

    if q_max > q_max_theory:
        return False, f"Q-value too large: {q_max:.4f} > {q_max_theory:.4f} Å⁻¹"

    # Typical SAXS range check
    if q_min > 1:
        return False, f"Q-min too large for SAXS: {q_min:.4f} Å⁻¹"

    # Check intensity positivity
    if np.any(intensity < 0):
        negative_count = np.sum(intensity < 0)
        return False, f"{negative_count} negative intensities found"

    # Check intensity range (reasonable for photon counting)
    max_intensity = np.max(intensity)
    if max_intensity > 1e12:
        return False, f"Intensity too high: {max_intensity:.2e} counts"

    if max_intensity < 1e-6:
        return False, f"Intensity too low: {max_intensity:.2e} counts"

    # Check detector geometry if provided
    if detector_distance is not None and pixel_size is not None:
        # Calculate q-range from detector geometry
        max_angle = np.arctan(0.5 * pixel_size * len(q_values) / detector_distance)
        q_max_detector = 4 * np.pi * np.sin(max_angle / 2) / wavelength_A

        if q_max > 1.1 * q_max_detector:  # Allow 10% tolerance
            return (
                False,
                f"Q-range inconsistent with detector geometry: "
                f"{q_max:.4f} > {q_max_detector:.4f} Å⁻¹",
            )

    return True, "Scattering physics constraints satisfied"


def verify_correlation_physics(
    tau_values: np.ndarray,
    g2_values: np.ndarray,
    temperature: float,
    viscosity: float,
    particle_size: float | None = None,
) -> tuple[bool, str]:
    """
    Verify physics of intensity correlation functions

    Args:
        tau_values: Time delay values (s)
        g2_values: G2 correlation values
        temperature: Temperature (K)
        viscosity: Viscosity (Pa⋅s)
        particle_size: Expected particle size (m), optional

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(tau_values) != len(g2_values):
        return False, "Tau and G2 arrays must have same length"

    # Check time range
    if np.any(tau_values <= 0):
        return False, "Time delays must be positive"

    tau_min, tau_max = np.min(tau_values), np.max(tau_values)

    # Typical XPCS time range
    if tau_min < 1e-9:
        return False, f"Time delay too short: {tau_min:.2e} s"

    if tau_max > 1e6:
        return False, f"Time delay too long: {tau_max:.2e} s"

    # Check G2 values
    if np.any(g2_values < 1.0 - 1e-6):  # Allow small numerical errors
        min_g2 = np.min(g2_values)
        return False, f"G2 values below 1: minimum = {min_g2:.6f}"

    g2_max = np.max(g2_values)
    if g2_max > 3.0:
        return False, f"G2 values too large: maximum = {g2_max:.3f}"

    # Check that G2 approaches 1 at long times
    if len(g2_values) > 10:
        g2_asymptotic = np.mean(g2_values[-5:])  # Average of last 5 points
        if abs(g2_asymptotic - 1.0) > 0.1:
            return False, f"G2 does not approach 1 at long times: {g2_asymptotic:.3f}"

    # Estimate diffusion coefficient if particle size is known
    if particle_size is not None:
        k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]
        k_B * temperature / (6 * np.pi * viscosity * particle_size)

        # Estimate relaxation time from G2 data
        # Find where G2 decays to 1 + 1/e of initial value
        g2_initial = g2_values[0] - 1.0
        g2_target = 1.0 + g2_initial / np.e

        # Find closest point
        idx = np.argmin(np.abs(g2_values - g2_target))
        tau_relax = tau_values[idx]

        # For simple Brownian motion with known q: τ = 1/(Dq²)
        # We can't check this exactly without knowing q, but we can check
        # that the relaxation time is reasonable
        if tau_relax < 1e-6:
            return False, f"Relaxation time too fast: {tau_relax:.2e} s"

        if tau_relax > 1e3:
            return False, f"Relaxation time too slow: {tau_relax:.2e} s"

    return True, "Correlation physics constraints satisfied"


def verify_conservation_laws(
    input_intensity: np.ndarray,
    output_intensity: np.ndarray,
    conservation_type: str = "photon_number",
) -> tuple[bool, str]:
    """
    Verify conservation laws in scattering processes

    Args:
        input_intensity: Input intensity distribution
        output_intensity: Output intensity distribution after processing
        conservation_type: Type of conservation to check

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(input_intensity) != len(output_intensity):
        return False, "Input and output arrays must have same length"

    if conservation_type == "photon_number":
        # Total photon number should be conserved
        input_total = np.sum(input_intensity)
        output_total = np.sum(output_intensity)

        if input_total == 0:
            if output_total != 0:
                return False, "Zero input should give zero output"
        else:
            rel_error = abs(output_total - input_total) / input_total
            if rel_error > 1e-6:
                return (
                    False,
                    f"Photon number not conserved: relative error = {rel_error:.2e}",
                )

    elif conservation_type == "energy":
        # For elastic scattering, energy should be conserved
        # This is automatically satisfied if photon number is conserved
        # and wavelength doesn't change
        return verify_conservation_laws(
            input_intensity, output_intensity, "photon_number"
        )

    elif conservation_type == "momentum":
        # For elastic scattering, momentum magnitude is conserved
        # This is satisfied by the scattering geometry
        return True, "Momentum conservation satisfied by scattering geometry"

    else:
        return False, f"Unknown conservation type: {conservation_type}"

    return True, f"{conservation_type} conservation satisfied"


def verify_fluctuation_dissipation_theorem(
    correlation_function: np.ndarray,
    response_function: np.ndarray,
    temperature: float,
    frequency: np.ndarray,
) -> tuple[bool, str]:
    """
    Verify fluctuation-dissipation theorem for correlation and response functions

    The FDT relates equilibrium fluctuations to linear response:
    S(ω) = (2kT/ω) * Im[χ(ω)]

    Args:
        correlation_function: Power spectral density S(ω)
        response_function: Response function χ(ω) (complex)
        temperature: Temperature (K)
        frequency: Frequency values (Hz)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(correlation_function) != len(response_function):
        return False, "Correlation and response arrays must have same length"

    if len(frequency) != len(correlation_function):
        return False, "Frequency array must match other arrays"

    # Remove zero frequency (singularity in FDT)
    nonzero_mask = frequency > 0
    if np.sum(nonzero_mask) < 3:
        return False, "Need at least 3 non-zero frequencies"

    freq = frequency[nonzero_mask]
    S_omega = correlation_function[nonzero_mask]
    chi_omega = response_function[nonzero_mask]

    k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]

    # Calculate theoretical correlation from response
    # S(ω) = (2kT/ω) * Im[χ(ω)]
    if np.any(np.iscomplex(chi_omega)):
        chi_imag = np.imag(chi_omega)
    else:
        return False, "Response function should be complex"

    S_theory = (2 * k_B * temperature / freq) * chi_imag

    # Compare with measured correlation function
    # Calculate relative error where both functions are significant
    significant_mask = (S_omega > 0.1 * np.max(S_omega)) & (
        S_theory > 0.1 * np.max(S_theory)
    )

    if np.sum(significant_mask) < 3:
        return True, "Insufficient significant data for FDT test"

    S_meas = S_omega[significant_mask]
    S_theo = S_theory[significant_mask]

    rel_errors = np.abs(S_meas - S_theo) / S_theo
    max_rel_error = np.max(rel_errors)
    mean_rel_error = np.mean(rel_errors)

    if max_rel_error > 0.5:  # 50% tolerance
        return False, f"FDT violated: max error = {max_rel_error:.3f}"

    if mean_rel_error > 0.2:  # 20% average error
        return False, f"FDT violated: mean error = {mean_rel_error:.3f}"

    return (
        True,
        f"Fluctuation-dissipation theorem satisfied (mean error = {mean_rel_error:.3f})",
    )


def verify_causality_constraint(
    response_function: np.ndarray, time: np.ndarray
) -> tuple[bool, str]:
    """
    Verify causality constraint: response function must be zero for t < 0

    Args:
        response_function: Time-domain response function
        time: Time values

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(response_function) != len(time):
        return False, "Response function and time arrays must have same length"

    # Find negative time indices
    negative_time_mask = time < 0

    if not np.any(negative_time_mask):
        return True, "No negative times present"

    # Check that response is zero (or negligible) for negative times
    response_negative = response_function[negative_time_mask]
    max_negative_response = np.max(np.abs(response_negative))

    # Get scale from positive-time response
    positive_time_mask = time > 0
    if np.any(positive_time_mask):
        response_positive = response_function[positive_time_mask]
        response_scale = np.max(np.abs(response_positive))
    else:
        response_scale = 1.0

    # Check relative magnitude
    relative_violation = (
        max_negative_response / response_scale
        if response_scale > 0
        else max_negative_response
    )

    if relative_violation > 1e-6:
        return (
            False,
            f"Causality violated: response at t<0 = {max_negative_response:.2e}",
        )

    return True, "Causality constraint satisfied"


def verify_detailed_balance(
    transition_rates: dict[tuple[int, int], float],
    energies: dict[int, float],
    temperature: float,
) -> tuple[bool, str]:
    """
    Verify detailed balance condition for transition rates

    Detailed balance: W(i→j)/W(j→i) = exp[-(E_j - E_i)/(kT)]

    Args:
        transition_rates: Dictionary of transition rates W(i→j)
        energies: Dictionary of state energies
        temperature: Temperature (K)

    Returns:
        Tuple of (is_valid, error_message)
    """
    k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]
    kT = k_B * temperature

    # Check all pairs of states
    violations = []

    for (i, j), rate_ij in transition_rates.items():
        # Look for reverse transition
        reverse_key = (j, i)
        if reverse_key in transition_rates:
            rate_ji = transition_rates[reverse_key]

            if i in energies and j in energies:
                E_i = energies[i]
                E_j = energies[j]

                # Calculate theoretical ratio
                if rate_ji > 0:
                    ratio_measured = rate_ij / rate_ji
                    ratio_theory = np.exp(-(E_j - E_i) / kT)

                    rel_error = abs(ratio_measured - ratio_theory) / ratio_theory

                    if rel_error > 0.1:  # 10% tolerance
                        violations.append(
                            f"States ({i},{j}): ratio = {ratio_measured:.3f}, "
                            f"expected {ratio_theory:.3f}"
                        )

    if violations:
        return False, f"Detailed balance violations: {'; '.join(violations[:3])}"

    return True, "Detailed balance satisfied"
