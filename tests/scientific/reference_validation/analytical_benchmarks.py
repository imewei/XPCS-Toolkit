"""
Analytical Benchmarks for XPCS Algorithm Validation

This module provides comprehensive analytical benchmarks against which XPCS
algorithms can be validated. These benchmarks include exact analytical solutions
for various physical systems and mathematical models.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import special


class AnalyticalBenchmarkSuite:
    """Suite of analytical benchmarks for XPCS algorithm validation"""

    def __init__(self):
        self.benchmarks = {}
        self._initialize_benchmarks()

    def _initialize_benchmarks(self):
        """Initialize all analytical benchmarks"""

        # G2 correlation function benchmarks
        self.benchmarks["g2_single_exponential"] = {
            "function": self.g2_single_exponential,
            "description": "Single exponential G2 decay",
            "parameters": {"baseline": 1.0, "beta": 0.8, "gamma": 1000.0},
            "domain": np.logspace(-6, 1, 100),
            "properties": ["normalization", "monotonicity", "asymptotic_behavior"],
        }

        self.benchmarks["g2_double_exponential"] = {
            "function": self.g2_double_exponential,
            "description": "Double exponential G2 decay (two-component system)",
            "parameters": {
                "baseline": 1.0,
                "beta1": 0.5,
                "gamma1": 5000.0,
                "beta2": 0.3,
                "gamma2": 500.0,
            },
            "domain": np.logspace(-6, 0, 100),
            "properties": ["normalization", "asymptotic_behavior"],
        }

        self.benchmarks["g2_stretched_exponential"] = {
            "function": self.g2_stretched_exponential,
            "description": "Stretched exponential (KWW) G2 decay",
            "parameters": {
                "baseline": 1.0,
                "amplitude": 0.8,
                "tau_char": 0.001,
                "beta": 0.7,
            },
            "domain": np.logspace(-6, 1, 100),
            "properties": ["normalization", "asymptotic_behavior"],
        }

        # SAXS scattering benchmarks
        self.benchmarks["sphere_form_factor"] = {
            "function": self.sphere_form_factor,
            "description": "Sphere form factor (Rayleigh scattering)",
            "parameters": {"radius": 25.0, "intensity_0": 1000.0},
            "domain": np.logspace(-3, 0, 200),
            "properties": ["positivity", "forward_scattering", "oscillations"],
        }

        self.benchmarks["cylinder_form_factor"] = {
            "function": self.cylinder_form_factor,
            "description": "Infinite cylinder form factor",
            "parameters": {"radius": 20.0, "intensity_0": 1000.0},
            "domain": np.logspace(-3, 0, 200),
            "properties": ["positivity", "forward_scattering", "asymptotic_decay"],
        }

        self.benchmarks["guinier_approximation"] = {
            "function": self.guinier_approximation,
            "description": "Guinier approximation for globular particles",
            "parameters": {"intensity_0": 1000.0, "rg": 20.0},
            "domain": np.logspace(-3, -1, 100),
            "properties": ["positivity", "exponential_decay", "rg_relation"],
        }

        self.benchmarks["power_law_scattering"] = {
            "function": self.power_law_scattering,
            "description": "Power law scattering (Porod region)",
            "parameters": {"amplitude": 100.0, "exponent": 2.5, "background": 1.0},
            "domain": np.logspace(-2, 0, 100),
            "properties": ["positivity", "power_law_scaling"],
        }

        # Diffusion analysis benchmarks
        self.benchmarks["brownian_diffusion"] = {
            "function": self.brownian_diffusion,
            "description": "Simple Brownian diffusion τ(q)",
            "parameters": {"diffusion_coefficient": 1e-12},
            "domain": np.logspace(-3, -1, 50),
            "properties": ["positivity", "q_squared_scaling"],
        }

        self.benchmarks["subdiffusion"] = {
            "function": self.subdiffusion,
            "description": "Subdiffusive motion (fractional Brownian)",
            "parameters": {"amplitude": 1e-4, "exponent": 1.6},
            "domain": np.logspace(-3, -1, 50),
            "properties": ["positivity", "power_law_scaling"],
        }

        self.benchmarks["cooperative_diffusion"] = {
            "function": self.cooperative_diffusion,
            "description": "Cooperative diffusion with hydrodynamic interactions",
            "parameters": {"D0": 2e-12, "xi": 100e-10},
            "domain": np.logspace(-3, -1, 50),
            "properties": ["positivity", "q_dependence"],
        }

    # G2 correlation function benchmarks

    @staticmethod
    def g2_single_exponential(
        tau: np.ndarray, baseline: float, beta: float, gamma: float
    ) -> np.ndarray:
        """
        Single exponential G2 correlation function

        G2(τ) = baseline + β * exp(-γτ)

        Args:
            tau: Time delay values
            baseline: Asymptotic baseline (typically 1.0)
            beta: Amplitude factor (coherence factor)
            gamma: Decay rate (1/τ_relax)

        Returns:
            G2 correlation values
        """
        return baseline + beta * np.exp(-gamma * tau)

    @staticmethod
    def g2_double_exponential(
        tau: np.ndarray,
        baseline: float,
        beta1: float,
        gamma1: float,
        beta2: float,
        gamma2: float,
    ) -> np.ndarray:
        """
        Double exponential G2 correlation function

        G2(τ) = baseline + β₁*exp(-γ₁τ) + β₂*exp(-γ₂τ)

        Args:
            tau: Time delay values
            baseline: Asymptotic baseline
            beta1, beta2: Amplitude factors for fast and slow components
            gamma1, gamma2: Decay rates for fast and slow components

        Returns:
            G2 correlation values
        """
        return baseline + beta1 * np.exp(-gamma1 * tau) + beta2 * np.exp(-gamma2 * tau)

    @staticmethod
    def g2_stretched_exponential(
        tau: np.ndarray, baseline: float, amplitude: float, tau_char: float, beta: float
    ) -> np.ndarray:
        """
        Stretched exponential (Kohlrausch-Williams-Watts) G2 correlation

        G2(τ) = baseline + A * exp(-(τ/τ₀)^β)

        Args:
            tau: Time delay values
            baseline: Asymptotic baseline
            amplitude: Amplitude factor
            tau_char: Characteristic time scale
            beta: Stretching exponent (0 < β ≤ 1)

        Returns:
            G2 correlation values
        """
        return baseline + amplitude * np.exp(-((tau / tau_char) ** beta))

    # SAXS scattering benchmarks

    @staticmethod
    def sphere_form_factor(
        q: np.ndarray, radius: float, intensity_0: float
    ) -> np.ndarray:
        """
        Sphere form factor (Rayleigh scattering)

        F(q) = 3[sin(qR) - qR*cos(qR)]/(qR)³
        I(q) = I₀ * |F(q)|²

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            radius: Sphere radius (Å)
            intensity_0: Forward scattering intensity

        Returns:
            Scattering intensity values
        """
        qR = q * radius

        # Handle q=0 case
        qR = np.where(qR == 0, 1e-10, qR)

        # Sphere form factor
        form_factor = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR**3)

        return intensity_0 * form_factor**2

    @staticmethod
    def cylinder_form_factor(
        q: np.ndarray, radius: float, intensity_0: float
    ) -> np.ndarray:
        """
        Infinite cylinder form factor (averaged over orientations)

        I(q) = I₀ * [2*J₁(qR)/(qR)]²

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            radius: Cylinder radius (Å)
            intensity_0: Forward scattering intensity

        Returns:
            Scattering intensity values
        """
        qR = q * radius

        # Handle q=0 case
        qR = np.where(qR == 0, 1e-10, qR)

        # Cylinder form factor using Bessel function
        form_factor = 2 * special.j1(qR) / qR

        return intensity_0 * form_factor**2

    @staticmethod
    def guinier_approximation(
        q: np.ndarray, intensity_0: float, rg: float
    ) -> np.ndarray:
        """
        Guinier approximation for globular particles

        I(q) = I₀ * exp(-q²*Rg²/3)

        Valid for qRg < 1.3

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            intensity_0: Forward scattering intensity
            rg: Radius of gyration (Å)

        Returns:
            Scattering intensity values
        """
        return intensity_0 * np.exp(-(q**2) * rg**2 / 3)

    @staticmethod
    def power_law_scattering(
        q: np.ndarray, amplitude: float, exponent: float, background: float
    ) -> np.ndarray:
        """
        Power law scattering (Porod region)

        I(q) = A * q^(-α) + B

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            amplitude: Power law amplitude
            exponent: Power law exponent (typically 2-4)
            background: Constant background

        Returns:
            Scattering intensity values
        """
        return amplitude * q ** (-exponent) + background

    # Diffusion analysis benchmarks

    @staticmethod
    def brownian_diffusion(q: np.ndarray, diffusion_coefficient: float) -> np.ndarray:
        """
        Brownian diffusion relaxation times

        τ(q) = 1/(D*q²)

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            diffusion_coefficient: Diffusion coefficient (m²/s)

        Returns:
            Relaxation time values (s)
        """
        # Convert q from Å⁻¹ to m⁻¹
        q_m = q * 1e10

        return 1.0 / (diffusion_coefficient * q_m**2)

    @staticmethod
    def subdiffusion(q: np.ndarray, amplitude: float, exponent: float) -> np.ndarray:
        """
        Subdiffusive motion (anomalous diffusion)

        τ(q) = A * q^(-α) where α < 2

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            amplitude: Amplitude factor
            exponent: Power law exponent (0 < α < 2)

        Returns:
            Relaxation time values (s)
        """
        return amplitude * q ** (-exponent)

    @staticmethod
    def cooperative_diffusion(q: np.ndarray, D0: float, xi: float) -> np.ndarray:
        """
        Cooperative diffusion with hydrodynamic interactions

        D(q) = D₀ * (1 + (qξ)²)
        τ(q) = 1/(D(q)*q²)

        Args:
            q: Scattering vector magnitude (Å⁻¹)
            D0: Bare diffusion coefficient (m²/s)
            xi: Correlation length (m)

        Returns:
            Relaxation time values (s)
        """
        # Convert units
        q_m = q * 1e10

        # Cooperative diffusion coefficient
        D_coop = D0 * (1 + (q_m * xi) ** 2)

        return 1.0 / (D_coop * q_m**2)

    # Utility methods

    def get_benchmark(self, benchmark_name: str) -> dict[str, Any]:
        """Get benchmark data by name"""
        if benchmark_name not in self.benchmarks:
            available = list(self.benchmarks.keys())
            raise ValueError(
                f"Benchmark '{benchmark_name}' not found. Available: {available}"
            )

        return self.benchmarks[benchmark_name]

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        custom_parameters: dict[str, Any] | None = None,
        custom_domain: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a benchmark with given or default parameters

        Args:
            benchmark_name: Name of the benchmark to evaluate
            custom_parameters: Custom parameters (overrides defaults)
            custom_domain: Custom domain values (overrides defaults)

        Returns:
            Tuple of (domain_values, function_values)
        """
        benchmark = self.get_benchmark(benchmark_name)

        # Use custom or default parameters
        parameters = benchmark["parameters"].copy()
        if custom_parameters:
            parameters.update(custom_parameters)

        # Use custom or default domain
        domain = custom_domain if custom_domain is not None else benchmark["domain"]

        # Evaluate function
        function_values = benchmark["function"](domain, **parameters)

        return domain, function_values

    def validate_benchmark_properties(
        self, benchmark_name: str, domain: np.ndarray, values: np.ndarray
    ) -> dict[str, bool]:
        """
        Validate that benchmark results satisfy expected properties

        Args:
            benchmark_name: Name of the benchmark
            domain: Domain values used for evaluation
            values: Function values to validate

        Returns:
            Dictionary of property validation results
        """
        benchmark = self.get_benchmark(benchmark_name)
        properties = benchmark.get("properties", [])

        validation_results = {}

        for prop in properties:
            if prop == "normalization":
                # G2 normalization: G2(τ=0) ≥ 1
                if len(domain) > 0:
                    zero_idx = np.argmin(np.abs(domain))
                    g2_zero = values[zero_idx]
                    validation_results[prop] = g2_zero >= 1.0 - 1e-10
                else:
                    validation_results[prop] = False

            elif prop == "monotonicity":
                # Monotonic decrease
                diff = np.diff(values)
                validation_results[prop] = np.all(
                    diff <= 1e-12
                )  # Allow numerical errors

            elif prop == "asymptotic_behavior":
                # G2(τ→∞) → baseline
                if len(values) > 10:
                    asymptotic_value = np.mean(values[-5:])
                    expected_baseline = 1.0  # Assumed baseline
                    validation_results[prop] = (
                        abs(asymptotic_value - expected_baseline) < 0.01
                    )
                else:
                    validation_results[prop] = False

            elif prop == "positivity":
                # All values should be non-negative
                validation_results[prop] = np.all(
                    values >= -1e-12
                )  # Allow tiny numerical errors

            elif prop == "forward_scattering":
                # Forward scattering should be maximum
                if len(values) > 0:
                    max_intensity = np.max(values)
                    forward_intensity = values[0]  # Assume q=0 is first point
                    validation_results[prop] = forward_intensity >= 0.9 * max_intensity
                else:
                    validation_results[prop] = False

            elif prop == "q_squared_scaling":
                # Brownian diffusion: τ ∝ q⁻²
                if len(domain) > 5:
                    log_q = np.log10(domain)
                    log_tau = np.log10(values)

                    # Linear fit in log-log space
                    fit_coeff = np.polyfit(log_q, log_tau, 1)
                    measured_exponent = fit_coeff[0]

                    validation_results[prop] = abs(measured_exponent + 2.0) < 0.1
                else:
                    validation_results[prop] = False

            elif prop == "power_law_scaling":
                # General power law scaling
                if len(domain) > 5:
                    log_x = np.log10(domain)
                    log_y = np.log10(values)

                    # Check for reasonable power law fit
                    correlation = np.corrcoef(log_x, log_y)[0, 1]
                    validation_results[prop] = abs(correlation) > 0.95
                else:
                    validation_results[prop] = False

            else:
                # Unknown property
                validation_results[prop] = None

        return validation_results

    def run_comprehensive_benchmark_validation(self) -> dict[str, Any]:
        """
        Run validation on all benchmarks with their default parameters

        Returns:
            Comprehensive validation report
        """
        validation_report = {
            "timestamp": str(np.datetime64("now")),
            "n_benchmarks": len(self.benchmarks),
            "benchmark_results": {},
            "summary": {},
        }

        all_passed = True

        for benchmark_name in self.benchmarks:
            try:
                # Evaluate benchmark
                domain, values = self.evaluate_benchmark(benchmark_name)

                # Validate properties
                property_results = self.validate_benchmark_properties(
                    benchmark_name, domain, values
                )

                # Check for any failures - convert numpy booleans to Python booleans
                benchmark_passed = all(
                    bool(result) is True
                    for result in property_results.values()
                    if result is not None
                )

                if not benchmark_passed:
                    all_passed = False

                validation_report["benchmark_results"][benchmark_name] = {
                    "passed": benchmark_passed,
                    "property_validations": property_results,
                    "n_points": len(values),
                    "value_range": (float(np.min(values)), float(np.max(values))),
                }

            except Exception as e:
                validation_report["benchmark_results"][benchmark_name] = {
                    "passed": False,
                    "error": str(e),
                }
                all_passed = False

        # Summary
        passed_benchmarks = sum(
            1
            for result in validation_report["benchmark_results"].values()
            if result.get("passed", False)
        )

        validation_report["summary"] = {
            "all_benchmarks_passed": all_passed,
            "passed_benchmarks": passed_benchmarks,
            "total_benchmarks": len(self.benchmarks),
            "success_rate": passed_benchmarks / len(self.benchmarks),
        }

        return validation_report


def create_custom_analytical_benchmark(
    function: Callable,
    parameters: dict[str, Any],
    domain: np.ndarray,
    properties: list[str] | None = None,
    description: str = "Custom benchmark",
) -> dict[str, Any]:
    """
    Create a custom analytical benchmark

    Args:
        function: Analytical function to use as benchmark
        parameters: Default parameters for the function
        domain: Default domain for evaluation
        properties: List of properties to validate
        description: Description of the benchmark

    Returns:
        Benchmark dictionary
    """
    if properties is None:
        properties = ["positivity"]  # Default property

    return {
        "function": function,
        "description": description,
        "parameters": parameters,
        "domain": domain,
        "properties": properties,
    }


if __name__ == "__main__":
    # Example usage and validation
    benchmark_suite = AnalyticalBenchmarkSuite()

    print("Available benchmarks:")
    for name in benchmark_suite.benchmarks:
        print(f"  - {name}: {benchmark_suite.benchmarks[name]['description']}")

    print("\nRunning comprehensive benchmark validation...")
    validation_report = benchmark_suite.run_comprehensive_benchmark_validation()

    print("\nValidation Summary:")
    print(f"  Total benchmarks: {validation_report['summary']['total_benchmarks']}")
    print(f"  Passed benchmarks: {validation_report['summary']['passed_benchmarks']}")
    print(f"  Success rate: {validation_report['summary']['success_rate']:.1%}")

    if validation_report["summary"]["all_benchmarks_passed"]:
        print("  ✓ All analytical benchmarks validated successfully!")
    else:
        print("  ✗ Some benchmarks failed validation:")
        for name, result in validation_report["benchmark_results"].items():
            if not result.get("passed", False):
                print(
                    f"    - {name}: {result.get('error', 'Property validation failed')}"
                )
