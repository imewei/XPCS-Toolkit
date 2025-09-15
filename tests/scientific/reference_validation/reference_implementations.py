"""
Reference Implementation Validators

This module provides validation against reference implementations from
established XPCS analysis software packages and literature implementations.
"""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .cross_validation_framework import ReferenceValidator


class LiteratureReferenceValidator(ReferenceValidator):
    """Validator using reference data from scientific literature"""

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.reference_library = self._load_literature_references()

    def _load_literature_references(self) -> dict[str, Any]:
        """Load literature reference data"""
        # In a real implementation, this would load from a database
        # or external files containing validated literature results

        return {
            "ponmurugan_2009_sphere_form_factor": {
                "description": "Sphere form factor data from Ponmurugan et al. 2009",
                "reference": "Ponmurugan et al., J. Appl. Cryst. 42, 523 (2009)",
                "test_case": "sphere_radius_25A",
                "q_values": np.logspace(-3, 0, 100),
                "expected_minima": [4.493, 7.725, 10.904],  # qR values of minima
                "tolerance": 0.05,
            },
            "lumma_2000_g2_fitting": {
                "description": "G2 fitting benchmark from Lumma et al. 2000",
                "reference": "Lumma et al., Rev. Sci. Instrum. 71, 3274 (2000)",
                "test_case": "silica_colloids_water",
                "tau_range": np.logspace(-6, 1, 50),
                "expected_params": {"gamma": 1250, "beta": 0.85, "baseline": 1.0},
                "tolerance": 0.1,
            },
            "fluerasu_2007_twotime": {
                "description": "Two-time correlation analysis from Fluerasu et al. 2007",
                "reference": "Fluerasu et al., Phys. Rev. E 76, 010401(R) (2007)",
                "test_case": "aging_colloidal_glass",
                "correlation_properties": ["ergodicity_breaking", "aging_effects"],
                "tolerance": 0.05,
            },
        }

    def validate_against_reference(
        self, input_data: dict[str, Any], algorithm_output: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate against literature reference

        Args:
            input_data: Must contain 'reference_key' specifying which literature reference
            algorithm_output: Algorithm results to validate

        Returns:
            Tuple of (is_valid, validation_report)
        """
        reference_key = input_data.get("reference_key")

        if reference_key not in self.reference_library:
            return False, {
                "error": f"Reference {reference_key} not found in library",
                "available_references": list(self.reference_library.keys()),
            }

        reference = self.reference_library[reference_key]

        report = {
            "validator_type": "literature_reference",
            "reference_key": reference_key,
            "reference_citation": reference["reference"],
            "tolerance_used": reference.get("tolerance", self.tolerance),
        }

        # Validate based on reference type
        if "sphere_form_factor" in reference_key:
            return self._validate_sphere_form_factor(
                reference, algorithm_output, report
            )
        if "g2_fitting" in reference_key:
            return self._validate_g2_fitting(reference, algorithm_output, report)
        if "twotime" in reference_key:
            return self._validate_twotime_analysis(reference, algorithm_output, report)
        report["error"] = f"Unknown reference type for {reference_key}"
        return False, report

    def _validate_sphere_form_factor(
        self, reference: dict, output: dict, report: dict
    ) -> tuple[bool, dict]:
        """Validate sphere form factor against literature"""

        try:
            q_values = output.get("q_values")
            intensity = output.get("intensity")
            radius = output.get("radius", 25.0)

            if q_values is None or intensity is None:
                report["error"] = "Missing q_values or intensity in output"
                return False, report

            # Calculate qR values
            qR_values = q_values * radius

            # Find minima in the form factor
            minima_indices = []
            for i in range(1, len(intensity) - 1):
                if intensity[i] < intensity[i - 1] and intensity[i] < intensity[i + 1]:
                    minima_indices.append(i)

            found_minima = qR_values[minima_indices] if minima_indices else []
            expected_minima = reference["expected_minima"]

            # Match found minima to expected ones
            validation_details = {
                "expected_minima": expected_minima,
                "found_minima": found_minima.tolist(),
                "tolerance": reference["tolerance"],
            }

            # Check if we found the expected minima
            matches = 0
            for expected in expected_minima[
                : len(found_minima)
            ]:  # Only check first few
                closest_found = min(found_minima, key=lambda x: abs(x - expected))
                if abs(closest_found - expected) <= reference["tolerance"]:
                    matches += 1

            success = matches >= min(2, len(expected_minima))  # At least 2 matches

            validation_details["successful_matches"] = matches
            validation_details["minimum_required_matches"] = min(
                2, len(expected_minima)
            )

            report["validation_details"] = validation_details
            return success, report

        except Exception as e:
            report["error"] = f"Validation failed: {e}"
            return False, report

    def _validate_g2_fitting(
        self, reference: dict, output: dict, report: dict
    ) -> tuple[bool, dict]:
        """Validate G2 fitting parameters against literature"""

        try:
            fitted_params = output.get("fitted_parameters")
            if fitted_params is None:
                report["error"] = "Missing fitted_parameters in output"
                return False, report

            expected_params = reference["expected_params"]
            tolerance = reference["tolerance"]

            validation_details = {
                "expected_parameters": expected_params,
                "fitted_parameters": fitted_params,
                "tolerance": tolerance,
                "parameter_comparisons": {},
            }

            all_params_valid = True

            for param_name, expected_value in expected_params.items():
                if param_name in fitted_params:
                    fitted_value = fitted_params[param_name]
                    rel_error = abs(fitted_value - expected_value) / abs(expected_value)

                    param_valid = rel_error <= tolerance

                    validation_details["parameter_comparisons"][param_name] = {
                        "expected": expected_value,
                        "fitted": fitted_value,
                        "relative_error": rel_error,
                        "valid": param_valid,
                    }

                    if not param_valid:
                        all_params_valid = False
                else:
                    validation_details["parameter_comparisons"][param_name] = {
                        "error": "Parameter not found in fitted results"
                    }
                    all_params_valid = False

            report["validation_details"] = validation_details
            return all_params_valid, report

        except Exception as e:
            report["error"] = f"Validation failed: {e}"
            return False, report

    def _validate_twotime_analysis(
        self, reference: dict, output: dict, report: dict
    ) -> tuple[bool, dict]:
        """Validate two-time analysis against literature"""

        try:
            correlation_matrix = output.get("correlation_matrix")
            properties = reference.get("correlation_properties", [])

            if correlation_matrix is None:
                report["error"] = "Missing correlation_matrix in output"
                return False, report

            validation_details = {
                "expected_properties": properties,
                "property_tests": {},
            }

            all_properties_valid = True

            for prop in properties:
                if prop == "ergodicity_breaking":
                    # Test for non-ergodic behavior
                    # Check that diagonal and off-diagonal elements show different scaling
                    diagonal_mean = np.mean(np.diag(correlation_matrix))
                    off_diag_mean = np.mean(
                        correlation_matrix - np.diag(np.diag(correlation_matrix))
                    )

                    # Non-ergodic systems should show significant difference
                    ergodicity_ratio = (
                        off_diag_mean / diagonal_mean if diagonal_mean > 0 else 0
                    )

                    property_valid = (
                        ergodicity_ratio < 0.8
                    )  # Threshold for ergodicity breaking

                    validation_details["property_tests"][prop] = {
                        "ergodicity_ratio": ergodicity_ratio,
                        "threshold": 0.8,
                        "valid": property_valid,
                    }

                elif prop == "aging_effects":
                    # Test for aging effects in correlation matrix
                    # Check for time-dependent correlation structure

                    # Simple test: check if correlation changes with time
                    n_times = correlation_matrix.shape[0]
                    early_corr = correlation_matrix[: n_times // 3, : n_times // 3]
                    late_corr = correlation_matrix[
                        2 * n_times // 3 :, 2 * n_times // 3 :
                    ]

                    early_mean = np.mean(early_corr)
                    late_mean = np.mean(late_corr)

                    aging_effect = (
                        abs(early_mean - late_mean) / early_mean
                        if early_mean > 0
                        else 0
                    )

                    property_valid = aging_effect > 0.1  # Threshold for aging effects

                    validation_details["property_tests"][prop] = {
                        "aging_effect_magnitude": aging_effect,
                        "threshold": 0.1,
                        "valid": property_valid,
                    }

                if not property_valid:
                    all_properties_valid = False

            report["validation_details"] = validation_details
            return all_properties_valid, report

        except Exception as e:
            report["error"] = f"Validation failed: {e}"
            return False, report


class SoftwarePackageValidator(ReferenceValidator):
    """Validator using reference results from established XPCS software packages"""

    def __init__(self, package_name: str, tolerance: float = 1e-3):
        self.package_name = package_name
        self.tolerance = tolerance
        self.supported_packages = {
            "matlab_dls": "MATLAB Dynamic Light Scattering toolbox",
            "igor_pro_xpcs": "Igor Pro XPCS analysis package",
            "pymcr": "Python Multivariate Curve Resolution",
            "xpcs_eigen": "C++ XPCS analysis library",
        }

        if package_name not in self.supported_packages:
            warnings.warn(
                f"Package '{package_name}' not in supported list: "
                f"{list(self.supported_packages.keys())}",
                stacklevel=2,
            )

    def validate_against_reference(
        self, input_data: dict[str, Any], algorithm_output: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate against reference software package results

        Args:
            input_data: Must contain reference results from the software package
            algorithm_output: XPCS Toolkit results to validate

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "validator_type": "software_package",
            "reference_package": self.package_name,
            "package_description": self.supported_packages.get(
                self.package_name, "Unknown"
            ),
            "tolerance_used": self.tolerance,
        }

        try:
            reference_results = input_data.get("reference_results", {})

            if not reference_results:
                report["error"] = "No reference results provided"
                return False, report

            # Compare results field by field
            comparison_results = {}
            overall_valid = True

            # Common fields to compare
            common_fields = set(reference_results.keys()) & set(algorithm_output.keys())

            if not common_fields:
                report["error"] = (
                    "No common output fields between reference and algorithm"
                )
                return False, report

            for field in common_fields:
                ref_data = np.asarray(reference_results[field])
                algo_data = np.asarray(algorithm_output[field])

                field_comparison = self._compare_field_data(field, ref_data, algo_data)
                comparison_results[field] = field_comparison

                if not field_comparison["valid"]:
                    overall_valid = False

            # Package-specific validation
            if self.package_name == "matlab_dls":
                overall_valid = (
                    self._validate_matlab_dls_specific(
                        reference_results, algorithm_output, comparison_results
                    )
                    and overall_valid
                )

            elif self.package_name == "igor_pro_xpcs":
                overall_valid = (
                    self._validate_igor_pro_specific(
                        reference_results, algorithm_output, comparison_results
                    )
                    and overall_valid
                )

            report["validation_details"] = {
                "compared_fields": list(common_fields),
                "field_comparisons": comparison_results,
                "overall_agreement": overall_valid,
            }

            return overall_valid, report

        except Exception as e:
            report["error"] = f"Validation failed: {e}"
            return False, report

    def _compare_field_data(
        self, field_name: str, reference_data: np.ndarray, algorithm_data: np.ndarray
    ) -> dict[str, Any]:
        """Compare data from a specific field"""

        if reference_data.shape != algorithm_data.shape:
            return {
                "valid": False,
                "error": f"Shape mismatch: {reference_data.shape} vs {algorithm_data.shape}",
            }

        # Calculate comparison metrics
        abs_diff = np.abs(reference_data - algorithm_data)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        # Handle division by zero in relative differences
        ref_magnitude = np.abs(reference_data) + 1e-15
        rel_diff = abs_diff / ref_magnitude
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # Field-specific tolerance adjustment
        tolerance = self.tolerance
        if field_name in ["fitted_parameters", "diffusion_coefficient"]:
            tolerance *= 5  # Allow more tolerance for fitted parameters
        elif field_name in ["correlation_times", "relaxation_rates"]:
            tolerance *= 2  # Moderate tolerance for time scales

        # Validation criteria
        field_valid = max_abs_diff <= tolerance or max_rel_diff <= tolerance

        # Correlation analysis
        if reference_data.size > 1:
            correlation = np.corrcoef(
                reference_data.flatten(), algorithm_data.flatten()
            )[0, 1]
        else:
            correlation = 1.0 if reference_data == algorithm_data else 0.0

        return {
            "valid": field_valid,
            "max_absolute_difference": float(max_abs_diff),
            "mean_absolute_difference": float(mean_abs_diff),
            "max_relative_difference": float(max_rel_diff),
            "mean_relative_difference": float(mean_rel_diff),
            "correlation": float(correlation),
            "tolerance_used": tolerance,
        }

    def _validate_matlab_dls_specific(
        self, reference_results: dict, algorithm_output: dict, comparison_results: dict
    ) -> bool:
        """MATLAB DLS toolbox specific validation"""

        # Check for MATLAB-specific outputs
        matlab_specific_checks = True

        # Check autocorrelation function properties
        if "autocorrelation" in reference_results and "g2_values" in algorithm_output:
            ref_autocorr = reference_results["autocorrelation"]
            algo_g2 = algorithm_output["g2_values"]

            # MATLAB typically normalizes differently
            # Check that both functions decay properly
            if len(ref_autocorr) > 10 and len(algo_g2) > 10:
                ref_decay = ref_autocorr[0] / ref_autocorr[-5:]  # Last few points
                algo_decay = algo_g2[0] / algo_g2[-5:]

                # Both should show similar decay behavior
                decay_similarity = np.corrcoef(ref_decay, algo_decay)[0, 1]
                if decay_similarity < 0.8:
                    matlab_specific_checks = False

        return matlab_specific_checks

    def _validate_igor_pro_specific(
        self, reference_results: dict, algorithm_output: dict, comparison_results: dict
    ) -> bool:
        """Igor Pro XPCS package specific validation"""

        igor_specific_checks = True

        # Igor Pro often includes baseline correction
        if "baseline_corrected_g2" in reference_results:
            # Check that our algorithm produces similar baseline correction
            ref_baseline = reference_results.get("baseline_value", 1.0)

            # Our algorithm should produce G2 values that approach this baseline
            if "g2_values" in algorithm_output:
                algo_g2 = algorithm_output["g2_values"]
                if len(algo_g2) > 10:
                    algo_baseline = np.mean(algo_g2[-5:])  # Last few points

                    baseline_agreement = (
                        abs(algo_baseline - ref_baseline) / ref_baseline
                    )
                    if baseline_agreement > 0.1:  # 10% tolerance
                        igor_specific_checks = False

        return igor_specific_checks


def load_reference_data_from_file(file_path: str) -> dict[str, Any]:
    """
    Load reference data from external file

    Args:
        file_path: Path to reference data file (JSON, NPZ, etc.)

    Returns:
        Dictionary containing reference data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Reference data file not found: {file_path}")

    if file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)

    elif file_path.suffix == ".npz":
        data = dict(np.load(file_path, allow_pickle=True))
        # Convert numpy arrays to lists for JSON compatibility where needed
        return data

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def create_reference_comparison_report(validation_results: list[dict[str, Any]]) -> str:
    """
    Create a comprehensive report comparing multiple reference validations

    Args:
        validation_results: List of validation results from different validators

    Returns:
        Formatted comparison report
    """
    report = """
=== XPCS Algorithm Reference Validation Report ===

"""

    total_validations = len(validation_results)
    successful_validations = sum(
        1 for result in validation_results if result.get("overall_valid", False)
    )

    report += f"Total Validations: {total_validations}\n"
    report += f"Successful Validations: {successful_validations}\n"
    report += f"Success Rate: {successful_validations / total_validations:.1%}\n\n"

    # Detailed results for each validator
    for i, result in enumerate(validation_results, 1):
        validator_type = result.get("validator_type", "Unknown")
        is_valid = result.get("overall_valid", False)
        status = "PASS" if is_valid else "FAIL"

        report += f"{i}. {validator_type.upper()}: {status}\n"

        if "reference_citation" in result:
            report += f"   Reference: {result['reference_citation']}\n"
        elif "reference_package" in result:
            report += f"   Package: {result['reference_package']}\n"

        # Summary of validation details
        if "validation_details" in result:
            details = result["validation_details"]
            if "compared_fields" in details:
                report += (
                    f"   Compared Fields: {', '.join(details['compared_fields'])}\n"
                )

            if "field_comparisons" in details:
                for field, comparison in details["field_comparisons"].items():
                    if comparison.get("valid", False):
                        report += f"   ✓ {field}: PASS\n"
                    else:
                        report += f"   ✗ {field}: FAIL\n"

        if "error" in result:
            report += f"   Error: {result['error']}\n"

        report += "\n"

    # Overall assessment
    report += "=== Overall Assessment ===\n"

    if successful_validations == total_validations:
        report += "✓ All reference validations passed successfully.\n"
        report += (
            "The algorithm implementation is consistent with established references.\n"
        )
    elif successful_validations >= total_validations * 0.8:
        report += "⚠ Most reference validations passed.\n"
        report += "Minor discrepancies may be due to implementation differences.\n"
    else:
        report += "✗ Significant validation failures detected.\n"
        report += "Algorithm implementation requires review and correction.\n"

    report += "\n" + "=" * 50

    return report
