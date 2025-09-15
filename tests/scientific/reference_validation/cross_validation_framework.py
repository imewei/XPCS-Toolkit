"""
Cross-Validation Framework for XPCS Algorithms

This module provides a comprehensive cross-validation framework to validate
XPCS algorithms against reference implementations, analytical solutions,
and established benchmarks from literature.

The framework supports:
1. Cross-validation with external software packages
2. Analytical solution validation
3. Statistical cross-validation procedures
4. Regression testing against previous versions
5. Inter-laboratory comparison validation
"""

import unittest
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import optimize


class ReferenceValidator(ABC):
    """Abstract base class for reference validation"""

    @abstractmethod
    def validate_against_reference(
        self, input_data: dict[str, Any], algorithm_output: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate algorithm output against reference implementation

        Args:
            input_data: Input data for the algorithm
            algorithm_output: Output from the algorithm being tested

        Returns:
            Tuple of (is_valid, validation_report)
        """


class AnalyticalValidator(ReferenceValidator):
    """Validator that compares against analytical solutions"""

    def __init__(self, tolerance: float = 1e-6, relative_tolerance: float = 1e-3):
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance

    def validate_against_reference(
        self, input_data: dict[str, Any], algorithm_output: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate against analytical solutions

        Args:
            input_data: Must contain 'analytical_function', 'parameters', 'x_values'
            algorithm_output: Must contain 'y_values'

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "validator_type": "analytical",
            "tolerance_used": self.tolerance,
            "relative_tolerance_used": self.relative_tolerance,
            "validation_details": {},
        }

        try:
            # Extract required data
            analytical_func = input_data["analytical_function"]
            parameters = input_data["parameters"]
            x_values = input_data["x_values"]
            y_computed = algorithm_output["y_values"]

            # Compute analytical reference
            y_analytical = analytical_func(x_values, **parameters)

            # Compare results
            absolute_errors = np.abs(y_computed - y_analytical)
            relative_errors = np.abs(absolute_errors / y_analytical)

            # Handle division by zero in relative errors
            relative_errors = np.where(
                np.abs(y_analytical) < 1e-15, absolute_errors, relative_errors
            )

            max_abs_error = np.max(absolute_errors)
            max_rel_error = np.max(relative_errors)
            mean_abs_error = np.mean(absolute_errors)
            mean_rel_error = np.mean(relative_errors)

            # Check validation criteria
            abs_valid = max_abs_error <= self.tolerance
            rel_valid = max_rel_error <= self.relative_tolerance

            is_valid = abs_valid or rel_valid  # Pass if either criterion is met

            # Detailed validation report
            report["validation_details"] = {
                "max_absolute_error": float(max_abs_error),
                "max_relative_error": float(max_rel_error),
                "mean_absolute_error": float(mean_abs_error),
                "mean_relative_error": float(mean_rel_error),
                "absolute_criterion_met": abs_valid,
                "relative_criterion_met": rel_valid,
                "n_points_compared": len(x_values),
            }

            # Statistical analysis
            if len(y_computed) > 10:
                # Correlation analysis
                correlation = np.corrcoef(y_computed, y_analytical)[0, 1]

                # R-squared
                ss_res = np.sum((y_computed - y_analytical) ** 2)
                ss_tot = np.sum((y_analytical - np.mean(y_analytical)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                report["validation_details"].update(
                    {"correlation": float(correlation), "r_squared": float(r_squared)}
                )

            return is_valid, report

        except Exception as e:
            report["validation_details"] = {"error": str(e)}
            return False, report


class StatisticalCrossValidator:
    """Cross-validator using statistical methods like k-fold validation"""

    def __init__(self, n_folds: int = 5, random_state: int | None = None):
        self.n_folds = n_folds
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def k_fold_validate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        fitting_function: Callable,
        parameter_bounds: tuple | None = None,
    ) -> dict[str, Any]:
        """
        Perform k-fold cross-validation on fitting algorithm

        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            fitting_function: Function to fit (scipy.optimize.curve_fit compatible)
            parameter_bounds: Bounds for fitting parameters

        Returns:
            Cross-validation results dictionary
        """
        n_data = len(x_data)
        if n_data < self.n_folds:
            raise ValueError(
                f"Not enough data points ({n_data}) for {self.n_folds}-fold validation"
            )

        # Create fold indices
        indices = np.arange(n_data)
        np.random.shuffle(indices)

        fold_size = n_data // self.n_folds
        fold_indices = []

        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_folds - 1 else n_data
            fold_indices.append(indices[start_idx:end_idx])

        # Cross-validation loop
        fold_results = []
        all_parameters = []
        all_errors = []

        for fold in range(self.n_folds):
            # Create training and validation sets
            val_indices = fold_indices[fold]
            train_indices = np.concatenate(
                [fold_indices[i] for i in range(self.n_folds) if i != fold]
            )

            x_train, y_train = x_data[train_indices], y_data[train_indices]
            x_val, y_val = x_data[val_indices], y_data[val_indices]

            try:
                # Fit on training data
                if parameter_bounds is not None:
                    popt, pcov = optimize.curve_fit(
                        fitting_function,
                        x_train,
                        y_train,
                        bounds=parameter_bounds,
                        maxfev=5000,
                    )
                else:
                    popt, pcov = optimize.curve_fit(
                        fitting_function, x_train, y_train, maxfev=5000
                    )

                # Evaluate on validation data
                y_pred = fitting_function(x_val, *popt)

                # Calculate validation metrics
                mse = np.mean((y_val - y_pred) ** 2)
                mae = np.mean(np.abs(y_val - y_pred))

                if len(y_val) > 1:
                    r2 = 1 - np.sum((y_val - y_pred) ** 2) / np.sum(
                        (y_val - np.mean(y_val)) ** 2
                    )
                else:
                    r2 = np.nan

                fold_result = {
                    "fold": fold,
                    "parameters": popt,
                    "parameter_errors": np.sqrt(np.diag(pcov)),
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "n_train": len(x_train),
                    "n_val": len(x_val),
                }

                fold_results.append(fold_result)
                all_parameters.append(popt)
                all_errors.append(np.sqrt(np.diag(pcov)))

            except Exception as e:
                fold_result = {"fold": fold, "error": str(e), "failed": True}
                fold_results.append(fold_result)

        # Aggregate results
        successful_folds = [r for r in fold_results if "error" not in r]
        n_successful = len(successful_folds)

        if n_successful == 0:
            return {
                "success": False,
                "error": "All folds failed",
                "fold_results": fold_results,
            }

        # Calculate aggregate statistics
        all_mse = [r["mse"] for r in successful_folds]
        all_mae = [r["mae"] for r in successful_folds]
        all_r2 = [r["r2"] for r in successful_folds if not np.isnan(r["r2"])]

        # Parameter consistency analysis
        param_arrays = np.array([r["parameters"] for r in successful_folds])
        param_means = np.mean(param_arrays, axis=0)
        param_stds = np.std(param_arrays, axis=0)
        param_cvs = param_stds / np.abs(param_means)  # Coefficient of variation

        cross_val_results = {
            "success": True,
            "n_folds": self.n_folds,
            "n_successful_folds": n_successful,
            "validation_metrics": {
                "mean_mse": np.mean(all_mse),
                "std_mse": np.std(all_mse),
                "mean_mae": np.mean(all_mae),
                "std_mae": np.std(all_mae),
                "mean_r2": np.mean(all_r2) if all_r2 else np.nan,
                "std_r2": np.std(all_r2) if all_r2 else np.nan,
            },
            "parameter_consistency": {
                "parameter_means": param_means,
                "parameter_stds": param_stds,
                "parameter_cv": param_cvs,
                "max_cv": np.max(param_cvs),
            },
            "fold_results": fold_results,
        }

        return cross_val_results


class ReferenceImplementationValidator(ReferenceValidator):
    """Validator that compares against reference software implementations"""

    def __init__(self, reference_software: str, tolerance: float = 1e-4):
        self.reference_software = reference_software
        self.tolerance = tolerance
        self.supported_software = ["matlab_dls", "igor_pro", "pymcr", "scikit_learn"]

        if reference_software not in self.supported_software:
            warnings.warn(
                f"Reference software '{reference_software}' not in supported list: "
                f"{self.supported_software}",
                stacklevel=2,
            )

    def validate_against_reference(
        self, input_data: dict[str, Any], algorithm_output: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate against reference software implementation

        Args:
            input_data: Contains reference data from external software
            algorithm_output: Output from XPCS Toolkit algorithm

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "validator_type": "reference_implementation",
            "reference_software": self.reference_software,
            "tolerance_used": self.tolerance,
            "validation_details": {},
        }

        try:
            # Extract reference and computed results
            reference_results = input_data["reference_results"]
            computed_results = algorithm_output

            # Compare common output fields
            common_fields = set(reference_results.keys()) & set(computed_results.keys())

            if not common_fields:
                return False, {
                    **report,
                    "validation_details": {
                        "error": "No common fields between reference and computed results"
                    },
                }

            field_validations = {}
            overall_valid = True

            for field in common_fields:
                ref_values = np.asarray(reference_results[field])
                comp_values = np.asarray(computed_results[field])

                if ref_values.shape != comp_values.shape:
                    field_validations[field] = {
                        "valid": False,
                        "error": f"Shape mismatch: {ref_values.shape} vs {comp_values.shape}",
                    }
                    overall_valid = False
                    continue

                # Calculate comparison metrics
                abs_diff = np.abs(ref_values - comp_values)
                max_abs_diff = np.max(abs_diff)
                mean_abs_diff = np.mean(abs_diff)

                # Relative differences
                rel_diff = np.abs(
                    abs_diff / (ref_values + 1e-15)
                )  # Avoid division by zero
                max_rel_diff = np.max(rel_diff)
                mean_rel_diff = np.mean(rel_diff)

                # Validation criteria
                field_valid = (
                    max_abs_diff <= self.tolerance or max_rel_diff <= self.tolerance
                )

                field_validations[field] = {
                    "valid": field_valid,
                    "max_absolute_difference": float(max_abs_diff),
                    "mean_absolute_difference": float(mean_abs_diff),
                    "max_relative_difference": float(max_rel_diff),
                    "mean_relative_difference": float(mean_rel_diff),
                    "correlation": float(
                        np.corrcoef(ref_values.flatten(), comp_values.flatten())[0, 1]
                    ),
                }

                if not field_valid:
                    overall_valid = False

            report["validation_details"] = {
                "compared_fields": list(common_fields),
                "field_validations": field_validations,
                "overall_agreement": overall_valid,
            }

            return overall_valid, report

        except Exception as e:
            report["validation_details"] = {"error": str(e)}
            return False, report


class ComprehensiveCrossValidationFramework:
    """Comprehensive framework combining multiple validation approaches"""

    def __init__(self):
        self.validators = {}
        self.validation_history = []

    def add_validator(self, name: str, validator: ReferenceValidator):
        """Add a validator to the framework"""
        self.validators[name] = validator

    def run_comprehensive_validation(
        self, algorithm_name: str, test_cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Run comprehensive validation across multiple test cases and validators

        Args:
            algorithm_name: Name of algorithm being validated
            test_cases: List of test cases, each containing 'input_data' and 'algorithm_output'

        Returns:
            Comprehensive validation report
        """
        validation_report = {
            "algorithm_name": algorithm_name,
            "timestamp": str(np.datetime64("now")),
            "n_test_cases": len(test_cases),
            "n_validators": len(self.validators),
            "validator_results": {},
            "summary": {},
        }

        # Run each validator on each test case
        for validator_name, validator in self.validators.items():
            validator_results = []

            for i, test_case in enumerate(test_cases):
                input_data = test_case["input_data"]
                algorithm_output = test_case["algorithm_output"]

                try:
                    is_valid, validation_details = validator.validate_against_reference(
                        input_data, algorithm_output
                    )

                    test_result = {
                        "test_case": i,
                        "valid": is_valid,
                        "details": validation_details,
                    }

                except Exception as e:
                    test_result = {"test_case": i, "valid": False, "error": str(e)}

                validator_results.append(test_result)

            validation_report["validator_results"][validator_name] = validator_results

        # Generate summary statistics
        summary = {}
        for validator_name, results in validation_report["validator_results"].items():
            valid_count = sum(1 for r in results if r.get("valid", False))
            total_count = len(results)
            success_rate = valid_count / total_count if total_count > 0 else 0

            summary[validator_name] = {
                "success_rate": success_rate,
                "passed_tests": valid_count,
                "total_tests": total_count,
            }

        # Overall summary
        all_success_rates = [s["success_rate"] for s in summary.values()]
        summary["overall"] = {
            "mean_success_rate": np.mean(all_success_rates) if all_success_rates else 0,
            "min_success_rate": np.min(all_success_rates) if all_success_rates else 0,
            "all_validators_passed": all(
                s["success_rate"] == 1.0 for s in summary.values()
            ),
        }

        validation_report["summary"] = summary

        # Store in history
        self.validation_history.append(validation_report)

        return validation_report

    def generate_validation_certificate(self, validation_report: dict[str, Any]) -> str:
        """Generate a validation certificate summarizing results"""

        summary = validation_report["summary"]
        overall = summary["overall"]

        certificate = f"""
=== XPCS Toolkit Algorithm Validation Certificate ===

Algorithm: {validation_report["algorithm_name"]}
Validation Date: {validation_report["timestamp"]}
Test Cases: {validation_report["n_test_cases"]}
Validators: {validation_report["n_validators"]}

VALIDATION RESULTS:
"""

        for validator_name, results in summary.items():
            if validator_name != "overall":
                status = (
                    "PASS"
                    if results["success_rate"] == 1.0
                    else "PARTIAL"
                    if results["success_rate"] > 0.5
                    else "FAIL"
                )
                certificate += f"  {validator_name}: {status} ({results['passed_tests']}/{results['total_tests']}) - {results['success_rate']:.1%}\n"

        certificate += "\nOVERALL STATUS: "
        if overall["all_validators_passed"]:
            certificate += "✓ VALIDATED - All tests passed\n"
        elif overall["mean_success_rate"] > 0.8:
            certificate += "⚠ PARTIALLY VALIDATED - Most tests passed\n"
        else:
            certificate += "✗ VALIDATION FAILED - Significant issues found\n"

        certificate += f"Mean Success Rate: {overall['mean_success_rate']:.1%}\n"
        certificate += "=" * 50

        return certificate


class TestCrossValidationFramework(unittest.TestCase):
    """Test cases for the cross-validation framework"""

    def setUp(self):
        """Set up test framework"""
        self.framework = ComprehensiveCrossValidationFramework()

        # Add analytical validator
        analytical_validator = AnalyticalValidator(tolerance=1e-6)
        self.framework.add_validator("analytical", analytical_validator)

    def test_analytical_validation_g2_single_exponential(self):
        """Test analytical validation for G2 single exponential"""

        # Define analytical function
        def g2_single_exp(tau, baseline, beta, gamma):
            return baseline + beta * np.exp(-gamma * tau)

        # Test case
        tau_values = np.logspace(-6, 0, 50)
        true_params = {"baseline": 1.0, "beta": 0.8, "gamma": 1000.0}

        # Generate reference data
        g2_analytical = g2_single_exp(tau_values, **true_params)

        # Simulate algorithm output (perfect match)
        g2_computed = g2_analytical.copy()

        test_case = {
            "input_data": {
                "analytical_function": g2_single_exp,
                "parameters": true_params,
                "x_values": tau_values,
            },
            "algorithm_output": {"y_values": g2_computed},
        }

        # Run validation
        report = self.framework.run_comprehensive_validation(
            "G2_SingleExponential_Perfect", [test_case]
        )

        # Check results
        self.assertTrue(report["summary"]["overall"]["all_validators_passed"])
        self.assertEqual(report["summary"]["analytical"]["success_rate"], 1.0)

    def test_analytical_validation_with_noise(self):
        """Test analytical validation with noisy data"""

        def g2_single_exp(tau, baseline, beta, gamma):
            return baseline + beta * np.exp(-gamma * tau)

        tau_values = np.logspace(-6, 0, 50)
        true_params = {"baseline": 1.0, "beta": 0.8, "gamma": 1000.0}

        # Generate reference data
        g2_analytical = g2_single_exp(tau_values, **true_params)

        # Add noise to simulate realistic algorithm output
        noise_level = 1e-4
        np.random.seed(42)
        g2_noisy = g2_analytical + noise_level * np.random.normal(
            size=len(g2_analytical)
        )

        test_case = {
            "input_data": {
                "analytical_function": g2_single_exp,
                "parameters": true_params,
                "x_values": tau_values,
            },
            "algorithm_output": {"y_values": g2_noisy},
        }

        # Run validation with appropriate tolerance
        analytical_validator = AnalyticalValidator(
            tolerance=1e-3, relative_tolerance=1e-3
        )
        framework = ComprehensiveCrossValidationFramework()
        framework.add_validator("analytical_noisy", analytical_validator)

        report = framework.run_comprehensive_validation(
            "G2_SingleExponential_Noisy", [test_case]
        )

        # Should still pass with appropriate tolerance
        self.assertTrue(report["summary"]["overall"]["all_validators_passed"])

    def test_statistical_cross_validation(self):
        """Test statistical cross-validation framework"""

        # Generate synthetic data for cross-validation
        np.random.seed(123)
        x_data = np.linspace(0.1, 5, 100)

        # True function: exponential decay
        def true_function(x, A, tau):
            return A * np.exp(-x / tau)

        true_params = [2.0, 1.5]
        y_true = true_function(x_data, *true_params)

        # Add noise
        noise_level = 0.05
        y_data = y_true + noise_level * y_true * np.random.normal(size=len(y_true))

        # Cross-validation
        cross_validator = StatisticalCrossValidator(n_folds=5, random_state=42)

        results = cross_validator.k_fold_validate(
            x_data, y_data, true_function, parameter_bounds=([0.1, 0.1], [10.0, 10.0])
        )

        # Check results
        self.assertTrue(results["success"])
        self.assertEqual(results["n_successful_folds"], 5)

        # Check parameter consistency
        max_cv = results["parameter_consistency"]["max_cv"]
        self.assertLess(max_cv, 0.2)  # Parameters should be consistent across folds

        # Check validation metrics
        mean_r2 = results["validation_metrics"]["mean_r2"]
        self.assertGreater(mean_r2, 0.9)  # Should have good fit quality


if __name__ == "__main__":
    unittest.main(verbosity=2)
