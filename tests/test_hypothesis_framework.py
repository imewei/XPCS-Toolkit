#!/usr/bin/env python3
"""
Test validation for the Hypothesis property-based testing framework.

This module validates that the property-based testing framework works correctly
and integrates properly with the XPCS Toolkit testing infrastructure.
"""

import unittest

import numpy as np

from tests.hypothesis_framework import (
    HYPOTHESIS_AVAILABLE,
    HypothesisTestRunner,
    MathematicalInvariants,
    XPCSPropertyTests,
    calculate_mock_g2,
    generate_test_matrix,
    mock_exponential_fit,
)


class TestHypothesisFramework(unittest.TestCase):
    """Test cases for the property-based testing framework."""

    def setUp(self):
        """Set up test environment."""
        self.runner = HypothesisTestRunner(max_examples=10, deadline=5000)

    def test_hypothesis_availability_detection(self):
        """Test that hypothesis availability is correctly detected."""
        # This should always be True in our test environment or False if not installed
        self.assertIsInstance(HYPOTHESIS_AVAILABLE, bool)

    def test_runner_initialization(self):
        """Test that the test runner initializes correctly."""
        self.assertEqual(self.runner.max_examples, 10)
        self.assertEqual(self.runner.deadline, 5000)
        self.assertIsInstance(self.runner.test_results, dict)

    def test_mock_g2_calculation(self):
        """Test the mock G2 calculation function."""
        intensities = np.random.uniform(1, 100, 50)
        g2_values = calculate_mock_g2(intensities)

        # Basic properties
        self.assertGreater(len(g2_values), 0)
        self.assertTrue(np.all(g2_values >= 0))
        self.assertGreaterEqual(g2_values[0], 1.0)  # G2(0) >= 1

        # Check decay pattern
        if len(g2_values) > 5:
            self.assertGreaterEqual(g2_values[0], g2_values[4])

    def test_mock_exponential_fit(self):
        """Test the mock exponential fitting function."""
        x_data = np.linspace(0, 5, 50)
        y_data = 2.0 * np.exp(-0.5 * x_data) + np.random.normal(0, 0.1, 50)

        fitted_params = mock_exponential_fit(x_data, y_data)

        # Check return structure
        required_keys = ["amplitude", "decay_rate", "r_squared"]
        for key in required_keys:
            self.assertIn(key, fitted_params)

        # Check value ranges
        self.assertGreater(fitted_params["amplitude"], 0)
        self.assertGreater(fitted_params["decay_rate"], 0)
        self.assertGreater(fitted_params["r_squared"], 0)
        self.assertLessEqual(fitted_params["r_squared"], 1)

    def test_test_matrix_generation(self):
        """Test the test matrix generation function."""
        size = 5
        condition_number = 10.0

        matrix = generate_test_matrix(size, condition_number)

        # Check dimensions
        self.assertEqual(matrix.shape, (size, size))

        # Check that matrix is invertible
        det = np.linalg.det(matrix)
        self.assertNotAlmostEqual(det, 0, places=10)

        # Check condition number is approximately correct
        actual_cond = np.linalg.cond(matrix)
        self.assertLess(actual_cond, condition_number * 2)  # Allow some tolerance

    def test_mathematical_invariants_structure(self):
        """Test that mathematical invariants are properly structured."""
        math_invariants = MathematicalInvariants()

        # Test that each property method returns a list of test functions
        test_groups = [
            math_invariants.correlation_function_properties(),
            math_invariants.fitting_properties(),
            math_invariants.numerical_stability_properties(),
            math_invariants.fourier_transform_properties(),
        ]

        for test_group in test_groups:
            self.assertIsInstance(test_group, list)
            self.assertGreater(len(test_group), 0)

            for test_func in test_group:
                self.assertTrue(callable(test_func))
                self.assertTrue(hasattr(test_func, "__name__"))

    def test_xpcs_property_tests_structure(self):
        """Test that XPCS property tests are properly structured."""
        xpcs_tests = XPCSPropertyTests()

        test_groups = [
            xpcs_tests.intensity_statistics_properties(),
            xpcs_tests.scattering_properties(),
        ]

        for test_group in test_groups:
            self.assertIsInstance(test_group, list)
            self.assertGreater(len(test_group), 0)

            for test_func in test_group:
                self.assertTrue(callable(test_func))
                self.assertTrue(hasattr(test_func, "__name__"))

    def test_runner_mathematical_tests(self):
        """Test running mathematical invariant tests."""
        results = self.runner.run_mathematical_invariant_tests()

        if HYPOTHESIS_AVAILABLE:
            # Should have results for each test group
            expected_groups = [
                "correlation_functions",
                "fitting_algorithms",
                "numerical_stability",
                "fourier_transforms",
            ]

            for group in expected_groups:
                self.assertIn(group, results)
                self.assertIsInstance(results[group], dict)
        else:
            # Should indicate that hypothesis is not available
            self.assertEqual(results["status"], "skipped")
            self.assertIn("reason", results)

    def test_runner_xpcs_tests(self):
        """Test running XPCS-specific tests."""
        results = self.runner.run_xpcs_specific_tests()

        if HYPOTHESIS_AVAILABLE:
            # Should have results for each test group
            expected_groups = ["intensity_statistics", "scattering_calculations"]

            for group in expected_groups:
                self.assertIn(group, results)
                self.assertIsInstance(results[group], dict)
        else:
            # Should indicate that hypothesis is not available
            self.assertEqual(results["status"], "skipped")
            self.assertIn("reason", results)

    def test_report_generation(self):
        """Test that test reports are generated correctly."""
        report = self.runner.generate_test_report()

        self.assertIsInstance(report, str)
        self.assertIn("Property-Based Testing Report", report)

        if HYPOTHESIS_AVAILABLE:
            self.assertIn("Summary:", report)
            self.assertIn("Total tests:", report)
            self.assertIn("Success rate:", report)
        else:
            self.assertIn("Hypothesis library not available", report)
            self.assertIn("pip install hypothesis", report)

    def test_edge_case_inputs(self):
        """Test framework behavior with edge case inputs."""
        # Test with very small arrays
        small_intensities = np.array([1.0, 2.0])
        g2_small = calculate_mock_g2(small_intensities)
        self.assertGreater(len(g2_small), 0)

        # Test with constant intensities
        constant_intensities = np.ones(20)
        g2_constant = calculate_mock_g2(constant_intensities)
        self.assertGreater(len(g2_constant), 0)
        self.assertGreaterEqual(g2_constant[0], 1.0)

        # Test exponential fit with problematic data
        x_data = np.array([0, 1, 2])
        y_data = np.array([0, 0, 0])  # All zeros
        fitted_params = mock_exponential_fit(x_data, y_data)
        self.assertIn("amplitude", fitted_params)
        self.assertGreater(fitted_params["amplitude"], 0)

    def test_numerical_precision_properties(self):
        """Test that numerical precision is maintained in calculations."""
        # Test with small numbers
        small_intensities = np.random.uniform(1e-6, 1e-5, 100)
        g2_small = calculate_mock_g2(small_intensities)
        self.assertTrue(np.all(np.isfinite(g2_small)))

        # Test with large numbers
        large_intensities = np.random.uniform(1e5, 1e6, 100)
        g2_large = calculate_mock_g2(large_intensities)
        self.assertTrue(np.all(np.isfinite(g2_large)))

    def test_framework_integration(self):
        """Test that the framework integrates with standard test infrastructure."""
        # This test verifies that the framework can be used as part of the standard test suite
        runner = HypothesisTestRunner(max_examples=5, deadline=1000)

        # Should be able to generate a report without errors
        report = runner.generate_test_report()
        self.assertIsInstance(report, str)

        # Should handle both available and unavailable hypothesis scenarios
        if HYPOTHESIS_AVAILABLE:
            math_results = runner.run_mathematical_invariant_tests()
            self.assertIsInstance(math_results, dict)
        else:
            self.assertIn("Hypothesis library not available", report)

    def test_property_test_isolation(self):
        """Test that property tests are properly isolated."""
        # Run the same test multiple times to ensure no state interference
        runner1 = HypothesisTestRunner(max_examples=5)
        runner2 = HypothesisTestRunner(max_examples=5)

        if HYPOTHESIS_AVAILABLE:
            results1 = runner1.run_mathematical_invariant_tests()
            results2 = runner2.run_mathematical_invariant_tests()

            # Results should be independent (structure should be the same)
            self.assertEqual(set(results1.keys()), set(results2.keys()))


if __name__ == "__main__":
    unittest.main()
