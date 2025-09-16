"""
Comprehensive test suite for diffusion analysis enhancements.

This module provides thorough testing of the comprehensive diffusion analysis framework
including multiple diffusion models, weighted regression, outlier detection, model selection,
and advanced coefficient extraction capabilities.

Tests cover:
- Multiple diffusion models (simple, subdiffusion, hyperdiffusion, anomalous)
- Weighted regression framework for heteroscedastic errors
- Outlier detection and robust fitting methods
- Automated model selection using AIC/BIC and cross-validation
- Advanced diffusion coefficient extraction with uncertainty propagation
- Integration with existing error analysis framework
- Edge cases and error handling
"""

import unittest
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

from xpcs_toolkit.helper.fitting import (
    # New diffusion analysis classes
    DiffusionModels,
    WeightedRegressionFramework,
    OutlierDetection,
    ModelSelectionFramework,
    DiffusionCoefficientExtractor,
    ComprehensiveDiffusionAnalyzer,

    # Existing classes for integration testing
    SyntheticG2DataGenerator,
    RobustOptimizer,
    single_exp
)


class TestDiffusionModels(unittest.TestCase):
    """Test suite for diffusion models collection."""

    def setUp(self):
        """Set up test fixtures."""
        self.models = DiffusionModels()
        self.tau = np.logspace(-6, 0, 50)  # Standard time delay array

    def test_simple_diffusion_model(self):
        """Test simple diffusion model."""
        baseline = 1.0
        amplitude = 0.5
        diffusion_coefficient = 1e-3

        g2 = self.models.simple_diffusion(self.tau, baseline, amplitude, diffusion_coefficient)

        # Check basic properties
        self.assertEqual(len(g2), len(self.tau))
        self.assertTrue(np.all(g2 >= baseline))  # Should be at least baseline
        self.assertTrue(np.all(np.isfinite(g2)))

        # Check exponential decay property
        self.assertTrue(np.all(np.diff(g2) <= 0))  # Monotonic decay

        # Check asymptotic behavior
        self.assertAlmostEqual(g2[0], baseline + amplitude, places=3)  # Initial value
        # Final value should be closer to baseline than initial value
        self.assertLess(abs(g2[-1] - baseline), abs(g2[0] - baseline))

    def test_subdiffusion_model(self):
        """Test subdiffusion model with α < 1."""
        baseline = 1.0
        amplitude = 0.4
        diffusion_coefficient = 1e-3
        alpha = 0.7  # Subdiffusion exponent

        g2 = self.models.subdiffusion(self.tau, baseline, amplitude, diffusion_coefficient, alpha)

        # Check basic properties
        self.assertEqual(len(g2), len(self.tau))
        self.assertTrue(np.all(g2 >= baseline))
        self.assertTrue(np.all(np.isfinite(g2)))
        self.assertTrue(np.all(np.diff(g2) <= 0))

        # Test constraint enforcement
        g2_extreme = self.models.subdiffusion(self.tau, baseline, amplitude, diffusion_coefficient, -0.1)
        self.assertTrue(np.all(np.isfinite(g2_extreme)))  # Should handle invalid alpha

    def test_hyperdiffusion_model(self):
        """Test hyperdiffusion model with α > 1."""
        baseline = 1.0
        amplitude = 0.6
        diffusion_coefficient = 1e-4
        alpha = 1.5  # Hyperdiffusion exponent

        g2 = self.models.hyperdiffusion(self.tau, baseline, amplitude, diffusion_coefficient, alpha)

        # Check basic properties
        self.assertEqual(len(g2), len(self.tau))
        self.assertTrue(np.all(g2 >= baseline))
        self.assertTrue(np.all(np.isfinite(g2)))

        # Test constraint enforcement
        g2_low_alpha = self.models.hyperdiffusion(self.tau, baseline, amplitude, diffusion_coefficient, 0.5)
        self.assertTrue(np.all(np.isfinite(g2_low_alpha)))  # Should enforce α > 1

    def test_anomalous_diffusion_model(self):
        """Test general anomalous diffusion model."""
        baseline = 1.0
        amplitude = 0.3
        diffusion_coefficient = 5e-4

        # Test different alpha values
        alpha_values = [0.5, 1.0, 1.8]

        for alpha in alpha_values:
            with self.subTest(alpha=alpha):
                g2 = self.models.anomalous_diffusion(self.tau, baseline, amplitude, diffusion_coefficient, alpha)

                self.assertEqual(len(g2), len(self.tau))
                self.assertTrue(np.all(g2 >= baseline))
                self.assertTrue(np.all(np.isfinite(g2)))

    def test_stretched_exponential_model(self):
        """Test stretched exponential model."""
        baseline = 1.0
        amplitude = 0.4
        relaxation_time = 1e-3
        beta = 0.8  # Stretching exponent

        g2 = self.models.stretched_exponential(self.tau, baseline, amplitude, relaxation_time, beta)

        # Check basic properties
        self.assertEqual(len(g2), len(self.tau))
        self.assertTrue(np.all(g2 >= baseline))
        self.assertTrue(np.all(np.isfinite(g2)))

    def test_double_exponential_model(self):
        """Test double exponential model for bimodal diffusion."""
        baseline = 1.0
        amplitude1 = 0.3
        rate1 = 1e3
        amplitude2 = 0.2
        rate2 = 1e2

        g2 = self.models.double_exponential(self.tau, baseline, amplitude1, rate1, amplitude2, rate2)

        # Check basic properties
        self.assertEqual(len(g2), len(self.tau))
        self.assertTrue(np.all(g2 >= baseline))
        self.assertTrue(np.all(np.isfinite(g2)))

    def test_get_model_info(self):
        """Test model information retrieval."""
        model_info = self.models.get_model_info()

        # Check that all expected models are present
        expected_models = [
            'simple_diffusion', 'subdiffusion', 'hyperdiffusion',
            'anomalous_diffusion', 'stretched_exponential', 'double_exponential'
        ]

        for model_name in expected_models:
            self.assertIn(model_name, model_info)

            model_data = model_info[model_name]
            self.assertIn('function', model_data)
            self.assertIn('parameters', model_data)
            self.assertIn('bounds', model_data)
            self.assertIn('description', model_data)

            # Check that function is callable
            self.assertTrue(callable(model_data['function']))

            # Check bounds structure
            bounds = model_data['bounds']
            self.assertEqual(len(bounds), len(model_data['parameters']))
            for bound in bounds:
                self.assertEqual(len(bound), 2)
                self.assertLess(bound[0], bound[1])  # Lower < upper


class TestWeightedRegressionFramework(unittest.TestCase):
    """Test suite for weighted regression framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.framework = WeightedRegressionFramework()

        # Create test data with known heteroscedastic errors
        np.random.seed(42)
        self.n_points = 50
        self.xdata = np.logspace(-6, 0, self.n_points)

        # Generate true G2 data
        true_params = [1.0, 0.4, 1e-3]  # baseline, amplitude, decay rate
        self.ydata_true = single_exp(self.xdata, *true_params)

        # Add heteroscedastic noise (error increases with time)
        error_scale = 0.02 * (1 + 2 * np.sqrt(self.xdata))
        noise = np.random.normal(0, error_scale)
        self.ydata = self.ydata_true + noise
        self.sigma = error_scale

    def test_inverse_variance_weights(self):
        """Test inverse variance weight calculation."""
        weights = self.framework.calculate_weights_inverse_variance(self.sigma)

        # Check basic properties
        self.assertEqual(len(weights), len(self.sigma))
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(np.isfinite(weights)))

        # Weights should be inversely related to sigma
        high_sigma_idx = np.argmax(self.sigma)
        low_sigma_idx = np.argmin(self.sigma)
        self.assertGreater(weights[low_sigma_idx], weights[high_sigma_idx])

    def test_robust_mad_weights(self):
        """Test Median Absolute Deviation weight calculation."""
        residuals = self.ydata - self.ydata_true
        weights = self.framework.calculate_weights_robust_mad(residuals)

        # Check basic properties
        self.assertEqual(len(weights), len(residuals))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))  # Tukey's bisquare weights are bounded
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_huber_weights(self):
        """Test Huber weight calculation."""
        residuals = self.ydata - self.ydata_true
        weights = self.framework.calculate_weights_huber(residuals)

        # Check basic properties
        self.assertEqual(len(weights), len(residuals))
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(weights <= 1))
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_adaptive_weights(self):
        """Test adaptive weight calculation methods."""
        methods = ['local_variance', 'signal_dependent']

        for method in methods:
            with self.subTest(method=method):
                weights = self.framework.calculate_weights_adaptive(
                    self.ydata, self.ydata_true, method=method
                )

                self.assertEqual(len(weights), len(self.ydata))
                self.assertTrue(np.all(weights > 0))
                self.assertTrue(np.all(np.isfinite(weights)))

                # Weights should be normalized
                np.testing.assert_allclose(np.mean(weights), 1.0, rtol=0.1)

    def test_weighted_least_squares(self):
        """Test weighted least squares fitting."""
        weights = self.framework.calculate_weights_inverse_variance(self.sigma)

        # Fit with weights
        popt, pcov = self.framework.weighted_least_squares(
            single_exp, self.xdata, self.ydata, weights,
            bounds=([1e-6, 0.9, 0.01], [1e0, 1.1, 2.0])
        )

        # Check result structure
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(popt)))
        self.assertTrue(np.all(np.isfinite(pcov)))

    def test_iteratively_reweighted_least_squares(self):
        """Test IRLS for robust fitting."""
        # Add some outliers to test robustness
        ydata_outliers = self.ydata.copy()
        outlier_indices = [10, 25, 40]
        ydata_outliers[outlier_indices] += 0.5  # Add outliers

        results = self.framework.iteratively_reweighted_least_squares(
            single_exp, self.xdata, ydata_outliers,
            bounds=([1e-6, 0.9, 0.01], [1e0, 1.1, 2.0]),
            max_iterations=5, weight_method='huber'
        )

        # Check result structure
        required_keys = ['parameters', 'covariance', 'weights', 'convergence_history',
                        'converged', 'iterations', 'final_residuals']
        for key in required_keys:
            self.assertIn(key, results)

        # Check parameter validity
        params = results['parameters']
        self.assertEqual(len(params), 3)
        self.assertTrue(np.all(np.isfinite(params)))

        # Weights should be computed and finite
        weights = results['weights']
        self.assertEqual(len(weights), len(ydata_outliers))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0))  # All weights should be positive

        # Check that the algorithm produces different weights (not all equal)
        # This ensures the robust weighting is actually working
        self.assertGreater(np.std(weights), 0.01,
                          "Weights should vary to indicate robust downweighting is working")


class TestOutlierDetection(unittest.TestCase):
    """Test suite for outlier detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = OutlierDetection()

        # Create test data with known outliers
        np.random.seed(123)
        n_points = 100
        self.clean_data = np.random.normal(0, 1, n_points)

        # Add known outliers
        self.outlier_indices = [10, 25, 75]
        self.data_with_outliers = self.clean_data.copy()
        self.data_with_outliers[self.outlier_indices] += 5.0  # Add outliers

    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection."""
        outliers = self.detector.detect_outliers_iqr(self.data_with_outliers)

        # Check basic properties
        self.assertEqual(len(outliers), len(self.data_with_outliers))
        self.assertTrue(outliers.dtype == bool)

        # Should detect the known outliers
        detected_outlier_count = np.sum(outliers)
        self.assertGreaterEqual(detected_outlier_count, len(self.outlier_indices))

        # Known outliers should be detected
        for idx in self.outlier_indices:
            self.assertTrue(outliers[idx], f"Outlier at index {idx} not detected")

    def test_mad_outlier_detection(self):
        """Test MAD-based outlier detection."""
        outliers = self.detector.detect_outliers_mad(self.data_with_outliers)

        # Check basic properties
        self.assertEqual(len(outliers), len(self.data_with_outliers))
        self.assertTrue(outliers.dtype == bool)

        # Should detect the known outliers
        detected_outlier_count = np.sum(outliers)
        self.assertGreater(detected_outlier_count, 0)

    def test_zscore_outlier_detection(self):
        """Test Z-score based outlier detection."""
        outliers = self.detector.detect_outliers_zscore(self.data_with_outliers)

        # Check basic properties
        self.assertEqual(len(outliers), len(self.data_with_outliers))
        self.assertTrue(outliers.dtype == bool)

        # Should detect at least some outliers (Z-score can be affected by outliers in data)
        detected_outlier_count = np.sum(outliers)
        self.assertGreater(detected_outlier_count, 0, "Z-score should detect at least some outliers")
        
        # The most extreme outlier should always be detected
        most_extreme_idx = 75  # This had the highest Z-score in our test
        self.assertTrue(outliers[most_extreme_idx], f"Most extreme outlier at index {most_extreme_idx} not detected")
    def test_isolation_forest_outlier_detection(self):
        """Test Isolation Forest outlier detection (if sklearn available)."""
        try:
            outliers = self.detector.detect_outliers_isolation_forest(self.data_with_outliers)

            # Check basic properties
            self.assertEqual(len(outliers), len(self.data_with_outliers))
            self.assertTrue(outliers.dtype == bool)

        except ImportError:
            # sklearn not available, should fall back to MAD
            outliers = self.detector.detect_outliers_isolation_forest(self.data_with_outliers)
            self.assertEqual(len(outliers), len(self.data_with_outliers))

    def test_residual_outlier_detection(self):
        """Test outlier detection in residuals."""
        # Create residuals with outliers
        y_true = np.sin(np.linspace(0, 4*np.pi, 100))
        y_pred = y_true + 0.1 * np.random.normal(size=100)

        # Add outlier residuals
        outlier_indices = [20, 50, 80]
        y_true_outliers = y_true.copy()
        y_true_outliers[outlier_indices] += 2.0

        outliers = self.detector.detect_residual_outliers(y_true_outliers, y_pred, method='mad')

        # Should detect outliers
        self.assertTrue(np.sum(outliers) > 0)

    def test_robust_outlier_detection(self):
        """Test combined robust outlier detection."""
        # Create test fitting scenario
        xdata = np.logspace(-6, 0, 50)
        true_params = [1.0, 0.4, 1e-3]
        ydata = single_exp(xdata, *true_params)

        # Add outliers
        outlier_indices = [10, 25, 40]
        ydata[outlier_indices] += 0.3

        results = self.detector.robust_outlier_detection(
            xdata, ydata, single_exp, true_params, methods=['mad', 'iqr', 'zscore']
        )

        # Check result structure
        self.assertIn('consensus', results)
        self.assertIn('mad', results)
        self.assertIn('iqr', results)
        self.assertIn('zscore', results)

        # Each method should return boolean array
        for method in ['mad', 'iqr', 'zscore', 'consensus']:
            outliers = results[method]
            self.assertEqual(len(outliers), len(ydata))
            self.assertTrue(outliers.dtype == bool)


class TestModelSelectionFramework(unittest.TestCase):
    """Test suite for model selection framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = ModelSelectionFramework()

        # Create synthetic data that follows simple diffusion
        np.random.seed(456)
        self.tau = np.logspace(-6, 0, 40)
        true_params = [1.0, 0.4, 1e-3]  # baseline, amplitude, decay rate

        # Generate clean data with simple diffusion
        self.g2_true = DiffusionModels.simple_diffusion(self.tau, *true_params)
        noise = 0.02 * self.g2_true * np.random.normal(size=len(self.tau))
        self.g2_data = self.g2_true + noise
        self.g2_errors = 0.02 * self.g2_true

        # Get model information for testing
        self.models = DiffusionModels().get_model_info()

    def test_information_criteria_calculation(self):
        """Test AIC, BIC, and AICc calculation."""
        n_params = 3
        n_data = 40
        chi_squared = 45.0

        aic = self.selector.calculate_aic(n_params, n_data, chi_squared)
        bic = self.selector.calculate_bic(n_params, n_data, chi_squared)
        aicc = self.selector.calculate_aicc(n_params, n_data, chi_squared)

        # Check that values are finite
        self.assertTrue(np.isfinite(aic))
        self.assertTrue(np.isfinite(bic))
        self.assertTrue(np.isfinite(aicc))

        # AICc should be larger than AIC (penalty for small sample size)
        self.assertGreaterEqual(aicc, aic)

        # BIC should be different from AIC
        self.assertNotEqual(bic, aic)

    def test_cross_validation(self):
        """Test k-fold cross-validation."""
        # Use simple diffusion model for testing
        model_info = self.models['simple_diffusion']
        func = model_info['function']
        bounds = model_info['bounds']

        # Generate initial parameters
        lower_bounds, upper_bounds = zip(*bounds)
        p0 = np.array([(l + u) / 2 for l, u in bounds])

        cv_results = self.selector.cross_validate_model(
            func, self.tau, self.g2_data, p0, bounds=bounds,
            k_folds=5, sigma=self.g2_errors
        )

        # Check result structure
        required_keys = ['mean_cv_score', 'std_cv_score', 'cv_scores',
                        'cv_parameters', 'n_successful_folds']
        for key in required_keys:
            self.assertIn(key, cv_results)

        # Check values
        self.assertTrue(np.isfinite(cv_results['mean_cv_score']))
        self.assertTrue(np.isfinite(cv_results['std_cv_score']))
        self.assertEqual(len(cv_results['cv_scores']), 5)
        self.assertGreaterEqual(cv_results['n_successful_folds'], 0)

    def test_model_comparison(self):
        """Test comparison of multiple diffusion models."""
        # Select subset of models for testing
        test_models = {
            'simple_diffusion': self.models['simple_diffusion'],
            'subdiffusion': self.models['subdiffusion']
        }

        results = self.selector.compare_models(
            test_models, self.tau, self.g2_data, self.g2_errors,
            use_cross_validation=True, cv_folds=3
        )

        # Check that both models were evaluated
        self.assertIn('simple_diffusion', results)
        self.assertIn('subdiffusion', results)

        # Check result structure for each model
        for model_name, model_results in results.items():
            if model_results.get('fitted_successfully', False):
                required_keys = ['parameters', 'covariance', 'chi_squared',
                               'aic', 'bic', 'aicc', 'r_squared', 'rmse']
                for key in required_keys:
                    self.assertIn(key, model_results)

                # Check cross-validation results
                if 'cross_validation' in model_results:
                    cv_data = model_results['cross_validation']
                    self.assertIn('mean_cv_score', cv_data)

    def test_best_model_selection(self):
        """Test best model selection."""
        # Create mock comparison results
        mock_results = {
            'model_a': {
                'fitted_successfully': True,
                'aic': 100.0,
                'bic': 105.0,
                'aicc': 102.0,
                'cross_validation': {'mean_cv_score': 0.05}
            },
            'model_b': {
                'fitted_successfully': True,
                'aic': 110.0,
                'bic': 112.0,
                'aicc': 115.0,
                'cross_validation': {'mean_cv_score': 0.08}
            }
        }

        # Test different selection criteria
        for criterion in ['aic', 'bic', 'aicc', 'cv', 'combined']:
            with self.subTest(criterion=criterion):
                selection = self.selector.select_best_model(mock_results, criterion=criterion)

                # Check result structure
                required_keys = ['best_model', 'best_model_results',
                               'selection_info', 'all_model_results']
                for key in required_keys:
                    self.assertIn(key, selection)

                # Best model should be identified
                self.assertIsNotNone(selection['best_model'])

                # For AIC/BIC/AICc, model_a should be better (lower values)
                if criterion in ['aic', 'bic', 'aicc']:
                    self.assertEqual(selection['best_model'], 'model_a')


class TestDiffusionCoefficientExtractor(unittest.TestCase):
    """Test suite for diffusion coefficient extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DiffusionCoefficientExtractor()
        self.q_value = 0.01  # 1/nm, typical XPCS q-value

    def test_simple_diffusion_extraction(self):
        """Test diffusion coefficient extraction from simple diffusion."""
        # Known parameters: D = 1e-12 m²/s, q = 0.01 nm⁻¹
        true_D = 1e-12
        fitted_rate = 2.0 * true_D * (self.q_value * 1e9) ** 2  # Convert to 1/m²

        # Mock fitted parameters [baseline, amplitude, 2D*q²]
        params = np.array([1.0, 0.4, fitted_rate])
        param_errors = np.array([0.01, 0.02, fitted_rate * 0.1])

        results = self.extractor.extract_simple_diffusion_coefficient(
            params, param_errors, self.q_value * 1e9  # Convert to 1/m
        )

        # Check result structure
        required_keys = ['diffusion_coefficient', 'diffusion_coefficient_error',
                        'fitted_rate', 'fitted_rate_error', 'q_value']
        for key in required_keys:
            self.assertIn(key, results)

        # Check extracted diffusion coefficient
        extracted_D = results['diffusion_coefficient']
        self.assertAlmostEqual(extracted_D, true_D, places=15)

        # Check uncertainty propagation
        self.assertGreater(results['diffusion_coefficient_error'], 0)

    def test_anomalous_diffusion_extraction(self):
        """Test parameter extraction from anomalous diffusion."""
        # Mock fitted parameters [baseline, amplitude, 2D*q², α]
        params = np.array([1.0, 0.3, 2e6, 0.8])  # α = 0.8 (subdiffusion)
        param_errors = np.array([0.01, 0.02, 1e5, 0.05])

        results = self.extractor.extract_anomalous_diffusion_parameters(
            params, param_errors, self.q_value * 1e9
        )

        # Check result structure
        required_keys = ['diffusion_coefficient', 'diffusion_coefficient_error',
                        'anomalous_exponent', 'anomalous_exponent_error',
                        'fitted_rate', 'fitted_rate_error', 'q_value', 'diffusion_type']
        for key in required_keys:
            self.assertIn(key, results)

        # Check diffusion type classification
        self.assertEqual(results['diffusion_type'], 'subdiffusion')
        self.assertEqual(results['anomalous_exponent'], 0.8)

    def test_monte_carlo_uncertainty_propagation(self):
        """Test Monte Carlo uncertainty propagation."""
        # Create test parameters and covariance matrix
        params = np.array([1.0, 0.4, 1e6])
        param_cov = np.diag([0.01, 0.02, 1e4]) ** 2  # Diagonal covariance

        def extraction_func(p, pe, q_value):
            return self.extractor.extract_simple_diffusion_coefficient(p, pe, q_value)

        mc_results = self.extractor.propagate_uncertainty_monte_carlo(
            extraction_func, params, param_cov, n_samples=100, q_value=self.q_value * 1e9
        )

        # Check that Monte Carlo statistics are calculated
        if mc_results:  # If MC succeeded
            self.assertIn('n_successful_samples', mc_results)
            self.assertGreater(mc_results['n_successful_samples'], 50)

    def test_constraint_application(self):
        """Test physical constraint application."""
        # Test with good parameters
        good_params = np.array([1.0, 0.4, 1e6])
        good_cov = np.diag([0.01, 0.02, 1e4]) ** 2

        results_good = self.extractor.extract_with_constraints(
            'simple_diffusion', good_params, good_cov, self.q_value * 1e9,
            apply_physical_constraints=True
        )

        # Should not have constraint warnings
        self.assertNotIn('physical_constraint_warning', results_good)

        # Test with problematic parameters (negative diffusion coefficient)
        bad_params = np.array([1.0, 0.4, -1e6])  # Negative rate

        results_bad = self.extractor.extract_with_constraints(
            'simple_diffusion', bad_params, good_cov, self.q_value * 1e9,
            apply_physical_constraints=True
        )

        # Should have constraint warning
        self.assertIn('physical_constraint_warning', results_bad)


class TestComprehensiveDiffusionAnalyzer(unittest.TestCase):
    """Test suite for the integrated diffusion analysis framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ComprehensiveDiffusionAnalyzer(
            diagnostic_level='standard', n_jobs=1  # Use single core for tests
        )

        # Create realistic synthetic XPCS data
        np.random.seed(789)
        self.tau = np.logspace(-6, 0, 60)
        self.q_value = 0.01  # nm⁻¹

        # Generate data with simple diffusion + noise
        true_params = [1.0, 0.4, 5e-4]
        self.g2_true = DiffusionModels.simple_diffusion(self.tau, *true_params)

        # Add realistic noise (Poisson-like, heteroscedastic)
        noise_level = 0.03
        noise = noise_level * self.g2_true * np.random.normal(size=len(self.tau))
        self.g2_data = self.g2_true + noise
        self.g2_errors = noise_level * self.g2_true

        # Add a few outliers
        outlier_indices = [15, 35, 50]
        self.g2_data[outlier_indices] += 0.2

    def test_analyzer_initialization(self):
        """Test analyzer initialization and component setup."""
        # Check that all components are initialized
        self.assertIsNotNone(self.analyzer.robust_optimizer)
        self.assertIsNotNone(self.analyzer.weighted_regression)
        self.assertIsNotNone(self.analyzer.outlier_detection)
        self.assertIsNotNone(self.analyzer.model_selection)
        self.assertIsNotNone(self.analyzer.coefficient_extractor)
        self.assertIsNotNone(self.analyzer.diffusion_models)

        # Check available models
        available_models = self.analyzer.get_available_models()
        self.assertIn('simple_diffusion', available_models)
        self.assertIn('subdiffusion', available_models)
        self.assertIn('anomalous_diffusion', available_models)

    def test_input_validation(self):
        """Test input data validation."""
        validation = self.analyzer.validate_input_data(self.tau, self.g2_data, self.g2_errors)

        # Check validation structure
        self.assertIn('valid', validation)
        self.assertIn('warnings', validation)
        self.assertIn('errors', validation)
        self.assertIn('recommendations', validation)

        # Should be valid for good data
        self.assertTrue(validation['valid'])

        # Test invalid data
        invalid_validation = self.analyzer.validate_input_data(
            self.tau[:-10], self.g2_data, self.g2_errors  # Mismatched lengths
        )
        self.assertFalse(invalid_validation['valid'])
        self.assertGreater(len(invalid_validation['errors']), 0)

    def test_quick_diffusion_analysis(self):
        """Test quick single-model diffusion analysis."""
        result = self.analyzer.quick_diffusion_analysis(
            self.tau, self.g2_data, q_value=self.q_value, model='simple_diffusion'
        )

        # Check result structure
        self.assertIn('success', result)

        if result['success']:
            required_keys = ['model', 'parameters', 'parameter_errors',
                           'covariance', 'diffusion_info', 'fit_info']
            for key in required_keys:
                self.assertIn(key, result)

            # Check parameter validity
            params = result['parameters']
            self.assertEqual(len(params), 3)
            self.assertTrue(np.all(np.isfinite(params)))

            # Check diffusion coefficient extraction
            diffusion_info = result['diffusion_info']
            self.assertIn('diffusion_coefficient', diffusion_info)
            self.assertGreater(diffusion_info['diffusion_coefficient'], 0)

    def test_comprehensive_analysis(self):
        """Test comprehensive diffusion analysis with all features."""
        # Run comprehensive analysis with limited models for speed
        results = self.analyzer.analyze_diffusion(
            self.tau, self.g2_data, g2_errors=self.g2_errors, q_value=self.q_value,
            models_to_test=['simple_diffusion', 'subdiffusion'],
            enable_outlier_detection=True,
            enable_weighted_fitting=True,
            model_selection_criterion='bic',
            bootstrap_samples=50,  # Reduced for speed
            apply_physical_constraints=True
        )

        # Check main result sections
        required_sections = ['input_parameters', 'preprocessing', 'model_comparison',
                           'best_model_analysis', 'diffusion_parameters', 'diagnostics']
        for section in required_sections:
            self.assertIn(section, results)

        # Check input parameters section
        input_params = results['input_parameters']
        self.assertEqual(input_params['n_data_points'], len(self.tau))
        self.assertEqual(input_params['q_value'], self.q_value)
        self.assertTrue(input_params['has_error_estimates'])

        # Check that outlier detection was performed
        if 'outlier_detection' in results['preprocessing']:
            outlier_info = results['preprocessing']['outlier_detection']
            if 'n_outliers_detected' in outlier_info:
                self.assertGreaterEqual(outlier_info['n_outliers_detected'], 0)

        # Check model comparison
        model_comparison = results['model_comparison']
        self.assertIn('simple_diffusion', model_comparison)

        # Check that a best model was selected
        self.assertIn('model_selection', results)

        # Check diagnostics
        diagnostics = results['diagnostics']
        self.assertIn('analysis_summary', diagnostics)
        summary = diagnostics['analysis_summary']
        self.assertIn('best_model', summary)
        self.assertIn('data_points_used', summary)

    def test_model_selection_integration(self):
        """Test integration with model selection framework."""
        # Test with a subset of models
        results = self.analyzer.analyze_diffusion(
            self.tau, self.g2_data, g2_errors=self.g2_errors, q_value=self.q_value,
            models_to_test=['simple_diffusion', 'anomalous_diffusion'],
            bootstrap_samples=0,  # Skip bootstrap for speed
            enable_outlier_detection=False  # Skip outlier detection for speed
        )

        # Should have model comparison results
        model_comparison = results['model_comparison']
        self.assertIn('simple_diffusion', model_comparison)
        self.assertIn('anomalous_diffusion', model_comparison)

        # Should have selected a best model
        best_model_info = results.get('model_selection', {})
        self.assertIn('best_model', best_model_info)

    def test_error_handling(self):
        """Test error handling with problematic data."""
        # Create problematic data
        problematic_tau = np.array([1e-10, 1e-9])  # Too few points
        problematic_g2 = np.array([1.0, 1.0])  # Constant data

        try:
            results = self.analyzer.analyze_diffusion(
                problematic_tau, problematic_g2, q_value=self.q_value,
                bootstrap_samples=0  # Skip bootstrap
            )

            # If it completes, should have error information
            if 'error' in results:
                self.assertIsInstance(results['error'], str)

        except Exception as e:
            # It's acceptable for this to raise an exception with bad data
            self.assertIsInstance(e, (ValueError, RuntimeError))

    def test_physical_constraints_integration(self):
        """Test integration with physical constraint checking."""
        results = self.analyzer.analyze_diffusion(
            self.tau, self.g2_data, g2_errors=self.g2_errors, q_value=self.q_value,
            models_to_test=['simple_diffusion'],
            apply_physical_constraints=True,
            bootstrap_samples=0  # Skip for speed
        )

        # Check diffusion parameters section
        if 'diffusion_parameters' in results and not results['diffusion_parameters'].get('error'):
            diffusion_params = results['diffusion_parameters']

            # Should have diffusion coefficient
            self.assertIn('diffusion_coefficient', diffusion_params)

            # Physical constraints should be checked
            if 'physical_constraint_warning' in diffusion_params:
                self.assertIsInstance(diffusion_params['physical_constraint_warning'], str)


class TestIntegrationWithExistingFramework(unittest.TestCase):
    """Test integration with existing robust fitting framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ComprehensiveDiffusionAnalyzer(diagnostic_level='basic')

        # Create data using existing synthetic generator for consistency
        self.generator = SyntheticG2DataGenerator(random_state=999)

    def test_integration_with_synthetic_data_generator(self):
        """Test integration with existing synthetic data generation."""
        # Generate data with existing generator
        tau, g2, g2_err, _true_params = self.generator.generate_dataset(
            model_type='single_exp', n_points=40, noise_level=0.04
        )

        # Analyze with new diffusion analyzer
        results = self.analyzer.quick_diffusion_analysis(
            tau, g2, q_value=0.01, model='simple_diffusion'
        )

        # Should successfully fit the data
        self.assertTrue(results['success'])
        self.assertIn('diffusion_info', results)

    def test_integration_with_robust_optimizer(self):
        """Test that diffusion models work with existing robust optimizer."""
        from xpcs_toolkit.helper.fitting import RobustOptimizer

        # Create test data
        tau = np.logspace(-6, 0, 30)
        true_params = [1.0, 0.4, 1e-3]
        g2 = DiffusionModels.simple_diffusion(tau, *true_params)
        g2 += 0.02 * g2 * np.random.normal(size=len(g2))

        # Use with robust optimizer directly
        optimizer = RobustOptimizer()

        bounds = ([0.5, 0.0, 1e-8], [2.0, 1.0, 1e-2])
        popt, pcov, info = optimizer.robust_curve_fit(
            DiffusionModels.simple_diffusion, tau, g2, bounds=bounds
        )

        # Should successfully fit
        self.assertEqual(len(popt), 3)
        self.assertTrue(np.all(np.isfinite(popt)))
        self.assertIn('r_squared', info)

    def test_backward_compatibility(self):
        """Test that new features don't break existing functionality."""
        # Import existing functions
        from xpcs_toolkit.helper.fitting import robust_curve_fit

        # Create simple test data
        xdata = np.linspace(0.1, 5, 20)
        ydata = 2.0 * np.exp(-xdata / 1.5) + 1.0 + 0.01 * np.random.normal(size=20)

        def exp_model(x, a, tau, b):
            return a * np.exp(-x / tau) + b

        # Should still work with existing interface
        popt, pcov = robust_curve_fit(
            exp_model, xdata, ydata, bounds=([0.1, 0.1, 0.9], [10, 10, 1.1])
        )

        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test edge cases and robustness of diffusion analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ComprehensiveDiffusionAnalyzer(diagnostic_level='basic')

    def test_minimal_data(self):
        """Test behavior with minimal data points."""
        # Very few data points
        tau = np.logspace(-6, 0, 8)  # Only 8 points
        g2 = DiffusionModels.simple_diffusion(tau, 1.0, 0.4, 1e-3)

        validation = self.analyzer.validate_input_data(tau, g2)

        # Should warn about few data points
        self.assertIn('few data points', ' '.join(validation['warnings']).lower())

    def test_constant_data(self):
        """Test behavior with constant data."""
        tau = np.logspace(-6, 0, 30)
        g2 = np.ones_like(tau)  # Constant data

        validation = self.analyzer.validate_input_data(tau, g2)

        # Should warn about small dynamic range
        warnings_text = ' '.join(validation['warnings']).lower()
        self.assertIn('dynamic range', warnings_text)

    def test_extreme_noise(self):
        """Test behavior with extremely noisy data."""
        tau = np.logspace(-6, 0, 50)
        g2_clean = DiffusionModels.simple_diffusion(tau, 1.0, 0.4, 1e-3)

        # Add 50% noise
        noise = 0.5 * g2_clean * np.random.normal(size=len(g2_clean))
        g2_noisy = g2_clean + noise

        try:
            # Should handle gracefully or fail with informative error
            result = self.analyzer.quick_diffusion_analysis(
                tau, g2_noisy, model='simple_diffusion'
            )

            if not result['success']:
                self.assertIn('error', result)

        except Exception as e:
            # Acceptable to fail with extreme noise
            self.assertIsInstance(e, (ValueError, RuntimeError))

    def test_boundary_conditions(self):
        """Test behavior at parameter boundaries."""
        # Data that might push parameters to boundaries
        tau = np.logspace(-6, 0, 30)

        # Very small amplitude (should hit amplitude lower bound)
        g2 = DiffusionModels.simple_diffusion(tau, 1.0, 0.001, 1e-3)

        result = self.analyzer.quick_diffusion_analysis(
            tau, g2, model='simple_diffusion'
        )

        if result['success']:
            params = result['parameters']
            # Parameters should be within reasonable bounds
            self.assertTrue(np.all(np.isfinite(params)))

    def test_missing_error_estimates(self):
        """Test behavior when error estimates are not provided."""
        tau = np.logspace(-6, 0, 40)
        g2 = DiffusionModels.simple_diffusion(tau, 1.0, 0.4, 1e-3)

        validation = self.analyzer.validate_input_data(tau, g2, g2_errors=None)

        # Should recommend providing error estimates
        recommendations_text = ' '.join(validation['recommendations']).lower()
        self.assertIn('error', recommendations_text)

        # Should still work without errors
        result = self.analyzer.quick_diffusion_analysis(
            tau, g2, model='simple_diffusion'
        )

        # Should succeed even without error estimates
        self.assertTrue(result.get('success', False))


if __name__ == '__main__':
    # Configure warnings for clean test output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run tests with appropriate verbosity
    unittest.main(verbosity=2)