"""Tests for NLSQResult delegation to CurveFitResult (US1 - T008-T017).

This module tests that NLSQResult properly delegates statistical properties
and methods to the native NLSQ 0.6.0 CurveFitResult.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from xpcsviewer.fitting.results import NLSQResult


class TestNLSQResultDelegation:
    """Tests for NLSQResult property delegation to native_result."""

    @pytest.fixture
    def mock_native_result(self) -> MagicMock:
        """Create a mock CurveFitResult with expected attributes."""
        mock = MagicMock()
        # Statistical properties
        mock.r_squared = 0.95
        mock.adj_r_squared = 0.94
        mock.rmse = 0.05
        mock.mae = 0.03
        mock.aic = 100.5
        mock.bic = 105.2
        mock.residuals = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        mock.predictions = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        mock.pcov = np.array([[0.01, 0.001], [0.001, 0.02]])
        mock.confidence_intervals = {
            "tau": (0.9, 1.1),
            "baseline": (0.95, 1.05),
        }
        mock.prediction_interval.return_value = (
            np.array([0.8, 0.9, 1.0]),
            np.array([1.2, 1.3, 1.4]),
        )
        # Diagnostics
        mock.diagnostics = MagicMock()
        mock.diagnostics.status = "healthy"
        mock.diagnostics.health_score = 95
        mock.diagnostics.identifiability = MagicMock()
        mock.diagnostics.identifiability.condition_number = 10.5
        return mock

    @pytest.fixture
    def result_with_native(self, mock_native_result: MagicMock) -> NLSQResult:
        """Create NLSQResult with mock native_result."""
        result = NLSQResult(
            params={"tau": 1.0, "baseline": 1.0},
            chi_squared=1.0,
            converged=True,
            native_result=mock_native_result,
            _param_names=["tau", "baseline"],
        )
        # Set legacy covariance for backward compat tests
        result._covariance = np.eye(2)
        return result

    # T009: test_r_squared_delegation
    def test_r_squared_delegation(self, result_with_native: NLSQResult) -> None:
        """Test r_squared delegates to native_result."""
        assert result_with_native.r_squared == 0.95

    # T010: test_adj_r_squared_delegation
    def test_adj_r_squared_delegation(self, result_with_native: NLSQResult) -> None:
        """Test adj_r_squared delegates to native_result."""
        assert result_with_native.adj_r_squared == 0.94

    # T011: test_rmse_mae_delegation
    def test_rmse_delegation(self, result_with_native: NLSQResult) -> None:
        """Test rmse delegates to native_result."""
        assert result_with_native.rmse == 0.05

    def test_mae_delegation(self, result_with_native: NLSQResult) -> None:
        """Test mae delegates to native_result."""
        assert result_with_native.mae == 0.03

    # T012: test_aic_bic_delegation
    def test_aic_delegation(self, result_with_native: NLSQResult) -> None:
        """Test aic delegates to native_result."""
        assert result_with_native.aic == 100.5

    def test_bic_delegation(self, result_with_native: NLSQResult) -> None:
        """Test bic delegates to native_result."""
        assert result_with_native.bic == 105.2

    # T013: test_residuals_predictions_delegation
    def test_residuals_delegation(self, result_with_native: NLSQResult) -> None:
        """Test residuals delegates to native_result and returns numpy array."""
        residuals = result_with_native.residuals
        assert isinstance(residuals, np.ndarray)
        np.testing.assert_array_almost_equal(
            residuals, [0.01, -0.02, 0.015, -0.01, 0.02]
        )

    def test_predictions_delegation(self, result_with_native: NLSQResult) -> None:
        """Test predictions delegates to native_result and returns numpy array."""
        predictions = result_with_native.predictions
        assert isinstance(predictions, np.ndarray)
        np.testing.assert_array_almost_equal(predictions, [1.0, 1.1, 1.2, 1.3, 1.4])

    # T014: test_confidence_intervals_delegation
    def test_confidence_intervals_delegation(
        self, result_with_native: NLSQResult
    ) -> None:
        """Test confidence_intervals property delegates to native_result."""
        ci = result_with_native.confidence_intervals
        assert ci == {"tau": (0.9, 1.1), "baseline": (0.95, 1.05)}

    def test_get_confidence_interval_single_param(
        self, result_with_native: NLSQResult
    ) -> None:
        """Test get_confidence_interval method for single parameter."""
        lower, upper = result_with_native.get_confidence_interval("tau")
        assert lower == 0.9
        assert upper == 1.1

    # T015: test_prediction_interval_delegation
    def test_get_prediction_interval_delegation(
        self, result_with_native: NLSQResult, mock_native_result: MagicMock
    ) -> None:
        """Test get_prediction_interval delegates to native_result."""
        x_new = np.array([1.0, 2.0, 3.0])
        lower, upper = result_with_native.get_prediction_interval(x_new, alpha=0.05)

        # Verify delegation
        mock_native_result.prediction_interval.assert_called_once()
        call_args = mock_native_result.prediction_interval.call_args
        # Check x argument (passed as keyword)
        np.testing.assert_array_equal(call_args.kwargs["x"], x_new)
        assert call_args.kwargs["alpha"] == 0.05

        # Verify return type
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)

    # T016: test_diagnostics_delegation
    def test_diagnostics_property(self, result_with_native: NLSQResult) -> None:
        """Test diagnostics property returns native diagnostics."""
        diagnostics = result_with_native.diagnostics
        assert diagnostics.status == "healthy"
        assert diagnostics.health_score == 95

    def test_is_healthy_property(self, result_with_native: NLSQResult) -> None:
        """Test is_healthy property."""
        assert result_with_native.is_healthy is True

    def test_health_score_property(self, result_with_native: NLSQResult) -> None:
        """Test health_score property."""
        assert result_with_native.health_score == 95

    def test_condition_number_property(self, result_with_native: NLSQResult) -> None:
        """Test condition_number property from diagnostics."""
        assert result_with_native.condition_number == 10.5

    # T017: test_backward_compatibility
    def test_backward_compat_params(self, result_with_native: NLSQResult) -> None:
        """Test params dict still accessible."""
        assert result_with_native.params == {"tau": 1.0, "baseline": 1.0}

    def test_backward_compat_converged(self, result_with_native: NLSQResult) -> None:
        """Test converged flag still accessible."""
        assert result_with_native.converged is True

    def test_backward_compat_chi_squared(self, result_with_native: NLSQResult) -> None:
        """Test chi_squared still accessible."""
        assert result_with_native.chi_squared == 1.0

    def test_backward_compat_pcov_valid(self, result_with_native: NLSQResult) -> None:
        """Test pcov_valid still accessible."""
        assert result_with_native.pcov_valid is True

    def test_backward_compat_covariance(
        self, result_with_native: NLSQResult, mock_native_result: MagicMock
    ) -> None:
        """Test covariance matrix delegates to native pcov when available."""
        # When native_result is present, covariance delegates to pcov
        np.testing.assert_array_almost_equal(
            result_with_native.covariance, mock_native_result.pcov
        )

    def test_backward_compat_get_param_uncertainty(
        self, result_with_native: NLSQResult, mock_native_result: MagicMock
    ) -> None:
        """Test get_param_uncertainty() still works using native pcov."""
        uncertainty = result_with_native.get_param_uncertainty("tau")
        # Uses native_result.pcov which is [[0.01, 0.001], [0.001, 0.02]]
        # sqrt(0.01) = 0.1
        assert np.isclose(uncertainty, 0.1)


class TestNLSQResultWithoutNativeResult:
    """Tests for NLSQResult when native_result is None (backward compat)."""

    @pytest.fixture
    def result_without_native(self) -> NLSQResult:
        """Create legacy NLSQResult without native_result."""
        result = NLSQResult(
            params={"tau": 1.0, "baseline": 1.0},
            chi_squared=1.05,
            converged=True,
        )
        # Set legacy fields via setters
        result.covariance = np.eye(2) * 0.01
        result.residuals = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        result.r_squared = 0.92
        result.adj_r_squared = 0.91
        result.rmse = 0.06
        result.mae = 0.04
        result.aic = 102.0
        result.bic = 107.0
        return result

    def test_fallback_r_squared(self, result_without_native: NLSQResult) -> None:
        """Test r_squared uses stored value when native_result is None."""
        assert result_without_native.r_squared == 0.92

    def test_fallback_residuals(self, result_without_native: NLSQResult) -> None:
        """Test residuals returns stored array when native_result is None."""
        np.testing.assert_array_almost_equal(
            result_without_native.residuals, [0.01, -0.02, 0.015, -0.01, 0.02]
        )

    def test_diagnostics_none(self, result_without_native: NLSQResult) -> None:
        """Test diagnostics returns None when native_result is None."""
        assert result_without_native.diagnostics is None

    def test_is_healthy_default_true(self, result_without_native: NLSQResult) -> None:
        """Test is_healthy defaults to True when no diagnostics."""
        assert result_without_native.is_healthy is True

    def test_health_score_default(self, result_without_native: NLSQResult) -> None:
        """Test health_score returns 100 when no diagnostics."""
        assert result_without_native.health_score == 100


class TestNLSQResultCovariance:
    """Tests for covariance matrix handling."""

    def test_covariance_from_native(self) -> None:
        """Test covariance property delegates to native pcov."""
        mock_native = MagicMock()
        mock_native.pcov = np.array([[0.02, 0.005], [0.005, 0.03]])

        result = NLSQResult(
            params={"a": 1.0, "b": 2.0},
            chi_squared=1.0,
            converged=True,
            native_result=mock_native,
        )

        cov = result.covariance
        np.testing.assert_array_almost_equal(cov, [[0.02, 0.005], [0.005, 0.03]])

    def test_covariance_fallback_to_legacy(self) -> None:
        """Test covariance falls back to _covariance when no native_result."""
        result = NLSQResult(
            params={"a": 1.0, "b": 2.0},
            chi_squared=1.0,
            converged=True,
        )
        result.covariance = np.array([[0.01, 0.0], [0.0, 0.02]])

        cov = result.covariance
        np.testing.assert_array_almost_equal(cov, [[0.01, 0.0], [0.0, 0.02]])
