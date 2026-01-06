"""Tests for pcov validation (T092).

Tests for validate_pcov() function that validates covariance matrices
for finite values and positive semi-definiteness (FR-021).
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.visualization import validate_pcov


class TestValidatePcov:
    """Tests for validate_pcov function."""

    def test_valid_covariance_matrix(self) -> None:
        """Test validation passes for valid covariance matrix."""
        pcov = np.array([[1.0, 0.2], [0.2, 1.0]])
        is_valid, message = validate_pcov(pcov)
        assert is_valid is True
        assert "valid" in message.lower()

    def test_none_covariance(self) -> None:
        """Test validation fails for None covariance."""
        is_valid, message = validate_pcov(None)
        assert is_valid is False
        assert "None" in message

    def test_inf_values(self) -> None:
        """Test validation fails for inf values."""
        pcov = np.array([[1.0, np.inf], [np.inf, 1.0]])
        is_valid, message = validate_pcov(pcov)
        assert is_valid is False
        assert "inf" in message.lower() or "nan" in message.lower()

    def test_nan_values(self) -> None:
        """Test validation fails for nan values."""
        pcov = np.array([[1.0, np.nan], [np.nan, 1.0]])
        is_valid, message = validate_pcov(pcov)
        assert is_valid is False
        assert "inf" in message.lower() or "nan" in message.lower()

    def test_non_positive_semidefinite(self) -> None:
        """Test validation fails for non-positive semi-definite matrix."""
        # Matrix with negative eigenvalue
        pcov = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        is_valid, message = validate_pcov(pcov)
        assert is_valid is False
        assert "positive semi-definite" in message.lower()

    def test_identity_matrix(self) -> None:
        """Test validation passes for identity matrix."""
        pcov = np.eye(3)
        is_valid, message = validate_pcov(pcov)
        assert is_valid is True

    def test_diagonal_matrix(self) -> None:
        """Test validation passes for diagonal matrix with positive elements."""
        pcov = np.diag([1.0, 2.0, 3.0])
        is_valid, message = validate_pcov(pcov)
        assert is_valid is True

    def test_zero_matrix(self) -> None:
        """Test validation passes for zero matrix (borderline positive semi-definite)."""
        pcov = np.zeros((2, 2))
        is_valid, message = validate_pcov(pcov)
        assert is_valid is True

    def test_1x1_matrix(self) -> None:
        """Test validation works for 1x1 matrix."""
        pcov = np.array([[0.5]])
        is_valid, message = validate_pcov(pcov)
        assert is_valid is True

    def test_large_valid_matrix(self) -> None:
        """Test validation works for larger matrix."""
        # Generate a valid covariance matrix using A @ A.T
        rng = np.random.default_rng(42)
        a = rng.random((5, 5))
        pcov = a @ a.T
        is_valid, message = validate_pcov(pcov)
        assert is_valid is True
