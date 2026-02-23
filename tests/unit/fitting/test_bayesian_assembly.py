"""Tests for Bayesian assembly — converting per-Q FitResults to fit_summary."""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.bayesian_assembly import (
    _DOUBLE_EXP_NPARAMS,
    _SINGLE_EXP_NPARAMS,
    _extract_double_exp_params,
    _extract_single_exp_params,
    assemble_fit_summary,
)
from xpcsviewer.fitting.results import FitResult


def _make_fit_result(params: dict[str, float], n_samples: int = 100) -> FitResult:
    """Create a FitResult with constant samples (mean = value, std ≈ 0)."""
    rng = np.random.default_rng(42)
    samples = {
        name: np.full(n_samples, val) + rng.normal(0, abs(val) * 0.01, n_samples)
        for name, val in params.items()
    }
    return FitResult(samples=samples)


def _identity_model(x, tau, baseline, contrast):
    """Simplified single-exp model for testing."""
    return baseline + contrast * np.exp(-2 * x / max(tau, 1e-30))


def _identity_double_model(x, tau1, tau2, baseline, contrast1, contrast2):
    """Simplified double-exp model for testing."""
    return (
        baseline
        + contrast1 * np.exp(-2 * x / max(tau1, 1e-30))
        + contrast2 * np.exp(-2 * x / max(tau2, 1e-30))
    )


class TestExtractSingleExpParams:
    """Test _extract_single_exp_params mapping."""

    def test_param_count(self):
        fr = _make_fit_result({"contrast": 0.3, "tau": 1.5, "baseline": 1.0})
        vals, errs = _extract_single_exp_params(fr)
        assert vals.shape == (_SINGLE_EXP_NPARAMS,)
        assert errs.shape == (_SINGLE_EXP_NPARAMS,)

    def test_param_mapping(self):
        fr = _make_fit_result({"contrast": 0.3, "tau": 1.5, "baseline": 1.0})
        vals, errs = _extract_single_exp_params(fr)
        # idx 0 = contrast, idx 1 = tau, idx 2 = stretching (1.0), idx 3 = baseline
        assert vals[0] == pytest.approx(0.3, abs=0.01)
        assert vals[1] == pytest.approx(1.5, abs=0.05)
        assert vals[2] == 1.0  # fixed stretching
        assert vals[3] == pytest.approx(1.0, abs=0.01)

    def test_stretching_error_is_zero(self):
        fr = _make_fit_result({"contrast": 0.3, "tau": 1.5, "baseline": 1.0})
        _, errs = _extract_single_exp_params(fr)
        assert errs[2] == 0.0


class TestExtractDoubleExpParams:
    """Test _extract_double_exp_params mapping."""

    def test_param_count(self):
        fr = _make_fit_result(
            {
                "contrast1": 0.2,
                "tau1": 1.0,
                "baseline": 1.0,
                "tau2": 5.0,
                "contrast2": 0.1,
            }
        )
        vals, errs = _extract_double_exp_params(fr)
        assert vals.shape == (_DOUBLE_EXP_NPARAMS,)
        assert errs.shape == (_DOUBLE_EXP_NPARAMS,)

    def test_param_mapping(self):
        fr = _make_fit_result(
            {
                "contrast1": 0.2,
                "tau1": 1.0,
                "baseline": 1.0,
                "tau2": 5.0,
                "contrast2": 0.1,
            }
        )
        vals, _ = _extract_double_exp_params(fr)
        assert vals[0] == pytest.approx(0.2, abs=0.01)  # contrast1
        assert vals[1] == pytest.approx(1.0, abs=0.05)  # tau1
        assert vals[2] == 1.0  # stretching (fixed)
        assert vals[3] == pytest.approx(1.0, abs=0.01)  # baseline
        assert vals[4] == pytest.approx(5.0, abs=0.1)  # tau2
        assert vals[5] == pytest.approx(0.1, abs=0.01)  # contrast2
        assert vals[6] == 1.0  # stretching (fixed)


class TestAssembleFitSummary:
    """Test assemble_fit_summary end-to-end."""

    @pytest.fixture()
    def single_exp_results(self):
        """Create results for 5 Q-bins, one failed."""
        results = {}
        for i in range(5):
            if i == 2:
                results[i] = None  # failed Q-bin
            else:
                results[i] = _make_fit_result(
                    {"contrast": 0.3, "tau": 1.0 + i * 0.5, "baseline": 1.0}
                )
        return results

    def test_fit_val_shape(self, single_exp_results):
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        assert summary["fit_val"].shape == (5, 2, _SINGLE_EXP_NPARAMS)

    def test_fit_line_shape(self, single_exp_results):
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        n_fit_x = len(summary["fit_x"])
        assert summary["fit_line"].shape == (5, n_fit_x)

    def test_failed_q_bin_is_nan(self, single_exp_results):
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        # Q-index 2 failed — should be NaN, not zero
        assert np.all(np.isnan(summary["fit_val"][2]))
        assert np.all(np.isnan(summary["fit_line"][2]))

    def test_successful_q_bin_has_tau(self, single_exp_results):
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        # Q-index 0 should have tau ≈ 1.0 at fit_val[0, 0, 1]
        assert summary["fit_val"][0, 0, 1] == pytest.approx(1.0, abs=0.05)

    def test_tauq_reads_tau_values(self, single_exp_results):
        """Verify tau-q pipeline can read tau values via fit_val[:, 0, 1]."""
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        tau_values = summary["fit_val"][:, 0, 1]
        assert tau_values.shape == (5,)
        # Q-index 2 failed → tau is NaN (not 0)
        assert np.isnan(tau_values[2])
        # Others should be positive
        assert all(tau_values[i] > 0 for i in [0, 1, 3, 4])

    def test_failed_mask_present(self, single_exp_results):
        """failed_mask must be in summary with correct shape and values."""
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        assert "failed_mask" in summary
        mask = summary["failed_mask"]
        assert mask.shape == (5,)
        assert mask.dtype == bool
        # Q-index 2 failed
        assert mask[2] is np.True_
        # Others succeeded
        for i in [0, 1, 3, 4]:
            assert mask[i] is np.False_

    def test_q_val_preserved(self, single_exp_results):
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results, q_arr, t_el, "single", _identity_model
        )
        np.testing.assert_array_equal(summary["q_val"], q_arr)

    def test_metadata_fields(self, single_exp_results):
        q_arr = np.linspace(0.001, 0.01, 5)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            single_exp_results,
            q_arr,
            t_el,
            "single",
            _identity_model,
            label="test_file",
        )
        assert summary["fit_func"] == "single"
        assert summary["label"] == "test_file"
        assert "fit_x" in summary
        assert "fit_line" in summary

    def test_empty_results(self):
        """Empty results dict should produce all-NaN arrays."""
        q_arr = np.linspace(0.001, 0.01, 3)
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary({}, q_arr, t_el, "single", _identity_model)
        assert summary["fit_val"].shape == (3, 2, _SINGLE_EXP_NPARAMS)
        assert np.all(np.isnan(summary["fit_val"]))

    def test_double_exp_assembly(self):
        results = {
            0: _make_fit_result(
                {
                    "contrast1": 0.2,
                    "tau1": 1.0,
                    "baseline": 1.0,
                    "tau2": 5.0,
                    "contrast2": 0.1,
                }
            )
        }
        q_arr = np.array([0.005])
        t_el = np.logspace(-5, 1, 50)
        summary = assemble_fit_summary(
            results, q_arr, t_el, "double", _identity_double_model
        )
        assert summary["fit_val"].shape == (1, 2, _DOUBLE_EXP_NPARAMS)
        # tau1 at index 1
        assert summary["fit_val"][0, 0, 1] == pytest.approx(1.0, abs=0.05)
        # tau2 at index 4
        assert summary["fit_val"][0, 0, 4] == pytest.approx(5.0, abs=0.1)
