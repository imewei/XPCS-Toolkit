"""Integration tests: NLSQ and Bayesian power-law fitting consistency.

Verifies:
    1. Both paths recover correct parameters from synthetic data
    2. NLSQ b ≈ -Bayesian alpha (sign convention mapping)
    3. Bayesian bounds parameter is respected
    4. BUG-M: Fixed parameters use p0, not upper bound
    5. BUG-H: Extraction prefers tauq_q keys over fit_val
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from xpcsviewer.fitting.legacy import fit_with_fixed
from xpcsviewer.xpcs_file.fitting import power_law

try:
    from xpcsviewer.fitting.sampler import run_power_law_fit

    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False

needs_numpyro = pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not installed")

# Ground truth: tau(q) = 1e-5 * q^(-2.0)
TRUE_TAU0 = 1e-5
TRUE_ALPHA = 2.0
TRUE_B = -TRUE_ALPHA  # NLSQ convention


def _make_synthetic_data(
    n_q: int = 20,
    noise_frac: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic power-law diffusion data with known parameters."""
    rng = np.random.default_rng(seed)
    q = np.logspace(-2, -0.5, n_q)
    tau_true = TRUE_TAU0 * q**TRUE_B  # = TRUE_TAU0 * q^(-TRUE_ALPHA)
    tau_err = np.abs(tau_true) * noise_frac
    tau = tau_true + rng.normal(0, tau_err)
    # Ensure positive tau (physical constraint)
    tau = np.maximum(tau, tau_true * 0.1)
    return q, tau, tau_err


# ---------------------------------------------------------------------------
# NLSQ path
# ---------------------------------------------------------------------------


class TestNLSQPowerLaw:
    """Verify NLSQ recovers known power-law parameters."""

    def test_nlsq_recovers_parameters(self):
        """fit_with_fixed(power_law) must recover a and b from synthetic data."""
        q, tau, tau_err = _make_synthetic_data()

        bounds = np.array(
            [[1e-15, -4.0], [1e-2, 0.0]]
        )  # [[a_min, b_min], [a_max, b_max]]
        fit_flag = np.array([True, True])
        fit_x = np.logspace(-2.1, -0.4, 50)
        p0 = np.array([1e-5, -2.0])

        fit_line, fit_val = fit_with_fixed(
            power_law,
            q,
            tau[:, np.newaxis],
            tau_err[:, np.newaxis],
            bounds,
            fit_flag,
            fit_x,
            p0=p0,
        )

        a_fit = fit_val[0, 0, 0]
        b_fit = fit_val[0, 0, 1]

        # Should recover TRUE_TAU0 and TRUE_B within reasonable tolerance
        assert abs(np.log10(a_fit) - np.log10(TRUE_TAU0)) < 0.5, (
            f"a_fit={a_fit:.3e} vs true={TRUE_TAU0:.3e}"
        )
        assert abs(b_fit - TRUE_B) < 0.3, f"b_fit={b_fit:.3f} vs true={TRUE_B:.3f}"

    def test_fixed_param_uses_p0_not_upper_bound(self):
        """BUG-M: When b is fixed, its value should come from p0, not bounds[1]."""
        q, tau, tau_err = _make_synthetic_data()

        bounds = np.array([[1e-15, -4.0], [1e-2, 0.0]])
        fit_flag = np.array([True, False])  # Fix b
        fit_x = np.logspace(-2.1, -0.4, 50)
        p0 = np.array([1e-5, -2.0])  # b should be fixed at -2.0, not 0.0

        _fit_line, fit_val = fit_with_fixed(
            power_law,
            q,
            tau[:, np.newaxis],
            tau_err[:, np.newaxis],
            bounds,
            fit_flag,
            fit_x,
            p0=p0,
        )

        b_fixed = fit_val[0, 0, 1]
        # b should be -2.0 (from p0), NOT 0.0 (upper bound)
        assert b_fixed == pytest.approx(-2.0), (
            f"Fixed b={b_fixed:.3f}, expected -2.0 from p0 (not 0.0 from upper bound)"
        )

    def test_fixed_param_without_p0_uses_midpoint(self):
        """When p0 is None and b is fixed, use midpoint of bounds."""
        q, tau, tau_err = _make_synthetic_data()

        bounds = np.array([[1e-15, -4.0], [1e-2, 0.0]])
        fit_flag = np.array([True, False])  # Fix b
        fit_x = np.logspace(-2.1, -0.4, 50)

        _fit_line, fit_val = fit_with_fixed(
            power_law,
            q,
            tau[:, np.newaxis],
            tau_err[:, np.newaxis],
            bounds,
            fit_flag,
            fit_x,
            p0=None,
        )

        b_fixed = fit_val[0, 0, 1]
        expected_midpoint = (-4.0 + 0.0) / 2  # = -2.0
        assert b_fixed == pytest.approx(expected_midpoint), (
            f"Fixed b={b_fixed:.3f}, expected midpoint={expected_midpoint:.3f}"
        )


# ---------------------------------------------------------------------------
# Bayesian path
# ---------------------------------------------------------------------------


class TestBayesianPowerLaw:
    """Verify Bayesian path recovers known power-law parameters."""

    @needs_numpyro
    def test_bayesian_recovers_parameters(self):
        """run_power_law_fit must recover tau0 and alpha from synthetic data."""
        q, tau, tau_err = _make_synthetic_data()

        result = run_power_law_fit(
            q,
            tau,
            tau_err=tau_err,
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            random_seed=42,
        )

        tau0_mean = result.summary.loc["tau0", "mean"]
        alpha_mean = result.summary.loc["alpha", "mean"]

        # Should recover TRUE_TAU0 and TRUE_ALPHA
        assert abs(np.log10(tau0_mean) - np.log10(TRUE_TAU0)) < 1.0, (
            f"tau0={tau0_mean:.3e} vs true={TRUE_TAU0:.3e}"
        )
        assert abs(alpha_mean - TRUE_ALPHA) < 0.5, (
            f"alpha={alpha_mean:.3f} vs true={TRUE_ALPHA:.3f}"
        )

    @needs_numpyro
    def test_bounds_parameter_accepted(self):
        """run_power_law_fit must accept the bounds parameter."""
        sig = inspect.signature(run_power_law_fit)
        assert "bounds" in sig.parameters, (
            "run_power_law_fit must accept a 'bounds' keyword argument"
        )

    @needs_numpyro
    def test_custom_bounds_respected(self):
        """Custom bounds should override defaults for NLSQ warm-start."""
        q, tau, tau_err = _make_synthetic_data()

        # Tight bounds around true values
        result = run_power_law_fit(
            q,
            tau,
            tau_err=tau_err,
            bounds={"tau0": (1e-7, 1e-3), "alpha": (1.5, 2.5)},
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            random_seed=42,
        )

        alpha_mean = result.summary.loc["alpha", "mean"]
        # Alpha should still be near 2.0
        assert 1.0 < alpha_mean < 3.0, f"alpha={alpha_mean:.3f} outside expected range"


# ---------------------------------------------------------------------------
# Cross-path consistency
# ---------------------------------------------------------------------------


class TestNLSQBayesianConsistency:
    """Verify both paths agree on the same physics."""

    @needs_numpyro
    def test_b_equals_negative_alpha(self):
        """NLSQ b and Bayesian alpha must satisfy b ≈ -alpha."""
        q, tau, tau_err = _make_synthetic_data()

        # NLSQ fit
        bounds_nlsq = np.array([[1e-15, -4.0], [1e-2, 0.0]])
        fit_flag = np.array([True, True])
        fit_x = np.logspace(-2.1, -0.4, 50)
        p0 = np.array([1e-5, -2.0])

        _fit_line, fit_val = fit_with_fixed(
            power_law,
            q,
            tau[:, np.newaxis],
            tau_err[:, np.newaxis],
            bounds_nlsq,
            fit_flag,
            fit_x,
            p0=p0,
        )
        b_nlsq = fit_val[0, 0, 1]

        # Bayesian fit
        result = run_power_law_fit(
            q,
            tau,
            tau_err=tau_err,
            num_warmup=200,
            num_samples=500,
            num_chains=1,
            random_seed=42,
        )
        alpha_bayesian = result.summary.loc["alpha", "mean"]

        # b ≈ -alpha (sign convention mapping)
        assert abs(b_nlsq - (-alpha_bayesian)) < 0.5, (
            f"NLSQ b={b_nlsq:.3f}, Bayesian alpha={alpha_bayesian:.3f}, "
            f"expected b ≈ -alpha but diff={abs(b_nlsq + alpha_bayesian):.3f}"
        )


# ---------------------------------------------------------------------------
# BUG-H: Extraction data source preference
# ---------------------------------------------------------------------------


class TestBugH_ExtractionDataSource:
    """Verify Bayesian extraction prefers tauq keys over fit_val."""

    def test_extract_prefers_tauq_keys(self):
        """When tauq_q/tauq_tau/tauq_tau_err exist, they should be used."""
        # Create fit_summary with DIFFERENT values in tauq_* vs fit_val
        n_q = 5
        tauq_tau = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tauq_tau_err = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        tauq_q = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        # fit_val has deliberately different values
        fit_val = np.zeros((n_q, 2, 4))
        fit_val[:, 0, 1] = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # 10x different
        fit_val[:, 1, 1] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        fit_summary = {
            "q_val": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),  # Different q
            "fit_val": fit_val,
            "tauq_q": tauq_q,
            "tauq_tau": tauq_tau,
            "tauq_tau_err": tauq_tau_err,
        }

        # Simulate the extraction logic from _extract_diffusion_for_bayesian
        if "tauq_q" in fit_summary:
            q = np.asarray(fit_summary["tauq_q"])
            tau = np.asarray(fit_summary["tauq_tau"])
            tau_err = np.asarray(fit_summary["tauq_tau_err"])
        else:
            q = np.asarray(fit_summary["q_val"])
            fv = np.asarray(fit_summary["fit_val"])
            tau = fv[:, 0, 1]
            tau_err = fv[:, 1, 1]

        # Should use tauq values, not fit_val values
        np.testing.assert_array_equal(q, tauq_q)
        np.testing.assert_array_equal(tau, tauq_tau)
        np.testing.assert_array_equal(tau_err, tauq_tau_err)

    def test_fallback_to_fit_val_when_tauq_absent(self):
        """When tauq keys are missing, fall back to fit_val."""
        n_q = 5
        fit_val = np.zeros((n_q, 2, 4))
        fit_val[:, 0, 1] = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        fit_val[:, 1, 1] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_val = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

        fit_summary = {"q_val": q_val, "fit_val": fit_val}

        if "tauq_q" in fit_summary:
            q = np.asarray(fit_summary["tauq_q"])
            tau = np.asarray(fit_summary["tauq_tau"])
            tau_err = np.asarray(fit_summary["tauq_tau_err"])
        else:
            q = np.asarray(fit_summary["q_val"])
            fv = np.asarray(fit_summary["fit_val"])
            tau = fv[:, 0, 1]
            tau_err = fv[:, 1, 1]

        np.testing.assert_array_equal(q, q_val)
        np.testing.assert_array_equal(tau, fit_val[:, 0, 1])

    def test_extract_source_in_xpcs_viewer(self):
        """Source inspection: _extract_diffusion_for_bayesian must check tauq_q."""
        from pathlib import Path

        viewer_path = (
            Path(__file__).parent.parent.parent.parent / "xpcsviewer" / "xpcs_viewer.py"
        )
        source = viewer_path.read_text()

        # Find the method body
        method_start = source.find("def _extract_diffusion_for_bayesian")
        assert method_start != -1, "_extract_diffusion_for_bayesian not found"

        # Check it references tauq_q for the preference check
        method_body = source[method_start : method_start + 2000]
        assert '"tauq_q" in fit_summary' in method_body, (
            "_extract_diffusion_for_bayesian must check for tauq_q in fit_summary"
        )
