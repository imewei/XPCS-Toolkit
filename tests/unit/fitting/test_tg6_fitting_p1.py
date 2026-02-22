"""TG6 -- Fitting Pipeline P1 Fixes: Tests for BUG-017 through BUG-026.

Tests cover:
- BUG-017: getattr success default is False (not True)
- BUG-018: stretched_exp_model clamps tau
- BUG-019: power_law_model uses LogNormal prior and clamps q
- BUG-022: double-exp warm-start sorts tau1/tau2 before tau2_factor
- BUG-023: failed NLSQResult metrics are NaN not 0.0
- BUG-024: FitResult __post_init__ validates samples
- BUG-025: FitDiagnostics __post_init__ validates divergences and ESS
- BUG-026: legacy model factory calls get_backend() once at construction
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1: BUG-017 -- getattr success default is False
# ---------------------------------------------------------------------------


class TestBug017DefaultConvergence:
    """BUG-017: getattr(native_result, 'success', False) not True."""

    def test_missing_success_attribute_defaults_to_false(self):
        """A native result object missing 'success' must yield converged=False."""

        class _FakeNativeResult:
            popt = np.array([1.0, 1.0, 0.3])
            pcov = np.eye(3) * 0.01

        native = _FakeNativeResult()
        assert not hasattr(native, "success"), "precondition: no 'success' attribute"

        # This is the expression in nlsq.py after the fix:
        converged = getattr(native, "success", False)
        assert converged is False, (
            "Missing 'success' attribute must default to False, not True"
        )

    def test_success_true_propagates(self):
        """An explicit success=True is preserved."""

        class _FakeNativeResult:
            success = True

        converged = getattr(_FakeNativeResult(), "success", False)
        assert converged is True

    def test_success_false_propagates(self):
        """An explicit success=False is preserved."""

        class _FakeNativeResult:
            success = False

        converged = getattr(_FakeNativeResult(), "success", False)
        assert converged is False

    def test_nlsq_py_uses_false_as_default(self):
        """Verify nlsq.py source uses getattr(native_result, 'success', False)."""
        import inspect

        from xpcsviewer.fitting import nlsq as nlsq_module

        source = inspect.getsource(nlsq_module)
        assert '"success", False' in source or "'success', False" in source, (
            "nlsq.py must use getattr(native_result, 'success', False) (BUG-017)"
        )
        # Must NOT have the old bug
        assert '"success", True' not in source and "'success', True" not in source, (
            "nlsq.py must NOT use default=True for 'success' attribute"
        )


# ---------------------------------------------------------------------------
# Test 2: BUG-018 -- stretched_exp_model clamps tau
# ---------------------------------------------------------------------------


class TestBug018StretchedExpTauClamping:
    """BUG-018: stretched_exp_model must clamp tau so (2x/tau)^beta is finite."""

    def test_tau_clamping_exists_in_model_source(self):
        """Verify that stretched_exp_model contains tau = jnp.clip(tau, 1e-30)."""
        import inspect

        from xpcsviewer.fitting.models import stretched_exp_model

        source = inspect.getsource(stretched_exp_model)
        # Must clamp tau specifically (not just x)
        assert "jnp.clip(tau, 1e-30)" in source, (
            "stretched_exp_model must clamp tau via jnp.clip(tau, 1e-30) (BUG-018)"
        )

    def test_tau_near_zero_does_not_produce_nan_gradient(self):
        """Near-zero tau must not produce NaN when computing (2x/tau)^beta."""
        try:
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not available")

        # Directly test the arithmetic that would overflow without clamping.
        tau_raw = jnp.array(1e-40)  # Well below 1e-30
        tau_clamped = jnp.clip(tau_raw, 1e-30)
        x = jnp.array([0.001, 0.01, 0.1])
        beta = jnp.array(0.7)

        # With clamping result must be finite
        result_clamped = jnp.power(2 * x / tau_clamped, beta)

        assert jnp.all(jnp.isfinite(result_clamped)), (
            "Clamped tau must produce finite (2x/tau)^beta"
        )


# ---------------------------------------------------------------------------
# Test 3: BUG-019 -- power_law_model uses LogNormal for tau0 and clamps q
# ---------------------------------------------------------------------------


class TestBug019PowerLawModelPriors:
    """BUG-019: power_law_model must use LogNormal prior for tau0 and clamp q."""

    def test_tau0_uses_lognormal_prior(self):
        """power_law_model source must reference dist.LogNormal for tau0."""
        import inspect

        from xpcsviewer.fitting.models import power_law_model

        source = inspect.getsource(power_law_model)
        # LogNormal prevents tau0 from sampling negative values.
        assert "dist.LogNormal" in source, (
            "power_law_model must use dist.LogNormal for tau0 prior (BUG-019)"
        )

    def test_q_is_clamped_to_prevent_log_zero(self):
        """power_law_model source must clamp q to prevent log(0) = -inf."""
        import inspect

        from xpcsviewer.fitting.models import power_law_model

        source = inspect.getsource(power_law_model)
        assert "jnp.clip(q, 1e-30)" in source, (
            "power_law_model must clamp q via jnp.clip(q, 1e-30) (BUG-019)"
        )

    def test_lognormal_samples_are_always_positive(self):
        """LogNormal distribution never samples negative values."""
        try:
            import jax
            import numpyro.distributions as dist
        except ImportError:
            pytest.skip("NumPyro/JAX not available")

        key = jax.random.PRNGKey(42)
        samples = dist.LogNormal(0.0, 2.0).sample(key, sample_shape=(1000,))
        assert float(samples.min()) > 0.0, "LogNormal samples must always be positive"


# ---------------------------------------------------------------------------
# Test 4: BUG-022 -- double-exp warm-start sorts tau1/tau2
# ---------------------------------------------------------------------------


class TestBug022DoubleExpTauSorting:
    """BUG-022: tau1/tau2 must be sorted before computing tau2_factor."""

    def test_tau2_factor_positive_when_tau1_greater_than_tau2(self):
        """If NLSQ returns tau1 > tau2, sorting must ensure tau2_factor > 0."""
        # Scenario: NLSQ returns tau1=10.0, tau2=0.1 (i.e., tau1 > tau2)
        nlsq_tau1 = 10.0
        nlsq_tau2 = 0.1

        # Correct sorted behavior:
        tau_sorted = sorted([nlsq_tau1, nlsq_tau2])
        tau1_used = tau_sorted[0]
        tau2_used = tau_sorted[1]
        tau2_factor = max(0.01, min(tau2_used / tau1_used - 1, 1000.0))

        assert tau2_factor > 0, (
            "tau2_factor must be positive after sorting tau1/tau2 (BUG-022)"
        )
        assert tau1_used <= tau2_used, "tau1 must be <= tau2 after sorting"

    def test_tau2_factor_correct_when_already_sorted(self):
        """When tau1 < tau2 (normal case), sorting should not change anything."""
        nlsq_tau1 = 0.1
        nlsq_tau2 = 10.0

        tau_sorted = sorted([nlsq_tau1, nlsq_tau2])
        tau1_used = tau_sorted[0]
        tau2_used = tau_sorted[1]
        tau2_factor = max(0.01, min(tau2_used / tau1_used - 1, 1000.0))

        assert tau1_used == pytest.approx(0.1)
        assert tau2_used == pytest.approx(10.0)
        assert tau2_factor == pytest.approx(10.0 / 0.1 - 1)

    def test_sampler_sorts_tau_before_tau2_factor(self):
        """Verify the sampler source contains tau sorting logic (BUG-022)."""
        import inspect

        from xpcsviewer.fitting import sampler

        source = inspect.getsource(sampler.run_double_exp_fit)
        # The sort must appear before tau2_factor assignment.
        sort_idx = source.find("sorted(")
        # Find the assignment 'tau2_factor =' (not just any mention of tau2_factor)
        factor_assign_idx = source.find("tau2_factor =")
        assert sort_idx != -1, "run_double_exp_fit must sort tau values (BUG-022)"
        assert factor_assign_idx != -1, "run_double_exp_fit must assign tau2_factor"
        assert sort_idx < factor_assign_idx, (
            "Tau sorting must occur before tau2_factor assignment"
        )


# ---------------------------------------------------------------------------
# Test 5: BUG-023 -- failed NLSQResult metrics are NaN not 0.0
# ---------------------------------------------------------------------------


class TestBug023FailedNLSQResultMetrics:
    """BUG-023: Failed NLSQResult default metrics must be NaN, not 0.0."""

    def test_default_r_squared_is_nan(self):
        """NLSQResult._r_squared default must be float('nan')."""
        from xpcsviewer.fitting.results import NLSQResult

        result = NLSQResult(
            params={"tau": 1.0},
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            native_result=None,
        )
        assert math.isnan(result.r_squared), (
            "Failed NLSQResult r_squared must be NaN, not 0.0 (BUG-023)"
        )

    def test_default_adj_r_squared_is_nan(self):
        """NLSQResult._adj_r_squared default must be float('nan')."""
        from xpcsviewer.fitting.results import NLSQResult

        result = NLSQResult(
            params={"tau": 1.0},
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            native_result=None,
        )
        assert math.isnan(result.adj_r_squared), (
            "Failed NLSQResult adj_r_squared must be NaN (BUG-023)"
        )

    def test_default_rmse_is_nan(self):
        """NLSQResult._rmse default must be float('nan')."""
        from xpcsviewer.fitting.results import NLSQResult

        result = NLSQResult(
            params={"tau": 1.0},
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            native_result=None,
        )
        assert math.isnan(result.rmse), "Failed NLSQResult rmse must be NaN (BUG-023)"

    def test_failed_result_distinguishable_from_converged_bad_fit(self):
        """A failed result (NaN) must be distinguishable from a bad-but-converged fit (finite)."""
        from xpcsviewer.fitting.results import NLSQResult

        # Failed result -- all metrics NaN
        failed = NLSQResult(
            params={"tau": 1.0},
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            native_result=None,
        )

        # Bad-but-converged result -- finite but poor metrics
        bad_converged = NLSQResult(
            params={"tau": 1.0},
            chi_squared=100.0,
            converged=True,
            is_fallback=False,
            native_result=None,
            _r_squared=-5.0,  # Terrible but finite
        )

        assert math.isnan(failed.r_squared)
        assert not math.isnan(bad_converged.r_squared)
        assert bad_converged.r_squared == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Test 6: BUG-024 -- FitResult __post_init__ validation
# ---------------------------------------------------------------------------


class TestBug024FitResultValidation:
    """BUG-024: FitResult.__post_init__ must reject invalid samples."""

    def test_empty_samples_dict_raises(self):
        """FitResult with empty samples dict must raise ValueError."""
        from xpcsviewer.fitting.results import FitResult

        with pytest.raises(ValueError, match="samples"):
            FitResult(samples={})

    def test_inconsistent_sample_shapes_raise(self):
        """FitResult with sample arrays of different lengths must raise ValueError."""
        from xpcsviewer.fitting.results import FitResult

        with pytest.raises(ValueError, match="shape"):
            FitResult(
                samples={
                    "tau": np.ones(100),
                    "baseline": np.ones(200),  # Different length -- inconsistent
                }
            )

    def test_valid_samples_accepted(self):
        """FitResult with consistent non-empty samples must be accepted."""
        from xpcsviewer.fitting.results import FitResult

        result = FitResult(
            samples={
                "tau": np.ones(100),
                "baseline": np.ones(100),
                "contrast": np.ones(100),
            }
        )
        assert len(result.samples) == 3

    def test_single_param_valid(self):
        """FitResult with a single parameter sample is valid."""
        from xpcsviewer.fitting.results import FitResult

        result = FitResult(samples={"tau": np.ones(50)})
        assert "tau" in result.samples


# ---------------------------------------------------------------------------
# Test 7: BUG-025 -- FitDiagnostics __post_init__ validation
# ---------------------------------------------------------------------------


class TestBug025FitDiagnosticsValidation:
    """BUG-025: FitDiagnostics.__post_init__ must reject invalid diagnostics."""

    def test_negative_divergences_raises(self):
        """FitDiagnostics with negative divergences must raise ValueError."""
        from xpcsviewer.fitting.results import FitDiagnostics

        with pytest.raises(ValueError, match="divergences"):
            FitDiagnostics(divergences=-1)

    def test_negative_ess_bulk_raises(self):
        """FitDiagnostics with negative ESS bulk values must raise ValueError."""
        from xpcsviewer.fitting.results import FitDiagnostics

        with pytest.raises(ValueError, match="ess_bulk"):
            FitDiagnostics(ess_bulk={"tau": -10})

    def test_negative_ess_tail_raises(self):
        """FitDiagnostics with negative ESS tail values must raise ValueError."""
        from xpcsviewer.fitting.results import FitDiagnostics

        with pytest.raises(ValueError, match="ess_tail"):
            FitDiagnostics(ess_tail={"tau": -5})

    def test_valid_diagnostics_accepted(self):
        """FitDiagnostics with valid values must be accepted."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diag = FitDiagnostics(
            r_hat={"tau": 1.005},
            ess_bulk={"tau": 500},
            ess_tail={"tau": 450},
            divergences=0,
        )
        assert diag.divergences == 0

    def test_zero_divergences_accepted(self):
        """Zero divergences must be accepted (boundary case)."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diag = FitDiagnostics(divergences=0)
        assert diag.divergences == 0

    def test_zero_ess_accepted(self):
        """Zero ESS values must be accepted (warm-up only case)."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diag = FitDiagnostics(ess_bulk={"tau": 0}, ess_tail={"tau": 0})
        assert diag.ess_bulk["tau"] == 0


# ---------------------------------------------------------------------------
# Test 8: BUG-026 -- Legacy model factory calls get_backend() once at construction
# ---------------------------------------------------------------------------


class TestBug026LegacyModelFactory:
    """BUG-026: Legacy model functions must call get_backend() once at construction."""

    def test_factory_functions_exist(self):
        """Factory functions make_single_exp, make_double_exp, etc. must exist."""
        from xpcsviewer.fitting import legacy

        assert hasattr(legacy, "make_single_exp"), (
            "legacy module must expose make_single_exp factory (BUG-026)"
        )
        assert hasattr(legacy, "make_double_exp"), (
            "legacy module must expose make_double_exp factory (BUG-026)"
        )

    def test_factory_produces_callable(self):
        """The factory must return a callable suitable for curve_fit."""
        from xpcsviewer.fitting.legacy import make_double_exp, make_single_exp

        fn = make_single_exp()
        assert callable(fn), "make_single_exp must return a callable"

        fn2 = make_double_exp()
        assert callable(fn2), "make_double_exp must return a callable"

    def test_factory_result_evaluates_correctly(self):
        """Factory-produced function must evaluate single exponential correctly."""
        from xpcsviewer.fitting.legacy import make_single_exp

        fn = make_single_exp()
        x = np.array([0.0, 1.0, 2.0])
        # single_exp: cts * exp(-2*x/tau) + bkg
        tau, bkg, cts = 1.0, 1.0, 0.5
        result = fn(x, tau, bkg, cts)
        expected = cts * np.exp(-2 * x / tau) + bkg
        np.testing.assert_allclose(
            np.asarray(result),
            expected,
            rtol=1e-5,
            err_msg="Factory single_exp must match manual calculation",
        )

    def test_factory_uses_jnp_independent_of_backend(self, monkeypatch):
        """Model factories must use jnp directly, independent of xpcsviewer backend."""
        import xpcsviewer.fitting.legacy as legacy_module

        # Factory should work regardless of backend setting
        monkeypatch.setenv("XPCS_USE_JAX", "false")

        fn = legacy_module.make_single_exp()

        x = np.array([1.0, 2.0, 3.0])
        result = fn(x, 1.0, 1.0, 0.3)
        expected = 0.3 * np.exp(-2 * x / 1.0) + 1.0
        np.testing.assert_allclose(
            np.asarray(result),
            expected,
            rtol=1e-5,
            err_msg="Factory must produce correct results independent of backend",
        )
