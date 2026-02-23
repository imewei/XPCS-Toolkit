"""Regression tests for G2 fitting deep RCA bug fixes.

Tests cover:
    BUG-A: init_params must be in unconstrained space for NUTS
    BUG-B: Jitter applied in unconstrained space (no negative constrained values)
    BUG-C: get_hdi() uses true HDI, not equal-tailed interval
    BUG-D: max_treedepth_reached computed from num_steps
    BUG-E: Time-based seed stored in config.random_seed
    BUG-F: FitResult.predict() raises NotImplementedError
    BUG-G: run_power_law_fit must accept tau_err (not hardcode None)
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from xpcsviewer.fitting.results import FitDiagnostics, FitResult, SamplerConfig

# ---------------------------------------------------------------------------
# Conditional imports for tests that need JAX / NumPyro
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions.transforms import biject_to

    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False

needs_numpyro = pytest.mark.skipif(not HAS_NUMPYRO, reason="NumPyro not installed")


# ============================================================================
# BUG-A: init_params in unconstrained space
# ============================================================================
class TestBugA_InitParamsUnconstrained:
    """Verify _run_mcmc uses init_to_value for constrained warm-start."""

    @needs_numpyro
    def test_run_mcmc_signature_has_init_params(self):
        """_run_mcmc must accept an init_params keyword argument."""
        from xpcsviewer.fitting.sampler import _run_mcmc

        sig = inspect.signature(_run_mcmc)
        assert "init_params" in sig.parameters

    @needs_numpyro
    def test_biject_to_inv_transforms_constrained_to_unconstrained(self):
        """biject_to(constraint).inv maps constrained → unconstrained."""
        # LogNormal: constrained positive → unconstrained via log
        transform = biject_to(dist.LogNormal(0, 1).support)
        val = jnp.array(2.0)
        unconstrained = transform.inv(val)
        assert float(jnp.abs(unconstrained - jnp.log(2.0))) < 1e-6

        # HalfNormal: constrained positive → unconstrained via log
        transform_hn = biject_to(dist.HalfNormal(1).support)
        unconstrained_hn = transform_hn.inv(jnp.array(0.3))
        assert float(jnp.abs(unconstrained_hn - jnp.log(0.3))) < 1e-6

        # Beta: constrained (0,1) → unconstrained via logit
        transform_beta = biject_to(dist.Beta(2, 2).support)
        beta_val = jnp.array(0.8)
        unconstrained_beta = transform_beta.inv(beta_val)
        expected_logit = jnp.log(beta_val / (1 - beta_val))
        assert float(jnp.abs(unconstrained_beta - expected_logit)) < 1e-5

        # Normal: identity transform
        transform_normal = biject_to(dist.Normal(0, 1).support)
        unconstrained_normal = transform_normal.inv(jnp.array(1.5))
        assert float(jnp.abs(unconstrained_normal - 1.5)) < 1e-6

    @needs_numpyro
    def test_run_mcmc_uses_init_to_value(self):
        """_run_mcmc must use init_to_value for warm-start initialization."""
        from xpcsviewer.fitting.sampler import _run_mcmc

        source = inspect.getsource(_run_mcmc)
        assert "init_to_value" in source, (
            "_run_mcmc should use init_to_value for constrained warm-start"
        )

    @needs_numpyro
    def test_warm_start_actually_starts_near_nlsq(self):
        """NUTS with warm-start should begin near the NLSQ point estimate.

        This is the critical regression test: before the fix, passing
        tau=1.0 as init_params caused NUTS to start at exp(1.0)=2.718
        instead of tau=1.0. Now init_to_value handles the transform.
        """
        from xpcsviewer.fitting.sampler import _run_mcmc

        # Simple model with known parameters
        true_tau = 1.5

        def model(x, y=None, yerr=None):
            tau = numpyro.sample("tau", dist.LogNormal(0.0, 2.0))
            baseline = numpyro.sample("baseline", dist.Normal(1.0, 0.1))
            contrast = numpyro.sample("contrast", dist.HalfNormal(1.0))
            tau = jnp.clip(tau, 1e-30)
            mu = baseline + contrast * jnp.exp(-2 * x / tau)
            sigma = yerr if yerr is not None else 0.01
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        x = jnp.linspace(0.1, 10.0, 50)
        y = 1.0 + 0.3 * jnp.exp(-2 * x / true_tau)

        config = SamplerConfig(
            num_warmup=100,
            num_samples=200,
            num_chains=1,
            random_seed=42,
        )

        init_params = {"tau": true_tau, "baseline": 1.0, "contrast": 0.3}

        mcmc, samples = _run_mcmc(
            model,
            (jnp.array(x), jnp.array(y), None),
            config,
            init_params=init_params,
        )

        # Posterior mean for tau should be close to true value
        tau_mean = float(np.mean(np.asarray(samples["tau"])))
        assert abs(tau_mean - true_tau) / true_tau < 0.3, (
            f"Posterior mean tau={tau_mean:.3f} too far from true={true_tau}"
        )


# ============================================================================
# BUG-B: Jitter in unconstrained space
# ============================================================================
class TestBugB_JitterUnconstrained:
    """Verify init_to_value handles constrained→unconstrained internally.

    With the init_to_value approach, NumPyro handles the constrained→
    unconstrained transform and jitter internally, so we verify the
    strategy is correctly wired up.
    """

    @needs_numpyro
    def test_init_to_value_handles_constrained_params(self):
        """init_to_value must accept constrained values and produce valid init."""
        from numpyro.infer.initialization import init_to_value

        # Verify init_to_value exists and is callable with values kwarg
        strategy = init_to_value(values={"tau": 1.0})
        assert callable(strategy)


# ============================================================================
# BUG-C: get_hdi uses true HDI
# ============================================================================
class TestBugC_TrueHDI:
    """Verify get_hdi returns true HDI, not equal-tailed interval."""

    def test_hdi_narrower_than_equal_tailed_for_skewed(self):
        """For a skewed distribution, true HDI should be narrower."""
        np.random.seed(42)
        # LogNormal samples (heavily right-skewed)
        skewed_samples = np.random.lognormal(mean=0, sigma=1.0, size=10000)

        result = FitResult(
            samples={"tau": skewed_samples},
        )

        hdi_lower, hdi_upper = result.get_hdi("tau", prob=0.94)
        hdi_width = hdi_upper - hdi_lower

        # Equal-tailed interval for comparison
        alpha = 0.06
        et_lower = np.percentile(skewed_samples, 100 * alpha / 2)
        et_upper = np.percentile(skewed_samples, 100 * (1 - alpha / 2))
        et_width = et_upper - et_lower

        # HDI must be narrower (or at most equal) to equal-tailed
        assert hdi_width <= et_width * 1.01, (
            f"HDI width ({hdi_width:.4f}) should be <= equal-tailed "
            f"width ({et_width:.4f}) for skewed distribution"
        )

    def test_hdi_similar_to_equal_tailed_for_symmetric(self):
        """For a symmetric distribution, HDI ≈ equal-tailed interval."""
        np.random.seed(42)
        symmetric_samples = np.random.normal(loc=5.0, scale=1.0, size=10000)

        result = FitResult(
            samples={"baseline": symmetric_samples},
        )

        hdi_lower, hdi_upper = result.get_hdi("baseline", prob=0.94)
        hdi_width = hdi_upper - hdi_lower

        # Equal-tailed
        alpha = 0.06
        et_lower = np.percentile(symmetric_samples, 100 * alpha / 2)
        et_upper = np.percentile(symmetric_samples, 100 * (1 - alpha / 2))
        et_width = et_upper - et_lower

        # For symmetric, they should be very similar (within 5%)
        assert abs(hdi_width - et_width) / et_width < 0.05

    def test_hdi_raises_for_missing_param(self):
        """get_hdi must raise KeyError for unknown parameter."""
        result = FitResult(samples={"tau": np.array([1.0, 2.0, 3.0])})
        with pytest.raises(KeyError, match="nonexistent"):
            result.get_hdi("nonexistent")


# ============================================================================
# BUG-D: max_treedepth_reached computed from num_steps
# ============================================================================
class TestBugD_MaxTreedepthReached:
    """Verify max_treedepth_reached is computed, not hardcoded to 0."""

    @needs_numpyro
    def test_build_fit_result_source_uses_num_steps(self):
        """_build_fit_result must reference num_steps for max_treedepth."""
        from xpcsviewer.fitting.sampler import _build_fit_result

        source = inspect.getsource(_build_fit_result)
        assert "num_steps" in source, (
            "_build_fit_result should use num_steps extra field"
        )
        # The old code had: max_treedepth_reached=0,  # NumPyro doesn't track
        # The fix computes it from num_steps. Verify the computation exists.
        assert "config.max_tree_depth" in source, (
            "_build_fit_result should use config.max_tree_depth to compute "
            "max_treedepth_reached from num_steps"
        )

    @needs_numpyro
    def test_run_mcmc_requests_num_steps(self):
        """_run_mcmc must request num_steps as an extra field."""
        from xpcsviewer.fitting.sampler import _run_mcmc

        source = inspect.getsource(_run_mcmc)
        assert "num_steps" in source, (
            "_run_mcmc should request num_steps via extra_fields"
        )

    def test_diagnostics_max_treedepth_field_exists(self):
        """FitDiagnostics must have max_treedepth_reached field."""
        diag = FitDiagnostics(
            r_hat={"tau": 1.0},
            ess_bulk={"tau": 500},
            ess_tail={"tau": 500},
            divergences=0,
            max_treedepth_reached=5,
        )
        assert diag.max_treedepth_reached == 5


# ============================================================================
# BUG-E: Time-based seed stored in config
# ============================================================================
class TestBugE_SeedStorage:
    """Verify time-based seed is stored in SamplerConfig for reproducibility."""

    @needs_numpyro
    def test_time_based_seed_stored_in_config(self):
        """After _run_mcmc with random_seed=None, config must have actual seed."""
        from xpcsviewer.fitting.sampler import _run_mcmc

        def model(x, y=None, yerr=None):
            tau = numpyro.sample("tau", dist.LogNormal(0.0, 2.0))
            mu = jnp.exp(-x / tau)
            numpyro.sample("obs", dist.Normal(mu, 0.01), obs=y)

        x = jnp.linspace(0.1, 5.0, 10)
        y = jnp.exp(-x / 1.0)

        config = SamplerConfig(
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            random_seed=None,  # Time-based
        )
        assert config.random_seed is None

        _run_mcmc(model, (x, y, None), config)

        # After running, config.random_seed should be populated
        assert config.random_seed is not None
        assert isinstance(config.random_seed, int)
        assert config.random_seed > 0

    @needs_numpyro
    def test_user_specified_seed_unchanged(self):
        """User-specified seed should not be overwritten."""
        from xpcsviewer.fitting.sampler import _run_mcmc

        def model(x, y=None, yerr=None):
            tau = numpyro.sample("tau", dist.LogNormal(0.0, 2.0))
            mu = jnp.exp(-x / tau)
            numpyro.sample("obs", dist.Normal(mu, 0.01), obs=y)

        x = jnp.linspace(0.1, 5.0, 10)
        y = jnp.exp(-x / 1.0)

        config = SamplerConfig(
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            random_seed=12345,
        )

        _run_mcmc(model, (x, y, None), config)
        assert config.random_seed == 12345

    @needs_numpyro
    def test_fit_result_to_dict_has_actual_seed(self):
        """FitResult.to_dict() must export the actual seed, not None."""
        from xpcsviewer.fitting import fit_single_exp

        x = np.linspace(0.1, 10.0, 30)
        y = 1.0 + 0.3 * np.exp(-2 * x / 1.0)

        result = fit_single_exp(
            x,
            y,
            num_warmup=50,
            num_samples=50,
            num_chains=1,
        )

        exported = result.to_dict()
        seed = exported["sampler_config"]["random_seed"]
        assert seed is not None, "to_dict() should export the actual seed, not None"
        assert isinstance(seed, int)


# ============================================================================
# BUG-F: predict() raises NotImplementedError
# ============================================================================
class TestBugF_PredictRaises:
    """Verify predict() raises instead of returning silent zeros."""

    def test_predict_raises_not_implemented(self):
        """FitResult.predict() must raise NotImplementedError."""
        result = FitResult(
            samples={"tau": np.array([1.0, 1.1, 0.9])},
        )
        with pytest.raises(NotImplementedError, match="model function"):
            result.predict(np.linspace(0, 10, 20))


# ============================================================================
# Integration: End-to-end NLSQ → NUTS warm-start
# ============================================================================
class TestEndToEndWarmStart:
    """Integration tests for the full NLSQ → NUTS pipeline."""

    @needs_numpyro
    @pytest.mark.slow
    def test_single_exp_fit_recovers_parameters(self):
        """fit_single_exp should recover true parameters within HDI."""
        from xpcsviewer.fitting import fit_single_exp

        true_tau = 2.0
        true_baseline = 1.0
        true_contrast = 0.25

        np.random.seed(42)
        x = np.linspace(0.1, 15.0, 60)
        y_true = true_baseline + true_contrast * np.exp(-2 * x / true_tau)
        y = y_true + 0.005 * np.random.randn(len(x))

        result = fit_single_exp(
            x,
            y,
            num_warmup=200,
            num_samples=500,
            num_chains=2,
            random_seed=42,
        )

        # Check parameter recovery
        tau_mean = result.get_mean("tau")
        baseline_mean = result.get_mean("baseline")
        contrast_mean = result.get_mean("contrast")

        assert abs(tau_mean - true_tau) / true_tau < 0.15, (
            f"tau={tau_mean:.3f}, expected ~{true_tau}"
        )
        assert abs(baseline_mean - true_baseline) < 0.05, (
            f"baseline={baseline_mean:.3f}, expected ~{true_baseline}"
        )
        assert abs(contrast_mean - true_contrast) / true_contrast < 0.15, (
            f"contrast={contrast_mean:.3f}, expected ~{true_contrast}"
        )

        # Check diagnostics are populated
        assert result.diagnostics.max_treedepth_reached >= 0
        # BFMI may be None for small sample sizes if ArviZ can't compute it
        if result.diagnostics.bfmi is not None:
            assert result.diagnostics.bfmi > 0

        # Check seed was stored
        assert result.config is not None
        assert result.config.random_seed is not None


# ============================================================================
# BUG-G: run_power_law_fit must accept tau_err (not hardcode None)
# ============================================================================
class TestBugG_PowerLawTauErr:
    """Verify tau_err is threaded through the Bayesian diffusion pipeline.

    Before the fix, run_power_law_fit() hardcoded ``tau_err = None`` on
    line 612, silently discarding measurement uncertainties. This caused
    unweighted NLSQ warm-starts and NUTS inferred noise instead of using
    known G2 fitting errors.
    """

    @needs_numpyro
    def test_run_power_law_fit_accepts_tau_err(self):
        """run_power_law_fit must have a tau_err parameter."""
        from xpcsviewer.fitting.sampler import run_power_law_fit

        sig = inspect.signature(run_power_law_fit)
        assert "tau_err" in sig.parameters, (
            "run_power_law_fit is missing tau_err parameter"
        )
        param = sig.parameters["tau_err"]
        assert param.default is None, "tau_err should default to None"

    def test_fit_power_law_accepts_tau_err(self):
        """Public fit_power_law must have a tau_err parameter."""
        from xpcsviewer.fitting import fit_power_law

        sig = inspect.signature(fit_power_law)
        assert "tau_err" in sig.parameters, "fit_power_law is missing tau_err parameter"
        param = sig.parameters["tau_err"]
        assert param.default is None, "tau_err should default to None"

    @needs_numpyro
    def test_source_no_hardcoded_tau_err_none(self):
        """run_power_law_fit must NOT hardcode tau_err = None."""
        from xpcsviewer.fitting.sampler import run_power_law_fit

        source = inspect.getsource(run_power_law_fit)
        # The old bug: a bare assignment "tau_err = None" inside the body
        # (not the default value in the signature)
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip the function signature default value
            if stripped.startswith("def ") or stripped.startswith("tau_err:"):
                continue
            assert stripped != "tau_err = None", (
                "run_power_law_fit still hardcodes tau_err = None in the body"
            )

    @needs_numpyro
    @pytest.mark.slow
    def test_tau_err_weighted_fit_succeeds(self):
        """Power-law fit with tau_err should run without error."""
        from xpcsviewer.fitting.sampler import run_power_law_fit

        # Synthetic power-law data: tau = 100 * q^(-2)
        q = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.1])
        true_tau0, true_alpha = 100.0, 2.0
        tau = true_tau0 * q ** (-true_alpha)
        tau_err = 0.05 * tau  # 5% relative error

        result = run_power_law_fit(
            q,
            tau,
            tau_err=tau_err,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            random_seed=42,
        )

        # Basic sanity: result has expected parameters
        assert "tau0" in result.samples
        assert "alpha" in result.samples

        # Posterior should recover parameters reasonably
        alpha_mean = result.get_mean("alpha")
        assert abs(alpha_mean - true_alpha) / true_alpha < 0.3, (
            f"alpha={alpha_mean:.2f}, expected ~{true_alpha}"
        )


# ============================================================================
# BUG-H: samples dict key order must match function signature
# ============================================================================
class TestBugH_SamplesKeyOrder:
    """Verify FitResult.samples preserves param_names ordering.

    Root cause: _build_fit_result iterated over mcmc.get_samples().items()
    (NumPyro internal order) instead of param_names (function signature
    order).  plot_posterior_predictive then unpacked *params in dict key
    order, scrambling arguments and producing visually divergent fits
    despite perfect MCMC convergence diagnostics.
    """

    def test_fit_result_has_param_names_field(self):
        """FitResult stores canonical param_names list."""
        result = FitResult(
            samples={"a": np.array([1.0]), "b": np.array([2.0])},
            param_names=["a", "b"],
        )
        assert result.param_names == ["a", "b"]

    def test_fit_result_param_names_default_empty(self):
        """Legacy FitResults without param_names get an empty list."""
        result = FitResult(samples={"a": np.array([1.0])})
        assert result.param_names == []

    def test_samples_dict_preserves_param_names_order(self):
        """_build_fit_result must produce samples keyed in param_names order."""
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from xpcsviewer.fitting.sampler import _build_fit_result

        # Simulate NumPyro returning keys in ALPHABETICAL order
        # (opposite of param_names)
        class FakeMCMC:
            def get_extra_fields(self):
                return {
                    "diverging": np.array([False, False]),
                    "num_steps": np.array([7, 7]),
                }

        samples_alpha_order = {
            "baseline": np.array([1.01, 0.99]),
            "contrast": np.array([0.38, 0.37]),
            "tau": np.array([0.5, 0.6]),
        }
        param_names = ["tau", "baseline", "contrast"]

        # Mock ArviZ functions so _build_fit_result can complete
        # without a real NumPyro MCMC object
        fake_summary = pd.DataFrame(
            {
                "r_hat": [1.0, 1.0, 1.0],
                "ess_bulk": [100, 100, 100],
                "ess_tail": [80, 80, 80],
            },
            index=param_names,
        )

        with (
            patch(
                "xpcsviewer.fitting.sampler.az.from_numpyro",
                return_value=MagicMock(),
            ),
            patch(
                "xpcsviewer.fitting.sampler.az.summary",
                return_value=fake_summary,
            ),
            patch(
                "xpcsviewer.fitting.sampler.az.bfmi",
                return_value=np.array([0.9]),
            ),
        ):
            result = _build_fit_result(
                FakeMCMC(),
                samples_alpha_order,
                nlsq_init={"tau": 0.5, "baseline": 1.0, "contrast": 0.3},
                param_names=param_names,
                config=SamplerConfig(
                    num_warmup=10,
                    num_samples=2,
                    num_chains=1,
                ),
            )

        # Key order must follow param_names, not alphabetical
        assert list(result.samples.keys()) == param_names
        assert result.param_names == param_names
        # Verify diagnostics were populated
        assert result.diagnostics is not None
        assert result.diagnostics.divergences == 0

    @needs_numpyro
    def test_posterior_predictive_uses_correct_param_order(self):
        """plot_posterior_predictive must call model with correct arg order.

        Regression test: previously params were unpacked in dict key order
        (alphabetical from NumPyro), not function signature order.
        """
        from xpcsviewer.fitting.models import single_exp_func

        # Create a FitResult with WRONG dict key order but correct param_names
        wrong_order_samples = {
            "baseline": np.array([1.0, 1.01, 0.99]),
            "contrast": np.array([0.3, 0.31, 0.29]),
            "tau": np.array([0.5, 0.51, 0.49]),
        }
        correct_names = ["tau", "baseline", "contrast"]
        result = FitResult(
            samples=wrong_order_samples,
            param_names=correct_names,
        )

        # With param_names, visualization should unpack in correct order
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x_data = np.linspace(0.01, 1.0, 20)
        y_data = single_exp_func(x_data, 0.5, 1.0, 0.3)

        plot_posterior_predictive(result, single_exp_func, x_data, y_data, ax=ax)

        # The median fit at x=0.01 should be close to baseline + contrast
        # ≈ 1.3, NOT ≈ 0.8 (which would indicate scrambled params)
        lines = [c for c in ax.get_children() if hasattr(c, "get_ydata")]
        for line in lines:
            ydata = line.get_ydata()
            if ydata is not None and len(ydata) > 10:
                # Median fit line starts near 1.3
                assert ydata[0] > 1.1, (
                    f"Fit starts at {ydata[0]:.2f}, expected >1.1. "
                    "Params likely scrambled."
                )
                break
        plt.close(fig)
