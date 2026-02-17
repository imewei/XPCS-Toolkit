"""TG1: Tests for the silent scientific corruption chain (BUG-006, BUG-003, BUG-004, BUG-020, BUG-021).

Six focused tests that verify the corruption chain is broken:
1. HDF5Facade read_g2_data uses g2.shape[1] (n_q) not g2.shape[0] (n_delay)
2. NLSQ exception handler returns sentinel with converged=False, is_fallback=True, _r_squared=nan
3. NumPyro models clamp tau via jnp.clip(tau, 1e-30) -- no NaN gradients near zero
4. Sampler PRNG key is non-deterministic (not PRNGKey(0)) and seed is logged
5. Per-chain init params have jitter (not identical via broadcast_to)
6. End-to-end chain: missing Q values -> NLSQ fails gracefully -> sentinel propagates
"""

from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import numpyro

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import nlsq  # noqa: F401

    NLSQ_AVAILABLE = True
except ImportError:
    NLSQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Test 1: HDF5Facade read_g2_data uses g2.shape[1] for dummy Q values (BUG-006)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
class TestHDF5FacadeG2AxisBug:
    """BUG-006: Dummy Q values must use n_q axis (shape[1]), not n_delay axis (shape[0])."""

    def test_dummy_q_values_use_n_q_axis(self, tmp_path):
        """read_g2_data generates len(n_q) dummy Q values when q_values key is missing.

        G2 data shape is (n_delay, n_q). Before BUG-006 fix, the fallback used
        g2.shape[0] = n_delay instead of g2.shape[1] = n_q, producing the wrong
        number of dummy Q values.
        """
        # Create an HDF5 file with g2 shape (n_delay, n_q) = (100, 5)
        n_delay = 100
        n_q = 5
        hdf5_file = tmp_path / "test_g2_axis.h5"

        with h5py.File(hdf5_file, "w") as f:
            g2_group = f.create_group("/xpcs/g2")
            g2_group.create_dataset(
                "g2", data=np.ones((n_delay, n_q), dtype=np.float64)
            )
            g2_group.create_dataset(
                "g2_err", data=np.ones((n_delay, n_q), dtype=np.float64) * 0.01
            )
            g2_group.create_dataset("delay_times", data=np.logspace(-3, 2, n_delay))
            # Deliberately omit q_values and q_list to trigger the fallback path

        from xpcsviewer.io.hdf5_facade import HDF5Facade

        facade = HDF5Facade()
        result = facade.read_g2_data(str(hdf5_file))

        # After BUG-006 fix: dummy Q count == n_q (5), not n_delay (100)
        assert len(result.q_values) == n_q, (
            f"Expected {n_q} dummy Q values (shape[1]=n_q), "
            f"got {len(result.q_values)}. "
            f"If count is {n_delay}, BUG-006 is not fixed (shape[0] still used)."
        )

    def test_dummy_q_values_are_sequential_integers(self, tmp_path):
        """Dummy Q values are sequential integers 0..n_q-1."""
        n_delay = 50
        n_q = 3
        hdf5_file = tmp_path / "test_g2_seq.h5"

        with h5py.File(hdf5_file, "w") as f:
            g2_group = f.create_group("/xpcs/g2")
            g2_group.create_dataset(
                "g2", data=np.ones((n_delay, n_q), dtype=np.float64)
            )
            g2_group.create_dataset(
                "g2_err", data=np.ones((n_delay, n_q), dtype=np.float64) * 0.01
            )
            g2_group.create_dataset("delay_times", data=np.logspace(-3, 2, n_delay))
            # No q_values, no q_list -- triggers fallback

        from xpcsviewer.io.hdf5_facade import HDF5Facade

        facade = HDF5Facade()
        result = facade.read_g2_data(str(hdf5_file))

        assert list(result.q_values) == list(range(n_q)), (
            f"Expected sequential Q indices {list(range(n_q))}, "
            f"got {list(result.q_values)}"
        )


# ---------------------------------------------------------------------------
# Test 2: NLSQ exception handler returns sentinel (BUG-003)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NLSQ_AVAILABLE, reason="nlsq not installed")
class TestNLSQSentinelResult:
    """BUG-003: NLSQ exception handler must return a proper sentinel, not disguise p0 as a result."""

    def test_exception_returns_sentinel_with_converged_false(self):
        """When nlsq.fit raises, NLSQResult has converged=False."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x = np.logspace(-3, 2, 10)
        y = np.ones(10)
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {"tau": (1e-6, 1e6), "baseline": (0.0, 2.0), "contrast": (0.0, 1.0)}

        with patch("nlsq.fit", side_effect=RuntimeError("simulated nlsq failure")):
            result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds)

        assert result.converged is False, "Sentinel result must have converged=False"

    def test_exception_returns_sentinel_with_is_fallback_true(self):
        """When nlsq.fit raises, NLSQResult has is_fallback=True."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x = np.logspace(-3, 2, 10)
        y = np.ones(10)
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {"tau": (1e-6, 1e6), "baseline": (0.0, 2.0), "contrast": (0.0, 1.0)}

        with patch("nlsq.fit", side_effect=RuntimeError("simulated nlsq failure")):
            result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds)

        assert hasattr(result, "is_fallback"), (
            "NLSQResult must have is_fallback attribute"
        )
        assert result.is_fallback is True, "Sentinel result must have is_fallback=True"

    def test_exception_returns_sentinel_with_nan_r_squared(self):
        """When nlsq.fit raises, NLSQResult._r_squared is float('nan'), not 0.0."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x = np.logspace(-3, 2, 10)
        y = np.ones(10)
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {"tau": (1e-6, 1e6), "baseline": (0.0, 2.0), "contrast": (0.0, 1.0)}

        with patch("nlsq.fit", side_effect=RuntimeError("simulated nlsq failure")):
            result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds)

        # _r_squared must be NaN to distinguish failed fits from bad-but-converged fits
        assert math.isnan(result._r_squared), (
            f"Sentinel _r_squared must be NaN, got {result._r_squared!r}. "
            f"If it is 0.0, BUG-003 is not fixed."
        )

    def test_exception_sentinel_has_no_native_result(self):
        """Sentinel must not have native_result set (prevents delegation to fake result)."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x = np.logspace(-3, 2, 10)
        y = np.ones(10)
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {"tau": (1e-6, 1e6), "baseline": (0.0, 2.0), "contrast": (0.0, 1.0)}

        with patch("nlsq.fit", side_effect=RuntimeError("simulated nlsq failure")):
            result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds)

        assert result.native_result is None, (
            "Sentinel result must have native_result=None (no delegation to fake result)"
        )


# ---------------------------------------------------------------------------
# Test 3: NumPyro models clamp tau via jnp.clip(tau, 1e-30) (BUG-004)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestNumPyroModelTauClamping:
    """BUG-004: All NumPyro models must clamp tau to avoid NaN gradients near zero."""

    def test_single_exp_model_clamps_tau(self):
        """single_exp_model: tau is clipped before use, preventing NaN in mu.

        Without jnp.clip(tau, 1e-30), tau sampled near zero causes
        exp(-2*x/tau) -> exp(-inf) -> 0 or NaN depending on the gradient path.
        """
        import jax
        import jax.numpy as jnp
        import numpyro

        from xpcsviewer.fitting.models import single_exp_model

        x = jnp.array([1e-3, 1.0, 10.0])

        # Force near-zero tau using numpyro condition handler
        conditioned = numpyro.handlers.condition(
            single_exp_model,
            {
                "tau": jnp.array(1e-30),
                "baseline": jnp.array(1.0),
                "contrast": jnp.array(0.3),
            },
        )
        seeded = numpyro.handlers.seed(conditioned, rng_seed=0)
        trace = numpyro.handlers.trace(seeded).get_trace(x)
        mu = trace["obs"]["fn"].loc

        assert jnp.all(jnp.isfinite(mu)), (
            f"single_exp_model produces non-finite mu with tau=1e-30: mu={mu}. "
            f"Add tau = jnp.clip(tau, 1e-30) to fix BUG-004."
        )

    def test_stretched_exp_model_clamps_tau(self):
        """stretched_exp_model: tau is clipped to prevent overflow in (2x/tau)^beta.

        When tau approaches zero, 2x/tau -> inf, and inf^beta overflows.
        """
        import jax.numpy as jnp
        import numpyro

        from xpcsviewer.fitting.models import stretched_exp_model

        x = jnp.array([1e-3, 1.0, 10.0])

        conditioned = numpyro.handlers.condition(
            stretched_exp_model,
            {
                "tau": jnp.array(1e-30),
                "baseline": jnp.array(1.0),
                "contrast": jnp.array(0.3),
                "beta": jnp.array(0.5),
            },
        )
        seeded = numpyro.handlers.seed(conditioned, rng_seed=0)
        trace = numpyro.handlers.trace(seeded).get_trace(x)
        mu = trace["obs"]["fn"].loc

        assert jnp.all(jnp.isfinite(mu)), (
            f"stretched_exp_model produces non-finite mu with tau=1e-30: mu={mu}. "
            f"Add tau = jnp.clip(tau, 1e-30) to fix BUG-004."
        )

    def test_power_law_model_clamps_q(self):
        """power_law_model: q is clipped before log(q) to prevent log(0) = -inf.

        The model uses jnp.log(q) in log-space computation; q=0 yields -inf -> NaN.
        """
        import jax.numpy as jnp
        import numpyro

        from xpcsviewer.fitting.models import power_law_model

        # Include near-zero q values that would trigger log(0)
        q = jnp.array([1e-30, 0.1, 1.0])

        conditioned = numpyro.handlers.condition(
            power_law_model,
            {"tau0": jnp.array(1.0), "alpha": jnp.array(2.0)},
        )
        seeded = numpyro.handlers.seed(conditioned, rng_seed=0)
        trace = numpyro.handlers.trace(seeded).get_trace(q)
        mu = trace["obs"]["fn"].loc

        assert jnp.all(jnp.isfinite(mu)), (
            f"power_law_model produces non-finite mu with q=1e-30: mu={mu}. "
            f"Add q = jnp.clip(q, 1e-30) to fix BUG-004."
        )


# ---------------------------------------------------------------------------
# Test 4: Sampler PRNG key is non-deterministic and seed is logged (BUG-020)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestSamplerPRNGNonDeterministic:
    """BUG-020: PRNG key must not be hardcoded PRNGKey(0); seed must be logged."""

    def test_rng_key_is_not_hardcoded_zero(self):
        """When config.random_seed is None, rng_key is NOT created with PRNGKey(0)."""
        from xpcsviewer.fitting.results import SamplerConfig
        from xpcsviewer.fitting.sampler import _run_mcmc

        config = SamplerConfig(
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            random_seed=None,  # No explicit seed -> must NOT use PRNGKey(0)
        )

        captured_keys = []
        original_prng_key = jax.random.PRNGKey

        def capturing_prng_key(seed):
            captured_keys.append(seed)
            return original_prng_key(seed)

        with patch("jax.random.PRNGKey", side_effect=capturing_prng_key):
            with patch("xpcsviewer.fitting.sampler.MCMC") as mock_mcmc_cls:
                mock_mcmc = MagicMock()
                mock_mcmc.get_samples.return_value = {}
                mock_mcmc_cls.return_value = mock_mcmc

                with patch("xpcsviewer.fitting.sampler.NUTS"):
                    try:
                        _run_mcmc(lambda x: None, (), config)
                    except Exception:
                        pass  # We only care about captured_keys

        assert len(captured_keys) >= 1, "PRNGKey must have been called at least once"

        # All seeds used for None config must NOT be 0 (time-based, non-deterministic)
        # Probability of time_ns() % 2**31 == 0 is ~4.6e-10 per run (acceptable)
        non_zero_seeds = [k for k in captured_keys if k != 0]
        assert len(non_zero_seeds) >= 1, (
            f"All PRNGKey calls used seed=0, indicating hardcoded PRNGKey(0). "
            f"Seeds used: {captured_keys}. BUG-020 not fixed."
        )

    def test_seed_is_logged(self, caplog):
        """When a time-based seed is generated, it must be logged for reproducibility."""
        from xpcsviewer.fitting.results import SamplerConfig
        from xpcsviewer.fitting.sampler import _run_mcmc

        config = SamplerConfig(
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            random_seed=None,
        )

        with caplog.at_level(logging.DEBUG, logger="xpcsviewer.fitting.sampler"):
            with patch("xpcsviewer.fitting.sampler.MCMC") as mock_mcmc_cls:
                mock_mcmc = MagicMock()
                mock_mcmc.get_samples.return_value = {}
                mock_mcmc_cls.return_value = mock_mcmc

                with patch("xpcsviewer.fitting.sampler.NUTS"):
                    try:
                        _run_mcmc(lambda x: None, (), config)
                    except Exception:
                        pass

        seed_logged = any(
            "seed" in record.message.lower() or "prng" in record.message.lower()
            for record in caplog.records
        )
        assert seed_logged, (
            f"PRNG seed was not logged. BUG-020 requires seed logging for reproducibility. "
            f"Log records: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Test 5: Per-chain init params have jitter (BUG-021)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestPerChainJitter:
    """BUG-021: MCMC chains must start from distinct points, not identical broadcast_to values."""

    def test_multi_chain_init_params_are_not_identical(self):
        """When num_chains > 1, each chain's init param value differs by jitter.

        Before BUG-021 fix, jnp.broadcast_to(val, (num_chains,)) produces
        identical values for all chains, making R-hat diagnostics meaningless.
        """
        from xpcsviewer.fitting.results import SamplerConfig
        from xpcsviewer.fitting.sampler import _run_mcmc

        n_chains = 4
        config = SamplerConfig(
            num_warmup=10,
            num_samples=10,
            num_chains=n_chains,
            random_seed=42,
        )

        init_params = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        captured_init = {}

        def capture_run(rng_key, *args, init_params=None, **kwargs):
            if init_params is not None:
                captured_init.update(init_params)

        with patch("xpcsviewer.fitting.sampler.MCMC") as mock_mcmc_cls:
            mock_mcmc = MagicMock()
            mock_mcmc.get_samples.return_value = {}
            mock_mcmc.run.side_effect = capture_run
            mock_mcmc_cls.return_value = mock_mcmc

            with patch("xpcsviewer.fitting.sampler.NUTS"):
                try:
                    _run_mcmc(lambda x: None, (), config, init_params=init_params)
                except Exception:
                    pass

        assert "tau" in captured_init, "tau must be in init_params passed to MCMC.run"

        tau_vals = jnp.asarray(captured_init["tau"])
        assert len(tau_vals) == n_chains, (
            f"Expected {n_chains} per-chain tau init values, got {len(tau_vals)}"
        )

        all_identical = bool(jnp.all(tau_vals == tau_vals[0]))
        assert not all_identical, (
            f"All {n_chains} chains have identical tau init={float(tau_vals[0]):.6f}. "
            f"jnp.broadcast_to produces identical values. BUG-021 not fixed: "
            f"use val + 0.01 * jax.random.normal(subkey, shape) per chain."
        )

    def test_jitter_magnitude_preserves_scale(self):
        """Jitter at 1% scale does not corrupt the warm-start value significantly."""
        from xpcsviewer.fitting.results import SamplerConfig
        from xpcsviewer.fitting.sampler import _run_mcmc

        n_chains = 4
        config = SamplerConfig(
            num_warmup=10,
            num_samples=10,
            num_chains=n_chains,
            random_seed=42,
        )

        tau_init = 5.0
        init_params = {"tau": tau_init, "baseline": 1.0, "contrast": 0.3}
        captured_init = {}

        def capture_run(rng_key, *args, init_params=None, **kwargs):
            if init_params is not None:
                captured_init.update(init_params)

        with patch("xpcsviewer.fitting.sampler.MCMC") as mock_mcmc_cls:
            mock_mcmc = MagicMock()
            mock_mcmc.get_samples.return_value = {}
            mock_mcmc.run.side_effect = capture_run
            mock_mcmc_cls.return_value = mock_mcmc

            with patch("xpcsviewer.fitting.sampler.NUTS"):
                try:
                    _run_mcmc(lambda x: None, (), config, init_params=init_params)
                except Exception:
                    pass

        if "tau" in captured_init:
            tau_vals = np.asarray(captured_init["tau"])
            if len(tau_vals) > 1:
                mean_val = float(np.mean(tau_vals))
                # Jitter at 0.01 * normal noise: mean should stay close to original
                assert abs(mean_val - tau_init) < tau_init * 0.5, (
                    f"Jitter too large: mean tau={mean_val:.3f}, expected ~{tau_init:.3f}"
                )


# ---------------------------------------------------------------------------
# Test 6: End-to-end chain test (BUG-006 + BUG-003 together)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py not installed")
@pytest.mark.skipif(not NLSQ_AVAILABLE, reason="nlsq not installed")
class TestEndToEndCorruptionChain:
    """End-to-end: missing Q values -> NLSQ fails -> sentinel propagates with is_fallback=True."""

    def test_missing_q_values_produce_correct_shape(self, tmp_path):
        """Missing Q values in HDF5 produce correct number of dummy Q entries (BUG-006 entry point)."""
        from xpcsviewer.io.hdf5_facade import HDF5Facade

        # G2 data: 80 delay points, 7 Q bins -- would produce 80 dummy Q values before fix
        n_delay, n_q = 80, 7
        hdf5_file = tmp_path / "missing_q.h5"

        with h5py.File(hdf5_file, "w") as f:
            grp = f.create_group("/xpcs/g2")
            grp.create_dataset("g2", data=np.ones((n_delay, n_q), dtype=np.float64))
            grp.create_dataset(
                "g2_err", data=np.ones((n_delay, n_q), dtype=np.float64) * 0.01
            )
            grp.create_dataset("delay_times", data=np.logspace(-3, 2, n_delay))
            # No q_values, no q_list

        facade = HDF5Facade()
        g2_data = facade.read_g2_data(str(hdf5_file))

        assert len(g2_data.q_values) == n_q, (
            f"Q values count {len(g2_data.q_values)} != n_q {n_q}. "
            f"BUG-006 not fixed: shape[0] still used instead of shape[1]."
        )

    def test_nlsq_failure_produces_detectable_sentinel(self):
        """NLSQ failure returns is_fallback=True sentinel (not silent p0 return).

        Before BUG-003 fix: popt=p0_array, converged=False, but no is_fallback flag.
        After BUG-003 fix: is_fallback=True, _r_squared=NaN.
        """
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x = np.logspace(-3, 2, 20)
        y = 1.0 + 0.3 * np.exp(-2 * x / 1.0)
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {"tau": (1e-6, 1e6), "baseline": (0.0, 2.0), "contrast": (0.0, 1.0)}

        with patch("nlsq.fit", side_effect=ValueError("deliberate failure")):
            result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds)

        # The full corruption chain requires ALL three to be true:
        assert result.converged is False, "Sentinel must be unconverged"
        assert result.is_fallback is True, "Sentinel must be flagged as fallback"
        assert math.isnan(result._r_squared), "Sentinel R-squared must be NaN"

    def test_sentinel_params_shape_consistent_for_downstream(self):
        """Sentinel result params dict has correct keys for downstream use."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x = np.logspace(-3, 2, 20)
        y = np.ones(20)
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {"tau": (1e-6, 1e6), "baseline": (0.0, 2.0), "contrast": (0.0, 1.0)}

        with patch("nlsq.fit", side_effect=RuntimeError("simulated failure")):
            result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds)

        # Params dict must still be well-formed for downstream consumers
        assert set(result.params.keys()) == {"tau", "baseline", "contrast"}, (
            f"Sentinel params keys mismatch: {set(result.params.keys())}"
        )
        # All param values must be finite scalars
        for key, val in result.params.items():
            assert math.isfinite(val), f"Sentinel param '{key}'={val} is not finite"
