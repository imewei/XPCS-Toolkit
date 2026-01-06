"""Tests for NumPyro model definitions (T034).

This module tests the NumPyro probabilistic models for G2
correlation fitting.
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if NumPyro is available
try:
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestSingleExpModel:
    """Tests for single exponential model."""

    def test_model_can_be_imported(self) -> None:
        """Test model function can be imported."""
        from xpcsviewer.fitting.models import single_exp_model

        assert callable(single_exp_model)

    def test_model_prior_predictive(self) -> None:
        """Test model generates valid prior predictive samples."""
        import jax

        from xpcsviewer.fitting.models import single_exp_model

        x = jnp.logspace(-3, 2, 50)

        predictive = Predictive(single_exp_model, num_samples=100)
        rng_key = jax.random.PRNGKey(0)
        samples = predictive(rng_key, x=x)

        assert "tau" in samples
        assert "baseline" in samples
        assert "contrast" in samples
        assert samples["tau"].shape == (100,)
        assert np.all(samples["tau"] > 0)  # Tau is LogNormal

    def test_model_shape_consistency(self) -> None:
        """Test model output shape matches input."""
        import jax

        from xpcsviewer.fitting.models import single_exp_model

        x = jnp.logspace(-3, 2, 50)
        y = jnp.ones(50)

        predictive = Predictive(single_exp_model, num_samples=10)
        rng_key = jax.random.PRNGKey(0)
        samples = predictive(rng_key, x=x, y=None)

        assert "obs" in samples
        assert samples["obs"].shape == (10, 50)


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestStretchedExpModel:
    """Tests for stretched exponential model."""

    def test_model_can_be_imported(self) -> None:
        """Test model function can be imported."""
        from xpcsviewer.fitting.models import stretched_exp_model

        assert callable(stretched_exp_model)

    def test_model_has_beta_parameter(self) -> None:
        """Test model includes stretching exponent beta."""
        import jax

        from xpcsviewer.fitting.models import stretched_exp_model

        x = jnp.logspace(-3, 2, 50)

        predictive = Predictive(stretched_exp_model, num_samples=100)
        rng_key = jax.random.PRNGKey(0)
        samples = predictive(rng_key, x=x)

        assert "beta" in samples
        # Beta should be in (0, 1) from Beta distribution
        assert np.all(samples["beta"] > 0)
        assert np.all(samples["beta"] < 1)


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestDoubleExpModel:
    """Tests for double exponential model."""

    def test_model_can_be_imported(self) -> None:
        """Test model function can be imported."""
        from xpcsviewer.fitting.models import double_exp_model

        assert callable(double_exp_model)

    def test_model_has_two_taus(self) -> None:
        """Test model includes two relaxation times."""
        import jax

        from xpcsviewer.fitting.models import double_exp_model

        x = jnp.logspace(-3, 2, 50)

        predictive = Predictive(double_exp_model, num_samples=100)
        rng_key = jax.random.PRNGKey(0)
        samples = predictive(rng_key, x=x)

        assert "tau1" in samples
        assert "tau2" in samples
        # tau2 should be > tau1 by construction
        assert np.all(np.asarray(samples["tau2"]) >= np.asarray(samples["tau1"]))


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestPowerLawModel:
    """Tests for power law model."""

    def test_model_can_be_imported(self) -> None:
        """Test model function can be imported."""
        from xpcsviewer.fitting.models import power_law_model

        assert callable(power_law_model)

    def test_model_has_alpha_parameter(self) -> None:
        """Test model includes power law exponent alpha."""
        import jax

        from xpcsviewer.fitting.models import power_law_model

        q = jnp.logspace(-2, 0, 20)

        predictive = Predictive(power_law_model, num_samples=100)
        rng_key = jax.random.PRNGKey(0)
        samples = predictive(rng_key, q=q)

        assert "tau0" in samples
        assert "alpha" in samples
        assert samples["alpha"].shape == (100,)


class TestModelFunctions:
    """Tests for model functions (non-probabilistic versions)."""

    def test_single_exp_func(self) -> None:
        """Test single exponential function."""
        from xpcsviewer.fitting.models import single_exp_func

        x = np.logspace(-3, 2, 50)
        y = single_exp_func(x, tau=1.0, baseline=1.0, contrast=0.3)

        assert len(y) == 50
        # At x=0, y should be baseline + contrast
        assert np.isclose(y[0], 1.3, rtol=0.1)
        # At x >> tau, y should approach baseline
        assert y[-1] < 1.1

    def test_stretched_exp_func(self) -> None:
        """Test stretched exponential function."""
        from xpcsviewer.fitting.models import stretched_exp_func

        x = np.logspace(-3, 2, 50)
        y = stretched_exp_func(x, tau=1.0, baseline=1.0, contrast=0.3, beta=0.8)

        assert len(y) == 50
        # At x=0, y should be baseline + contrast
        assert np.isclose(y[0], 1.3, rtol=0.1)

    def test_power_law_func(self) -> None:
        """Test power law function."""
        from xpcsviewer.fitting.models import power_law_func

        q = np.array([0.01, 0.1, 1.0])
        tau = power_law_func(q, tau0=1.0, alpha=2.0)

        # tau ~ 1/q^2
        assert tau[0] > tau[1] > tau[2]
        assert np.isclose(tau[2], 1.0)  # tau0 * 1^-2 = 1
