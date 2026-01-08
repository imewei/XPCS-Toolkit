"""Baseline benchmarks for model functions.

Establishes performance baselines before JIT optimization.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import model functions to benchmark
try:
    import jax.numpy as jnp

    from xpcsviewer.fitting.models import (
        double_exp_func,
        single_exp_func,
        stretched_exp_func,
    )

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture
def model_benchmark_data() -> dict:
    """Generate data for model function benchmarks."""
    n_points = 1000
    tau_values = np.logspace(-6, 2, n_points)
    return {
        "x": jnp.array(tau_values) if JAX_AVAILABLE else tau_values,
        "tau": 1.0,
        "tau1": 0.1,
        "tau2": 10.0,
        "baseline": 1.0,
        "contrast": 0.3,
        "contrast1": 0.2,
        "contrast2": 0.1,
        "beta": 0.8,
        "n_points": n_points,
    }


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSingleExpBaseline:
    """Baseline benchmark for single exponential function."""

    def test_single_exp_baseline_correctness(self, model_benchmark_data: dict) -> None:
        """Verify single_exp_func produces correct results."""
        data = model_benchmark_data
        result = single_exp_func(
            data["x"], data["tau"], data["baseline"], data["contrast"]
        )

        # Should return an array of same length
        assert len(result) == data["n_points"]

        # Values should be between baseline and baseline + contrast
        assert float(jnp.min(result)) >= data["baseline"] - 0.01
        assert float(jnp.max(result)) <= data["baseline"] + data["contrast"] + 0.01

    def test_single_exp_baseline_timing(
        self, model_benchmark_data: dict, benchmark
    ) -> None:
        """Record baseline timing for single_exp_func."""
        data = model_benchmark_data

        def run_single_exp():
            return single_exp_func(
                data["x"], data["tau"], data["baseline"], data["contrast"]
            )

        # Run benchmark
        result = benchmark(run_single_exp)

        # Verify result is valid
        assert result is not None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestDoubleExpBaseline:
    """Baseline benchmark for double exponential function."""

    def test_double_exp_baseline_correctness(self, model_benchmark_data: dict) -> None:
        """Verify double_exp_func produces correct results."""
        data = model_benchmark_data
        result = double_exp_func(
            data["x"],
            data["tau1"],
            data["tau2"],
            data["baseline"],
            data["contrast1"],
            data["contrast2"],
        )

        # Should return an array of same length
        assert len(result) == data["n_points"]

        # Values should be finite
        assert jnp.all(jnp.isfinite(result))

    def test_double_exp_baseline_timing(
        self, model_benchmark_data: dict, benchmark
    ) -> None:
        """Record baseline timing for double_exp_func."""
        data = model_benchmark_data

        def run_double_exp():
            return double_exp_func(
                data["x"],
                data["tau1"],
                data["tau2"],
                data["baseline"],
                data["contrast1"],
                data["contrast2"],
            )

        result = benchmark(run_double_exp)
        assert result is not None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestStretchedExpBaseline:
    """Baseline benchmark for stretched exponential function."""

    def test_stretched_exp_baseline_correctness(
        self, model_benchmark_data: dict
    ) -> None:
        """Verify stretched_exp_func produces correct results."""
        data = model_benchmark_data
        result = stretched_exp_func(
            data["x"],
            data["tau"],
            data["baseline"],
            data["contrast"],
            data["beta"],
        )

        assert len(result) == data["n_points"]
        assert jnp.all(jnp.isfinite(result))

    def test_stretched_exp_baseline_timing(
        self, model_benchmark_data: dict, benchmark
    ) -> None:
        """Record baseline timing for stretched_exp_func."""
        data = model_benchmark_data

        def run_stretched_exp():
            return stretched_exp_func(
                data["x"],
                data["tau"],
                data["baseline"],
                data["contrast"],
                data["beta"],
            )

        result = benchmark(run_stretched_exp)
        assert result is not None
