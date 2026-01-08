"""Benchmark test configuration and fixtures.

Provides timing fixtures and utilities for performance benchmarks.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    name: str
    iterations: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_mb: float | None = None
    speedup: float | None = None
    baseline_time_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "memory_mb": self.memory_mb,
            "speedup": self.speedup,
            "baseline_time_ms": self.baseline_time_ms,
        }


@dataclass
class BenchmarkTimer:
    """Context manager for timing code blocks."""

    times: list[float] = field(default_factory=list)

    def __enter__(self) -> "BenchmarkTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000  # Convert to ms
        self.times.append(elapsed)

    @property
    def mean(self) -> float:
        """Mean execution time in milliseconds."""
        return float(np.mean(self.times)) if self.times else 0.0

    @property
    def std(self) -> float:
        """Standard deviation in milliseconds."""
        return float(np.std(self.times)) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        """Minimum execution time in milliseconds."""
        return float(np.min(self.times)) if self.times else 0.0

    @property
    def max(self) -> float:
        """Maximum execution time in milliseconds."""
        return float(np.max(self.times)) if self.times else 0.0


def run_benchmark(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    iterations: int = 100,
    warmup: int = 10,
    name: str | None = None,
) -> BenchmarkResult:
    """Run a benchmark on a function.

    Parameters
    ----------
    func : Callable
        Function to benchmark
    args : tuple
        Positional arguments to pass to function
    kwargs : dict, optional
        Keyword arguments to pass to function
    iterations : int
        Number of timing iterations
    warmup : int
        Number of warmup iterations (not timed)
    name : str, optional
        Name for the benchmark result

    Returns
    -------
    BenchmarkResult
        Benchmark timing results
    """
    kwargs = kwargs or {}

    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Force garbage collection before timing
    gc.collect()

    # Timed runs
    timer = BenchmarkTimer()
    for _ in range(iterations):
        with timer:
            func(*args, **kwargs)

    return BenchmarkResult(
        name=name or func.__name__,
        iterations=iterations,
        mean_time_ms=timer.mean,
        std_time_ms=timer.std,
        min_time_ms=timer.min,
        max_time_ms=timer.max,
    )


def compare_benchmarks(
    baseline: BenchmarkResult,
    optimized: BenchmarkResult,
    min_speedup: float = 1.0,
) -> tuple[float, bool]:
    """Compare two benchmark results.

    Parameters
    ----------
    baseline : BenchmarkResult
        Baseline (unoptimized) benchmark result
    optimized : BenchmarkResult
        Optimized benchmark result
    min_speedup : float
        Minimum acceptable speedup factor

    Returns
    -------
    tuple[float, bool]
        (speedup factor, whether min_speedup was achieved)
    """
    if optimized.mean_time_ms <= 0:
        return float("inf"), True

    speedup = baseline.mean_time_ms / optimized.mean_time_ms
    return speedup, speedup >= min_speedup


@pytest.fixture
def benchmark_timer() -> type[BenchmarkTimer]:
    """Provide BenchmarkTimer class for tests."""
    return BenchmarkTimer


@pytest.fixture
def run_benchmark_fixture() -> Callable[..., BenchmarkResult]:
    """Provide run_benchmark function for tests."""
    return run_benchmark


@pytest.fixture
def synthetic_g2_data() -> dict[str, np.ndarray]:
    """Generate synthetic G2 correlation data for benchmarks."""
    rng = np.random.default_rng(42)
    n_tau = 100
    n_q = 50

    tau = np.logspace(-6, 2, n_tau)
    g2 = np.zeros((n_q, n_tau))
    g2_err = np.zeros((n_q, n_tau))

    for i in range(n_q):
        tau_c = 10 ** rng.uniform(-3, 1)
        beta = rng.uniform(0.1, 0.5)
        baseline = 1.0
        g2[i] = baseline + beta * np.exp(-tau / tau_c)
        g2_err[i] = 0.01 * np.ones(n_tau)

    return {
        "tau": tau,
        "g2": g2,
        "g2_err": g2_err,
        "n_q": n_q,
        "n_tau": n_tau,
    }


@pytest.fixture
def synthetic_detector_data() -> dict[str, np.ndarray]:
    """Generate synthetic detector data for benchmarks."""
    rng = np.random.default_rng(42)
    size = (512, 512)

    return {
        "image": rng.random(size, dtype=np.float64),
        "mask": rng.integers(0, 2, size=size, dtype=np.int32),
        "qmap": rng.random(size, dtype=np.float64) * 0.1,
        "size": size,
    }
