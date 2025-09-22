"""Example tests demonstrating the new pytest infrastructure.

This module shows how to use the fixtures and utilities provided
by the modernized test infrastructure.
"""

import h5py
import numpy as np
import pytest

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

    # Mock h5py for basic testing
    class MockH5pyFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_dataset(self, *args, **kwargs):
            pass

        def create_group(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

        def __setitem__(self, key, value):
            pass

    class MockH5py:
        File = MockH5pyFile

    h5py = MockH5py()

from tests.fixtures.synthetic_data import (
    create_synthetic_g2_data,
    create_synthetic_saxs_data,
)


@pytest.mark.unit
def test_synthetic_g2_data_generation(synthetic_correlation_data):
    """Test synthetic G2 data generation."""
    data = synthetic_correlation_data

    # Check data structure
    assert "tau" in data
    assert "g2" in data
    assert "g2_err" in data
    assert "g2_theory" in data

    # Check data properties
    assert len(data["tau"]) == 50  # Default fixture size
    assert len(data["g2"]) == 50
    assert np.all(data["tau"] > 0)
    assert (
        np.mean(data["g2"]) >= 1.0
    )  # Average should be >= 1 (noise may cause individual points < 1)

    # Check monotonicity
    assert np.all(np.diff(data["tau"]) > 0)


@pytest.mark.unit
def test_synthetic_saxs_data_generation(synthetic_scattering_data):
    """Test synthetic SAXS data generation."""
    data = synthetic_scattering_data

    # Check data structure
    assert "q" in data
    assert "intensity" in data
    assert "intensity_err" in data

    # Check properties
    assert np.all(data["q"] > 0)
    assert np.all(data["intensity"] > 0)
    assert np.all(data["intensity_err"] >= 0)


@pytest.mark.integration
def test_minimal_hdf5_fixture(minimal_xpcs_hdf5):
    """Test minimal HDF5 file fixture."""
    hdf_path = minimal_xpcs_hdf5
    if hdf_path:
        assert hdf_path.endswith(".h5")

        # Check file exists and has correct structure
        with h5py.File(hdf_path, "r") as f:
            assert "exchange" in f

            # Check essential datasets
            assert "exchange/tau" in f
            assert "exchange/g2" in f

            # Check metadata
            assert f.attrs["instrument"] == "APS 8-ID-I"


@pytest.mark.integration
def test_comprehensive_hdf5_fixture(comprehensive_xpcs_hdf5):
    """Test comprehensive HDF5 file fixture."""
    hdf_path = comprehensive_xpcs_hdf5
    if hdf_path:
        with h5py.File(hdf_path, "r") as f:
            # Check enhanced structure
            assert "exchange" in f
            assert "saxs_1d" in f
            assert "qmap" in f

            # Check correlation data
            assert "exchange/tau" in f
            assert "exchange/g2" in f
            assert "exchange/g2_err" in f

            # Check metadata
            assert f.attrs["instrument"] == "APS 8-ID-I"

            # Check data dimensions
            g2_data = f["exchange/g2"][:]
            assert len(g2_data) == 50  # default tau points


@pytest.mark.unit
def test_correlation_function_validator(correlation_function_validator):
    """Test correlation function validation fixture."""
    # Valid data
    tau = np.logspace(-5, 0, 20)
    g2 = 1 + 0.5 * np.exp(-tau / 1e-3)
    g2_err = 0.01 * g2

    # Should not raise
    correlation_function_validator(tau, g2, g2_err)

    # Invalid data should raise
    with pytest.raises(AssertionError):
        # Negative tau
        correlation_function_validator(np.array([-1, 1, 2]), np.array([1.5, 1.3, 1.1]))

    with pytest.raises(AssertionError):
        # G2 < 1 (violates correlation inequality)
        correlation_function_validator(np.array([1, 2, 3]), np.array([0.5, 1.3, 1.1]))


@pytest.mark.unit
def test_array_comparison_fixture(assert_arrays_close):
    """Test custom array comparison fixture."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0001, 1.9999, 3.0002])

    # Should pass with appropriate tolerance
    assert_arrays_close(a, b, rtol=1e-3)

    # Should fail with stricter tolerance
    with pytest.raises(AssertionError):
        assert_arrays_close(a, b, rtol=1e-6)


@pytest.mark.unit
def test_performance_timer_fixture(performance_timer):
    """Test performance timer fixture."""
    import time

    performance_timer.start()
    time.sleep(0.01)  # 10ms
    elapsed = performance_timer.stop()

    assert elapsed >= 0.01
    assert elapsed < 0.1  # Should be much less than 100ms


@pytest.mark.parametrize(
    "q_min,q_max,expected_points",
    [
        (0.001, 0.1, 50),
        (0.005, 0.05, 25),
        (0.01, 0.2, 100),
    ],
)
@pytest.mark.unit
def test_parametrized_q_range(q_min, q_max, expected_points):
    """Test parametrized Q-range configurations."""
    data = create_synthetic_saxs_data(q_min=q_min, q_max=q_max)

    assert data["q"].min() >= q_min
    assert data["q"].max() <= q_max
    assert len(data["q"]) == 100  # Default number of points


@pytest.mark.scientific
def test_g2_decay_properties():
    """Test scientific properties of G2 decay."""
    # Generate data with known parameters
    tau_c = 5e-3  # 5ms correlation time
    beta = 0.8

    data = create_synthetic_g2_data(
        tau_min=1e-6,
        tau_max=1e0,
        n_tau=100,
        beta=beta,
        tau_c=tau_c,
        noise_level=0.001,  # Very low noise
    )

    # Check initial value
    g2_initial = data["g2"][0]  # At shortest time
    expected_initial = 1 + beta
    assert abs(g2_initial - expected_initial) < 0.1

    # Check long-time limit
    g2_final = data["g2"][-1]  # At longest time
    assert abs(g2_final - 1.0) < 0.1  # Should approach baseline

    # Check decay at correlation time
    tau_idx = np.argmin(abs(data["tau"] - tau_c))
    g2_at_tau_c = data["g2"][tau_idx]
    expected_at_tau_c = 1 + beta * np.exp(-1)  # e^(-1) ≈ 0.368
    assert abs(g2_at_tau_c - expected_at_tau_c) < 0.2


@pytest.mark.slow
def test_large_dataset_handling():
    """Test handling of large datasets (marked as slow)."""
    # This test would be skipped by default unless --slow is passed
    large_data = create_synthetic_g2_data(
        n_tau=1000,  # Large number of points
        noise_level=0.05,
    )

    assert len(large_data["tau"]) == 1000
    assert len(large_data["g2"]) == 1000

    # Verify memory usage is reasonable
    import sys

    size_bytes = sys.getsizeof(large_data["g2"])
    assert size_bytes < 100_000  # Less than 100KB for 1000 points
