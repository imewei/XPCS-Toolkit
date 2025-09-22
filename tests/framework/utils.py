"""Test utilities and helper functions for XPCS Toolkit tests.

This module provides reusable utilities, patterns, and helper functions that
standardize testing practices across the XPCS Toolkit test suite.

Features:
- Common test patterns and templates
- Scientific assertion helpers
- Mock object factories
- Performance testing utilities
- Data validation helpers
- Test debugging tools
"""

import contextlib
import functools
import inspect
import logging
import os
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

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

# ============================================================================
# Test Decorators and Markers
# ============================================================================


def scientific_test(tolerance: float = 1e-7, array_tolerance: float = 1e-12):
    """
    Decorator for tests requiring scientific accuracy validation.

    Args:
        tolerance: Default floating-point comparison tolerance
        array_tolerance: Default array comparison tolerance

    Usage:
        @scientific_test(tolerance=1e-6)
        def test_correlation_calculation(self):
            # Test implementation
    """

    def decorator(func):
        func._scientific_test = True
        func._tolerance = tolerance
        func._array_tolerance = array_tolerance
        return func

    return decorator


def performance_test(max_time: float = 1.0, memory_limit_mb: float = 100.0):
    """
    Decorator for performance-sensitive tests.

    Args:
        max_time: Maximum execution time in seconds
        memory_limit_mb: Maximum memory usage in MB
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            start_memory = _get_memory_usage()

            result = func(*args, **kwargs)

            elapsed_time = time.perf_counter() - start_time
            memory_used = _get_memory_usage() - start_memory

            if elapsed_time > max_time:
                pytest.fail(
                    f"Test exceeded time limit: {elapsed_time:.3f}s > {max_time}s"
                )

            if memory_used > memory_limit_mb:
                pytest.fail(
                    f"Test exceeded memory limit: {memory_used:.1f}MB > {memory_limit_mb}MB"
                )

            return result

        wrapper._performance_test = True
        wrapper._max_time = max_time
        wrapper._memory_limit = memory_limit_mb
        return wrapper

    return decorator


def gui_test(requires_display: bool = True):
    """
    Decorator for GUI tests.

    Args:
        requires_display: Whether test requires actual display
    """

    def decorator(func):
        if requires_display and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return pytest.mark.skip("Test requires display")(func)
        return pytest.mark.gui(func)

    return decorator


# ============================================================================
# Scientific Assertion Helpers
# ============================================================================


class ScientificAssertions:
    """Helper class providing scientific assertion methods with proper tolerances."""

    @staticmethod
    def assert_arrays_close(
        actual: np.ndarray,
        expected: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 1e-14,
        msg: str = "",
    ):
        """
        Assert that two arrays are element-wise close within tolerances.

        Args:
            actual: Actual array values
            expected: Expected array values
            rtol: Relative tolerance
            atol: Absolute tolerance
            msg: Custom error message
        """
        try:
            np.testing.assert_allclose(
                actual, expected, rtol=rtol, atol=atol, err_msg=msg
            )
        except AssertionError as e:
            # Enhanced error reporting
            if actual.shape != expected.shape:
                raise AssertionError(
                    f"Shape mismatch: {actual.shape} != {expected.shape}"
                )

            diff = np.abs(actual - expected)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            max_diff = diff[max_diff_idx]

            enhanced_msg = (
                f"{msg}\nMaximum difference: {max_diff:.2e} at index {max_diff_idx}"
            )
            if msg:
                enhanced_msg = f"{msg}\n{enhanced_msg}"

            raise AssertionError(f"{enhanced_msg}\n{e!s}")

    @staticmethod
    def assert_correlation_properties(
        tau: np.ndarray, g2: np.ndarray, g2_err: np.ndarray | None = None
    ):
        """
        Assert that correlation function data satisfies physical constraints.

        Args:
            tau: Time lag array
            g2: G2 correlation values
            g2_err: Optional error bars
        """
        # Basic shape checks
        assert len(tau) == len(g2), "tau and g2 must have same length"
        if g2_err is not None:
            assert len(g2_err) == len(g2), "g2_err must have same length as g2"

        # Physical constraints
        assert np.all(tau > 0), "All tau values must be positive"
        assert np.all(g2 >= 1.0), "G2 values must be >= 1 (correlation inequality)"

        # Monotonicity (generally expected for simple systems)
        if len(tau) > 1:
            assert np.all(np.diff(tau) > 0), (
                "tau values must be monotonically increasing"
            )

        # Error bars should be positive
        if g2_err is not None:
            assert np.all(g2_err >= 0), "Error bars must be non-negative"

    @staticmethod
    def assert_scattering_properties(q: np.ndarray, intensity: np.ndarray):
        """
        Assert that scattering data satisfies physical constraints.

        Args:
            q: Scattering vector magnitude
            intensity: Scattering intensity
        """
        assert len(q) == len(intensity), "q and intensity must have same length"
        assert np.all(q >= 0), "All q values must be non-negative"
        assert np.all(intensity >= 0), "All intensity values must be non-negative"

        # Q values should be monotonically increasing (typical)
        if len(q) > 1 and not np.allclose(np.diff(q), 0):
            assert np.all(np.diff(q) > 0), "q values should be monotonically increasing"

    @staticmethod
    def assert_fits_quality(
        chi_squared: float, dof: int, p_value_threshold: float = 0.05
    ):
        """
        Assert that fit quality metrics are reasonable.

        Args:
            chi_squared: Chi-squared value
            dof: Degrees of freedom
            p_value_threshold: Minimum p-value for acceptable fit
        """
        assert chi_squared >= 0, "Chi-squared must be non-negative"
        assert dof > 0, "Degrees of freedom must be positive"

        reduced_chi_squared = chi_squared / dof

        # Warn if fit is suspiciously good or bad
        if reduced_chi_squared < 0.1:
            warnings.warn(
                f"Suspiciously good fit: χ²/dof = {reduced_chi_squared:.3f}",
                stacklevel=2,
            )
        elif reduced_chi_squared > 10.0:
            warnings.warn(
                f"Poor fit quality: χ²/dof = {reduced_chi_squared:.3f}", stacklevel=2
            )


# ============================================================================
# Mock Object Factories
# ============================================================================


class MockFactory:
    """Factory for creating consistent mock objects for testing."""

    @staticmethod
    def create_mock_xpcs_file(
        data_type: str = "multitau", q_points: int = 10, tau_points: int = 50
    ) -> Mock:
        """
        Create mock XpcsFile object with realistic data structure.

        Args:
            data_type: Type of XPCS data ("multitau", "twotime", "saxs")
            q_points: Number of Q points
            tau_points: Number of tau points

        Returns:
            Mock XpcsFile object
        """
        mock_file = Mock()
        mock_file.fname = "test_data.hdf"
        mock_file.atype = data_type.title()

        # Generate realistic test data
        if data_type == "multitau":
            mock_file.get_g2_data.return_value = (
                np.logspace(-3, -1, q_points),  # q values
                np.logspace(-6, 2, tau_points),  # tau values
                1
                + 0.8
                * np.exp(-np.logspace(-6, 2, tau_points).reshape(1, -1) / 1e-3),  # g2
                0.02 * np.ones((q_points, tau_points)),  # g2_err
                [f"q={q:.3f}" for q in np.logspace(-3, -1, q_points)],  # labels
            )

        return mock_file

    @staticmethod
    def create_mock_hdf5_file(temp_dir: Path, structure: dict[str, Any]) -> Path:
        """
        Create mock HDF5 file with specified structure.

        Args:
            temp_dir: Temporary directory for file creation
            structure: Dictionary defining HDF5 structure

        Returns:
            Path to created HDF5 file
        """
        hdf_path = temp_dir / "mock_data.hdf"

        with h5py.File(hdf_path, "w") as f:
            _create_hdf5_structure(f, structure)

        return hdf_path

    @staticmethod
    def create_mock_detector_geometry() -> dict[str, Any]:
        """Create mock detector geometry parameters."""
        return {
            "pixel_size": 75e-6,  # 75 μm pixels
            "det_dist": 5.0,  # 5 m detector distance
            "beam_center_x": 512.0,  # pixels
            "beam_center_y": 512.0,  # pixels
            "X_energy": 8.0,  # keV
            "detector_size": (1024, 1024),  # pixels
            "wavelength": 1.55e-10,  # meters
        }


# ============================================================================
# Test Data Generators
# ============================================================================


class TestDataGenerator:
    """Generators for creating synthetic test data with controlled properties."""

    @staticmethod
    def generate_correlation_data(
        tau_range: tuple[float, float] = (1e-6, 1e2),
        n_points: int = 50,
        beta: float = 0.8,
        tau_c: float = 1e-3,
        noise_level: float = 0.02,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Generate synthetic G2 correlation function data.

        Args:
            tau_range: Range of tau values (min, max)
            n_points: Number of tau points
            beta: Correlation amplitude
            tau_c: Correlation time
            noise_level: Relative noise level
            seed: Random seed for reproducibility

        Returns:
            Dictionary with tau, g2, g2_err, and metadata
        """
        np.random.seed(seed)

        # Generate logarithmic time spacing
        tau = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_points)

        # Theoretical correlation function
        g2_theory = 1 + beta * np.exp(-tau / tau_c)

        # Add realistic noise
        noise = np.random.normal(0, noise_level * g2_theory)
        g2 = g2_theory + noise
        g2_err = noise_level * g2_theory

        return {
            "tau": tau,
            "g2": g2,
            "g2_err": g2_err,
            "g2_theory": g2_theory,
            "parameters": {"beta": beta, "tau_c": tau_c, "noise_level": noise_level},
        }

    @staticmethod
    def generate_scattering_data(
        q_range: tuple[float, float] = (0.001, 0.1),
        n_points: int = 100,
        power_law: float = -2.5,
        background: float = 100,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Generate synthetic SAXS scattering data.

        Args:
            q_range: Range of Q values (min, max)
            n_points: Number of Q points
            power_law: Power law exponent
            background: Background intensity
            seed: Random seed

        Returns:
            Dictionary with q, intensity, and metadata
        """
        np.random.seed(seed)

        q = np.linspace(q_range[0], q_range[1], n_points)

        # Power law scattering + background
        intensity_theory = 1e6 * (q**power_law) + background

        # Add Poisson noise
        intensity = np.random.poisson(intensity_theory)
        intensity_err = np.sqrt(intensity)

        return {
            "q": q,
            "intensity": intensity,
            "intensity_err": intensity_err,
            "intensity_theory": intensity_theory,
            "parameters": {"power_law": power_law, "background": background},
        }


# ============================================================================
# Performance Testing Utilities
# ============================================================================


class PerformanceTimer:
    """Context manager for measuring execution time with statistical analysis."""

    def __init__(self, name: str = "", num_runs: int = 1):
        self.name = name
        self.num_runs = num_runs
        self.times = []
        self.current_start = None

    def __enter__(self):
        self.current_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.current_start
        self.times.append(elapsed)

    @property
    def mean_time(self) -> float:
        """Mean execution time."""
        return np.mean(self.times) if self.times else 0.0

    @property
    def std_time(self) -> float:
        """Standard deviation of execution time."""
        return np.std(self.times) if len(self.times) > 1 else 0.0

    @property
    def min_time(self) -> float:
        """Minimum execution time."""
        return min(self.times) if self.times else 0.0

    @property
    def max_time(self) -> float:
        """Maximum execution time."""
        return max(self.times) if self.times else 0.0

    def report(self) -> str:
        """Generate performance report."""
        if not self.times:
            return f"{self.name}: No measurements"

        return (
            f"{self.name}: "
            f"mean={self.mean_time:.3f}s ± {self.std_time:.3f}s, "
            f"min={self.min_time:.3f}s, max={self.max_time:.3f}s "
            f"({len(self.times)} runs)"
        )


# ============================================================================
# Test Debugging Tools
# ============================================================================


class TestDebugger:
    """Tools for debugging test failures and analyzing test behavior."""

    @staticmethod
    def capture_context(func: Callable) -> Callable:
        """
        Decorator to capture test context on failure.

        Usage:
            @TestDebugger.capture_context
            def test_something(self):
                # Test implementation
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Capture local variables
                frame = inspect.currentframe()
                locals_dict = frame.f_locals

                # Enhanced error message
                debug_info = f"\nTest Debug Information for {func.__name__}:\n"
                debug_info += f"Arguments: {args[1:] if args else []}\n"  # Skip self
                debug_info += f"Keyword Arguments: {kwargs}\n"
                debug_info += f"Local Variables: {list(locals_dict.keys())}\n"

                # Add to exception message
                enhanced_msg = f"{e!s}\n{debug_info}"
                raise type(e)(enhanced_msg) from e

        return wrapper

    @staticmethod
    def log_test_steps(func: Callable) -> Callable:
        """
        Decorator to log test execution steps.

        Usage:
            @TestDebugger.log_test_steps
            def test_something(self):
                # Test implementation
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info(f"Starting test: {func.__name__}")

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.info(f"Test {func.__name__} passed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"Test {func.__name__} failed after {elapsed:.3f}s: {e}")
                raise

        return wrapper


# ============================================================================
# Fixture Helpers
# ============================================================================


def create_temp_hdf5(structure: dict[str, Any], filename: str = "test.hdf") -> str:
    """
    Create temporary HDF5 file with specified structure.

    Args:
        structure: Dictionary defining HDF5 structure
        filename: Name of temporary file

    Returns:
        Path to temporary HDF5 file
    """
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, filename)

    with h5py.File(temp_file, "w") as f:
        _create_hdf5_structure(f, structure)

    return temp_file


def suppress_warnings(*warning_types):
    """
    Context manager to suppress specific warning types during testing.

    Usage:
        with suppress_warnings(UserWarning, DeprecationWarning):
            # Code that might generate warnings
    """
    return warnings.catch_warnings() if warning_types else contextlib.nullcontext()


# ============================================================================
# Internal Helper Functions
# ============================================================================


def _create_hdf5_structure(parent_group: h5py.Group, structure: dict[str, Any]):
    """Recursively create HDF5 structure from dictionary."""
    for key, value in structure.items():
        if isinstance(value, dict):
            # Create subgroup
            subgroup = parent_group.create_group(key)
            _create_hdf5_structure(subgroup, value)
        elif isinstance(value, np.ndarray):
            # Create dataset
            parent_group.create_dataset(key, data=value)
        else:
            # Create scalar dataset
            parent_group.create_dataset(key, data=value)


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# ============================================================================
# Test Pattern Templates
# ============================================================================


class TestTemplate:
    """Templates for common test patterns."""

    UNIT_TEST_TEMPLATE = '''
def test_{function_name}(self):
    """Test {description}."""
    # Arrange
    {setup_code}

    # Act
    result = {function_call}

    # Assert
    {assertions}
    '''

    PARAMETERIZED_TEST_TEMPLATE = '''
@pytest.mark.parametrize("{param_names}", {param_values})
def test_{function_name}(self, {param_names}):
    """Test {description} with various parameters."""
    # Arrange
    {setup_code}

    # Act
    result = {function_call}

    # Assert
    {assertions}
    '''

    EXCEPTION_TEST_TEMPLATE = '''
def test_{function_name}_raises_{exception_name}(self):
    """Test that {function_name} raises {exception_name} for invalid input."""
    # Arrange
    {setup_code}

    # Act & Assert
    with pytest.raises({exception_type}) as exc_info:
        {function_call}

    assert "{expected_message}" in str(exc_info.value)
    '''


# Export commonly used items
__all__ = [
    # Mock factories
    "MockFactory",
    # Performance utilities
    "PerformanceTimer",
    # Assertion helpers
    "ScientificAssertions",
    # Data generators
    "TestDataGenerator",
    # Debugging tools
    "TestDebugger",
    # Templates
    "TestTemplate",
    # Fixture helpers
    "create_temp_hdf5",
    "gui_test",
    "performance_test",
    # Decorators
    "scientific_test",
    "suppress_warnings",
]
