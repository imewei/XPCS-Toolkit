"""
Shared H5py mock objects for testing.

This module provides standardized mock implementations of h5py objects
to eliminate duplication across test files.
"""

from pathlib import Path

import numpy as np


class MockH5pyDataset:
    """Mock h5py Dataset for testing."""

    def __init__(self, data: np.ndarray | None = None, shape: tuple | None = None):
        if data is not None:
            self._data = data
            self.shape = data.shape
            self.dtype = data.dtype
        elif shape is not None:
            self.shape = shape
            self.dtype = np.float64
            self._data = np.zeros(shape, dtype=self.dtype)
        else:
            self.shape = (100, 100)
            self.dtype = np.float64
            self._data = np.random.random(self.shape)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def value(self):
        return self._data


class MockH5pyGroup:
    """Mock h5py Group for testing."""

    def __init__(self):
        self._items = {}
        self.attrs = {}

    def __getitem__(self, key):
        return self._items.get(key, MockH5pyDataset())

    def __setitem__(self, key, value):
        self._items[key] = value

    def __contains__(self, key):
        return key in self._items

    def keys(self):
        return self._items.keys()

    def items(self):
        return self._items.items()

    def get(self, key, default=None):
        return self._items.get(key, default)

    def create_dataset(
        self,
        name: str,
        data: np.ndarray | None = None,
        shape: tuple | None = None,
        dtype: type | None = None,
    ):
        """Create a mock dataset."""
        dataset = MockH5pyDataset(data, shape)
        self._items[name] = dataset
        return dataset

    def create_group(self, name: str):
        """Create a mock group."""
        group = MockH5pyGroup()
        self._items[name] = group
        return group


class MockH5pyFile(MockH5pyGroup):
    """Mock h5py File for testing."""

    def __init__(
        self, filename: str | Path | None = None, mode: str = "r", *args, **kwargs
    ):
        super().__init__()
        self.filename = filename or "mock_file.h5"
        self.mode = mode
        self.closed = False

        # Create some default structure for XPCS files
        self._setup_default_structure()

    def _setup_default_structure(self):
        """Set up default XPCS file structure."""
        # Create common XPCS groups and datasets
        self.create_group("exchange")
        exchange = self["exchange"]
        exchange.create_dataset("data", shape=(100, 100, 50))

        # Add some metadata
        self.attrs["instrument"] = "APS 8-ID-I"
        self.attrs["sample"] = "test_sample"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the mock file."""
        self.closed = True

    def flush(self):
        """Mock flush operation."""


class MockH5py:
    """Mock h5py module for testing."""

    File = MockH5pyFile
    Dataset = MockH5pyDataset
    Group = MockH5pyGroup

    @staticmethod
    def is_hdf5(filename):
        """Mock HDF5 file check."""
        return str(filename).endswith((".h5", ".hdf5"))


# Convenience function for creating mock files
def create_mock_hdf5_file(
    filename: str = "test.h5", data_shape: tuple = (100, 100, 50)
) -> MockH5pyFile:
    """Create a mock HDF5 file with standard XPCS structure."""
    mock_file = MockH5pyFile(filename)

    # Add common XPCS datasets
    exchange = mock_file.create_group("exchange")
    exchange.create_dataset("data", shape=data_shape)

    # Add G2 analysis group
    g2_group = mock_file.create_group("g2_analysis")
    g2_group.create_dataset("correlation_function", shape=(50,))
    g2_group.create_dataset("tau", shape=(50,))

    return mock_file


# ============================================================================
# Additional Mock Utilities
# ============================================================================


class MockXpcsFile:
    """Mock XPCS file object for testing."""

    def __init__(self, filename: str = "test.h5"):
        self.filename = filename
        self.h5_file = create_mock_hdf5_file(filename)
        self._data_cache = {}

    def get_correlation_data(self):
        """Mock correlation data retrieval."""
        return {
            "tau": np.logspace(-6, 2, 50),
            "g2": 1.0 + 0.8 * np.exp(-np.logspace(-6, 2, 50) / 1e-3),
            "g2_err": 0.02 * np.ones(50),
        }

    def get_scattering_data(self):
        """Mock scattering data retrieval."""
        q = np.linspace(0.001, 0.1, 100)
        return {
            "q": q,
            "intensity": 1e6 * q ** (-2.5) + 100,
            "intensity_err": np.sqrt(1e6 * q ** (-2.5) + 100),
        }

    def close(self):
        """Mock close operation."""
        if hasattr(self.h5_file, "close"):
            self.h5_file.close()


def create_mock_xpcs_file(filename: str = "test_xpcs.h5") -> MockXpcsFile:
    """Create a comprehensive mock XPCS file."""
    return MockXpcsFile(filename)
