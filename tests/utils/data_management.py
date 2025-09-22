#!/usr/bin/env python3
"""
Advanced Test Data Management for XPCS Toolkit

This module provides enhanced test data generation, caching, and management
capabilities for comprehensive testing across the XPCS Toolkit test suite.

Created: 2025-09-16
"""

import hashlib
import json
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

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


@dataclass
class TestDataSpec:
    """Specification for test data generation."""

    data_type: str  # 'xpcs', 'saxs', 'correlation', 'detector'
    shape: tuple | None = None
    size_mb: float | None = None
    noise_level: float = 0.02
    seed: int = 42
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for hashing."""
        return asdict(self)

    def hash_key(self) -> str:
        """Generate unique hash key for this specification."""
        spec_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(spec_str.encode()).hexdigest()


class TestDataCache:
    """Cache for generated test data to avoid regeneration."""

    def __init__(self, cache_dir: Path | None = None, max_cache_mb: float = 100.0):
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "xpcs_test_cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_mb = max_cache_mb
        self._index_file = self.cache_dir / "cache_index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file) as f:
                    self._index = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2)
        except OSError:
            pass  # Ignore cache save errors

    def get(self, spec: TestDataSpec) -> dict[str, Any] | None:
        """Retrieve cached data if available."""
        key = spec.hash_key()

        if key not in self._index:
            return None

        entry = self._index[key]
        cache_file = self.cache_dir / f"{key}.npz"

        if not cache_file.exists():
            # Remove stale index entry
            del self._index[key]
            self._save_index()
            return None

        try:
            # Load cached data
            loaded_data = dict(np.load(cache_file, allow_pickle=True))

            # Update access time
            entry["last_access"] = time.time()
            self._save_index()

            return loaded_data

        except (OSError, ValueError):
            # Remove corrupted cache entry
            cache_file.unlink(missing_ok=True)
            del self._index[key]
            self._save_index()
            return None

    def put(self, spec: TestDataSpec, data: dict[str, Any]) -> None:
        """Store data in cache."""
        key = spec.hash_key()
        cache_file = self.cache_dir / f"{key}.npz"

        try:
            # Save data
            np.savez_compressed(cache_file, **data)

            # Update index
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            self._index[key] = {
                "spec": spec.to_dict(),
                "file_size_mb": file_size_mb,
                "created": time.time(),
                "last_access": time.time(),
                "file_path": str(cache_file),
            }

            # Clean up if cache is too large
            self._cleanup_cache()
            self._save_index()

        except (OSError, ValueError):
            # Remove failed cache entry
            cache_file.unlink(missing_ok=True)

    def _cleanup_cache(self) -> None:
        """Clean up cache if it exceeds size limit."""
        total_size = sum(entry["file_size_mb"] for entry in self._index.values())

        if total_size <= self.max_cache_mb:
            return

        # Remove oldest accessed files first
        sorted_entries = sorted(self._index.items(), key=lambda x: x[1]["last_access"])

        for key, entry in sorted_entries:
            cache_file = Path(entry["file_path"])
            cache_file.unlink(missing_ok=True)
            del self._index[key]

            total_size -= entry["file_size_mb"]
            if total_size <= self.max_cache_mb * 0.8:  # Leave some headroom
                break

    def clear(self) -> None:
        """Clear all cached data."""
        for entry in self._index.values():
            cache_file = Path(entry["file_path"])
            cache_file.unlink(missing_ok=True)

        self._index.clear()
        self._save_index()


class AdvancedTestDataFactory:
    """Factory for creating sophisticated test datasets."""

    def __init__(self, cache: TestDataCache | None = None):
        self.cache = cache or TestDataCache()
        self._generators = {
            "xpcs": self._generate_xpcs_data,
            "saxs": self._generate_saxs_data,
            "correlation": self._generate_correlation_data,
            "detector": self._generate_detector_data,
            "qmap": self._generate_qmap_data,
            "twotime": self._generate_twotime_data,
        }

    def create_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Create test data according to specification."""
        # Check cache first
        cached_data = self.cache.get(spec)
        if cached_data is not None:
            return cached_data

        # Generate new data
        generator = self._generators.get(spec.data_type)
        if generator is None:
            raise ValueError(f"Unknown data type: {spec.data_type}")

        np.random.seed(spec.seed)
        data = generator(spec)

        # Cache the result
        self.cache.put(spec, data)

        return data

    def _generate_xpcs_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Generate complete XPCS dataset."""
        # Determine size
        if spec.size_mb is not None:
            # Estimate array sizes based on target memory usage
            total_elements = int(
                (spec.size_mb * 1024 * 1024) / 8
            )  # 8 bytes per float64
            n_frames = min(1000, max(10, int(total_elements**0.5)))
            detector_size = int((total_elements / n_frames) ** 0.5)
            shape = (detector_size, detector_size)
        else:
            shape = spec.shape or (512, 512)
            n_frames = 100

        # Generate SAXS 2D data
        saxs_2d = np.random.poisson(100, (n_frames, *shape)).astype(np.float32)

        # Add realistic structure (rings)
        center = (shape[0] // 2, shape[1] // 2)
        y, x = np.ogrid[: shape[0], : shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        # Add scattering rings
        for ring_r in [50, 100, 150]:
            if ring_r < min(shape) // 2:
                ring_mask = np.abs(r - ring_r) < 5
                saxs_2d[:, ring_mask] *= 2.0

        # Generate correlation data
        n_tau = 50
        tau = np.logspace(-6, 2, n_tau)
        g2 = 1 + 0.8 * np.exp(-tau / 1e-3)
        g2_noise = g2 + np.random.normal(0, spec.noise_level * g2)
        g2_err = spec.noise_level * g2

        return {
            "saxs_2d": saxs_2d,
            "g2": g2_noise,
            "g2_err": g2_err,
            "tau": tau,
            "n_frames": n_frames,
            "detector_shape": shape,
            "metadata": spec.metadata,
        }

    def _generate_saxs_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Generate SAXS scattering data."""
        shape = spec.shape or (1024, 1024)

        # Create realistic SAXS pattern
        center = (shape[0] // 2, shape[1] // 2)
        y, x = np.ogrid[: shape[0], : shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        # Power-law scattering with Poisson noise
        base_intensity = 1000 * (1 + r) ** (-2.5)
        intensity = np.random.poisson(base_intensity).astype(np.float32)

        # Add beam stop
        beam_stop_mask = r < 20
        intensity[beam_stop_mask] = 0

        return {
            "intensity": intensity,
            "shape": shape,
            "center": center,
            "metadata": spec.metadata,
        }

    def _generate_correlation_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Generate correlation function data."""
        n_tau = spec.metadata.get("n_tau", 50)
        tau_min = spec.metadata.get("tau_min", 1e-6)
        tau_max = spec.metadata.get("tau_max", 1e2)

        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

        # Multi-exponential decay
        beta1 = spec.metadata.get("beta1", 0.6)
        beta2 = spec.metadata.get("beta2", 0.2)
        tau1 = spec.metadata.get("tau1", 1e-3)
        tau2 = spec.metadata.get("tau2", 1e-2)

        g2 = 1 + beta1 * np.exp(-tau / tau1) + beta2 * np.exp(-tau / tau2)
        g2_noise = g2 + np.random.normal(0, spec.noise_level * g2)
        g2_err = spec.noise_level * g2

        return {
            "tau": tau,
            "g2": g2_noise,
            "g2_clean": g2,
            "g2_err": g2_err,
            "fit_params": {"beta1": beta1, "beta2": beta2, "tau1": tau1, "tau2": tau2},
            "metadata": spec.metadata,
        }

    def _generate_detector_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Generate detector geometry and calibration data."""
        shape = spec.shape or (1024, 1024)

        # Standard detector parameters
        pixel_size = spec.metadata.get("pixel_size", 75e-6)  # meters
        det_dist = spec.metadata.get("det_dist", 5.0)  # meters
        wavelength = spec.metadata.get("wavelength", 1.55e-10)  # meters
        beam_center = spec.metadata.get("beam_center", (shape[0] / 2, shape[1] / 2))

        # Create coordinate grids
        y_coords = (np.arange(shape[0]) - beam_center[0]) * pixel_size
        x_coords = (np.arange(shape[1]) - beam_center[1]) * pixel_size
        X, Y = np.meshgrid(x_coords, y_coords)

        # Calculate Q-space mapping
        r_pixel = np.sqrt(X**2 + Y**2)
        theta = 0.5 * np.arctan(r_pixel / det_dist)
        q_magnitude = 4 * np.pi * np.sin(theta) / wavelength
        phi = np.arctan2(Y, X) * 180 / np.pi

        # Create mask (detector active area)
        mask = np.ones(shape, dtype=bool)
        # Remove beam stop
        mask[r_pixel < pixel_size * 20] = False
        # Remove detector edges
        mask[:10, :] = False
        mask[-10:, :] = False
        mask[:, :10] = False
        mask[:, -10:] = False

        return {
            "q_magnitude": q_magnitude.astype(np.float32),
            "phi": phi.astype(np.float32),
            "mask": mask,
            "beam_center": beam_center,
            "pixel_size": pixel_size,
            "det_dist": det_dist,
            "wavelength": wavelength,
            "shape": shape,
            "metadata": spec.metadata,
        }

    def _generate_qmap_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Generate Q-space mapping data."""
        detector_data = self._generate_detector_data(spec)

        # Create Q-space bins
        q_max = np.nanmax(detector_data["q_magnitude"])
        n_q_bins = spec.metadata.get("n_q_bins", 50)
        n_phi_bins = spec.metadata.get("n_phi_bins", 36)

        q_bins = np.linspace(0, q_max, n_q_bins + 1)
        phi_bins = np.linspace(-180, 180, n_phi_bins + 1)

        # Create digital maps
        dqmap = np.digitize(detector_data["q_magnitude"], q_bins) - 1
        sqmap = dqmap.copy()  # For simplicity, use same mapping

        # Phi mapping
        dphi_map = np.digitize(detector_data["phi"], phi_bins) - 1
        sphi_map = dphi_map.copy()

        # Apply mask
        mask = detector_data["mask"]
        dqmap[~mask] = -1
        sqmap[~mask] = -1
        dphi_map[~mask] = -1
        sphi_map[~mask] = -1

        return {
            "dqmap": dqmap.astype(np.int32),
            "sqmap": sqmap.astype(np.int32),
            "dqlist": q_bins[:-1],
            "sqlist": q_bins[:-1],
            "dplist": phi_bins[:-1],
            "splist": phi_bins[:-1],
            "dynamic_num_pts": n_q_bins,
            "static_num_pts": n_q_bins,
            "mask": mask.astype(np.int32),
            "q_magnitude": detector_data["q_magnitude"],
            "phi": detector_data["phi"],
            **{
                k: v
                for k, v in detector_data.items()
                if k not in ["q_magnitude", "phi", "mask"]
            },
            "metadata": spec.metadata,
        }

    def _generate_twotime_data(self, spec: TestDataSpec) -> dict[str, Any]:
        """Generate two-time correlation data."""
        shape = spec.shape or (100, 100)

        # Generate realistic two-time correlation matrix
        g2_matrix = np.ones(shape)

        # Add diagonal structure (equal time correlations)
        for i in range(shape[0]):
            for j in range(shape[1]):
                time_diff = abs(i - j)
                decay_factor = np.exp(-time_diff / 10.0)  # Correlation time
                g2_matrix[i, j] = 1.0 + 0.5 * decay_factor

        # Add noise
        noise = np.random.normal(0, spec.noise_level, shape)
        g2_matrix += noise

        # Ensure physical constraint g2 >= 1
        g2_matrix = np.maximum(g2_matrix, 1.0)

        # Time axes
        elapsed_time = np.linspace(0, 100, shape[0])  # seconds

        return {
            "g2": g2_matrix.astype(np.float32),
            "elapsed_time": elapsed_time,
            "shape": shape,
            "metadata": spec.metadata,
        }


class HDF5TestDataManager:
    """Manager for creating and managing HDF5 test files."""

    def __init__(self, factory: AdvancedTestDataFactory | None = None):
        self.factory = factory or AdvancedTestDataFactory()
        self._temp_files = []

    def create_xpcs_file(
        self,
        file_path: Path,
        data_specs: dict[str, TestDataSpec],
        nexus_structure: bool = True,
    ) -> Path:
        """Create comprehensive XPCS HDF5 file."""

        # Generate all required data
        generated_data = {}
        for name, spec in data_specs.items():
            generated_data[name] = self.factory.create_data(spec)

        with h5py.File(file_path, "w") as f:
            if nexus_structure:
                self._create_nexus_structure(f)

            # Create XPCS group
            xpcs = f.create_group("xpcs")

            # Add Q-mapping data
            if "qmap" in generated_data:
                qmap_group = xpcs.create_group("qmap")
                qmap_data = generated_data["qmap"]
                for key, value in qmap_data.items():
                    if key != "metadata":
                        qmap_group.create_dataset(key, data=value)

            # Add multi-tau correlation data
            if "correlation" in generated_data:
                multitau = xpcs.create_group("multitau")
                corr_data = generated_data["correlation"]

                # Reshape for multi-q analysis
                n_q = 5  # Multiple Q-points
                multitau.create_dataset("g2", data=np.tile(corr_data["g2"], (n_q, 1)))
                multitau.create_dataset(
                    "g2_err", data=np.tile(corr_data["g2_err"], (n_q, 1))
                )
                multitau.create_dataset("tau", data=corr_data["tau"])

                # Add fit results if available
                if "fit_params" in corr_data:
                    fit_group = multitau.create_group("fit_results")
                    for param, value in corr_data["fit_params"].items():
                        fit_group.create_dataset(param, data=np.full(n_q, value))

            # Add SAXS data
            if "xpcs" in generated_data:
                xpcs_data = generated_data["xpcs"]
                multitau = xpcs.get("multitau") or xpcs.create_group("multitau")

                # 1D SAXS
                saxs_1d = np.mean(xpcs_data["saxs_2d"], axis=(1, 2))
                multitau.create_dataset("saxs_1d", data=saxs_1d.reshape(1, -1))

                # Full 2D data (compressed)
                multitau.create_dataset(
                    "saxs_2d",
                    data=xpcs_data["saxs_2d"],
                    compression="gzip",
                    compression_opts=9,
                )

            # Add two-time data if requested
            if "twotime" in generated_data:
                twotime = xpcs.create_group("twotime")
                tt_data = generated_data["twotime"]
                twotime.create_dataset("g2", data=tt_data["g2"])
                twotime.create_dataset("elapsed_time", data=tt_data["elapsed_time"])

            # Add metadata
            metadata_group = f.create_group("metadata")
            metadata_group.create_dataset("created", data=str(time.time()))
            metadata_group.create_dataset("generator", data="AdvancedTestDataFactory")

        self._temp_files.append(file_path)
        return file_path

    def _create_nexus_structure(self, f: h5py.File) -> None:
        """Create basic NeXus structure."""
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("start_time", data="2024-01-01T00:00:00")

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("detector_number", data=1)

        source = instrument.create_group("source")
        source.attrs["NX_class"] = "NXsource"
        source.create_dataset("name", data="Advanced Photon Source")

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for file_path in self._temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError:
                pass
        self._temp_files.clear()


# Global instances
_data_cache = TestDataCache()
_data_factory = AdvancedTestDataFactory(_data_cache)
_hdf5_manager = HDF5TestDataManager(_data_factory)


def get_test_data_factory() -> AdvancedTestDataFactory:
    """Get the global test data factory."""
    return _data_factory


def get_hdf5_manager() -> HDF5TestDataManager:
    """Get the global HDF5 test data manager."""
    return _hdf5_manager


@contextmanager
def temporary_xpcs_file(data_specs: dict[str, TestDataSpec], **kwargs):
    """Context manager for temporary XPCS file."""

    temp_file = Path(tempfile.mktemp(suffix=".hdf"))
    try:
        file_path = _hdf5_manager.create_xpcs_file(temp_file, data_specs, **kwargs)
        yield str(file_path)
    finally:
        if temp_file.exists():
            temp_file.unlink()


# Convenience functions for common test data patterns
def create_minimal_test_data(seed: int = 42) -> dict[str, Any]:
    """Create minimal test data for basic tests."""
    spec = TestDataSpec("correlation", seed=seed, metadata={"n_tau": 20})
    return _data_factory.create_data(spec)


def create_performance_test_data(
    size_mb: float = 10.0, seed: int = 42
) -> dict[str, Any]:
    """Create larger datasets for performance testing."""
    spec = TestDataSpec("xpcs", size_mb=size_mb, seed=seed)
    return _data_factory.create_data(spec)


def create_realistic_xpcs_dataset(
    detector_shape: tuple = (512, 512), n_frames: int = 100, seed: int = 42
) -> dict[str, Any]:
    """Create realistic XPCS dataset for integration tests."""
    metadata = {
        "n_frames": n_frames,
        "pixel_size": 75e-6,
        "det_dist": 5.0,
        "wavelength": 1.55e-10,
    }

    spec = TestDataSpec("xpcs", shape=detector_shape, seed=seed, metadata=metadata)
    return _data_factory.create_data(spec)
