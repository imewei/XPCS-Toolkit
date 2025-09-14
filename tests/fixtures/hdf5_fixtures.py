"""HDF5 test file fixtures for XPCS Toolkit testing.

This module provides utilities for creating and managing HDF5 test files
that conform to the XPCS NeXus format used by the toolkit.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Dict, List, Optional

import h5py
import numpy as np

from .synthetic_data import SyntheticXPCSGenerator, SyntheticXPCSParameters


class XPCSTestFile:
    """Context manager for creating and managing XPCS test HDF5 files."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        file_type: str = "minimal",
        cleanup: bool = True,
        **kwargs,
    ):
        """Initialize XPCS test file.

        Parameters:
        -----------
        file_path : str, optional
            Path for the test file. If None, creates temporary file.
        file_type : str
            Type of test file: "minimal", "comprehensive", "realistic"
        cleanup : bool
            Whether to clean up file on exit
        **kwargs : dict
            Additional parameters for synthetic data generation
        """
        self.file_path = file_path
        self.file_type = file_type
        self.cleanup = cleanup
        self.temp_dir = None
        self._is_temporary = file_path is None
        self.kwargs = kwargs

    def __enter__(self) -> str:
        """Create and return path to test file."""
        if self._is_temporary:
            self.temp_dir = tempfile.mkdtemp(prefix="xpcs_test_")
            self.file_path = os.path.join(self.temp_dir, f"test_{self.file_type}.hdf")

        # Create the test file
        self._create_test_file()
        return self.file_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary files if requested."""
        if self.cleanup and self._is_temporary and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_file(self):
        """Create test file based on type."""
        if self.file_type == "minimal":
            create_minimal_hdf5(self.file_path, **self.kwargs)
        elif self.file_type == "comprehensive":
            create_comprehensive_hdf5(self.file_path, **self.kwargs)
        elif self.file_type == "realistic":
            create_realistic_hdf5(self.file_path, **self.kwargs)
        else:
            raise ValueError(f"Unknown file type: {self.file_type}")


def create_minimal_hdf5(file_path: str, n_tau: int = 20, n_q: int = 5, **kwargs) -> str:
    """Create minimal XPCS HDF5 file for basic testing.

    Parameters:
    -----------
    file_path : str
        Path where to create the file
    n_tau : int
        Number of tau points
    n_q : int
        Number of Q points
    **kwargs : dict
        Additional parameters for synthetic data

    Returns:
    --------
    str : Path to created file
    """
    # Generate synthetic data with limited size for speed
    params = SyntheticXPCSParameters(
        n_tau=n_tau,
        detector_size=(100, 100),  # Small for testing
        n_q_bins=n_q,
        **kwargs,
    )
    generator = SyntheticXPCSGenerator(params)

    # Generate required data
    qmap_data = generator.generate_qmap_data()
    multitau_data = generator.generate_multitau_data(n_q)

    with h5py.File(file_path, "w") as f:
        # Basic NeXus structure
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        # Minimal instrument info
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"

        # XPCS data structure
        xpcs = f.create_group("xpcs")

        # Q-map group
        qmap_group = xpcs.create_group("qmap")
        for key, value in qmap_data.items():
            qmap_group.create_dataset(key, data=value)

        # Multi-tau correlation data
        multitau = xpcs.create_group("multitau")
        for key, value in multitau_data.items():
            multitau.create_dataset(key, data=value)

        # Add minimal SAXS data
        saxs_data = generator.generate_scattering_data()
        multitau.create_dataset("saxs_1d", data=saxs_data["intensity"].reshape(1, -1))
        multitau.create_dataset("Iqp", data=np.random.rand(n_q, 50))
        multitau.create_dataset("Int_t", data=np.random.rand(2, 100))

    return file_path


def create_comprehensive_hdf5(
    file_path: str,
    n_tau: int = 50,
    n_q: int = 10,
    include_twotime: bool = True,
    include_stability: bool = True,
    **kwargs,
) -> str:
    """Create comprehensive XPCS HDF5 file with all analysis results.

    Parameters:
    -----------
    file_path : str
        Path where to create the file
    n_tau : int
        Number of tau points
    n_q : int
        Number of Q points
    include_twotime : bool
        Whether to include two-time correlation data
    include_stability : bool
        Whether to include stability analysis data
    **kwargs : dict
        Additional parameters for synthetic data

    Returns:
    --------
    str : Path to created file
    """
    # Generate comprehensive synthetic data
    params = SyntheticXPCSParameters(
        n_tau=n_tau,
        detector_size=(512, 512),  # Realistic size
        n_q_bins=n_q,
        **kwargs,
    )
    generator = SyntheticXPCSGenerator(params)

    # Generate all data types
    qmap_data = generator.generate_qmap_data()
    multitau_data = generator.generate_multitau_data(n_q)
    saxs_data = generator.generate_scattering_data()

    with h5py.File(file_path, "w") as f:
        # Enhanced NeXus structure
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("start_time", data="2024-01-01T00:00:00")
        entry.create_dataset("end_time", data="2024-01-01T01:00:00")
        entry.create_dataset("title", data="Comprehensive XPCS Test Dataset")

        # Detailed instrument information
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        # Source information
        source = instrument.create_group("source")
        source.attrs["NX_class"] = "NXsource"
        source.create_dataset("name", data="Advanced Photon Source")
        source.create_dataset("probe", data="x-ray")
        source.create_dataset("energy", data=params.X_energy)

        # Detector information
        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("detector_number", data=1)
        detector.create_dataset(
            "pixel_size", data=[params.pixel_size, params.pixel_size]
        )
        detector.create_dataset("distance", data=params.det_dist)

        # Sample information
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample.create_dataset("name", data="Test Sample")
        sample.create_dataset("temperature", data=298.15)  # Room temperature

        # XPCS analysis groups
        xpcs = f.create_group("xpcs")

        # Enhanced Q-map with metadata
        qmap_group = xpcs.create_group("qmap")
        for key, value in qmap_data.items():
            qmap_group.create_dataset(key, data=value)

        # Add Q-map metadata
        qmap_group.attrs["q_min"] = params.q_min
        qmap_group.attrs["q_max"] = params.q_max
        qmap_group.attrs["n_q_bins"] = params.n_q_bins
        qmap_group.attrs["n_phi_bins"] = params.n_phi_bins

        # Multi-tau with fitting results
        multitau = xpcs.create_group("multitau")
        for key, value in multitau_data.items():
            multitau.create_dataset(key, data=value)

        # Add fitting results
        fit_group = multitau.create_group("fit_results")
        fit_group.create_dataset("beta", data=np.full(n_q, params.beta))
        fit_group.create_dataset("tau_c", data=np.full(n_q, params.tau_c))
        fit_group.create_dataset("baseline", data=np.ones(n_q))
        fit_group.create_dataset("chi_squared", data=np.random.rand(n_q))
        fit_group.create_dataset("fit_quality", data=np.random.rand(n_q))

        # Enhanced SAXS data
        multitau.create_dataset("saxs_1d", data=saxs_data["intensity"].reshape(1, -1))
        multitau.create_dataset(
            "saxs_1d_err", data=saxs_data["intensity_err"].reshape(1, -1)
        )
        multitau.create_dataset("q_saxs", data=saxs_data["q"])
        multitau.create_dataset("Iqp", data=np.random.rand(n_q, len(saxs_data["q"])))

        # Intensity vs time data
        n_time_points = 1000
        int_t_data = np.random.rand(n_q, n_time_points) * 1000 + 5000
        multitau.create_dataset("Int_t", data=int_t_data)
        multitau.create_dataset("time_points", data=np.linspace(0, 1000, n_time_points))

        # Two-time correlation (if requested)
        if include_twotime:
            twotime_data = generator.generate_twotime_data(
                n_time_points=50
            )  # Smaller for testing
            twotime = xpcs.create_group("twotime")
            for key, value in twotime_data.items():
                twotime.create_dataset(key, data=value)

            # Add two-time metadata
            twotime.attrs["analysis_type"] = "two_time_correlation"
            twotime.attrs["n_time_points"] = 50

        # Stability analysis (if requested)
        if include_stability:
            stability = xpcs.create_group("stability")
            stability_time = np.linspace(0, 1000, 1000)
            stability_intensity = (
                5000
                + 500 * np.sin(2 * np.pi * stability_time / 100)
                + np.random.normal(0, 50, 1000)
            )

            stability.create_dataset("intensity_time", data=stability_intensity)
            stability.create_dataset("time_points", data=stability_time)
            stability.create_dataset(
                "mean_intensity", data=np.mean(stability_intensity)
            )
            stability.create_dataset("std_intensity", data=np.std(stability_intensity))
            stability.create_dataset("drift_rate", data=0.1)  # per second

            # Add stability metadata
            stability.attrs["analysis_type"] = "stability_analysis"
            stability.attrs["analysis_duration"] = 1000.0

    return file_path


def create_realistic_hdf5(
    file_path: str, measurement_type: str = "dynamics", **kwargs
) -> str:
    """Create realistic XPCS HDF5 file mimicking real experimental data.

    Parameters:
    -----------
    file_path : str
        Path where to create the file
    measurement_type : str
        Type of measurement: "dynamics", "static", "mixed"
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    str : Path to created file
    """
    # Use realistic parameters based on APS-8IDI beamline
    if measurement_type == "dynamics":
        params = SyntheticXPCSParameters(
            tau_min=1e-5,
            tau_max=1e1,
            n_tau=64,  # Typical multi-tau points
            beta=0.6,  # Realistic contrast
            tau_c=5e-3,  # 5ms correlation time
            noise_type="mixed",
            noise_level=0.03,
            statistics=1000,
            detector_size=(1024, 1024),
            pixel_size=75e-6,
            det_dist=8.0,  # 8m typical distance
            X_energy=7.35,  # Typical APS energy
            n_q_bins=64,
            n_phi_bins=36,
            **kwargs,
        )
    elif measurement_type == "static":
        params = SyntheticXPCSParameters(
            scattering_law="power_law",
            power_law_exponent=-3.2,  # Typical for soft matter
            background=150.0,
            detector_size=(1024, 1024),
            **kwargs,
        )
    else:  # mixed
        params = SyntheticXPCSParameters(**kwargs)

    return create_comprehensive_hdf5(
        file_path,
        n_tau=params.n_tau,
        n_q=params.n_q_bins,
        include_twotime=True,
        include_stability=True,
        **kwargs,
    )


class HDF5TestGenerator:
    """Generator class for creating multiple related test files."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize generator.

        Parameters:
        -----------
        base_dir : str, optional
            Base directory for test files. If None, uses temporary directory.
        """
        self.base_dir = base_dir
        self.temp_dir = None
        self._created_files = []

    def __enter__(self):
        """Setup temporary directory if needed."""
        if self.base_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="xpcs_test_suite_")
            self.base_dir = self.temp_dir
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary files."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_file_series(
        self,
        base_name: str = "test_series",
        n_files: int = 5,
        file_type: str = "minimal",
        vary_parameters: bool = True,
        **kwargs,
    ) -> List[str]:
        """Create series of related test files.

        Parameters:
        -----------
        base_name : str
            Base name for the file series
        n_files : int
            Number of files to create
        file_type : str
            Type of files to create
        vary_parameters : bool
            Whether to vary parameters between files
        **kwargs : dict
            Base parameters for file generation

        Returns:
        --------
        List[str] : Paths to created files
        """
        file_paths = []

        for i in range(n_files):
            file_path = os.path.join(self.base_dir, f"{base_name}_{i:03d}.hdf")

            # Vary parameters if requested
            if vary_parameters:
                varied_kwargs = kwargs.copy()
                varied_kwargs["tau_c"] = kwargs.get("tau_c", 1e-3) * (
                    1 + 0.2 * (i - n_files / 2)
                )
                varied_kwargs["beta"] = kwargs.get("beta", 0.8) * (
                    1 + 0.1 * np.random.normal()
                )
            else:
                varied_kwargs = kwargs

            # Create file
            if file_type == "minimal":
                create_minimal_hdf5(file_path, **varied_kwargs)
            elif file_type == "comprehensive":
                create_comprehensive_hdf5(file_path, **varied_kwargs)
            elif file_type == "realistic":
                create_realistic_hdf5(file_path, **varied_kwargs)

            file_paths.append(file_path)
            self._created_files.append(file_path)

        return file_paths

    def create_benchmark_files(self) -> Dict[str, str]:
        """Create standard benchmark files for performance testing."""
        benchmark_files = {}

        # Small file for unit tests
        benchmark_files["small"] = create_minimal_hdf5(
            os.path.join(self.base_dir, "benchmark_small.hdf"), n_tau=10, n_q=3
        )

        # Medium file for integration tests
        benchmark_files["medium"] = create_comprehensive_hdf5(
            os.path.join(self.base_dir, "benchmark_medium.hdf"), n_tau=50, n_q=10
        )

        # Large file for performance tests
        benchmark_files["large"] = create_comprehensive_hdf5(
            os.path.join(self.base_dir, "benchmark_large.hdf"), n_tau=100, n_q=50
        )

        self._created_files.extend(benchmark_files.values())
        return benchmark_files


# Convenience context managers
@contextmanager
def minimal_xpcs_file(**kwargs):
    """Context manager for minimal XPCS test file."""
    with XPCSTestFile(file_type="minimal", **kwargs) as file_path:
        yield file_path


@contextmanager
def comprehensive_xpcs_file(**kwargs):
    """Context manager for comprehensive XPCS test file."""
    with XPCSTestFile(file_type="comprehensive", **kwargs) as file_path:
        yield file_path


@contextmanager
def realistic_xpcs_file(**kwargs):
    """Context manager for realistic XPCS test file."""
    with XPCSTestFile(file_type="realistic", **kwargs) as file_path:
        yield file_path
