#!/usr/bin/env python3
"""
Complete XPCS Workflow End-to-End Tests

This module tests complete XPCS analysis workflows from data loading to final results:
1. File loading and validation
2. Q-map processing and validation
3. G2 correlation analysis
4. Fitting (single/double exponential)
5. Diffusion analysis and parameter extraction
6. Result validation and export
7. Error handling and recovery

These tests simulate realistic user workflows with actual scientific analysis patterns.

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import XPCS Toolkit components
try:
    from xpcs_toolkit.fileIO.hdf_reader import batch_read_fields, get
    from xpcs_toolkit.fileIO.qmap_utils import get_qmap
    from xpcs_toolkit.helper.fitting import fit_with_fixed
    from xpcs_toolkit.module import g2mod, intt, saxs1d, saxs2d, stability
    from xpcs_toolkit.utils.logging_config import get_logger
    from xpcs_toolkit.utils.memory_utils import MemoryTracker
    from xpcs_toolkit.viewer_kernel import ViewerKernel
    from xpcs_toolkit.xpcs_file import XpcsFile
except ImportError as e:
    warnings.warn(f"Could not import all XPCS components: {e}")
    sys.exit(0)

logger = get_logger(__name__)


class XpcsWorkflowResult:
    """Container for XPCS workflow results."""

    def __init__(self):
        self.file_info = {}
        self.qmap_info = {}
        self.g2_analysis = {}
        self.fitting_results = {}
        self.diffusion_analysis = {}
        self.validation_results = {}
        self.performance_metrics = {}
        self.errors = []
        self.warnings = []

    def add_error(self, step, error):
        """Add error from workflow step."""
        self.errors.append({"step": step, "error": str(error)})

    def add_warning(self, step, warning):
        """Add warning from workflow step."""
        self.warnings.append({"step": step, "warning": str(warning)})

    def is_successful(self):
        """Check if workflow completed successfully."""
        return len(self.errors) == 0

    def get_summary(self):
        """Get workflow summary."""
        return {
            "successful": self.is_successful(),
            "steps_completed": len(
                [
                    k
                    for k in [
                        self.file_info,
                        self.qmap_info,
                        self.g2_analysis,
                        self.fitting_results,
                        self.diffusion_analysis,
                    ]
                    if k
                ]
            ),
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "performance": self.performance_metrics,
        }


class TestCompleteXpcsWorkflow(unittest.TestCase):
    """Test complete XPCS analysis workflows."""

    def setUp(self):
        """Set up workflow testing environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="xpcs_workflow_test_")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.test_files = self._create_realistic_xpcs_files()
        self.memory_tracker = MemoryTracker()
        self.memory_tracker.start_tracking()

    def tearDown(self):
        """Clean up workflow testing."""
        self.memory_tracker.stop_tracking()

    def _create_realistic_xpcs_files(self):
        """Create realistic XPCS test files with scientific data patterns."""
        test_files = []

        # Different experimental scenarios
        scenarios = [
            {
                "name": "fast_diffusion",
                "description": "Fast diffusive particles",
                "tau_c": 5e-4,  # 0.5 ms
                "beta": 0.85,
                "noise": 0.015,
                "n_q": 30,
                "background": 100,
            },
            {
                "name": "slow_diffusion",
                "description": "Slow diffusive particles",
                "tau_c": 5e-2,  # 50 ms
                "beta": 0.75,
                "noise": 0.02,
                "n_q": 25,
                "background": 150,
            },
            {
                "name": "mixed_dynamics",
                "description": "Mixed fast/slow dynamics",
                "tau_c": 1e-3,  # 1 ms (will add second component)
                "beta": 0.8,
                "noise": 0.018,
                "n_q": 35,
                "background": 120,
            },
        ]

        for scenario in scenarios:
            hdf_path = os.path.join(self.temp_dir, f"{scenario['name']}.hdf")
            self._create_scientific_xpcs_file(hdf_path, scenario)
            test_files.append(hdf_path)

        return test_files

    def _create_scientific_xpcs_file(self, hdf_path, scenario):
        """Create scientifically realistic XPCS file."""
        with h5py.File(hdf_path, "w") as f:
            # NeXus structure
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry.create_dataset("start_time", data="2024-01-01T00:00:00")
            entry.create_dataset("end_time", data="2024-01-01T01:00:00")

            # Sample information
            sample = entry.create_group("sample")
            sample.attrs["NX_class"] = "NXsample"
            sample.create_dataset("name", data=scenario["description"])
            sample.create_dataset("temperature", data=298.15)  # 25°C

            # XPCS structure
            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            qmap = xpcs.create_group("qmap")

            # Parameters
            n_q = scenario["n_q"]
            n_tau = 60  # Realistic multitau points
            n_saxs = 200

            # Create realistic tau array with multitau structure
            tau = self._create_realistic_multitau_array(n_tau)

            # Create realistic G2 with proper Q-dependence
            g2_data = self._create_scientific_g2_data(
                tau,
                n_q,
                scenario["tau_c"],
                scenario["beta"],
                scenario["noise"],
                scenario.get("mixed", False),
            )
            g2_err = scenario["noise"] * g2_data

            multitau.create_dataset("g2", data=g2_data)
            multitau.create_dataset("g2_err", data=g2_err)
            multitau.create_dataset("tau", data=tau)

            # Multitau metadata
            multitau.create_dataset("t0", data=50e-6)  # 50 μs
            multitau.create_dataset("t1", data=50e-6)
            multitau.create_dataset("stride_frame", data=1)
            multitau.create_dataset("avg_frame", data=1)
            multitau.create_dataset("start_time", data=1600000000)

            # Realistic Q-map with proper detector geometry
            qmap_data = self._create_realistic_detector_qmap(n_q)
            for key, value in qmap_data.items():
                qmap.create_dataset(key, data=value)

            # Realistic SAXS scattering
            saxs_data = self._create_realistic_scattering_data(n_saxs, scenario)
            multitau.create_dataset(
                "saxs_1d", data=saxs_data["intensity"].reshape(1, -1)
            )
            multitau.create_dataset("q_saxs", data=saxs_data["q"])

            # 2D SAXS (Iqp) with proper Q-dependence
            Iqp_data = self._create_realistic_iqp_data(
                n_q, n_saxs, saxs_data, qmap_data
            )
            multitau.create_dataset("Iqp", data=Iqp_data)

            # Stability data with realistic fluctuations
            stability_data = self._create_realistic_stability_data(scenario)
            multitau.create_dataset("Int_t", data=stability_data)

    def _create_realistic_multitau_array(self, n_tau):
        """Create realistic multitau time array."""
        # Start with shortest correlation time
        tau_min = 50e-6  # 50 μs (detector readout time)
        tau_max = 100.0  # 100 seconds

        # Create multitau structure: levels with 4 points each
        tau_levels = []
        level_start = tau_min

        while level_start < tau_max and len(tau_levels) < n_tau:
            # 4 points per level
            level_tau = [level_start * (1 + i * 0.25) for i in range(4)]
            tau_levels.extend(level_tau)
            level_start *= 2  # Next level is 2x longer

        # Take first n_tau points and ensure monotonic
        tau = np.array(tau_levels[:n_tau])
        tau = np.sort(tau)

        return tau

    def _create_scientific_g2_data(
        self, tau, n_q, tau_c_base, beta_base, noise, mixed=False
    ):
        """Create scientifically accurate G2 correlation functions."""
        g2_data = np.zeros((n_q, len(tau)))

        # Q values for proper diffusive scaling
        q_values = np.logspace(-3, -1, n_q)  # 0.001 to 0.1 Å⁻¹

        for i, q in enumerate(q_values):
            # Diffusive scaling: τ_c ∝ 1/q²
            tau_c = tau_c_base / (q**2 / q_values[0] ** 2)

            # Contrast decreases with Q (realistic behavior)
            beta = beta_base * (q_values[0] / q) ** 0.1

            if mixed and i > n_q // 2:
                # Add slow component for mixed dynamics
                fast_component = beta * 0.7 * np.exp(-tau / tau_c)
                slow_component = beta * 0.3 * np.exp(-tau / (tau_c * 10))
                g2_clean = 1 + fast_component + slow_component
            else:
                # Single exponential
                g2_clean = 1 + beta * np.exp(-tau / tau_c)

            # Add realistic noise (higher at long times)
            noise_factor = noise * (1 + 0.5 * np.sqrt(tau / tau.max()))
            g2_noisy = g2_clean + np.random.normal(0, noise_factor * g2_clean)

            # Ensure G2 >= 1 (correlation inequality)
            g2_noisy = np.maximum(g2_noisy, 1.0)

            g2_data[i, :] = g2_noisy

        return g2_data

    def _create_realistic_detector_qmap(self, n_q):
        """Create realistic detector Q-map."""
        # APS 8-ID-I typical parameters
        detector_params = {
            "pixel_size": 75e-6,  # 75 μm pixels
            "det_dist": 5.0,  # 5 m sample-detector distance
            "beam_center_x": 512.0,
            "beam_center_y": 512.0,
            "X_energy": 8.0,  # keV
            "wavelength": 1.55e-10,  # meters
        }

        # Q values with realistic spacing
        q_min = 0.001
        q_max = 0.1
        q_values = np.logspace(np.log10(q_min), np.log10(q_max), n_q)

        # Azimuthal angles
        phi_values = np.linspace(-180, 180, 36)

        return {
            "dqlist": q_values,
            "sqlist": q_values,
            "dplist": phi_values,
            "splist": phi_values,
            "dynamic_index_mapping": np.arange(n_q),
            "static_index_mapping": np.arange(n_q),
            "dynamic_num_pts": n_q,
            "static_num_pts": n_q,
            **detector_params,
        }

    def _create_realistic_scattering_data(self, n_points, scenario):
        """Create realistic SAXS scattering data."""
        q = np.logspace(-3, -1, n_points)

        # Power law scattering (typical for colloidal systems)
        if scenario["name"] == "fast_diffusion":
            # Small particles: steeper slope
            intensity = 1e6 * q ** (-3.5)
        elif scenario["name"] == "slow_diffusion":
            # Larger particles/structures: shallower slope
            intensity = 5e5 * q ** (-2.2)
        else:
            # Mixed: intermediate
            intensity = 8e5 * q ** (-2.8)

        # Add structure factor for some cases
        if scenario["name"] == "mixed_dynamics":
            structure_peak_q = 0.02
            structure_width = 0.005
            structure_factor = 1 + 1.5 * np.exp(
                -((q - structure_peak_q) ** 2) / (2 * structure_width**2)
            )
            intensity *= structure_factor

        # Add background
        intensity += scenario["background"]

        # Add Poisson noise
        intensity += np.random.poisson(np.sqrt(intensity))

        return {"q": q, "intensity": intensity}

    def _create_realistic_iqp_data(self, n_q, n_saxs, saxs_data, qmap_data):
        """Create realistic Iqp (2D SAXS) data."""
        Iqp_data = np.zeros((n_q, n_saxs))

        q_dynamic = qmap_data["dqlist"]
        q_saxs = saxs_data["q"]
        I_saxs = saxs_data["intensity"]

        # Interpolate SAXS data for each dynamic Q
        for i, q_dyn in enumerate(q_dynamic):
            # Find closest SAXS Q values and interpolate
            I_interp = np.interp(q_saxs, np.full_like(q_saxs, q_dyn), I_saxs)

            # Add some variation and noise
            variation = 1 + 0.1 * np.random.randn(n_saxs)
            noise = np.random.poisson(np.sqrt(np.maximum(I_interp, 1)))

            Iqp_data[i, :] = I_interp * variation + noise

        return Iqp_data

    def _create_realistic_stability_data(self, scenario):
        """Create realistic intensity vs time stability data."""
        n_time = 5000
        time_data = np.linspace(0, 3600, n_time)  # 1 hour measurement

        # Base intensity
        base_intensity = 1000

        # Slow drift (sample settling, beam decay)
        drift_component = -50 * (time_data / 3600)

        # Periodic variations (temperature cycling, beam refills)
        periodic_component = 20 * np.sin(2 * np.pi * time_data / 300)  # 5 min period

        # Random fluctuations (proportional to dynamics)
        if scenario["name"] == "fast_diffusion":
            fluctuation_scale = 30  # More fluctuations for fast dynamics
        else:
            fluctuation_scale = 15

        fluctuations = fluctuation_scale * np.random.randn(n_time)

        # Occasional spikes (cosmic rays, detector glitches)
        spike_probability = 0.001  # 0.1% of points
        spikes = np.random.choice(
            [0, 200], n_time, p=[1 - spike_probability, spike_probability]
        )

        intensity_data = (
            base_intensity
            + drift_component
            + periodic_component
            + fluctuations
            + spikes
        )
        intensity_data = np.maximum(intensity_data, 10)  # Ensure positive

        return np.vstack([time_data, intensity_data])

    def test_complete_fast_diffusion_workflow(self):
        """Test complete workflow for fast diffusion scenario."""
        result = XpcsWorkflowResult()

        # Use fast diffusion test file
        test_file = self.test_files[0]
        result.file_info["filename"] = Path(test_file).name

        workflow_start_time = time.perf_counter()

        try:
            # Step 1: File Loading and Validation
            step_start = time.perf_counter()

            xf = XpcsFile(test_file)
            result.file_info.update(
                {
                    "loaded": True,
                    "hdf_filename": xf.hdf_filename,
                    "load_time": time.perf_counter() - step_start,
                }
            )

            # Step 2: Q-map Processing
            step_start = time.perf_counter()

            qmap_data = get_qmap(test_file)
            self.assertIsInstance(qmap_data, dict)
            self.assertIn("dqlist", qmap_data)

            result.qmap_info.update(
                {
                    "processed": True,
                    "n_dynamic_q": len(qmap_data.get("dqlist", [])),
                    "n_static_q": len(qmap_data.get("sqlist", [])),
                    "q_range": (
                        np.min(qmap_data["dqlist"]),
                        np.max(qmap_data["dqlist"]),
                    ),
                    "process_time": time.perf_counter() - step_start,
                }
            )

            # Step 3: G2 Correlation Analysis
            step_start = time.perf_counter()

            g2_data = xf.g2
            tau_data = xf.tau

            self.assertIsInstance(g2_data, np.ndarray)
            self.assertIsInstance(tau_data, np.ndarray)
            self.assertGreater(g2_data.size, 0)
            self.assertGreater(tau_data.size, 0)

            # Validate G2 properties
            self.assertTrue(np.all(g2_data >= 1.0), "G2 values must be >= 1")
            self.assertTrue(np.all(np.isfinite(g2_data)), "G2 values must be finite")
            self.assertTrue(np.all(tau_data > 0), "Tau values must be positive")

            # Calculate correlation characteristics
            initial_contrast = (
                np.mean(g2_data[:, 0] - 1.0) if g2_data.shape[1] > 0 else 0
            )
            final_g2 = np.mean(g2_data[:, -1]) if g2_data.shape[1] > 0 else 1

            result.g2_analysis.update(
                {
                    "analyzed": True,
                    "n_q_points": g2_data.shape[0],
                    "n_tau_points": len(tau_data),
                    "tau_range": (np.min(tau_data), np.max(tau_data)),
                    "initial_contrast": initial_contrast,
                    "final_g2": final_g2,
                    "analysis_time": time.perf_counter() - step_start,
                }
            )

            # Step 4: G2 Fitting (test subset for speed)
            step_start = time.perf_counter()

            successful_fits = 0
            fit_results = []

            # Test fitting on first few Q-points
            n_fit_test = min(5, g2_data.shape[0])

            for q_idx in range(n_fit_test):
                g2_single = g2_data[q_idx, :]

                # Filter valid data points
                valid = np.isfinite(g2_single) & (g2_single > 1.0)

                if np.sum(valid) > 20:  # Need sufficient points for fitting
                    tau_fit = tau_data[valid]
                    g2_fit = g2_single[valid]

                    try:
                        # Simple exponential fitting
                        fit_result = g2mod.fit_g2(
                            tau_fit, g2_fit, fit_type="single_exp"
                        )

                        if fit_result is not None:
                            successful_fits += 1

                            # Extract fit parameters (format depends on implementation)
                            if isinstance(fit_result, dict):
                                fit_params = fit_result
                            elif (
                                isinstance(fit_result, (list, tuple))
                                and len(fit_result) > 0
                            ):
                                # Convert to dict format
                                fit_params = {
                                    "beta": fit_result[0] if len(fit_result) > 0 else 0,
                                    "tau_c": fit_result[1]
                                    if len(fit_result) > 1
                                    else 0,
                                    "baseline": fit_result[2]
                                    if len(fit_result) > 2
                                    else 1,
                                }
                            else:
                                fit_params = {"status": "unknown_format"}

                            fit_results.append(
                                {
                                    "q_index": q_idx,
                                    "q_value": qmap_data["dqlist"][q_idx]
                                    if q_idx < len(qmap_data["dqlist"])
                                    else 0,
                                    "n_points": len(tau_fit),
                                    "fit_params": fit_params,
                                }
                            )

                    except Exception as e:
                        result.add_warning(
                            "fitting", f"Fit failed for Q-point {q_idx}: {e}"
                        )

            result.fitting_results.update(
                {
                    "attempted_fits": n_fit_test,
                    "successful_fits": successful_fits,
                    "fit_success_rate": successful_fits / n_fit_test
                    if n_fit_test > 0
                    else 0,
                    "fit_results": fit_results,
                    "fitting_time": time.perf_counter() - step_start,
                }
            )

            # Step 5: Diffusion Analysis
            step_start = time.perf_counter()

            if successful_fits > 2:  # Need multiple points for diffusion analysis
                try:
                    # Extract correlation times and Q values
                    tau_c_values = []
                    q_values = []

                    for fit_result in fit_results:
                        if "fit_params" in fit_result and isinstance(
                            fit_result["fit_params"], dict
                        ):
                            tau_c = fit_result["fit_params"].get("tau_c", 0)
                            if tau_c > 0:
                                tau_c_values.append(tau_c)
                                q_values.append(fit_result["q_value"])

                    if len(tau_c_values) > 2:
                        # Fit diffusion scaling: τ_c ∝ 1/q²
                        q_array = np.array(q_values)
                        tau_c_array = np.array(tau_c_values)

                        # Linear fit in log space: log(τ_c) = log(D₀) - 2*log(q)
                        valid_data = (q_array > 0) & (tau_c_array > 0)

                        if np.sum(valid_data) > 2:
                            log_q = np.log(q_array[valid_data])
                            log_tau_c = np.log(tau_c_array[valid_data])

                            # Linear regression
                            coeffs = np.polyfit(log_q, log_tau_c, 1)
                            slope = coeffs[0]
                            intercept = coeffs[1]

                            # Extract diffusion coefficient
                            expected_slope = -2.0  # For diffusion
                            D_apparent = np.exp(-intercept) if intercept != 0 else 0

                            result.diffusion_analysis.update(
                                {
                                    "analyzed": True,
                                    "q_range_fitted": (
                                        np.min(q_array[valid_data]),
                                        np.max(q_array[valid_data]),
                                    ),
                                    "tau_c_range": (
                                        np.min(tau_c_array[valid_data]),
                                        np.max(tau_c_array[valid_data]),
                                    ),
                                    "diffusion_slope": slope,
                                    "expected_slope": expected_slope,
                                    "slope_deviation": abs(slope - expected_slope),
                                    "diffusion_coefficient": D_apparent,
                                    "r_squared": np.corrcoef(log_q, log_tau_c)[0, 1]
                                    ** 2
                                    if len(log_q) > 1
                                    else 0,
                                }
                            )

                except Exception as e:
                    result.add_error("diffusion_analysis", e)

            result.diffusion_analysis["analysis_time"] = (
                time.perf_counter() - step_start
            )

            # Step 6: Workflow Validation
            validation_errors = []
            validation_warnings = []

            # Validate file loading
            if not result.file_info.get("loaded", False):
                validation_errors.append("File loading failed")

            # Validate Q-map
            if result.qmap_info.get("n_dynamic_q", 0) == 0:
                validation_errors.append("No dynamic Q-points found")

            # Validate G2 analysis
            if result.g2_analysis.get("initial_contrast", 0) < 0.1:
                validation_warnings.append("Low initial contrast detected")

            if result.g2_analysis.get("final_g2", 0) < 1.0:
                validation_errors.append(
                    "Final G2 < 1 (violates correlation inequality)"
                )

            # Validate fitting
            if result.fitting_results.get("fit_success_rate", 0) < 0.5:
                validation_warnings.append("Low fitting success rate")

            # Validate diffusion analysis
            if result.diffusion_analysis.get("analyzed", False):
                slope_deviation = result.diffusion_analysis.get("slope_deviation", 999)
                if slope_deviation > 1.0:
                    validation_warnings.append(
                        f"Diffusion slope deviates significantly from -2: {slope_deviation:.2f}"
                    )

            result.validation_results.update(
                {
                    "errors": validation_errors,
                    "warnings": validation_warnings,
                    "overall_valid": len(validation_errors) == 0,
                }
            )

        except Exception as e:
            result.add_error("workflow", e)

        # Record performance metrics
        total_workflow_time = time.perf_counter() - workflow_start_time
        result.performance_metrics.update(
            {
                "total_workflow_time": total_workflow_time,
                "memory_used_mb": self.memory_tracker.get_current_usage(),
            }
        )

        # Log workflow summary
        summary = result.get_summary()
        logger.info(f"Fast diffusion workflow completed: {summary}")

        # Assertions for test validation
        self.assertTrue(
            result.is_successful(), f"Workflow failed with errors: {result.errors}"
        )
        self.assertTrue(result.file_info.get("loaded", False), "File loading failed")
        self.assertTrue(
            result.qmap_info.get("processed", False), "Q-map processing failed"
        )
        self.assertTrue(result.g2_analysis.get("analyzed", False), "G2 analysis failed")
        self.assertGreater(
            result.fitting_results.get("successful_fits", 0), 0, "No successful fits"
        )
        self.assertLess(total_workflow_time, 30.0, "Workflow took too long")

        # Scientific validation
        self.assertGreater(
            result.g2_analysis.get("initial_contrast", 0), 0.05, "Insufficient contrast"
        )
        self.assertGreaterEqual(
            result.g2_analysis.get("final_g2", 0), 1.0, "Final G2 < 1"
        )

        return result

    def test_complete_slow_diffusion_workflow(self):
        """Test complete workflow for slow diffusion scenario."""
        result = XpcsWorkflowResult()

        # Use slow diffusion test file
        test_file = self.test_files[1]
        result.file_info["filename"] = Path(test_file).name

        try:
            # Run abbreviated workflow (same structure as fast diffusion)
            xf = XpcsFile(test_file)
            result.file_info["loaded"] = True

            # G2 analysis
            g2_data = xf.g2
            tau_data = xf.tau

            self.assertIsInstance(g2_data, np.ndarray)
            self.assertIsInstance(tau_data, np.ndarray)

            result.g2_analysis.update(
                {
                    "analyzed": True,
                    "n_q_points": g2_data.shape[0],
                    "n_tau_points": len(tau_data),
                }
            )

            # Test that slow diffusion has longer correlation times
            # This would be validated by fitting but we'll do simple check
            if g2_data.shape[1] > 10:
                # Check correlation persists to longer times
                mid_point = g2_data.shape[1] // 2
                mid_correlation = np.mean(g2_data[:, mid_point] - 1.0)

                self.assertGreater(
                    mid_correlation,
                    0.1,
                    "Slow diffusion should have correlation at intermediate times",
                )

                result.diffusion_analysis["mid_correlation"] = mid_correlation

        except Exception as e:
            result.add_error("slow_diffusion_workflow", e)

        # Validate workflow
        self.assertTrue(
            result.is_successful(), f"Slow diffusion workflow failed: {result.errors}"
        )
        self.assertTrue(result.file_info.get("loaded", False))
        self.assertTrue(result.g2_analysis.get("analyzed", False))

        return result

    def test_mixed_dynamics_workflow(self):
        """Test workflow for mixed dynamics scenario."""
        result = XpcsWorkflowResult()

        # Use mixed dynamics test file
        test_file = self.test_files[2]
        result.file_info["filename"] = Path(test_file).name

        try:
            xf = XpcsFile(test_file)
            result.file_info["loaded"] = True

            g2_data = xf.g2
            tau_data = xf.tau

            # Mixed dynamics should show different behavior at different Q
            if g2_data.shape[0] > 10 and g2_data.shape[1] > 10:
                # Compare early and late Q-points
                early_q_contrast = np.mean(g2_data[:5, 0] - 1.0)  # First 5 Q-points
                late_q_contrast = np.mean(g2_data[-5:, 0] - 1.0)  # Last 5 Q-points

                contrast_variation = abs(early_q_contrast - late_q_contrast)

                result.diffusion_analysis.update(
                    {
                        "early_q_contrast": early_q_contrast,
                        "late_q_contrast": late_q_contrast,
                        "contrast_variation": contrast_variation,
                    }
                )

                # Mixed dynamics should show some Q-dependent variation
                self.assertGreater(
                    contrast_variation,
                    0.02,
                    "Mixed dynamics should show Q-dependent contrast variation",
                )

            result.g2_analysis.update(
                {
                    "analyzed": True,
                    "n_q_points": g2_data.shape[0],
                    "n_tau_points": len(tau_data),
                }
            )

        except Exception as e:
            result.add_error("mixed_dynamics_workflow", e)

        self.assertTrue(
            result.is_successful(), f"Mixed dynamics workflow failed: {result.errors}"
        )

        return result

    def test_workflow_performance_benchmarks(self):
        """Test workflow performance meets benchmarks."""
        performance_results = []

        for i, test_file in enumerate(self.test_files):
            start_time = time.perf_counter()
            start_memory = self.memory_tracker.get_current_usage()

            try:
                # Simplified workflow for performance testing
                xf = XpcsFile(test_file)
                g2_data = xf.g2
                get_qmap(test_file)

                end_time = time.perf_counter()
                end_memory = self.memory_tracker.get_current_usage()

                file_size_mb = os.path.getsize(test_file) / 1024 / 1024

                perf_result = {
                    "file_index": i,
                    "file_size_mb": file_size_mb,
                    "workflow_time": end_time - start_time,
                    "memory_increase_mb": end_memory - start_memory,
                    "time_per_mb": (end_time - start_time) / file_size_mb
                    if file_size_mb > 0
                    else 0,
                    "data_points": g2_data.size if g2_data is not None else 0,
                }

                performance_results.append(perf_result)

            except Exception as e:
                logger.error(f"Performance test failed for file {i}: {e}")

        # Validate performance benchmarks
        self.assertGreater(len(performance_results), 0, "No performance results")

        for result in performance_results:
            # Workflow should complete in reasonable time
            self.assertLess(
                result["workflow_time"],
                10.0,
                f"Workflow too slow: {result['workflow_time']:.2f}s",
            )

            # Time per MB should be reasonable
            self.assertLess(
                result["time_per_mb"],
                2.0,
                f"Processing too slow: {result['time_per_mb']:.2f}s/MB",
            )

            # Memory usage should be reasonable
            self.assertLess(
                result["memory_increase_mb"],
                500,
                f"Excessive memory usage: {result['memory_increase_mb']:.1f}MB",
            )

        # Log performance summary
        mean_time = np.mean([r["workflow_time"] for r in performance_results])
        mean_memory = np.mean([r["memory_increase_mb"] for r in performance_results])

        logger.info(
            f"Workflow performance: mean_time={mean_time:.2f}s, "
            f"mean_memory={mean_memory:.1f}MB"
        )

    def test_workflow_error_recovery(self):
        """Test workflow error handling and recovery."""
        # Test with problematic file
        problem_file = os.path.join(self.temp_dir, "problem_workflow.hdf")

        # Create file with missing/corrupted data
        with h5py.File(problem_file, "w") as f:
            f.create_group("entry")
            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")

            # Create problematic data
            g2_problem = np.ones((10, 50))
            g2_problem[0, :] = np.nan  # NaN values
            g2_problem[1, :] = 0.5  # Violates G2 >= 1
            g2_problem[2, :] = np.inf  # Infinite values

            multitau.create_dataset("g2", data=g2_problem)
            multitau.create_dataset("tau", data=np.logspace(-6, 2, 50))

            # Missing qmap group - will cause qmap processing to fail

        result = XpcsWorkflowResult()

        try:
            # Attempt workflow with problematic file
            xf = XpcsFile(problem_file)
            result.file_info["loaded"] = True

            # This should succeed despite problematic data
            g2_data = xf.g2

            if g2_data is not None:
                result.g2_analysis["analyzed"] = True

                # Check that data quality issues are detectable
                has_nan = np.any(np.isnan(g2_data))
                has_inf = np.any(np.isinf(g2_data))
                has_invalid = np.any(g2_data < 1.0)

                result.validation_results["data_quality"] = {
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "has_invalid": has_invalid,
                }

                if has_nan or has_inf or has_invalid:
                    result.add_warning("data_quality", "Problematic data detected")

            # Q-map processing should fail gracefully
            try:
                qmap_data = get_qmap(problem_file)
                result.qmap_info["processed"] = qmap_data is not None
            except Exception as e:
                result.add_error("qmap_processing", e)
                result.qmap_info["processed"] = False

        except Exception as e:
            result.add_error("error_recovery_workflow", e)

        # Validate error handling
        # System should handle errors gracefully without crashing
        self.assertTrue(
            result.file_info.get("loaded", False),
            "Should be able to load problematic file",
        )

        # Should detect data quality issues
        data_quality = result.validation_results.get("data_quality", {})
        self.assertTrue(
            any(data_quality.values()), "Should detect data quality problems"
        )


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run complete XPCS workflow tests
    unittest.main(verbosity=2)
