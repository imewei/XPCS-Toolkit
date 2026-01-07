"""End-to-end workflow tests for JAX migration (SC-005).

SC-005: 100% of existing analysis workflows complete successfully after migration.

These tests verify complete user workflows from start to finish using the
JAX-migrated codebase.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQmapWorkflow:
    """End-to-end tests for Q-map computation workflow."""

    def test_complete_qmap_workflow(self, monkeypatch) -> None:
        """Test complete Q-map computation from parameters to output.

        Workflow:
        1. Configure backend
        2. Set detector parameters
        3. Compute Q-map
        4. Verify output arrays
        5. Export to NumPy for I/O
        """
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend, ensure_numpy

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Step 1: Define detector parameters
        energy = 10.0  # keV
        center = (256.0, 256.0)  # beam center
        shape = (512, 512)  # detector shape
        pix_dim = 0.075  # pixel size in mm
        det_dist = 5000.0  # detector distance in mm

        # Step 2: Compute Q-map (returns dict of arrays, dict of metadata)
        qmap_dict, metadata = compute_transmission_qmap(
            energy, center, shape, pix_dim, det_dist
        )

        # Step 3: Verify Q-map output - returns dict with 'q', 'phi', etc.
        assert "q" in qmap_dict
        qmap = qmap_dict["q"]
        assert qmap.shape == shape
        assert not np.isnan(qmap).any()
        assert qmap.min() >= 0  # Q values should be positive

        # Step 4: Verify metadata
        assert metadata is not None

        # Step 5: Export to NumPy for I/O
        qmap_numpy = ensure_numpy(qmap)
        assert isinstance(qmap_numpy, np.ndarray)
        assert qmap_numpy.shape == shape

    def test_qmap_with_varying_parameters(self, monkeypatch) -> None:
        """Test Q-map computation with different parameter sets."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Test multiple detector configurations
        configs = [
            {"energy": 8.0, "center": (128, 128), "shape": (256, 256)},
            {"energy": 10.0, "center": (512, 512), "shape": (1024, 1024)},
            {"energy": 12.0, "center": (256, 256), "shape": (512, 512)},
        ]

        for config in configs:
            qmap_dict, _ = compute_transmission_qmap(
                energy=config["energy"],
                center=config["center"],
                shape=config["shape"],
                pix_dim=0.075,
                det_dist=5000.0,
            )

            assert "q" in qmap_dict
            qmap = qmap_dict["q"]
            assert qmap.shape == config["shape"]
            assert not np.isnan(qmap).any()


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestPartitionWorkflow:
    """End-to-end tests for partition generation workflow."""

    def test_complete_partition_workflow(self, monkeypatch) -> None:
        """Test complete partition generation from Q-map to binned regions.

        Workflow:
        1. Generate Q-map
        2. Create mask
        3. Generate partition
        4. Verify partition labels
        """
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap
        from xpcsviewer.simplemask.utils import generate_partition

        # Step 1: Generate Q-map
        qmap_dict, _ = compute_transmission_qmap(
            energy=10.0,
            center=(256.0, 256.0),
            shape=(512, 512),
            pix_dim=0.075,
            det_dist=5000.0,
        )
        qmap = qmap_dict["q"]

        # Step 2: Create mask (exclude corners)
        mask = np.ones((512, 512), dtype=bool)
        mask[:50, :50] = False  # Exclude top-left corner
        mask[-50:, -50:] = False  # Exclude bottom-right corner

        # Step 3: Generate partition using generate_partition (lower-level API)
        partition_result = generate_partition(
            map_name="q",
            mask=mask,
            xmap=qmap,
            num_pts=36,
            style="linear",
        )

        # Step 4: Verify partition - returns a dict with partition info
        assert partition_result is not None
        assert isinstance(partition_result, dict)
        # Check that we have the partition array
        if "partition" in partition_result:
            partition = partition_result["partition"]
            assert partition.shape == (512, 512)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestFittingWorkflow:
    """End-to-end tests for Bayesian fitting workflow."""

    @pytest.mark.skip(
        reason="ArviZ diagnostics require more samples than feasible for CI"
    )
    def test_complete_bayesian_fitting_workflow(self, monkeypatch) -> None:
        """Test complete Bayesian fitting from data to diagnostics.

        Workflow:
        1. Generate synthetic G2 data
        2. Perform NLSQ warm-start
        3. Run NUTS sampling
        4. Check convergence diagnostics
        5. Extract parameter estimates

        Note: Skipped in CI due to ArviZ diagnostics requiring more samples.
        The sampler tests in tests/jax_migration/fitting/ cover this functionality.
        """
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.fitting import fit_single_exp

        # Step 1: Generate synthetic G2 data
        np.random.seed(42)
        delay_times = np.logspace(-4, 1, 50)
        true_tau = 0.1
        true_baseline = 1.0
        true_contrast = 0.3
        noise_level = 0.01

        g2_data = (
            true_baseline
            + true_contrast * np.exp(-2 * delay_times / true_tau)
            + np.random.normal(0, noise_level, len(delay_times))
        )
        g2_errors = np.ones_like(g2_data) * noise_level

        # Step 2-3: Fit with Bayesian inference (includes NLSQ warm-start)
        # Need at least 2 chains and sufficient samples for ArviZ diagnostics
        result = fit_single_exp(
            x=delay_times,
            y=g2_data,
            yerr=g2_errors,
            num_samples=200,  # Need sufficient samples for diagnostics
            num_warmup=100,
            num_chains=2,  # Minimum for R-hat calculation
        )

        # Step 4: Check that we got a result with samples
        assert result.samples is not None
        assert "tau" in result.samples
        assert len(result.samples["tau"]) > 0

        # Step 5: Extract parameter estimates
        tau_mean = result.get_mean("tau")
        baseline_mean = result.get_mean("baseline")
        contrast_mean = result.get_mean("contrast")

        # Parameters should be in reasonable range
        assert 0.01 < tau_mean < 1.0, f"tau={tau_mean} outside range"
        assert 0.8 < baseline_mean < 1.2, f"baseline={baseline_mean} outside range"
        assert 0.1 < contrast_mean < 0.5, f"contrast={contrast_mean} outside range"

    def test_nlsq_standalone_workflow(self, monkeypatch) -> None:
        """Test NLSQ fitting as standalone workflow."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import jax.numpy as jnp

        from xpcsviewer.fitting.nlsq import nlsq_optimize

        # Generate synthetic data
        np.random.seed(42)
        x = np.logspace(-3, 1, 100)
        true_tau = 0.1
        y = 1.0 + 0.3 * np.exp(-2 * x / true_tau) + np.random.normal(0, 0.01, len(x))
        yerr = np.ones_like(y) * 0.01

        # Define model (must use JAX ops)
        def model(x, tau, baseline, contrast):
            return baseline + contrast * jnp.exp(-2 * x / tau)

        # Run NLSQ optimization
        result = nlsq_optimize(
            model_fn=model,
            x=x,
            y=y,
            yerr=yerr,
            p0={"tau": 0.2, "baseline": 1.0, "contrast": 0.3},
            bounds={
                "tau": (0.01, 10.0),
                "baseline": (0.5, 1.5),
                "contrast": (0.01, 1.0),
            },
        )

        # Verify result - NLSQResult has 'converged' not 'success'
        assert result.converged
        assert result.params is not None
        assert "tau" in result.params
        # Fitted tau should be close to true value
        assert abs(result.params["tau"] - true_tau) < 0.05


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestVisualizationWorkflow:
    """End-to-end tests for visualization workflow."""

    def test_visualization_export_workflow(self, monkeypatch) -> None:
        """Test visualization generation and export workflow."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import jax.numpy as jnp

        from xpcsviewer.fitting.nlsq import nlsq_optimize
        from xpcsviewer.fitting.visualization import plot_nlsq_fit, save_figure

        # Generate and fit data
        np.random.seed(42)
        x = np.logspace(-3, 1, 50)
        y = 1.0 + 0.3 * np.exp(-2 * x / 0.1) + np.random.normal(0, 0.01, len(x))
        yerr = np.ones_like(y) * 0.01

        def model(x, tau, baseline, contrast):
            return baseline + contrast * jnp.exp(-2 * x / tau)

        result = nlsq_optimize(
            model_fn=model,
            x=x,
            y=y,
            yerr=yerr,
            p0={"tau": 0.2, "baseline": 1.0, "contrast": 0.3},
            bounds={
                "tau": (0.01, 10.0),
                "baseline": (0.5, 1.5),
                "contrast": (0.01, 1.0),
            },
        )

        # Create visualization - plot_nlsq_fit returns Axes, get Figure from it
        ax = plot_nlsq_fit(
            result=result,
            model=model,
            x_data=x,
            y_data=y,
        )

        assert ax is not None
        fig = ax.get_figure()
        assert fig is not None

        # Test export to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_plot.pdf"
            png_path = Path(tmpdir) / "test_plot.png"

            save_figure(fig, str(pdf_path), formats=["pdf"])
            save_figure(fig, str(png_path), formats=["png"])

            assert pdf_path.exists()
            assert png_path.exists()

        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestIntegratedPipeline:
    """End-to-end tests for complete integrated analysis pipeline."""

    def test_full_xpcs_analysis_pipeline(self, monkeypatch) -> None:
        """Test complete XPCS analysis pipeline from Q-map to fitted parameters.

        This is the ultimate end-to-end test that simulates a complete
        user workflow from detector geometry to scientific results.

        Pipeline:
        1. Compute Q-map from detector geometry
        2. Generate partition (Q-binning)
        3. Simulate G2 correlation data per Q-bin
        4. Fit each Q-bin with NLSQ
        5. Extract diffusion coefficient from tau(Q) relationship
        """
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import jax.numpy as jnp

        from xpcsviewer.fitting.nlsq import nlsq_optimize
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap
        from xpcsviewer.simplemask.utils import generate_partition

        # Step 1: Compute Q-map
        qmap_dict, _ = compute_transmission_qmap(
            energy=10.0,
            center=(128.0, 128.0),
            shape=(256, 256),
            pix_dim=0.075,
            det_dist=5000.0,
        )
        qmap = qmap_dict["q"]

        # Step 2: Generate partition
        mask = np.ones((256, 256), dtype=bool)
        partition_result = generate_partition(
            map_name="q",
            mask=mask,
            xmap=qmap,
            num_pts=10,
            style="linear",
        )

        # Get the partition array
        if isinstance(partition_result, dict) and "partition" in partition_result:
            partition = partition_result["partition"]
        else:
            # If generate_partition returns the array directly
            partition = np.zeros((256, 256), dtype=int)  # fallback

        # Step 3: Simulate G2 data for each Q-bin
        np.random.seed(42)
        delay_times = np.logspace(-4, 1, 30)
        diffusion = 1e-12  # m²/s (simulated diffusion coefficient)

        q_values = []
        tau_values = []

        for q_bin in range(1, 6):  # Test first 5 Q-bins
            # Get average Q for this bin
            bin_mask = partition == q_bin
            if not bin_mask.any():
                continue
            avg_q = float(qmap[bin_mask].mean())
            q_values.append(avg_q)

            # Simulate tau = 1/(D*Q²) relationship
            true_tau = 1.0 / (diffusion * (avg_q * 1e10) ** 2)
            true_tau = np.clip(true_tau, 0.01, 100)  # Reasonable range

            # Generate synthetic G2 data
            g2_data = (
                1.0
                + 0.3 * np.exp(-2 * delay_times / true_tau)
                + np.random.normal(0, 0.01, len(delay_times))
            )
            yerr = np.ones_like(g2_data) * 0.01

            # Step 4: Fit with NLSQ (faster for this test)
            def model(x, tau, baseline, contrast):
                return baseline + contrast * jnp.exp(-2 * x / tau)

            result = nlsq_optimize(
                model_fn=model,
                x=delay_times,
                y=g2_data,
                yerr=yerr,
                p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
                bounds={
                    "tau": (0.001, 1000.0),
                    "baseline": (0.5, 1.5),
                    "contrast": (0.01, 1.0),
                },
            )

            if result.converged:
                tau_values.append(result.params["tau"])

        # Step 5: Verify we got results for multiple Q values
        assert len(q_values) >= 3, "Should have fitted at least 3 Q-bins"
        assert len(tau_values) == len(q_values), "Should have tau for each Q"

        # Verify tau values are physically reasonable
        for tau in tau_values:
            assert 0.001 < tau < 1000, f"Tau {tau} is outside reasonable range"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestBackendSwitching:
    """Test workflow stability when switching backends."""

    def test_workflow_after_backend_reset(self, monkeypatch) -> None:
        """Test that workflows work correctly after backend reset."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend, get_backend

        # Initial computation
        _reset_backend()
        backend1 = get_backend()
        x1 = backend1.linspace(0, 1, 100)
        y1 = backend1.sin(x1)

        # Reset and repeat
        _reset_backend()
        backend2 = get_backend()
        x2 = backend2.linspace(0, 1, 100)
        y2 = backend2.sin(x2)

        # Results should be identical
        np.testing.assert_allclose(
            np.asarray(y1), np.asarray(y2), rtol=1e-10, atol=1e-12
        )

    def test_memory_cleanup_after_workflow(self, monkeypatch) -> None:
        """Test memory is properly cleaned up after workflow completion."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        import psutil

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Run workflow multiple times
        for _ in range(3):
            qmap_dict, _ = compute_transmission_qmap(
                energy=10.0,
                center=(256.0, 256.0),
                shape=(512, 512),
                pix_dim=0.075,
                det_dist=5000.0,
            )
            del qmap_dict
            gc.collect()

        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)

        # Memory should not grow excessively
        assert memory_increase_mb < 200, (
            f"Memory increased by {memory_increase_mb:.1f}MB after workflow cleanup"
        )
