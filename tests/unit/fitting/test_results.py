"""Tests for fitting results module - Bayesian artifacts compliance.

Tests for Technical Guidelines compliance:
- T029: FitDiagnostics.bfmi field exists and defaults to None
- T030: FitDiagnostics.converged includes BFMI check
- T031: FitResult.to_dict() includes versions dict
- T032: FitResult.to_dict() includes sampler_config dict
- T033: FitResult.to_dict() includes data_metadata dict
"""

import numpy as np
import pytest


class TestFitDiagnosticsBFMI:
    """Test FitDiagnostics BFMI field and convergence check (T029, T030)."""

    def test_bfmi_field_exists_and_defaults_to_none(self):
        """T029: Verify bfmi field exists and defaults to None."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diagnostics = FitDiagnostics()
        assert hasattr(diagnostics, "bfmi")
        assert diagnostics.bfmi is None

    def test_bfmi_can_be_set(self):
        """Verify bfmi field can be set to a float value."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diagnostics = FitDiagnostics(bfmi=0.95)
        assert diagnostics.bfmi == 0.95

    def test_converged_passes_with_good_bfmi(self):
        """T030: Verify converged is True when bfmi >= 0.2."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diagnostics = FitDiagnostics(
            r_hat={"param": 1.0},
            ess_bulk={"param": 500},
            ess_tail={"param": 500},
            divergences=0,
            bfmi=0.85,
        )
        assert diagnostics.converged is True

    def test_converged_fails_with_low_bfmi(self):
        """T030: Verify converged is False when bfmi < 0.2."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diagnostics = FitDiagnostics(
            r_hat={"param": 1.0},
            ess_bulk={"param": 500},
            ess_tail={"param": 500},
            divergences=0,
            bfmi=0.15,  # Below threshold
        )
        assert diagnostics.converged is False

    def test_converged_passes_with_none_bfmi(self):
        """T030: Verify converged ignores bfmi when None (not computed)."""
        from xpcsviewer.fitting.results import FitDiagnostics

        diagnostics = FitDiagnostics(
            r_hat={"param": 1.0},
            ess_bulk={"param": 500},
            ess_tail={"param": 500},
            divergences=0,
            bfmi=None,  # Not computed
        )
        assert diagnostics.converged is True

    def test_converged_boundary_bfmi(self):
        """T030: Verify bfmi boundary at exactly 0.2."""
        from xpcsviewer.fitting.results import FitDiagnostics

        # Exactly at threshold - should pass
        diagnostics = FitDiagnostics(
            r_hat={"param": 1.0},
            ess_bulk={"param": 500},
            ess_tail={"param": 500},
            divergences=0,
            bfmi=0.2,
        )
        assert diagnostics.converged is True

        # Just below threshold - should fail
        diagnostics_low = FitDiagnostics(
            r_hat={"param": 1.0},
            ess_bulk={"param": 500},
            ess_tail={"param": 500},
            divergences=0,
            bfmi=0.199,
        )
        assert diagnostics_low.converged is False


class TestFitResultToDict:
    """Test FitResult.to_dict() includes required metadata (T031-T033)."""

    def test_to_dict_includes_versions(self):
        """T031: Verify to_dict() includes versions dict with 6 packages."""
        from xpcsviewer.fitting.results import FitResult

        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
        )
        data = result.to_dict()

        assert "versions" in data
        versions = data["versions"]
        assert isinstance(versions, dict)

        # Must have exactly these 6 packages per Technical Guidelines
        expected_packages = ["xpcsviewer", "numpyro", "jax", "arviz", "nlsq", "python"]
        for pkg in expected_packages:
            assert pkg in versions, f"Missing version for {pkg}"
            assert isinstance(versions[pkg], str), f"Version for {pkg} must be string"

        assert len(versions) == 6

    def test_to_dict_includes_sampler_config(self):
        """T032: Verify to_dict() includes sampler_config with 6 params."""
        from xpcsviewer.fitting.results import FitResult, SamplerConfig

        config = SamplerConfig(
            num_warmup=500,
            num_samples=1000,
            num_chains=4,
            target_accept_prob=0.8,
            max_tree_depth=10,
            random_seed=42,
        )

        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
            config=config,
        )
        data = result.to_dict()

        assert "sampler_config" in data
        sampler_config = data["sampler_config"]
        assert isinstance(sampler_config, dict)

        # Must have exactly these 6 params per SamplerConfig
        expected_params = [
            "num_warmup",
            "num_samples",
            "num_chains",
            "target_accept_prob",
            "max_tree_depth",
            "random_seed",
        ]
        for param in expected_params:
            assert param in sampler_config, f"Missing param {param}"

        assert sampler_config["num_warmup"] == 500
        assert sampler_config["num_samples"] == 1000
        assert sampler_config["num_chains"] == 4
        assert sampler_config["target_accept_prob"] == 0.8
        assert sampler_config["max_tree_depth"] == 10
        assert sampler_config["random_seed"] == 42

    def test_to_dict_sampler_config_empty_when_no_config(self):
        """T032: Verify sampler_config is empty dict when config is None."""
        from xpcsviewer.fitting.results import FitResult

        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
            config=None,
        )
        data = result.to_dict()

        assert "sampler_config" in data
        assert data["sampler_config"] == {}

    def test_to_dict_includes_data_metadata(self):
        """T033: Verify to_dict() includes data_metadata with n_points and x_range."""
        from xpcsviewer.fitting.results import FitResult

        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
            x=x_data,
        )
        data = result.to_dict()

        assert "data_metadata" in data
        data_metadata = data["data_metadata"]
        assert isinstance(data_metadata, dict)

        assert "n_points" in data_metadata
        assert data_metadata["n_points"] == 5

        assert "x_range" in data_metadata
        assert data_metadata["x_range"] == [1.0, 5.0]

    def test_to_dict_data_metadata_empty_when_no_x(self):
        """T033: Verify data_metadata is empty dict when x is None."""
        from xpcsviewer.fitting.results import FitResult

        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
            x=None,
        )
        data = result.to_dict()

        assert "data_metadata" in data
        assert data["data_metadata"] == {}

    def test_to_dict_includes_bfmi_in_diagnostics(self):
        """Verify to_dict() includes bfmi in diagnostics."""
        from xpcsviewer.fitting.results import FitDiagnostics, FitResult

        diagnostics = FitDiagnostics(bfmi=0.85)
        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
            diagnostics=diagnostics,
        )
        data = result.to_dict()

        assert "diagnostics" in data
        assert "bfmi" in data["diagnostics"]
        assert data["diagnostics"]["bfmi"] == 0.85

    def test_to_dict_bfmi_none_when_not_computed(self):
        """Verify to_dict() includes bfmi=None when not computed."""
        from xpcsviewer.fitting.results import FitResult

        result = FitResult(
            samples={"param": np.array([1.0, 2.0, 3.0])},
        )
        data = result.to_dict()

        assert "diagnostics" in data
        assert "bfmi" in data["diagnostics"]
        assert data["diagnostics"]["bfmi"] is None


class TestSafeVersion:
    """Test safe_version helper function (T025)."""

    def test_safe_version_returns_string(self):
        """Verify safe_version returns a string."""
        from xpcsviewer.fitting.results import safe_version

        version = safe_version("numpy")
        assert isinstance(version, str)

    def test_safe_version_returns_unknown_for_nonexistent(self):
        """Verify safe_version returns 'unknown' for non-existent packages."""
        from xpcsviewer.fitting.results import safe_version

        version = safe_version("nonexistent_package_12345")
        assert version == "unknown"

    def test_safe_version_never_raises(self):
        """Verify safe_version never raises exceptions."""
        from xpcsviewer.fitting.results import safe_version

        # Should not raise for any input
        safe_version("numpy")
        safe_version("")
        safe_version("nonexistent_package")
        safe_version("sys")  # Built-in module
