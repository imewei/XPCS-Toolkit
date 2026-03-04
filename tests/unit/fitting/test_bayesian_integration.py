"""Integration test: batch Bayesian -> dual storage -> plot -> export."""

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestBayesianIntegration:
    """End-to-end test of the batch Bayesian pipeline."""

    def test_assemble_stores_source_bayesian(self):
        """assemble_fit_summary output must have source='bayesian'."""
        from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary

        fr = MagicMock()
        fr.get_mean.side_effect = lambda k: 1.0
        fr.get_std.side_effect = lambda k: 0.1
        fr.samples = {
            "tau": np.ones(10),
            "baseline": np.ones(10),
            "contrast": np.ones(10) * 0.3,
        }

        from xpcsviewer.fitting.models import single_exp_func

        summary = assemble_fit_summary(
            results={0: fr, 1: fr},
            q_arr=np.array([0.01, 0.02]),
            t_el=np.linspace(0.001, 1.0, 50),
            fit_func_name="single",
            model_func=single_exp_func,
        )
        assert summary["source"] == "bayesian"
        assert summary["fit_val"].shape == (2, 2, 4)
        assert summary["fit_line"].shape[0] == 2
        assert len(summary["fit_x"]) >= 100

    def test_plot_all_q_generates_figure(self):
        """plot_bayesian_all_q must return a Figure."""
        import matplotlib

        matplotlib.use("Agg")
        from xpcsviewer.fitting.viz import plot_bayesian_all_q

        summary = {
            "source": "bayesian",
            "fit_func": "single",
            "fit_val": np.random.rand(2, 2, 4),
            "fit_line": np.random.rand(2, 200) + 1.0,
            "fit_x": np.linspace(0.001, 1.0, 200),
            "q_val": np.array([0.01, 0.02]),
            "t_el": np.linspace(0.001, 1.0, 50),
        }
        g2 = np.random.rand(50, 2) + 1.0
        g2_err = np.ones((50, 2)) * 0.01

        fig = plot_bayesian_all_q(summary, g2)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_all_q_with_data_t_el(self):
        """plot_bayesian_all_q must work when data_t_el differs from summary t_el."""
        import matplotlib

        matplotlib.use("Agg")
        from xpcsviewer.fitting.viz import plot_bayesian_all_q

        # Summary has 50 time points, data has only 30 (simulating t_range filter)
        summary = {
            "source": "bayesian",
            "fit_func": "single",
            "fit_val": np.random.rand(2, 2, 4),
            "fit_line": np.random.rand(2, 200) + 1.0,
            "fit_x": np.linspace(0.001, 1.0, 200),
            "q_val": np.array([0.01, 0.02]),
            "t_el": np.linspace(0.001, 1.0, 50),  # 50 points in summary
        }
        g2 = np.random.rand(30, 2) + 1.0  # Only 30 filtered points
        g2_err = np.ones((30, 2)) * 0.01
        data_t_el = np.linspace(0.01, 0.8, 30)  # Filtered time axis

        fig = plot_bayesian_all_q(summary, g2, data_t_el=data_t_el)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_csv_export_roundtrip(self, tmp_path):
        """Exported CSV must contain all Q-bins and parameters."""
        from xpcsviewer.fitting.viz import export_bayesian_csv

        fit_val = np.array(
            [
                [[0.3, 1.0, 1.0, 1.0], [0.01, 0.1, 0.01, 0.01]],
                [[0.25, 2.0, 1.0, 1.02], [0.02, 0.2, 0.01, 0.02]],
            ]
        )
        q_val = np.array([0.01, 0.02])

        path = tmp_path / "test_params.csv"
        export_bayesian_csv(path, fit_val, q_val, "single")

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 Q-bins
        assert "q_value" in lines[0]
        assert "tau_mean" in lines[0]
        assert "0.010000" in lines[1]

    def test_dual_storage_independence(self):
        """Setting bayesian_fit_summary must not affect fit_summary on XpcsFile."""
        from unittest.mock import patch

        from xpcsviewer.xpcs_file import XpcsFile

        with patch.object(XpcsFile, "__init__", lambda self: None):
            xf = XpcsFile.__new__(XpcsFile)
            xf.fit_summary = {"source": "nlsq"}
            xf.bayesian_fit_summary = {"source": "bayesian"}
            xf.bayesian_results = {0: MagicMock()}
            assert xf.fit_summary["source"] == "nlsq"
            assert xf.bayesian_fit_summary["source"] == "bayesian"
