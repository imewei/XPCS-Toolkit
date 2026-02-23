"""Tests for Bayesian all-Q visualization and export."""
import numpy as np
import pytest


class TestPlotBayesianAllQ:
    """plot_bayesian_all_q must generate a matplotlib figure."""

    def test_returns_figure(self):
        """Must return a matplotlib Figure with axes."""
        import matplotlib
        matplotlib.use("Agg")
        from xpcsviewer.fitting.viz import plot_bayesian_all_q

        bayesian_summary = {
            "source": "bayesian",
            "fit_func": "single",
            "fit_val": np.random.rand(3, 2, 4),
            "fit_line": np.random.rand(3, 200) + 1.0,
            "fit_x": np.linspace(0.001, 1.0, 200),
            "q_val": np.array([0.01, 0.02, 0.03]),
            "t_el": np.linspace(0.001, 1.0, 50),
        }
        g2_data = np.random.rand(50, 3) + 1.0
        g2_err = np.ones((50, 3)) * 0.01

        fig = plot_bayesian_all_q(bayesian_summary, g2_data, g2_err)
        assert fig is not None
        assert len(fig.axes) >= 1
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_returns_none_without_data(self):
        """Must return None when bayesian_summary is None."""
        from xpcsviewer.fitting.viz import plot_bayesian_all_q
        assert plot_bayesian_all_q(None, None, None) is None


class TestExportBayesianResults:
    """export_bayesian_results must write CSV and figure files."""

    def test_csv_export(self, tmp_path):
        """Must write a CSV with parameter columns."""
        from xpcsviewer.fitting.viz import export_bayesian_csv

        fit_val = np.array([
            [[0.3, 1.0, 1.0, 1.0], [0.01, 0.1, 0.01, 0.01]],
        ])
        q_val = np.array([0.01])

        path = tmp_path / "params.csv"
        export_bayesian_csv(path, fit_val, q_val, "single")
        assert path.exists()
        content = path.read_text()
        assert "q_value" in content
        assert "tau" in content
