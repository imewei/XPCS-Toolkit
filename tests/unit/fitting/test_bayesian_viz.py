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

        fig = plot_bayesian_all_q(bayesian_summary, g2_data)
        assert fig is not None
        assert len(fig.axes) >= 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_none_without_data(self):
        """Must return None when bayesian_summary is None."""
        from xpcsviewer.fitting.viz import plot_bayesian_all_q

        assert plot_bayesian_all_q(None, None) is None

    def test_skips_failed_q_bins(self):
        """LineCollection should exclude Q-bins flagged as failed."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.collections import LineCollection

        from xpcsviewer.fitting.viz import plot_bayesian_all_q

        # 3 Q-bins, Q-index 1 failed
        failed_mask = np.array([False, True, False])
        bayesian_summary = {
            "source": "bayesian",
            "fit_func": "single",
            "fit_val": np.random.rand(3, 2, 4),
            "fit_line": np.random.rand(3, 200) + 1.0,
            "fit_x": np.linspace(0.001, 1.0, 200),
            "q_val": np.array([0.01, 0.02, 0.03]),
            "t_el": np.linspace(0.001, 1.0, 50),
            "failed_mask": failed_mask,
        }
        # Make failed Q-bin line all NaN to be realistic
        bayesian_summary["fit_line"][1, :] = np.nan
        bayesian_summary["fit_val"][1, :, :] = np.nan

        g2_data = np.random.rand(50, 3) + 1.0

        fig = plot_bayesian_all_q(bayesian_summary, g2_data)
        assert fig is not None

        ax = fig.axes[0]
        line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_collections) == 1
        # Only 2 segments (Q-index 0 and 2), not 3
        assert len(line_collections[0].get_segments()) == 2

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_failure_indicator_shown(self):
        """Title should include success count when failed_mask is present."""
        import matplotlib

        matplotlib.use("Agg")
        from xpcsviewer.fitting.viz import plot_bayesian_all_q

        failed_mask = np.array([False, True, False])
        bayesian_summary = {
            "source": "bayesian",
            "fit_func": "single",
            "fit_val": np.random.rand(3, 2, 4),
            "fit_line": np.random.rand(3, 200) + 1.0,
            "fit_x": np.linspace(0.001, 1.0, 200),
            "q_val": np.array([0.01, 0.02, 0.03]),
            "t_el": np.linspace(0.001, 1.0, 50),
            "failed_mask": failed_mask,
        }
        g2_data = np.random.rand(50, 3) + 1.0

        fig = plot_bayesian_all_q(bayesian_summary, g2_data)
        title = fig.axes[0].get_title()
        assert "2/3" in title
        assert "succeeded" in title

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestExportBayesianResults:
    """export_bayesian_results must write CSV and figure files."""

    def test_csv_export(self, tmp_path):
        """Must write a CSV with parameter columns."""
        from xpcsviewer.fitting.viz import export_bayesian_csv

        fit_val = np.array(
            [
                [[0.3, 1.0, 1.0, 1.0], [0.01, 0.1, 0.01, 0.01]],
            ]
        )
        q_val = np.array([0.01])

        path = tmp_path / "params.csv"
        export_bayesian_csv(path, fit_val, q_val, "single")
        assert path.exists()
        content = path.read_text()
        assert "q_value" in content
        assert "tau" in content

    def test_csv_export_status_column(self, tmp_path):
        """CSV should include status column when failed_mask is provided."""
        from xpcsviewer.fitting.viz import export_bayesian_csv

        fit_val = np.array(
            [
                [[0.3, 1.0, 1.0, 1.0], [0.01, 0.1, 0.01, 0.01]],
                [[np.nan] * 4, [np.nan] * 4],
            ]
        )
        q_val = np.array([0.01, 0.02])
        failed_mask = np.array([False, True])

        path = tmp_path / "params_status.csv"
        export_bayesian_csv(path, fit_val, q_val, "single", failed_mask=failed_mask)
        assert path.exists()
        content = path.read_text()
        lines = content.strip().split("\n")
        # Header has status column
        assert "status" in lines[0]
        # First row: ok, second row: failed
        assert lines[1].endswith("ok")
        assert lines[2].endswith("failed")
