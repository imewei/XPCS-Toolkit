"""Bayesian diagnosis window for MCMC fit results.

Provides a shared diagnostic window used by both the G2 Fitting tab
and the Diffusion tab.  Contains:

- Tab 1: Posterior predictive plot with 95% credible interval
- Tab 2: Six ArviZ diagnostic plots (pair, forest, energy, autocorr, rank, ESS)
- Bottom panel: Convergence summary (R-hat, ESS, divergences, BFMI)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
)

from xpcsviewer.gui.qt_compat import (
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    Qt,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from xpcsviewer.fitting.results import FitDiagnostics, FitResult

logger = logging.getLogger(__name__)

# NOTE: Do not call matplotlib.use() here — backend is managed by
# plothandler/matplot_qt.py or the application entry point.


# ---------------------------------------------------------------------------
# Convergence summary widget
# ---------------------------------------------------------------------------


class ConvergenceSummaryWidget(QWidget):
    """Rich-text summary of MCMC convergence diagnostics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setText("<i>No results yet</i>")
        layout.addWidget(self._label)

    def update_diagnostics(self, diag: FitDiagnostics) -> None:
        """Refresh the summary from a ``FitDiagnostics`` instance."""
        rows: list[str] = []

        # Overall status
        status = (
            "<span style='color:green;font-weight:bold;'>CONVERGED</span>"
            if diag.converged
            else "<span style='color:red;font-weight:bold;'>NOT CONVERGED</span>"
        )
        rows.append(f"<b>Status:</b> {status}")

        # R-hat
        if diag.r_hat:
            rhat_parts = [f"{k}={v:.4f}" for k, v in diag.r_hat.items()]
            rows.append(f"<b>R-hat:</b> {', '.join(rhat_parts)}")

        # ESS bulk
        if diag.ess_bulk:
            ess_parts = [f"{k}={v}" for k, v in diag.ess_bulk.items()]
            rows.append(f"<b>ESS bulk:</b> {', '.join(ess_parts)}")

        # ESS tail
        if diag.ess_tail:
            ess_parts = [f"{k}={v}" for k, v in diag.ess_tail.items()]
            rows.append(f"<b>ESS tail:</b> {', '.join(ess_parts)}")

        # Divergences
        rows.append(f"<b>Divergences:</b> {diag.divergences}")

        # BFMI
        if diag.bfmi is not None:
            bfmi_color = "green" if diag.bfmi >= 0.2 else "red"
            rows.append(
                f"<b>BFMI:</b> <span style='color:{bfmi_color};'>{diag.bfmi:.3f}</span>"
            )

        self._label.setText("<br>".join(rows))


# ---------------------------------------------------------------------------
# Main diagnosis window
# ---------------------------------------------------------------------------


class BayesianDiagnosisWindow(QMainWindow):
    """Diagnostic window showing Bayesian fit results and MCMC diagnostics.

    Reused for both G2 single-Q fits and diffusion power-law fits by
    calling :meth:`update_results` with different data/labels.
    """

    _ARVIZ_PLOT_NAMES = ("pair", "forest", "energy", "autocorr", "rank", "ess")

    def __init__(
        self, parent: QWidget | None = None, title: str = "Bayesian Diagnosis"
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 750)

        # Central widget
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Splitter: top = tabs, bottom = convergence summary
        splitter = QSplitter(Qt.Orientation.Vertical, central)
        main_layout.addWidget(splitter)

        # --- Tab widget ---
        self._tabs = QTabWidget()
        splitter.addWidget(self._tabs)

        # Tab 1: Posterior predictive
        self._pp_widget = QWidget()
        pp_layout = QVBoxLayout(self._pp_widget)
        self._pp_canvas = FigureCanvasQTAgg(Figure(figsize=(7, 4)))
        pp_layout.addWidget(self._pp_canvas)
        # Export buttons for Tab 1
        pp_btn_row = QHBoxLayout()
        pp_btn_row.addStretch()
        self._btn_export_fit_plot = QPushButton("Export Plot")
        self._btn_export_fit_plot.setToolTip("Save fit + 95% CI plot as PNG/PDF")
        self._btn_export_fit_plot.clicked.connect(self._export_fit_plot)
        pp_btn_row.addWidget(self._btn_export_fit_plot)
        self._btn_export_fit_data = QPushButton("Export Data")
        self._btn_export_fit_data.setToolTip(
            "Save raw data, fitted curve, 95% CI, and residuals as CSV"
        )
        self._btn_export_fit_data.clicked.connect(self._export_fit_data)
        pp_btn_row.addWidget(self._btn_export_fit_data)
        pp_layout.addLayout(pp_btn_row)
        self._tabs.addTab(self._pp_widget, "Fit + 95% CI")

        # Tab 2: ArviZ diagnostics (2x3 grid inside scroll area)
        diag_container = QWidget()
        diag_container_layout = QVBoxLayout(diag_container)
        diag_container_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._diag_widget = QWidget()
        self._diag_layout = QGridLayout(self._diag_widget)
        self._diag_layout.setContentsMargins(4, 4, 4, 4)
        self._diag_layout.setSpacing(6)
        self._arviz_canvases: dict[str, FigureCanvasQTAgg] = {}
        for idx, name in enumerate(self._ARVIZ_PLOT_NAMES):
            canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4)))
            canvas.setMinimumSize(420, 360)
            canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            row, col = divmod(idx, 3)
            self._diag_layout.addWidget(canvas, row, col)
            self._arviz_canvases[name] = canvas
        scroll.setWidget(self._diag_widget)
        diag_container_layout.addWidget(scroll)
        # Export buttons for Tab 2
        diag_btn_row = QHBoxLayout()
        diag_btn_row.addStretch()
        self._btn_export_diag_plots = QPushButton("Export Plots")
        self._btn_export_diag_plots.setToolTip(
            "Save all 6 diagnostic plots as individual PNGs"
        )
        self._btn_export_diag_plots.clicked.connect(self._export_diag_plots)
        diag_btn_row.addWidget(self._btn_export_diag_plots)
        self._btn_export_traces = QPushButton("Export Traces")
        self._btn_export_traces.setToolTip(
            "Save ArviZ InferenceData as netCDF for further analysis"
        )
        self._btn_export_traces.clicked.connect(self._export_traces)
        diag_btn_row.addWidget(self._btn_export_traces)
        diag_container_layout.addLayout(diag_btn_row)
        self._tabs.addTab(diag_container, "MCMC Diagnostics")

        # --- Convergence summary ---
        self._summary = ConvergenceSummaryWidget()
        splitter.addWidget(self._summary)

        # Give most space to tabs
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)

        # Track the init-time figures so closeEvent can clean them up
        # even if _update_arviz_diagnostics replaces the canvas figures.
        self._original_figures = [self._pp_canvas.figure] + [
            c.figure for c in self._arviz_canvases.values()
        ]

        # Store axis labels (customisable per context)
        self._xlabel = "x"
        self._ylabel = "y"
        # Store latest results for export
        self._result: FitResult | None = None
        self._model_func: Any = None
        self._x_data: np.ndarray | None = None
        self._y_data: np.ndarray | None = None
        self._yerr: np.ndarray | None = None
        self._q_value: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_axis_labels(self, xlabel: str, ylabel: str) -> None:
        """Set axis labels for the posterior predictive plot."""
        self._xlabel = xlabel
        self._ylabel = ylabel

    def update_results(
        self,
        result: FitResult,
        model_func: Any,
        x_data: Any,
        y_data: Any,
        yerr: Any | None = None,
        q_value: float | None = None,
        title: str | None = None,
    ) -> None:
        """Refresh all panels with new fit results.

        Parameters
        ----------
        result : FitResult
            Bayesian fit result with posterior samples and diagnostics.
        model_func : callable
            Model function ``f(x, *params)`` for posterior predictive.
        x_data, y_data : array-like
            Original data used for fitting.
        yerr : array-like, optional
            Measurement uncertainties (shown as errorbars if provided).
        q_value : float, optional
            Q value for display in the title.
        title : str, optional
            Window title override.
        """
        if title is not None:
            self.setWindowTitle(title)

        self._result = result
        self._model_func = model_func
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._yerr = np.asarray(yerr) if yerr is not None else None
        self._q_value = q_value
        self._update_posterior_predictive(
            result, model_func, x_data, y_data, yerr, q_value
        )
        self._update_arviz_diagnostics(result)
        self._summary.update_diagnostics(result.diagnostics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_posterior_predictive(
        self,
        result: FitResult,
        model_func: Any,
        x_data: Any,
        y_data: Any,
        yerr: Any | None,
        q_value: float | None,
    ) -> None:
        """Redraw the posterior predictive plot (Tab 1)."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        fig = self._pp_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Plot data with errorbars if available
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        if yerr is not None:
            yerr = np.asarray(yerr)
            ax.errorbar(
                x_data,
                y_data,
                yerr=yerr,
                fmt="ko",
                ms=4,
                alpha=0.7,
                label="Data",
                zorder=3,
            )
        else:
            ax.scatter(x_data, y_data, c="k", s=20, alpha=0.7, label="Data", zorder=3)

        # Log-spaced prediction for smooth curve on log-x axis
        pos = x_data[x_data > 0]
        if len(pos) < 2:
            x_pred = np.linspace(x_data.min(), x_data.max(), 200)
        else:
            x_pred = np.geomspace(pos.min(), pos.max(), 200)

        plot_posterior_predictive(
            result,
            model_func,
            x_data,
            y_data,
            x_pred=x_pred,
            ax=ax,
        )

        # Remove the duplicate "Data" scatter that plot_posterior_predictive adds
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        unique_handles, unique_labels = [], []
        for h, lbl in zip(handles, labels, strict=False):
            if lbl not in seen:
                seen.add(lbl)
                unique_handles.append(h)
                unique_labels.append(lbl)
        ax.legend(unique_handles, unique_labels)

        ax.set_xscale("log")
        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)
        subtitle = f"Q = {q_value:.4g}" if q_value is not None else ""
        ax.set_title(f"Posterior Predictive {subtitle}".strip())

        fig.tight_layout()
        self._pp_canvas.draw()

    def _update_arviz_diagnostics(self, result: FitResult) -> None:
        """Redraw ArviZ diagnostic plots (Tab 2)."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        if result.arviz_data is None:
            logger.warning("No ArviZ data available — skipping diagnostic plots")
            return

        import matplotlib.pyplot as plt

        var_names = list(result.samples.keys())
        try:
            figures = generate_arviz_diagnostics(result.arviz_data, var_names=var_names)
        except Exception:
            logger.exception("Failed to generate ArviZ diagnostics")
            return

        for name, canvas in self._arviz_canvases.items():
            fig = figures.get(name)
            if fig is None:
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    f"{name}\n(not available)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                canvas.draw()
                continue

            # Close the old figure before replacing to prevent leak
            old_fig = canvas.figure
            canvas.figure = fig
            fig.set_canvas(canvas)
            # Reduce font sizes so labels fit within the canvas
            for ax in fig.get_axes():
                ax.tick_params(labelsize=8)
                ax.xaxis.label.set_size(9)
                ax.yaxis.label.set_size(9)
                if ax.get_title():
                    ax.title.set_size(9)
            fig.tight_layout(pad=1.2, h_pad=1.0, w_pad=1.0)
            canvas.draw()
            plt.close(old_fig)

    # ------------------------------------------------------------------
    # Export handlers
    # ------------------------------------------------------------------

    def _export_fit_plot(self) -> None:
        """Save the Fit + 95% CI plot to file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Plot",
            "bayesian_fit.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        fig = self._pp_canvas.figure
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Exported fit plot to %s", path)

    def _export_fit_data(self) -> None:
        """Save raw data, fitted curve, 95% CI, and residuals as CSV."""
        if (
            self._result is None
            or self._x_data is None
            or self._y_data is None
            or self._model_func is None
        ):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fit Data",
            "bayesian_fit.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        import csv
        from pathlib import Path

        result = self._result
        param_names = result.param_names or list(result.samples.keys())
        n_samples = len(result.samples[param_names[0]])
        x = self._x_data
        model = self._model_func

        # Compute posterior predictions at each data point
        predictions = np.empty((n_samples, len(x)))
        for i in range(n_samples):
            params = [result.samples[name][i] for name in param_names]
            predictions[i] = model(x, *params)

        y_fit = np.median(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        residual = self._y_data - y_fit

        with Path(path).open("w", newline="") as f:
            writer = csv.writer(f)
            # Header with metadata comment
            f.write(f"# Q = {self._q_value}\n" if self._q_value is not None else "")
            header = [
                self._xlabel,
                self._ylabel,
                "yerr",
                "fit_median",
                "ci_lower_2.5%",
                "ci_upper_97.5%",
                "residual",
            ]
            writer.writerow(header)
            for j in range(len(x)):
                row = [
                    f"{x[j]:.6g}",
                    f"{self._y_data[j]:.6g}",
                    f"{self._yerr[j]:.6g}" if self._yerr is not None else "",
                    f"{y_fit[j]:.6g}",
                    f"{ci_lower[j]:.6g}",
                    f"{ci_upper[j]:.6g}",
                    f"{residual[j]:.6g}",
                ]
                writer.writerow(row)

        logger.info("Exported fit data (%d points) to %s", len(x), path)

    def _export_diag_plots(self) -> None:
        """Save all 6 diagnostic plots as individual files."""
        from pathlib import Path

        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not dir_path:
            return
        out = Path(dir_path)
        for name, canvas in self._arviz_canvases.items():
            fig = canvas.figure
            filepath = out / f"mcmc_{name}.png"
            fig.savefig(str(filepath), dpi=300, bbox_inches="tight")
        logger.info(
            "Exported %d diagnostic plots to %s", len(self._arviz_canvases), out
        )

    def _export_traces(self) -> None:
        """Save ArviZ InferenceData as netCDF."""
        if self._result is None or self._result.arviz_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Traces (netCDF)",
            "mcmc_traces.nc",
            "netCDF (*.nc)",
        )
        if not path:
            return
        self._result.arviz_data.to_netcdf(path)
        logger.info("Exported ArviZ traces to %s", path)

    def closeEvent(self, event: Any) -> None:
        """Clean up matplotlib figures to avoid memory leaks."""
        import matplotlib.pyplot as plt

        # Close current canvas figures
        for canvas in self._arviz_canvases.values():
            plt.close(canvas.figure)
        plt.close(self._pp_canvas.figure)

        # Also close init-time figures that may have been swapped out
        for fig in self._original_figures:
            plt.close(fig)

        super().closeEvent(event)
