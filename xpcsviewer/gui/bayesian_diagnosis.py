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

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QGridLayout

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

# Ensure non-interactive backend for embedded canvases
matplotlib.use("QtAgg")


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
        self._tabs.addTab(self._pp_widget, "Fit + 95% CI")

        # Tab 2: ArviZ diagnostics (2x3 grid inside scroll area)
        self._diag_widget = QWidget()
        self._diag_layout = QGridLayout(self._diag_widget)
        self._arviz_canvases: dict[str, FigureCanvasQTAgg] = {}
        for idx, name in enumerate(self._ARVIZ_PLOT_NAMES):
            canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
            canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            row, col = divmod(idx, 3)
            self._diag_layout.addWidget(canvas, row, col)
            self._arviz_canvases[name] = canvas
        self._tabs.addTab(self._diag_widget, "MCMC Diagnostics")

        # --- Convergence summary ---
        self._summary = ConvergenceSummaryWidget()
        splitter.addWidget(self._summary)

        # Give most space to tabs
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)

        # Store axis labels (customisable per context)
        self._xlabel = "x"
        self._ylabel = "y"

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

        # Use the fitting visualization (it will plot on our existing axes)
        # We pass ax so it draws on top of our data
        x_pred = np.linspace(x_data.min(), x_data.max(), 200)
        plot_posterior_predictive(
            result,
            model_func,
            x_data,
            y_data,
            x_pred=x_pred,
            ax=ax,
        )

        # Remove the duplicate "Data" scatter that plot_posterior_predictive adds
        # by keeping only the first Data legend entry
        handles, labels = ax.get_legend_handles_labels()
        seen: set[str] = set()
        unique_handles, unique_labels = [], []
        for h, lbl in zip(handles, labels, strict=False):
            if lbl not in seen:
                seen.add(lbl)
                unique_handles.append(h)
                unique_labels.append(lbl)
        ax.legend(unique_handles, unique_labels)

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

            # Replace the canvas figure with the ArviZ-generated one
            canvas.figure = fig
            fig.set_canvas(canvas)
            fig.tight_layout()
            canvas.draw()

    def closeEvent(self, event: Any) -> None:
        """Clean up matplotlib figures to avoid memory leaks."""
        import matplotlib.pyplot as plt

        # Close all figures owned by our canvases
        for canvas in self._arviz_canvases.values():
            plt.close(canvas.figure)
        plt.close(self._pp_canvas.figure)
        super().closeEvent(event)
