"""Regression tests for tauq.plot rendering.

Tests cover:
    BUG-F: Fit line must render when tauq_success=True and tauq_fit_line is a dict
    BUG-G: Error bars must use tauq_tau_err (not fit_val[:, 1, 1]) when available
    Fallback: Graceful handling when tauq keys are absent (early-return path)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from xpcsviewer.module.tauq import plot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xf(
    q: np.ndarray,
    tau: np.ndarray,
    tau_err: np.ndarray,
    *,
    tauq_success: bool = False,
    fit_line_x: np.ndarray | None = None,
    fit_line_y: np.ndarray | None = None,
    include_tauq_keys: bool = True,
    label: str = "test_file",
) -> SimpleNamespace:
    """Build a minimal xf-like object with fit_summary.

    Parameters
    ----------
    q, tau, tau_err : ndarray
        Per-q-point data (already filtered if ``include_tauq_keys``).
    tauq_success : bool
        Whether the power-law fit succeeded.
    fit_line_x, fit_line_y : ndarray, optional
        Fit curve x/y arrays (stored as dict in ``tauq_fit_line``).
    include_tauq_keys : bool
        If True, populate ``tauq_q``/``tauq_tau``/``tauq_tau_err`` keys.
        If False, only populate ``fit_val`` (simulates early-return path).
    label : str
        File label for legend.
    """
    n_q = len(q)

    # Build the full G2-style fit_val array (n_q, 2, 4)
    fit_val = np.zeros((n_q, 2, 4))
    fit_val[:, 0, 1] = tau  # tau values
    fit_val[:, 1, 1] = tau_err  # tau errors

    fit_summary: dict = {
        "q_val": q,
        "fit_val": fit_val,
        "tauq_success": tauq_success,
    }

    if include_tauq_keys:
        fit_summary["tauq_q"] = q
        fit_summary["tauq_tau"] = tau
        fit_summary["tauq_tau_err"] = tau_err

    if tauq_success and fit_line_x is not None and fit_line_y is not None:
        fit_summary["tauq_fit_line"] = {"fit_x": fit_line_x, "fit_y": fit_line_y}
    elif tauq_success:
        fit_summary["tauq_fit_line"] = None

    return SimpleNamespace(fit_summary=fit_summary, label=label)


def _make_hdl() -> MagicMock:
    """Build a mock matplotlib canvas handle."""
    hdl = MagicMock()
    ax = MagicMock()
    hdl.subplots.return_value = ax
    return hdl


# ---------------------------------------------------------------------------
# BUG-F: Fit line rendering
# ---------------------------------------------------------------------------


class TestBugF_FitLineRendering:
    """Verify power-law fit line is drawn when tauq_success=True."""

    def test_fit_line_drawn_when_success(self):
        """ax.plot() must be called with fit_x/fit_y when tauq_success."""
        q = np.array([0.01, 0.02, 0.03])
        tau = np.array([1.0, 0.5, 0.25])
        tau_err = np.array([0.1, 0.05, 0.02])
        fit_x = np.logspace(-2.1, -1.4, 50)
        fit_y = 100 * fit_x ** (-2.0)

        xf = _make_xf(
            q,
            tau,
            tau_err,
            tauq_success=True,
            fit_line_x=fit_x,
            fit_line_y=fit_y,
        )
        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf], hdl, q_range=(0.01, 0.03), offset=0, plot_type=3)

        # ax.plot must have been called (for the fit line)
        ax.plot.assert_called_once()
        call_args = ax.plot.call_args
        np.testing.assert_array_equal(call_args[0][0], fit_x)
        np.testing.assert_array_equal(call_args[0][1], fit_y)

    def test_no_fit_line_when_tauq_success_false(self):
        """ax.plot() must NOT be called when tauq_success is False."""
        q = np.array([0.01, 0.02, 0.03])
        tau = np.array([1.0, 0.5, 0.25])
        tau_err = np.array([0.1, 0.05, 0.02])

        xf = _make_xf(q, tau, tau_err, tauq_success=False)
        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf], hdl, q_range=(0.01, 0.03), offset=0, plot_type=3)

        ax.plot.assert_not_called()

    def test_no_fit_line_when_tauq_fit_line_none(self):
        """ax.plot() must NOT be called when tauq_fit_line is None."""
        q = np.array([0.01, 0.02])
        tau = np.array([1.0, 0.5])
        tau_err = np.array([0.1, 0.05])

        xf = _make_xf(
            q,
            tau,
            tau_err,
            tauq_success=True,
            fit_line_x=None,
            fit_line_y=None,
        )
        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf], hdl, q_range=(0.01, 0.03), offset=0, plot_type=3)

        ax.plot.assert_not_called()


# ---------------------------------------------------------------------------
# BUG-G: Error bar data source
# ---------------------------------------------------------------------------


class TestBugG_ErrorBarSource:
    """Verify error bars use tauq_tau_err when available."""

    def test_errorbar_uses_tauq_tau_err(self):
        """yerr in ax.errorbar() must match tauq_tau_err, not fit_val."""
        q = np.array([0.01, 0.02, 0.03])
        tauq_tau = np.array([1.0, 0.5, 0.25])
        tauq_tau_err = np.array([0.1, 0.05, 0.02])

        # Put DIFFERENT errors in fit_val to detect if the wrong source is used
        xf = _make_xf(q, tauq_tau, tauq_tau_err, include_tauq_keys=True)
        different_err = np.array([0.9, 0.8, 0.7])
        xf.fit_summary["fit_val"][:, 1, 1] = different_err

        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf], hdl, q_range=(0.01, 0.03), offset=0, plot_type=3)

        # Check yerr passed to errorbar is tauq_tau_err, not different_err
        ax.errorbar.assert_called_once()
        call_kwargs = ax.errorbar.call_args
        np.testing.assert_array_equal(call_kwargs[1]["yerr"], tauq_tau_err)

    def test_fallback_to_fit_val_when_tauq_keys_absent(self):
        """When tauq keys are missing, fall back to fit_val[:, :, 1]."""
        q = np.array([0.01, 0.02, 0.03])
        tau = np.array([1.0, 0.5, 0.25])
        tau_err = np.array([0.1, 0.05, 0.02])

        xf = _make_xf(q, tau, tau_err, include_tauq_keys=False)
        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf], hdl, q_range=(0.01, 0.03), offset=0, plot_type=3)

        # Should still plot using fit_val data
        ax.errorbar.assert_called_once()
        call_kwargs = ax.errorbar.call_args
        np.testing.assert_array_equal(call_kwargs[1]["yerr"], tau_err)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestTauqPlotEdgeCases:
    """Edge cases for robustness."""

    def test_empty_xf_list(self):
        """No crash with empty file list."""
        hdl = _make_hdl()
        plot([], hdl, q_range=(0.01, 0.1), offset=0, plot_type=3)
        hdl.draw.assert_called_once()

    def test_all_zero_sigma_skipped(self):
        """Files with all-zero tau_err are skipped gracefully."""
        q = np.array([0.01, 0.02])
        tau = np.array([1.0, 0.5])
        tau_err = np.array([0.0, 0.0])  # all zero

        xf = _make_xf(q, tau, tau_err, include_tauq_keys=True)
        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf], hdl, q_range=(0.01, 0.03), offset=0, plot_type=3)

        ax.errorbar.assert_not_called()

    def test_offset_scales_data(self):
        """Decade offset divides y and yerr values correctly."""
        q = np.array([0.01, 0.02])
        tau = np.array([10.0, 5.0])
        tau_err = np.array([1.0, 0.5])

        xf0 = _make_xf(q, tau, tau_err, label="file0")
        xf1 = _make_xf(q, tau, tau_err, label="file1")
        hdl = _make_hdl()
        ax = hdl.subplots.return_value

        plot([xf0, xf1], hdl, q_range=(0.01, 0.03), offset=1, plot_type=3)

        assert ax.errorbar.call_count == 2

        # First file: scale = 10^(1*0) = 1
        first_call = ax.errorbar.call_args_list[0]
        np.testing.assert_array_almost_equal(first_call[0][1], tau)

        # Second file: scale = 10^(1*1) = 10
        second_call = ax.errorbar.call_args_list[1]
        np.testing.assert_array_almost_equal(second_call[0][1], tau / 10)
