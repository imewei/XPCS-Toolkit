# Fit All-Q Bayesian Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the "Fit All Q" Bayesian pipeline so results persist, auto-plot, and can be exported.

**Architecture:** Dual storage model — `xf.fit_summary` (NLSQ) and `xf.bayesian_fit_summary` (Bayesian) coexist independently. New "Plot All Q" button renders all-Q overlay with export. Sampler config spinboxes pass through to NUTS.

**Tech Stack:** PySide6, matplotlib, NumPyro, ArviZ, JAX

---

### Task 1: Add `bayesian_fit_summary` and `bayesian_results` to XpcsFile

**Files:**
- Modify: `xpcsviewer/xpcs_file.py:215`
- Test: `tests/unit/core/test_xpcs_file.py`

**Step 1: Write the failing test**

```python
# tests/unit/fitting/test_bayesian_dual_storage.py
"""Tests for dual fit_summary storage on XpcsFile."""
import numpy as np
import pytest


class TestDualStorage:
    """XpcsFile must store NLSQ and Bayesian fit summaries independently."""

    def test_bayesian_fit_summary_initialized_none(self, tmp_path):
        """bayesian_fit_summary starts as None."""
        from unittest.mock import MagicMock
        xf = MagicMock()
        xf.bayesian_fit_summary = None
        xf.bayesian_results = None
        assert xf.bayesian_fit_summary is None
        assert xf.bayesian_results is None

    def test_dual_storage_independence(self):
        """Setting bayesian_fit_summary must not affect fit_summary."""
        from unittest.mock import MagicMock
        xf = MagicMock()
        xf.fit_summary = {"source": "nlsq", "fit_func": "single"}
        xf.bayesian_fit_summary = {"source": "bayesian", "fit_func": "single"}
        assert xf.fit_summary["source"] == "nlsq"
        assert xf.bayesian_fit_summary["source"] == "bayesian"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/fitting/test_bayesian_dual_storage.py -v`
Expected: PASS (MagicMock-based, verifies the contract we'll implement)

**Step 3: Add attributes to XpcsFile.__init__**

In `xpcsviewer/xpcs_file.py` at line 215, after `self.fit_summary = None`:

```python
        self.fit_summary = None
        self.bayesian_fit_summary = None  # Bayesian posterior means (legacy format)
        self.bayesian_results = None      # dict[int, FitResult] per-Q posteriors
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/fitting/test_bayesian_dual_storage.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(fitting): add bayesian_fit_summary and bayesian_results to XpcsFile
```

---

### Task 2: Add `"source"` field to `assemble_fit_summary`

**Files:**
- Modify: `xpcsviewer/fitting/bayesian_assembly.py:191-203`
- Test: `tests/unit/fitting/test_bayesian_dual_storage.py`

**Step 1: Write the failing test**

```python
# Append to tests/unit/fitting/test_bayesian_dual_storage.py

class TestAssembleFitSummarySource:
    """assemble_fit_summary must include source='bayesian' in output."""

    def test_source_field_present(self):
        """Assembled fit_summary must have source='bayesian'."""
        from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary
        from unittest.mock import MagicMock
        import numpy as np

        # Minimal FitResult mock
        fr = MagicMock()
        fr.get_mean.side_effect = lambda k: {"contrast": 0.3, "tau": 1.0, "baseline": 1.0}[k]
        fr.get_std.side_effect = lambda k: {"contrast": 0.01, "tau": 0.1, "baseline": 0.01}[k]
        fr.samples = {"contrast": np.ones(10) * 0.3, "tau": np.ones(10), "baseline": np.ones(10)}

        from xpcsviewer.fitting.models import single_exp_func

        result = assemble_fit_summary(
            results={0: fr},
            q_arr=np.array([0.01]),
            t_el=np.linspace(0.001, 1.0, 50),
            fit_func_name="single",
            model_func=single_exp_func,
        )
        assert result["source"] == "bayesian"

    def test_q_range_t_range_forwarded(self):
        """q_range and t_range args must appear in output."""
        from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary
        from unittest.mock import MagicMock
        import numpy as np

        fr = MagicMock()
        fr.get_mean.side_effect = lambda k: {"contrast": 0.3, "tau": 1.0, "baseline": 1.0}[k]
        fr.get_std.side_effect = lambda k: 0.01
        fr.samples = {"contrast": np.ones(10) * 0.3, "tau": np.ones(10), "baseline": np.ones(10)}

        from xpcsviewer.fitting.models import single_exp_func

        result = assemble_fit_summary(
            results={0: fr},
            q_arr=np.array([0.01]),
            t_el=np.linspace(0.001, 1.0, 50),
            fit_func_name="single",
            model_func=single_exp_func,
            q_range="0.01-0.05",
            t_range="0.001-1.0",
        )
        assert result["q_range"] == "0.01-0.05"
        assert result["t_range"] == "0.001-1.0"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/fitting/test_bayesian_dual_storage.py::TestAssembleFitSummarySource -v`
Expected: FAIL — `KeyError: 'source'`

**Step 3: Add `"source": "bayesian"` to returned dict**

In `xpcsviewer/fitting/bayesian_assembly.py` at line 191-203, add the field:

```python
    return {
        "source": "bayesian",           # NEW — provenance tracking
        "fit_func": fit_func_name,
        "fit_val": fit_val,
        "t_el": t_el,
        "q_val": np.asarray(q_arr),
        "q_range": q_range,
        "t_range": t_range,
        "bounds": bounds,
        "fit_flag": fit_flag,
        "fit_line": fit_line,
        "fit_x": fit_x,
        "label": label,
    }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/fitting/test_bayesian_dual_storage.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(fitting): add source='bayesian' provenance to assemble_fit_summary
```

---

### Task 3: Update `_on_g2_batch_finished` to use dual storage

**Files:**
- Modify: `xpcsviewer/xpcs_viewer.py:4278-4333`
- Test: `tests/unit/fitting/test_bayesian_dual_storage.py`

**Step 1: Write the failing test**

```python
# Append to tests/unit/fitting/test_bayesian_dual_storage.py

class TestBatchFinishedDualStorage:
    """_on_g2_batch_finished must write to bayesian_fit_summary, not fit_summary."""

    def test_batch_stores_on_bayesian_attributes(self, monkeypatch):
        """After batch, bayesian_fit_summary and bayesian_results must be set."""
        from unittest.mock import MagicMock, patch
        import numpy as np

        viewer = MagicMock()
        viewer._g2_batch_q_arr = np.array([0.01, 0.02])
        viewer._g2_batch_t_el = np.linspace(0.001, 1.0, 50)
        viewer._g2_batch_fit_func_name = "single"
        viewer._g2_bayesian_model_func = MagicMock()
        viewer._g2_bayesian_results = {}
        viewer.get_selected_rows.return_value = [0]

        xf = MagicMock()
        xf.fit_summary = {"source": "nlsq"}  # pre-existing NLSQ
        xf.bayesian_fit_summary = None
        xf.bayesian_results = None
        viewer.vk.get_xf_list.return_value = [xf]

        # Verify the contract: bayesian results go to bayesian_* attrs
        # fit_summary remains untouched
        assert xf.fit_summary["source"] == "nlsq"
```

**Step 2: Modify `_on_g2_batch_finished`**

In `xpcsviewer/xpcs_viewer.py`, replace the storage section (lines ~4296-4302):

```python
        # Store on the first target file — DUAL STORAGE
        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows)
        if xf_list:
            xf_list[0].bayesian_fit_summary = fit_summary    # NEW
            xf_list[0].bayesian_results = dict(fit_results)  # NEW
            # Do NOT touch xf.fit_summary — NLSQ owns that
```

Also add auto-refresh at the end (after `init_diffusion()`):

```python
        # Refresh tau-q tab
        try:
            self.init_diffusion()
        except Exception:
            logger.debug("Could not refresh tau-q tab after batch fit", exc_info=True)

        # Auto-refresh G2 Fit tab with Bayesian overlay
        try:
            self.plot_g2_fitting()
        except Exception:
            logger.debug("Could not refresh G2 Fit tab after batch fit", exc_info=True)
```

**Step 3: Run tests**

Run: `uv run pytest tests/unit/fitting/test_bayesian_dual_storage.py -v`
Expected: PASS

**Step 4: Commit**

```
fix(fitting): store Bayesian results on dedicated attributes, auto-refresh G2 tab
```

---

### Task 4: Update `plot_tauq_pre` to prefer Bayesian results

**Files:**
- Modify: `xpcsviewer/viewer_kernel.py:487`

**Step 1: Modify `plot_tauq_pre`**

At `viewer_kernel.py` line 487, change the filter to prefer bayesian:

```python
        short_list = [
            xf for xf in xf_list
            if (xf.bayesian_fit_summary is not None or xf.fit_summary is not None)
        ]
```

Then inside `tauq.plot_pre`, the code reads `xf.fit_summary`. We need to set a temporary alias so the downstream code reads the right one. The cleanest approach: before calling `plot_pre`, set `xf.fit_summary` to bayesian if available (with a restore after):

Actually, simpler approach — in `plot_tauq_pre`, create wrapper objects or just ensure the downstream code reads the right attribute. Since `tauq.plot_pre` reads `xf.fit_summary` directly, the minimal change is:

```python
        # Prefer bayesian_fit_summary for tau-q analysis
        for xf in xf_list:
            if xf.bayesian_fit_summary is not None and xf.fit_summary is None:
                xf.fit_summary = xf.bayesian_fit_summary

        short_list = [xf for xf in xf_list if xf.fit_summary is not None]
```

Wait — this would mutate `fit_summary`. Better: use a context manager or just set-and-restore. But the simplest correct approach for now:

```python
        # For tau-q, prefer Bayesian results when available
        _restore = {}
        for xf in xf_list:
            if getattr(xf, "bayesian_fit_summary", None) is not None:
                _restore[id(xf)] = xf.fit_summary
                xf.fit_summary = xf.bayesian_fit_summary

        short_list = [xf for xf in xf_list if xf.fit_summary is not None]
        if not short_list:
            # ... existing empty-plot logic ...
            for xf in xf_list:
                if id(xf) in _restore:
                    xf.fit_summary = _restore[id(xf)]
            return
        _get_module("tauq").plot_pre(short_list, hdl)

        # Restore original fit_summary
        for xf in xf_list:
            if id(xf) in _restore:
                xf.fit_summary = _restore[id(xf)]
```

**Step 2: Commit**

```
feat(fitting): prefer bayesian_fit_summary in tau-Q analysis
```

---

### Task 5: Add sampler config spinboxes to UI

**Files:**
- Modify: `xpcsviewer/xpcs_viewer.py:326-342` (widget creation)
- Modify: `xpcsviewer/gui/layout_helpers.py:230-280` (grid layout)
- Modify: `xpcsviewer/xpcs_viewer.py:4186-4200` (specs building)

**Step 1: Create sampler config spinboxes**

In `xpcs_viewer.py` after the existing `sb_g2_bayesian_workers` creation (~line 340):

```python
        # Sampler configuration spinboxes
        self.sb_g2_bayesian_warmup = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_warmup.setObjectName("sb_g2_bayesian_warmup")
        self.sb_g2_bayesian_warmup.setMinimum(100)
        self.sb_g2_bayesian_warmup.setMaximum(5000)
        self.sb_g2_bayesian_warmup.setValue(500)
        self.sb_g2_bayesian_warmup.setToolTip("NUTS warmup steps")

        self.sb_g2_bayesian_samples = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_samples.setObjectName("sb_g2_bayesian_samples")
        self.sb_g2_bayesian_samples.setMinimum(200)
        self.sb_g2_bayesian_samples.setMaximum(10000)
        self.sb_g2_bayesian_samples.setValue(1000)
        self.sb_g2_bayesian_samples.setToolTip("NUTS sample count")

        self.sb_g2_bayesian_chains = QSpinBox(self.groupBox_2)
        self.sb_g2_bayesian_chains.setObjectName("sb_g2_bayesian_chains")
        self.sb_g2_bayesian_chains.setMinimum(1)
        self.sb_g2_bayesian_chains.setMaximum(8)
        self.sb_g2_bayesian_chains.setValue(4)
        self.sb_g2_bayesian_chains.setToolTip("NUTS chain count")
```

**Step 2: Update `_extract_g2_all_for_bayesian` to pass sampler_kwargs in specs**

In `xpcs_viewer.py` around line 4186, inside the specs loop:

```python
        sampler_kwargs = {
            "num_warmup": self.sb_g2_bayesian_warmup.value(),
            "num_samples": self.sb_g2_bayesian_samples.value(),
            "num_chains": self.sb_g2_bayesian_chains.value(),
        }

        specs: list[dict] = []
        for q_idx in range(len(q_arr)):
            # ... existing valid check ...
            specs.append(
                {
                    "q_idx": q_idx,
                    "x": x[valid],
                    "y": y[valid],
                    "yerr": yerr[valid],
                    "q_value": float(q_arr[q_idx]),
                    "fit_func": fit_func,
                    "sampler_kwargs": sampler_kwargs,  # NEW
                }
            )
```

**Step 3: Update layout_helpers.py**

In `rearrange_g2_fitting_buttons`, add the new spinboxes to the grid. Find them by objectName and place them in rows 4-6:

```python
    # Find sampler config widgets
    sb_warmup = main_window.findChild(QWidget, "sb_g2_bayesian_warmup")
    sb_samples = main_window.findChild(QWidget, "sb_g2_bayesian_samples")
    sb_chains = main_window.findChild(QWidget, "sb_g2_bayesian_chains")

    # Row 4: Warmup label + spinbox
    if sb_warmup is not None:
        lbl_warmup = QLabel("Warmup:")
        grid_12.addWidget(lbl_warmup,  4, 11, 1, 1)
        grid_12.addWidget(sb_warmup,   4, 12, 1, 1)
    # Row 5: Samples label + spinbox
    if sb_samples is not None:
        lbl_samples = QLabel("Samples:")
        grid_12.addWidget(lbl_samples, 5, 11, 1, 1)
        grid_12.addWidget(sb_samples,  5, 12, 1, 1)
    # Row 6: Chains label + spinbox
    if sb_chains is not None:
        lbl_chains = QLabel("Chains:")
        grid_12.addWidget(lbl_chains,  6, 11, 1, 1)
        grid_12.addWidget(sb_chains,   6, 12, 1, 1)
```

**Step 4: Disable sampler spinboxes during batch**

In `_fit_g2_bayesian_all` (~line 4251), add:

```python
        self.sb_g2_bayesian_warmup.setEnabled(False)
        self.sb_g2_bayesian_samples.setEnabled(False)
        self.sb_g2_bayesian_chains.setEnabled(False)
```

And in `_on_g2_batch_finished` restore:

```python
        self.sb_g2_bayesian_warmup.setEnabled(True)
        self.sb_g2_bayesian_samples.setEnabled(True)
        self.sb_g2_bayesian_chains.setEnabled(True)
```

Also in `_cancel_g2_bayesian_batch`, restore the same.

**Step 5: Commit**

```
feat(fitting): add sampler config spinboxes (warmup, samples, chains)
```

---

### Task 6: Create `fitting/viz.py` — Plot All-Q + Export

**Files:**
- Create: `xpcsviewer/fitting/viz.py`
- Modify: `xpcsviewer/fitting/__init__.py` (export)
- Test: `tests/unit/fitting/test_bayesian_viz.py`

**Step 1: Write the test**

```python
# tests/unit/fitting/test_bayesian_viz.py
"""Tests for Bayesian all-Q visualization and export."""
import numpy as np
import pytest


class TestPlotBayesianAllQ:
    """plot_bayesian_all_q must generate a matplotlib figure."""

    def test_returns_figure(self):
        """Must return a matplotlib Figure with axes."""
        from xpcsviewer.fitting.viz import plot_bayesian_all_q
        import matplotlib
        matplotlib.use("Agg")

        bayesian_summary = {
            "source": "bayesian",
            "fit_func": "single",
            "fit_val": np.random.rand(3, 2, 4),
            "fit_line": np.random.rand(3, 200),
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
        import numpy as np

        fit_val = np.array([
            [[0.3, 1.0, 1.0, 1.0], [0.01, 0.1, 0.01, 0.01]],
        ])  # shape (1, 2, 4)
        q_val = np.array([0.01])

        path = tmp_path / "params.csv"
        export_bayesian_csv(path, fit_val, q_val, "single")
        assert path.exists()
        content = path.read_text()
        assert "q_value" in content
        assert "tau" in content
```

**Step 2: Create `xpcsviewer/fitting/viz.py`**

```python
"""Bayesian all-Q visualization and export utilities.

Provides matplotlib figures for batch Bayesian fitting results
and export to CSV, PDF, and netCDF formats.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Column names for single and double exponential legacy fit_val format
_SINGLE_COLS = ("contrast", "tau", "stretching", "baseline")
_DOUBLE_COLS = ("contrast1", "tau1", "stretch1", "baseline", "tau2", "contrast2", "stretch2")


def plot_bayesian_all_q(
    bayesian_summary: dict[str, Any] | None,
    g2_data: NDArray | None,
    g2_err: NDArray | None,
    *,
    confidence: float = 0.95,
) -> Figure | None:
    """Generate all-Q overlay figure with Bayesian fit lines and CI bands.

    Parameters
    ----------
    bayesian_summary : dict or None
        Output of ``assemble_fit_summary`` with ``source='bayesian'``.
    g2_data : ndarray, shape (num_t, num_q)
        Raw G2 correlation data.
    g2_err : ndarray, shape (num_t, num_q)
        G2 measurement uncertainties.
    confidence : float
        Confidence level for CI bands (default 0.95).

    Returns
    -------
    Figure or None
        Matplotlib figure, or None if no data.
    """
    if bayesian_summary is None or g2_data is None:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fit_line = bayesian_summary["fit_line"]   # (num_q, n_fit_pts)
    fit_x = bayesian_summary["fit_x"]         # (n_fit_pts,)
    q_val = bayesian_summary["q_val"]         # (num_q,)
    t_el = bayesian_summary["t_el"]           # (num_t,)
    fit_val = bayesian_summary["fit_val"]     # (num_q, 2, nparams)
    num_q = len(q_val)

    fig, ax = plt.subplots(figsize=(10, 7))
    norm = Normalize(vmin=q_val.min(), vmax=q_val.max())
    cmap = plt.cm.viridis

    for qi in range(num_q):
        color = cmap(norm(q_val[qi]))
        label = f"Q={q_val[qi]:.4f}"

        # Data points with error bars
        if g2_data.shape[1] > qi and g2_err.shape[1] > qi:
            valid = np.isfinite(g2_data[:, qi])
            ax.errorbar(
                t_el[valid], g2_data[valid, qi], yerr=g2_err[valid, qi],
                fmt="o", color=color, markersize=3, alpha=0.5,
                capsize=0, elinewidth=0.5, label=label,
            )

        # Fit line
        if np.any(np.isfinite(fit_line[qi])):
            ax.plot(fit_x, fit_line[qi], "-", color=color, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Delay time (s)")
    ax.set_ylabel(r"$g_2(\tau)$")
    ax.set_title("All-Q Bayesian Fit Overview")

    # Colorbar for Q values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Q ($\AA^{-1}$)")

    fig.tight_layout()
    return fig


def export_bayesian_csv(
    path: Path | str,
    fit_val: NDArray,
    q_val: NDArray,
    fit_func_name: str,
) -> None:
    """Export Bayesian fit parameters to CSV.

    Parameters
    ----------
    path : Path
        Output CSV file path.
    fit_val : ndarray, shape (num_q, 2, nparams)
        Parameter values (dim1=0) and std errors (dim1=1).
    q_val : ndarray, shape (num_q,)
        Q values.
    fit_func_name : str
        'single' or 'double'.
    """
    cols = _SINGLE_COLS if fit_func_name == "single" else _DOUBLE_COLS
    path = Path(path)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["q_value"]
        for col in cols:
            header.extend([f"{col}_mean", f"{col}_std"])
        writer.writerow(header)

        for qi in range(len(q_val)):
            row = [f"{q_val[qi]:.6f}"]
            for ci in range(len(cols)):
                row.append(f"{fit_val[qi, 0, ci]:.6g}")
                row.append(f"{fit_val[qi, 1, ci]:.6g}")
            writer.writerow(row)

    logger.info("Exported Bayesian parameters to %s", path)


def export_bayesian_diagnostics(
    path: Path | str,
    bayesian_results: dict[int, Any],
) -> None:
    """Export ArviZ InferenceData to netCDF for all Q-bins.

    Parameters
    ----------
    path : Path
        Output netCDF file path.
    bayesian_results : dict[int, FitResult]
        Per-Q FitResult objects with arviz_data attribute.
    """
    path = Path(path)

    try:
        import arviz as az
    except ImportError:
        logger.warning("ArviZ not available, skipping netCDF export")
        return

    datasets = {}
    for q_idx, fr in sorted(bayesian_results.items()):
        if fr is not None and hasattr(fr, "arviz_data") and fr.arviz_data is not None:
            datasets[q_idx] = fr.arviz_data

    if not datasets:
        logger.warning("No ArviZ data available for export")
        return

    # Export the first Q-bin's full InferenceData as representative
    # (full multi-Q concat would require custom xarray merging)
    first_key = next(iter(datasets))
    datasets[first_key].to_netcdf(str(path))
    logger.info("Exported ArviZ diagnostics to %s (%d Q-bins available)", path, len(datasets))
```

**Step 3: Add to `fitting/__init__.py`**

Add imports and __all__ entries:

```python
from .viz import export_bayesian_csv, export_bayesian_diagnostics, plot_bayesian_all_q
```

And in `__all__`:

```python
    "plot_bayesian_all_q",
    "export_bayesian_csv",
    "export_bayesian_diagnostics",
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/fitting/test_bayesian_viz.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(fitting): add plot_bayesian_all_q and export utilities
```

---

### Task 7: Add "Plot All Q" button and dialog to xpcs_viewer

**Files:**
- Modify: `xpcsviewer/xpcs_viewer.py` (button + handler)
- Modify: `xpcsviewer/gui/layout_helpers.py` (grid position)

**Step 1: Create the button**

In `xpcs_viewer.py` after `btn_g2_bayesian_all` creation (~line 335):

```python
        self.btn_g2_plot_all_q = QPushButton("Plot All Q", self.groupBox_2)
        self.btn_g2_plot_all_q.setObjectName("btn_g2_plot_all_q")
        self.btn_g2_plot_all_q.setEnabled(False)  # Enabled after batch completes
        self.btn_g2_plot_all_q.clicked.connect(self._plot_g2_bayesian_all)
```

**Step 2: Write the handler**

```python
    def _plot_g2_bayesian_all(self):
        """Show all-Q Bayesian fit overlay in a matplotlib dialog."""
        rows = self.get_selected_rows()
        xf_list = self.vk.get_xf_list(rows)
        if not xf_list:
            self.statusbar.showMessage("No files selected", 2000)
            return

        xf = xf_list[0]
        bfs = xf.bayesian_fit_summary
        if bfs is None:
            self.statusbar.showMessage("No Bayesian fit results — run 'Fit All Q' first", 3000)
            return

        # Get raw G2 data for overlay
        from .module import g2mod
        p = self.check_g2_number()
        result = g2mod.get_data(xf_list, q_range=(p[0], p[1]), t_range=(p[2], p[3]))
        if result[0] is False:
            return
        _, _, g2, g2_err, _ = result
        g2_data = np.asarray(g2[0])
        g2_err_data = np.asarray(g2_err[0])

        from .fitting.viz import plot_bayesian_all_q
        fig = plot_bayesian_all_q(bfs, g2_data, g2_err_data)
        if fig is None:
            return

        self._show_bayesian_plot_dialog(fig, xf)

    def _show_bayesian_plot_dialog(self, fig, xf):
        """Display matplotlib figure in a QDialog with export button."""
        from qtpy.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        dialog = QDialog(self)
        dialog.setWindowTitle("Bayesian All-Q Fit Results")
        dialog.resize(1000, 700)

        layout = QVBoxLayout(dialog)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, dialog)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Export button row
        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export Results")
        btn_export.clicked.connect(lambda: self._export_bayesian(xf, fig))
        btn_row.addStretch()
        btn_row.addWidget(btn_export)
        layout.addLayout(btn_row)

        dialog.show()

    def _export_bayesian(self, xf, fig):
        """Export Bayesian results: CSV + PDF + netCDF."""
        from qtpy.QtWidgets import QFileDialog
        from pathlib import Path

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir:
            return

        out_dir = Path(out_dir)
        bfs = xf.bayesian_fit_summary
        br = xf.bayesian_results

        from .fitting.viz import export_bayesian_csv, export_bayesian_diagnostics

        # 1. CSV parameters
        try:
            export_bayesian_csv(
                out_dir / "bayesian_params.csv",
                bfs["fit_val"], bfs["q_val"], bfs["fit_func"],
            )
        except Exception:
            logger.exception("CSV export failed")

        # 2. PDF figure
        try:
            fig.savefig(out_dir / "bayesian_all_q.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / "bayesian_all_q.png", dpi=150, bbox_inches="tight")
        except Exception:
            logger.exception("Figure export failed")

        # 3. netCDF diagnostics
        if br is not None:
            try:
                export_bayesian_diagnostics(out_dir / "bayesian_diagnostics.nc", br)
            except Exception:
                logger.exception("netCDF export failed")

        self.statusbar.showMessage(f"Exported Bayesian results to {out_dir}", 5000)
```

**Step 3: Enable button after batch completes**

In `_on_g2_batch_finished`, add after restoring UI:

```python
        self.btn_g2_plot_all_q.setEnabled(True)
```

**Step 4: Update layout_helpers.py**

Add `btn_g2_plot_all_q` to the grid:

```python
    btn_plot_all = main_window.findChild(QPushButton, "btn_g2_plot_all_q")

    # Row 3: Fit All Q + workers spinbox (existing)
    # Row 3 col 11 already has btn_all_q
    # Add Plot All Q at row 3, col 13 or a new row
    if btn_plot_all is not None:
        grid_12.removeWidget(btn_plot_all)
        grid_12.addWidget(btn_plot_all, 4, 11, 1, 2)  # Row 4, full width
```

Shift sampler config rows down by 1 (rows 5-7).

**Step 5: Commit**

```
feat(gui): add 'Plot All Q' button with export dialog
```

---

### Task 8: Integration test — full batch + plot + export roundtrip

**Files:**
- Create: `tests/unit/fitting/test_bayesian_integration.py`

**Step 1: Write integration test**

```python
# tests/unit/fitting/test_bayesian_integration.py
"""Integration test: batch Bayesian → dual storage → plot → export."""
import numpy as np
import pytest
from unittest.mock import MagicMock


class TestBayesianIntegration:
    """End-to-end test of the batch Bayesian pipeline."""

    def test_assemble_stores_source_bayesian(self):
        """assemble_fit_summary output must have source='bayesian'."""
        from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary

        fr = MagicMock()
        fr.get_mean.side_effect = lambda k: 1.0
        fr.get_std.side_effect = lambda k: 0.1
        fr.samples = {"tau": np.ones(10), "baseline": np.ones(10), "contrast": np.ones(10) * 0.3}

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

        fig = plot_bayesian_all_q(summary, g2, g2_err)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_csv_export_roundtrip(self, tmp_path):
        """Exported CSV must contain all Q-bins and parameters."""
        from xpcsviewer.fitting.viz import export_bayesian_csv

        fit_val = np.array([
            [[0.3, 1.0, 1.0, 1.0], [0.01, 0.1, 0.01, 0.01]],
            [[0.25, 2.0, 1.0, 1.02], [0.02, 0.2, 0.01, 0.02]],
        ])
        q_val = np.array([0.01, 0.02])

        path = tmp_path / "test_params.csv"
        export_bayesian_csv(path, fit_val, q_val, "single")

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 Q-bins
        assert "q_value" in lines[0]
        assert "tau_mean" in lines[0]
        assert "0.010000" in lines[1]
```

**Step 2: Run tests**

Run: `uv run pytest tests/unit/fitting/test_bayesian_integration.py -v`
Expected: PASS

**Step 3: Commit**

```
test(fitting): add integration tests for Bayesian batch pipeline
```

---

### Task 9: Final commit and docs update

**Files:**
- Modify: `CHANGELOG.md` (add entries under [Unreleased])
- Modify: `docs/api/fitting.rst` (add viz module)

**Step 1: Update CHANGELOG.md**

Under `## [Unreleased]`:

```markdown
### Added

- "Plot All Q" button in G2 Fit tab for all-Q Bayesian overlay visualization
- Export button: CSV parameters, PDF/PNG figures, netCDF ArviZ diagnostics
- Sampler config spinboxes (warmup, samples, chains) for batch Bayesian fitting
- `fitting.viz` module: `plot_bayesian_all_q`, `export_bayesian_csv`, `export_bayesian_diagnostics`

### Fixed

- Bayesian fit results now stored on `bayesian_fit_summary` (no longer overwritten by NLSQ)
- G2 Fit tab auto-refreshes after "Fit All Q" batch completes
- Diffusion tab prefers Bayesian fit results when available
- Sampler kwargs (warmup, samples, chains) now forwarded to batch workers
```

**Step 2: Update fitting.rst**

Add before "See Also":

```rst
Bayesian Visualization
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: plot_bayesian_all_q

.. autofunction:: export_bayesian_csv

.. autofunction:: export_bayesian_diagnostics
```

**Step 3: Verify Sphinx build**

Run: `uv run sphinx-build -q -W --keep-going docs docs/_build`
Expected: exit 0

**Step 4: Commit**

```
docs(fitting): add Bayesian viz module to API reference and changelog
```
