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
_DOUBLE_COLS = (
    "contrast1",
    "tau1",
    "stretch1",
    "baseline",
    "tau2",
    "contrast2",
    "stretch2",
)


def plot_bayesian_all_q(
    bayesian_summary: dict[str, Any] | None,
    g2_data: NDArray | None,
    *,
    data_t_el: NDArray | None = None,
) -> Figure | None:
    """Generate all-Q overlay figure with Bayesian fit lines.

    Parameters
    ----------
    bayesian_summary : dict or None
        Output of ``assemble_fit_summary`` with ``source='bayesian'``.
    g2_data : ndarray, shape (num_t, num_q)
        Raw G2 correlation data.
    data_t_el : ndarray or None
        Time axis matching ``g2_data`` rows.  When the caller applies a
        ``t_range`` filter the resulting array may be shorter than the
        summary's ``t_el``.  Pass the filtered time array here so that
        data points are plotted at the correct times.  Falls back to
        ``bayesian_summary["t_el"]`` when *None*.

    Returns
    -------
    Figure or None
        Matplotlib figure, or None if no data.
    """
    if bayesian_summary is None or g2_data is None:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    fit_line = bayesian_summary["fit_line"]
    fit_x = bayesian_summary["fit_x"]
    q_val = bayesian_summary["q_val"]
    t_el = data_t_el if data_t_el is not None else bayesian_summary["t_el"]
    failed_mask = bayesian_summary.get("failed_mask")
    num_q = len(q_val)

    fig, ax = plt.subplots(figsize=(10, 7))
    q_min, q_max = q_val.min(), q_val.max()
    if q_min == q_max:
        q_max = q_min + 1.0
    norm = Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.get_cmap("viridis")

    # --- Batch data points with a single scatter call ---
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    for qi in range(min(num_q, g2_data.shape[1])):
        valid = np.isfinite(g2_data[:, qi])
        n_valid = int(valid.sum())
        if n_valid > 0:
            all_t.append(t_el[valid])
            all_y.append(g2_data[valid, qi])
            all_c.append(np.full(n_valid, q_val[qi]))

    if all_t:
        t_cat = np.concatenate(all_t)
        y_cat = np.concatenate(all_y)
        c_cat = np.concatenate(all_c)
        ax.scatter(
            t_cat,
            y_cat,
            c=c_cat,
            cmap=cmap,
            norm=norm,
            s=9,
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )

    # --- Batch fit lines with a single LineCollection ---
    segments = []
    seg_colors = []
    for qi in range(num_q):
        if failed_mask is not None and failed_mask[qi]:
            continue
        if np.any(np.isfinite(fit_line[qi])):
            pts = np.column_stack([fit_x, fit_line[qi]])
            segments.append(pts)
            seg_colors.append(cmap(norm(q_val[qi])))

    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
        ax.add_collection(lc)
        ax.autoscale_view()

    ax.set(xscale="log", xlabel="Delay time (s)", ylabel=r"$g_2(\tau)$")
    title = "All-Q Bayesian Fit Overview"
    if failed_mask is not None:
        title += f"\n{int((~failed_mask).sum())}/{num_q} Q-bins succeeded"
    ax.set_title(title)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label(r"Q ($\AA^{-1}$)")

    fig.tight_layout()
    return fig


def export_bayesian_csv(
    path: Path | str,
    fit_val: NDArray,
    q_val: NDArray,
    fit_func_name: str,
    *,
    failed_mask: NDArray | None = None,
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
    failed_mask : ndarray or None
        Boolean array (True = failed). Adds a ``status`` column when provided.
    """
    cols = _SINGLE_COLS if fit_func_name == "single" else _DOUBLE_COLS
    path = Path(path)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["q_value"]
        for col in cols:
            header.extend([f"{col}_mean", f"{col}_std"])
        if failed_mask is not None:
            header.append("status")
        writer.writerow(header)

        for qi in range(len(q_val)):
            row = [f"{q_val[qi]:.6f}"]
            for ci in range(len(cols)):
                row.append(f"{fit_val[qi, 0, ci]:.6g}")
                row.append(f"{fit_val[qi, 1, ci]:.6g}")
            if failed_mask is not None:
                row.append("failed" if failed_mask[qi] else "ok")
            writer.writerow(row)

    logger.info("Exported Bayesian parameters to %s", path)


def export_bayesian_diagnostics(
    path: Path | str,
    bayesian_results: dict[int, Any],
) -> None:
    """Export ArviZ DataTree to netCDF for all Q-bins.

    Parameters
    ----------
    path : Path
        Output netCDF file path.
    bayesian_results : dict[int, FitResult]
        Per-Q FitResult objects with arviz_data attribute.
    """
    path = Path(path)

    try:
        import arviz as az  # noqa: F401
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

    # Write one netCDF file per Q-bin to preserve all diagnostics
    for q_idx, idata in sorted(datasets.items()):
        q_path = path.with_stem(f"{path.stem}_q{q_idx:03d}")
        idata.to_netcdf(str(q_path))

    logger.info(
        "Exported ArviZ diagnostics to %s (%d Q-bin files)", path.parent, len(datasets)
    )
