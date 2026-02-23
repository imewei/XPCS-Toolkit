# Fit All-Q Bayesian Enhancement Design

**Date:** 2026-02-22
**Status:** Approved
**Scope:** Bug fixes + Plot All-Q + Export + Sampler Config

## Problem Statement

The "Fit All Q" button in the G2 Fit tab runs batch Bayesian fitting via
`BatchBayesianCoordinator` but has five integration gaps:

1. G2 Fit tab is not auto-refreshed after batch completes
2. `xf.fit_summary` is overwritten by NLSQ when user re-plots, losing Bayesian results
3. No "Plot All-Q" button to visualize all Q-bin Bayesian fits at once
4. No export for Bayesian results, traces, or diagnostics
5. `sampler_kwargs` (warmup, samples, chains) not configurable from UI

## Design

### Dual Storage Model

| Attribute | Source | Content |
|-----------|--------|---------|
| `xf.fit_summary` | NLSQ "Fit" button | Point-estimate fit (unchanged) |
| `xf.bayesian_fit_summary` | "Fit All Q" | Bayesian posterior means in legacy format |
| `xf.bayesian_results` | "Fit All Q" | `dict[int, FitResult]` per-Q posteriors + ArviZ data |

- NLSQ "Fit" never touches `bayesian_fit_summary`
- "Fit All Q" never touches `fit_summary`
- Diffusion tab: prefers `bayesian_fit_summary` when available
- Both can coexist independently

### Auto-Refresh After Batch

`_on_g2_batch_finished` stores results on `xf.bayesian_fit_summary` and
`xf.bayesian_results`, then calls `plot_g2_fitting()` to refresh the G2
Fit tab immediately.

### "Plot All-Q" Button

New `QPushButton("Plot All Q")` next to "Fit All Q". Handler:

1. Guard: `xf.bayesian_fit_summary` must exist
2. Generate matplotlib figure via `fitting.viz.plot_bayesian_all_q()`:
   - All-Q overlay: data + Bayesian fit lines + 95% CI bands, color-coded by Q
3. Display in `QDialog` with `NavigationToolbar2QT`
4. Include "Export" button

### Export

Three outputs from a single "Export" action:

| Output | Format | Content |
|--------|--------|---------|
| Parameters | CSV | Per-Q: param means/stds, R-hat, ESS |
| Figures | PDF/PNG | All-Q overlay with CI bands |
| Diagnostics | netCDF | ArviZ InferenceData (full posteriors) |

Export uses `QFileDialog.getExistingDirectory()` then writes all files.

### Sampler Config UI

Collapsible group box with three `QSpinBox`es:

| Setting | Default | Range |
|---------|---------|-------|
| Warmup | 500 | 100-5000 |
| Samples | 1000 | 200-10000 |
| Chains | 4 | 1-8 |

Values packed into `sampler_kwargs` and passed via spec dicts to
`BayesianFitWorker`.

### `bayesian_assembly.py` Changes

- Add `"source": "bayesian"` to returned `fit_summary` dict
- Accept and forward `q_range`, `t_range` arguments

### Files Changed

| File | Scope |
|------|-------|
| `xpcs_viewer.py` | New buttons, sampler config, handlers, dual storage |
| `xpcs_file.py` | Add `bayesian_fit_summary`, `bayesian_results` attributes |
| `bayesian_assembly.py` | Add `source` field, accept range args |
| `viewer_kernel.py` | `plot_tauq_pre` prefers bayesian results |
| `gui/layout_helpers.py` | Grid positions for new widgets |
| `fitting/viz.py` (new) | `plot_bayesian_all_q()` + export logic |
