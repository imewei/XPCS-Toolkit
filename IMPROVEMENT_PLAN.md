# Technical Guidelines Compliance Improvement Plan

Generated: 2026-01-15

## Executive Summary

This document identifies gaps between the xpcsviewer codebase and the Technical Guidelines defined in `~/.claude/CLAUDE.md`, with prioritized action items for remediation.

---

## 1. Package Management ✅ COMPLIANT

**Status: No action required**

- Python 3.12+ enforced via `requires-python = ">=3.12"`
- `uv` is primary package manager with intelligent fallback
- Local `.venv/` only; no global site-packages
- `uv.lock` is single source of truth

---

## 2. JAX-First Computational Core

**Status: B+ (Good foundation, minor gaps)**

### 2.1 Compliant Areas
- JIT-compiled fitting models (`@jax.jit` on 4 core functions)
- NLSQ 0.6.0 as primary optimizer (not scipy.optimize)
- Backend abstraction with `get_backend()` and `ensure_numpy()`
- `interpax` infrastructure built in `backends/scipy_replacements/`

### 2.2 Action Items

#### HIGH PRIORITY: Migrate scipy.interpolate usage to interpax

| File | Line | Function | Current | Target |
|------|------|----------|---------|--------|
| `module/g2mod.py` | 711 | `interpolate_g2_data()` | `scipy.interpolate.interp1d` | `backends.scipy_replacements.interpolate.Interp1d` |
| `module/saxs1d.py` | 435 | `subtract_background()` | `scipy.interpolate.interp1d` | `backends.scipy_replacements.interpolate.Interp1d` |
| `module/twotime_utils.py` | 675 | `adaptive_sampling()` | `scipy.ndimage.zoom` | JAX-based resize or `backends.scipy_replacements` |

**Implementation:**
```python
# Before (g2mod.py:711)
from scipy.interpolate import interp1d

# After
from xpcsviewer.backends.scipy_replacements.interpolate import Interp1d as interp1d
```

#### MEDIUM PRIORITY: Expand vmap usage in analysis modules

Currently vmap is used in:
- `backends/scipy_replacements/ndimage.py` (convolve1d)
- `utils/vectorized_roi.py` (batch ROI processing)

Consider vmap for:
- Batch G2 fitting across multiple q-values
- Parallel SAXS background subtraction

---

## 3. Data Integrity

**Status: B- (Violations in visualization layer)**

### 3.1 CRITICAL: Silent Downsampling in Visualization

**Files with violations:**

| File | Line | Issue | Severity |
|------|------|-------|----------|
| `utils/visualization_optimizer.py` | 145-177 | `ImageDisplayOptimizer.optimize_image_for_display()` silently downsamples | HIGH |
| `utils/visualization_optimizer.py` | 350-356 | `PlotPerformanceOptimizer.optimize_matplotlib_plot()` downsamples to 10k points | HIGH |
| `utils/visualization_optimizer.py` | 746-768 | `AdvancedGUIRenderer.optimize_large_dataset_display()` peak-preserving downsample | MEDIUM |
| `utils/validation.py` | 201-216 | `validate_array_compatibility()` silently trims arrays | MEDIUM |

**Required Fix:**
```python
# Add to config or function signature
class VisualizationConfig:
    allow_downsampling: bool = False  # Default OFF per guidelines
    downsample_threshold: int = 100_000
    log_downsampling: bool = True  # Always log when active
```

### 3.2 CRITICAL: Uncontrolled Posterior Sampling

**Location:** `fitting/visualization.py` lines 406, 615

```python
# Current: silent subsampling
np.random.choice(n_samples, min(n_draws, n_samples), replace=False)

# Required: explicit parameter with logging
def plot_credible_intervals(
    ...,
    max_draws: int | None = None,  # None = use all samples
    subsample_seed: int | None = None,
):
    if max_draws is not None and n_samples > max_draws:
        logger.info(f"Subsampling posterior: {n_samples} → {max_draws} draws")
```

### 3.3 Action Items

| Priority | Task | File(s) |
|----------|------|---------|
| HIGH | Add `disable_visualization_optimization` config flag | `utils/visualization_optimizer.py` |
| HIGH | Make downsampling opt-in with explicit user consent | `utils/visualization_optimizer.py` |
| HIGH | Add `max_draws` parameter to posterior visualization | `fitting/visualization.py` |
| MEDIUM | Change array trimming to raise ValidationError instead of silent truncation | `utils/validation.py` |
| LOW | Add data integrity audit trail (log all subsampling decisions) | Global |

---

## 4. Bayesian Inference Pipeline

**Status: B (Solid core, missing artifacts)**

### 4.1 Compliant Areas
- NumPyro NUTS sampler with proper configuration
- NLSQ → NUTS warm-start pipeline implemented
- ArviZ diagnostics: R-hat, ESS bulk/tail, divergences
- Random seed support via `SamplerConfig.random_seed`

### 4.2 Action Items

#### HIGH PRIORITY: Add BFMI Diagnostic

**Location:** `fitting/sampler.py` line ~119 (`_build_fit_result`)

```python
# Add BFMI computation
import arviz as az

def _build_fit_result(...):
    # Existing diagnostics...

    # Add BFMI
    idata = az.from_numpyro(mcmc)
    bfmi = az.bfmi(idata)

    diagnostics = FitDiagnostics(
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        divergences=num_divergent,
        bfmi=float(bfmi.mean()),  # NEW FIELD
    )
```

**Also update:** `fitting/results.py` `FitDiagnostics` dataclass:
```python
@dataclass
class FitDiagnostics:
    r_hat: dict[str, float]
    ess_bulk: dict[str, int]
    ess_tail: dict[str, int]
    divergences: int
    bfmi: float | None = None  # NEW FIELD
```

#### HIGH PRIORITY: Software Version Tracking in Artifacts

**Location:** `fitting/results.py` `FitResult.to_dict()` (line ~644)

```python
import sys
import jax
import numpyro
import arviz as az
import nlsq
from xpcsviewer import __version__

def to_dict(self) -> dict[str, Any]:
    return {
        # Existing fields...
        "versions": {
            "xpcsviewer": __version__,
            "numpyro": numpyro.__version__,
            "jax": jax.__version__,
            "arviz": az.__version__,
            "nlsq": nlsq.__version__,
            "python": sys.version,
        },
        "sampler_config": {
            "num_warmup": self.config.num_warmup,
            "num_samples": self.config.num_samples,
            "num_chains": self.config.num_chains,
            "target_accept_prob": self.config.target_accept_prob,
            "max_tree_depth": self.config.max_tree_depth,
            "random_seed": self.config.random_seed,
        },
        "data_metadata": {
            "n_points": len(self.x),
            "x_range": [float(self.x.min()), float(self.x.max())],
        },
    }
```

#### LOW PRIORITY: Alternative Samplers (Oryx, Blackjax)

Not critical but would add redundancy. Consider for future:
- Blackjax for alternative HMC implementations
- Oryx for functional composition

---

## 5. Code Quality

**Status: A- (Strong compliance)**

### 5.1 Compliant Areas
- Zero wildcard imports in production code
- Comprehensive type hints at public APIs
- 1,260+ structured logging calls
- Custom exception hierarchy with context and recovery suggestions
- No silent fallbacks (0 bare `except: pass` patterns)

### 5.2 Action Items

#### LOW PRIORITY: Add missing type hints

**Location:** `helper/utils.py`

```python
# Current
def get_min_max(data, min_percent=0, max_percent=100, **kwargs):

# Required
def get_min_max(
    data: np.ndarray,
    min_percent: float = 0,
    max_percent: float = 100,
    **kwargs: Any,
) -> tuple[float, float]:
```

Functions to annotate:
- `get_min_max()` (line 18)
- `norm_saxs_data()` (line 47)
- `create_slice()` (line 65)

---

## 6. GUI & Visualization

**Status: B+ (Hard-locked to PySide6)**

### 6.1 Compliant Areas
- GUI is pure view layer (zero numerical code)
- Full Light/Dark theming with system detection
- Token-based design system with QSS modularity
- PyQtGraph for interactive, Matplotlib for publication
- Unified color palette across backends

### 6.2 NON-COMPLIANT: Hard-locked to PySide6

**Current state:** All imports use `from PySide6 import ...` directly

**Affected files:**
- `xpcsviewer/xpcs_viewer.py`
- `xpcsviewer/plothandler/matplot_qt.py`
- `xpcsviewer/plothandler/pyqtgraph_handler.py`
- `xpcsviewer/gui/theme/manager.py`
- `xpcsviewer/gui/layout_helpers.py`
- All files in `xpcsviewer/gui/widgets/`
- All files in `xpcsviewer/gui/shortcuts/`

### 6.3 Action Items

#### MEDIUM PRIORITY: Adopt qtpy abstraction layer

**Step 1:** Add dependency
```toml
# pyproject.toml
dependencies = [
    "qtpy>=2.4.0",
    # Remove hard pyside6 requirement or make optional
]
```

**Step 2:** Create compatibility shim
```python
# xpcsviewer/gui/qt_compat.py
"""Qt binding compatibility layer."""
import os

# Allow runtime override via environment variable
# Default to PySide6 for backwards compatibility
os.environ.setdefault("QT_API", "pyside6")

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Signal, Slot, QObject
from qtpy.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QGroupBox,
    # ... etc
)

__all__ = [
    "QtCore", "QtGui", "QtWidgets",
    "Signal", "Slot", "QObject",
    "QMainWindow", "QWidget", "QPushButton", "QGroupBox",
]
```

**Step 3:** Update imports across codebase
```python
# Before
from PySide6.QtWidgets import QMainWindow, QPushButton

# After
from xpcsviewer.gui.qt_compat import QMainWindow, QPushButton
```

**Estimated effort:** 2-3 hours for mechanical refactor

---

## Priority Matrix

| Priority | Category | Task | Effort |
|----------|----------|------|--------|
| **P0** | Data Integrity | Add `disable_visualization_optimization` flag | 1h |
| **P0** | Data Integrity | Make posterior subsampling explicit | 1h |
| **P1** | Bayesian | Add BFMI diagnostic | 30m |
| **P1** | Bayesian | Add version tracking to artifacts | 1h |
| **P1** | JAX-First | Migrate 3x scipy.interpolate to interpax | 2h |
| **P2** | GUI | Adopt qtpy abstraction | 3h |
| **P2** | Data Integrity | Change array trimming to raise error | 1h |
| **P3** | Code Quality | Add type hints to helper/utils.py | 30m |

---

## Implementation Order

### Phase 1: Critical Data Integrity (Week 1)
1. Add visualization optimization config flag
2. Make posterior subsampling explicit with logging
3. Change silent array trimming to validation error

### Phase 2: Bayesian Completeness (Week 1-2)
4. Add BFMI to FitDiagnostics
5. Add version/config tracking to FitResult.to_dict()

### Phase 3: JAX Migration (Week 2)
6. Migrate g2mod.py interpolation
7. Migrate saxs1d.py interpolation
8. Migrate twotime_utils.py zoom

### Phase 4: GUI Flexibility (Week 3)
9. Create qt_compat.py shim
10. Refactor imports to use qtpy

### Phase 5: Polish (Ongoing)
11. Add missing type hints
12. Expand vmap usage in analysis

---

## Verification Checklist

After implementation, verify:

- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] No scipy.interpolate imports in module/
- [ ] `FitResult.to_dict()` includes versions and config
- [ ] Visualization downsampling is OFF by default
- [ ] Posterior plots warn when subsampling
- [ ] GUI launches with both `QT_API=pyside6` and `QT_API=pyqt6`
