# XPCS Viewer Dependency Diagram

## Legend

```
┌─────────┐
│ Module  │  Regular module
└─────────┘

┏━━━━━━━━━┓
┃ Module  ┃  High fan-in (integration hotspot)
┗━━━━━━━━━┛

╔═════════╗
║ Module  ║  High fan-out (brittle dependencies)
╚═════════╝

───────>    Dependency (A depends on B)
========>   Critical I/O boundary
- - - ->    Signal-based coupling (loose)
```

---

## Layer 1: Backend Abstraction (Foundation)

```
                    ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
                    ┃ backends._conversions   ┃  (FAN-IN: 9)
                    ┃ - ensure_numpy()        ┃
                    ┃ - ensure_backend_array()┃
                    ┗━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              ▲
                              │
                    ┏━━━━━━━━━┻━━━━━━━━━┓
                    ┃ backends          ┃  (FAN-IN: 8)
                    ┃ - get_backend()   ┃
                    ┃ - set_backend()   ┃
                    ┗━━━━━━━━━━━━━━━━━━━┛
                        ▲         ▲
            ┌───────────┴─────────┴────────────┐
            │                                  │
  ┌──────────────────┐              ┌──────────────────┐
  │ _numpy_backend   │              │ _jax_backend     │
  │ - NumPy impl     │              │ - JAX impl       │
  │                  │              │ - JIT support    │
  └──────────────────┘              └──────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │
                  ┌────────┴─────────┐
                  │ scipy_replacements│
                  │ - interpolate     │
                  │ - ndimage         │
                  │ - optimize        │
                  └───────────────────┘
```

---

## Layer 2: Data Access & I/O

```
┏━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ utils.logging_config   ┃  (FAN-IN: 24) - Used everywhere
┗━━━━━━━━━━━━━━━━━━━━━━━━┛

┌─────────────────────────────────────────────────────────┐
│                    HDF5 I/O Layer                       │
└─────────────────────────────────────────────────────────┘

       ┌──────────────────┐
       │ fileIO.aps_8idi  │  Beamline key mapping
       │ - key["g2"]      │
       │ - key["saxs1d"]  │
       └────────┬─────────┘
                │
                ▼
   ┏━━━━━━━━━━━━━━━━━━━━━┓
   ┃ fileIO.hdf_reader   ┃  (FAN-IN: 2)
   ┃ - HDF5ConnectionPool┃  Connection pooling
   ┃ - get(), put()      ┃
   ┗━━━━━━━━━┳━━━━━━━━━━━┛
             │
             │ Uses
             ▼
   ┌──────────────────────┐
   │ fileIO.qmap_utils    │
   │ - QMap class         │
   │ - get_qmap()         │
   └──────────────────────┘
```

---

## Layer 3: Core Data Model

```
                    ╔═════════════════════╗
                    ║ xpcs_file.py        ║  (FAN-IN: 2, FAN-OUT: 3)
                    ║ - XpcsFile class    ║  ** GOD OBJECT **
                    ║ - 39 public methods ║
                    ╚═══════════╤═════════╝
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
   ┌─────────────────┐ ┌────────────────┐ ┌─────────────────┐
   │ xpcs_file/cache │ │ xpcs_file/     │ │ xpcs_file/      │
   │ - DataCache     │ │ memory         │ │ fitting         │
   │ - CacheItem     │ │ - MemoryMonitor│ │ - legacy funcs  │
   └─────────────────┘ └────────────────┘ └─────────────────┘
                │               │
                └───────┬───────┘
                        │ Used by
                        ▼
            ┌─────────────────────┐
            │ viewer_kernel.py    │  (FAN-IN: 1)
            │ - ViewerKernel      │  Coordinates analysis
            │ - Lazy module load  │
            └──────────┬──────────┘
                       │
                       │ Loads analysis modules on demand
                       ▼
```

---

## Layer 4: Analysis Modules (Consumer Layer)

```
┌────────────────────────────────────────────────────────────┐
│                     Analysis Modules                        │
│  (All depend on: utils.logging_config, backends/_conversions)│
└────────────────────────────────────────────────────────────┘

        ┌──────────────┐        ╔══════════════════╗
        │ module.g2mod │        ║ module.twotime   ║  (FAN-OUT: 4)
        │ - G2 plots   │        ║ - Two-time corr  ║  ** BRITTLE **
        └──────────────┘        ╚══════════════════╝
                │                       │
                │                       ├───> backends
                │                       ├───> backends._conversions
                │                       ├───> backends.scipy_replacements
                │                       └───> utils.logging_config
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
┌──────────────┐  ┌──────────────┐
│ module.      │  │ module.      │
│ saxs1d       │  │ saxs2d       │
│ - 1D SAXS    │  │ - 2D SAXS    │
└──────────────┘  └──────────────┘
        │                │
        └────────┬───────┘
                 │ Both use
                 ▼
        ┌──────────────────┐
        │ backends.        │
        │ _conversions     │
        │ - ensure_numpy() │  ** I/O BOUNDARY **
        └────────┬─────────┘
                 │
                 ▼
        [ PyQtGraph Plots ]
```

---

## Layer 5: SimpleMask Subsystem (Loosely Coupled)

```
┌─────────────────────────────────────────────────────────┐
│              SimpleMask (Self-Contained)                 │
└─────────────────────────────────────────────────────────┘

    ┌─────────────────────────┐
    │ simplemask_window.py    │  QMainWindow
    │ - Signals:              │
    │   * mask_exported       │ ─ ─ ─ ─> [ XPCS Viewer ]
    │   * qmap_exported       │ ─ ─ ─ ─> [ XPCS Viewer ]
    └────────────┬────────────┘
                 │ Owns
                 ▼
    ╔═══════════════════════════╗
    ║ simplemask_kernel.py      ║  (FAN-IN: 2, FAN-OUT: 4)
    ║ - SimpleMaskKernel        ║
    ║ - compute_qmap()          ║
    ║ - compute_partition()     ║
    ╚═══════════╤═══════════════╝
                │
        ┌───────┼────────┬───────────┐
        │       │        │           │
        ▼       ▼        ▼           ▼
  ┌──────┐ ┏━━━━━┓ ┏━━━━━━━┓ ┌───────────┐
  │area_ │ ┃qmap ┃ ┃utils  ┃ │pyqtgraph_ │
  │mask  │ ┃     ┃ ┃       ┃ │mod        │
  └──┬───┘ ┗━━┳━━┛ ┗━━━┳━━━┛ └─────┬─────┘
     │        │        │           │
     │        │        │           │ (FAN-IN: 3)
     └────────┴────────┴───────────┘
              │ All use
              ▼
     ┏━━━━━━━━━━━━━━━━━┓
     ┃ backends        ┃
     ┃ backends.       ┃
     ┃ _conversions    ┃
     ┗━━━━━━━━━━━━━━━━━┛
```

---

## Layer 6: Fitting Module (JAX-Dependent)

```
┌─────────────────────────────────────────────────────────┐
│                  Fitting Subsystem                       │
│  (Requires JAX backend for NumPyro)                     │
└─────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │ fitting/__init__.py  │
    │ - fit_single_exp()   │
    │ - fit_double_exp()   │
    │ - nlsq_fit()         │
    └──────────┬───────────┘
               │
       ┌───────┼────────┐
       │       │        │
       ▼       ▼        ▼
  ┌────────┐ ┌─────┐ ┌────────┐
  │models  │ │nlsq │ │sampler │
  │- NumPyro│ │- JAX│ │- NUTS  │
  └────────┘ └──┬──┘ └────┬───┘
                │         │
                └────┬────┘
                     │ Requires
                     ▼
            ┏━━━━━━━━━━━━━┓
            ┃ backends    ┃  MUST be JAX backend
            ┃ - value_and_┃  (raises error otherwise)
            ┃   grad()    ┃
            ┗━━━━━━━━━━━━━┛
                     │
                     │ Cross-dependency
                     ▼
       ┌─────────────────────────┐
       │ simplemask.calibration  │
       │ - minimize_with_grad()  │
       │ - beam center fitting   │
       └─────────────────────────┘
```

---

## Critical I/O Boundaries (Conversion Points)

```
┌──────────────────────────────────────────────────────────┐
│        I/O Boundary Conversion (ensure_numpy)             │
└──────────────────────────────────────────────────────────┘

        [ JAX/NumPy Arrays ]
                 │
                 │ backends._conversions.ensure_numpy()
                 │
        ┌────────┴────────┬────────────┬───────────┐
        │                 │            │           │
        ▼                 ▼            ▼           ▼
┌─────────────┐  ┌─────────────┐  ┌─────────┐  ┌─────────┐
│ PyQtGraph   │  │ HDF5        │  │Matplotlib│  │Signals  │
│ plot.setData│  │ h5.create_  │  │plt.plot()│  │.emit()  │
│ (np.ndarray)│  │ dataset     │  │          │  │         │
└─────────────┘  └─────────────┘  └─────────┘  └─────────┘

** 9 modules use ensure_numpy() at these boundaries **

Modules with I/O conversions:
  - module.saxs1d  (PyQtGraph)
  - module.saxs2d  (PyQtGraph)
  - module.twotime (PyQtGraph + HDF5)
  - simplemask.qmap (Signals)
  - simplemask.area_mask (HDF5)
  - simplemask.utils (Partition export)
  - fitting.visualization (Matplotlib)
  - (2 more in tests/utils)
```

---

## Data Flow: XPCS Analysis Pipeline

```
┌──────────┐
│ HDF5 File│
│ /xpcs/   │
└────┬─────┘
     │ fileIO.hdf_reader.get()
     ▼
┌─────────────────────┐
│ XpcsFile            │ NumPy arrays (from HDF5)
│ - metadata          │
│ - qmap (dict)       │
│ - g2 data           │
└──────┬──────────────┘
       │
       │ viewer_kernel.plot_g2()
       ▼
┌─────────────────────┐
│ module.g2mod        │
│ - convert to backend│ <- ensure_backend_array()
│ - compute with JAX  │
│ - fit_g2()          │
└──────┬──────────────┘
       │
       │ fitting.fit_single_exp()
       ▼
┌─────────────────────┐
│ fitting.sampler     │ JAX arrays (computation)
│ - NLSQ warm-start   │
│ - NumPyro NUTS      │
└──────┬──────────────┘
       │
       │ FitResult.get_mean()
       ▼
┌─────────────────────┐
│ visualization       │
│ - ensure_numpy()    │ <- Convert back to NumPy
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Matplotlib Plot     │ NumPy arrays (display)
│ - Posterior plot    │
│ - Trace plot        │
└─────────────────────┘
```

---

## Data Flow: SimpleMask Integration

```
┌──────────────────┐
│ User Loads Image │
│ (NumPy array)    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│ SimpleMaskWindow        │
│ - User draws ROIs       │
│ - Edits mask            │
└────────┬────────────────┘
         │
         │ kernel.compute_qmap()
         ▼
┌─────────────────────────┐
│ simplemask.qmap         │
│ - Get backend           │ <- get_backend() (JAX or NumPy)
│ - JIT compile if JAX    │
│ - Compute Q-map         │
└────────┬────────────────┘
         │
         │ kernel.compute_partition()
         ▼
┌─────────────────────────┐
│ simplemask.utils        │
│ - generate_partition()  │ <- JIT compiled for JAX
│ - ensure_numpy()        │ <- Convert for export
└────────┬────────────────┘
         │
         │ window.export_to_viewer()
         ▼
┌─────────────────────────┐
│ Signal: qmap_exported   │ NumPy arrays + dict
│ {                       │
│   "partition_map": np,  │
│   "val_list": list,     │
│   "num_list": list      │
│ }                       │
└────────┬────────────────┘
         │ - - - - - (Signal, loose coupling)
         ▼
┌─────────────────────────┐
│ XPCS Viewer             │
│ - apply_qmap_result()   │
│ - Update analysis       │
└─────────────────────────┘
```

---

## Circular Dependency Check (None Found ✅)

```
Analysis of key modules:

xpcs_file.py
  ├─> xpcs_file.cache
  ├─> xpcs_file.memory
  └─> xpcs_file.fitting
      └─> (no xpcs_file import) ✅

viewer_kernel.py
  ├─> xpcs_file
  └─> module.* (lazy loaded)
      └─> (no viewer_kernel import) ✅

backends
  └─> (no internal dependencies) ✅

fitting
  ├─> backends
  └─> simplemask.calibration
      ├─> backends
      └─> (no fitting import) ✅

module.twotime
  ├─> backends
  ├─> backends._conversions
  ├─> backends.scipy_replacements
  └─> xpcs_file (for MemoryMonitor only)
      └─> (no module.twotime import) ✅

simplemask.simplemask_kernel
  ├─> simplemask.area_mask
  ├─> simplemask.qmap
  ├─> simplemask.utils
  └─> simplemask.pyqtgraph_mod
      └─> (no simplemask_kernel import) ✅

Conclusion: Clean dependency tree, no cycles detected.
```

---

## Proposed Architecture After Facade Migration

```
┌─────────────────────────────────────────────────────────┐
│                   NEW: Facade Layer                      │
└─────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │ io.hdf5_facade       │  ** NEW **
    │ - read_qmap()        │  Schema validation
    │ - write_partition()  │  Versioning
    │ - get_connection()   │  Connection pooling
    └──────────┬───────────┘
               │
               │ Uses
               ▼
    ┌──────────────────────┐
    │ repositories.        │  ** NEW **
    │ xpcs_repository      │  Repository pattern
    │ - get_g2_data()      │
    │ - save_fit_result()  │
    └──────────┬───────────┘
               │
               │ Used by
               ▼
    ┌──────────────────────┐
    │ services.            │  ** NEW **
    │ g2_analysis          │  Service layer
    │ - analyze_g2()       │  (extracted from XpcsFile)
    │ - fit_correlation()  │
    └──────────────────────┘


┌─────────────────────────────────────────────────────────┐
│              NEW: Backend I/O Adapters                   │
└─────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │ backends.io_adapter  │  ** NEW **
    │ - PyQtGraphAdapter   │  Centralized conversions
    │ - HDF5Adapter        │
    │ - MatplotlibAdapter  │
    └──────────────────────┘


BENEFITS:
  - Single source of truth for I/O patterns
  - Schema validation at boundaries
  - Easy to test with mocks
  - Performance monitoring in one place
  - Consistent error handling
```

---

## Anti-Patterns Identified

```
❌ God Object: xpcs_file.py
   - 39 public methods
   - Knows about fitting, SAXS, G2, two-time, tau-Q
   - Violates Single Responsibility Principle

   FIX: Extract to service layer
   ✅ services/g2_analysis.py
   ✅ services/saxs_analysis.py
   ✅ services/twotime_analysis.py


❌ Scattered I/O Conversions
   - 9 modules use ensure_numpy() directly
   - No centralized adapter pattern
   - Hard to audit I/O boundaries

   FIX: Backend I/O adapters
   ✅ backends/io_adapter.py


❌ Implicit Data Contracts
   - QMapDict passed as plain dict
   - No runtime validation
   - Typos cause silent failures

   FIX: Schema validators
   ✅ schemas/validators.py
   ✅ QMapSchema dataclass with validation


✅ Good Pattern: Signal-Based Integration
   - SimpleMask uses signals for export
   - Loose coupling, testable
   - KEEP THIS PATTERN for other integrations
```

---

## Complexity Metrics

```
Module Complexity (by Fan-In × Fan-Out):

High Risk (refactor priority):
  1. backends._conversions:  9 × 0 = 9  (high fan-in, stable)
  2. backends:               8 × 0 = 8  (high fan-in, stable)
  3. module.twotime:         0 × 4 = 4  (high fan-out, brittle)

Medium Risk:
  4. simplemask_kernel:      2 × 4 = 8  (balanced)
  5. xpcs_file:              2 × 3 = 6  (god object, needs refactor)

Low Risk:
  6. utils.logging_config:  24 × 0 = 0  (utility, stable)
  7. module.saxs1d:          0 × 2 = 2  (low coupling)
  8. module.saxs2d:          0 × 2 = 2  (low coupling)
```

---

**Document Version:** 1.0
**Generated:** 2026-01-06
**Maintained:** Sync with dependency_analysis.md
