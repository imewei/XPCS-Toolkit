# System Overview

High-level architecture of the XPCS Viewer codebase, showing module dependencies
and subsystem boundaries.

## Module Dependency Diagram

```{mermaid}
graph TB
    subgraph GUI["GUI Layer (PySide6)"]
        XV[xpcs_viewer.py<br/>Main Window]
        VK[viewer_kernel.py<br/>Analysis Orchestrator]
        VU[viewer_ui.py<br/>UI Layout]
        GM[gui/<br/>Theme, State, Shortcuts]
    end

    subgraph Analysis["Analysis Modules"]
        G2[module/g2mod<br/>G2 Correlation]
        S1[module/saxs1d<br/>1D SAXS]
        S2[module/saxs2d<br/>2D SAXS]
        TT[module/twotime<br/>Two-Time Correlation]
        TQ[module/tauq<br/>Tau-Q Analysis]
        IT[module/intt<br/>Intensity-Time]
    end

    subgraph SM["SimpleMask Subsystem"]
        SMW[simplemask_window.py<br/>QMainWindow]
        SMK[simplemask_kernel.py<br/>Computation Core]
        QM[qmap.py<br/>Q-Map Computation]
        AM[area_mask.py<br/>Mask Assembly]
        SU[utils.py<br/>Partition Generation]
    end

    subgraph Fitting["Fitting Pipeline"]
        FI[fitting/__init__<br/>Public API]
        NL[fitting/nlsq<br/>NLSQ 0.6.0 Wrapper]
        FS[fitting/sampler<br/>NumPyro NUTS]
        FM[fitting/models<br/>Probabilistic Models]
        FR[fitting/results<br/>NLSQResult, FitResult]
    end

    subgraph Backend["Backend Abstraction"]
        BE[backends/__init__<br/>get_backend]
        BP[backends/_base<br/>BackendProtocol]
        JB[backends/_jax_backend<br/>JAXBackend]
        NB[backends/_numpy_backend<br/>NumPyBackend]
        DM[backends/_device<br/>DeviceManager]
        CV[backends/_conversions<br/>ensure_numpy]
        IA[backends/io_adapter<br/>I/O Adapters]
    end

    subgraph IO["I/O Layer"]
        HF[io/hdf5_facade<br/>HDF5Facade]
        HR[fileIO/hdf_reader<br/>Connection Pool]
        QU[fileIO/qmap_utils<br/>Q-Map I/O]
        AK[fileIO/aps_8idi<br/>Beamline Key Map]
    end

    subgraph Schema["Schema Validation"]
        SV[schemas/validators<br/>QMapSchema, G2Data, etc.]
    end

    subgraph Core["Core Data Model"]
        XF[xpcs_file.py<br/>XpcsFile]
        XC[xpcs_file/cache<br/>DataCache]
        XM[xpcs_file/memory<br/>MemoryMonitor]
        FL[file_locator.py<br/>FileLocator]
    end

    subgraph Utils["Utilities"]
        LC[utils/logging_config<br/>LoggingConfig]
        LU[utils/log_utils<br/>LoggingContext, RateLimitedLogger]
    end

    subgraph Threading["Threading"]
        UT[threading/unified_threading<br/>UnifiedThreadingManager]
        PW[threading/plot_workers<br/>Plot Workers]
        AW[threading/async_workers<br/>Async Workers]
    end

    %% GUI dependencies
    XV --> VK
    XV --> VU
    XV --> GM
    XV --> UT
    VK --> XF
    XV --> FL
    FL --> XF
    VK -.->|lazy load| G2
    VK -.->|lazy load| S1
    VK -.->|lazy load| S2
    VK -.->|lazy load| TT
    VK -.->|lazy load| TQ
    VK -.->|lazy load| IT

    %% Analysis dependencies
    G2 --> FI
    G2 --> CV
    S1 --> CV
    S2 --> CV
    TT --> CV
    TT --> BE
    IT --> CV
    TQ --> CV

    %% SimpleMask
    SMW -.->|signals| XV
    SMW --> SMK
    SMK --> QM
    SMK --> AM
    SMK --> SU
    QM --> BE
    QM --> CV
    SU --> BE

    %% Fitting
    FI --> NL
    FI --> FS
    NL --> FR
    FS --> FM
    FS --> FR
    FM --> BE

    %% I/O
    HF --> HR
    HF --> SV
    XF --> HR
    XF --> QU
    QU --> AK

    %% Backend
    BE --> JB
    BE --> NB
    JB --> BP
    NB --> BP
    JB --> DM
    IA --> CV

    %% Cross-cutting
    LC -.->|used by all| XV
```

## Subsystem Responsibilities

### GUI Layer
The GUI layer uses PySide6 and PyQtGraph for interactive visualization. `xpcs_viewer.py`
is the main window; `viewer_kernel.py` coordinates lazy-loaded analysis modules. The
`gui/` subpackage provides theme management (light/dark), session state persistence,
keyboard shortcuts, and a command palette.

### Core Data Model
`XpcsFile` is the in-memory representation of an XPCS dataset loaded from HDF5.
`DataCache` provides LRU caching for array reads, and `MemoryMonitor` tracks
system memory to prevent out-of-memory conditions. `FileLocator` discovers and
manages multiple HDF5 files for batch operations, providing search and filtering
across dataset directories.

### Analysis Modules
Each analysis module (`g2mod`, `saxs1d`, `saxs2d`, `twotime`, `tauq`, `intt`) handles a
specific XPCS analysis task. Modules are loaded lazily by `viewer_kernel.py` to minimize
startup time. All modules convert arrays to NumPy at I/O boundaries via `ensure_numpy()`.

### SimpleMask Subsystem
A self-contained subsystem for mask editing and Q-map computation. Communicates with the
main viewer exclusively through Qt signals (`mask_exported`, `qmap_exported`), maintaining
loose coupling.

### Fitting Pipeline
Two-stage fitting: NLSQ 0.6.0 for fast point estimates (warm-start), followed by NumPyro
NUTS for full Bayesian inference. Results are wrapped in `NLSQResult` (point estimates) or
`FitResult` (posterior samples with ArviZ diagnostics).

### Backend Abstraction
Protocol-based abstraction over JAX and NumPy. Auto-detects JAX availability. Provides
I/O adapters for boundary conversions and a device manager for GPU/CPU selection.

### I/O Layer
`HDF5Facade` provides unified HDF5 access with schema validation and connection pooling.
The existing `HDF5ConnectionPool` manages file handle reuse.

### Schema Validation
Frozen dataclasses (`QMapSchema`, `GeometryMetadata`, `G2Data`, `PartitionSchema`,
`MaskSchema`) enforce type safety, shape consistency, and physical constraints at I/O
boundaries.

### Threading
`UnifiedThreadingManager` coordinates background tasks with priority queues, pool
management, and progress reporting for GUI responsiveness.
