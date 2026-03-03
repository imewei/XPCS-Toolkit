# Data Flow

This document describes the primary data flows through XPCS Viewer, from HDF5 file
ingestion through computation to GUI visualization.

## XPCS Analysis Pipeline

The main data flow for G2 correlation analysis:

```{mermaid}
flowchart TD
    HDF5["HDF5 File<br/>/xpcs/ hierarchy"]
    CP["HDF5ConnectionPool<br/>fileIO/hdf_reader.py"]
    XF["XpcsFile<br/>In-memory data model"]
    DC["DataCache<br/>LRU cache for arrays"]
    VK["ViewerKernel<br/>Analysis coordinator"]

    subgraph Backend["Backend Computation"]
        BE["get_backend()<br/>JAX or NumPy"]
        COMP["Array Operations<br/>backend.exp(), backend.mean(), ..."]
        JIT["JIT Cache<br/>(JAX only)"]
    end

    subgraph Fitting["Fitting Pipeline"]
        NLSQ["NLSQ 0.6.0<br/>Point estimates"]
        NUTS["NumPyro NUTS<br/>Posterior samples"]
        FRES["FitResult<br/>ArviZ diagnostics"]
    end

    subgraph Output["Output Boundaries"]
        PQG["PyQtGraph<br/>Interactive plots"]
        MPL["Matplotlib<br/>Publication plots"]
        H5W["HDF5 Write<br/>Fit results"]
    end

    EN["ensure_numpy()<br/>I/O boundary conversion"]

    HDF5 -->|"pool.get_connection()"| CP
    CP -->|"np.ndarray (float64)"| XF
    XF -->|"cached reads"| DC
    XF -->|"g2_data, qmap"| VK
    VK -->|"lazy module load"| COMP
    VK -->|"fit request"| NLSQ

    BE -->|"backend arrays"| COMP
    COMP -->|"JIT-compiled"| JIT

    NLSQ -->|"warm-start init"| NUTS
    NUTS -->|"posterior samples"| FRES

    COMP -->|"analysis results"| EN
    FRES -->|"plot data"| EN

    EN -->|"np.ndarray"| PQG
    EN -->|"np.ndarray"| MPL
    EN -->|"np.ndarray"| H5W

    style EN fill:#f96,stroke:#333,color:#000
    style BE fill:#69f,stroke:#333,color:#000
    style NLSQ fill:#9c6,stroke:#333,color:#000
    style NUTS fill:#9c6,stroke:#333,color:#000
```

### Array Type Transitions

| Stage | Array Type | Notes |
|-------|-----------|-------|
| HDF5 read | `np.ndarray` | Always NumPy from h5py |
| XpcsFile storage | `np.ndarray` | Cached as NumPy |
| Backend computation | JAX array or `np.ndarray` | Depends on active backend |
| JIT cache | JAX array | Compiled traces cached |
| NLSQ fitting | `np.ndarray` | NLSQ operates on NumPy |
| NumPyro NUTS | JAX array | Requires JAX backend |
| I/O boundary | `np.ndarray` | `ensure_numpy()` conversion |
| PyQtGraph/Matplotlib | `np.ndarray` | Display libraries require NumPy |

## SimpleMask Data Flow

Mask editing and Q-map computation flow:

```{mermaid}
flowchart TD
    IMG["Detector Image<br/>np.ndarray (H, W)"]
    SMW["SimpleMaskWindow<br/>User draws ROIs"]
    SMK["SimpleMaskKernel<br/>Computation core"]

    subgraph MaskOps["Mask Operations"]
        AM["AreaMask<br/>Undo/redo history"]
        DT["DrawingTools<br/>Rect, Circle, Polygon, ..."]
        MASK["Final Mask<br/>int32, values 0/1"]
    end

    subgraph QMapComp["Q-Map Computation"]
        GEO["GeometryMetadata<br/>bcx, bcy, det_dist, ..."]
        QM["qmap.compute_qmap()<br/>JIT-compiled (JAX)"]
        QRES["QMapSchema<br/>sqmap, dqmap, phis"]
    end

    subgraph Partition["Partition Generation"]
        PG["utils.generate_partition()<br/>JIT-compiled (JAX)"]
        PS["PartitionSchema<br/>partition_map, val_list, num_list"]
    end

    subgraph Export["Signal Export"]
        SIG1["mask_exported<br/>Signal(np.ndarray)"]
        SIG2["qmap_exported<br/>Signal(dict)"]
    end

    XV["XPCS Viewer<br/>apply_mask(), apply_qmap_result()"]
    H5["HDF5 File<br/>Mask/Partition persistence"]

    IMG --> SMW
    SMW --> SMK
    SMK --> AM
    DT --> AM
    AM --> MASK

    SMK --> QM
    GEO --> QM
    QM --> QRES

    QRES --> PG
    MASK --> PG
    PG --> PS

    MASK --> SIG1
    PS --> SIG2

    SIG1 -.->|"loose coupling"| XV
    SIG2 -.->|"loose coupling"| XV

    SMK --> H5

    style SIG1 fill:#f96,stroke:#333,color:#000
    style SIG2 fill:#f96,stroke:#333,color:#000
```

### Signal Payload Schemas

**`mask_exported(np.ndarray)`**:
- Shape: `(H, W)`, dtype: `int32`
- Values: 0 (masked) or 1 (valid)

**`qmap_exported(dict)`**:
```python
{
    "partition_map": np.ndarray,  # int32, (H, W), Q-bin indices
    "num_pts": int,               # Number of Q-bins
    "val_list": list[float],      # Q-bin center values
    "num_list": list[int],        # Pixels per Q-bin
}
```

## Fitting Data Flow

The two-stage Bayesian fitting pipeline:

```{mermaid}
flowchart TD
    G2["G2 Data<br/>delay_times, g2, g2_err"]

    subgraph Stage1["Stage 1: NLSQ Warm-Start"]
        NF["nlsq.nlsq_optimize()"]
        NR["NLSQResult<br/>Point estimates + CurveFitResult"]
        P0["Initial params<br/>tau, baseline, contrast"]
    end

    subgraph Stage2["Stage 2: Bayesian Inference"]
        MC["NumPyro Model<br/>single_exp_model()"]
        NU["NUTS Sampler<br/>4 chains x 1000 samples"]
        PS["Posterior Samples<br/>dict[str, ndarray]"]
    end

    subgraph Diagnostics["Convergence Diagnostics"]
        RH["R-hat < 1.01"]
        ESS["ESS bulk/tail > 400"]
        DIV["Divergences == 0"]
        BF["BFMI >= 0.2"]
        FD["FitDiagnostics"]
    end

    subgraph Results["Result Objects"]
        FR["FitResult<br/>samples + diagnostics"]
        AZ["xarray DataTree<br/>Posterior plots"]
    end

    G2 --> NF
    NF --> NR
    NR -->|"warm-start"| P0
    P0 --> MC
    MC --> NU
    NU --> PS

    PS --> RH
    PS --> ESS
    PS --> DIV
    PS --> BF
    RH --> FD
    ESS --> FD
    DIV --> FD
    BF --> FD

    PS --> FR
    FD --> FR
    FR --> AZ

    style NR fill:#9c6,stroke:#333,color:#000
    style FR fill:#69f,stroke:#333,color:#000
    style FD fill:#f96,stroke:#333,color:#000
```

### Convergence Thresholds

| Diagnostic | Threshold | Meaning |
|-----------|-----------|---------|
| R-hat | < 1.01 | All chains converge to same distribution |
| ESS bulk | > 400 | Sufficient effective samples for mean/variance |
| ESS tail | > 400 | Sufficient effective samples for quantiles |
| Divergences | == 0 | No divergent transitions |
| BFMI | >= 0.2 | Adequate energy exploration |

## HDF5 File Schema

The standard HDF5 file structure for XPCS data:

```{mermaid}
graph LR
    subgraph HDF5["HDF5 File Structure"]
        ROOT["/"]
        XPCS["/xpcs/"]
        QMAP["/xpcs/qmap/"]
        G2["/xpcs/g2/"]
        META["/xpcs/metadata/"]
        SM["/simplemask/"]
        SMMASK["/simplemask/mask/"]
        SMPART["/simplemask/partition/"]
    end

    ROOT --> XPCS
    ROOT --> SM
    XPCS --> QMAP
    XPCS --> G2
    XPCS --> META
    SM --> SMMASK
    SM --> SMPART

    subgraph QMapDS["Q-Map Datasets"]
        SQ["sqmap<br/>float64, (H,W)"]
        DQ["dqmap<br/>float64, (H,W)"]
        PH["phis<br/>float64, (H,W)"]
        MK["mask<br/>int32, (H,W)"]
        PM["partition_map<br/>int32, (H,W)"]
    end

    subgraph G2DS["G2 Datasets"]
        G2V["g2<br/>float64, (n_delay, n_q)"]
        G2E["g2_err<br/>float64, (n_delay, n_q)"]
        DT["delay_times<br/>float64, (n_delay,)"]
        QV["q_values<br/>float64, (n_q,)"]
    end

    subgraph MetaDS["Metadata Attributes"]
        BCX["bcx: float"]
        BCY["bcy: float"]
        DD["det_dist: float (mm)"]
        LAM["lambda_: float (A)"]
        PD["pix_dim: float (mm)"]
        SHP["shape: (H, W)"]
    end

    QMAP --> QMapDS
    G2 --> G2DS
    META --> MetaDS
```

## I/O Boundary Summary

All I/O boundaries where `ensure_numpy()` conversion occurs:

| Boundary | Direction | Source Module | Target |
|----------|-----------|--------------|--------|
| PyQtGraph plotting | Backend -> NumPy | `module/saxs1d`, `module/saxs2d`, `module/g2mod`, `module/twotime` | `plot.setData()` |
| Matplotlib plotting | Backend -> NumPy | `fitting/visualization` | `plt.plot()` |
| HDF5 write | Backend -> NumPy | `simplemask/area_mask`, `simplemask/simplemask_kernel` | `h5py.create_dataset()` |
| HDF5 read | NumPy -> Backend | `xpcs_file`, `fileIO/hdf_reader` | `backend.from_numpy()` |
| Signal export | Backend -> NumPy | `simplemask/qmap`, `simplemask/utils` | `Signal.emit()` |
