# ADR-003: HDF5 Facade Pattern with Connection Pooling

## Status

Accepted

## Context

HDF5 file I/O in XPCS Viewer was historically scattered across 12+ modules, each opening files independently with `h5py.File`. This created several problems:

1. **No connection reuse**: Each module opened and closed HDF5 files independently. For interactive analysis workflows where multiple plots read from the same file, this meant repeated open/close cycles.
2. **Inconsistent error handling**: Some modules raised exceptions on missing datasets, others returned `None`, and some silently returned empty arrays.
3. **No schema validation**: HDF5 datasets were read as raw NumPy arrays without checking shapes, dtypes, NaN values, or physical constraints (e.g., non-negative delay times).
4. **Implicit data contracts**: The structure of HDF5 groups and datasets was documented in comments but never enforced at runtime. Typos in dataset paths caused `KeyError` at unpredictable points.
5. **No versioning**: Schema changes to HDF5 file format had no migration path. Old files could silently produce wrong results with new code.

The codebase already had a connection pool (`fileIO/hdf_reader.py:HDF5ConnectionPool`) for basic connection reuse, but it was used directly by only a few modules.

## Decision

We introduced a **facade pattern** with two complementary layers:

1. **Schema validators** (`xpcsviewer/schemas/validators.py`): Frozen dataclasses with `__post_init__` validation for all shared data structures.
2. **HDF5 Facade** (`xpcsviewer/io/hdf5_facade.py`): A unified entry point for all HDF5 operations that combines connection pooling with schema validation.

### Architecture

```
xpcsviewer/schemas/
  validators.py    # QMapSchema, GeometryMetadata, G2Data, PartitionSchema, MaskSchema
  __init__.py      # Public re-exports

xpcsviewer/io/
  hdf5_facade.py   # HDF5Facade: read/write with validation + pooling
  __init__.py      # Public re-exports
```

### Schema Design

All schemas are **frozen dataclasses** (`@dataclass(frozen=True)`) to enforce immutability after construction. Each schema validates in `__post_init__`:

| Schema | Validates | Fields |
|--------|-----------|--------|
| `QMapSchema` | Shape consistency, float64 dtype, no NaN, valid units, mask values 0/1 | sqmap, dqmap, phis, units, mask, partition_map |
| `GeometryMetadata` | Positive det_dist/lambda_/pix_dim, 2-tuple shape, beam center bounds | bcx, bcy, det_dist, lambda_, pix_dim, shape |
| `G2Data` | Shape consistency, float64 dtype, no NaN in g2/delay_times, non-negative errors, monotonic delay_times | g2, g2_err, delay_times, q_values |
| `PartitionSchema` | Positive num_pts, integer partition_map, matching list lengths, non-negative num_list | partition_map, num_pts, val_list, num_list, metadata |
| `MaskSchema` | 2D integer array, values 0/1, shape matches metadata | mask, metadata, version |

Key validation patterns:
- **Defensive copies** on construction: `object.__setattr__(self, "sqmap", np.copy(self.sqmap))` prevents external mutation of frozen dataclass arrays.
- **Immutable collections** (BUG-010): Mutable lists inside frozen dataclasses are converted to tuples: `object.__setattr__(self, "q_values", tuple(self.q_values))`.
- **dtype coercion** in `from_dict()` (BUG-011, BUG-058): Float32 HDF5 data is coerced to float64 via `np.asarray(data, dtype=np.float64)`.
- **NaN/Inf rejection** (BUG-048): `GeometryMetadata.from_dict()` explicitly checks for NaN and infinite values in critical fields.

### Facade Design

`HDF5Facade` provides methods for each data type with consistent patterns:

```python
class HDF5Facade:
    def __init__(self, pool=None, validate=True):
        self.pool = pool or _connection_pool  # Global connection pool
        self.validate = validate

    def read_qmap(self, file_path, group="/xpcs/qmap") -> QMapSchema: ...
    def write_mask(self, file_path, mask_schema, group, compression) -> None: ...
    def write_partition(self, file_path, partition_schema, group) -> None: ...
    def read_g2_data(self, file_path, q_idx=None, group="/xpcs/g2") -> G2Data: ...
    def read_geometry_metadata(self, file_path, group="/xpcs/metadata") -> GeometryMetadata: ...
    def get_pool_stats(self) -> dict: ...
    def clear_pool(self) -> None: ...
```

Each read method:
1. Opens the file via the connection pool (`self.pool.get_connection(file_path, "r")`).
2. Reads raw datasets from the HDF5 group.
3. Handles backward compatibility (missing optional datasets, bytes vs. string attributes).
4. Constructs and returns a validated schema object.
5. Wraps validation errors in `HDF5ValidationError` for consistent error handling.

The `validate=False` option (BUG-029) returns raw dictionaries instead of schema objects, bypassing `__post_init__` validation for performance-critical paths.

All read/write methods are decorated with `@log_timing(threshold_ms=...)` for automatic performance monitoring.

### Connection Pooling

The facade delegates connection management to the existing `HDF5ConnectionPool` from `fileIO/hdf_reader.py`. The pool:
- Caches open file handles keyed by `(file_path, mode)`.
- Provides context manager access via `pool.get_connection(path, mode)`.
- Tracks cache hit statistics via `pool.get_pool_stats()`.
- Can be cleared via `pool.clear_pool()` for application shutdown.

## Consequences

### What became easier

- **Type-safe access**: `qmap.sqmap` instead of `qmap_dict["sqmap"]` -- IDE autocomplete, no `KeyError` risk.
- **Fail-fast validation**: Shape mismatches, NaN values, and invalid units are caught at the I/O boundary, not deep in analysis code.
- **Consistent error handling**: All HDF5 errors are wrapped in `HDF5ValidationError`, making error handling uniform across the codebase.
- **Backward compatibility**: `from_dict()` and `to_dict()` methods allow gradual migration from legacy dict-passing patterns.
- **Monitoring**: `get_pool_stats()` exposes cache hit ratios and connection counts for production diagnostics.
- **Versioning**: `MaskSchema.version` and `PartitionSchema.version` fields enable future schema migration.

### What became more difficult

- **Validation overhead**: Schema validation adds ~1ms per construction. For high-frequency reads in tight loops, `validate=False` is available.
- **Frozen dataclass limitations**: In-place mutation of arrays is not possible. Operations that modify data must create new schema instances.
- **Migration effort**: Existing code that passes raw dicts must be updated to use schemas. The `from_dict()`/`to_dict()` bridge eases this transition.
- **Unit consistency** (BUG-028): The default unit in `QMapSchema.from_dict()` was changed from `"A^-1"` to `"nm^-1"` to match `hdf5_facade.py`. Legacy files with implicit units may need attention.
