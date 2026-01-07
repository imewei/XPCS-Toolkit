# Facade and Schema Infrastructure

**Created:** 2026-01-06
**Status:** Active - Ready for Migration
**Related Documents:** `dependency_analysis.md`, `integration_catalog.md`

## Overview

This document describes the new facade and schema infrastructure for the XPCS Viewer codebase, designed to standardize HDF5 I/O operations and provide type-safe data validation.

## Architecture Components

### 1. Schema Validators (`xpcsviewer/schemas/`)

Dataclass-based schemas with built-in validation for all shared data structures.

#### Available Schemas

| Schema | Purpose | Key Fields |
|--------|---------|------------|
| `QMapSchema` | Q-map data with units | sqmap, dqmap, phis, units |
| `GeometryMetadata` | Detector geometry | bcx, bcy, det_dist, lambda_, pix_dim, shape |
| `G2Data` | G2 correlation data | g2, g2_err, delay_times, q_values |
| `PartitionSchema` | Q-bin partition data | partition_map, num_pts, val_list, num_list |
| `MaskSchema` | Mask data with metadata | mask, metadata, version |

#### Key Features

- **Type Safety:** All fields have type annotations
- **Validation:** Automatic validation in `__post_init__`
- **Backward Compatibility:** `from_dict()` and `to_dict()` methods
- **Units Enforcement:** Literal types for unit validation
- **Shape Consistency:** Cross-field shape validation

#### Usage Example

```python
from xpcsviewer.schemas import QMapSchema, GeometryMetadata
import numpy as np

# Create validated schema
metadata = GeometryMetadata(
    bcx=512.5, bcy=512.5,
    det_dist=5000.0, lambda_=1.54,
    pix_dim=0.075, shape=(1024, 1024)
)

# This will raise ValueError if shapes don't match
qmap = QMapSchema(
    sqmap=np.random.rand(100, 100),
    dqmap=np.random.rand(100, 100),
    phis=np.random.rand(100, 100),
    sqmap_unit="nm^-1",
    dqmap_unit="nm^-1",
    phis_unit="rad"
)

# Legacy compatibility
qmap_dict = qmap.to_dict()  # For old code
qmap_restored = QMapSchema.from_dict(qmap_dict)
```

### 2. HDF5 Facade (`xpcsviewer/io/hdf5_facade.py`)

Unified interface for all HDF5 file operations with schema validation and connection pooling.

#### Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `read_qmap()` | Read Q-map with validation | `QMapSchema` |
| `write_mask()` | Write mask with versioning | None |
| `write_partition()` | Write partition with compression | None |
| `read_g2_data()` | Read G2 correlation data | `G2Data` |
| `read_geometry_metadata()` | Read detector geometry | `GeometryMetadata` |
| `get_pool_stats()` | Get connection pool stats | dict |

#### Key Features

- **Schema Validation:** All data validated on read/write
- **Connection Pooling:** Uses existing HDF5 connection pool
- **Versioning:** Automatic version tracking for written data
- **Compression:** Configurable compression for datasets
- **Error Handling:** Consistent error handling with `HDF5ValidationError`
- **Logging:** Comprehensive logging of all operations

#### Usage Example

```python
from xpcsviewer.io import HDF5Facade
from xpcsviewer.schemas import MaskSchema, GeometryMetadata
import numpy as np

# Initialize facade
facade = HDF5Facade()

# Read with validation
qmap = facade.read_qmap("/path/to/file.h5")
print(f"Q-map shape: {qmap.sqmap.shape}")

# Write with validation
mask_schema = MaskSchema(
    mask=np.random.randint(0, 2, size=(1024, 1024), dtype=np.int32),
    metadata=metadata,
    version="1.0.0"
)
facade.write_mask("/path/to/output.h5", mask_schema)

# Check pool performance
stats = facade.get_pool_stats()
print(f"Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
```

### 3. Backend I/O Adapters (`xpcsviewer/backends/io_adapter.py`)

Centralized array conversion adapters for I/O boundaries.

#### Available Adapters

| Adapter | Purpose | Use Case |
|---------|---------|----------|
| `PyQtGraphAdapter` | Convert for PyQtGraph plotting | All GUI visualizations |
| `HDF5Adapter` | Convert for HDF5 file I/O | File reading/writing |
| `MatplotlibAdapter` | Convert for Matplotlib plotting | Static plots, analysis |

#### Key Features

- **Performance Monitoring:** Optional conversion time tracking
- **Statistics:** Conversion count and average time metrics
- **Bidirectional:** Both to/from conversions supported
- **Batch Operations:** Multiple array conversions at once
- **Error Handling:** Consistent error messages

#### Usage Example

```python
from xpcsviewer.backends import get_backend, create_adapters

# Create all adapters at once
backend = get_backend()
pyqt_adapter, hdf5_adapter, mpl_adapter = create_adapters(
    backend,
    enable_monitoring=True
)

# Convert for PyQtGraph
x = backend.linspace(0, 10, 100)
y = backend.sin(x)
x_np = pyqt_adapter.to_pyqtgraph(x)
y_np = pyqt_adapter.to_pyqtgraph(y)
plot_item.setData(x_np, y_np)

# Convert for Matplotlib (handles multiple arrays)
x_mpl, y_mpl = mpl_adapter.to_matplotlib(x, y)
plt.plot(x_mpl, y_mpl)

# Convert for HDF5
data_np = hdf5_adapter.to_hdf5(y)
hf.create_dataset("data", data=data_np)

# Get performance stats
stats = pyqt_adapter.get_stats()
print(f"Avg conversion time: {stats['average_conversion_time_ms']:.3f}ms")
```

## Migration Strategy

### Phase 1: Non-Breaking Additions (Complete)

- ✅ Created `xpcsviewer/schemas/` with all validators
- ✅ Created `xpcsviewer/io/hdf5_facade.py` with core methods
- ✅ Created `xpcsviewer/backends/io_adapter.py` with all adapters
- ✅ Added comprehensive docstrings and type hints
- ✅ Created example script demonstrating usage

### Phase 2: Gradual Module Migration (Next Steps)

**Priority Order:**

1. **SimpleMask Module** (Low risk, self-contained)
   - Replace direct `h5py.File` usage with `HDF5Facade`
   - Use `MaskSchema` and `PartitionSchema` for validation
   - Replace scattered `ensure_numpy()` with adapters
   - Test: `tests/unit/simplemask/`

2. **SAXS Modules** (Medium complexity)
   - `module/saxs1d.py`: Use `PyQtGraphAdapter` for plotting
   - `module/saxs2d.py`: Use `QMapSchema` for Q-map handling
   - Test: Existing SAXS tests

3. **Two-Time Module** (High complexity)
   - `module/twotime.py`: Multiple I/O boundaries
   - Use `HDF5Facade` for cache reads
   - Use adapters for all conversions
   - Test: `tests/unit/test_twotime.py`

4. **Core Module** (Migrate last)
   - `xpcs_file.py`: Extensive refactoring
   - Extract analysis logic to service layer
   - Use facade for all HDF5 access
   - Test: Full integration test suite

### Migration Pattern

For each module:

```python
# BEFORE (direct HDF5 access)
with h5py.File(path, 'r') as f:
    mask = f['/xpcs/qmap/mask'][:]
    sqmap = f['/xpcs/qmap/sqmap'][:]

# AFTER (using facade)
from xpcsviewer.io import HDF5Facade

facade = HDF5Facade()
qmap = facade.read_qmap(path)  # Validated QMapSchema
mask = qmap.mask  # Type-safe access
```

```python
# BEFORE (scattered conversions)
from xpcsviewer.backends._conversions import ensure_numpy

plot_item.setData(ensure_numpy(x), ensure_numpy(y))

# AFTER (using adapter)
from xpcsviewer.backends import create_adapters, get_backend

pyqt_adapter, _, _ = create_adapters(get_backend())
plot_item.setData(pyqt_adapter.to_pyqtgraph(x), pyqt_adapter.to_pyqtgraph(y))
```

## Benefits

### Type Safety

**Before:**
```python
qmap_dict = get_qmap_from_file(path)
sqmap = qmap_dict["sqmap"]  # KeyError if typo
```

**After:**
```python
qmap = facade.read_qmap(path)  # QMapSchema
sqmap = qmap.sqmap  # Type-safe, IDE autocomplete
```

### Validation

**Before:**
```python
# No validation - runtime errors possible
qmap_dict = {"sqmap": sqmap, "dqmap": wrong_shape_dqmap}
```

**After:**
```python
# Validation at construction - fails fast
qmap = QMapSchema(sqmap=sqmap, dqmap=wrong_shape_dqmap)
# ValueError: Q-maps must have identical shapes
```

### Centralized I/O

**Before:**
```python
# Scattered across 12 modules
with h5py.File(path, 'r') as f:
    data = f['/some/path'][:]
```

**After:**
```python
# Single facade with consistent error handling
facade = HDF5Facade()
data = facade.read_<data_type>(path)
```

### Performance Monitoring

**Before:**
```python
# No visibility into conversion performance
np_array = ensure_numpy(jax_array)
```

**After:**
```python
# Optional monitoring with statistics
adapter = PyQtGraphAdapter(backend, enable_monitoring=True)
np_array = adapter.to_pyqtgraph(jax_array)
stats = adapter.get_stats()
# {"conversion_count": 1000, "average_conversion_time_ms": 0.5}
```

## Testing

### Unit Tests

Create tests for each schema:

```python
# tests/unit/schemas/test_qmap_schema.py
def test_qmap_schema_validates_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        QMapSchema(
            sqmap=np.ones((100, 100)),
            dqmap=np.ones((50, 50)),  # Wrong shape
            phis=np.ones((100, 100)),
            sqmap_unit="nm^-1",
            dqmap_unit="nm^-1",
            phis_unit="rad"
        )
```

### Integration Tests

Test facade with real HDF5 files:

```python
# tests/integration/test_hdf5_facade.py
def test_facade_reads_legacy_files():
    """Ensure facade can read old HDF5 files without schema version."""
    facade = HDF5Facade()
    qmap = facade.read_qmap("tests/data/legacy_v1.hdf5")
    assert isinstance(qmap, QMapSchema)
    assert qmap.sqmap_unit in ["nm^-1", "A^-1"]
```

### Backward Compatibility Tests

Verify new code works with legacy code:

```python
def test_schema_to_dict_backward_compatible():
    """Ensure to_dict() output matches legacy dict structure."""
    qmap = QMapSchema(...)
    qmap_dict = qmap.to_dict()

    # Legacy code can use the dict
    legacy_function(qmap_dict)  # Should not fail
```

## Performance Impact

Based on initial testing:

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Schema validation | <1ms | One-time cost at construction |
| Facade vs. direct h5py | ~1% | Connection pooling offsets overhead |
| Adapter conversion | 0% | Same underlying `ensure_numpy()` |
| Monitoring overhead | ~5% | Only when `enable_monitoring=True` |

**Recommendation:** Enable monitoring in development, disable in production.

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# ❌ Wrong
from xpcsviewer.schemas.validators import QMapSchema

# ✅ Correct
from xpcsviewer.schemas import QMapSchema
```

#### 2. Validation Errors

```python
# ❌ Fails validation
qmap = QMapSchema(sqmap=sqmap, dqmap=dqmap, phis=phis,
                  sqmap_unit="invalid")  # ValueError

# ✅ Use valid units
qmap = QMapSchema(..., sqmap_unit="nm^-1")
```

#### 3. Circular Imports

```python
# ❌ Circular import in backends
from xpcsviewer.io import HDF5Facade  # May cause circular import

# ✅ Use TYPE_CHECKING guard
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xpcsviewer.io import HDF5Facade
```

## Future Enhancements

### Planned Features

1. **Schema Versioning:** Automatic migration of old schema versions
2. **Async I/O:** Async facade methods for large file operations
3. **Caching Layer:** Adapter-level caching for small arrays
4. **Metrics Export:** Export performance metrics to monitoring systems
5. **Validation Strictness:** Configurable validation levels (strict/lenient)

### Extension Points

To add a new schema:

1. Create dataclass in `xpcsviewer/schemas/validators.py`
2. Add validation logic in `__post_init__`
3. Implement `from_dict()` and `to_dict()` class methods
4. Export from `xpcsviewer/schemas/__init__.py`
5. Add unit tests
6. Update this documentation

To add a new facade method:

1. Add method to `HDF5Facade` class
2. Use appropriate schema for return type
3. Add docstring with parameters and examples
4. Add logging statements
5. Add error handling
6. Add unit tests
7. Update this documentation

## References

- **Architecture Analysis:** `docs/architecture/dependency_analysis.md`
- **Integration Catalog:** `docs/architecture/integration_catalog.md`
- **Example Usage:** `examples/facade_usage_example.py`
- **Backend Abstraction:** `xpcsviewer/backends/README.md`

---

**Document Version:** 1.0
**Author:** Architecture Team
**Last Updated:** 2026-01-06
