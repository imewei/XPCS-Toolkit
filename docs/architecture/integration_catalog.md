# Integration Points Catalog

**Purpose:** Quick reference guide for developers working on cross-module integrations during JAX migration and beyond.

**Status:** Active (001-jax-migration branch)
**Last Updated:** 2026-01-06

---

## Quick Reference

### When to Use Each Integration Pattern

| Use Case | Pattern | Example |
|----------|---------|---------|
| New HDF5 read/write | **Facade** | `hdf5_facade.read_qmap()` |
| Array conversion for plotting | **Adapter** | `adapter.to_pyqtgraph(array)` |
| Cross-module communication | **Signals** | `mask_exported.emit(mask)` |
| Shared data validation | **Schema** | `validate_qmap(data)` |
| Data access abstraction | **Repository** | `repo.get_g2_data()` |

---

## Integration Point Catalog

### 1. HDF5 I/O Integration Points

#### 1.1 Primary Data Files (XPCS Data)

**Location:** `xpcs_file.py`, `fileIO/hdf_reader.py`

**Schema:** `/xpcs/` hierarchy
```text
/xpcs/
  ├── qmap/mask           # int32, (H, W)
  ├── qmap/sqmap          # float64, (H, W)
  ├── qmap/dqmap          # float64, (H, W)
  ├── g2/                 # Correlation results
  └── metadata/           # Geometry
```

**Current Access Pattern:**
```python
# BEFORE (scattered across modules)
with h5py.File(path, 'r') as f:
    mask = f['/xpcs/qmap/mask'][:]
```

**Recommended Pattern:**
```python
# AFTER (via facade)
from xpcsviewer.io.hdf5_facade import HDF5Facade
facade = HDF5Facade()
qmap = facade.read_qmap(path)  # Returns validated QMapSchema
```

**Migration Status:** 🔴 Not started
**Priority:** HIGH
**Affected Modules:** 8 modules

---

#### 1.2 SimpleMask Mask Files

**Location:** `simplemask/simplemask_kernel.py`, `simplemask/area_mask.py`

**Schema:** SimpleMask HDF5 format
```python
{
    "version": "1.0.0",
    "mask": np.ndarray,           # int32, (H, W)
    "metadata": {
        "bcx": float,
        "bcy": float,
        "det_dist": float,
        ...
    }
}
```

**Current Access Pattern:**
```python
# simplemask_kernel.py:save_mask()
with h5py.File(save_name, 'w') as hf:
    hf.create_dataset('mask', data=mask)
    hf.attrs['version'] = __version__
```

**Recommended Pattern:**
```python
# Use facade with schema validation
mask_schema = MaskSchema(mask=mask, metadata=metadata, version="1.0.0")
facade.write_mask(save_name, mask_schema)
```

**Migration Status:** 🔴 Not started
**Priority:** MEDIUM
**Affected Modules:** 2 modules

---

#### 1.3 SimpleMask Partition Files

**Location:** `simplemask/simplemask_kernel.py`

**Schema:** Partition HDF5 format
```python
{
    "partition_map": np.ndarray,  # int32, (H, W)
    "num_pts": int,
    "val_list": list[float],
    "num_list": list[int],
    "metadata": dict
}
```

**Current Access Pattern:**
```python
# simplemask_kernel.py:save_partition()
with h5py.File(save_fname, 'w') as hf:
    hf.create_dataset('partition_map', data=partition_map)
    hf.attrs['num_pts'] = num_pts
```

**Recommended Pattern:**
```python
partition = PartitionSchema.from_dict(self.new_partition)
facade.write_partition(save_fname, partition)
```

**Migration Status:** 🔴 Not started
**Priority:** LOW (less frequently used)
**Affected Modules:** 1 module

---

#### 1.4 Two-Time Correlation Cache

**Location:** `module/twotime_utils.py`

**Schema:** `/exchange/C2T_all` dataset

**Current Access Pattern:**
```python
# twotime_utils.py:load_data_slicing()
with h5py.File(full_path, 'r') as f:
    data = f[key][:]
```

**Recommended Pattern:**
```python
# Use repository pattern for data access
repo = XPCSRepository(facade)
twotime_data = repo.get_twotime_data(full_path, key, slice_spec)
```

**Migration Status:** 🔴 Not started
**Priority:** MEDIUM
**Affected Modules:** 1 module

---

### 2. Backend Array Conversion Points

#### 2.1 PyQtGraph Plotting Boundary

**Modules:** `module/saxs1d.py`, `module/saxs2d.py`, `module/g2mod.py`, `module/twotime.py`

**Current Pattern:**
```python
from xpcsviewer.backends._conversions import ensure_numpy

# Before plotting
plot_item.setData(ensure_numpy(x), ensure_numpy(y))
```

**Issue:** Scattered across 7+ locations, hard to audit.

**Recommended Pattern:**
```python
from xpcsviewer.backends.io_adapter import PyQtGraphAdapter

adapter = PyQtGraphAdapter(backend)
plot_item.setData(adapter.to_pyqtgraph(x), adapter.to_pyqtgraph(y))
```

**Benefits:**
- Centralized conversion logic
- Easy to add caching for small arrays
- Performance monitoring in one place
- Consistent error handling

**Migration Status:** 🔴 Not started
**Priority:** HIGH
**Affected Modules:** 7 modules

---

#### 2.2 HDF5 Writing Boundary

**Modules:** `simplemask/area_mask.py`, `simplemask/simplemask_kernel.py`, `fileIO/hdf_reader.py`

**Current Pattern:**
```python
from xpcsviewer.backends._conversions import ensure_numpy

# Before HDF5 write
hf.create_dataset('data', data=ensure_numpy(jax_array))
```

**Recommended Pattern:**
```python
from xpcsviewer.backends.io_adapter import HDF5Adapter

adapter = HDF5Adapter(backend)
hf.create_dataset('data', data=adapter.to_hdf5(array))
```

**Migration Status:** 🔴 Not started
**Priority:** MEDIUM
**Affected Modules:** 3 modules

---

#### 2.3 Matplotlib Plotting Boundary

**Modules:** `fitting/visualization.py`

**Current Pattern:**
```python
import matplotlib.pyplot as plt
from xpcsviewer.backends._conversions import ensure_numpy

plt.plot(ensure_numpy(x), ensure_numpy(y))
```

**Recommended Pattern:**
```python
from xpcsviewer.backends.io_adapter import MatplotlibAdapter

adapter = MatplotlibAdapter(backend)
plt.plot(*adapter.to_matplotlib(x, y))  # Unpacks to (x_np, y_np)
```

**Migration Status:** 🔴 Not started
**Priority:** LOW
**Affected Modules:** 1 module

---

### 3. Signal-Based Integration Points

#### 3.1 SimpleMask → XPCS Viewer Communication

**Location:** `simplemask/simplemask_window.py` → `xpcs_viewer.py`

**Signals:**
```python
class SimpleMaskWindow(QMainWindow):
    mask_exported = Signal(np.ndarray)      # Mask array
    qmap_exported = Signal(dict)            # Partition dict
```

**Data Flow:**
```
SimpleMaskWindow.export_to_viewer()
    → mask_exported.emit(mask_array)
    → XPCS Viewer.apply_mask(mask_array)
```

**Schema Contract:**
```python
# mask_exported payload
mask: np.ndarray[int32]  # Shape=(H, W), values 0 or 1

# qmap_exported payload
{
    "partition_map": np.ndarray[int32],
    "num_pts": int,
    "val_list": list[float],
    "num_list": list[int]
}
```

**Current Status:** ✅ Well-designed, loose coupling
**Recommendation:** Keep this pattern, use for other integrations

**Migration Status:** ✅ Complete (already uses signals)
**Priority:** N/A (reference implementation)

---

#### 3.2 Recommended Signal Pattern for Future Integrations

**Example:** If adding a new analysis module that exports results to main viewer:

```python
# In new module
class NewAnalysisModule(QWidget):
    results_ready = Signal(dict)  # Emit structured results

    def compute_analysis(self):
        result = self._compute()
        # Validate before emitting
        validated = validate_analysis_result(result)
        self.results_ready.emit(validated)

# In main viewer
def connect_new_module(self):
    self.new_module.results_ready.connect(self.apply_analysis_results)

def apply_analysis_results(self, result: dict):
    # Type-safe handling
    schema = AnalysisResultSchema(**result)
    self.update_plot(schema.data)
```

---

### 4. Shared Data Schema Integration Points

#### 4.1 QMapDict Contract

**Producers:**
- `simplemask/qmap.py` → `compute_qmap()`
- `fileIO/qmap_utils.py` → `QMap.get_qmap()`

**Consumers:**
- `xpcs_file.py` → `get_cropped_qmap()`
- `viewer_kernel.py` → `plot_qmap()`
- `module/saxs1d.py` → `plot()`
- `module/saxs2d.py` → `plot()`

**Schema:**
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class QMapSchema:
    sqmap: np.ndarray      # float64, shape=(H, W)
    dqmap: np.ndarray      # float64, shape=(H, W)
    phis: np.ndarray       # float64, shape=(H, W)
    sqmap_unit: str        # "nm^-1" or "A^-1"
    dqmap_unit: str
    phis_unit: str         # "rad" or "deg"

    def __post_init__(self):
        assert self.sqmap.shape == self.dqmap.shape == self.phis.shape
        assert self.sqmap_unit in ["nm^-1", "A^-1"]
        # ... more validation
```

**Current Pattern (UNSAFE):**
```python
# Producer (simplemask/qmap.py)
qmap_dict = {
    "sqmap": sqmap,
    "dqmap": dqmap,
    # ... (typos possible)
}

# Consumer (viewer_kernel.py)
sqmap = qmap_dict["sqmap"]  # KeyError if typo
```

**Recommended Pattern (SAFE):**
```python
# Producer
from xpcsviewer.schemas.validators import QMapSchema
qmap = QMapSchema(sqmap=sqmap, dqmap=dqmap, ...)  # Validated

# Consumer
sqmap = qmap.sqmap  # Type-safe attribute access
```

**Migration Status:** 🔴 Not started
**Priority:** HIGH (prevents silent failures)
**Affected Modules:** 6 modules

---

#### 4.2 GeometryMetadata Contract

**Producers:**
- `fileIO/hdf_reader.py` → loads from HDF5 `/xpcs/metadata`
- `simplemask/simplemask_window.py` → user input via GUI

**Consumers:**
- `simplemask/qmap.py` → `compute_qmap(metadata)`
- `xpcs_file.py` → `self.metadata`
- `viewer_kernel.py` → geometry calculations

**Schema:**
```python
from dataclasses import dataclass

@dataclass
class GeometryMetadata:
    bcx: float           # Beam center X (column, pixels)
    bcy: float           # Beam center Y (row, pixels)
    det_dist: float      # Detector distance (mm)
    lambda_: float       # Wavelength (Å)
    pix_dim: float       # Pixel size (mm)
    shape: tuple[int, int]  # (height, width)

    def __post_init__(self):
        assert self.det_dist > 0, "Detector distance must be positive"
        assert self.lambda_ > 0, "Wavelength must be positive"
        # ... more validation
```

**Migration Status:** 🔴 Not started
**Priority:** MEDIUM
**Affected Modules:** 4 modules

---

#### 4.3 G2Data Contract

**Producers:**
- `xpcs_file.py` → `get_g2_data(q_idx)`

**Consumers:**
- `module/g2mod.py` → plotting
- `fitting/sampler.py` → Bayesian fitting

**Schema:**
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class G2Data:
    g2: np.ndarray          # shape=(n_q, n_delay)
    g2_err: np.ndarray      # shape=(n_q, n_delay)
    delay_times: np.ndarray # shape=(n_delay,)
    q_values: list[float]   # length=n_q

    def __post_init__(self):
        assert self.g2.shape == self.g2_err.shape
        assert self.g2.shape[1] == len(self.delay_times)
        assert self.g2.shape[0] == len(self.q_values)
```

**Migration Status:** 🔴 Not started
**Priority:** MEDIUM
**Affected Modules:** 3 modules

---

### 5. Module Lazy Loading Integration

**Location:** `viewer_kernel.py`

**Current Pattern:**
```python
# viewer_kernel.py:_get_module()
def _get_module(module_name: str) -> ModuleType:
    if module_name not in _module_cache:
        if module_name == "g2mod":
            from .module import g2mod
            _module_cache[module_name] = g2mod
    return _module_cache[module_name]
```

**Purpose:** Improve startup time by deferring imports

**Status:** ✅ Working well
**Recommendation:** Keep this pattern

**Adding a New Module:**
```python
# 1. Add to _get_module() mapping
elif module_name == "new_analysis":
    from .module import new_analysis
    _module_cache[module_name] = new_analysis

# 2. Add wrapper method in ViewerKernel
def plot_new_analysis(self, **kwargs):
    module = self._get_module("new_analysis")
    return module.plot(**kwargs)
```

---

## Migration Checklist

### Phase 1: Create New Infrastructure (No Breaking Changes)

- [ ] Create `xpcsviewer/io/hdf5_facade.py`
  - [ ] `read_qmap()` with schema validation
  - [ ] `write_mask()` with versioning
  - [ ] `write_partition()` with compression
  - [ ] Unit tests (100% coverage)

- [ ] Create `xpcsviewer/schemas/validators.py`
  - [ ] `QMapSchema` dataclass
  - [ ] `GeometryMetadata` dataclass
  - [ ] `G2Data` dataclass
  - [ ] `PartitionSchema` dataclass
  - [ ] Unit tests for validation logic

- [ ] Create `xpcsviewer/backends/io_adapter.py`
  - [ ] `PyQtGraphAdapter`
  - [ ] `HDF5Adapter`
  - [ ] `MatplotlibAdapter`
  - [ ] Unit tests for conversions

- [ ] Create `xpcsviewer/repositories/xpcs_repository.py`
  - [ ] `get_qmap()`
  - [ ] `get_g2_data()`
  - [ ] `save_fit_result()`
  - [ ] Integration tests with facade

### Phase 2: Migrate Modules (One at a Time)

**Priority Order:**

1. [ ] **simplemask/simplemask_kernel.py** (self-contained, low risk)
   - [ ] Use `HDF5Facade` for mask I/O
   - [ ] Use `HDF5Adapter` for array conversions
   - [ ] Validate partition schema before export
   - [ ] Integration tests
   - [ ] Monitor for 2 weeks

2. [ ] **module/saxs1d.py** (simple, good reference)
   - [ ] Use `PyQtGraphAdapter` for plotting
   - [ ] Use `QMapSchema` for qmap handling
   - [ ] Unit tests
   - [ ] Monitor for 1 week

3. [ ] **module/saxs2d.py** (similar to saxs1d)
   - [ ] Same pattern as saxs1d
   - [ ] Integration tests
   - [ ] Monitor for 1 week

4. [ ] **module/twotime.py** (high complexity)
   - [ ] Use `XPCSRepository` for cache access
   - [ ] Use `PyQtGraphAdapter` for plotting
   - [ ] Use `HDF5Adapter` for cache writes
   - [ ] Extensive integration tests
   - [ ] Monitor for 2 weeks

5. [ ] **xpcs_file.py** (core module, migrate last)
   - [ ] Extract analysis logic to service layer
   - [ ] Use `XPCSRepository` for all HDF5 access
   - [ ] Keep thin compatibility wrappers
   - [ ] Extensive integration tests
   - [ ] Monitor for 1 month

### Phase 3: Cleanup

- [ ] Remove deprecated direct `h5py.File` usage
- [ ] Remove scattered `ensure_numpy()` calls
- [ ] Add linting rules to enforce patterns
- [ ] Update documentation
- [ ] Final performance benchmarks

---

## Testing Strategy for Integration Points

### Unit Tests (New Code)

```python
# tests/unit/io/test_hdf5_facade.py
def test_read_qmap_validates_schema():
    facade = HDF5Facade()
    with pytest.raises(SchemaValidationError):
        facade.read_qmap("invalid_qmap.h5")

# tests/unit/backends/test_io_adapter.py
def test_pyqtgraph_adapter_converts_jax_array():
    backend = get_backend()  # JAX
    adapter = PyQtGraphAdapter(backend)
    jax_array = backend.array([1, 2, 3])
    np_array = adapter.to_pyqtgraph(jax_array)
    assert isinstance(np_array, np.ndarray)
```

### Integration Tests (Cross-Module)

```python
# tests/integration/test_simplemask_integration.py
def test_simplemask_exports_valid_partition():
    """Ensure SimpleMask → XPCS Viewer integration works."""
    window = SimpleMaskWindow()
    window.load_detector_image(test_image)
    window.compute_partition()

    # Capture signal
    received = []
    window.qmap_exported.connect(lambda x: received.append(x))
    window.export_to_viewer()

    # Validate schema
    partition = PartitionSchema.from_dict(received[0])
    assert partition.num_pts > 0
    assert len(partition.val_list) == partition.num_pts
```

### Backward Compatibility Tests

```python
# tests/integration/test_backward_compatibility.py
def test_old_hdf5_files_still_load():
    """Ensure facade can read old HDF5 files without schema version."""
    facade = HDF5Facade()
    qmap = facade.read_qmap("tests/data/legacy_v1.hdf5")
    assert qmap.sqmap_unit in ["nm^-1", "A^-1"]  # Default inferred
```

---

## Performance Monitoring

### Key Metrics to Track During Migration

1. **HDF5 I/O Performance:**
   - Connection pool hit rate (target: >95%)
   - Average read latency (target: <10ms for cached)
   - Memory usage per connection (target: <50MB)

2. **Backend Conversion Performance:**
   - `ensure_numpy()` overhead (baseline: ~0.5ms/1024x1024 array)
   - Adapter overhead (target: <5% vs. direct conversion)

3. **Schema Validation Performance:**
   - Validation time per schema (target: <1ms)
   - Memory overhead per validated object (target: <1KB)

### Instrumentation Example

```python
# xpcsviewer/io/hdf5_facade.py
import time
from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)

class HDF5Facade:
    def read_qmap(self, file_path: str) -> QMapSchema:
        start = time.perf_counter()
        try:
            # ... read logic
            qmap = QMapSchema(**data)
            elapsed = time.perf_counter() - start
            logger.debug(f"read_qmap() took {elapsed*1000:.2f}ms")
            return qmap
        except Exception as e:
            logger.error(f"read_qmap() failed: {e}")
            raise
```

---

## Code Review Checklist for Integration Changes

When reviewing PRs that modify integration points:

### HDF5 I/O Changes
- [ ] Uses `HDF5Facade` instead of direct `h5py.File`?
- [ ] Includes schema validation?
- [ ] Handles file versioning?
- [ ] Includes backward compatibility test?
- [ ] Connection properly released (no leaks)?

### Backend Array Conversions
- [ ] Uses adapter pattern instead of direct `ensure_numpy()`?
- [ ] Conversion happens at I/O boundary (not earlier)?
- [ ] No unnecessary conversions in hot loops?
- [ ] Includes performance benchmark?

### Shared Data Schemas
- [ ] Uses dataclass with validation, not plain dict?
- [ ] Includes type hints?
- [ ] Validation logic tested?
- [ ] Schema version documented?

### Signal-Based Integration
- [ ] Signal payload is documented?
- [ ] Payload uses validated schema?
- [ ] Signal connection tested in integration test?
- [ ] Error handling for disconnected signals?

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting I/O Boundary Conversion

**Problem:**
```python
# BAD: Passing JAX array directly to PyQtGraph
plot_item.setData(jax_array, jax_array)  # Crashes!
```

**Solution:**
```python
# GOOD: Use adapter at I/O boundary
adapter = PyQtGraphAdapter(backend)
plot_item.setData(adapter.to_pyqtgraph(x), adapter.to_pyqtgraph(y))
```

### Pitfall 2: Schema Validation Too Late

**Problem:**
```python
# BAD: Validation after use
qmap = get_qmap_from_hdf5(path)
plot(qmap["sqmap"])  # Might crash here if key missing
validate_qmap(qmap)   # Too late!
```

**Solution:**
```python
# GOOD: Validate at boundary
qmap = facade.read_qmap(path)  # Validated during read
plot(qmap.sqmap)  # Type-safe, cannot fail
```

### Pitfall 3: Direct HDF5 Access Bypassing Pool

**Problem:**
```python
# BAD: Bypasses connection pool
with h5py.File(path, 'r') as f:
    data = f['/xpcs/g2'][:]
```

**Solution:**
```python
# GOOD: Use facade with pooling
data = facade.read_g2_data(path)
```

### Pitfall 4: Circular Import from Facade

**Problem:**
```python
# In backends/__init__.py
from xpcsviewer.io.hdf5_facade import HDF5Facade  # Circular import!
```

**Solution:**
```python
# Use TYPE_CHECKING guard
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from xpcsviewer.io.hdf5_facade import HDF5Facade
```

---

## Contact and Maintenance

**Document Owner:** Architecture Team
**Review Frequency:** Monthly during migration, quarterly after
**Update Process:** Update after each module migration completes

**Key Stakeholders:**
- Backend migration team (JAX integration)
- SimpleMask integration team
- Data I/O team (HDF5 access patterns)
- Testing team (integration test coverage)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-06
**Next Review:** 2026-02-06
