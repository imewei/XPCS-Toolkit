# ADR-004: Protocol-Based Backend Abstraction Layer

## Status

Accepted

## Context

When introducing JAX as an optional computational backend (see [ADR-001](ADR-001-jax-migration.md)), we needed an abstraction layer that:

1. **Does not require JAX at import time**: Users without JAX should be able to import and use all non-JAX features.
2. **Provides identical API for both backends**: Analysis code should not contain `if backend == "jax"` conditionals.
3. **Supports capabilities introspection**: Code that requires gradients or JIT must be able to check backend capabilities at runtime.
4. **Handles I/O boundary conversions**: External libraries (h5py, PyQtGraph, Matplotlib) require NumPy arrays; conversions must be centralized.
5. **Manages device placement**: GPU-enabled workflows need explicit control over which device holds the data.

We considered three approaches:
- **Abstract base class**: Would require JAX imports at class definition time (even for the base class docstrings or type hints).
- **Duck typing**: No enforced contract; easy to drift between backends.
- **Protocol (structural subtyping)**: Defines the contract without requiring inheritance; works with `isinstance()` checks via `@runtime_checkable`.

## Decision

We chose a **`typing.Protocol`-based design** with the following components:

### BackendProtocol (`_base.py`)

A `@runtime_checkable` Protocol defining 60+ abstract methods across 8 categories:

| Category | Methods | Purpose |
|----------|---------|---------|
| Properties | `name`, `supports_gpu`, `supports_jit`, `supports_grad`, `pi` | Capability introspection |
| Array Creation | `zeros`, `ones`, `arange`, `linspace`, `logspace`, `meshgrid`, `full`, `array` | Factory methods |
| Trigonometry | `sin`, `cos`, `arctan`, `arctan2`, `hypot`, `deg2rad`, `rad2deg` | Angle computations for Q-map |
| Statistics | `mean`, `std`, `nanmean`, `nanmin`, `nanmax`, `percentile`, `sum`, `min`, `max` | Data analysis |
| Binning | `digitize`, `bincount`, `unique` | Q-bin partition |
| Boolean/Masking | `logical_and`, `logical_or`, `logical_not`, `where`, `nonzero`, `isnan`, `isfinite` | Mask operations |
| Math | `exp`, `log`, `log10`, `sqrt`, `abs`, `power` | Model evaluation |
| Functional | `jit`, `grad`, `value_and_grad`, `vmap`, `scan`, `fori_loop` | JAX transformations |

### NumPyBackend (`_numpy_backend.py`)

- Implements all `BackendProtocol` methods using NumPy.
- `supports_gpu = False`, `supports_jit = False`, `supports_grad = False`.
- `jit()` returns the function unchanged (no-op).
- `grad()` and `value_and_grad()` raise `NotImplementedError`.
- `vmap()` uses a simple Python loop fallback.
- `scan()` uses a Python `for` loop.
- `fori_loop()` uses `range()`.

### JAXBackend (`_jax_backend.py`)

- Implements all `BackendProtocol` methods using `jax.numpy`.
- `supports_gpu = True` (if GPU device available), `supports_jit = True`, `supports_grad = True`.
- `jit()` delegates to `jax.jit` with optional `static_argnums`.
- `grad()` and `value_and_grad()` delegate to JAX's automatic differentiation.
- `vmap()` delegates to `jax.vmap`.
- `scan()` delegates to `jax.lax.scan`.
- `fori_loop()` delegates to `jax.lax.fori_loop`.
- JAX is imported lazily via module-level variables (`_jax`, `_jnp`) and an `_ensure_jax()` function, avoiding import-time failures.

### DeviceManager (`_device.py`)

A **thread-safe singleton** for device lifecycle management:

```python
class DeviceManager:
    _instance: DeviceManager | None = None
    _lock = threading.RLock()

    def __new__(cls) -> DeviceManager:
        # Double-checked locking pattern
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
```

Key features:
- **`DeviceConfig` dataclass**: Configurable via `DeviceConfig.from_environment()`, reads `XPCS_USE_GPU`, `XPCS_GPU_FALLBACK`, `XPCS_GPU_MEMORY_FRACTION`.
- **`DeviceInfo` dataclass**: Stores device type, ID, name, and memory info. Validates non-negative fields (BUG-044).
- **Graceful GPU fallback**: If GPU is requested but unavailable and `allow_gpu_fallback=True`, falls back to CPU with a warning.
- **`place_on_device(array)`**: Moves arrays to the configured device via `jax.device_put`.
- **`reset()` classmethod**: Destroys the singleton for testing.

### Conversion Utilities (`_conversions.py`)

```python
ensure_numpy(array)          # Any array -> writable np.ndarray
ensure_backend_array(array)  # Any array -> current backend's array type
is_jax_array(array)          # Type check
is_numpy_array(array)        # Type check
get_array_backend(array)     # Returns "numpy", "jax", or "unknown"
arrays_compatible(a, b)      # Same backend check
```

`ensure_numpy()` handles:
- NumPy arrays (fast path; copies if read-only).
- JAX arrays (copies to CPU via `np.array()`).
- Objects with `__array__` method.
- Lists and scalars (via `np.array()`).

### I/O Adapters (`io_adapter.py`)

Three adapter classes wrap `ensure_numpy()` with domain-specific semantics and optional performance monitoring:

| Adapter | Methods | Use Case |
|---------|---------|----------|
| `PyQtGraphAdapter` | `to_pyqtgraph()`, `from_pyqtgraph()` | GUI visualization |
| `HDF5Adapter` | `to_hdf5()`, `from_hdf5()` | File I/O |
| `MatplotlibAdapter` | `to_matplotlib(*arrays)` | Static plots |

Each adapter tracks conversion statistics (`get_stats()`, `reset_stats()`) and logs slow conversions (>10ms) at DEBUG level.

The `create_adapters()` factory function creates all three adapters from a single backend instance.

### Usage Pattern

```python
from xpcsviewer.backends import get_backend, ensure_numpy

backend = get_backend()  # Auto-detects JAX or NumPy

# Computation (backend-agnostic)
x = backend.linspace(0, 10, 100)
y = backend.exp(-x)

# I/O boundary
import pyqtgraph as pg
plot.setData(ensure_numpy(x), ensure_numpy(y))
```

## Consequences

### What became easier

- **Backend-agnostic code**: Analysis modules call `backend.method()` and work identically on JAX and NumPy.
- **Capability checks**: `if backend.supports_grad: result = backend.value_and_grad(fn)(x)` enables graceful degradation.
- **Testing**: `set_backend("numpy")` forces NumPy for deterministic tests. `reset_backend()` restores auto-detection.
- **I/O monitoring**: Adapters track conversion counts and timing, enabling performance profiling of I/O boundaries.
- **Device management**: A single `DeviceManager` instance coordinates all GPU/CPU decisions.

### What became more difficult

- **Protocol maintenance**: Adding a new mathematical operation requires updating `BackendProtocol`, `NumPyBackend`, and `JAXBackend` simultaneously.
- **No polymorphic dispatch on arrays**: Unlike PyTorch's `torch.Tensor` methods, backend arrays do not carry their backend. Operations like `x.mean()` are not available; callers must use `backend.mean(x)`.
- **JAX semantic constraints**: JIT-compiled functions must be functionally pure. Mutable state (e.g., counters, caches) must be refactored into explicit state passing or use JAX's `scan`/`fori_loop`.
- **Fan-in coupling**: 8+ modules depend on `get_backend()` and 9+ depend on `ensure_numpy()`. Changes to these functions cascade widely.
