# ADR-001: JAX Migration and Backend Abstraction

## Status

Accepted

## Context

XPCS Viewer originally used NumPy exclusively for all array computations. As datasets grew larger and fitting workloads became more demanding, several pain points emerged:

1. **Performance ceiling**: NumPy computations on large 2D detector images (1024x1024+) and multi-Q-bin correlation analysis were becoming the bottleneck in interactive workflows.
2. **No GPU acceleration**: Synchrotron beamlines increasingly provide GPU-equipped workstations, but NumPy cannot utilize them.
3. **No automatic differentiation**: The Bayesian fitting pipeline requires gradients for NUTS sampling. With NumPy, gradient computation had to be done numerically (finite differences), which is slow and imprecise.
4. **No JIT compilation**: Repeated Q-map computations and partition generation could not be compiled and cached.

The team evaluated JAX as the primary computational backend because:

- JAX provides a NumPy-compatible API (`jax.numpy`), minimizing migration effort.
- JAX supports JIT compilation via XLA, yielding 5-10x speedups on repeated computations.
- JAX supports automatic differentiation (`jax.grad`, `jax.value_and_grad`), which NumPyro requires for Hamiltonian Monte Carlo (NUTS) sampling.
- JAX supports GPU/TPU execution with the same code, via `jax.device_put`.

Initially, JAX was an optional dependency to support lightweight installations. As the project matured, JAX and its ecosystem (NumPyro, optimistix, interpax, optax) became core dependencies (listed in `pyproject.toml` under `dependencies`), since the Bayesian fitting pipeline and JIT-compiled Q-map computation are now integral to the application. The backend abstraction layer remains useful for testing (forcing NumPy behavior) and for graceful degradation when JAX is installed but GPU hardware is unavailable.

## Decision

We introduced a **backend abstraction layer** (`xpcsviewer/backends/`) that provides a unified `BackendProtocol` interface for array operations, with two concrete implementations:

### Architecture

```
xpcsviewer/backends/
  _base.py            # BackendProtocol (runtime_checkable Protocol)
  _jax_backend.py     # JAXBackend: full JIT, grad, GPU support
  _numpy_backend.py   # NumPyBackend: CPU-only fallback (no-op JIT, raises on grad)
  _device.py          # DeviceManager singleton: GPU/CPU selection, memory config
  _conversions.py     # ensure_numpy(), ensure_backend_array() at I/O boundaries
  io_adapter.py       # PyQtGraphAdapter, HDF5Adapter, MatplotlibAdapter
  __init__.py          # get_backend(), set_backend(), auto-detection
```

### Key Design Choices

1. **Protocol-based interface** (`typing.Protocol` with `@runtime_checkable`): Both backends implement `BackendProtocol`, which defines 60+ methods covering array creation, mathematical operations, statistical functions, JIT, grad, vmap, scan, and fori_loop. This uses structural subtyping -- no base class inheritance required.

2. **Auto-detection with environment override**: `get_backend()` auto-detects JAX availability. Users can force a backend via `XPCS_USE_JAX=true|false|auto`. When JAX is requested but not installed, a clear `ImportError` with install instructions is raised.

3. **float64 by default**: Scientific computing requires double precision. JAX is configured with `JAX_ENABLE_X64=true` on first use to ensure float64 arrays are supported.

4. **Singleton DeviceManager**: `DeviceManager` uses double-checked locking for thread-safe singleton creation. It reads `XPCS_USE_GPU`, `XPCS_GPU_FALLBACK`, and `XPCS_GPU_MEMORY_FRACTION` from the environment and configures the JAX runtime accordingly.

5. **`ensure_numpy()` at I/O boundaries**: All external libraries (h5py, PyQtGraph, Matplotlib) require NumPy arrays. The `ensure_numpy()` function in `_conversions.py` handles all array types (JAX arrays, read-only NumPy arrays, lists) and guarantees a writable NumPy ndarray.

6. **I/O Adapters**: `PyQtGraphAdapter`, `HDF5Adapter`, and `MatplotlibAdapter` wrap `ensure_numpy()` with optional performance monitoring (conversion count, timing statistics). These centralize boundary conversions instead of scattering `ensure_numpy()` calls across modules.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `XPCS_USE_JAX` | `auto` | `true`, `false`, or `auto` |
| `XPCS_USE_GPU` | `false` | Enable GPU device |
| `XPCS_GPU_FALLBACK` | `true` | Fall back to CPU if GPU unavailable |
| `XPCS_GPU_MEMORY_FRACTION` | `0.9` | Max GPU memory fraction |
| `JAX_PLATFORMS` | (unset) | Force JAX platform (`cpu`, `gpu`) |

## Consequences

### What became easier

- **Performance**: JIT-compiled Q-map computation runs ~10x faster on repeated calls. GPU acceleration provides further speedups for large datasets.
- **Bayesian fitting**: NumPyro NUTS sampling works natively with JAX arrays and automatic differentiation, eliminating the need for numerical gradients.
- **Backend flexibility**: The NumPy backend remains available via `XPCS_USE_JAX=false` for debugging and deterministic testing, even though JAX is installed by default.
- **Testing**: Both backends can be tested independently. The `reset_backend()` function allows tests to switch backends between test cases.

### What became more difficult

- **Debugging**: JAX's tracing semantics (functional purity requirements) can produce confusing error messages when Python control flow depends on array values.
- **Dependency size**: JAX + jaxlib adds ~500MB to the installation. Since JAX is now a core dependency, all installations carry this cost.
- **I/O boundary discipline**: Every interaction with external libraries must go through `ensure_numpy()`. Missing conversions produce runtime errors (JAX arrays are not accepted by h5py or PyQtGraph).
- **NumPy backend limitations**: `NumPyBackend.grad()` and `value_and_grad()` raise `NotImplementedError`. Code that requires gradients must check `backend.supports_grad` first.
