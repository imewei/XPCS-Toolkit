JAX Backend Architecture
=========================

This page explains the design and performance model of XPCS Viewer's backend
abstraction layer, which provides transparent switching between JAX and NumPy
computational backends.

Why a Backend Abstraction?
---------------------------

XPCS Viewer performs intensive numerical computation: Q-map generation over
million-pixel detectors, correlation function fitting across dozens of Q-bins,
and two-time correlation on thousands of frames. JAX provides JIT compilation,
automatic differentiation, and optional GPU acceleration that can deliver
10-100x speedups for these workloads.

However, the codebase must also work in environments where JAX is available but
GPU hardware is not, or where users want deterministic NumPy behavior for
debugging. The backend abstraction makes this transparent: analysis code calls
``backend.exp(x)`` and gets the right implementation regardless of whether the
active backend is JAX or NumPy.

Architecture Overview
----------------------

.. code-block:: text

   Application Code
          |
          v
   get_backend()  -->  BackendProtocol (60+ methods)
          |                   |                |
          v                   v                v
     JAXBackend         NumPyBackend     DeviceManager
     - jax.numpy        - numpy          - GPU/CPU selection
     - jax.jit          - no-op jit      - Memory config
     - jax.grad         - raises         - Device placement
          |
          v
   ensure_numpy()  -->  I/O Adapters
                        - PyQtGraphAdapter
                        - HDF5Adapter
                        - MatplotlibAdapter

The Protocol Pattern
^^^^^^^^^^^^^^^^^^^^^

The abstraction uses Python's ``typing.Protocol`` with ``@runtime_checkable``
rather than an abstract base class. This means:

- **No inheritance required**: Each backend independently implements the
  same interface through structural subtyping (duck typing with contracts).
- **No JAX import at definition time**: The protocol is defined using only
  standard Python types. ``JAXBackend`` imports JAX lazily.
- **Runtime checking**: ``isinstance(backend, BackendProtocol)`` works at
  runtime, enabling defensive checks.

The protocol defines methods across eight categories: array creation,
trigonometry, statistics, binning, boolean/masking, math, and functional
transformations (jit, grad, vmap, scan, fori_loop).

Backend Selection
------------------

Backend selection follows a priority chain:

1. **Explicit environment variable**: ``XPCS_USE_JAX=true`` forces JAX;
   ``XPCS_USE_JAX=false`` forces NumPy.
2. **Auto-detection** (``XPCS_USE_JAX=auto``, the default): Attempts to
   import JAX. If successful, uses JAXBackend; otherwise falls back to
   NumPyBackend with no error.
3. **Programmatic override**: ``set_backend("numpy")`` in code or tests.

Once selected, the backend is cached as a module-level singleton.
``reset_backend()`` clears the cache (used in tests to switch backends
between test cases).

JAX Configuration
^^^^^^^^^^^^^^^^^^

When JAX is first used, XPCS Viewer configures it for scientific computing:

- ``JAX_ENABLE_X64=true``: Enables float64 support. Without this, JAX
  defaults to float32, which is insufficient for the precision requirements
  of correlation function fitting.
- ``XLA_PYTHON_CLIENT_MEM_FRACTION``: Set from ``XPCS_GPU_MEMORY_FRACTION``
  to control GPU memory allocation.

These environment variables must be set **before** JAX is imported, which is
why the configuration happens in ``_configure_jax()`` called from
``get_backend()``.

JIT Compilation
----------------

JIT (Just-In-Time) compilation is the primary performance benefit of the JAX
backend. When a function is decorated with ``@jax.jit`` or called via
``backend.jit(fn)``, JAX traces the function with abstract values to build an
XLA computation graph, then compiles it to optimized machine code.

.. list-table:: Typical JIT speedups in XPCS Viewer
   :header-rows: 1
   :widths: 40 20 20 20

   * - Operation
     - First Call
     - Subsequent
     - Speedup
   * - Q-map computation (1024x1024)
     - ~200 ms
     - ~5 ms
     - 40x
   * - Partition generation
     - ~100 ms
     - ~3 ms
     - 33x
   * - Model function evaluation
     - ~50 ms
     - ~0.5 ms
     - 100x

**Caching strategy**: The Q-map module uses a dict-based JIT cache
(``_JIT_CACHE``) because JAX arrays are not hashable and cannot be used as
``lru_cache`` keys. The first call for a given scattering geometry type
creates and caches the JIT-compiled function; subsequent calls reuse it.

**Tracing constraints**: JIT-compiled functions must be **functionally pure**
-- they cannot use Python-level control flow that depends on array values
(e.g., ``if x > 0``). The codebase uses ``static_argnums`` for arguments
that control code paths (like string mode selectors) and JAX primitives
(``jax.lax.cond``, ``jax.lax.fori_loop``) for array-dependent branching.

NumPy Backend Behavior
^^^^^^^^^^^^^^^^^^^^^^^

When JAX is not active, the NumPy backend provides the same API with
no-op transformations:

- ``backend.jit(fn)`` returns ``fn`` unchanged (no compilation).
- ``backend.grad()`` and ``backend.value_and_grad()`` raise
  ``NotImplementedError``. Code that requires gradients must check
  ``backend.supports_grad`` first.
- ``backend.vmap(fn)`` falls back to a Python loop.
- ``backend.scan()`` and ``backend.fori_loop()`` use standard Python
  ``for`` loops.

This means NumPy backend code is functionally identical but slower for
repeated computations (no compilation cache) and cannot perform automatic
differentiation (required for Bayesian NUTS sampling).

I/O Boundaries
---------------

External libraries (h5py, PyQtGraph, Matplotlib) require NumPy arrays.
When the JAX backend is active, arrays must be converted at every I/O
boundary.

The Boundary Problem
^^^^^^^^^^^^^^^^^^^^^

Missing ``ensure_numpy()`` calls produce runtime errors:

- h5py raises ``TypeError`` when given a JAX array
- PyQtGraph renders nothing or crashes silently
- Matplotlib raises ``ValueError`` on non-NumPy input

These errors are easy to introduce during development because code works
fine with the NumPy backend but fails with JAX.

I/O Adapters
^^^^^^^^^^^^^

To centralize and monitor boundary conversions, three adapter classes wrap
``ensure_numpy()``:

- **PyQtGraphAdapter**: ``to_pyqtgraph(array)`` / ``from_pyqtgraph(array)``
  for all GUI visualization
- **HDF5Adapter**: ``to_hdf5(array)`` / ``from_hdf5(array)`` for file I/O
- **MatplotlibAdapter**: ``to_matplotlib(*arrays)`` for static plots

Each adapter optionally tracks conversion statistics (count, average time,
slowest conversion) for performance profiling. When the backend is NumPy,
conversions are essentially free (the input array is already NumPy).

Conversion overhead for JAX arrays:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Array Size
     - CPU JAX
     - GPU JAX
   * - 100x100
     - < 0.01 ms
     - ~0.1 ms
   * - 1024x1024
     - < 0.01 ms
     - ~0.5 ms
   * - 4096x4096
     - < 0.01 ms
     - ~5 ms

GPU data transfer dominates the cost for GPU arrays. CPU JAX arrays share
memory with NumPy and convert with near-zero overhead.

Device Management
------------------

The ``DeviceManager`` is a thread-safe singleton that controls GPU/CPU
device selection:

- **Configuration**: Reads ``XPCS_USE_GPU``, ``XPCS_GPU_FALLBACK``, and
  ``XPCS_GPU_MEMORY_FRACTION`` from the environment.
- **Graceful fallback**: If GPU is requested but unavailable and
  ``XPCS_GPU_FALLBACK=true`` (the default), falls back to CPU with a
  warning instead of crashing.
- **Device placement**: ``place_on_device(array)`` moves arrays to the
  configured device via ``jax.device_put``.
- **Thread safety**: Uses double-checked locking for singleton creation
  to avoid race conditions in multi-threaded GUI applications.

When to Use GPU
^^^^^^^^^^^^^^^^

GPU acceleration helps most for:

- Large detector images (>1024x1024 pixels) during Q-map computation
- Batch fitting across many Q-bins (parallelizable across GPU cores)
- Two-time correlation on large frame sequences

GPU does **not** help (and may hurt due to transfer overhead) for:

- Small arrays (<10,000 elements)
- Single scalar operations
- I/O-bound workflows (time spent in HDF5 reads, not computation)

Automatic Differentiation
--------------------------

JAX's ``grad`` and ``value_and_grad`` transformations are essential for the
Bayesian fitting pipeline. NumPyro's NUTS sampler requires gradients of the
log-posterior to perform Hamiltonian Monte Carlo sampling.

The backend exposes this capability through:

- ``backend.supports_grad``: Boolean capability check
- ``backend.grad(fn)``: Returns a function that computes the gradient
- ``backend.value_and_grad(fn)``: Returns both the function value and
  its gradient in a single forward/backward pass

The fitting module's model functions (``single_exp_func``,
``double_exp_func``, ``stretched_exp_func``) are decorated with
``@_maybe_jit``, which applies ``jax.jit`` when JAX is available. These
JIT-compiled functions are automatically differentiable by JAX's tracing
mechanism.
