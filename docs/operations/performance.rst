Performance Tuning
==================

This guide covers performance optimization techniques for XPCS Viewer,
including JIT compilation, GPU acceleration, and connection pooling.


JIT Compilation (JAX Backend)
-----------------------------

When the JAX backend is active (``XPCS_USE_JAX=true`` or auto-detected),
functions decorated with ``@jax.jit`` or called via ``backend.jit()`` are
compiled to XLA on first invocation. Subsequent calls use the cached compiled
version.

**Typical speedups**:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Operation
     - First Call
     - Subsequent Calls
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

**Best practices for JIT**:

1. **Avoid Python control flow that depends on array values.** JAX traces
   functions abstractly. Use ``jax.lax.cond`` instead of ``if``/``else``
   on traced values.

2. **Use ``static_argnums`` for non-array arguments.** When a function
   takes integer or string arguments that affect control flow, mark them
   as static::

       @jax.jit
       def compute(data, mode):  # 'mode' changes code path
           ...

       # Better:
       @functools.partial(jax.jit, static_argnums=(1,))
       def compute(data, mode):
           ...

3. **Warm up JIT caches at startup.** For interactive workflows, call
   JIT-compiled functions once with representative data during initialization
   to avoid compilation latency on the first user interaction.

4. **Monitor compilation via logging.** Set ``PYXPCS_LOG_LEVEL=DEBUG`` to see
   ``@log_timing`` output for JIT-compiled functions. First-call times that
   are 10-100x longer than subsequent calls indicate JIT compilation overhead.


GPU Acceleration
----------------

Enable GPU computation for large datasets::

    export XPCS_USE_JAX=true
    export XPCS_USE_GPU=true

**When GPU helps**:

- Large detector images (>1024x1024 pixels)
- Batch fitting across many Q-bins
- Two-time correlation computation on large frame sequences

**When GPU does not help** (or hurts):

- Small arrays (<10,000 elements): Host-to-device transfer overhead dominates
- Single-array operations: Not enough parallelism to saturate the GPU
- I/O-bound workflows: Time is spent in HDF5 reads, not computation

**Memory management**:

Control GPU memory allocation via ``XPCS_GPU_MEMORY_FRACTION``::

    # Use only 50% of GPU memory (share with other processes)
    export XPCS_GPU_MEMORY_FRACTION=0.5

The ``DeviceManager`` reports GPU status::

    from xpcsviewer.backends import DeviceManager

    dm = DeviceManager()
    print(f"GPU available: {dm.gpu_available}")
    print(f"GPU enabled: {dm.is_gpu_enabled}")
    print(f"Current device: {dm.current_device}")
    print(f"All devices: {dm.available_devices}")

**Explicit device placement**::

    from xpcsviewer.backends import DeviceManager

    dm = DeviceManager()
    gpu_array = dm.place_on_device(numpy_array)  # Moves to configured device


HDF5 Connection Pooling
------------------------

The ``HDF5ConnectionPool`` in ``fileIO/hdf_reader.py`` caches open file
handles to avoid repeated open/close cycles during interactive analysis.

**Monitoring pool performance**::

    from xpcsviewer.io import HDF5Facade

    facade = HDF5Facade()
    stats = facade.get_pool_stats()
    # {
    #     "pool_size": 3,
    #     "cache_hits": 150,
    #     "cache_misses": 5,
    #     "cache_hit_ratio": 0.968,
    # }

**Target metrics**:

- Cache hit ratio: > 95% in typical interactive use
- Average read latency: < 10 ms for cached connections
- Pool size: 3--5 concurrent files for normal workflows

**Clearing the pool**:

Release all pooled connections before application shutdown or when switching
to a different dataset directory::

    facade.clear_pool()


I/O Boundary Optimization
--------------------------

Array conversion at I/O boundaries (``ensure_numpy()``) is the most frequent
performance-sensitive operation. The I/O adapters provide monitoring to
identify bottlenecks.

**Enable monitoring**::

    from xpcsviewer.backends import get_backend, create_adapters

    backend = get_backend()
    pyqt_adapter, hdf5_adapter, mpl_adapter = create_adapters(
        backend,
        enable_monitoring=True,
    )

    # ... use adapters in analysis workflow ...

    # Check statistics
    stats = pyqt_adapter.get_stats()
    print(f"Conversions: {stats['conversion_count']}")
    print(f"Avg time: {stats['average_conversion_time_ms']:.3f} ms")

**Typical conversion overhead**:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Array Size
     - CPU (NumPy -> NumPy)
     - GPU (JAX -> NumPy)
   * - 100x100 (10K elements)
     - < 0.01 ms
     - ~0.1 ms
   * - 1024x1024 (1M elements)
     - < 0.01 ms
     - ~0.5 ms
   * - 4096x4096 (16M elements)
     - < 0.01 ms
     - ~5 ms

When the backend is NumPy, ``ensure_numpy()`` is essentially free (returns the
input array if it is already a writable NumPy array). When the backend is JAX,
data must be copied from the device (GPU or CPU-backed JAX arrays) to a
standard NumPy array.

**Minimize unnecessary conversions**:

- Convert at the I/O boundary, not inside computation loops.
- Avoid converting the same array multiple times; store the NumPy result.
- For small arrays used only for display, the overhead is negligible.


Fitting Performance
-------------------

**NLSQ warm-start**:

The NLSQ solver provides fast point estimates to initialize the Bayesian
sampler. Use presets to control the speed/robustness tradeoff:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Preset
     - Typical Time
     - Use Case
   * - ``fast``
     - 1--5 ms
     - Quick preview, known-good data
   * - ``robust``
     - 5--50 ms
     - Default for production use
   * - ``global``
     - 50--500 ms
     - Multi-modal or difficult optimization landscapes
   * - ``large``
     - varies
     - Large parameter spaces (>10 parameters)

**NumPyro NUTS tuning**:

Adjust ``SamplerConfig`` for the speed/accuracy tradeoff::

    from xpcsviewer.fitting.results import SamplerConfig

    # Fast exploration (fewer samples, fewer chains)
    fast_config = SamplerConfig(
        num_warmup=200,
        num_samples=500,
        num_chains=2,
    )

    # Production quality
    production_config = SamplerConfig(
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
        target_accept_prob=0.8,
        random_seed=42,  # Reproducibility
    )

**Batch fitting tips**:

1. Fit multiple Q-bins in parallel using the threading manager.
2. Check ``FitDiagnostics.converged`` to skip diagnostic plots for
   well-converged fits.
3. Use ``nlsq_optimize()`` alone (without NUTS) for exploratory analysis
   where full Bayesian inference is not needed.


Profiling Workflow
------------------

1. **Identify the bottleneck** using ``@log_timing`` output::

       export PYXPCS_LOG_LEVEL=DEBUG
       python -m xpcsviewer

   Look for methods that exceed their ``threshold_ms`` (logged at WARNING).

2. **Check I/O adapter stats** for excessive conversions::

       stats = pyqt_adapter.get_stats()
       if stats['average_conversion_time_ms'] > 1.0:
           print("Conversion overhead detected")

3. **Check HDF5 pool stats** for cache misses::

       pool_stats = facade.get_pool_stats()
       if pool_stats['cache_hit_ratio'] < 0.9:
           print("Poor cache hit ratio -- check file access patterns")

4. **Check JIT compilation** by comparing first-call vs. subsequent-call
   times in DEBUG logs. First calls that are >100x slower indicate JIT
   overhead that may benefit from warm-up.
