Backend Selection: JAX vs NumPy
================================

This tutorial explains the backend abstraction layer, how to switch between
JAX and NumPy backends, and when to use each for optimal performance.

.. contents:: In this tutorial
   :local:
   :depth: 2

Architecture Overview
----------------------

xpcsviewer uses a backend abstraction that provides a unified API for
array operations. All computation code uses the backend instead of
importing ``numpy`` or ``jax.numpy`` directly. This allows:

- **Automatic fallback**: JAX if available, NumPy otherwise
- **GPU acceleration**: JAX backend enables JIT compilation and GPU offloading
- **Gradient computation**: Needed for advanced fitting and optimization
- **Same API**: Switch backends without changing analysis code

The backend protocol is defined in :class:`~xpcsviewer.backends._base.BackendProtocol`.

Checking the Active Backend
-----------------------------

.. code-block:: python

   from xpcsviewer.backends import get_backend

   backend = get_backend()
   print(f"Backend: {backend.name}")
   print(f"GPU support: {backend.supports_gpu}")
   print(f"JIT support: {backend.supports_jit}")
   print(f"Grad support: {backend.supports_grad}")

Switching Backends
-------------------

Programmatic Selection
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from xpcsviewer.backends import set_backend, get_backend

   # Force NumPy backend
   set_backend("numpy")
   print(f"Now using: {get_backend().name}")

   # Force JAX backend (requires JAX installation)
   set_backend("jax")
   print(f"Now using: {get_backend().name}")

Environment Variable Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``XPCS_USE_JAX=1`` before importing xpcsviewer to enable JAX:

.. code-block:: bash

   # Enable JAX backend
   export XPCS_USE_JAX=1

   # Force CPU-only JAX (useful for debugging)
   export JAX_PLATFORMS=cpu

   # Then run your script
   python my_analysis.py

If ``XPCS_USE_JAX`` is not set, the backend auto-detects: JAX if importable,
NumPy as fallback.

Using the Backend API
----------------------

Basic Array Operations
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from xpcsviewer.backends import get_backend

   backend = get_backend()

   # Array creation
   x = backend.linspace(0, 10, 100)
   y = backend.zeros((3, 3))
   z = backend.arange(0, 100, 0.5)

   # Math operations
   result = backend.sin(x)
   result = backend.exp(-x)
   result = backend.sqrt(backend.sum(x ** 2))

   # Statistics
   mean_val = backend.mean(x)
   std_val = backend.std(x)
   min_val = backend.min(x)

   # Array manipulation
   stacked = backend.stack([x, result])
   reshaped = backend.reshape(x, (10, 10))

All backend methods mirror the NumPy/JAX API and return arrays in the
active backend's format.

JIT Compilation (JAX Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   backend = get_backend()

   if backend.supports_jit:
       @backend.jit
       def compute_g2(t, tau, baseline, contrast):
           return baseline + contrast * backend.exp(-2 * t / tau)

       # First call compiles; subsequent calls are fast
       t = backend.linspace(0.01, 1000, 10000)
       g2 = compute_g2(t, 5.0, 1.0, 0.3)

Gradient Computation (JAX Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   backend = get_backend()

   if backend.supports_grad:
       def loss_fn(params, x, y_obs):
           tau, baseline, contrast = params
           y_pred = baseline + contrast * backend.exp(-2 * x / tau)
           return backend.sum((y_pred - y_obs) ** 2)

       # Get loss value and gradients simultaneously
       value_and_grad_fn = backend.value_and_grad(loss_fn)
       params = backend.array([5.0, 1.0, 0.3])
       loss, grads = value_and_grad_fn(params, x, y_obs)
       print(f"Loss: {loss}, Gradients: {grads}")

I/O Boundary Conversions
--------------------------

JAX arrays are not directly compatible with HDF5, PyQtGraph, or Matplotlib.
Always convert at I/O boundaries using :func:`~xpcsviewer.backends._conversions.ensure_numpy`:

.. code-block:: python

   from xpcsviewer.backends import get_backend, ensure_numpy
   import matplotlib.pyplot as plt

   backend = get_backend()

   # Compute with backend (may be JAX or NumPy)
   x = backend.linspace(0, 10, 100)
   y = backend.sin(x)

   # Convert at the plotting boundary
   plt.plot(ensure_numpy(x), ensure_numpy(y))
   plt.show()

The :func:`~xpcsviewer.backends._conversions.ensure_numpy` function:

- Returns NumPy arrays unchanged (no copy)
- Copies JAX arrays to host memory
- Handles lists and other array-like objects
- Ensures the result is writable (important for HDF5)

I/O Adapters
^^^^^^^^^^^^^^

For structured I/O, use the dedicated adapter classes:

.. code-block:: python

   from xpcsviewer.backends.io_adapter import PyQtGraphAdapter, HDF5Adapter

   # For PyQtGraph plots
   pg_adapter = PyQtGraphAdapter()
   plot_data = pg_adapter.to_pyqtgraph(jax_array)

   # For HDF5 writing
   h5_adapter = HDF5Adapter()
   h5_data = h5_adapter.to_hdf5(jax_array)

When to Use Each Backend
--------------------------

Use NumPy When:
^^^^^^^^^^^^^^^

- Running on machines without JAX/GPU
- Debugging (NumPy error messages are more descriptive)
- Small datasets where JIT compilation overhead dominates
- Interfacing with libraries that require NumPy arrays
- Running on macOS without GPU support

Use JAX When:
^^^^^^^^^^^^^

- Fitting many Q bins (JIT + vmap parallelism)
- Running Bayesian inference (NumPyro requires JAX)
- Needing gradient computation for optimization
- Working with large datasets (> 10,000 delay points)
- GPU hardware is available

Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time
   from xpcsviewer.backends import set_backend, get_backend

   def benchmark_g2_compute(backend, n_points=100000, n_trials=10):
       t = backend.linspace(0.01, 1000, n_points)
       tau = backend.array(5.0)

       # Warm-up (important for JIT)
       _ = backend.exp(-2 * t / tau)

       start = time.perf_counter()
       for _ in range(n_trials):
           _ = backend.exp(-2 * t / tau)
       elapsed = (time.perf_counter() - start) / n_trials

       return elapsed

   # NumPy benchmark
   set_backend("numpy")
   np_time = benchmark_g2_compute(get_backend())
   print(f"NumPy: {np_time*1000:.2f} ms")

   # JAX benchmark (if available)
   try:
       set_backend("jax")
       jax_time = benchmark_g2_compute(get_backend())
       print(f"JAX:   {jax_time*1000:.2f} ms")
       print(f"Speedup: {np_time/jax_time:.1f}x")
   except (ImportError, ValueError):
       print("JAX not available for comparison")

Troubleshooting
----------------

JAX Not Detected
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Check if JAX is importable
   try:
       import jax
       print(f"JAX version: {jax.__version__}")
       print(f"JAX devices: {jax.devices()}")
   except ImportError:
       print("JAX is not installed. Install with: pip install jax jaxlib")

Memory Issues with JAX
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Limit JAX GPU memory pre-allocation
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

   # Force CPU-only for debugging
   export JAX_PLATFORMS=cpu

Float64 Precision
^^^^^^^^^^^^^^^^^^

JAX defaults to float32. xpcsviewer configures float64 precision automatically
via ``jax.config.update("jax_enable_x64", True)``. Verify:

.. code-block:: python

   from xpcsviewer.backends import get_backend

   backend = get_backend()
   x = backend.linspace(0, 1, 10)
   print(f"Default dtype: {x.dtype}")  # Should be float64

Next Steps
----------

- :doc:`fitting_guide` -- Use JAX acceleration for fitting
- :doc:`cookbook` -- Backend-aware analysis patterns
- :doc:`getting_started` -- Loading data and basic analysis
