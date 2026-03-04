Backends
========

Backend abstraction layer for JAX/NumPy array operations with device management
and I/O boundary adapters.

.. currentmodule:: xpcsviewer.backends

Overview
--------

The backends module provides a unified interface for array computations that can
run on either JAX (with GPU support) or NumPy (CPU fallback). The backend is
selected automatically based on JAX availability, or explicitly via environment
variables.

Quick Start
-----------

.. code-block:: python

   from xpcsviewer.backends import get_backend, ensure_numpy

   backend = get_backend()  # Auto-detects JAX or NumPy
   print(f"Using {backend.name} backend")

   # Use backend for array operations
   x = backend.linspace(0, 1, 100)
   y = backend.sin(x)

   # Convert to NumPy at I/O boundaries
   import matplotlib.pyplot as plt
   plt.plot(ensure_numpy(x), ensure_numpy(y))

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``XPCS_USE_JAX``
     - ``auto``
     - ``'true'``, ``'false'``, or ``'auto'`` (detect JAX availability)
   * - ``XPCS_USE_GPU``
     - ``false``
     - Enable GPU computation
   * - ``XPCS_GPU_FALLBACK``
     - ``true``
     - Fall back to CPU if GPU unavailable
   * - ``XPCS_GPU_MEMORY_FRACTION``
     - ``0.9``
     - Maximum GPU memory fraction (0.0-1.0)

Backend Functions
-----------------

.. autofunction:: get_backend

.. autofunction:: set_backend

.. autofunction:: reset_backend

Backend Protocol
----------------

.. autoclass:: BackendProtocol
   :members:
   :no-index:

Array Conversions
-----------------

Utilities for converting between array types at I/O boundaries.

.. automodule:: xpcsviewer.backends._conversions
   :members:
   :no-index:

.. note::

   Always use ``ensure_numpy()`` at I/O boundaries (HDF5, PyQtGraph,
   Matplotlib) to avoid passing JAX arrays to libraries that expect NumPy.

I/O Adapters
------------

Centralized adapters for array conversion at I/O boundaries with optional
performance monitoring.

.. autoclass:: PyQtGraphAdapter
   :members:
   :no-index:

.. autoclass:: HDF5Adapter
   :members:
   :no-index:

.. autoclass:: MatplotlibAdapter
   :members:
   :no-index:

.. autofunction:: create_adapters

Device Management
-----------------

Singleton manager for compute device selection and GPU/CPU placement.

.. autoclass:: xpcsviewer.backends.DeviceType
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: xpcsviewer.backends.DeviceConfig
   :members:
   :no-index:

.. autoclass:: xpcsviewer.backends.DeviceManager
   :members:
   :no-index:

SciPy Replacements
-------------------

JAX-compatible replacements for SciPy functions using interpax, optimistix,
and optax. Used by the SimpleMask module for GPU-accelerated operations.

.. automodule:: xpcsviewer.backends.scipy_replacements
   :members:
   :no-index:

.. automodule:: xpcsviewer.backends.scipy_replacements.interpolate
   :members:
   :no-index:

.. automodule:: xpcsviewer.backends.scipy_replacements.ndimage
   :members:
   :no-index:

.. automodule:: xpcsviewer.backends.scipy_replacements.optimize
   :members:
   :no-index:

See Also
--------

- :doc:`schemas` - Type-safe data structures using backend arrays
- :doc:`io` - HDF5 facade with backend-aware I/O
- :doc:`fitting` - Fitting module using JAX backend for acceleration
