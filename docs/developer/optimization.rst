Performance Optimization
=======================

XPCS Toolkit includes comprehensive performance optimizations across all major
subsystems. This guide covers the optimization architecture and usage guidelines.

Optimization Overview
---------------------

The toolkit has been optimized through a multi-phase approach targeting:

- **Threading System**: 25-40% performance improvement
- **Memory Management**: 40-60% reduction in overhead
- **I/O Operations**: Connection pooling and batch processing
- **Scientific Computing**: Vectorized algorithms and parallel processing

Core Optimizations
------------------

Threading System
~~~~~~~~~~~~~~~~

The async worker framework provides:

- Non-blocking GUI operations
- Parallel processing for CPU-intensive tasks
- Automatic thread pool management
- Progress reporting and cancellation support

.. code-block:: python

   from xpcs_toolkit.threading import AsyncWorker

   # Example usage
   worker = AsyncWorker(compute_intensive_function, data)
   worker.start()

Memory Management
~~~~~~~~~~~~~~~~~

LRU caching and memory pressure detection:

- Automatic cleanup of large data arrays
- Lazy loading of SAXS data
- Memory usage monitoring
- Configurable cache sizes

.. code-block:: python

   from xpcs_toolkit.utils import MemoryManager

   # Configure memory limits
   MemoryManager.set_max_memory("8GB")
   MemoryManager.enable_pressure_detection()

I/O Performance
~~~~~~~~~~~~~~~

HDF5 connection pooling and batch operations:

- Reduced file open/close overhead
- Batch reading of multiple datasets
- Cached metadata access
- Connection health monitoring

Scientific Computing
~~~~~~~~~~~~~~~~~~~~~

Vectorized operations and parallel algorithms:

- NumPy-optimized array operations
- Multiprocessing for two-time correlation
- Parallel file averaging
- Optimized fitting routines

Configuration
-------------

Performance settings can be configured via environment variables:

.. code-block:: bash

   # Memory management
   export PYXPCS_MAX_MEMORY=16GB
   export PYXPCS_ENABLE_CACHING=1

   # Threading
   export PYXPCS_MAX_WORKERS=8
   export PYXPCS_THREAD_POOL_SIZE=4

   # I/O optimization
   export PYXPCS_CONNECTION_POOL_SIZE=25
   export PYXPCS_BATCH_SIZE=100

Monitoring
----------

Built-in performance monitoring:

.. code-block:: python

   from xpcs_toolkit.utils import PerformanceMonitor

   # Start monitoring
   monitor = PerformanceMonitor()
   monitor.start()

   # View metrics
   metrics = monitor.get_metrics()
   print(f"Memory usage: {metrics['memory_usage']:.1f}%")
   print(f"CPU usage: {metrics['cpu_usage']:.1f}%")

Best Practices
--------------

1. **Memory**: Set appropriate memory limits for your system
2. **Threading**: Use async workers for long-running operations
3. **I/O**: Process files in batches when possible
4. **Caching**: Enable caching for repeated data access
5. **Monitoring**: Use performance monitoring in production

For detailed optimization information, see the complete `optimization guide <../OPTIMIZATION_GUIDE.md>`_.
