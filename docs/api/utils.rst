Utilities
=========

Utility modules providing logging, memory management, performance monitoring,
and other support functions.

.. currentmodule:: xpcsviewer.utils

Logging Configuration
---------------------

Advanced logging setup with hierarchical loggers, custom formatters, and
environment-variable driven configuration.

.. automodule:: xpcsviewer.utils.logging_config
   :members: get_logger, initialize_logging, set_log_level, setup_logging, log_system_info, setup_exception_logging
   :no-index:

Logging Utilities
-----------------

Session context, rate-limited logging, method timing, and path sanitization.

.. automodule:: xpcsviewer.utils.log_utils
   :members: LoggingContext, SessionContextFilter, RateLimitedLogger, log_timing, sanitize_path
   :no-index:

**Environment Variables:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``PYXPCS_LOG_LEVEL``
     - ``INFO``
     - DEBUG/INFO/WARNING/ERROR/CRITICAL
   * - ``PYXPCS_LOG_FORMAT``
     - ``TEXT``
     - TEXT or JSON (for log aggregation)
   * - ``PYXPCS_LOG_RATE_LIMIT``
     - ``10.0``
     - Rate limit (msgs/sec)
   * - ``PYXPCS_LOG_SANITIZE_PATHS``
     - ``home``
     - none/home/hash

Log Formatters
--------------

Custom log formatters for console, file, JSON, and performance output.

.. automodule:: xpcsviewer.utils.log_formatters
   :members: ColoredConsoleFormatter, JSONFormatter, PerformanceFormatter, StructuredFileFormatter, create_formatter
   :no-index:

Memory Management
-----------------

Intelligent memory management with LRU caching, memory pressure detection,
and automatic cleanup.

.. automodule:: xpcsviewer.utils.memory_manager
   :members:
   :no-index:

Performance Monitoring
----------------------

Real-time performance tracking for optimization and debugging.

.. automodule:: xpcsviewer.utils.performance_monitor
   :members:
   :no-index:

Reliability
-----------

Error handling, validation, and fallback mechanisms.

.. automodule:: xpcsviewer.utils.reliability
   :members:
   :no-index:

Validation
----------

Input validation utilities for data integrity checks.

.. automodule:: xpcsviewer.utils.validation
   :members:
   :no-index:

Exceptions
----------

Custom exception hierarchy for XPCS-specific error handling.

The validation system uses ``XPCSValidationError`` for structured error
reporting. Analysis modules catch these exceptions to skip incompatible
files gracefully rather than silently trimming data.

.. automodule:: xpcsviewer.utils.exceptions
   :members:
   :no-index:

Common Checks
-------------

Lightweight guard functions for quick input pre-checks.

.. automodule:: xpcsviewer.utils.common_checks
   :members:
   :no-index:

Data Processing
---------------

Optimized data processing utilities.

**Lazy Loader** -- Deferred module loading for startup performance:

.. automodule:: xpcsviewer.utils.lazy_loader
   :members:
   :no-index:

**Streaming Processor** -- Chunk-based data processing:

.. automodule:: xpcsviewer.utils.streaming_processor
   :members:
   :no-index:

**Vectorized ROI** -- Optimized region-of-interest extraction:

.. automodule:: xpcsviewer.utils.vectorized_roi
   :members:
   :no-index:

Atomic I/O
----------

Crash-safe JSON writes via atomic file replacement.

.. automodule:: xpcsviewer.utils.atomic_io
   :members:
   :no-index:

Device Detection
----------------

GPU detection, system CUDA discovery, and JAX backend diagnostics.

.. automodule:: xpcsviewer.utils.device
   :members:
   :no-index:

Health Monitoring
-----------------

Background health monitoring for resource usage and reliability metrics.

.. automodule:: xpcsviewer.utils.health_monitor
   :members:
   :no-index:

Memory Prediction
-----------------

Predictive memory pressure detection for proactive memory management
in XPCS data analysis workflows.

.. automodule:: xpcsviewer.utils.memory_predictor
   :members:
   :no-index:

Reliability Manager
-------------------

Unified entry point for enabling and configuring reliability features.

.. automodule:: xpcsviewer.utils.reliability_manager
   :members:
   :no-index:

Startup Optimization
--------------------

Application startup performance optimization via lazy loading,
parallel initialization, and resource preloading.

.. automodule:: xpcsviewer.utils.startup_optimizer
   :members:
   :no-index:

State Validation
----------------

Lock-free state consistency validation using atomic operations
and weak references.

.. automodule:: xpcsviewer.utils.state_validator
   :members:
   :no-index:

Visualization Optimization
--------------------------

Performance optimization for the visualization layer including
PyQtGraph and matplotlib backends.

This module conditionally imports PyQtGraph and cannot be introspected
during the documentation build.  See the source at
``xpcsviewer/utils/visualization_optimizer.py`` for the full API.

See Also
--------

- :doc:`constants` - Application-wide configuration constants
- :doc:`backends` - Backend abstraction layer
