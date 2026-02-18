Configuration Reference
=======================

XPCS Viewer is configured through environment variables. All variables have
sensible defaults and are optional.

Backend Configuration
---------------------

These variables control the computational backend (JAX vs. NumPy) and GPU
device selection.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``XPCS_USE_JAX``
     - ``auto``
     - Backend selection. ``true`` forces JAX (raises ``ImportError`` if not
       installed), ``false`` forces NumPy, ``auto`` uses JAX if available.
   * - ``XPCS_USE_GPU``
     - ``false``
     - Enable GPU device for JAX computations. Requires JAX with GPU support
       (``jax[cuda12]`` or ``jax[metal]``).
   * - ``XPCS_GPU_FALLBACK``
     - ``true``
     - When GPU is requested but unavailable, fall back to CPU instead of
       raising an error.
   * - ``XPCS_GPU_MEMORY_FRACTION``
     - ``0.9``
     - Maximum fraction of GPU memory JAX may use (0.0 to 1.0). Maps to
       ``XLA_PYTHON_CLIENT_MEM_FRACTION``.

**JAX Platform Variables** (passed through to JAX):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``JAX_PLATFORMS``
     - (unset)
     - Force JAX platform: ``cpu``, ``gpu``, or ``tpu``.
   * - ``JAX_ENABLE_X64``
     - ``true``
     - Enable float64 support (required for scientific precision). XPCS Viewer
       sets this automatically.
   * - ``XLA_PYTHON_CLIENT_MEM_FRACTION``
     - ``0.9``
     - GPU memory fraction. Set automatically from ``XPCS_GPU_MEMORY_FRACTION``.

Examples
^^^^^^^^

Force NumPy backend (no JAX dependency)::

    export XPCS_USE_JAX=false
    python -m xpcsviewer

Enable GPU acceleration::

    export XPCS_USE_JAX=true
    export XPCS_USE_GPU=true
    export XPCS_GPU_MEMORY_FRACTION=0.8
    python -m xpcsviewer

Force CPU-only JAX (useful for debugging)::

    export XPCS_USE_JAX=true
    export JAX_PLATFORMS=cpu
    python -m xpcsviewer


Logging Configuration
---------------------

These variables control the logging system. See :doc:`logging` for detailed
usage examples.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``PYXPCS_LOG_LEVEL``
     - ``INFO``
     - Log level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
   * - ``PYXPCS_LOG_FORMAT``
     - ``TEXT``
     - Log format: ``TEXT`` for human-readable, ``JSON`` for structured
       logging (suitable for log aggregation tools).
   * - ``PYXPCS_LOG_DIR``
     - ``~/.xpcsviewer/logs``
     - Directory for log files. Created automatically if it does not exist.
   * - ``PYXPCS_LOG_FILE``
     - ``<log_dir>/xpcsviewer.log``
     - Custom log file path. Overrides the default file within ``PYXPCS_LOG_DIR``.
   * - ``PYXPCS_LOG_MAX_SIZE``
     - ``10``
     - Maximum log file size in megabytes before rotation.
   * - ``PYXPCS_LOG_BACKUP_COUNT``
     - ``5``
     - Number of rotated log files to keep.
   * - ``PYXPCS_LOG_RATE_LIMIT``
     - ``10.0``
     - Default rate limit for ``RateLimitedLogger`` in messages per second.
   * - ``PYXPCS_LOG_SANITIZE_PATHS``
     - ``home``
     - Path sanitization mode in logs: ``none`` (full paths), ``home``
       (replace home directory with ``~``), ``hash`` (hash filenames).
   * - ``PYXPCS_LOG_SESSION_ID``
     - ``1``
     - Enable session correlation IDs in log records. Set to ``0`` to disable.
   * - ``PYXPCS_SUPPRESS_QT_WARNINGS``
     - ``0``
     - Set to ``1`` to suppress Qt-related log messages (useful in headless
       environments).

Examples
^^^^^^^^

Enable debug logging with JSON format::

    export PYXPCS_LOG_LEVEL=DEBUG
    export PYXPCS_LOG_FORMAT=JSON
    python -m xpcsviewer

Reduce log file size for constrained environments::

    export PYXPCS_LOG_MAX_SIZE=2
    export PYXPCS_LOG_BACKUP_COUNT=3
    python -m xpcsviewer

Disable path sanitization for development::

    export PYXPCS_LOG_SANITIZE_PATHS=none
    python -m xpcsviewer


Documentation Build
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``BUILDING_DOCS``
     - (unset)
     - Set to ``1`` during Sphinx documentation builds to skip heavy
       imports and enable mock modules.


Combining Configuration
-----------------------

A typical production configuration for a GPU-equipped workstation at a
beamline::

    # Backend: JAX with GPU
    export XPCS_USE_JAX=true
    export XPCS_USE_GPU=true
    export XPCS_GPU_MEMORY_FRACTION=0.7

    # Logging: structured JSON, moderate verbosity
    export PYXPCS_LOG_LEVEL=INFO
    export PYXPCS_LOG_FORMAT=JSON
    export PYXPCS_LOG_SANITIZE_PATHS=hash

    # Suppress Qt noise
    export PYXPCS_SUPPRESS_QT_WARNINGS=1

    python -m xpcsviewer

A development configuration for debugging::

    # Backend: NumPy for deterministic behavior
    export XPCS_USE_JAX=false

    # Logging: verbose, unsanitized paths
    export PYXPCS_LOG_LEVEL=DEBUG
    export PYXPCS_LOG_FORMAT=TEXT
    export PYXPCS_LOG_SANITIZE_PATHS=none

    python -m xpcsviewer
