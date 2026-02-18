Logging Guide
=============

XPCS Viewer provides a comprehensive logging system with session correlation,
rate limiting, method timing, and path sanitization. This guide covers setup
and usage patterns.

Quick Start
-----------

Get a configured logger in any module::

    from xpcsviewer.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")

The logging system initializes automatically on first use. Configuration is
read from environment variables (see :doc:`configuration`).


Log Output
----------

Logs are written to two destinations simultaneously:

**Console** (stdout):
    Colored output using ``ColoredConsoleFormatter``. Includes timestamp,
    level, logger name, and message.

**Rotating file** (``~/.xpcsviewer/logs/xpcsviewer.log``):
    Uses ``RotatingFileHandler`` with configurable maximum size (default 10 MB)
    and backup count (default 5). File format is either structured text or JSON
    depending on ``PYXPCS_LOG_FORMAT``.


Session Context (LoggingContext)
--------------------------------

Use ``LoggingContext`` to correlate related log entries with a session ID.
This is valuable for tracing operations across modules in a complex analysis
workflow.

::

    from xpcsviewer.utils.log_utils import LoggingContext

    with LoggingContext(operation="batch_analysis") as ctx:
        logger.info("Starting batch")  # Includes session_id in log record

        for file_path in file_list:
            ctx.update_file(file_path)
            process_file(file_path)

        ctx.update_operation("fitting")
        run_fits()

The ``LoggingContext`` sets three ``contextvars`` that are thread-safe:

- **session_id**: Auto-generated UUID4 prefix (8 chars) or custom string
- **operation**: Current high-level operation name (max 100 chars)
- **current_file**: Currently loaded file path (auto-sanitized)

The ``SessionContextFilter`` (added automatically to handlers when
``PYXPCS_LOG_SESSION_ID=1``) enriches each log record with these fields.

Retrieving current context::

    from xpcsviewer.utils.log_utils import get_session_context

    ctx = get_session_context()
    # {'session_id': 'a1b2c3d4', 'operation': 'fitting', 'current_file': '~/data/...'}


Rate-Limited Logging (RateLimitedLogger)
-----------------------------------------

For high-frequency events (mouse moves, progress updates, per-pixel
operations), use ``RateLimitedLogger`` to prevent log flooding::

    from xpcsviewer.utils.log_utils import RateLimitedLogger

    rate_limited = RateLimitedLogger(logger, rate_per_second=5.0)

    def on_mouse_move(event):
        rate_limited.debug(f"Mouse at ({event.x}, {event.y})")  # Max 5/sec

The rate limiter uses a **token bucket algorithm**:

- Each unique message template (first 50 characters) gets its own bucket.
- Tokens refill at ``rate_per_second``.
- Burst size defaults to ``rate_per_second`` (allows short bursts).

Available methods: ``debug()``, ``info()``, ``warning()``, ``error()``,
``critical()``. Each returns ``True`` if the message was logged, ``False``
if suppressed.

Checking suppressed messages::

    count = rate_limited.get_suppressed_count()       # Total suppressed
    count = rate_limited.get_suppressed_count("Mouse") # For specific prefix
    rate_limited.reset()                               # Clear all state


Method Timing (log_timing)
--------------------------

Use the ``@log_timing`` decorator to measure and log method execution time::

    from xpcsviewer.utils.log_utils import log_timing

    @log_timing()
    def process_data(data):
        ...  # Logs: "process_data completed in 234.56ms" at DEBUG level

    @log_timing(threshold_ms=5000)
    def long_operation():
        ...  # Logs at WARNING level if execution exceeds 5 seconds

    @log_timing(threshold_ms=1000, include_args=True)
    def fit_q_bin(q_idx, model_name):
        ...  # Logs: "fit_q_bin(0, 'single_exp') completed in 1234.56ms"

Parameters:

- **logger**: Logger to use (default: module logger of decorated function)
- **level**: Normal log level (default: ``DEBUG``)
- **threshold_ms**: Time threshold for elevated logging (optional)
- **threshold_level**: Level when threshold exceeded (default: ``WARNING``)
- **include_args**: Include function arguments in log message (default: ``False``)

If the decorated function raises an exception, the timing is still logged
at ``ERROR`` level with the traceback.


Path Sanitization
-----------------

Always sanitize file paths before including them in log messages to protect
user privacy::

    from xpcsviewer.utils.log_utils import sanitize_path

    logger.info(f"Loaded: {sanitize_path(file_path)}")

Sanitization modes (configured via ``PYXPCS_LOG_SANITIZE_PATHS``):

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Mode
     - Input
     - Output
   * - ``none``
     - ``/Users/john/data/experiment.h5``
     - ``/Users/john/data/experiment.h5``
   * - ``home``
     - ``/Users/john/data/experiment.h5``
     - ``~/data/experiment.h5``
   * - ``hash``
     - ``/Users/john/data/experiment.h5``
     - ``~/data/a1b2c3d4.h5``

The ``home`` mode replaces the user's home directory with ``~``. The ``hash``
mode additionally replaces the filename with an 8-character SHA-256 hash
while preserving the extension.


Array Logging Guards
--------------------

For expensive debug operations involving array metadata, use the
``isEnabledFor`` guard to avoid computing values when debug logging is
disabled::

    import logging

    if logger.isEnabledFor(logging.DEBUG):
        shape = getattr(array, "shape", "N/A")
        dtype = getattr(array, "dtype", "N/A")
        has_nan = bool(np.any(np.isnan(array)))
        logger.debug(f"Array: shape={shape}, dtype={dtype}, has_nan={has_nan}")

This pattern is used throughout the codebase (``backends/io_adapter.py``,
``backends/_device.py``, etc.) to avoid unnecessary computation when logging
at INFO or higher levels.


JSON Structured Logging
-----------------------

Set ``PYXPCS_LOG_FORMAT=JSON`` to output structured JSON log records. Each
line is a valid JSON object::

    {"timestamp": "2026-02-17T10:30:00.123", "level": "INFO",
     "logger": "xpcsviewer.fitting.nlsq", "message": "nlsq_optimize completed in 45.67ms",
     "session_id": "a1b2c3d4", "operation": "fitting", "app_name": "XPCS Viewer"}

JSON format is suitable for log aggregation tools (ELK stack, Datadog, etc.)
in multi-user beamline environments.


Programmatic Configuration
--------------------------

The logging system can also be configured programmatically::

    from xpcsviewer.utils.logging_config import (
        get_logging_config,
        set_log_level,
        get_log_file_path,
        setup_exception_logging,
        log_system_info,
    )

    # Change log level at runtime
    set_log_level("DEBUG")

    # Get current log file path
    log_path = get_log_file_path()

    # Enable uncaught exception logging
    setup_exception_logging()

    # Log system information (Python version, NumPy version, etc.)
    log_system_info()

    # Get full configuration
    config = get_logging_config()
    info = config.get_logger_info()
    # {'log_level': 'DEBUG', 'log_file': '...', 'format': 'TEXT', ...}
