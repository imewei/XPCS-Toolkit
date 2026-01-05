Constants
=========

Centralized constants for application-wide configuration values.

.. note::
   Constants are organized into domain-specific submodules for better
   maintainability and discoverability.

.. currentmodule:: xpcsviewer.constants

Overview
--------

The constants module provides centralized configuration values organized
by domain:

- **thresholds**: Numeric comparison thresholds (e.g., MIN_Q_VALUE)
- **timeouts**: Time-related constants (e.g., FILE_LOAD_TIMEOUT)
- **limits**: Size and count limits (e.g., MAX_CACHE_SIZE_MB)
- **defaults**: Default configuration values (e.g., DEFAULT_THEME)
- **fitting**: Curve fitting parameters (e.g., SINGLE_EXP_PARAMS)

Usage
-----

Import constants directly from the main module:

.. code-block:: python

    from xpcsviewer.constants import MIN_Q_VALUE, MAX_CACHE_SIZE_MB

    # Or import from specific submodule
    from xpcsviewer.constants.timeouts import FILE_LOAD_TIMEOUT
    from xpcsviewer.constants.limits import MAX_PLOT_POINTS

Submodules
----------

Thresholds
~~~~~~~~~~

Numeric thresholds for comparisons and validation.

See :mod:`xpcsviewer.constants.thresholds` for complete API documentation.

Timeouts
~~~~~~~~

Time-related constants for operations, file loading, and cleanup.

See :mod:`xpcsviewer.constants.timeouts` for complete API documentation.

Limits
~~~~~~

Size and count limits for caching, memory management, and UI elements.

See :mod:`xpcsviewer.constants.limits` for complete API documentation.

Defaults
~~~~~~~~

Default configuration values for application settings.

See :mod:`xpcsviewer.constants.defaults` for complete API documentation.

Fitting
~~~~~~~

Parameters for curve fitting operations (single/double exponential).

See :mod:`xpcsviewer.constants.fitting` for complete API documentation.
