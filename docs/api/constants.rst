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

.. automodule:: xpcsviewer.constants.thresholds
   :members:
   :no-index:

Timeouts
~~~~~~~~

Time-related constants for operations, file loading, and cleanup.
All values are in seconds unless otherwise noted.

.. automodule:: xpcsviewer.constants.timeouts
   :members:
   :no-index:

Limits
~~~~~~

Size and count limits for caching, memory management, and UI elements.

.. automodule:: xpcsviewer.constants.limits
   :members:
   :no-index:

Defaults
~~~~~~~~

Default configuration values for application settings.

.. automodule:: xpcsviewer.constants.defaults
   :members:
   :no-index:

Fitting
~~~~~~~

Parameters for curve fitting operations (single/double exponential).

.. automodule:: xpcsviewer.constants.fitting
   :members:
   :no-index:
