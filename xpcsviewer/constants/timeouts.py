"""Timeout constants for xpcsviewer operations.

These constants define time limits for various operations.
All values are in seconds unless otherwise noted.
"""

# File operation timeouts (seconds)
FILE_LOAD_TIMEOUT = 30
HDF5_CONNECTION_TIMEOUT = 5

# UI operation timeouts (seconds)
PLOT_RENDER_TIMEOUT = 10
PROGRESS_DIALOG_DELAY = 0.5  # Delay before showing progress dialog

# Cache expiration timeouts (seconds)
CACHE_ENTRY_TIMEOUT = 300  # 5 minutes
CACHE_CLEANUP_INTERVAL = 900  # 15 minutes
CACHE_LONG_EXPIRY = 1800  # 30 minutes
