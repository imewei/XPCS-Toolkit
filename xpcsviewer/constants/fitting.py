"""Fitting and model parameter constants for xpcsviewer.

These constants define parameters for fitting algorithms and models.
"""

# Single/double exponential fitting parameters
SINGLE_EXP_PARAMS = 4
DOUBLE_EXP_PARAMS = 7

# Fitting bounds and quality
PARALLEL_FITTING_THRESHOLD = 50
BATCH_SIZE_LARGE = 500
BATCH_SIZE_SMALL = 200
WORKER_THRESHOLD_LARGE = 1000
WORKER_THRESHOLD_MEDIUM = 500

# Memory efficiency thresholds for fitting
MEMORY_EFFICIENCY_HIGH = 0.85
MEMORY_EFFICIENCY_MEDIUM = 0.75
MEMORY_EFFICIENCY_LOW = 0.70

# FFT and processing sizes
FFT_DEFAULT_SIZE = 4096
MAX_PARALLEL_WORKERS = 800
MIN_PARALLEL_BATCH = 400

# Retry and timeout limits
MAX_RETRY_ATTEMPTS = 5
