"""Size and count limit constants for xpcsviewer.

These constants define maximum sizes and counts for various operations.
"""

# File list limits
MAX_FILES_IN_LIST = 1000
MAX_RECENT_PATHS = 10

# Cache limits
MAX_CACHE_SIZE_MB = 500
MIN_CACHE_ENTRIES = 50

# Thread pool limits
MAX_THREAD_POOL_SIZE = 8

# Display limits
MAX_PLOT_POINTS = 10000
MAX_LEGEND_ITEMS = 20
MIN_DOWNSAMPLE_POINTS = 50
MIN_DISPLAY_POINTS = 10
MIN_AVERAGING_FILES = 2  # Minimum files required for averaging operation

# Memory cleanup limits
MEMORY_CLEANUP_TIMEOUT_S = 0.050  # 50ms timeout for memory cleanup loops

# Dataset size limits (MB)
LARGE_DATASET_THRESHOLD_MB = 10
DIRECT_READ_LIMIT_MB = 100
MEMORY_WARNING_THRESHOLD_MB = 200
CRITICAL_MEMORY_THRESHOLD_MB = 500
STREAMING_CHUNK_SIZE_MB = 20

# History/learning limits
MIN_HISTORY_SAMPLES = 3
MIN_LEARNING_SAMPLES = 10
MAX_HISTORY_ENTRIES = 100

# Visualization limits
POINTS_THRESHOLD_HIGH = 1000000  # 1e6
POINTS_THRESHOLD_MEDIUM = 100000  # 1e5

# Array dimension constants
NDIM_2D = 2
NDIM_3D = 3
