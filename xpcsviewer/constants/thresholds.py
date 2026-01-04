"""Numeric threshold constants for xpcsviewer.

These constants define comparison thresholds for data validation and analysis.
"""

# Q-value range thresholds (inverse Angstroms)
MIN_Q_VALUE = 0.001
MAX_Q_VALUE = 10.0

# Convergence thresholds
G2_CONVERGENCE_THRESHOLD = 1e-6

# Intensity thresholds
INTENSITY_FLOOR = 1e-10

# Data validation thresholds
MIN_FRAMES = 2
MIN_Q_POINTS = 1

# Memory pressure thresholds (percentage of total memory)
MEMORY_PRESSURE_CRITICAL = 0.9
MEMORY_PRESSURE_HIGH = 0.8
MEMORY_PRESSURE_MODERATE = 0.65

# Downsampling/decimation thresholds
DECIMATION_FACTOR_THRESHOLD = 1.5
MIN_DECIMATION_DIM = 2
MAX_DECIMATION_DIM = 3

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.3
