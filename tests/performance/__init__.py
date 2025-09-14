"""
Performance testing and benchmarking suite for XPCS Toolkit.

This package provides comprehensive performance benchmarking, regression detection,
and scalability testing for all critical components of XPCS Toolkit.

Key Features:
- HDF5 I/O performance benchmarking
- Scientific analysis algorithm benchmarks
- Memory management and threading performance tests
- GUI performance monitoring
- Regression detection with CI/CD integration
- Scalability testing for large datasets
- Performance optimization validation

Usage:
    # Run all performance benchmarks
    pytest tests/performance/ --benchmark-only

    # Run with baseline comparison
    pytest tests/performance/ --benchmark-only --benchmark-compare=baseline

    # Generate detailed reports
    pytest tests/performance/ --benchmark-only --benchmark-save=results

    # Memory profiling
    pytest tests/performance/ --profile-mem

    # Scalability tests
    pytest tests/performance/test_scalability.py -k large_dataset
"""

__version__ = "1.0.0"
__author__ = "XPCS Toolkit Performance Team"

# Performance test configuration
PERFORMANCE_CONFIG = {
    "default_rounds": 10,  # Number of benchmark rounds
    "warmup_rounds": 3,  # Number of warmup rounds
    "max_time": 30.0,  # Maximum time per benchmark (seconds)
    "min_rounds": 5,  # Minimum rounds for statistical significance
    "memory_threshold_mb": 1024,  # Memory usage threshold for regression
    "performance_threshold": 0.2,  # 20% performance degradation threshold
    "statistical_confidence": 0.95,  # Confidence level for statistical tests
}

# Test data sizes for scalability testing
DATA_SIZES = {
    "tiny": {"width": 64, "height": 64, "frames": 100},
    "small": {"width": 128, "height": 128, "frames": 500},
    "medium": {"width": 256, "height": 256, "frames": 1000},
    "large": {"width": 512, "height": 512, "frames": 2000},
    "xlarge": {"width": 1024, "height": 1024, "frames": 5000},
}

# Benchmark categories
BENCHMARK_CATEGORIES = [
    "io_operations",  # File I/O, HDF5 operations
    "scientific_analysis",  # G2, SAXS, two-time analysis
    "memory_operations",  # Caching, memory management
    "threading_operations",  # Async workers, parallel processing
    "gui_operations",  # Widget rendering, plotting
    "integration",  # End-to-end workflows
    "scalability",  # Performance vs data size
]
