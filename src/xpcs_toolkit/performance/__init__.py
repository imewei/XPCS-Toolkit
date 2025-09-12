"""
XPCS Toolkit Performance Testing Framework

This module provides comprehensive performance testing, benchmarking, and regression
detection capabilities for the XPCS Toolkit.

Components:
- CPU Performance Test Suite: Comprehensive CPU and threading performance tests
- Benchmark Database: Historical performance data storage and analysis
- Regression Detector: Statistical regression detection and CI/CD integration
- Multi-Core Test Automation: Multi-core scaling performance validation
"""

from .cpu_performance_test_suite import CPUPerformanceTestSuite, CPUTestConfig
from .benchmark_database import (
    BenchmarkDatabase,
    PerformanceRecord,
    SystemConfiguration,
)
from .regression_detector import RegressionDetector, RegressionAnalysisConfig
from .multicore_test_automation import MultiCoreTestAutomation, MultiCoreTestConfig

__all__ = [
    "CPUPerformanceTestSuite",
    "CPUTestConfig",
    "BenchmarkDatabase",
    "PerformanceRecord",
    "SystemConfiguration",
    "RegressionDetector",
    "RegressionAnalysisConfig",
    "MultiCoreTestAutomation",
    "MultiCoreTestConfig",
]
