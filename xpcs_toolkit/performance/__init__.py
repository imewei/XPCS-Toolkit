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

from .benchmark_database import (
    BenchmarkDatabase,
    PerformanceRecord,
    SystemConfiguration,
)
from .cpu_performance_test_suite import CPUPerformanceTestSuite, CPUTestConfig
from .multicore_test_automation import MultiCoreTestAutomation, MultiCoreTestConfig
from .regression_detector import RegressionAnalysisConfig, RegressionDetector

__all__ = [
    "BenchmarkDatabase",
    "CPUPerformanceTestSuite",
    "CPUTestConfig",
    "MultiCoreTestAutomation",
    "MultiCoreTestConfig",
    "PerformanceRecord",
    "RegressionAnalysisConfig",
    "RegressionDetector",
    "SystemConfiguration",
]
