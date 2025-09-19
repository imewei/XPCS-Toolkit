"""
Algorithm-specific validation tests

This module contains comprehensive validation tests for each scientific algorithm
implemented in the XPCS Toolkit. Each algorithm is tested for:

1. Mathematical correctness
2. Physical validity
3. Numerical stability
4. Edge case handling
5. Performance characteristics

Available modules:
- test_fitting_algorithms: Fitting algorithm validation

Note: G2, SAXS, and two-time analysis tests have been moved to tests/analysis/
"""

# Only import modules that exist in this directory
from .test_fitting_algorithms import *
