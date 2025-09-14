"""
Reference Validation and Cross-Validation Framework

This module provides cross-validation capabilities for XPCS algorithms against
reference implementations, analytical solutions, and established benchmarks.
"""

from .analytical_benchmarks import *

# Import cross-validation modules
from .cross_validation_framework import *
from .reference_implementations import *
