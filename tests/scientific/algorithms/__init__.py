"""
Algorithm-specific validation tests

This module contains comprehensive validation tests for each scientific algorithm
implemented in the XPCS Toolkit. Each algorithm is tested for:

1. Mathematical correctness
2. Physical validity
3. Numerical stability
4. Edge case handling
5. Performance characteristics

Modules:
- test_g2_analysis: G2 correlation function analysis validation
- test_saxs_analysis: SAXS scattering analysis validation
- test_twotime_analysis: Two-time correlation analysis validation
- test_diffusion_analysis: Tau-Q diffusion analysis validation
- test_fitting_algorithms: Fitting algorithm validation
"""

from .test_diffusion_analysis import *
from .test_fitting_algorithms import *

# Import all algorithm test modules
from .test_g2_analysis import *
from .test_saxs_analysis import *
from .test_twotime_analysis import *
