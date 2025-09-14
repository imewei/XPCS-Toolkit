"""
Mathematical Properties and Invariants Module

This module defines mathematical properties and invariants that scientific
algorithms must satisfy. These are used by property-based tests to ensure
correctness across parameter ranges.

Modules:
- mathematical_invariants: Core mathematical properties for XPCS algorithms
- statistical_properties: Statistical properties and constraints
- physical_constraints: Physical laws and constraints validation
"""

# Import all property modules
from .mathematical_invariants import *
from .physical_constraints import *
from .statistical_properties import *
