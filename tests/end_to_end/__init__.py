"""
End-to-End Workflow Tests for XPCS Toolkit

This package contains comprehensive end-to-end workflow tests that validate
complete user scenarios from data loading to analysis completion and result export.

Test Categories:
- Complete XPCS Workflow: File loading → G2 analysis → Fitting → Diffusion analysis → Export
- SAXS Analysis Workflow: Data loading → 1D/2D analysis → Integration → Visualization
- Two-Time Analysis Workflow: Data loading → Two-time correlation → Analysis → Plotting
- Multi-File Analysis: Batch processing, averaging, comparative analysis
- Cross-Platform Validation: Test workflows on different OS configurations
- Large Dataset Handling: Test scalability with realistic data sizes

Each end-to-end test validates:
1. Complete user workflows from start to finish
2. Data integrity throughout the entire analysis chain
3. Performance benchmarks in realistic usage scenarios
4. Error recovery and graceful handling of edge cases
5. Resource management during complete workflows
6. Scientific accuracy in integrated analysis pipelines

These tests simulate actual user interaction patterns and ensure the integrated
system works reliably in practical usage scenarios.

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import logging
import sys
from pathlib import Path

# Configure logging for end-to-end tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add XPCS Toolkit to path if needed
toolkit_path = Path(__file__).parent.parent.parent / "xpcs_toolkit"
if toolkit_path.exists() and str(toolkit_path) not in sys.path:
    sys.path.insert(0, str(toolkit_path))

__version__ = "1.0.0"
__author__ = "Integration and Workflow Tester Agent"
