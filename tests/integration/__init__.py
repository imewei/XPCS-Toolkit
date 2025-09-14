"""
Integration Testing Package for XPCS Toolkit

This package contains comprehensive integration tests that validate the interaction
between different components of the XPCS Toolkit. These tests ensure that the
various modules, optimization phases, and systems work correctly together.

Test Categories:
- Component Integration: Cross-module interactions
- Data Pipeline Integration: File I/O → Data processing → Analysis → Results
- Memory Management Integration: Caching systems working with analysis modules
- Threading Integration: Async workers + GUI + analysis coordination
- Performance System Integration: All optimization phases working together
- Error Handling Integration: Error propagation through analysis chains
- Resource Management Integration: Memory, CPU, and I/O resource handling

Each integration test validates that:
1. Components work correctly when combined
2. Data flows properly through the entire system
3. Performance optimizations function as expected
4. Error handling is robust across component boundaries
5. Resource management is effective throughout workflows
6. Scientific accuracy is maintained in integrated operations

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import logging
import sys
from pathlib import Path

# Configure logging for integration tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add XPCS Toolkit to path if needed
toolkit_path = Path(__file__).parent.parent.parent / "xpcs_toolkit"
if toolkit_path.exists() and str(toolkit_path) not in sys.path:
    sys.path.insert(0, str(toolkit_path))

__version__ = "1.0.0"
__author__ = "Integration and Workflow Tester Agent"
