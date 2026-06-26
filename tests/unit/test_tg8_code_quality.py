"""Tests for Task Group 8: Code Quality and Cleanup.

Covers BUG-035 through BUG-062 (P2 improvements).
"""

from __future__ import annotations

import math
import pathlib
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Derive project root from this test file's location
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Test 1: logger.exception used instead of traceback.print_exc (BUG-035)
# ---------------------------------------------------------------------------


def test_xpcs_viewer_no_traceback_print_exc():
    """Verify xpcs_viewer.py uses logger.exception() not traceback.print_exc()."""
    import ast

    viewer_path = _PROJECT_ROOT / "xpcsviewer" / "xpcs_viewer.py"
    source = viewer_path.read_text()
    tree = ast.parse(source)

    # Collect all Call nodes that invoke traceback.print_exc
    class PrintExcFinder(ast.NodeVisitor):
        def __init__(self):
            self.found = []

        def visit_Call(self, node):
            # Match: traceback.print_exc(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "print_exc"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "traceback"
            ):
                self.found.append(node.lineno)
            self.generic_visit(node)

    finder = PrintExcFinder()
    finder.visit(tree)

    assert finder.found == [], (
        f"traceback.print_exc() still present at lines: {finder.found}. "
        "Replace with logger.exception()."
    )


# ---------------------------------------------------------------------------
# Test 2: NLSQResult property setters warn when delegation is active (BUG-039)
# ---------------------------------------------------------------------------


def test_nlsq_result_setter_warns_when_native_result_active():
    """NLSQResult property setters should log a warning when native_result exists."""
    from xpcsviewer.fitting.results import NLSQResult

    # Build a mock native_result with required properties
    mock_native = MagicMock()
    mock_native.r_squared = 0.99
    mock_native.adj_r_squared = 0.985
    mock_native.rmse = 0.01
    mock_native.mae = 0.008
    mock_native.aic = -100.0
    mock_native.bic = -95.0
    mock_native.diagnostics = None

    result = NLSQResult(
        params={"tau": 1.0},
        chi_squared=1.0,
        converged=True,
        native_result=mock_native,
    )

    # When delegation is active, setters should log a warning
    import logging

    with patch("xpcsviewer.fitting.results.logger") as mock_logger:
        result.r_squared = 0.5  # This should warn since native_result is active
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert (
            "delegation" in warning_msg.lower()
            or "native_result" in warning_msg.lower()
            or "no-op" in warning_msg.lower()
        ), (
            f"Warning message should mention delegation/native_result/no-op: {warning_msg}"
        )

    # After warning, the internal _r_squared should not be changed (getter still returns
    # native_result.r_squared)
    assert result.r_squared == 0.99  # native_result value persists


# ---------------------------------------------------------------------------
# Test 3: is_healthy uses enum/attribute check not string comparison (BUG-040)
# ---------------------------------------------------------------------------


def test_is_healthy_does_not_use_string_comparison():
    """is_healthy should not use str(status) == 'healthy' fragile string check."""
    import ast

    results_path = _PROJECT_ROOT / "xpcsviewer" / "fitting" / "results.py"
    source = results_path.read_text()

    # Check the string comparison pattern is not present
    assert 'str(self.diagnostics.status) == "healthy"' not in source, (
        "Fragile string comparison still present in is_healthy. "
        "Replace with enum or attribute check."
    )


# ---------------------------------------------------------------------------
# Test 4-6: Removed unused backends/_device.py and backends/io_adapter.py tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Additional targeted tests for other BUGs in TG8
# ---------------------------------------------------------------------------


def test_qmap_schema_to_dict_returns_defensive_copies(tmp_path):
    """QMapSchema.to_dict() should return copies of internal arrays (BUG-047)."""
    from xpcsviewer.schemas.validators import QMapSchema

    sqmap = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dqmap = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    phis = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64)

    schema = QMapSchema(
        sqmap=sqmap,
        dqmap=dqmap,
        phis=phis,
        sqmap_unit="nm^-1",
        dqmap_unit="nm^-1",
        phis_unit="rad",
    )
    d = schema.to_dict()

    # Mutating the dict's arrays should NOT affect the schema's internal arrays
    d["sqmap"][0, 0] = 999.0
    assert schema.sqmap[0, 0] != 999.0, (
        "to_dict() leaked internal array reference. Use np.copy()."
    )


def test_geometry_metadata_from_dict_rejects_nan():
    """GeometryMetadata.from_dict() should reject NaN values in critical fields (BUG-048)."""
    from xpcsviewer.schemas.validators import GeometryMetadata

    data = {
        "bcx": float("nan"),
        "bcy": 100.0,
        "det_dist": 1000.0,
        "lambda_": 1.54,
        "pix_dim": 0.075,
        "shape": (512, 512),
    }
    with pytest.raises((ValueError, AssertionError)):
        GeometryMetadata.from_dict(data)


def test_geometry_metadata_from_dict_rejects_inf():
    """GeometryMetadata.from_dict() should reject inf values (BUG-048)."""
    from xpcsviewer.schemas.validators import GeometryMetadata

    data = {
        "bcx": 100.0,
        "bcy": float("inf"),
        "det_dist": 1000.0,
        "lambda_": 1.54,
        "pix_dim": 0.075,
        "shape": (512, 512),
    }
    with pytest.raises((ValueError, AssertionError)):
        GeometryMetadata.from_dict(data)


def test_g2data_from_dict_coerces_to_float64():
    """G2Data.from_dict() should coerce arrays to float64 (BUG-058)."""
    from xpcsviewer.schemas.validators import G2Data

    data = {
        "g2": np.ones((10, 5), dtype=np.float32),
        "g2_err": np.ones((10, 5), dtype=np.float32) * 0.1,
        "delay_times": np.arange(1, 11, dtype=np.float32),
        "q_values": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    g2 = G2Data.from_dict(data)
    assert g2.g2.dtype == np.float64, f"Expected float64, got {g2.g2.dtype}"
    assert g2.g2_err.dtype == np.float64, f"Expected float64, got {g2.g2_err.dtype}"
    assert g2.delay_times.dtype == np.float64, (
        f"Expected float64, got {g2.delay_times.dtype}"
    )


def test_worker_result_validates_execution_time():
    """WorkerResult __post_init__ should reject negative execution_time (BUG-061)."""
    from xpcsviewer.threading.async_workers import WorkerResult

    with pytest.raises((ValueError, AssertionError)):
        WorkerResult(success=True, data=None, error="", execution_time=-1.0)


def test_worker_stats_validates_counts():
    """WorkerStats __post_init__ should reject negative job counts (BUG-061)."""
    from xpcsviewer.threading.async_workers import WorkerStats

    with pytest.raises((ValueError, AssertionError)):
        WorkerStats(
            total_jobs=-1,
            completed_jobs=0,
            failed_jobs=0,
            average_execution_time=0.0,
        )


def test_completed_tasks_bounded():
    """_completed_tasks should have a max-size bound to prevent memory growth (BUG-062)."""
    import ast

    unified_path = _PROJECT_ROOT / "xpcsviewer" / "threading" / "unified_threading.py"
    source = unified_path.read_text()

    # Check that there's a size check/bound on _completed_tasks
    assert "_completed_tasks" in source, "_completed_tasks not found"
    # Verify there's a max-size enforcement (len check or similar)
    assert "len(self._completed_tasks)" in source or "maxlen" in source, (
        "_completed_tasks has no size bound. Add periodic cleanup or maxlen."
    )
