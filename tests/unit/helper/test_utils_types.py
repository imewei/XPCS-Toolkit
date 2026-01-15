"""Tests for type annotations in helper/utils.py (Task T064).

This test verifies that all functions in helper/utils.py have complete type annotations.
"""

import ast

import numpy as np
import pytest

from xpcsviewer.helper.utils import create_slice, get_min_max, norm_saxs_data


class TestTypeAnnotations:
    """Verify helper/utils.py functions have complete type annotations."""

    def test_all_functions_typed(self):
        """T064: Verify all functions have return and parameter type annotations."""
        import xpcsviewer.helper.utils as utils_module

        with open(utils_module.__file__) as f:
            source = f.read()

        tree = ast.parse(source)
        functions = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]

        for func in functions:
            # Check return annotation
            assert func.returns is not None, f"{func.name} missing return type annotation"

            # Check all args have annotations (except self and **kwargs)
            for arg in func.args.args:
                if arg.arg not in ("self",):
                    assert arg.annotation is not None, (
                        f"{func.name} parameter '{arg.arg}' missing type annotation"
                    )

    def test_get_min_max_signature(self):
        """T060: Verify get_min_max has correct type signature."""
        import inspect

        sig = inspect.signature(get_min_max)

        # Check return annotation
        assert sig.return_annotation == tuple[float, float]

        # Check parameter annotations
        params = sig.parameters
        assert "data" in params
        assert "min_percent" in params
        assert "max_percent" in params

    def test_norm_saxs_data_signature(self):
        """T061: Verify norm_saxs_data has correct type signature."""
        import inspect

        sig = inspect.signature(norm_saxs_data)

        # Check return annotation includes tuple
        assert "tuple" in str(sig.return_annotation).lower()

        # Check parameter annotations
        params = sig.parameters
        assert "Iq" in params
        assert "q" in params
        assert "plot_norm" in params

    def test_create_slice_signature(self):
        """T062: Verify create_slice has correct type signature."""
        import inspect

        sig = inspect.signature(create_slice)

        # Check return annotation
        assert sig.return_annotation == slice

        # Check parameter annotations
        params = sig.parameters
        assert "arr" in params
        assert "x_range" in params


class TestFunctionBehavior:
    """Test that typed functions work correctly."""

    def test_get_min_max_returns_tuple(self):
        """Verify get_min_max returns a tuple of floats."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = get_min_max(data)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_norm_saxs_data_returns_tuple(self):
        """Verify norm_saxs_data returns correct tuple."""
        Iq = np.array([1.0, 2.0, 3.0])
        q = np.array([0.1, 0.2, 0.3])
        result = norm_saxs_data(Iq, q)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[1], str)  # xlabel
        assert isinstance(result[2], str)  # ylabel

    def test_create_slice_returns_slice(self):
        """Verify create_slice returns a slice object."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = create_slice(arr, (2.0, 4.0))

        assert isinstance(result, slice)
