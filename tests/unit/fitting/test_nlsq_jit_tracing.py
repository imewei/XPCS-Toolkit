"""Regression tests for nlsq JIT-tracing compatibility.

Verifies that model functions in xpcs_file/fitting.py use jax.numpy
operations (not numpy) so they remain compatible when nlsq JIT-traces
them with JAX tracers.

Root cause: np.exp() calls __array__() on JAX tracers, raising
TracerArrayConversionError. Using jnp.exp() avoids this because
jnp operations are JAX-native and handle tracers correctly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from xpcsviewer.xpcs_file.fitting import (
    double_exp_all,
    power_law,
    single_exp_all,
)

try:
    from nlsq import curve_fit

    HAS_NLSQ = True
except ImportError:
    HAS_NLSQ = False

needs_nlsq = pytest.mark.skipif(not HAS_NLSQ, reason="nlsq not installed")


class TestModelFunctionsUseJaxNumpy:
    """Model functions must use jnp.exp, not np.exp."""

    def test_single_exp_all_uses_jnp_exp(self):
        source = inspect.getsource(single_exp_all)
        assert "jnp.exp" in source, "single_exp_all must use jnp.exp, not np.exp"

    def test_double_exp_all_uses_jnp_exp(self):
        source = inspect.getsource(double_exp_all)
        assert "jnp.exp" in source, "double_exp_all must use jnp.exp, not np.exp"


class TestNoNumpyMathInModelFile:
    """Lint rule: xpcs_file/fitting.py must not use numpy math functions.

    Model functions are JIT-traced by nlsq with JAX tracers, so any
    numpy math call (np.exp, np.log, np.sqrt, etc.) will trigger
    TracerArrayConversionError. This test scans the entire source file
    to catch violations before they reach production.
    """

    # Functions that accept array-like inputs and call __array__()
    # on JAX tracers, causing TracerArrayConversionError.
    BANNED_NP_CALLS = [
        "np.exp",
        "np.log",
        "np.log10",
        "np.log2",
        "np.sqrt",
        "np.sin",
        "np.cos",
        "np.tan",
        "np.abs",
        "np.power",
        "np.square",
    ]

    def test_no_numpy_math_in_fitting_module(self):
        """xpcs_file/fitting.py must not contain numpy math calls."""
        from pathlib import Path

        fitting_path = (
            Path(__file__).parent.parent.parent.parent
            / "xpcsviewer"
            / "xpcs_file"
            / "fitting.py"
        )
        source = fitting_path.read_text()

        # Strip comments to avoid false positives on explanatory text
        import re

        source_no_comments = re.sub(r"#.*$", "", source, flags=re.MULTILINE)

        violations = []
        for banned in self.BANNED_NP_CALLS:
            # Match the call as a token (not inside a string or jnp.exp)
            pattern = rf"(?<![a-zA-Z_]){re.escape(banned)}\s*\("
            if re.search(pattern, source_no_comments):
                violations.append(banned)

        assert not violations, (
            f"xpcs_file/fitting.py contains numpy math calls that break "
            f"JIT tracing: {violations}. Use jnp equivalents instead."
        )


class TestNlsqJitTracing:
    """Model functions must survive nlsq JIT-tracing without TracerArrayConversionError."""

    @needs_nlsq
    def test_single_exp_all_curve_fit(self):
        """single_exp_all must work with nlsq.curve_fit (JIT-traced)."""
        x = np.linspace(0.001, 10.0, 48)
        y = 0.3 * np.exp(-2 * (x / 1.5) ** 1.0) + 1.0
        bounds = ([0.0, 0.001, 0.5, 0.8], [1.0, 100.0, 2.0, 1.5])

        popt, pcov = curve_fit(
            single_exp_all, x, y, p0=[0.2, 1.0, 1.0, 1.0], bounds=bounds
        )
        assert popt is not None
        assert len(popt) == 4
        # Recovered tau should be close to true value
        assert abs(popt[1] - 1.5) / 1.5 < 0.1

    @needs_nlsq
    def test_double_exp_all_curve_fit(self):
        """double_exp_all must work with nlsq.curve_fit (JIT-traced)."""
        x = np.linspace(0.001, 10.0, 48)
        f = 0.6
        t1 = np.exp(-((x / 1.0) ** 1.0)) * f
        t2 = np.exp(-((x / 5.0) ** 1.5)) * (1 - f)
        y = 0.3 * (t1 + t2) ** 2 + 1.0
        bounds = (
            [0.0, 0.001, 0.5, 0.8, 0.001, 0.5, 0.0],
            [1.0, 100.0, 2.0, 1.5, 100.0, 2.0, 1.0],
        )

        popt, pcov = curve_fit(
            double_exp_all,
            x,
            y,
            p0=[0.2, 1.0, 1.0, 1.0, 5.0, 1.5, 0.5],
            bounds=bounds,
        )
        assert popt is not None
        assert len(popt) == 7

    @needs_nlsq
    def test_power_law_curve_fit(self):
        """power_law must work with nlsq.curve_fit (JIT-traced)."""
        x = np.linspace(0.01, 1.0, 30)
        y = 1e-7 * x ** (-2.0)
        bounds = ([1e-15, -4.0], [1e-2, 0.0])

        popt, pcov = curve_fit(power_law, x, y, p0=[1e-7, -2.0], bounds=bounds)
        assert popt is not None
        assert len(popt) == 2
        assert abs(popt[1] - (-2.0)) < 0.1
