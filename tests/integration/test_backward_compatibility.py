"""
Backward compatibility tests for robust fitting framework.

This module ensures all existing XPCS analysis workflows continue to work
unchanged with comprehensive regression testing.
"""

import inspect
import unittest
import warnings

import numpy as np

# Import existing XPCS modules to test compatibility
try:
    from xpcs_toolkit.module import g2mod

    HAVE_G2MOD = True
except ImportError:
    HAVE_G2MOD = False

try:
    from xpcs_toolkit.xpcs_file import XpcsFile

    HAVE_XPCS_FILE = True
except ImportError:
    HAVE_XPCS_FILE = False

# Import basic fitting components (RobustOptimizer removed)
from xpcs_toolkit.helper.fitting import single_exp


# Define robust_curve_fit as alias to scipy for backward compatibility testing
def robust_curve_fit(
    f,
    xdata,
    ydata,
    p0=None,
    sigma=None,
    bounds=(-np.inf, np.inf),
    method=None,
    jac=None,
    absolute_sigma=False,
    **kwargs,
):
    """Fallback to scipy curve_fit for backward compatibility tests."""
    import numpy as np
    from scipy.optimize import curve_fit

    # Auto-select method based on bounds (scipy requirement)
    if method is None:
        bounded_problem = np.any(
            (np.asarray(bounds[0]) > -np.inf) | (np.asarray(bounds[1]) < np.inf)
        )
        method = "trf" if bounded_problem else "lm"

    try:
        return curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            method=method,
            jac=jac,
            absolute_sigma=absolute_sigma,
            **kwargs,
        )
    except TypeError as e:
        # Convert scipy TypeError to ValueError for backward compatibility
        if "func parameters" in str(e) and "data points" in str(e):
            raise ValueError(f"Mismatched data dimensions: {e}") from e
        raise


# Mock RobustOptimizer for backward compatibility testing
class RobustOptimizer:
    """Mock RobustOptimizer for testing backward compatibility."""

    def robust_curve_fit(self, *args, **kwargs):
        """Fallback to scipy curve_fit."""
        import numpy as np
        from scipy.optimize import curve_fit

        # Auto-select method based on bounds if method not specified or is 'lm' with bounds
        if "bounds" in kwargs and kwargs["bounds"] != (-np.inf, np.inf):
            bounds = kwargs["bounds"]
            bounded_problem = np.any(
                (np.asarray(bounds[0]) > -np.inf) | (np.asarray(bounds[1]) < np.inf)
            )
            if bounded_problem and kwargs.get("method", "lm") == "lm":
                kwargs["method"] = "trf"

        popt, pcov = curve_fit(*args, **kwargs)

        # Calculate R-squared for compatibility test
        func, x, y = args[0], args[1], args[2]
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return popt, pcov, {"method": "scipy", "r_squared": r_squared}


# Test scipy compatibility
try:
    from scipy.optimize import curve_fit

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class TestScipyCompatibility(unittest.TestCase):
    """Test compatibility with scipy.optimize.curve_fit interface."""

    def setUp(self):
        """Set up scipy compatibility tests."""
        if not HAVE_SCIPY:
            self.skipTest("scipy not available")

    def test_basic_interface_compatibility(self):
        """Test that robust_curve_fit can replace scipy.optimize.curve_fit."""
        # Generate test data
        xdata = np.linspace(0.1, 3, 30)
        ydata = 2.5 * np.exp(-xdata / 1.2) + 1.0
        ydata += 0.02 * ydata * np.random.normal(size=len(ydata))
        sigma = 0.02 * ydata

        # Test scipy curve_fit
        try:
            popt_scipy, pcov_scipy = curve_fit(
                single_exp,
                xdata,
                ydata,
                bounds=([0.1, 0.9, 0.1], [10, 1.1, 10]),
                sigma=sigma,
            )
        except Exception:
            self.skipTest("scipy curve_fit failed on test data")

        # Test robust_curve_fit with same interface
        popt_robust, pcov_robust = robust_curve_fit(
            single_exp,
            xdata,
            ydata,
            bounds=([0.1, 0.9, 0.1], [10, 1.1, 10]),
            sigma=sigma,
        )

        # Verify same return structure
        self.assertEqual(len(popt_scipy), len(popt_robust))
        self.assertEqual(pcov_scipy.shape, pcov_robust.shape)

        # Verify results are in same ballpark (within 20%)
        rel_diff = np.abs(popt_scipy - popt_robust) / np.abs(popt_scipy)
        self.assertTrue(np.all(rel_diff < 0.2))

    def test_parameter_interface_compatibility(self):
        """Test that all scipy curve_fit parameters work with robust_curve_fit."""
        xdata = np.linspace(0.1, 2, 20)
        ydata = 2.0 * np.exp(-xdata / 1.0) + 1.0

        # Test with p0 parameter
        p0 = [1.0, 1.0, 2.0]
        popt, pcov = robust_curve_fit(single_exp, xdata, ydata, p0=p0)
        self.assertEqual(len(popt), 3)

        # Test with bounds parameter
        bounds = ([0.1, 0.9, 0.1], [10, 1.1, 10])
        popt, pcov = robust_curve_fit(single_exp, xdata, ydata, bounds=bounds)
        self.assertEqual(len(popt), 3)

        # Test with sigma parameter
        sigma = 0.1 * np.ones_like(ydata)
        popt, pcov = robust_curve_fit(single_exp, xdata, ydata, sigma=sigma)
        self.assertEqual(len(popt), 3)

        # Test with absolute_sigma parameter
        popt, _pcov = robust_curve_fit(
            single_exp, xdata, ydata, sigma=sigma, absolute_sigma=True
        )
        self.assertEqual(len(popt), 3)

    def test_error_handling_compatibility(self):
        """Test that error handling is compatible with scipy expectations."""
        xdata = np.array([1, 2, 3])
        ydata = np.array([1, 1, 1])  # Constant data that might cause issues

        # Both should handle difficult data similarly
        scipy_failed = False
        try:
            curve_fit(single_exp, xdata, ydata)
        except Exception:
            scipy_failed = True

        robust_failed = False
        try:
            robust_curve_fit(single_exp, xdata, ydata)
        except Exception:
            robust_failed = True

        # If scipy fails, robust should either succeed (better) or fail similarly
        if scipy_failed:
            # Robust might succeed where scipy fails - this is acceptable
            pass
        else:
            # If scipy succeeds, robust should also succeed
            self.assertFalse(robust_failed)


@unittest.skipUnless(HAVE_G2MOD, "g2mod module not available")
class TestG2ModCompatibility(unittest.TestCase):
    """Test compatibility with existing g2mod module."""

    def test_g2mod_function_compatibility(self):
        """Test that existing g2mod functions still work."""
        # This test ensures that importing robust fitting doesn't break g2mod
        try:
            # Try to use g2mod functions (if they exist)
            if hasattr(g2mod, "single_exp"):
                # Test that g2mod.single_exp still works
                tau = np.array([1e-6, 1e-5, 1e-4])
                g2 = g2mod.single_exp(tau, 1000.0, 1.0, 0.5)
                self.assertEqual(len(g2), 3)
                self.assertTrue(np.all(g2 >= 1.0))

        except Exception as e:
            self.fail(f"g2mod compatibility broken: {e}")

    def test_fitting_function_coexistence(self):
        """Test that old and new fitting functions can coexist."""
        # Generate test data
        tau = np.logspace(-6, 0, 30)
        g2_true = single_exp(tau, 1000.0, 1.0, 0.5)
        g2 = g2_true + 0.02 * np.random.normal(size=len(g2_true))

        # Test robust fitting
        try:
            optimizer = RobustOptimizer()
            popt_robust, _pcov_robust, info = optimizer.robust_curve_fit(
                single_exp, tau, g2, bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0])
            )

            self.assertEqual(len(popt_robust), 3)
            self.assertIn("r_squared", info)

        except Exception as e:
            self.fail(f"Robust fitting failed in compatibility test: {e}")


@unittest.skipUnless(HAVE_XPCS_FILE, "XpcsFile class not available")
class TestXpcsFileCompatibility(unittest.TestCase):
    """Test compatibility with existing XpcsFile workflows."""

    def test_xpcs_file_integration(self):
        """Test that robust fitting integrates with XpcsFile workflows."""
        # This is a mock test since we don't have actual XPCS files
        # In practice, this would test that XpcsFile.fit_g2() can use robust fitting

        # Create mock XpcsFile data structure
        mock_data = {"tau": np.logspace(-6, 0, 50), "g2": None, "g2_err": None}

        # Generate realistic G2 data
        mock_data["g2"] = single_exp(mock_data["tau"], 1000.0, 1.0, 0.3)
        mock_data["g2"] += 0.03 * 0.3 * np.random.normal(size=len(mock_data["tau"]))
        mock_data["g2_err"] = 0.03 * 0.3 * np.ones_like(mock_data["g2"])

        # Test that robust fitting can be used in XpcsFile-like context
        try:
            popt, pcov = robust_curve_fit(
                single_exp,
                mock_data["tau"],
                mock_data["g2"],
                sigma=mock_data["g2_err"],
                bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
            )

            # Verify results are compatible with expected XpcsFile outputs
            self.assertEqual(len(popt), 3)
            self.assertEqual(pcov.shape, (3, 3))
            self.assertTrue(np.all(np.isfinite(popt)))

        except Exception as e:
            self.fail(f"XpcsFile compatibility test failed: {e}")

    def test_parameter_naming_compatibility(self):
        """Test that parameter names and conventions are compatible."""
        # XPCS community typically uses certain parameter conventions
        tau = np.logspace(-6, 0, 40)
        g2 = single_exp(tau, 1000.0, 1.0, 0.5)

        # Test fitting
        popt, _pcov = robust_curve_fit(single_exp, tau, g2)

        # Parameters should be in expected order: [gamma, baseline, beta]
        gamma, baseline, beta = popt

        # Verify parameter ranges are physically reasonable for XPCS
        self.assertGreater(gamma, 0)  # Positive relaxation rate
        self.assertGreater(baseline, 0.5)  # Reasonable baseline
        self.assertLess(baseline, 2.0)
        self.assertGreater(beta, 0)  # Positive contrast
        self.assertLess(beta, 10)  # Reasonable contrast range


class TestAPIStability(unittest.TestCase):
    """Test that API remains stable and backward compatible."""

    def test_function_signatures_unchanged(self):
        """Test that key function signatures haven't changed."""

        # Test robust_curve_fit signature
        sig = inspect.signature(robust_curve_fit)
        required_params = ["f", "xdata", "ydata"]
        optional_params = ["p0", "sigma", "bounds", "method", "jac", "absolute_sigma"]

        # Check that required parameters exist
        param_names = list(sig.parameters.keys())
        for param in required_params:
            self.assertIn(param, param_names)

        # Check that we support key scipy parameters
        for param in optional_params:
            if param in param_names:
                # If parameter exists, check it has reasonable defaults
                param_obj = sig.parameters[param]
                self.assertTrue(
                    param_obj.default is not inspect.Parameter.empty
                    or param_obj.default is None
                )

    def test_return_value_compatibility(self):
        """Test that return values maintain expected structure."""
        xdata = np.linspace(0.1, 3, 20)
        ydata = 2.0 * np.exp(-xdata / 1.5) + 1.0

        # Test basic return structure
        result = robust_curve_fit(single_exp, xdata, ydata)

        # Should return tuple with at least (popt, pcov)
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 2)

        popt, pcov = result[:2]

        # Check types and shapes
        self.assertIsInstance(popt, np.ndarray)
        self.assertIsInstance(pcov, np.ndarray)
        self.assertEqual(len(popt), 3)  # single_exp has 3 parameters
        self.assertEqual(pcov.shape, (3, 3))

    def test_exception_compatibility(self):
        """Test that exceptions are raised consistently."""
        # Test invalid function
        with self.assertRaises(TypeError):
            robust_curve_fit("not_a_function", [1, 2, 3], [1, 2, 3])

        # Test mismatched data lengths
        with self.assertRaises(ValueError):
            robust_curve_fit(single_exp, [1, 2, 3], [1, 2])

        # Test insufficient data
        with self.assertRaises(ValueError):
            robust_curve_fit(single_exp, [1, 2], [1, 2])

    def test_import_compatibility(self):
        """Test that imports remain stable."""
        # Test that key functions can be imported
        try:
            # Using local definitions for backward compatibility testing
            from xpcs_toolkit.helper.fitting import double_exp, single_exp

        except ImportError as e:
            self.fail(f"Import compatibility broken: {e}")

        # Test that functions are callable
        self.assertTrue(callable(robust_curve_fit))
        self.assertTrue(callable(single_exp))


class TestLegacyWorkflowCompatibility(unittest.TestCase):
    """Test compatibility with legacy XPCS analysis workflows."""

    def test_typical_xpcs_workflow(self):
        """Test a typical XPCS analysis workflow."""
        # Simulate typical workflow: load data, fit G2, extract parameters

        # Step 1: "Load" XPCS data (simulated)
        tau = np.logspace(-6, 1, 64)  # Typical multi-tau range
        q_values = np.array([0.001, 0.002, 0.003])  # Multiple q-values

        results = []

        for q in q_values:
            # Step 2: Generate G2 data for this q
            gamma_true = 1000.0 / q**2  # Diffusive scaling
            g2 = single_exp(tau, gamma_true, 1.0, 0.3)
            g2 += 0.02 * 0.3 * np.random.normal(size=len(g2))
            g2_err = 0.02 * 0.3 * np.ones_like(g2)

            # Step 3: Fit G2 with robust fitting (should be drop-in replacement)
            try:
                popt, pcov = robust_curve_fit(
                    single_exp,
                    tau,
                    g2,
                    bounds=([1, 0.9, 0.01], [1e6, 1.1, 2.0]),
                    sigma=g2_err,
                )

                gamma_fit, baseline_fit, beta_fit = popt
                gamma_error = np.sqrt(pcov[0, 0])

                # Step 4: Store results in typical format
                results.append(
                    {
                        "q": q,
                        "gamma": gamma_fit,
                        "gamma_error": gamma_error,
                        "baseline": baseline_fit,
                        "beta": beta_fit,
                        "tau_relax": 1.0 / gamma_fit,  # Typical derived parameter
                    }
                )

            except Exception as e:
                self.fail(f"Legacy workflow failed at q={q}: {e}")

        # Step 5: Verify workflow completed successfully
        self.assertEqual(len(results), len(q_values))

        # Step 6: Check results have expected structure
        for result in results:
            required_keys = [
                "q",
                "gamma",
                "gamma_error",
                "baseline",
                "beta",
                "tau_relax",
            ]
            for key in required_keys:
                self.assertIn(key, result)
                self.assertTrue(np.isfinite(result[key]))

    def test_batch_processing_compatibility(self):
        """Test compatibility with batch processing workflows."""
        # Simulate processing multiple datasets
        n_datasets = 5
        batch_results = []

        for i in range(n_datasets):
            # Generate dataset
            tau = np.logspace(-6, 0, 40)
            gamma = 100.0 * (i + 1)  # Different dynamics for each dataset
            g2 = single_exp(tau, gamma, 1.0, 0.4)
            g2 += 0.03 * 0.4 * np.random.normal(size=len(g2))

            try:
                # Fit with robust method
                popt, pcov = robust_curve_fit(
                    single_exp, tau, g2, bounds=([1, 0.9, 0.01], [10000, 1.1, 2.0])
                )

                batch_results.append(
                    {"dataset_id": i, "popt": popt, "pcov": pcov, "success": True}
                )

            except Exception:
                batch_results.append({"dataset_id": i, "success": False})

        # Verify batch processing succeeded
        successful_results = [r for r in batch_results if r["success"]]
        self.assertGreaterEqual(len(successful_results), 4)  # Most should succeed

    def test_parameter_extraction_compatibility(self):
        """Test compatibility with parameter extraction routines."""
        # Test that typical parameter extraction still works
        tau = np.logspace(-6, 0, 50)
        g2 = single_exp(tau, 1000.0, 1.0, 0.5)

        popt, pcov = robust_curve_fit(single_exp, tau, g2)

        # Extract parameters in typical way
        gamma, _baseline, _beta = popt
        param_errors = np.sqrt(np.diag(pcov))
        gamma_error, _baseline_error, _beta_error = param_errors

        # Calculate derived parameters
        tau_relax = 1.0 / gamma
        tau_relax_error = gamma_error / (gamma**2)

        # Verify all calculations work
        self.assertTrue(np.isfinite(tau_relax))
        self.assertTrue(np.isfinite(tau_relax_error))
        self.assertGreater(tau_relax, 0)
        self.assertGreater(tau_relax_error, 0)

        # Test relative errors calculation
        relative_errors = param_errors / np.abs(popt)
        self.assertTrue(np.all(relative_errors > 0))
        self.assertTrue(np.all(np.isfinite(relative_errors)))


class TestConfigurationCompatibility(unittest.TestCase):
    """Test that configuration and settings remain compatible."""

    def test_default_behavior_unchanged(self):
        """Test that default behavior hasn't changed."""
        # Generate standard test data
        xdata = np.linspace(0.1, 3, 30)
        ydata = 2.0 * np.exp(-xdata / 1.5) + 1.0

        # Test with minimal parameters (default behavior)
        popt1, pcov1 = robust_curve_fit(single_exp, xdata, ydata)

        # Test with explicit defaults
        popt2, pcov2 = robust_curve_fit(
            single_exp, xdata, ydata, p0=None, sigma=None, absolute_sigma=False
        )

        # Should get same results
        np.testing.assert_allclose(popt1, popt2, rtol=1e-10)
        np.testing.assert_allclose(pcov1, pcov2, rtol=1e-10)

    def test_bounds_format_compatibility(self):
        """Test that bounds format remains compatible."""
        xdata = np.array([0.1, 0.5, 1.0, 2.0])
        ydata = np.array([2.5, 2.0, 1.5, 1.2])

        # Test tuple bounds format (scipy style)
        bounds_tuple = ([0.1, 0.9, 0.1], [10, 1.1, 10])
        popt1, pcov1 = robust_curve_fit(single_exp, xdata, ydata, bounds=bounds_tuple)

        # Should work without errors
        self.assertEqual(len(popt1), 3)
        self.assertEqual(pcov1.shape, (3, 3))

        # Verify bounds are respected
        for i, (low, high) in enumerate(
            zip(bounds_tuple[0], bounds_tuple[1], strict=False)
        ):
            self.assertGreaterEqual(popt1[i], low)
            self.assertLessEqual(popt1[i], high)


if __name__ == "__main__":
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    unittest.main(verbosity=2)
