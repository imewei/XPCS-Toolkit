#!/usr/bin/env python3
"""
Test suite for lazy loading functionality in the XPCS Toolkit.
Tests the lazy loading system implemented in viewer_kernel.py.
"""

import gc
import sys
import tempfile
import threading
import time
import unittest

try:
    from xpcs_toolkit.viewer_kernel import ViewerKernel, _get_module, _module_cache

    LAZY_LOADING_AVAILABLE = True
except ImportError:
    LAZY_LOADING_AVAILABLE = False


@unittest.skipUnless(LAZY_LOADING_AVAILABLE, "Lazy loading modules not available")
class TestLazyLoading(unittest.TestCase):
    """Test suite for lazy loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear module cache before each test
        _module_cache.clear()

    def tearDown(self):
        """Clean up after tests."""
        # Clear module cache after each test
        _module_cache.clear()

    def test_module_cache_initialization(self):
        """Test that module cache is properly initialized."""
        self.assertIsInstance(_module_cache, dict)
        self.assertEqual(len(_module_cache), 0, "Cache should be empty initially")

    def test_get_module_function_exists(self):
        """Test that the _get_module function exists and is callable."""
        self.assertTrue(callable(_get_module))

    def test_lazy_loading_caches_modules(self):
        """Test that lazy loading properly caches modules."""
        # Clear cache first
        _module_cache.clear()

        # Load a module
        module_name = "g2mod"
        module1 = _get_module(module_name)

        # Check that module is now in cache
        self.assertIn(module_name, _module_cache)
        self.assertIs(_module_cache[module_name], module1)

        # Load same module again
        module2 = _get_module(module_name)

        # Should be the same object (cached)
        self.assertIs(module1, module2)

    def test_lazy_loading_different_modules(self):
        """Test lazy loading of different analysis modules."""
        analysis_modules = ["g2mod", "saxs1d", "saxs2d", "stability", "intt", "twotime"]

        loaded_modules = {}
        for module_name in analysis_modules:
            try:
                module = _get_module(module_name)
                loaded_modules[module_name] = module
                self.assertIsNotNone(module, f"Module {module_name} should not be None")
            except ImportError:
                # Skip modules that aren't available in test environment
                continue

        # Check that different modules are different objects
        module_objects = list(loaded_modules.values())
        if len(module_objects) > 1:
            for i, mod1 in enumerate(module_objects):
                for j, mod2 in enumerate(module_objects):
                    if i != j:
                        self.assertIsNot(
                            mod1, mod2, "Different modules should be different objects"
                        )

    def test_invalid_module_name(self):
        """Test behavior with invalid module names."""
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            _get_module("nonexistent_module")

    def test_viewer_kernel_get_module_method(self):
        """Test the ViewerKernel.get_module method."""
        # Create a minimal ViewerKernel instance with a test path
        with tempfile.TemporaryDirectory() as temp_dir:
            vk = ViewerKernel(temp_dir)

            # Test that get_module method exists
            self.assertTrue(hasattr(vk, "get_module"))
            self.assertTrue(callable(vk.get_module))

            # Test loading a module through ViewerKernel
            try:
                module = vk.get_module("g2mod")
                self.assertIsNotNone(module)
            except ImportError:
                # Skip if module not available in test environment
                pass

    def test_lazy_loading_performance(self):
        """Test that lazy loading provides performance benefits."""

        # Clear cache
        _module_cache.clear()

        # Time first load (should be slower)
        start_time = time.perf_counter()
        try:
            _get_module("g2mod")
            first_load_time = time.perf_counter() - start_time

            # Time second load (should be faster due to caching)
            start_time = time.perf_counter()
            _get_module("g2mod")
            second_load_time = time.perf_counter() - start_time

            # Second load should be significantly faster
            self.assertLess(
                second_load_time,
                first_load_time,
                "Cached module loading should be faster than initial load",
            )

            # Should be much faster (at least 10x)
            if first_load_time > 0.001:  # Only test if first load was measurable
                self.assertLess(
                    second_load_time,
                    first_load_time / 10,
                    "Cached loading should be at least 10x faster",
                )

        except ImportError:
            # Skip performance test if module not available
            pass

    def test_memory_efficiency(self):
        """Test that lazy loading doesn't cause memory leaks."""

        # Clear cache and run garbage collection
        _module_cache.clear()
        gc.collect()

        # Get initial reference count for a module
        try:
            module = _get_module("g2mod")
            initial_refs = sys.getrefcount(module)

            # Load same module multiple times
            for _ in range(10):
                _get_module("g2mod")

            # Reference count should not increase significantly
            final_refs = sys.getrefcount(module)

            # Allow for some variance but should not leak significantly
            self.assertLess(
                final_refs - initial_refs,
                5,
                "Multiple module loads should not create significant ref count increase",
            )

        except ImportError:
            # Skip if module not available
            pass

    def test_thread_safety_basic(self):
        """Basic test for thread safety of lazy loading."""

        _module_cache.clear()
        results = []
        errors = []

        def load_module():
            try:
                module = _get_module("g2mod")
                results.append(module)
            except Exception as e:
                errors.append(e)

        # Create multiple threads that load the same module
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=load_module)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout

        # Check results
        if not errors:  # Only check if no import errors
            # All loaded modules should be the same object (properly cached)
            if results:
                first_module = results[0]
                for module in results[1:]:
                    self.assertIs(
                        module,
                        first_module,
                        "All threads should get the same cached module instance",
                    )

    def test_cache_persistence(self):
        """Test that module cache persists across multiple calls."""
        _module_cache.clear()

        # Load multiple modules
        modules_to_test = ["g2mod", "saxs1d"]
        loaded_modules = {}

        for module_name in modules_to_test:
            try:
                module = _get_module(module_name)
                loaded_modules[module_name] = module
            except ImportError:
                continue

        # Check cache contains all loaded modules
        for module_name, module in loaded_modules.items():
            self.assertIn(module_name, _module_cache)
            self.assertIs(_module_cache[module_name], module)

        # Clear a specific module from cache and reload
        if "g2mod" in loaded_modules:
            loaded_modules["g2mod"]
            del _module_cache["g2mod"]

            # Reload should create new instance
            _get_module("g2mod")

            # Should be back in cache
            self.assertIn("g2mod", _module_cache)

    def test_lazy_loading_integration(self):
        """Integration test for lazy loading with viewer kernel."""
        # This test simulates how lazy loading is used in the actual application
        with tempfile.TemporaryDirectory() as temp_dir:
            vk = ViewerKernel(temp_dir)

            # Test that we can access analysis modules through the kernel
            analysis_methods = [
                ("g2mod", "get_data"),
                ("saxs1d", "pg_plot"),
                ("stability", "plot"),  # stability uses 'plot', not 'pg_plot'
            ]

            for module_name, method_name in analysis_methods:
                try:
                    module = vk.get_module(module_name)
                    self.assertTrue(
                        hasattr(module, method_name),
                        f"Module {module_name} should have method {method_name}",
                    )
                except ImportError:
                    # Skip modules not available in test environment
                    continue
                except AttributeError:
                    # Skip if specific method doesn't exist (test environment may differ)
                    continue


if __name__ == "__main__":
    unittest.main()
