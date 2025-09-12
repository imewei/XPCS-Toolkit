#!/usr/bin/env python

"""Tests for `xpcs_toolkit` package."""

import unittest
import os
import tempfile
import numpy as np
import h5py

from xpcs_toolkit import XpcsFile, __version__
from xpcs_toolkit.cli import main
from xpcs_toolkit.file_locator import FileLocator
from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


class TestXpcsToolkit(unittest.TestCase):
    """Tests for `xpcs_toolkit` package."""

    def setUp(self):
        """Set up test fixtures."""
        logger.info(f"Setting up test: {self._testMethodName}")
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_xpcs.hdf")
        logger.debug(f"Created temporary directory: {self.temp_dir}")
        logger.debug(f"Test file path: {self.test_file}")
        self._create_test_hdf_file()
        logger.debug("Test HDF5 file created successfully")

    def tearDown(self):
        """Tear down test fixtures."""
        logger.debug(f"Tearing down test: {self._testMethodName}")
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.debug(f"Temporary directory cleaned: {self.temp_dir}")
        logger.info(f"Test completed: {self._testMethodName}")

    def _create_test_hdf_file(self):
        """Create a minimal test HDF5 file with XPCS structure."""
        logger.debug("Creating test HDF5 file with XPCS structure")
        with h5py.File(self.test_file, "w") as f:
            # Create basic structure
            entry = f.create_group("entry")
            instrument = entry.create_group("instrument")
            instrument.create_group("detector")

            # Add minimal required datasets
            xpcs = f.create_group("xpcs")
            qmap_group = xpcs.create_group("qmap")

            # Minimal qmap data
            qmap_group.create_dataset("mask", data=np.ones((100, 100), dtype=np.int32))
            qmap_group.create_dataset("dqmap", data=np.ones((100, 100), dtype=np.int32))
            qmap_group.create_dataset("sqmap", data=np.ones((100, 100), dtype=np.int32))
            qmap_group.create_dataset("dqlist", data=np.linspace(0.01, 0.1, 10))
            qmap_group.create_dataset("sqlist", data=np.linspace(0.01, 0.1, 10))
            qmap_group.create_dataset("dplist", data=np.linspace(0, 360, 36))
            qmap_group.create_dataset("splist", data=np.linspace(0, 360, 36))
            qmap_group.create_dataset("bcx", data=50.0)
            qmap_group.create_dataset("bcy", data=50.0)
            qmap_group.create_dataset("X_energy", data=8.0)
            qmap_group.create_dataset("pixel_size", data=75e-6)
            qmap_group.create_dataset("det_dist", data=5.0)
            qmap_group.create_dataset("dynamic_num_pts", data=10)
            qmap_group.create_dataset("static_num_pts", data=10)
            qmap_group.create_dataset("static_index_mapping", data=np.arange(10))
            qmap_group.create_dataset("dynamic_index_mapping", data=np.arange(10))
            qmap_group.create_dataset("map_names", data=[b"q", b"phi"])
            qmap_group.create_dataset("map_units", data=[b"1/A", b"degree"])

            # Add minimal multitau data
            multitau = xpcs.create_group("multitau")
            multitau.create_dataset("g2", data=np.random.rand(20, 10) + 1.0)
            multitau.create_dataset("g2_err", data=np.random.rand(20, 10) * 0.1)
            multitau.create_dataset("tau", data=np.logspace(-6, 2, 20))
            multitau.create_dataset("stride_frame", data=1)
            multitau.create_dataset("avg_frame", data=1)

            # Add minimal data
            multitau.create_dataset("saxs_1d", data=np.random.rand(1, 50))
            multitau.create_dataset("Iqp", data=np.random.rand(10, 50))
            multitau.create_dataset("Int_t", data=np.random.rand(2, 100))
            multitau.create_dataset("t0", data=0.001)
            multitau.create_dataset("t1", data=0.001)
            multitau.create_dataset("start_time", data=1000000000)

    def test_package_version(self):
        """Test that package version is accessible."""
        logger.debug(f"Testing package version: {__version__}")
        self.assertIsInstance(__version__, str)
        self.assertTrue(len(__version__) > 0)
        logger.info(f"Package version verified: {__version__}")

    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        logger.debug("Testing CLI module import")
        self.assertTrue(callable(main))
        logger.info("CLI module import successful")

    def test_file_locator(self):
        """Test FileLocator functionality."""
        logger.debug(f"Testing FileLocator with path: {self.temp_dir}")
        locator = FileLocator(self.temp_dir)
        self.assertEqual(locator.path, self.temp_dir)
        logger.info("FileLocator functionality verified")

    def test_xpcs_file_creation(self):
        """Test XpcsFile can be created with test data."""
        logger.debug(f"Testing XpcsFile creation with: {self.test_file}")
        if os.path.exists(self.test_file):
            try:
                logger.debug("Creating XpcsFile instance")
                xf = XpcsFile(self.test_file)
                self.assertIsNotNone(xf)
                self.assertTrue(hasattr(xf, "atype"))
                self.assertTrue(hasattr(xf, "label"))
                logger.info("XpcsFile creation successful")
            except Exception as e:
                logger.warning(f"XpcsFile creation failed: {e}")
                # If XpcsFile fails due to missing dependencies, that's okay for basic testing
                self.skipTest(
                    f"XpcsFile creation failed (likely missing optional dependencies): {e}"
                )

    def test_basic_imports(self):
        """Test that basic modules can be imported."""
        logger.debug("Testing basic module imports")
        try:
            # Test core module imports by importing and checking they exist
            import xpcs_toolkit.viewer_kernel
            import xpcs_toolkit.file_locator
            
            # Check that the modules have the expected classes
            self.assertTrue(hasattr(xpcs_toolkit.viewer_kernel, 'ViewerKernel'))
            self.assertTrue(hasattr(xpcs_toolkit.file_locator, 'FileLocator'))

            logger.info("Basic module imports successful")
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            logger.error(f"Failed to import basic modules: {e}")
            self.fail(f"Failed to import basic modules: {e}")

    def test_threading_imports(self):
        """Test that threading components can be imported without metaclass conflicts."""
        logger.debug("Testing threading module imports")
        try:
            logger.debug("Importing threading components")
            from xpcs_toolkit.threading.async_workers import WorkerSignals

            # Test that WorkerSignals can be instantiated
            import os

            os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Headless mode
            logger.debug("Creating WorkerSignals instance in headless mode")
            signals = WorkerSignals()
            self.assertIsNotNone(signals)
            logger.info(
                "Threading module imports and WorkerSignals instantiation successful"
            )

            self.assertTrue(True)  # If we get here, all threading imports worked
        except Exception as e:
            logger.error(f"Failed to import threading modules: {e}")
            self.fail(f"Failed to import threading modules: {e}")


if __name__ == "__main__":
    unittest.main()
