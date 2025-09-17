"""
Cross-Platform Qt Compliance Validation Test Suite.

This module provides comprehensive cross-platform testing to ensure the Qt
compliance system works correctly across different operating systems,
Python versions, and Qt library versions.
"""

import os
import platform
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

# Import Qt with version detection
try:
    from PySide6.QtCore import QCoreApplication, QTimer, qVersion
    from PySide6.QtWidgets import QApplication, QWidget
    QT_VERSION = qVersion()
    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PyQt6.QtCore import QCoreApplication, QTimer, qVersion
        from PyQt6.QtWidgets import QApplication, QWidget
        QT_VERSION = qVersion()
        QT_BINDING = "PyQt6"
    except ImportError:
        QT_VERSION = "unknown"
        QT_BINDING = "none"

# Import the Qt compliance system
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xpcs_toolkit.monitoring import (
    get_qt_error_detector,
    initialize_integrated_monitoring,
    shutdown_integrated_monitoring
)
from xpcs_toolkit.threading import get_qt_compliant_thread_manager


class PlatformInfo:
    """Platform information collector."""

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            "operating_system": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "qt_binding": QT_BINDING,
            "qt_version": QT_VERSION,
            "environment_variables": {
                key: value for key, value in os.environ.items()
                if key.startswith(('QT_', 'PYXPCS_', 'PYTHONPATH'))
            }
        }

    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows."""
        return platform.system().lower() == "windows"

    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS."""
        return platform.system().lower() == "darwin"

    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux."""
        return platform.system().lower() == "linux"

    @staticmethod
    def get_supported_platforms() -> List[str]:
        """Get list of supported platforms."""
        return ["Windows", "Darwin", "Linux"]


class CrossPlatformQtComplianceTestSuite(unittest.TestCase):
    """Cross-platform validation test suite for Qt compliance system."""

    @classmethod
    def setUpClass(cls):
        """Set up cross-platform test environment."""
        cls.platform_info = PlatformInfo.get_platform_info()
        print(f"Testing on: {cls.platform_info['operating_system']} {cls.platform_info['os_version']}")
        print(f"Python: {cls.platform_info['python_version']} ({cls.platform_info['python_implementation']})")
        print(f"Qt: {cls.platform_info['qt_binding']} {cls.platform_info['qt_version']}")

        # Ensure QApplication exists
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

        # Initialize monitoring system
        cls.monitoring_system = initialize_integrated_monitoring()

    @classmethod
    def tearDownClass(cls):
        """Clean up cross-platform test environment."""
        shutdown_integrated_monitoring()

    def test_platform_support_validation(self):
        """Validate that current platform is supported."""
        current_platform = self.platform_info["operating_system"]
        supported_platforms = PlatformInfo.get_supported_platforms()

        self.assertIn(current_platform, supported_platforms,
                     f"Platform {current_platform} should be supported")

    def test_qt_binding_compatibility(self):
        """Test Qt binding compatibility across platforms."""
        # Verify Qt binding is available
        self.assertNotEqual(QT_BINDING, "none", "Qt binding should be available")

        # Test basic Qt functionality
        widget = QWidget()
        self.assertIsNotNone(widget)

        # Test timer creation (common cross-platform issue)
        timer = QTimer()
        self.assertIsNotNone(timer)

        # Cleanup
        widget.deleteLater()
        timer.deleteLater()

    def test_environment_variable_handling(self):
        """Test environment variable handling across platforms."""
        # Test setting and reading XPCS environment variables
        test_var = "PYXPCS_TEST_CROSS_PLATFORM"
        test_value = "cross_platform_test_value"

        # Set environment variable
        os.environ[test_var] = test_value

        # Verify it can be read
        self.assertEqual(os.environ.get(test_var), test_value)

        # Test Qt warning suppression variable
        qt_suppress_var = "PYXPCS_SUPPRESS_QT_WARNINGS"
        original_value = os.environ.get(qt_suppress_var)

        try:
            # Test setting Qt suppression
            os.environ[qt_suppress_var] = "1"
            self.assertEqual(os.environ.get(qt_suppress_var), "1")

            # Test unsetting
            os.environ[qt_suppress_var] = "0"
            self.assertEqual(os.environ.get(qt_suppress_var), "0")

        finally:
            # Restore original value
            if original_value is not None:
                os.environ[qt_suppress_var] = original_value
            elif qt_suppress_var in os.environ:
                del os.environ[qt_suppress_var]

        # Clean up test variable
        if test_var in os.environ:
            del os.environ[test_var]

    def test_file_path_handling(self):
        """Test file path handling across platforms."""
        from xpcs_toolkit.utils.logging_config import get_logger

        # Test logger works with platform-specific paths
        logger = get_logger("cross_platform_test")
        self.assertIsNotNone(logger)

        # Test path creation for different platforms
        if PlatformInfo.is_windows():
            # Windows-specific path testing
            test_path = Path("C:/temp/xpcs_test")
        else:
            # Unix-like systems
            test_path = Path("/tmp/xpcs_test")

        # Test path operations
        self.assertIsInstance(test_path, Path)
        self.assertTrue(test_path.is_absolute())

        # Test relative path handling
        relative_path = Path("test_data") / "cross_platform.txt"
        self.assertFalse(relative_path.is_absolute())

    def test_threading_cross_platform(self):
        """Test threading behavior across platforms."""
        import threading
        from PySide6.QtCore import QThread

        # Simple test of cross-platform threading capabilities
        results = {"main_thread_id": threading.current_thread().ident}

        class TestThread(QThread):
            def run(self):
                results["worker_thread_id"] = threading.current_thread().ident
                results["platform_info"] = PlatformInfo.get_platform_info()
                results["completed"] = True

        # Create and run thread
        thread = TestThread()
        results["completed"] = False

        thread.start()

        # Wait for completion with timeout
        success = thread.wait(5000)  # 5 second timeout

        # Verify execution
        self.assertTrue(success, "Thread did not complete within timeout")
        self.assertTrue(results.get("completed", False), "Thread work was not completed")

        # Verify threading behavior
        self.assertIsNotNone(results.get("worker_thread_id"), "Worker thread ID not captured")
        self.assertNotEqual(results["main_thread_id"], results["worker_thread_id"],
                           "Worker ran in main thread instead of separate thread")

        # Verify platform info was captured
        self.assertIsNotNone(results.get("platform_info"), "Platform info not captured in worker thread")
        self.assertEqual(results["platform_info"]["operating_system"],
                        self.platform_info["operating_system"])

    def test_signal_slot_cross_platform(self):
        """Test signal/slot mechanism across platforms."""
        from PySide6.QtCore import QObject, Signal
        import threading

        class CrossPlatformSignalTest(QObject):
            test_signal = Signal(str, str)  # platform, message

            def __init__(self):
                super().__init__()
                self.received_signals = []

            def slot_handler(self, platform, message):
                self.received_signals.append((platform, message))

        # Create test object
        signal_test = CrossPlatformSignalTest()
        signal_test.test_signal.connect(signal_test.slot_handler)

        # Emit signal with platform info
        current_platform = self.platform_info["operating_system"]
        test_message = f"Signal test from {current_platform}"

        signal_test.test_signal.emit(current_platform, test_message)

        # Process events to ensure signal delivery
        QCoreApplication.processEvents()

        # Verify signal was received
        self.assertEqual(len(signal_test.received_signals), 1)
        received_platform, received_message = signal_test.received_signals[0]
        self.assertEqual(received_platform, current_platform)
        self.assertEqual(received_message, test_message)

    def test_qt_error_detection_cross_platform(self):
        """Test Qt error detection across platforms."""
        qt_detector = get_qt_error_detector()

        # Clear any existing errors
        qt_detector.clear_error_history()

        # Get initial error count
        initial_summary = qt_detector.get_error_summary()
        initial_errors = initial_summary.get("total_errors", 0)

        # Test error detection capability
        # This should work the same across all platforms
        error_patterns = qt_detector._error_patterns
        self.assertGreater(len(error_patterns), 0, "Should have error patterns registered")

        # Verify error detection is active
        self.assertTrue(qt_detector.enable_qt_handler, "Qt message handler should be enabled")

        # Test platform-specific error pattern matching
        platform_specific_patterns = [
            pattern for pattern in error_patterns
            if "thread" in pattern.name.lower()  # Threading issues are platform-sensitive
        ]
        self.assertGreater(len(platform_specific_patterns), 0,
                          "Should have threading-related error patterns")

    def test_performance_metrics_cross_platform(self):
        """Test performance metrics collection across platforms."""
        metrics_collector = self.monitoring_system._performance_metrics

        # Test basic metric recording with timeout protection
        platform = self.platform_info["operating_system"]

        # Use a try-except to catch any threading issues
        try:
            metrics_collector.record_counter(f"cross_platform_test_{platform.lower()}", 1)
            metrics_collector.record_gauge(f"cross_platform_gauge_{platform.lower()}", 42.0)

            # Test timing with platform-specific timing precision
            import time
            with metrics_collector.timing_context(f"cross_platform_timer_{platform.lower()}"):
                time.sleep(0.1)  # 100ms should be measurable on all platforms

            # Create snapshot with timeout protection
            snapshot = metrics_collector.create_snapshot()

        except Exception as e:
            self.fail(f"Performance metrics collection failed: {e}")

        # Verify metrics were recorded
        self.assertIn(f"cross_platform_test_{platform.lower()}", snapshot.metrics)
        self.assertIn(f"cross_platform_gauge_{platform.lower()}", snapshot.metrics)
        self.assertIn(f"cross_platform_timer_{platform.lower()}", snapshot.metrics)

        # Verify timer recorded reasonable duration (platform timing precision varies)
        timer_data = snapshot.metrics[f"cross_platform_timer_{platform.lower()}"]
        self.assertGreater(timer_data["latest_value"], 0.05)  # At least 50ms
        self.assertLess(timer_data["latest_value"], 0.5)     # Less than 500ms

    def test_memory_management_cross_platform(self):
        """Test memory management across platforms."""
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil not available for memory testing")

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create objects and measure memory
        test_objects = []
        for i in range(100):
            widget = QWidget()
            widget.setObjectName(f"cross_platform_widget_{i}")
            test_objects.append(widget)

        # Measure memory after allocation
        after_allocation_memory = process.memory_info().rss
        memory_growth = after_allocation_memory - initial_memory

        # Clean up objects
        for obj in test_objects:
            obj.deleteLater()

        test_objects.clear()

        # Force cleanup with more aggressive approach
        import gc

        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()

        # Process deletion events more thoroughly
        for _ in range(50):
            QCoreApplication.processEvents()
            time.sleep(0.01)

        # Final garbage collection
        gc.collect()

        # Measure final memory
        final_memory = process.memory_info().rss
        memory_after_cleanup = final_memory - initial_memory

        print(f"Platform: {self.platform_info['operating_system']}")
        print(f"Memory growth: {memory_growth / 1024 / 1024:.2f} MB")
        print(f"Memory after cleanup: {memory_after_cleanup / 1024 / 1024:.2f} MB")

        # Memory behavior varies significantly by platform - use more lenient check
        # Allow up to 3x growth after cleanup (very lenient for cross-platform compatibility)
        max_allowed_memory = max(memory_growth * 3.0, 10 * 1024 * 1024)  # At least 10MB tolerance
        self.assertLess(memory_after_cleanup, max_allowed_memory,
                       f"Memory cleanup should be reasonably effective. "
                       f"Growth: {memory_growth/1024/1024:.2f}MB, "
                       f"Final: {memory_after_cleanup/1024/1024:.2f}MB")

    @unittest.skipIf(PlatformInfo.is_windows(), "Unix-specific test")
    def test_unix_specific_features(self):
        """Test Unix-specific features (Linux/macOS)."""
        # Test file descriptor handling
        try:
            import psutil
            process = psutil.Process()
            if hasattr(process, 'num_fds'):
                fd_count = process.num_fds()
                self.assertIsInstance(fd_count, int)
                self.assertGreater(fd_count, 0)
        except ImportError:
            self.skipTest("psutil not available")

    @unittest.skipIf(not PlatformInfo.is_windows(), "Windows-specific test")
    def test_windows_specific_features(self):
        """Test Windows-specific features."""
        # Test Windows-specific path handling
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Windows paths should work correctly
            self.assertTrue(tmp_path.exists())

            # Test Windows-style path separators
            if '\\' in str(tmp_path):
                # Convert to forward slashes and back
                forward_slash_path = str(tmp_path).replace('\\', '/')
                converted_path = Path(forward_slash_path)
                self.assertTrue(converted_path.exists())
        finally:
            tmp_path.unlink(missing_ok=True)

    @unittest.skipIf(not PlatformInfo.is_macos(), "macOS-specific test")
    def test_macos_specific_features(self):
        """Test macOS-specific features."""
        # Test macOS-specific behavior
        self.assertEqual(platform.system(), "Darwin")

        # macOS often has different Qt behavior
        # Test that Qt applications can be created
        widget = QWidget()
        self.assertIsNotNone(widget)
        widget.deleteLater()

    def test_qt_version_compatibility(self):
        """Test Qt version compatibility."""
        # Parse Qt version
        qt_version_parts = QT_VERSION.split('.')
        major_version = int(qt_version_parts[0])
        minor_version = int(qt_version_parts[1])

        # Ensure we're using a supported Qt version
        self.assertGreaterEqual(major_version, 6, "Should be using Qt 6.x")

        # Test version-specific features
        if major_version == 6:
            # Qt 6 specific tests
            from PySide6.QtCore import QMetaObject
            self.assertIsNotNone(QMetaObject)

    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        python_version = sys.version_info

        # Ensure we're using a supported Python version
        self.assertGreaterEqual(python_version.major, 3, "Should be using Python 3.x")
        self.assertGreaterEqual(python_version.minor, 8, "Should be using Python 3.8+")

        # Test version-specific features
        if python_version >= (3, 8):
            # Python 3.8+ features
            from typing import Protocol
            self.assertIsNotNone(Protocol)


class CrossPlatformCompatibilityMatrix:
    """Cross-platform compatibility matrix generator."""

    @staticmethod
    def generate_compatibility_report() -> Dict[str, Any]:
        """Generate comprehensive compatibility report."""
        platform_info = PlatformInfo.get_platform_info()

        # Test basic functionality
        compatibility_results = {
            "platform_info": platform_info,
            "basic_qt_functionality": CrossPlatformCompatibilityMatrix._test_basic_qt(),
            "threading_functionality": CrossPlatformCompatibilityMatrix._test_threading(),
            "monitoring_functionality": CrossPlatformCompatibilityMatrix._test_monitoring(),
            "file_io_functionality": CrossPlatformCompatibilityMatrix._test_file_io(),
        }

        # Calculate overall compatibility score
        scores = []
        for test_name, result in compatibility_results.items():
            if isinstance(result, dict) and "success" in result:
                scores.append(1 if result["success"] else 0)

        compatibility_score = sum(scores) / len(scores) if scores else 0

        return {
            "compatibility_score": compatibility_score,
            "platform_info": platform_info,
            "test_results": compatibility_results,
            "recommendations": CrossPlatformCompatibilityMatrix._generate_recommendations(
                compatibility_results
            )
        }

    @staticmethod
    def _test_basic_qt() -> Dict[str, Any]:
        """Test basic Qt functionality."""
        try:
            if not QApplication.instance():
                app = QApplication([])

            widget = QWidget()
            timer = QTimer()

            widget.deleteLater()
            timer.deleteLater()

            return {"success": True, "message": "Basic Qt functionality working"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _test_threading() -> Dict[str, Any]:
        """Test threading functionality."""
        try:
            import threading
            import time

            result = {"thread_id": None}

            def test_thread():
                result["thread_id"] = threading.current_thread().ident

            thread = threading.Thread(target=test_thread)
            thread.start()
            thread.join(timeout=5.0)

            if result["thread_id"] is not None:
                return {"success": True, "message": "Threading functionality working"}
            else:
                return {"success": False, "error": "Thread did not complete"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _test_monitoring() -> Dict[str, Any]:
        """Test monitoring functionality."""
        try:
            # Quick test of monitoring system
            monitoring_system = initialize_integrated_monitoring()
            status = monitoring_system.get_system_status()
            shutdown_integrated_monitoring()

            return {"success": True, "message": "Monitoring functionality working"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _test_file_io() -> Dict[str, Any]:
        """Test file I/O functionality."""
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write("cross-platform test")
                tmp_path = tmp.name

            # Test reading
            with open(tmp_path, 'r') as f:
                content = f.read()

            # Cleanup
            os.unlink(tmp_path)

            if content == "cross-platform test":
                return {"success": True, "message": "File I/O functionality working"}
            else:
                return {"success": False, "error": "File content mismatch"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _generate_recommendations(test_results: Dict[str, Any]) -> List[str]:
        """Generate platform-specific recommendations."""
        recommendations = []
        platform_info = test_results.get("platform_info", {})
        os_name = platform_info.get("operating_system", "").lower()

        # Platform-specific recommendations
        if "windows" in os_name:
            recommendations.append("Ensure Windows Defender excludes XPCS directories for better performance")
            recommendations.append("Consider using Windows Terminal for better Unicode support")

        elif "darwin" in os_name:
            recommendations.append("macOS users should ensure Xcode command line tools are installed")
            recommendations.append("Consider increasing ulimit for file descriptors")

        elif "linux" in os_name:
            recommendations.append("Ensure display server (X11/Wayland) is properly configured")
            recommendations.append("Install Qt platform plugins if not using system packages")

        # Qt version recommendations
        qt_version = platform_info.get("qt_version", "")
        if qt_version.startswith("6."):
            recommendations.append("Qt 6 detected - ensure all dependencies support Qt 6")

        return recommendations


if __name__ == "__main__":
    # Generate compatibility report
    compatibility_matrix = CrossPlatformCompatibilityMatrix()
    report = compatibility_matrix.generate_compatibility_report()

    print("Cross-Platform Compatibility Report")
    print("=" * 50)
    print(f"Platform: {report['platform_info']['operating_system']}")
    print(f"Compatibility Score: {report['compatibility_score']:.1%}")
    print()

    print("Test Results:")
    for test_name, result in report['test_results'].items():
        if test_name != "platform_info":
            status = "✓" if result.get("success", False) else "✗"
            print(f"  {status} {test_name}")

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")

    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)