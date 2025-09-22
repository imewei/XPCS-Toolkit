"""Qt Test Runner for Error Detection.

Specialized test runner for Qt-related error detection and GUI testing
with comprehensive error capture and analysis capabilities.
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

from PySide6 import QtWidgets


class QtTestRunner:
    """Specialized test runner for Qt GUI components with error capture."""

    def __init__(self, capture_stdout=True, capture_stderr=True, timeout=30):
        """
        Initialize Qt test runner.

        Args:
            capture_stdout: Whether to capture stdout
            capture_stderr: Whether to capture stderr
            timeout: Test timeout in seconds
        """
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr
        self.timeout = timeout
        self.qt_errors = []
        self.test_results = {}
        self.app = None

    def setup_qt_application(self):
        """Set up Qt application for testing."""
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication([])
            self.app.setQuitOnLastWindowClosed(False)
            # Set up offscreen platform for headless testing
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return self.app

    def cleanup_qt_application(self):
        """Clean up Qt application after testing."""
        if self.app and self.app != QtWidgets.QApplication.instance():
            self.app.quit()
            self.app = None

    def capture_qt_errors(self):
        """Context manager to capture Qt error messages."""
        return self._QtErrorCapture(self)

    class _QtErrorCapture:
        """Context manager for capturing Qt errors."""

        def __init__(self, runner):
            self.runner = runner
            self.original_stderr = None
            self.captured_stderr = None
            self.qt_error_patterns = [
                r"QObject::startTimer: Timers can only be used with threads started with QThread",
                r"qt\.core\.qobject\.connect: QObject::connect.*unique connections require.*",
                r"QWidget: Cannot create a QWidget without QApplication",
                r"QPixmap: It is not safe to use pixmaps outside the GUI thread",
                r"QTimer: QTimer can only be used with threads started with QThread",
                r"QObject::moveToThread.*",
                r"Qt.*Warning.*",
                r"Qt.*Critical.*",
            ]

        def __enter__(self):
            if self.runner.capture_stderr:
                self.original_stderr = sys.stderr
                self.captured_stderr = io.StringIO()
                sys.stderr = self.captured_stderr
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.runner.capture_stderr and self.original_stderr:
                sys.stderr = self.original_stderr
                captured_output = self.captured_stderr.getvalue()

                # Parse captured errors
                self._parse_qt_errors(captured_output)

        def _parse_qt_errors(self, output):
            """Parse Qt errors from captured output."""
            lines = output.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                for pattern in self.qt_error_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        error_info = {
                            "message": line,
                            "pattern": pattern,
                            "timestamp": time.time(),
                            "thread": threading.current_thread().name,
                        }
                        self.runner.qt_errors.append(error_info)
                        break

    def run_qt_test_function(self, test_func, *args, **kwargs):
        """
        Run a single Qt test function with error capture.

        Args:
            test_func: Test function to run
            *args: Arguments for test function
            **kwargs: Keyword arguments for test function

        Returns:
            Dict with test results and captured errors
        """
        test_name = getattr(test_func, "__name__", str(test_func))
        start_time = time.time()

        result = {
            "test_name": test_name,
            "success": False,
            "error": None,
            "qt_errors": [],
            "execution_time": 0,
            "warnings": [],
        }

        try:
            self.setup_qt_application()

            with self.capture_qt_errors():
                # Run the test function
                test_result = test_func(*args, **kwargs)
                result["success"] = True
                result["return_value"] = test_result

        except Exception as e:
            result["error"] = str(e)
            result["exception_type"] = type(e).__name__

        finally:
            result["execution_time"] = time.time() - start_time
            result["qt_errors"] = self.qt_errors.copy()
            self.qt_errors.clear()

            # Process Qt events to ensure cleanup
            if self.app:
                self.app.processEvents()

        return result

    def run_qt_test_suite(self, test_functions):
        """
        Run a suite of Qt test functions.

        Args:
            test_functions: List of test functions or (func, args, kwargs) tuples

        Returns:
            Dict with overall test results
        """
        suite_results = {
            "total_tests": len(test_functions),
            "passed": 0,
            "failed": 0,
            "total_qt_errors": 0,
            "test_details": [],
            "error_summary": {},
        }

        for test_item in test_functions:
            if callable(test_item):
                # Simple function
                result = self.run_qt_test_function(test_item)
            elif isinstance(test_item, (tuple, list)) and len(test_item) >= 1:
                # Function with arguments
                func = test_item[0]
                args = test_item[1] if len(test_item) > 1 else []
                kwargs = test_item[2] if len(test_item) > 2 else {}
                result = self.run_qt_test_function(func, *args, **kwargs)
            else:
                continue

            suite_results["test_details"].append(result)

            if result["success"]:
                suite_results["passed"] += 1
            else:
                suite_results["failed"] += 1

            suite_results["total_qt_errors"] += len(result["qt_errors"])

        # Generate error summary
        suite_results["error_summary"] = self._generate_error_summary(
            suite_results["test_details"]
        )

        return suite_results

    def _generate_error_summary(self, test_details):
        """Generate summary of Qt errors across all tests."""
        error_counts = {}
        error_patterns = {}

        for test in test_details:
            for error in test["qt_errors"]:
                pattern = error["pattern"]
                message = error["message"]

                if pattern not in error_patterns:
                    error_patterns[pattern] = []
                error_patterns[pattern].append(message)

                if pattern not in error_counts:
                    error_counts[pattern] = 0
                error_counts[pattern] += 1

        return {
            "error_counts": error_counts,
            "error_patterns": error_patterns,
            "most_common_errors": sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            ),
        }

    def run_pytest_with_qt_capture(self, test_path, markers=None, verbose=True):
        """
        Run pytest with Qt error capture.

        Args:
            test_path: Path to test file or directory
            markers: Pytest markers to filter tests
            verbose: Whether to run in verbose mode

        Returns:
            Dict with pytest results and Qt error analysis
        """
        # Create temporary file for pytest output
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", delete=False
        ) as temp_file:
            temp_file_path = temp_file.name

        try:
            # Build pytest command
            cmd = [sys.executable, "-m", "pytest", str(test_path)]

            if verbose:
                cmd.append("-v")

            if markers:
                cmd.extend(["-m", markers])

            # Add capture options
            cmd.extend(["--tb=short", "--capture=no"])

            # Set environment for Qt testing
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["PYXPCS_SUPPRESS_QT_WARNINGS"] = "0"  # We want to capture warnings

            # Run pytest
            start_time = time.time()
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            execution_time = time.time() - start_time

            # Parse pytest output for Qt errors
            qt_errors = self._parse_pytest_output_for_qt_errors(
                result.stderr + result.stdout
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "qt_errors": qt_errors,
                "execution_time": execution_time,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "timeout": self.timeout,
            }

        finally:
            # Clean up temporary file
            with contextlib.suppress(OSError):
                os.unlink(temp_file_path)

    def _parse_pytest_output_for_qt_errors(self, output):
        """Parse pytest output for Qt error messages."""
        qt_errors = []
        lines = output.split("\n")

        qt_patterns = [
            r"QObject::startTimer.*",
            r"qt\.core\.qobject\.connect.*",
            r"QWidget.*QApplication.*",
            r"QTimer.*QThread.*",
        ]

        for line in lines:
            line = line.strip()
            for pattern in qt_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    qt_errors.append(
                        {"message": line, "pattern": pattern, "source": "pytest_output"}
                    )
                    break

        return qt_errors

    def generate_error_report(self, results, output_file=None):
        """
        Generate comprehensive error report.

        Args:
            results: Test results from run_qt_test_suite or run_pytest_with_qt_capture
            output_file: Optional file path to save report

        Returns:
            String containing the error report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Qt Error Detection Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        if "total_tests" in results:
            # Test suite results
            report_lines.append(f"Total Tests: {results['total_tests']}")
            report_lines.append(f"Passed: {results['passed']}")
            report_lines.append(f"Failed: {results['failed']}")
            report_lines.append(f"Total Qt Errors: {results['total_qt_errors']}")
            report_lines.append("")

            if results["error_summary"]["most_common_errors"]:
                report_lines.append("Most Common Qt Errors:")
                for pattern, count in results["error_summary"]["most_common_errors"][
                    :5
                ]:
                    report_lines.append(f"  {count}x: {pattern}")
                report_lines.append("")

        elif "qt_errors" in results:
            # Pytest results
            report_lines.append(
                f"Execution Time: {results.get('execution_time', 0):.2f}s"
            )
            report_lines.append(f"Success: {results['success']}")
            report_lines.append(f"Qt Errors Found: {len(results['qt_errors'])}")
            report_lines.append("")

            if results["qt_errors"]:
                report_lines.append("Qt Errors Detected:")
                for error in results["qt_errors"]:
                    report_lines.append(f"  - {error['message']}")
                report_lines.append("")

        # Recommendations
        report_lines.append("Recommendations:")
        if "total_qt_errors" in results and results["total_qt_errors"] > 0:
            report_lines.append(
                "  1. Fix Qt timer threading violations by using QThread-started threads"
            )
            report_lines.append(
                "  2. Update signal/slot connections to use Qt5+ syntax"
            )
            report_lines.append("  3. Ensure GUI operations happen in main thread")
        else:
            report_lines.append("  No Qt errors detected - good job!")

        report_lines.append("")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)

        return report


class XpcsQtTestRunner(QtTestRunner):
    """Specialized Qt test runner for XPCS Toolkit components."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xpcs_specific_patterns = [
            r"BackgroundCleanupManager.*timer.*",
            r"XpcsViewer.*initialization.*",
            r"ViewerKernel.*thread.*",
            r"ImageViewDev.*connection.*",
        ]

    def run_xpcs_viewer_test(self, test_path="./", **kwargs):
        """Run XPCS viewer-specific test with error detection."""

        def xpcs_viewer_test():
            try:
                from xpcs_toolkit.xpcs_viewer import XpcsViewer

                viewer = XpcsViewer(path=test_path, **kwargs)
                # Let the application process events
                self.app.processEvents()
                viewer.close()
                return True
            except Exception as e:
                print(f"XPCS viewer test failed: {e}")
                return False

        return self.run_qt_test_function(xpcs_viewer_test)

    def run_background_cleanup_test(self):
        """Test background cleanup system for Qt compliance."""

        def background_cleanup_test():
            try:
                from xpcs_toolkit.threading.cleanup_ops import BackgroundCleanupManager

                # This might trigger timer threading warnings
                cleanup_manager = BackgroundCleanupManager()
                cleanup_manager.start()
                time.sleep(0.1)  # Brief pause
                cleanup_manager.stop()
                return True
            except Exception as e:
                print(f"Background cleanup test failed: {e}")
                return False

        return self.run_qt_test_function(background_cleanup_test)

    def run_plot_handler_test(self):
        """Test plot handlers for Qt compliance."""

        def plot_handler_test():
            try:
                from xpcs_toolkit.plothandler import ImageViewDev

                # This might trigger QStyleHints connection warnings
                plot_handler = ImageViewDev()
                self.app.processEvents()
                plot_handler.deleteLater()
                return True
            except Exception as e:
                print(f"Plot handler test failed: {e}")
                return False

        return self.run_qt_test_function(plot_handler_test)

    def run_comprehensive_xpcs_test_suite(self):
        """Run comprehensive test suite for XPCS components."""
        test_functions = [
            self.run_xpcs_viewer_test,
            self.run_background_cleanup_test,
            self.run_plot_handler_test,
        ]

        return self.run_qt_test_suite(test_functions)


# CLI interface for the Qt test runner
def main():
    """Command-line interface for Qt test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Qt Error Detection Test Runner")
    parser.add_argument(
        "test_path",
        nargs="?",
        default="tests/unit/threading/test_qt_error_detection.py",
        help="Path to test file or directory",
    )
    parser.add_argument("--markers", "-m", help="Pytest markers to filter tests")
    parser.add_argument(
        "--timeout", "-t", type=int, default=30, help="Test timeout in seconds"
    )
    parser.add_argument("--output", "-o", help="Output file for error report")
    parser.add_argument("--xpcs", action="store_true", help="Run XPCS-specific tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.xpcs:
        runner = XpcsQtTestRunner(timeout=args.timeout)
        results = runner.run_comprehensive_xpcs_test_suite()
    else:
        runner = QtTestRunner(timeout=args.timeout)
        results = runner.run_pytest_with_qt_capture(
            args.test_path, markers=args.markers, verbose=args.verbose
        )

    # Generate and display report
    report = runner.generate_error_report(results, args.output)
    print(report)

    # Exit with error code if Qt errors were found
    if "total_qt_errors" in results:
        exit_code = 1 if results["total_qt_errors"] > 0 else 0
    elif "qt_errors" in results:
        exit_code = 1 if len(results["qt_errors"]) > 0 else 0
    else:
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
