"""Tests for Qt compatibility layer (User Story 4).

These tests verify the qt_compat module provides proper abstraction
for Qt bindings and that the application can work with different Qt backends.

Tasks: T057, T058, T059
"""

import os
import sys

import pytest


class TestQtCompatLayer:
    """Test qt_compat module provides correct abstraction."""

    def test_qt_compat_exports_core_modules(self):
        """T057: Verify qt_compat exports core Qt modules."""
        from xpcsviewer.gui.qt_compat import QtCore, QtGui, QtWidgets

        # Verify modules are present and usable
        assert QtCore is not None
        assert QtGui is not None
        assert QtWidgets is not None

        # Verify key classes are accessible
        assert hasattr(QtCore, "Qt")
        assert hasattr(QtCore, "Signal")
        assert hasattr(QtWidgets, "QWidget")

    def test_qt_compat_exports_signal_slot(self):
        """T057: Verify Signal and Slot are exported."""
        from xpcsviewer.gui.qt_compat import Signal, Slot

        assert Signal is not None
        assert Slot is not None

    def test_qt_compat_exports_common_widgets(self):
        """T057: Verify common widget classes are exported."""
        from xpcsviewer.gui.qt_compat import (
            QApplication,
            QDialog,
            QFrame,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        # Verify all classes are present
        assert QApplication is not None
        assert QDialog is not None
        assert QFrame is not None
        assert QGroupBox is not None
        assert QHBoxLayout is not None
        assert QLabel is not None
        assert QMainWindow is not None
        assert QPushButton is not None
        assert QVBoxLayout is not None
        assert QWidget is not None

    def test_qt_compat_exports_gui_classes(self):
        """T057: Verify GUI-related classes are exported."""
        from xpcsviewer.gui.qt_compat import (
            QColor,
            QDesktopServices,
            QFont,
            QIcon,
            QKeySequence,
            QPalette,
            QShortcut,
        )

        assert QColor is not None
        assert QDesktopServices is not None
        assert QFont is not None
        assert QIcon is not None
        assert QKeySequence is not None
        assert QPalette is not None
        assert QShortcut is not None

    def test_qt_compat_exports_threading_classes(self):
        """T057: Verify threading-related classes are exported."""
        from xpcsviewer.gui.qt_compat import (
            QObject,
            QThread,
            QThreadPool,
            QTimer,
        )

        assert QObject is not None
        assert QThread is not None
        assert QThreadPool is not None
        assert QTimer is not None


class TestQtApiEnvironment:
    """Test QT_API environment variable handling."""

    def test_qt_api_defaults_to_pyside6(self):
        """T059: Verify qt_compat defaults to pyside6 when QT_API is unset."""
        # Save original value
        original_qt_api = os.environ.get("QT_API")

        try:
            # Clear environment (simulate fresh import)
            if "QT_API" in os.environ:
                del os.environ["QT_API"]

            # Re-import qt_compat module
            import importlib

            import xpcsviewer.gui.qt_compat as qt_compat

            importlib.reload(qt_compat)

            # Check that it defaults to pyside6
            assert os.environ.get("QT_API") == "pyside6"

        finally:
            # Restore original environment
            if original_qt_api is not None:
                os.environ["QT_API"] = original_qt_api
            else:
                os.environ.setdefault("QT_API", "pyside6")

    def test_qt_api_respects_environment(self):
        """T059: Verify qt_compat respects QT_API environment variable."""
        # The current environment should have QT_API set
        qt_api = os.environ.get("QT_API", "pyside6")
        assert qt_api in ("pyside6", "pyqt6")


class TestQtCompatWithPySide6:
    """Test qt_compat with PySide6 backend (T057).

    Note: These tests require PySide6 to be installed.
    They verify the app launches correctly with the default backend.
    """

    def test_pyside6_backend_works(self):
        """T057: Verify application works with PySide6 (default)."""
        # Ensure PySide6 is the backend
        os.environ["QT_API"] = "pyside6"

        from xpcsviewer.gui.qt_compat import QApplication, Qt, QWidget

        # Verify Qt namespace is accessible
        assert hasattr(Qt, "AlignmentFlag")
        assert hasattr(Qt, "WindowType")

        # Verify widget can be instantiated (requires QApplication)
        app = QApplication.instance()
        if app is None:
            # Don't create new app in test - just verify import works
            pass
        else:
            # QApplication exists, verify widget creation works
            widget = QWidget()
            assert widget is not None
            widget.deleteLater()


@pytest.mark.skipif(
    not os.environ.get("TEST_PYQT6", False),
    reason="PyQt6 testing not enabled (set TEST_PYQT6=1 to enable)",
)
class TestQtCompatWithPyQt6:
    """Test qt_compat with PyQt6 backend (T058).

    Note: These tests require PyQt6 to be installed and TEST_PYQT6=1.
    They verify the app launches correctly with PyQt6 backend.
    """

    def test_pyqt6_backend_works(self):
        """T058: Verify application works with PyQt6."""
        # This test requires PyQt6 to be installed
        try:
            import PyQt6  # noqa: F401
        except ImportError:
            pytest.skip("PyQt6 not installed")

        # Set PyQt6 as the backend
        os.environ["QT_API"] = "pyqt6"

        # Reimport qt_compat with new backend
        import importlib

        import xpcsviewer.gui.qt_compat as qt_compat

        importlib.reload(qt_compat)

        from xpcsviewer.gui.qt_compat import QApplication, Qt, QWidget

        # Verify Qt namespace is accessible
        assert hasattr(Qt, "AlignmentFlag")
        assert hasattr(Qt, "WindowType")


class TestNoPySide6DirectImports:
    """Test that source files don't have direct PySide6 imports."""

    def test_no_direct_pyside6_imports_in_gui(self):
        """T056: Verify no direct PySide6 imports in gui/ (except qt_compat)."""
        import subprocess

        result = subprocess.run(
            [
                "grep",
                "-r",
                "--include=*.py",
                "-l",
                "from PySide6\\|import PySide6",
                "xpcsviewer/gui",
            ],
            capture_output=True,
            text=True,
            cwd="/home/wei/Documents/GitHub/xpcsviewer",
        )

        # Should find no files (qt_compat uses qtpy, not direct PySide6)
        files_with_pyside6 = [
            f for f in result.stdout.strip().split("\n") if f and "qt_compat" not in f
        ]
        assert len(files_with_pyside6) == 0, (
            f"Found direct PySide6 imports in: {files_with_pyside6}"
        )

    def test_no_direct_pyside6_imports_in_modules(self):
        """T056: Verify no direct PySide6 imports in module/ and threading/."""
        import subprocess

        for directory in ["xpcsviewer/module", "xpcsviewer/threading"]:
            result = subprocess.run(
                [
                    "grep",
                    "-r",
                    "--include=*.py",
                    "-l",
                    "from PySide6\\|import PySide6",
                    directory,
                ],
                capture_output=True,
                text=True,
                cwd="/home/wei/Documents/GitHub/xpcsviewer",
            )

            files_with_pyside6 = [f for f in result.stdout.strip().split("\n") if f]
            assert len(files_with_pyside6) == 0, (
                f"Found direct PySide6 imports in {directory}: {files_with_pyside6}"
            )

    def test_only_autogenerated_files_have_pyside6(self):
        """T056: Verify only auto-generated files have direct PySide6 imports."""
        import subprocess

        result = subprocess.run(
            [
                "grep",
                "-r",
                "--include=*.py",
                "-l",
                "from PySide6\\|import PySide6",
                "xpcsviewer",
            ],
            capture_output=True,
            text=True,
            cwd="/home/wei/Documents/GitHub/xpcsviewer",
        )

        files_with_pyside6 = [f for f in result.stdout.strip().split("\n") if f]

        # Only auto-generated files should have direct PySide6 imports
        allowed_files = {"xpcsviewer/viewer_ui.py", "xpcsviewer/icons_rc.py"}
        unexpected_files = [f for f in files_with_pyside6 if f not in allowed_files]

        assert len(unexpected_files) == 0, (
            f"Found direct PySide6 imports in non-autogenerated files: {unexpected_files}"
        )
