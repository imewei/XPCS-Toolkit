"""Qt and GUI testing fixtures for XPCS Toolkit tests.

This module provides fixtures for Qt application testing, GUI components,
and Qt-specific testing utilities.
"""

import os
import sys

import pytest

# Qt availability check
QT_AVAILABLE = True
QT_ERROR = None

try:
    from PySide6 import QtCore, QtTest, QtWidgets
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication
except ImportError as e:
    QT_AVAILABLE = False
    QT_ERROR = str(e)

    # Create mock Qt classes for testing without Qt
    class MockQt:
        def __getattr__(self, name):
            return None

    class MockQApplication:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 0

        def quit(self):
            pass

        def processEvents(self):
            pass

        @classmethod
        def instance(cls):
            return None

    class MockQWidget:
        def __init__(self, *args, **kwargs):
            pass

        def show(self):
            pass

        def close(self):
            return True

        def resize(self, w, h):
            pass

    # Mock Qt modules
    QtWidgets = type(
        "MockQtWidgets",
        (),
        {
            "QApplication": MockQApplication,
            "QWidget": MockQWidget,
            "QMainWindow": MockQWidget,
            "QVBoxLayout": MockQWidget,
            "QHBoxLayout": MockQWidget,
            "QPushButton": MockQWidget,
            "QLabel": MockQWidget,
        },
    )()

    QtCore = type(
        "MockQtCore",
        (),
        {
            "Qt": MockQt(),
            "QTimer": MockQWidget,
            "QThread": MockQWidget,
            "pyqtSignal": lambda *args, **kwargs: None,
        },
    )()

    QtTest = type(
        "MockQtTest",
        (),
        {
            "QTest": type(
                "QTest",
                (),
                {
                    "qWait": lambda x: None,
                    "mouseClick": lambda *args: None,
                    "keyClick": lambda *args: None,
                },
            )(),
        },
    )()


# ============================================================================
# Qt Application Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def qt_application():
    """Create QApplication instance for Qt-based tests."""
    if not QT_AVAILABLE:
        pytest.skip(f"Qt not available: {QT_ERROR}")

    # Set Qt platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    yield app

    # Don't quit the app here - let it persist for the session


@pytest.fixture(scope="function")
def qt_widget(qt_application):
    """Create a basic QWidget for testing."""
    if not QT_AVAILABLE:
        return MockQWidget()

    widget = QtWidgets.QWidget()
    yield widget
    widget.close()


@pytest.fixture(scope="function")
def qt_main_window(qt_application):
    """Create a QMainWindow for testing."""
    if not QT_AVAILABLE:
        return MockQWidget()

    window = QtWidgets.QMainWindow()
    yield window
    window.close()


# ============================================================================
# Qt Testing Utilities
# ============================================================================


@pytest.fixture(scope="function")
def qt_wait():
    """Provide Qt event loop waiting utility."""
    if not QT_AVAILABLE:

        def mock_wait(ms):
            import time

            time.sleep(ms / 1000.0)

        return mock_wait

    def wait_func(milliseconds):
        """Wait for Qt events to process."""
        QtTest.QTest.qWait(milliseconds)

    return wait_func


@pytest.fixture(scope="function")
def qt_click_helper():
    """Provide Qt widget clicking utility."""
    if not QT_AVAILABLE:

        def mock_click(widget, button=None):
            pass

        return mock_click

    def click_func(widget, button=QtCore.Qt.LeftButton):
        """Click a Qt widget."""
        QtTest.QTest.mouseClick(widget, button)

    return click_func


@pytest.fixture(scope="function")
def qt_key_helper():
    """Provide Qt keyboard input utility."""
    if not QT_AVAILABLE:

        def mock_key(widget, key):
            pass

        return mock_key

    def key_func(widget, key):
        """Send key press to Qt widget."""
        QtTest.QTest.keyClick(widget, key)

    return key_func


# ============================================================================
# Qt Mock Objects for Testing
# ============================================================================


class MockQtSignal:
    """Mock Qt signal for testing without Qt."""

    def __init__(self, *args):
        self.connections = []

    def connect(self, callback):
        self.connections.append(callback)

    def disconnect(self, callback=None):
        if callback is None:
            self.connections.clear()
        elif callback in self.connections:
            self.connections.remove(callback)

    def emit(self, *args, **kwargs):
        for callback in self.connections:
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Ignore callback errors in tests


@pytest.fixture(scope="function")
def mock_qt_signal():
    """Provide mock Qt signal for testing."""
    return MockQtSignal


class MockQtThread:
    """Mock Qt thread for testing without Qt."""

    def __init__(self):
        self.started = MockQtSignal()
        self.finished = MockQtSignal()
        self.is_running = False

    def start(self):
        self.is_running = True
        self.started.emit()

    def quit(self):
        self.is_running = False
        self.finished.emit()

    def wait(self, timeout=None):
        return True

    def isRunning(self):
        return self.is_running


@pytest.fixture(scope="function")
def mock_qt_thread():
    """Provide mock Qt thread for testing."""
    return MockQtThread


# ============================================================================
# Qt Error Testing Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def qt_error_catcher():
    """Fixture to catch Qt-related errors during testing."""
    errors = []

    def error_handler(msg_type, context, message):
        errors.append({"type": msg_type, "context": context, "message": message})

    if QT_AVAILABLE:
        # Install Qt message handler
        QtCore.qInstallMessageHandler(error_handler)

    yield errors

    if QT_AVAILABLE:
        # Reset message handler
        QtCore.qInstallMessageHandler(None)


@pytest.fixture(scope="function")
def qt_performance_monitor(qt_application):
    """Monitor Qt application performance during tests."""
    if not QT_AVAILABLE:
        yield {"event_count": 0, "processing_time": 0}
        return

    import time

    metrics = {"event_count": 0, "processing_time": 0, "start_time": time.time()}

    def count_events():
        metrics["event_count"] += 1

    # Connect to application's aboutToQuit signal if available
    if hasattr(qt_application, "aboutToQuit"):
        qt_application.aboutToQuit.connect(lambda: None)

    yield metrics

    metrics["processing_time"] = time.time() - metrics["start_time"]


# ============================================================================
# GUI Testing Helpers
# ============================================================================


@pytest.fixture(scope="function")
def gui_test_helper(qt_application, qt_wait):
    """Comprehensive GUI testing helper."""

    class GuiTestHelper:
        def __init__(self, app, wait_func):
            self.app = app
            self.wait = wait_func

        def process_events(self, timeout=100):
            """Process Qt events with timeout."""
            if QT_AVAILABLE and self.app:
                self.app.processEvents()
                self.wait(timeout)

        def create_test_widget(self, widget_class=None):
            """Create a test widget."""
            if not QT_AVAILABLE:
                return MockQWidget()

            widget_class = widget_class or QtWidgets.QWidget
            widget = widget_class()
            return widget

        def simulate_user_action(self, widget, action="click", **kwargs):
            """Simulate user interaction with widget."""
            if not QT_AVAILABLE:
                return

            if action == "click":
                QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton)
            elif action == "key":
                key = kwargs.get("key", QtCore.Qt.Key_Return)
                QtTest.QTest.keyClick(widget, key)

            self.process_events(50)

    return GuiTestHelper(qt_application, qt_wait)


# ============================================================================
# Qt Configuration
# ============================================================================


def configure_qt_for_testing():
    """Configure Qt environment for testing."""
    # Set platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Disable Qt warnings in tests
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false")

    # Set Qt application attributes for testing
    if QT_AVAILABLE:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


# Auto-configure Qt when this module is imported
configure_qt_for_testing()
