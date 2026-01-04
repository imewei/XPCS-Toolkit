"""Unit tests for ToastNotification widgets."""

import pytest
from PySide6.QtWidgets import QMainWindow

from xpcsviewer.gui.widgets.toast_notification import (
    ToastManager,
    ToastType,
    ToastWidget,
)


class TestToastType:
    """Tests for ToastType enum."""

    def test_info_type(self):
        """INFO type should have 'info' value."""
        assert ToastType.INFO.value == "info"

    def test_success_type(self):
        """SUCCESS type should have 'success' value."""
        assert ToastType.SUCCESS.value == "success"

    def test_warning_type(self):
        """WARNING type should have 'warning' value."""
        assert ToastType.WARNING.value == "warning"

    def test_error_type(self):
        """ERROR type should have 'error' value."""
        assert ToastType.ERROR.value == "error"

    def test_all_types_exist(self):
        """All expected toast types should exist."""
        types = [t for t in ToastType]
        assert len(types) == 4


class TestToastWidget:
    """Tests for ToastWidget."""

    def test_widget_creation(self, qtbot):
        """ToastWidget should be created with message."""
        widget = ToastWidget(message="Test message")
        qtbot.addWidget(widget)

        assert widget is not None

    def test_widget_stores_message(self, qtbot):
        """ToastWidget should store the message."""
        widget = ToastWidget(message="Test message")
        qtbot.addWidget(widget)

        assert widget._message == "Test message"

    def test_widget_stores_type(self, qtbot):
        """ToastWidget should store the toast type."""
        widget = ToastWidget(message="Test", toast_type=ToastType.ERROR)
        qtbot.addWidget(widget)

        assert widget._toast_type == ToastType.ERROR

    def test_widget_default_type_is_info(self, qtbot):
        """ToastWidget should default to INFO type."""
        widget = ToastWidget(message="Test")
        qtbot.addWidget(widget)

        assert widget._toast_type == ToastType.INFO

    def test_widget_stores_duration(self, qtbot):
        """ToastWidget should store duration."""
        widget = ToastWidget(message="Test", duration_ms=5000)
        qtbot.addWidget(widget)

        assert widget._duration_ms == 5000

    def test_widget_default_duration(self, qtbot):
        """ToastWidget should have 3000ms default duration."""
        widget = ToastWidget(message="Test")
        qtbot.addWidget(widget)

        assert widget._duration_ms == 3000

    def test_widget_dismissible_default(self, qtbot):
        """ToastWidget should be dismissible by default."""
        widget = ToastWidget(message="Test")
        qtbot.addWidget(widget)

        assert widget._dismissible is True

    def test_widget_not_dismissible(self, qtbot):
        """ToastWidget can be set non-dismissible."""
        widget = ToastWidget(message="Test", dismissible=False)
        qtbot.addWidget(widget)

        assert widget._dismissible is False

    def test_widget_object_name_matches_type(self, qtbot):
        """ToastWidget objectName should reflect toast type."""
        widget = ToastWidget(message="Test", toast_type=ToastType.SUCCESS)
        qtbot.addWidget(widget)

        assert widget.objectName() == "toast_success"

    def test_widget_has_label(self, qtbot):
        """ToastWidget should have a label with the message."""
        widget = ToastWidget(message="Test message")
        qtbot.addWidget(widget)

        assert hasattr(widget, "_label")
        assert widget._label.text() == "Test message"

    def test_opacity_property(self, qtbot):
        """ToastWidget should have working opacity property."""
        widget = ToastWidget(message="Test")
        qtbot.addWidget(widget)

        # Default opacity
        assert widget.opacity == 1.0

        # Set opacity
        widget.opacity = 0.5
        assert widget.opacity == 0.5


class TestToastManager:
    """Tests for ToastManager."""

    @pytest.fixture
    def main_window(self, qtbot):
        """Create a main window for testing."""
        window = QMainWindow()
        window.resize(800, 600)
        qtbot.addWidget(window)
        return window

    def test_manager_creation(self, main_window):
        """ToastManager should be created with parent."""
        manager = ToastManager(parent=main_window)

        assert manager is not None
        assert manager._parent == main_window

    def test_manager_empty_toasts_initially(self, main_window):
        """ToastManager should start with no toasts."""
        manager = ToastManager(parent=main_window)

        assert len(manager._toasts) == 0

    def test_manager_default_duration(self, main_window):
        """ToastManager should have 3000ms default duration."""
        manager = ToastManager(parent=main_window)

        assert manager._default_duration_ms == 3000

    def test_set_default_duration(self, main_window):
        """ToastManager should allow setting default duration."""
        manager = ToastManager(parent=main_window)
        manager.set_default_duration(5000)

        assert manager._default_duration_ms == 5000

    def test_show_toast(self, main_window, qtbot):
        """show_toast should create and display a toast."""
        manager = ToastManager(parent=main_window)
        manager.show_toast("Test message")

        assert len(manager._toasts) == 1
        assert manager._toasts[0]._message == "Test message"

    def test_show_toast_with_type(self, main_window, qtbot):
        """show_toast should accept toast type."""
        manager = ToastManager(parent=main_window)
        manager.show_toast("Error message", toast_type=ToastType.ERROR)

        assert len(manager._toasts) == 1
        assert manager._toasts[0]._toast_type == ToastType.ERROR

    def test_show_info(self, main_window, qtbot):
        """show_info should create INFO toast."""
        manager = ToastManager(parent=main_window)
        manager.show_info("Info message")

        assert len(manager._toasts) == 1
        assert manager._toasts[0]._toast_type == ToastType.INFO

    def test_show_success(self, main_window, qtbot):
        """show_success should create SUCCESS toast."""
        manager = ToastManager(parent=main_window)
        manager.show_success("Success message")

        assert len(manager._toasts) == 1
        assert manager._toasts[0]._toast_type == ToastType.SUCCESS

    def test_show_warning(self, main_window, qtbot):
        """show_warning should create WARNING toast."""
        manager = ToastManager(parent=main_window)
        manager.show_warning("Warning message")

        assert len(manager._toasts) == 1
        assert manager._toasts[0]._toast_type == ToastType.WARNING

    def test_show_error(self, main_window, qtbot):
        """show_error should create ERROR toast."""
        manager = ToastManager(parent=main_window)
        manager.show_error("Error message")

        assert len(manager._toasts) == 1
        assert manager._toasts[0]._toast_type == ToastType.ERROR

    def test_multiple_toasts(self, main_window, qtbot):
        """ToastManager should support multiple toasts."""
        manager = ToastManager(parent=main_window)
        manager.show_info("Message 1")
        manager.show_warning("Message 2")
        manager.show_error("Message 3")

        assert len(manager._toasts) == 3

    def test_dismiss_all(self, main_window, qtbot):
        """dismiss_all should remove all toasts."""
        manager = ToastManager(parent=main_window)
        manager.show_info("Message 1")
        manager.show_info("Message 2")

        assert len(manager._toasts) == 2

        manager.dismiss_all()

        assert len(manager._toasts) == 0

    def test_toast_custom_duration(self, main_window, qtbot):
        """show_toast should accept custom duration."""
        manager = ToastManager(parent=main_window)
        manager.show_toast("Test", duration_ms=5000)

        assert manager._toasts[0]._duration_ms == 5000


class TestToastStyling:
    """Tests for toast CSS styling."""

    def test_info_toast_object_name(self, qtbot):
        """INFO toast should have correct objectName for CSS."""
        widget = ToastWidget(message="Test", toast_type=ToastType.INFO)
        qtbot.addWidget(widget)

        assert widget.objectName() == "toast_info"

    def test_success_toast_object_name(self, qtbot):
        """SUCCESS toast should have correct objectName for CSS."""
        widget = ToastWidget(message="Test", toast_type=ToastType.SUCCESS)
        qtbot.addWidget(widget)

        assert widget.objectName() == "toast_success"

    def test_warning_toast_object_name(self, qtbot):
        """WARNING toast should have correct objectName for CSS."""
        widget = ToastWidget(message="Test", toast_type=ToastType.WARNING)
        qtbot.addWidget(widget)

        assert widget.objectName() == "toast_warning"

    def test_error_toast_object_name(self, qtbot):
        """ERROR toast should have correct objectName for CSS."""
        widget = ToastWidget(message="Test", toast_type=ToastType.ERROR)
        qtbot.addWidget(widget)

        assert widget.objectName() == "toast_error"
