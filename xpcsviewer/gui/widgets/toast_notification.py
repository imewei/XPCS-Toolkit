"""
Toast notification system for XPCS-TOOLKIT GUI.

This module provides non-blocking popup notifications for
operation feedback, errors, and warnings.
"""

from enum import Enum

from PySide6.QtCore import Property, QPoint, Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget


class ToastType(Enum):
    """Types of toast notifications with associated styling."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class ToastWidget(QFrame):
    """Individual toast notification widget with fade animation."""

    def __init__(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration_ms: int = 3000,
        dismissible: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        """
        Initialize a toast widget.

        Args:
            message: Text to display
            toast_type: Type determining color/icon
            duration_ms: Time before auto-dismiss (0 = no auto-dismiss)
            dismissible: Whether clicking dismisses the toast
            parent: Parent widget
        """
        super().__init__(parent)
        self._opacity = 1.0
        self._message = message
        self._toast_type = toast_type
        self._duration_ms = duration_ms
        self._dismissible = dismissible

        self._setup_ui()
        self._setup_style()

        if dismissible:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _setup_ui(self) -> None:
        """Set up the widget layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self._label = QLabel(self._message)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self.setFixedWidth(300)
        self.adjustSize()

    def _setup_style(self) -> None:
        """Apply styling based on toast type."""
        self.setObjectName(f"toast_{self._toast_type.value}")

    def mousePressEvent(self, event) -> None:
        """Handle click to dismiss."""
        if self._dismissible:
            self.close()
        super().mousePressEvent(event)

    def get_opacity(self) -> float:
        """Get current opacity."""
        return self._opacity

    def set_opacity(self, value: float) -> None:
        """Set opacity and update window."""
        self._opacity = value
        self.setWindowOpacity(value)

    opacity = Property(float, get_opacity, set_opacity)


class ToastManager:
    """
    Manages toast notifications for a parent window.

    Handles creating, displaying, stacking, and dismissing toasts.
    """

    def __init__(self, parent: QWidget) -> None:
        """
        Initialize the ToastManager.

        Args:
            parent: Parent window for toast positioning
        """
        self._parent = parent
        self._toasts: list[ToastWidget] = []
        self._default_duration_ms = 3000
        self._margin = 16
        self._spacing = 8

    def show_toast(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration_ms: int | None = None,
        dismissible: bool = True,
    ) -> None:
        """
        Show a toast notification.

        Args:
            message: Text to display
            toast_type: Type determining color/icon
            duration_ms: Time before auto-dismiss (0 = no auto-dismiss)
            dismissible: Whether clicking dismisses the toast
        """
        if duration_ms is None:
            duration_ms = self._default_duration_ms

        toast = ToastWidget(
            message=message,
            toast_type=toast_type,
            duration_ms=duration_ms,
            dismissible=dismissible,
            parent=self._parent,
        )

        self._toasts.append(toast)
        self._position_toasts()
        toast.show()

        # Set up auto-dismiss
        if duration_ms > 0:
            QTimer.singleShot(duration_ms, lambda: self._dismiss_toast(toast))

    def show_info(self, message: str) -> None:
        """Show an info toast."""
        self.show_toast(message, ToastType.INFO)

    def show_success(self, message: str) -> None:
        """Show a success toast."""
        self.show_toast(message, ToastType.SUCCESS)

    def show_warning(self, message: str) -> None:
        """Show a warning toast."""
        self.show_toast(message, ToastType.WARNING)

    def show_error(self, message: str) -> None:
        """Show an error toast."""
        self.show_toast(message, ToastType.ERROR)

    def dismiss_all(self) -> None:
        """Dismiss all visible toasts immediately."""
        for toast in self._toasts[:]:
            self._dismiss_toast(toast)

    def set_default_duration(self, duration_ms: int) -> None:
        """
        Set the default duration for toasts.

        Args:
            duration_ms: Default duration in milliseconds
        """
        self._default_duration_ms = duration_ms

    def _dismiss_toast(self, toast: ToastWidget) -> None:
        """Dismiss a specific toast."""
        if toast in self._toasts:
            self._toasts.remove(toast)
            # Check if the C++ object is still valid before operating on it
            try:
                toast.close()
                toast.deleteLater()
            except RuntimeError:
                # C++ object already deleted (e.g., parent window closed)
                pass
            self._position_toasts()

    def _position_toasts(self) -> None:
        """Position all toasts in bottom-right corner."""
        if not self._parent:
            return

        try:
            parent_rect = self._parent.rect()
        except RuntimeError:
            # Parent C++ object already deleted
            return
        y_offset = self._margin

        for toast in reversed(self._toasts):
            x = parent_rect.right() - toast.width() - self._margin
            y = parent_rect.bottom() - toast.height() - y_offset
            toast.move(QPoint(x, y))
            y_offset += toast.height() + self._spacing
