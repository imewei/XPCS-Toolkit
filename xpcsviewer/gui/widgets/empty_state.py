"""
Empty state widget for XPCS Viewer.

Displays informative messages when no data is available for display,
with optional action buttons to guide users.
"""

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EmptyStateWidget(QFrame):
    """
    Reusable empty state widget with icon, message, and action button.

    Displays a centered message with optional icon and action button
    when there's no data to show.
    """

    def __init__(
        self,
        message: str = "No data available",
        description: str = "",
        action_text: str = "",
        action_callback: Callable | None = None,
        icon_text: str = "📭",
        parent: QWidget | None = None,
    ):
        """
        Initialize the empty state widget.

        Args:
            message: Primary message to display
            description: Secondary description text
            action_text: Text for the action button (hidden if empty)
            action_callback: Function to call when action button clicked
            icon_text: Emoji or text to display as icon
            parent: Parent widget
        """
        super().__init__(parent)
        self._action_callback = action_callback
        self._setup_ui(message, description, action_text, icon_text)

    def _setup_ui(
        self, message: str, description: str, action_text: str, icon_text: str
    ) -> None:
        """Set up the widget UI."""
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet(
            """
            EmptyStateWidget {
                background-color: transparent;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        # Icon
        self.icon_label = QLabel(icon_text)
        self.icon_label.setStyleSheet("font-size: 48px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        # Message
        self.message_label = QLabel(message)
        self.message_label.setStyleSheet(
            """
            font-size: 16px;
            font-weight: bold;
            color: #666;
            """
        )
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        # Description
        if description:
            self.description_label = QLabel(description)
            self.description_label.setStyleSheet(
                """
                font-size: 13px;
                color: #888;
                """
            )
            self.description_label.setAlignment(Qt.AlignCenter)
            self.description_label.setWordWrap(True)
            layout.addWidget(self.description_label)
        else:
            self.description_label = None

        # Action button
        if action_text and self._action_callback:
            button_layout = QHBoxLayout()
            button_layout.setAlignment(Qt.AlignCenter)

            self.action_button = QPushButton(action_text)
            self.action_button.setStyleSheet(
                """
                QPushButton {
                    padding: 8px 24px;
                    font-size: 13px;
                }
                """
            )
            self.action_button.clicked.connect(self._on_action_clicked)
            button_layout.addWidget(self.action_button)

            layout.addLayout(button_layout)
        else:
            self.action_button = None

    def _on_action_clicked(self) -> None:
        """Handle action button click."""
        if self._action_callback:
            self._action_callback()

    def set_message(self, message: str) -> None:
        """Update the primary message."""
        self.message_label.setText(message)

    def set_description(self, description: str) -> None:
        """Update the description text."""
        if self.description_label:
            self.description_label.setText(description)
            self.description_label.setVisible(bool(description))

    def set_icon(self, icon_text: str) -> None:
        """Update the icon."""
        self.icon_label.setText(icon_text)


# Predefined empty states for common scenarios
class NoFilesLoadedState(EmptyStateWidget):
    """Empty state for when no files are loaded."""

    def __init__(
        self, browse_callback: Callable | None = None, parent: QWidget | None = None
    ):
        super().__init__(
            message="No data loaded",
            description="Browse to a folder containing XPCS data files to get started.",
            action_text="Browse for Data",
            action_callback=browse_callback,
            icon_text="📂",
            parent=parent,
        )


class NoFilesSelectedState(EmptyStateWidget):
    """Empty state for when no files are selected for analysis."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            message="No files selected",
            description="Add files from the source list to analyze.",
            action_text="",
            action_callback=None,
            icon_text="📋",
            parent=parent,
        )


class NoDataToPlotState(EmptyStateWidget):
    """Empty state for when there's no data available for plotting."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            message="No data to display",
            description="Select files and configure plot settings.",
            action_text="",
            action_callback=None,
            icon_text="📊",
            parent=parent,
        )


class FeatureNotAvailableState(EmptyStateWidget):
    """Empty state for when a feature isn't available for the current data."""

    def __init__(self, feature_name: str = "Feature", parent: QWidget | None = None):
        super().__init__(
            message=f"{feature_name} not available",
            description="This feature is not available for the current dataset.",
            action_text="",
            action_callback=None,
            icon_text="🚫",
            parent=parent,
        )
