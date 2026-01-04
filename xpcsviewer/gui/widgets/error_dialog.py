"""
Error dialog for XPCS Viewer.

Provides actionable error dialogs with cause/remedy information,
retry buttons, and log access links.
"""

from collections.abc import Callable

from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


class ErrorDialog(QDialog):
    """
    Actionable error dialog with cause, remedy, and retry options.

    Displays error information in a user-friendly format with:
    - Error message and cause
    - Suggested remedy
    - Retry button (if applicable)
    - View Logs link
    - Copy Details button
    """

    def __init__(
        self,
        title: str = "Error",
        message: str = "An error occurred.",
        cause: str = "",
        remedy: str = "",
        details: str = "",
        retry_callback: Callable | None = None,
        parent=None,
    ):
        """
        Initialize the error dialog.

        Args:
            title: Dialog window title
            message: Main error message
            cause: Explanation of what caused the error
            remedy: Suggested action to fix the error
            details: Technical details (traceback, etc.)
            retry_callback: Function to call when Retry is clicked
            parent: Parent widget
        """
        super().__init__(parent)
        self._retry_callback = retry_callback
        self._details = details

        self.setWindowTitle(title)
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui(message, cause, remedy, details)

    def _setup_ui(self, message: str, cause: str, remedy: str, details: str) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Error icon and message
        header_layout = QHBoxLayout()

        icon_label = QLabel("❌")
        icon_label.setStyleSheet("font-size: 32px;")
        header_layout.addWidget(icon_label)

        message_label = QLabel(message)
        message_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        message_label.setWordWrap(True)
        header_layout.addWidget(message_label, stretch=1)

        layout.addLayout(header_layout)

        # Cause section
        if cause:
            cause_label = QLabel(f"<b>Cause:</b> {cause}")
            cause_label.setWordWrap(True)
            cause_label.setStyleSheet("color: #666;")
            layout.addWidget(cause_label)

        # Remedy section
        if remedy:
            remedy_label = QLabel(f"<b>Remedy:</b> {remedy}")
            remedy_label.setWordWrap(True)
            remedy_label.setStyleSheet("color: #2E7D32;")
            layout.addWidget(remedy_label)

        # Details section (collapsible)
        if details:
            self.details_widget = QTextEdit()
            self.details_widget.setPlainText(details)
            self.details_widget.setReadOnly(True)
            self.details_widget.setMaximumHeight(150)
            self.details_widget.setVisible(False)  # Hidden by default
            self.details_widget.setStyleSheet(
                """
                QTextEdit {
                    font-family: monospace;
                    font-size: 11px;
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                """
            )
            layout.addWidget(self.details_widget)

            # Toggle button
            self.toggle_details_btn = QPushButton("Show Details ▼")
            self.toggle_details_btn.setFlat(True)
            self.toggle_details_btn.clicked.connect(self._toggle_details)
            layout.addWidget(self.toggle_details_btn)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # Retry button (if callback provided)
        if self._retry_callback:
            retry_btn = QPushButton("Retry")
            retry_btn.clicked.connect(self._on_retry)
            button_layout.addWidget(retry_btn)

        # View Logs button
        logs_btn = QPushButton("View Logs")
        logs_btn.setFlat(True)
        logs_btn.clicked.connect(self._open_logs)
        button_layout.addWidget(logs_btn)

        # Copy Details button
        if details:
            copy_btn = QPushButton("Copy Details")
            copy_btn.setFlat(True)
            copy_btn.clicked.connect(self._copy_details)
            button_layout.addWidget(copy_btn)

        button_layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _toggle_details(self) -> None:
        """Toggle visibility of details section."""
        is_visible = self.details_widget.isVisible()
        self.details_widget.setVisible(not is_visible)
        self.toggle_details_btn.setText(
            "Hide Details ▲" if not is_visible else "Show Details ▼"
        )

    def _on_retry(self) -> None:
        """Handle retry button click."""
        self.accept()
        if self._retry_callback:
            self._retry_callback()

    def _open_logs(self) -> None:
        """Open the logs directory."""
        from pathlib import Path

        from PySide6.QtCore import QUrl

        log_dir = Path.home() / ".xpcsviewer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    def _copy_details(self) -> None:
        """Copy error details to clipboard."""
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        clipboard.setText(self._details)


# Convenience functions for common error types


def show_file_error(
    parent,
    file_path: str,
    error: str,
    retry_callback: Callable | None = None,
) -> None:
    """Show error dialog for file-related errors."""
    dialog = ErrorDialog(
        title="File Error",
        message=f"Could not access file: {file_path}",
        cause=error,
        remedy="Check that the file exists and you have permission to read it.",
        details=f"File: {file_path}\nError: {error}",
        retry_callback=retry_callback,
        parent=parent,
    )
    dialog.exec()


def show_data_error(
    parent,
    data_type: str,
    error: str,
    details: str = "",
    retry_callback: Callable | None = None,
) -> None:
    """Show error dialog for data-related errors."""
    dialog = ErrorDialog(
        title="Data Error",
        message=f"Error processing {data_type}",
        cause=error,
        remedy="Try selecting different data or check the file format.",
        details=details,
        retry_callback=retry_callback,
        parent=parent,
    )
    dialog.exec()


def show_plot_error(
    parent,
    plot_type: str,
    error: str,
    details: str = "",
    retry_callback: Callable | None = None,
) -> None:
    """Show error dialog for plotting errors."""
    dialog = ErrorDialog(
        title="Plot Error",
        message=f"Could not generate {plot_type} plot",
        cause=error,
        remedy="Check that the data contains the required fields for this plot type.",
        details=details,
        retry_callback=retry_callback,
        parent=parent,
    )
    dialog.exec()


def show_connection_error(
    parent,
    target: str,
    error: str,
    retry_callback: Callable | None = None,
) -> None:
    """Show error dialog for connection/network errors."""
    dialog = ErrorDialog(
        title="Connection Error",
        message=f"Could not connect to {target}",
        cause=error,
        remedy="Check your network connection and try again.",
        details=f"Target: {target}\nError: {error}",
        retry_callback=retry_callback,
        parent=parent,
    )
    dialog.exec()
