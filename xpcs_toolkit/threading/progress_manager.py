"""
Progress management system for XPCS Toolkit.

This module provides progress indicators, status bars, and cancellation
controls for long-running operations in the GUI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OperationInfo:
    """Information about a running operation."""

    operation_id: str
    description: str
    start_time: float
    current: int = 0
    total: int = 100
    status: str = "Starting..."
    is_cancellable: bool = True
    estimated_completion: float | None = None


class ProgressIndicator(QWidget):
    """
    Individual progress indicator widget for a single operation.
    """

    cancel_requested = Signal(str)  # operation_id

    def __init__(self, operation_info: OperationInfo, parent=None):
        super().__init__(parent)
        self.operation_info = operation_info
        self.setup_ui()

    def setup_ui(self):
        """Set up the progress indicator UI."""
        layout = QVBoxLayout(self)

        # Description label
        self.description_label = QLabel(self.operation_info.description)
        self.description_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.description_label)

        # Progress bar and status layout
        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(self.operation_info.total)
        self.progress_bar.setValue(self.operation_info.current)
        progress_layout.addWidget(self.progress_bar)

        # Cancel button
        if self.operation_info.is_cancellable:
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.setMaximumWidth(80)
            self.cancel_btn.clicked.connect(self.cancel_operation)
            progress_layout.addWidget(self.cancel_btn)

        layout.addLayout(progress_layout)

        # Status label
        self.status_label = QLabel(self.operation_info.status)
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)

        # ETA label
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.eta_label)

        self.setMaximumHeight(120)
        self.setContentsMargins(10, 5, 10, 5)

    def update_progress(self, current: int, total: int, status: str = ""):
        """Update the progress indicator."""
        self.operation_info.current = current
        self.operation_info.total = total
        if status:
            self.operation_info.status = status

        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(self.operation_info.status)

        # Calculate and display ETA
        self._update_eta()

    def _update_eta(self):
        """Calculate and update estimated time to completion."""
        if self.operation_info.current <= 0 or self.operation_info.total <= 0:
            return

        elapsed = time.time() - self.operation_info.start_time
        if elapsed < 1.0:  # Wait at least 1 second for stable estimate
            return

        progress_ratio = self.operation_info.current / self.operation_info.total
        if progress_ratio <= 0:
            return

        estimated_total_time = elapsed / progress_ratio
        remaining_time = estimated_total_time - elapsed

        if remaining_time > 0:
            if remaining_time < 60:
                eta_text = f"ETA: {int(remaining_time)}s"
            elif remaining_time < 3600:
                minutes = int(remaining_time / 60)
                seconds = int(remaining_time % 60)
                eta_text = f"ETA: {minutes}m {seconds}s"
            else:
                hours = int(remaining_time / 3600)
                minutes = int((remaining_time % 3600) / 60)
                eta_text = f"ETA: {hours}h {minutes}m"

            self.eta_label.setText(eta_text)

    def set_completed(self, success: bool = True):
        """Mark the operation as completed."""
        if success:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.status_label.setText("Completed")
            self.status_label.setStyleSheet("color: #2E7D32; font-size: 11px;")
        else:
            self.status_label.setText("Failed")
            self.status_label.setStyleSheet("color: #C62828; font-size: 11px;")

        elapsed = time.time() - self.operation_info.start_time
        self.eta_label.setText(f"Total time: {elapsed:.1f}s")

        if hasattr(self, "cancel_btn"):
            self.cancel_btn.setEnabled(False)

    def set_cancelled(self):
        """Mark the operation as cancelled."""
        self.status_label.setText("Cancelled")
        self.status_label.setStyleSheet("color: #FF9800; font-size: 11px;")
        elapsed = time.time() - self.operation_info.start_time
        self.eta_label.setText(f"Cancelled after {elapsed:.1f}s")

        if hasattr(self, "cancel_btn"):
            self.cancel_btn.setEnabled(False)

    @Slot()
    def cancel_operation(self):
        """Request cancellation of the operation."""
        self.cancel_requested.emit(self.operation_info.operation_id)
        if hasattr(self, "cancel_btn"):
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")


class ProgressDialog(QDialog):
    """
    Dialog for showing progress of multiple operations.
    """

    cancel_operation = Signal(str)  # operation_id
    cancel_all_operations = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.operation_widgets = {}  # operation_id -> ProgressIndicator
        self.setup_ui()

    def setup_ui(self):
        """Set up the progress dialog UI."""
        self.setWindowTitle("Operation Progress")
        self.setMinimumSize(500, 300)
        self.setModal(False)  # Allow interaction with main window

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Active Operations")
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Progress indicators container
        self.progress_container = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_container)
        self.progress_layout.setSpacing(5)

        # Scroll area for multiple operations
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.progress_container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll)

        # Buttons
        button_layout = QHBoxLayout()

        self.cancel_all_btn = QPushButton("Cancel All")
        self.cancel_all_btn.clicked.connect(self.cancel_all_operations.emit)
        button_layout.addWidget(self.cancel_all_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.hide)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def add_operation(self, operation_info: OperationInfo):
        """Add a new operation to track."""
        if operation_info.operation_id in self.operation_widgets:
            return  # Already tracking this operation

        indicator = ProgressIndicator(operation_info)
        indicator.cancel_requested.connect(self.cancel_operation.emit)

        self.operation_widgets[operation_info.operation_id] = indicator
        self.progress_layout.addWidget(indicator)

        # Show dialog if not visible
        if not self.isVisible():
            self.show()

    def update_operation(
        self, operation_id: str, current: int, total: int, status: str = ""
    ):
        """Update progress for an operation."""
        if operation_id in self.operation_widgets:
            self.operation_widgets[operation_id].update_progress(current, total, status)

    def complete_operation(self, operation_id: str, success: bool = True):
        """Mark an operation as completed."""
        if operation_id in self.operation_widgets:
            self.operation_widgets[operation_id].set_completed(success)

            # Auto-hide if all operations are complete
            QTimer.singleShot(
                2000, lambda: self._remove_completed_operation(operation_id)
            )

    def cancel_operation_ui(self, operation_id: str):
        """Mark an operation as cancelled in the UI."""
        if operation_id in self.operation_widgets:
            self.operation_widgets[operation_id].set_cancelled()
            QTimer.singleShot(
                1000, lambda: self._remove_completed_operation(operation_id)
            )

    def _remove_completed_operation(self, operation_id: str):
        """Remove a completed operation from the dialog."""
        if operation_id in self.operation_widgets:
            widget = self.operation_widgets[operation_id]
            self.progress_layout.removeWidget(widget)
            widget.deleteLater()
            del self.operation_widgets[operation_id]

            # Hide dialog if no operations remaining
            if not self.operation_widgets:
                self.hide()


class ProgressManager(QObject):
    """
    Centralized progress management for the application.
    """

    # Signals for operation lifecycle
    operation_started = Signal(str, str)  # operation_id, description
    operation_progress = Signal(str, int, int, str)  # op_id, current, total, status
    operation_completed = Signal(str, bool)  # operation_id, success
    operation_cancelled = Signal(str)  # operation_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_operations: dict[str, OperationInfo] = {}
        self.progress_dialog = None
        self.statusbar = None
        self.current_statusbar_operation = None

    def set_statusbar(self, statusbar: QtWidgets.QStatusBar):
        """Set the main window status bar for simple progress display."""
        self.statusbar = statusbar

    def start_operation(
        self,
        operation_id: str,
        description: str,
        total: int = 100,
        is_cancellable: bool = True,
        show_in_statusbar: bool = False,
    ) -> str:
        """
        Start tracking a new operation.

        Args:
            operation_id: Unique identifier for the operation
            description: Human-readable description
            total: Total progress units
            is_cancellable: Whether the operation can be cancelled
            show_in_statusbar: Whether to show progress in status bar

        Returns:
            The operation ID for reference
        """
        operation_info = OperationInfo(
            operation_id=operation_id,
            description=description,
            start_time=time.time(),
            total=total,
            is_cancellable=is_cancellable,
        )

        self.active_operations[operation_id] = operation_info

        # Show in progress dialog
        if self.progress_dialog is None:
            self.progress_dialog = ProgressDialog()
            self.progress_dialog.cancel_operation.connect(self.request_cancel)
            self.progress_dialog.cancel_all_operations.connect(self.cancel_all)

        self.progress_dialog.add_operation(operation_info)

        # Show in status bar if requested and available
        if show_in_statusbar and self.statusbar:
            self.current_statusbar_operation = operation_id
            self.statusbar.showMessage(f"{description}...")

        self.operation_started.emit(operation_id, description)
        logger.info(f"Started operation: {operation_id} - {description}")

        return operation_id

    def update_progress(
        self,
        operation_id: str,
        current: int,
        total: int | None = None,
        status: str = "",
    ):
        """Update progress for an operation."""
        if operation_id not in self.active_operations:
            return

        operation_info = self.active_operations[operation_id]

        if total is not None:
            operation_info.total = total
        operation_info.current = current
        if status:
            operation_info.status = status

        # Update progress dialog
        if self.progress_dialog:
            self.progress_dialog.update_operation(
                operation_id, current, operation_info.total, status
            )

        # Update status bar if this is the current operation
        if (
            self.current_statusbar_operation == operation_id
            and self.statusbar
            and status
        ):
            progress_pct = (
                int((current / operation_info.total) * 100)
                if operation_info.total > 0
                else 0
            )
            self.statusbar.showMessage(
                f"{operation_info.description}: {status} ({progress_pct}%)"
            )

        self.operation_progress.emit(
            operation_id, current, operation_info.total, status
        )

    def complete_operation(
        self, operation_id: str, success: bool = True, final_message: str = ""
    ):
        """Mark an operation as completed."""
        if operation_id not in self.active_operations:
            return

        operation_info = self.active_operations[operation_id]
        elapsed = time.time() - operation_info.start_time

        # Update progress dialog
        if self.progress_dialog:
            self.progress_dialog.complete_operation(operation_id, success)

        # Update status bar
        if self.current_statusbar_operation == operation_id and self.statusbar:
            if success:
                message = final_message or f"{operation_info.description} completed"
            else:
                message = final_message or f"{operation_info.description} failed"
            self.statusbar.showMessage(f"{message} ({elapsed:.1f}s)", 3000)
            self.current_statusbar_operation = None

        self.operation_completed.emit(operation_id, success)

        # Clean up
        del self.active_operations[operation_id]

        status_text = "successfully" if success else "with errors"
        logger.info(
            f"Operation {operation_id} completed {status_text} in {elapsed:.1f}s"
        )

    def cancel_operation_ui(self, operation_id: str):
        """Handle operation cancellation in the UI."""
        if operation_id not in self.active_operations:
            return

        operation_info = self.active_operations[operation_id]
        elapsed = time.time() - operation_info.start_time

        # Update progress dialog
        if self.progress_dialog:
            self.progress_dialog.cancel_operation_ui(operation_id)

        # Update status bar
        if self.current_statusbar_operation == operation_id and self.statusbar:
            self.statusbar.showMessage(
                f"{operation_info.description} cancelled ({elapsed:.1f}s)", 3000
            )
            self.current_statusbar_operation = None

        self.operation_cancelled.emit(operation_id)

        # Clean up
        del self.active_operations[operation_id]

        logger.info(f"Operation {operation_id} cancelled after {elapsed:.1f}s")

    @Slot(str)
    def request_cancel(self, operation_id: str):
        """Request cancellation of an operation (to be connected to actual cancellation)."""
        logger.info(f"Cancel requested for operation: {operation_id}")
        # This signal should be connected to the actual cancellation mechanism
        # For now, just update the UI
        self.cancel_operation_ui(operation_id)

    @Slot()
    def cancel_all(self):
        """Request cancellation of all operations."""
        operation_ids = list(self.active_operations.keys())
        for operation_id in operation_ids:
            self.request_cancel(operation_id)

    def is_operation_active(self, operation_id: str) -> bool:
        """Check if an operation is currently active."""
        return operation_id in self.active_operations

    def get_active_operations(self) -> set[str]:
        """Get set of active operation IDs."""
        return set(self.active_operations.keys())

    def show_progress_dialog(self):
        """Show the progress dialog."""
        if self.progress_dialog:
            self.progress_dialog.show()
            self.progress_dialog.raise_()

    def hide_progress_dialog(self):
        """Hide the progress dialog."""
        if self.progress_dialog:
            self.progress_dialog.hide()
