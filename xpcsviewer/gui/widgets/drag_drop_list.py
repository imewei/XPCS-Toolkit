"""
Drag-and-drop enabled list view for XPCS-TOOLKIT GUI.

This module provides a QListView subclass that supports internal
drag-and-drop reordering of items.
"""

import logging

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QListView

logger = logging.getLogger(__name__)


class DragDropListView(QListView):
    """
    QListView subclass with internal drag-and-drop reordering.

    Emits items_reordered signal when items are moved via drag-and-drop.
    """

    # Signal emitted when items are reordered
    # Signature: items_reordered(old_index: int, new_index: int)
    items_reordered = Signal(int, int)

    def __init__(self, parent=None) -> None:
        """
        Initialize the DragDropListView.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._drag_enabled = True
        self._drag_start_index: int | None = None
        self._setup_drag_drop()

    def _setup_drag_drop(self) -> None:
        """Configure drag-and-drop settings."""
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def set_drag_enabled(self, enabled: bool) -> None:
        """
        Enable or disable drag-and-drop.

        Args:
            enabled: Whether drag-drop is enabled
        """
        self._drag_enabled = enabled
        self.setDragEnabled(enabled)
        self.setAcceptDrops(enabled)

    def is_drag_enabled(self) -> bool:
        """
        Check if drag-and-drop is enabled.

        Returns:
            True if enabled
        """
        return self._drag_enabled

    def startDrag(self, supportedActions: Qt.DropAction) -> None:
        """
        Override to track the starting index of a drag operation.

        Args:
            supportedActions: The drop actions supported
        """
        indexes = self.selectedIndexes()
        if indexes:
            self._drag_start_index = indexes[0].row()
            logger.debug(f"Drag started from row {self._drag_start_index}")
        super().startDrag(supportedActions)

    def dropEvent(self, event) -> None:
        """
        Handle drop event and emit items_reordered signal.

        Args:
            event: The drop event
        """
        # Get the target drop position
        drop_index = self.indexAt(event.position().toPoint())

        # Calculate the target row
        if drop_index.isValid():
            target_row = drop_index.row()
        else:
            # Dropped at the end
            model = self.model()
            target_row = model.rowCount() - 1 if model else 0

        # Call parent implementation to perform the actual move
        super().dropEvent(event)

        # Emit signal if we have a valid start index and it changed
        if self._drag_start_index is not None and self._drag_start_index != target_row:
            logger.debug(f"Item moved from {self._drag_start_index} to {target_row}")
            self.items_reordered.emit(self._drag_start_index, target_row)

        self._drag_start_index = None

    def get_item_order(self) -> list[int]:
        """
        Get current item order as indices.

        Returns:
            List of original indices in current display order
        """
        model = self.model()
        if model is None:
            return []

        order = []
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            # Get original index from user data if stored
            original_idx = model.data(index, Qt.ItemDataRole.UserRole)
            if original_idx is not None:
                order.append(original_idx)
            else:
                order.append(row)
        return order

    def move_item(self, from_index: int, to_index: int) -> bool:
        """
        Programmatically move an item from one position to another.

        This is useful for keyboard-based reordering (fallback to buttons).

        Args:
            from_index: Source row index
            to_index: Target row index

        Returns:
            True if move was successful
        """
        model = self.model()
        if model is None:
            return False

        row_count = model.rowCount()
        if from_index < 0 or from_index >= row_count:
            return False
        if to_index < 0 or to_index >= row_count:
            return False
        if from_index == to_index:
            return False

        # Use model's moveRows if available (QStringListModel doesn't have it)
        # Fall back to manual item manipulation
        if hasattr(model, "moveRow"):
            success = model.moveRow(QModelIndex(), from_index, QModelIndex(), to_index)
            if success:
                self.items_reordered.emit(from_index, to_index)
            return success

        # Manual fallback for models without moveRow
        if hasattr(model, "stringList") and hasattr(model, "setStringList"):
            # QStringListModel support
            strings = model.stringList()
            item = strings.pop(from_index)
            # Adjust target index if moving down (indices shift after pop)
            # When moving down, the target index is already correct after pop
            # When moving up, target index doesn't need adjustment
            if to_index > from_index:
                # After removing item at from_index, target is now one less
                # But we want to insert AFTER the item at to_index
                # So no adjustment needed for "insert at position"
                pass
            strings.insert(to_index, item)
            model.setStringList(strings)
            self.items_reordered.emit(from_index, to_index)
            return True

        return False
