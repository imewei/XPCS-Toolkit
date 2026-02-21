"""
Drag-and-drop enabled list view for XPCS-TOOLKIT GUI.

This module provides a QListView subclass that supports internal
drag-and-drop reordering of items.
"""

import logging

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import (
    QAbstractItemView,
    QListView,
    QModelIndex,
    Qt,
    Signal,
)

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
        from_index = self._drag_start_index

        # Call parent implementation to perform the actual move
        super().dropEvent(event)

        # Determine actual new position from current selection
        if from_index is not None:
            indexes = self.selectedIndexes()
            if indexes:
                target_row = indexes[0].row()
            else:
                target_row = from_index  # fallback: no change

            if from_index != target_row:
                logger.debug(f"Item moved from {from_index} to {target_row}")
                self.items_reordered.emit(from_index, target_row)

        self._drag_start_index = None

    def get_item_order(self) -> list[int]:
        """
        Get current item order as indices.

        Returns:
            List of original indices in current display order
        """
        model = self.model()
        # model check assumed valid by type checker or handled
        if not model:
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
        # model check assumed valid by type checker or handled
        if not model:
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
            # After pop, indices above from_index shift down by one.
            # When moving down, adjust to_index to compensate.
            insert_at = to_index if to_index <= from_index else to_index - 1
            strings.insert(insert_at, item)
            model.setStringList(strings)
            self.items_reordered.emit(from_index, to_index)
            return True

        return False
