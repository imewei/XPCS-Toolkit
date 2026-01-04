"""Unit tests for DragDropListView widget."""

import pytest
from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtWidgets import QAbstractItemView

from xpcsviewer.gui.widgets.drag_drop_list import DragDropListView


class TestDragDropListViewInit:
    """Tests for DragDropListView initialization."""

    def test_default_initialization(self, qtbot):
        """DragDropListView should initialize with drag-drop enabled."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        assert widget.dragEnabled() is True
        assert widget.acceptDrops() is True
        assert widget.showDropIndicator() is True
        assert widget.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove

    def test_default_drag_enabled(self, qtbot):
        """is_drag_enabled should return True by default."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        assert widget.is_drag_enabled() is True

    def test_set_drag_enabled_false(self, qtbot):
        """set_drag_enabled(False) should disable drag-drop."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        widget.set_drag_enabled(False)

        assert widget.is_drag_enabled() is False
        assert widget.dragEnabled() is False
        assert widget.acceptDrops() is False

    def test_set_drag_enabled_true(self, qtbot):
        """set_drag_enabled(True) should enable drag-drop."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        widget.set_drag_enabled(False)
        widget.set_drag_enabled(True)

        assert widget.is_drag_enabled() is True
        assert widget.dragEnabled() is True
        assert widget.acceptDrops() is True


class TestDragDropListViewWithModel:
    """Tests for DragDropListView with a model attached."""

    def test_get_item_order_empty_model(self, qtbot):
        """get_item_order should return empty list for empty model."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel([])
        widget.setModel(model)

        assert widget.get_item_order() == []

    def test_get_item_order_with_items(self, qtbot):
        """get_item_order should return indices in order."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel(["Item 1", "Item 2", "Item 3"])
        widget.setModel(model)

        # Without UserRole data, returns row indices
        order = widget.get_item_order()
        assert order == [0, 1, 2]

    def test_get_item_order_no_model(self, qtbot):
        """get_item_order should return empty list when no model."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        assert widget.get_item_order() == []


class TestDragDropListViewMoveItem:
    """Tests for programmatic item movement."""

    def test_move_item_no_model(self, qtbot):
        """move_item should return False when no model."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        result = widget.move_item(0, 1)
        assert result is False

    def test_move_item_invalid_from_index(self, qtbot):
        """move_item should return False for invalid source index."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel(["A", "B", "C"])
        widget.setModel(model)

        assert widget.move_item(-1, 1) is False
        assert widget.move_item(10, 1) is False

    def test_move_item_invalid_to_index(self, qtbot):
        """move_item should return False for invalid target index."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel(["A", "B", "C"])
        widget.setModel(model)

        assert widget.move_item(0, -1) is False
        assert widget.move_item(0, 10) is False

    def test_move_item_same_index(self, qtbot):
        """move_item should return False when source equals target."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel(["A", "B", "C"])
        widget.setModel(model)

        assert widget.move_item(1, 1) is False

    def test_move_item_success(self, qtbot):
        """move_item should successfully reorder items."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel(["A", "B", "C"])
        widget.setModel(model)

        # Move A from index 0 to index 2
        # QStringListModel.moveRow inserts BEFORE the destination index
        # So moving from 0 to 2: "A" goes before index 2 ("C"), giving ["B", "A", "C"]
        result = widget.move_item(0, 2)
        assert result is True

        # Check new order
        new_list = model.stringList()
        assert new_list == ["B", "A", "C"]

    def test_move_item_emits_signal(self, qtbot):
        """move_item should emit items_reordered signal on success."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        model = QStringListModel(["A", "B", "C"])
        widget.setModel(model)

        with qtbot.waitSignal(widget.items_reordered, timeout=1000) as blocker:
            widget.move_item(0, 2)

        assert blocker.args == [0, 2]


class TestDragDropListViewSignals:
    """Tests for signal emission."""

    def test_signal_exists(self, qtbot):
        """DragDropListView should have items_reordered signal."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        # Signal should be accessible
        assert hasattr(widget, "items_reordered")

    def test_signal_can_connect(self, qtbot):
        """items_reordered signal should be connectable."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        received = []

        def on_reordered(old_idx, new_idx):
            received.append((old_idx, new_idx))

        widget.items_reordered.connect(on_reordered)

        # Manually emit to test connection
        widget.items_reordered.emit(0, 2)

        assert received == [(0, 2)]


class TestDragDropListViewSelection:
    """Tests for selection mode."""

    def test_default_selection_mode(self, qtbot):
        """Default selection mode should be SingleSelection."""
        widget = DragDropListView()
        qtbot.addWidget(widget)

        assert widget.selectionMode() == QAbstractItemView.SelectionMode.SingleSelection
