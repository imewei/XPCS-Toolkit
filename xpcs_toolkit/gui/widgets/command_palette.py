"""
Command palette for XPCS-TOOLKIT GUI.

This module provides a searchable command palette dialog
for executing actions via keyboard.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class CommandAction:
    """A command that can be executed from the palette."""

    id: str  # Unique identifier
    name: str  # Display name
    category: str  # Category for grouping
    callback: Callable[[], None]  # Function to execute
    shortcut: str | None = None  # Display shortcut (optional)
    enabled: Callable[[], bool] | None = None  # Enabled check (optional)


class CommandPalette(QDialog):
    """
    Searchable dialog for executing registered actions.

    Provides fuzzy search across registered actions with
    keyboard navigation.
    """

    # Signal emitted when an action is selected
    # Signature: action_triggered(action_id: str)
    action_triggered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the CommandPalette.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._actions: dict[str, CommandAction] = {}
        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        self.setWindowTitle("Command Palette")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setMinimumWidth(400)
        self.setMaximumHeight(400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Search input
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Type to search actions...")
        self._search_input.textChanged.connect(self._filter_actions)
        layout.addWidget(self._search_input)

        # Results list
        self._results_list = QListWidget()
        self._results_list.itemActivated.connect(self._on_item_activated)
        layout.addWidget(self._results_list)

        self.setObjectName("commandPalette")

    def _setup_shortcuts(self) -> None:
        """Set up keyboard navigation."""
        self._search_input.returnPressed.connect(self._execute_selected)

    def register_action(
        self,
        action_id: str,
        name: str,
        category: str,
        callback: Callable[[], None],
        shortcut: str | None = None,
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        """
        Register an action with the command palette.

        Args:
            action_id: Unique identifier (e.g., "view.toggle_dark_mode")
            name: Display name (e.g., "Toggle Dark Mode")
            category: Category for grouping (e.g., "View")
            callback: Function to execute when action selected
            shortcut: Optional shortcut display string
            enabled: Optional callable returning whether action is enabled

        Raises:
            ValueError: If action_id is already registered
        """
        if action_id in self._actions:
            raise ValueError(f"Action '{action_id}' is already registered")

        self._actions[action_id] = CommandAction(
            id=action_id,
            name=name,
            category=category,
            callback=callback,
            shortcut=shortcut,
            enabled=enabled,
        )

    def unregister_action(self, action_id: str) -> bool:
        """
        Remove an action from the command palette.

        Args:
            action_id: Action identifier to remove

        Returns:
            True if action was removed, False if not found
        """
        if action_id in self._actions:
            del self._actions[action_id]
            return True
        return False

    def show(self) -> None:
        """Show the command palette dialog."""
        self._search_input.clear()
        self._populate_results()
        self._search_input.setFocus()
        super().show()

        # Center over parent
        if self.parent():
            parent_rect = self.parent().rect()
            x = parent_rect.center().x() - self.width() // 2
            y = parent_rect.top() + 100
            self.move(
                self.parent().mapToGlobal(self.pos()).x() + x - self.pos().x(),
                self.parent().mapToGlobal(self.pos()).y() + y - self.pos().y(),
            )

    def hide(self) -> None:
        """Hide the command palette dialog."""
        self._search_input.clear()
        super().hide()

    def set_placeholder(self, text: str) -> None:
        """
        Set the placeholder text for the search input.

        Args:
            text: Placeholder text
        """
        self._search_input.setPlaceholderText(text)

    def _filter_actions(self, query: str) -> None:
        """Filter results based on search query."""
        self._populate_results(query)

    def _populate_results(self, query: str = "") -> None:
        """Populate the results list with matching actions."""
        self._results_list.clear()

        query_lower = query.lower()

        for action in self._actions.values():
            # Check if enabled
            if action.enabled is not None and not action.enabled():
                continue

            # Fuzzy match
            if query and not self._fuzzy_match(query_lower, action.name.lower()):
                continue

            # Create item
            display_text = f"{action.category}: {action.name}"
            if action.shortcut:
                display_text += f"  ({action.shortcut})"

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, action.id)
            self._results_list.addItem(item)

        # Select first item
        if self._results_list.count() > 0:
            self._results_list.setCurrentRow(0)

    def _fuzzy_match(self, query: str, text: str) -> bool:
        """
        Check if query fuzzy-matches text.

        Args:
            query: Search query (lowercase)
            text: Text to match against (lowercase)

        Returns:
            True if matches
        """
        # Prefix match
        if text.startswith(query):
            return True

        # Substring match
        if query in text:
            return True

        # Initials match
        words = text.split()
        initials = "".join(w[0] for w in words if w)
        if initials.startswith(query):
            return True

        return False

    def _execute_selected(self) -> None:
        """Execute the currently selected action."""
        current = self._results_list.currentItem()
        if current:
            self._on_item_activated(current)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        """Handle action item activation."""
        action_id = item.data(Qt.ItemDataRole.UserRole)
        if action_id and action_id in self._actions:
            action = self._actions[action_id]
            self.hide()
            action.callback()
            self.action_triggered.emit(action_id)

    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        elif event.key() == Qt.Key.Key_Down:
            current = self._results_list.currentRow()
            if current < self._results_list.count() - 1:
                self._results_list.setCurrentRow(current + 1)
        elif event.key() == Qt.Key.Key_Up:
            current = self._results_list.currentRow()
            if current > 0:
                self._results_list.setCurrentRow(current - 1)
        else:
            super().keyPressEvent(event)
