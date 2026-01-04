"""
Keyboard shortcut management for XPCS-TOOLKIT GUI.

This module provides centralized shortcut registration and management.
"""

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget


class ShortcutManager(QObject):
    """
    Centralized keyboard shortcut manager.

    Handles registration, conflict detection, and management of
    keyboard shortcuts across the application.
    """

    # Signal emitted when a shortcut is triggered
    # Signature: shortcut_triggered(shortcut_id: str)
    shortcut_triggered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the ShortcutManager.

        Args:
            parent: Parent widget for shortcuts
        """
        super().__init__(parent)
        self._shortcuts: dict[str, QShortcut] = {}
        self._parent_widget = parent

    def register_shortcut(
        self,
        shortcut_id: str,
        key_sequence: str | QKeySequence,
        callback: callable,
        description: str = "",
    ) -> bool:
        """
        Register a keyboard shortcut.

        Args:
            shortcut_id: Unique identifier for the shortcut
            key_sequence: Key combination (e.g., "Ctrl+P", "Ctrl+Shift+Tab")
            callback: Function to call when shortcut is triggered
            description: Human-readable description

        Returns:
            True if registration successful, False if ID already exists
        """
        if shortcut_id in self._shortcuts:
            return False

        if self._parent_widget is None:
            return False

        if isinstance(key_sequence, str):
            key_sequence = QKeySequence(key_sequence)

        shortcut = QShortcut(key_sequence, self._parent_widget)
        shortcut.activated.connect(callback)
        shortcut.activated.connect(lambda: self.shortcut_triggered.emit(shortcut_id))

        self._shortcuts[shortcut_id] = shortcut
        return True

    def unregister_shortcut(self, shortcut_id: str) -> bool:
        """
        Unregister a keyboard shortcut.

        Args:
            shortcut_id: ID of shortcut to remove

        Returns:
            True if removed, False if not found
        """
        if shortcut_id not in self._shortcuts:
            return False

        shortcut = self._shortcuts.pop(shortcut_id)
        shortcut.setEnabled(False)
        shortcut.deleteLater()
        return True

    def get_registered_shortcuts(self) -> list[str]:
        """
        Get list of registered shortcut IDs.

        Returns:
            List of shortcut identifiers
        """
        return list(self._shortcuts.keys())

    def is_registered(self, shortcut_id: str) -> bool:
        """
        Check if a shortcut ID is registered.

        Args:
            shortcut_id: ID to check

        Returns:
            True if registered
        """
        return shortcut_id in self._shortcuts
