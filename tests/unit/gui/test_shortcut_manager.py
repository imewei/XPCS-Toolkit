"""Unit tests for ShortcutManager."""

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMainWindow

from xpcsviewer.gui.shortcuts.shortcut_manager import ShortcutManager


class TestShortcutManagerInit:
    """Tests for ShortcutManager initialization."""

    def test_manager_creation(self, qtbot):
        """ShortcutManager should be created."""
        window = QMainWindow()
        qtbot.addWidget(window)

        manager = ShortcutManager(parent=window)

        assert manager is not None

    def test_manager_has_signal(self, qtbot):
        """ShortcutManager should have shortcut_triggered signal."""
        window = QMainWindow()
        qtbot.addWidget(window)

        manager = ShortcutManager(parent=window)

        assert hasattr(manager, "shortcut_triggered")

    def test_manager_starts_empty(self, qtbot):
        """ShortcutManager should start with no shortcuts."""
        window = QMainWindow()
        qtbot.addWidget(window)

        manager = ShortcutManager(parent=window)

        assert len(manager._shortcuts) == 0


class TestShortcutRegistration:
    """Tests for shortcut registration."""

    @pytest.fixture
    def window_and_manager(self, qtbot):
        """Create window and manager for testing."""
        window = QMainWindow()
        qtbot.addWidget(window)
        manager = ShortcutManager(parent=window)
        return window, manager

    def test_register_shortcut_string(self, window_and_manager):
        """register_shortcut should accept string key sequence."""
        window, manager = window_and_manager

        result = manager.register_shortcut(
            shortcut_id="test.shortcut",
            key_sequence="Ctrl+T",
            callback=lambda: None,
        )

        assert result is True
        assert "test.shortcut" in manager._shortcuts

    def test_register_shortcut_qkeysequence(self, window_and_manager):
        """register_shortcut should accept QKeySequence."""
        window, manager = window_and_manager

        result = manager.register_shortcut(
            shortcut_id="test.shortcut",
            key_sequence=QKeySequence("Ctrl+T"),
            callback=lambda: None,
        )

        assert result is True
        assert "test.shortcut" in manager._shortcuts

    def test_register_duplicate_shortcut(self, window_and_manager):
        """register_shortcut should return False for duplicate ID."""
        window, manager = window_and_manager

        manager.register_shortcut("test.shortcut", "Ctrl+T", lambda: None)
        result = manager.register_shortcut("test.shortcut", "Ctrl+R", lambda: None)

        assert result is False

    def test_register_without_parent(self, qtbot):
        """register_shortcut should return False without parent."""
        manager = ShortcutManager(parent=None)

        result = manager.register_shortcut("test.shortcut", "Ctrl+T", lambda: None)

        assert result is False

    def test_unregister_shortcut(self, window_and_manager):
        """unregister_shortcut should remove shortcut."""
        window, manager = window_and_manager

        manager.register_shortcut("test.shortcut", "Ctrl+T", lambda: None)
        result = manager.unregister_shortcut("test.shortcut")

        assert result is True
        assert "test.shortcut" not in manager._shortcuts

    def test_unregister_nonexistent(self, window_and_manager):
        """unregister_shortcut should return False for unknown ID."""
        window, manager = window_and_manager

        result = manager.unregister_shortcut("nonexistent")

        assert result is False


class TestShortcutQuery:
    """Tests for shortcut query methods."""

    @pytest.fixture
    def manager_with_shortcuts(self, qtbot):
        """Create manager with shortcuts for testing."""
        window = QMainWindow()
        qtbot.addWidget(window)
        manager = ShortcutManager(parent=window)

        manager.register_shortcut("shortcut1", "Ctrl+1", lambda: None)
        manager.register_shortcut("shortcut2", "Ctrl+2", lambda: None)
        manager.register_shortcut("shortcut3", "Ctrl+3", lambda: None)

        return manager

    def test_get_registered_shortcuts(self, manager_with_shortcuts):
        """get_registered_shortcuts should return list of IDs."""
        manager = manager_with_shortcuts

        shortcuts = manager.get_registered_shortcuts()

        assert len(shortcuts) == 3
        assert "shortcut1" in shortcuts
        assert "shortcut2" in shortcuts
        assert "shortcut3" in shortcuts

    def test_is_registered_true(self, manager_with_shortcuts):
        """is_registered should return True for registered shortcut."""
        manager = manager_with_shortcuts

        assert manager.is_registered("shortcut1") is True

    def test_is_registered_false(self, manager_with_shortcuts):
        """is_registered should return False for unregistered shortcut."""
        manager = manager_with_shortcuts

        assert manager.is_registered("nonexistent") is False


class TestShortcutExecution:
    """Tests for shortcut execution."""

    def test_callback_connected(self, qtbot):
        """Shortcut callback should be connected."""
        window = QMainWindow()
        qtbot.addWidget(window)
        manager = ShortcutManager(parent=window)

        called = []
        manager.register_shortcut(
            shortcut_id="test.shortcut",
            key_sequence="Ctrl+T",
            callback=lambda: called.append(True),
        )

        # The shortcut is registered - we can verify the connection exists
        shortcut = manager._shortcuts["test.shortcut"]
        assert shortcut is not None
        assert shortcut.isEnabled()

    def test_signal_emitted_on_activation(self, qtbot):
        """shortcut_triggered signal should be emitted."""
        window = QMainWindow()
        qtbot.addWidget(window)
        manager = ShortcutManager(parent=window)

        manager.register_shortcut(
            shortcut_id="test.shortcut",
            key_sequence="Ctrl+T",
            callback=lambda: None,
        )

        # Manually trigger the shortcut's activated signal
        shortcut = manager._shortcuts["test.shortcut"]

        with qtbot.waitSignal(manager.shortcut_triggered, timeout=1000) as blocker:
            shortcut.activated.emit()

        assert blocker.args == ["test.shortcut"]
