"""Unit tests for CommandPalette."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from xpcsviewer.gui.widgets.command_palette import CommandAction, CommandPalette


class TestCommandAction:
    """Tests for CommandAction dataclass."""

    def test_command_action_creation(self):
        """CommandAction should store all fields."""
        callback = lambda: None
        action = CommandAction(
            id="test.action",
            name="Test Action",
            category="Test",
            callback=callback,
        )

        assert action.id == "test.action"
        assert action.name == "Test Action"
        assert action.category == "Test"
        assert action.callback == callback
        assert action.shortcut is None
        assert action.enabled is None

    def test_command_action_with_shortcut(self):
        """CommandAction should store shortcut."""
        action = CommandAction(
            id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
            shortcut="Ctrl+T",
        )

        assert action.shortcut == "Ctrl+T"

    def test_command_action_with_enabled(self):
        """CommandAction should store enabled callable."""
        enabled_fn = lambda: True
        action = CommandAction(
            id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
            enabled=enabled_fn,
        )

        assert action.enabled == enabled_fn
        assert action.enabled() is True


class TestCommandPaletteInit:
    """Tests for CommandPalette initialization."""

    def test_palette_creation(self, qtbot):
        """CommandPalette should be created."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert palette is not None

    def test_palette_has_search_input(self, qtbot):
        """CommandPalette should have search input."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert hasattr(palette, "_search_input")

    def test_palette_has_results_list(self, qtbot):
        """CommandPalette should have results list."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert hasattr(palette, "_results_list")

    def test_palette_has_action_triggered_signal(self, qtbot):
        """CommandPalette should have action_triggered signal."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert hasattr(palette, "action_triggered")

    def test_palette_object_name(self, qtbot):
        """CommandPalette should have correct objectName."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert palette.objectName() == "commandPalette"


class TestCommandPaletteActions:
    """Tests for action registration."""

    def test_register_action(self, qtbot):
        """register_action should add action to palette."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
        )

        assert "test.action" in palette._actions

    def test_register_action_with_shortcut(self, qtbot):
        """register_action should store shortcut."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
            shortcut="Ctrl+T",
        )

        assert palette._actions["test.action"].shortcut == "Ctrl+T"

    def test_register_duplicate_action_raises(self, qtbot):
        """register_action should raise ValueError for duplicate ID."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
        )

        with pytest.raises(ValueError, match="already registered"):
            palette.register_action(
                action_id="test.action",
                name="Another Action",
                category="Test",
                callback=lambda: None,
            )

    def test_unregister_action(self, qtbot):
        """unregister_action should remove action."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
        )

        result = palette.unregister_action("test.action")

        assert result is True
        assert "test.action" not in palette._actions

    def test_unregister_nonexistent_action(self, qtbot):
        """unregister_action should return False for unknown ID."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        result = palette.unregister_action("nonexistent")

        assert result is False


class TestCommandPaletteSearch:
    """Tests for search/filter functionality."""

    def test_set_placeholder(self, qtbot):
        """set_placeholder should update search input."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.set_placeholder("Search commands...")

        assert palette._search_input.placeholderText() == "Search commands..."

    def test_fuzzy_match_prefix(self, qtbot):
        """_fuzzy_match should match prefix."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert palette._fuzzy_match("tog", "toggle theme") is True

    def test_fuzzy_match_substring(self, qtbot):
        """_fuzzy_match should match substring."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert palette._fuzzy_match("theme", "toggle theme") is True

    def test_fuzzy_match_initials(self, qtbot):
        """_fuzzy_match should match word initials."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        # "tt" matches "Toggle Theme" initials
        assert palette._fuzzy_match("tt", "toggle theme") is True

    def test_fuzzy_match_no_match(self, qtbot):
        """_fuzzy_match should return False for no match."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        assert palette._fuzzy_match("xyz", "toggle theme") is False


class TestCommandPaletteKeyboard:
    """Tests for keyboard navigation."""

    @pytest.fixture
    def palette_with_actions(self, qtbot):
        """Create palette with test actions."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action("action1", "First Action", "Test", lambda: None)
        palette.register_action("action2", "Second Action", "Test", lambda: None)
        palette.register_action("action3", "Third Action", "Test", lambda: None)

        return palette

    def test_escape_hides_palette(self, palette_with_actions, qtbot):
        """Escape key should hide palette."""
        palette = palette_with_actions
        palette.show()

        # Simulate Escape key
        from PySide6.QtGui import QKeyEvent

        event = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
        palette.keyPressEvent(event)

        assert palette.isHidden()


class TestCommandPaletteExecution:
    """Tests for action execution."""

    def test_action_callback_called(self, qtbot):
        """Executing action should call callback."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        called = []
        palette.register_action(
            action_id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: called.append(True),
        )

        # Populate and select
        palette._populate_results()
        palette._execute_selected()

        assert called == [True]

    def test_action_triggered_signal_emitted(self, qtbot):
        """Executing action should emit action_triggered signal."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="test.action",
            name="Test Action",
            category="Test",
            callback=lambda: None,
        )

        # Populate and select
        palette._populate_results()

        with qtbot.waitSignal(palette.action_triggered, timeout=1000) as blocker:
            palette._execute_selected()

        assert blocker.args == ["test.action"]

    def test_disabled_action_not_shown(self, qtbot):
        """Disabled actions should not appear in results."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="disabled.action",
            name="Disabled Action",
            category="Test",
            callback=lambda: None,
            enabled=lambda: False,
        )

        palette._populate_results()

        assert palette._results_list.count() == 0

    def test_enabled_action_shown(self, qtbot):
        """Enabled actions should appear in results."""
        palette = CommandPalette()
        qtbot.addWidget(palette)

        palette.register_action(
            action_id="enabled.action",
            name="Enabled Action",
            category="Test",
            callback=lambda: None,
            enabled=lambda: True,
        )

        palette._populate_results()

        assert palette._results_list.count() == 1
