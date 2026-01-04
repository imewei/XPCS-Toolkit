"""Unit tests for RecentPathsManager."""

import json
from pathlib import Path

import pytest

from xpcsviewer.gui.state.recent_paths import (
    RecentPath,
    RecentPathsManager,
    RecentPathsState,
    get_recent_paths_file,
)


class TestRecentPath:
    """Tests for RecentPath dataclass."""

    def test_recent_path_creation(self):
        """RecentPath should store all fields."""
        path = RecentPath(
            path="/test/path",
            last_accessed="2024-01-01T00:00:00Z",
            access_count=5,
        )

        assert path.path == "/test/path"
        assert path.last_accessed == "2024-01-01T00:00:00Z"
        assert path.access_count == 5

    def test_recent_path_default_access_count(self):
        """RecentPath should default to access_count=1."""
        path = RecentPath(
            path="/test/path",
            last_accessed="2024-01-01T00:00:00Z",
        )

        assert path.access_count == 1


class TestRecentPathsState:
    """Tests for RecentPathsState dataclass."""

    def test_state_default_values(self):
        """RecentPathsState should have sensible defaults."""
        state = RecentPathsState()

        assert state.version == "1.0"
        assert state.max_entries == 10
        assert state.paths == []

    def test_state_custom_values(self):
        """RecentPathsState should accept custom values."""
        paths = [
            RecentPath("/test1", "2024-01-01T00:00:00Z"),
            RecentPath("/test2", "2024-01-02T00:00:00Z"),
        ]
        state = RecentPathsState(
            version="2.0",
            max_entries=20,
            paths=paths,
        )

        assert state.version == "2.0"
        assert state.max_entries == 20
        assert len(state.paths) == 2


class TestGetRecentPathsFile:
    """Tests for get_recent_paths_file function."""

    def test_returns_path_object(self):
        """get_recent_paths_file should return a Path object."""
        path = get_recent_paths_file()

        assert isinstance(path, Path)

    def test_path_in_xpcsviewer_directory(self):
        """Recent paths file should be in .xpcsviewer directory."""
        path = get_recent_paths_file()

        assert ".xpcsviewer" in str(path)
        assert path.name == "recent_paths.json"


class TestRecentPathsManagerInit:
    """Tests for RecentPathsManager initialization."""

    def test_manager_creation(self, tmp_path, monkeypatch):
        """RecentPathsManager should be created."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()

        assert manager is not None

    def test_manager_default_max_entries(self, tmp_path, monkeypatch):
        """RecentPathsManager should have default max_entries=10."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()

        assert manager._max_entries == 10

    def test_manager_custom_max_entries(self, tmp_path, monkeypatch):
        """RecentPathsManager should accept custom max_entries."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager(max_entries=5)

        assert manager._max_entries == 5


class TestRecentPathsManagerAddPath:
    """Tests for RecentPathsManager.add_path."""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        """Create manager with temporary storage."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )
        return RecentPathsManager()

    def test_add_path(self, manager, tmp_path):
        """add_path should add path to recent list."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        manager.add_path(str(test_dir))

        paths = manager.get_recent_paths()
        assert len(paths) == 1
        assert paths[0].path == str(test_dir)

    def test_add_path_updates_existing(self, manager, tmp_path):
        """add_path should update existing path."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        manager.add_path(str(test_dir))
        manager.add_path(str(test_dir))

        paths = manager.get_recent_paths()
        assert len(paths) == 1
        assert paths[0].access_count == 2

    def test_add_path_moves_to_front(self, manager, tmp_path):
        """add_path should move existing path to front."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        manager.add_path(str(dir1))
        manager.add_path(str(dir2))
        manager.add_path(str(dir1))  # Access dir1 again

        paths = manager.get_recent_paths()
        assert paths[0].path == str(dir1)  # dir1 should be first now

    def test_add_path_respects_max_entries(self, tmp_path, monkeypatch):
        """add_path should trim list to max_entries."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager(max_entries=3)

        for i in range(5):
            d = tmp_path / f"dir{i}"
            d.mkdir()
            manager.add_path(str(d))

        paths = manager.get_recent_paths()
        assert len(paths) == 3


class TestRecentPathsManagerGetPaths:
    """Tests for RecentPathsManager.get_recent_paths."""

    def test_get_recent_paths_empty(self, tmp_path, monkeypatch):
        """get_recent_paths should return empty list initially."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()
        paths = manager.get_recent_paths()

        assert paths == []

    def test_get_recent_paths_returns_copy(self, tmp_path, monkeypatch):
        """get_recent_paths should return a copy."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        manager.add_path(str(test_dir))

        paths1 = manager.get_recent_paths()
        paths2 = manager.get_recent_paths()

        assert paths1 is not paths2


class TestRecentPathsManagerRemoveInvalid:
    """Tests for RecentPathsManager.remove_invalid_path."""

    def test_remove_invalid_path(self, tmp_path, monkeypatch):
        """remove_invalid_path should remove the path."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        manager.add_path(str(test_dir))

        result = manager.remove_invalid_path(str(test_dir))

        assert result is True
        assert len(manager.get_recent_paths()) == 0

    def test_remove_nonexistent_path(self, tmp_path, monkeypatch):
        """remove_invalid_path should return False for unknown path."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()

        result = manager.remove_invalid_path("/nonexistent/path")

        assert result is False


class TestRecentPathsManagerClear:
    """Tests for RecentPathsManager.clear."""

    def test_clear(self, tmp_path, monkeypatch):
        """clear should remove all paths."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        manager = RecentPathsManager()
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        manager.add_path(str(test_dir))

        manager.clear()

        assert len(manager.get_recent_paths()) == 0


class TestRecentPathsManagerPersistence:
    """Tests for RecentPathsManager persistence."""

    def test_load_existing_file(self, tmp_path, monkeypatch):
        """Manager should load existing recent_paths.json."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        # Write existing data
        data = {
            "version": "1.0",
            "max_entries": 10,
            "paths": [
                {
                    "path": "/existing/path",
                    "last_accessed": "2024-01-01T00:00:00Z",
                    "access_count": 3,
                }
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager = RecentPathsManager()
        paths = manager.get_recent_paths()

        assert len(paths) == 1
        assert paths[0].path == "/existing/path"
        assert paths[0].access_count == 3

    def test_save_persists_data(self, tmp_path, monkeypatch):
        """Manager should persist data to file."""
        file_path = tmp_path / "recent_paths.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        # Create and add path
        manager = RecentPathsManager()
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        manager.add_path(str(test_dir))

        # Verify file was written
        assert file_path.exists()

        # Load in new manager and verify
        manager2 = RecentPathsManager()
        paths = manager2.get_recent_paths()
        assert len(paths) == 1

    def test_handles_corrupted_file(self, tmp_path, monkeypatch):
        """Manager should handle corrupted JSON gracefully."""
        file_path = tmp_path / "recent_paths.json"
        file_path.write_text("{ invalid json }")
        monkeypatch.setattr(
            "xpcsviewer.gui.state.recent_paths.get_recent_paths_file",
            lambda: file_path,
        )

        # Should not raise
        manager = RecentPathsManager()
        paths = manager.get_recent_paths()

        assert paths == []
