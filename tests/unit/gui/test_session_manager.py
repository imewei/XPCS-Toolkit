"""Unit tests for SessionManager."""

import json
from pathlib import Path

import pytest

from xpcsviewer.gui.state.session_manager import (
    AnalysisParameters,
    FileEntry,
    SessionManager,
    SessionState,
    WindowGeometry,
    get_session_path,
)


class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_file_entry_creation(self):
        """FileEntry should store path and order."""
        entry = FileEntry(path="/path/to/file.hdf5", order=0)

        assert entry.path == "/path/to/file.hdf5"
        assert entry.order == 0
        assert entry.exists is True

    def test_file_entry_exists_default(self):
        """FileEntry exists should default to True."""
        entry = FileEntry(path="/path/to/file.hdf5", order=0)

        assert entry.exists is True

    def test_file_entry_exists_override(self):
        """FileEntry exists can be set explicitly."""
        entry = FileEntry(path="/path/to/file.hdf5", order=0, exists=False)

        assert entry.exists is False


class TestWindowGeometry:
    """Tests for WindowGeometry dataclass."""

    def test_default_values(self):
        """WindowGeometry should have sensible defaults."""
        geom = WindowGeometry()

        assert geom.x == 100
        assert geom.y == 100
        assert geom.width == 1200
        assert geom.height == 800
        assert geom.maximized is False

    def test_custom_values(self):
        """WindowGeometry should accept custom values."""
        geom = WindowGeometry(x=50, y=50, width=800, height=600, maximized=True)

        assert geom.x == 50
        assert geom.y == 50
        assert geom.width == 800
        assert geom.height == 600
        assert geom.maximized is True


class TestAnalysisParameters:
    """Tests for AnalysisParameters dataclass."""

    def test_default_values(self):
        """AnalysisParameters should have sensible defaults."""
        params = AnalysisParameters()

        # SAXS 2D defaults
        assert params.saxs2d_colormap == "viridis"
        assert params.saxs2d_auto_level is True
        assert params.saxs2d_log_scale is False

        # SAXS 1D defaults
        assert params.saxs1d_log_x is False
        assert params.saxs1d_log_y is True

        # G2 defaults
        assert params.g2_fit_function == "single_exp"
        assert params.g2_q_index == 0
        assert params.g2_show_fit is True

    def test_custom_values(self):
        """AnalysisParameters should accept custom values."""
        params = AnalysisParameters(
            saxs2d_colormap="plasma",
            g2_fit_function="stretched_exp",
            g2_q_index=5,
        )

        assert params.saxs2d_colormap == "plasma"
        assert params.g2_fit_function == "stretched_exp"
        assert params.g2_q_index == 5


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_default_values(self):
        """SessionState should have sensible defaults."""
        session = SessionState()

        assert session.version == "1.0"
        assert session.timestamp == ""
        assert session.data_path is None
        assert session.target_files == []
        assert session.active_tab == 0
        assert isinstance(session.window_geometry, WindowGeometry)
        assert isinstance(session.analysis_params, AnalysisParameters)

    def test_custom_values(self):
        """SessionState should accept custom values."""
        files = [
            FileEntry(path="/path/file1.hdf5", order=0),
            FileEntry(path="/path/file2.hdf5", order=1),
        ]
        session = SessionState(
            data_path="/path/to/data",
            target_files=files,
            active_tab=3,
        )

        assert session.data_path == "/path/to/data"
        assert len(session.target_files) == 2
        assert session.active_tab == 3


class TestGetSessionPath:
    """Tests for get_session_path function."""

    def test_returns_path_object(self):
        """get_session_path should return a Path object."""
        path = get_session_path()

        assert isinstance(path, Path)

    def test_path_in_xpcsviewer_directory(self):
        """Session path should be in .xpcsviewer directory."""
        path = get_session_path()

        assert ".xpcsviewer" in str(path)
        assert path.name == "session.json"


class TestSessionManagerSave:
    """Tests for SessionManager.save_session."""

    def test_save_session_creates_file(self, tmp_path, monkeypatch):
        """save_session should create session.json file."""
        # Redirect session path to temp directory
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        session = SessionState(data_path="/test/path", active_tab=2)

        result = manager.save_session(session)

        assert result is True
        assert session_file.exists()

    def test_save_session_content(self, tmp_path, monkeypatch):
        """save_session should write correct JSON content."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        session = SessionState(
            data_path="/test/path",
            target_files=[FileEntry(path="/test/file.hdf5", order=0)],
            active_tab=2,
        )

        manager.save_session(session)

        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["data_path"] == "/test/path"
        assert data["active_tab"] == 2
        assert len(data["target_files"]) == 1
        assert data["target_files"][0]["path"] == "/test/file.hdf5"

    def test_save_session_updates_timestamp(self, tmp_path, monkeypatch):
        """save_session should update timestamp."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        session = SessionState()

        # Timestamp should be empty initially
        assert session.timestamp == ""

        manager.save_session(session)

        # Timestamp should be updated
        assert session.timestamp != ""
        assert "T" in session.timestamp  # ISO format


class TestSessionManagerLoad:
    """Tests for SessionManager.load_session."""

    def test_load_session_returns_none_if_no_file(self, tmp_path, monkeypatch):
        """load_session should return None if no session file exists."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        result = manager.load_session()

        assert result is None

    def test_load_session_roundtrip(self, tmp_path, monkeypatch):
        """load_session should restore saved session."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        original = SessionState(
            data_path="/test/path",
            active_tab=3,
            window_geometry=WindowGeometry(x=200, y=200, width=1000, height=700),
            analysis_params=AnalysisParameters(saxs2d_colormap="plasma"),
        )

        manager.save_session(original)
        loaded = manager.load_session()

        assert loaded is not None
        assert loaded.data_path == original.data_path
        assert loaded.active_tab == original.active_tab
        assert loaded.window_geometry.x == 200
        assert loaded.window_geometry.width == 1000
        assert loaded.analysis_params.saxs2d_colormap == "plasma"

    def test_load_session_validates_files(self, tmp_path, monkeypatch):
        """load_session should validate file existence."""
        session_file = tmp_path / "session.json"
        existing_file = tmp_path / "existing.hdf5"
        existing_file.touch()

        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        # Write session with mixed files
        data = {
            "version": "1.0",
            "target_files": [
                {"path": str(existing_file), "order": 0},
                {"path": "/nonexistent/file.hdf5", "order": 1},
            ],
            "active_tab": 0,
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager = SessionManager()
        loaded = manager.load_session()

        # Only existing file should be in target_files
        assert loaded is not None
        assert len(loaded.target_files) == 1
        assert loaded.target_files[0].path == str(existing_file)

        # Warning for missing file
        warnings = manager.get_warnings()
        assert len(warnings) == 1
        assert "not found" in warnings[0].lower()

    def test_load_session_handles_corrupted_json(self, tmp_path, monkeypatch):
        """load_session should handle corrupted JSON gracefully."""
        session_file = tmp_path / "session.json"
        session_file.write_text("{ invalid json }")

        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        result = manager.load_session()

        assert result is None
        warnings = manager.get_warnings()
        assert len(warnings) == 1
        assert "corrupted" in warnings[0].lower()

    def test_load_session_validates_active_tab(self, tmp_path, monkeypatch):
        """load_session should validate active_tab range."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        # Write session with invalid active_tab
        data = {"version": "1.0", "active_tab": 99}
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager = SessionManager()
        loaded = manager.load_session()

        assert loaded is not None
        assert loaded.active_tab == 0  # Reset to default


class TestSessionManagerWarnings:
    """Tests for SessionManager warning handling."""

    def test_get_warnings_returns_list(self):
        """get_warnings should return a list."""
        manager = SessionManager()
        warnings = manager.get_warnings()

        assert isinstance(warnings, list)

    def test_get_warnings_clears_after_call(self, tmp_path, monkeypatch):
        """get_warnings should clear warnings after being called."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        # Write session with missing file
        data = {
            "version": "1.0",
            "target_files": [{"path": "/nonexistent.hdf5", "order": 0}],
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        manager = SessionManager()
        manager.load_session()

        # First call should have warnings
        first_warnings = manager.get_warnings()
        assert len(first_warnings) > 0

        # Second call should be empty
        second_warnings = manager.get_warnings()
        assert len(second_warnings) == 0


class TestSessionManagerClear:
    """Tests for SessionManager.clear_session."""

    def test_clear_session_removes_file(self, tmp_path, monkeypatch):
        """clear_session should remove the session file."""
        session_file = tmp_path / "session.json"
        session_file.write_text("{}")
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        assert session_file.exists()

        manager.clear_session()

        assert not session_file.exists()

    def test_clear_session_no_error_if_no_file(self, tmp_path, monkeypatch):
        """clear_session should not error if no file exists."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()

        # Should not raise
        manager.clear_session()


class TestSessionManagerHasSavedSession:
    """Tests for SessionManager.has_saved_session."""

    def test_returns_false_if_no_file(self, tmp_path, monkeypatch):
        """has_saved_session should return False if no file exists."""
        session_file = tmp_path / "session.json"
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        assert manager.has_saved_session() is False

    def test_returns_true_if_file_exists(self, tmp_path, monkeypatch):
        """has_saved_session should return True if file exists."""
        session_file = tmp_path / "session.json"
        session_file.write_text("{}")
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        assert manager.has_saved_session() is True
