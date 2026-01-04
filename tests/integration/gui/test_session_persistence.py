"""Integration tests for session persistence functionality."""

import json
import time
from pathlib import Path

import pytest

from xpcsviewer.gui.state.session_manager import (
    AnalysisParameters,
    FileEntry,
    SessionManager,
    SessionState,
    WindowGeometry,
)


class TestSessionPersistenceIntegration:
    """Integration tests for session save/restore cycle."""

    @pytest.fixture
    def session_manager(self, tmp_path, monkeypatch):
        """Create SessionManager with temporary storage."""
        session_file = tmp_path / ".xpcsviewer" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )
        return SessionManager()

    @pytest.fixture
    def sample_session(self, tmp_path):
        """Create a sample session state for testing."""
        # Create test files
        test_files = []
        for i in range(5):
            f = tmp_path / f"test_file_{i}.hdf"
            f.touch()
            test_files.append(
                FileEntry(
                    path=str(f),
                    order=i,
                )
            )

        return SessionState(
            version="1.0",
            window_geometry=WindowGeometry(x=100, y=100, width=1200, height=800),
            active_tab=2,
            target_files=test_files,
            analysis_params=AnalysisParameters(
                saxs1d_log_x=True,
                saxs1d_log_y=True,
            ),
        )

    def test_save_and_restore_session(self, session_manager, sample_session):
        """Session should save and restore all state correctly."""
        # Save session
        session_manager.save_session(sample_session)

        # Load session
        loaded = session_manager.load_session()

        assert loaded is not None
        assert loaded.version == sample_session.version
        assert loaded.active_tab == sample_session.active_tab
        assert len(loaded.target_files) == len(sample_session.target_files)

    def test_window_geometry_preserved(self, session_manager, sample_session):
        """Window geometry should be preserved across sessions."""
        session_manager.save_session(sample_session)
        loaded = session_manager.load_session()

        assert loaded.window_geometry.x == 100
        assert loaded.window_geometry.y == 100
        assert loaded.window_geometry.width == 1200
        assert loaded.window_geometry.height == 800

    def test_analysis_parameters_preserved(self, session_manager, sample_session):
        """Analysis parameters should be preserved across sessions."""
        session_manager.save_session(sample_session)
        loaded = session_manager.load_session()

        assert loaded.analysis_params.saxs1d_log_x is True
        assert loaded.analysis_params.saxs1d_log_y is True

    def test_file_entries_preserved(self, session_manager, sample_session):
        """File entries should be preserved with all metadata."""
        session_manager.save_session(sample_session)
        loaded = session_manager.load_session()

        assert len(loaded.target_files) == 5
        for i, entry in enumerate(loaded.target_files):
            assert entry.order == i

    def test_missing_file_warnings(self, tmp_path, monkeypatch):
        """Session with missing files should generate warnings."""
        session_file = tmp_path / ".xpcsviewer" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        session_manager = SessionManager()

        # Create session with non-existent files
        session = SessionState(
            target_files=[
                FileEntry(
                    path="/nonexistent/file1.hdf",
                    order=0,
                ),
                FileEntry(
                    path="/nonexistent/file2.hdf",
                    order=1,
                ),
            ]
        )

        session_manager.save_session(session)
        loaded = session_manager.load_session()
        warnings = session_manager.get_warnings()

        assert len(warnings) == 2
        assert "file1.hdf" in warnings[0]
        assert "file2.hdf" in warnings[1]

    def test_valid_files_not_warned(self, session_manager, sample_session):
        """Valid files should not generate warnings."""
        session_manager.save_session(sample_session)
        session_manager.load_session()
        warnings = session_manager.get_warnings()

        assert len(warnings) == 0

    def test_corrupted_session_handled(self, tmp_path, monkeypatch):
        """Corrupted session file should not crash application."""
        session_file = tmp_path / ".xpcsviewer" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text("{ invalid json }")

        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        # Should not raise
        manager = SessionManager()
        loaded = manager.load_session()

        assert loaded is None

    def test_empty_session_file_handled(self, tmp_path, monkeypatch):
        """Empty session file should be handled gracefully."""
        session_file = tmp_path / ".xpcsviewer" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.write_text("")

        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        manager = SessionManager()
        loaded = manager.load_session()

        assert loaded is None

    def test_session_restore_performance(self, tmp_path, monkeypatch):
        """Session restore should complete within 3 seconds for 20 files."""
        session_file = tmp_path / ".xpcsviewer" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        session_manager = SessionManager()

        # Create 20 test files
        test_files = []
        for i in range(20):
            f = tmp_path / f"perf_test_{i}.hdf"
            f.touch()
            test_files.append(
                FileEntry(
                    path=str(f),
                    order=i,
                )
            )

        session = SessionState(target_files=test_files)

        # Save and measure restore time
        session_manager.save_session(session)

        start_time = time.time()
        loaded = session_manager.load_session()
        _ = session_manager.get_warnings()
        elapsed = time.time() - start_time

        assert elapsed < 3.0, f"Session restore took {elapsed:.2f}s"
        assert len(loaded.target_files) == 20

    def test_clear_session_removes_file(self, tmp_path, monkeypatch):
        """Clearing session should remove session file."""
        session_file = tmp_path / ".xpcsviewer" / "session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.session_manager.get_session_path",
            lambda: session_file,
        )

        session_manager = SessionManager()

        # Create and save a simple session
        test_file = tmp_path / "test.hdf"
        test_file.touch()
        session = SessionState(target_files=[FileEntry(path=str(test_file), order=0)])

        session_manager.save_session(session)
        assert session_file.exists()

        session_manager.clear_session()
        assert not session_file.exists()


class TestSessionStateDataclasses:
    """Integration tests for session state dataclasses."""

    def test_session_state_serialization_roundtrip(self, tmp_path):
        """SessionState should serialize and deserialize correctly."""
        test_file = tmp_path / "test.hdf"
        test_file.touch()

        original = SessionState(
            version="1.0",
            window_geometry=WindowGeometry(x=50, y=75, width=1000, height=600),
            active_tab=1,
            target_files=[
                FileEntry(
                    path=str(test_file),
                    order=0,
                )
            ],
            analysis_params=AnalysisParameters(
                saxs1d_log_x=True,
                g2_q_index=5,
            ),
        )

        # Convert to dict for JSON
        data = {
            "version": original.version,
            "window_geometry": {
                "x": original.window_geometry.x,
                "y": original.window_geometry.y,
                "width": original.window_geometry.width,
                "height": original.window_geometry.height,
                "maximized": original.window_geometry.maximized,
            },
            "active_tab": original.active_tab,
            "target_files": [
                {
                    "path": f.path,
                    "order": f.order,
                }
                for f in original.target_files
            ],
            "analysis_params": {
                "saxs1d_log_x": original.analysis_params.saxs1d_log_x,
                "g2_q_index": original.analysis_params.g2_q_index,
            },
        }

        # Roundtrip through JSON
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)

        # Reconstruct
        reconstructed = SessionState(
            version=loaded_data["version"],
            window_geometry=WindowGeometry(**loaded_data["window_geometry"]),
            active_tab=loaded_data["active_tab"],
            target_files=[FileEntry(**f) for f in loaded_data["target_files"]],
            analysis_params=AnalysisParameters(**loaded_data["analysis_params"]),
        )

        assert reconstructed.version == original.version
        assert reconstructed.window_geometry.x == original.window_geometry.x
        assert reconstructed.active_tab == original.active_tab
        assert len(reconstructed.target_files) == len(original.target_files)
        assert (
            reconstructed.analysis_params.saxs1d_log_x
            == original.analysis_params.saxs1d_log_x
        )
