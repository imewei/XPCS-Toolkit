"""Tests for atomic I/O utilities."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from xpcsviewer.utils.atomic_io import safe_json_write


class TestSafeJsonWrite:
    """Tests for safe_json_write."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Write a dict and read it back."""
        target = tmp_path / "out.json"
        data = {"key": "value", "num": 42}
        safe_json_write(target, data)

        assert target.exists()
        assert json.loads(target.read_text()) == data

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        """Overwriting an existing file replaces content."""
        target = tmp_path / "out.json"
        safe_json_write(target, {"old": True})
        safe_json_write(target, {"new": True})

        assert json.loads(target.read_text()) == {"new": True}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        target = tmp_path / "a" / "b" / "c" / "out.json"
        safe_json_write(target, {"nested": True})

        assert target.exists()
        assert json.loads(target.read_text()) == {"nested": True}

    def test_no_temp_file_on_success(self, tmp_path: Path) -> None:
        """No leftover .tmp_ files after successful write."""
        target = tmp_path / "out.json"
        safe_json_write(target, {"ok": True})

        leftover = list(tmp_path.glob(".tmp_*"))
        assert leftover == []

    def test_cleanup_on_serialisation_error(self, tmp_path: Path) -> None:
        """Temp file is removed if json.dump raises TypeError."""
        target = tmp_path / "out.json"
        bad_data = {"fn": lambda: None}  # not JSON-serialisable

        with pytest.raises(TypeError):
            safe_json_write(target, bad_data)

        assert not target.exists()
        leftover = list(tmp_path.glob(".tmp_*"))
        assert leftover == []

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """String paths are accepted as well as Path objects."""
        target = str(tmp_path / "out.json")
        safe_json_write(target, {"str_path": True})

        assert json.loads(Path(target).read_text()) == {"str_path": True}

    def test_cleanup_on_replace_failure(self, tmp_path: Path) -> None:
        """Temp file is cleaned up when os.replace fails."""
        target = tmp_path / "out.json"

        with patch(
            "xpcsviewer.utils.atomic_io.os.replace", side_effect=OSError("mock")
        ):
            with pytest.raises(OSError, match="mock"):
                safe_json_write(target, {"fail": True})

        assert not target.exists()
        leftover = list(tmp_path.glob(".tmp_*"))
        assert leftover == []
