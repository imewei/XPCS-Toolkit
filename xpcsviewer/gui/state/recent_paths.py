"""
Recent paths management for XPCS-TOOLKIT GUI.

This module handles tracking and persisting recently accessed directories.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RecentPath:
    """A recently accessed directory."""

    path: str  # Absolute directory path
    last_accessed: str  # ISO 8601 timestamp
    access_count: int = 1  # Number of times accessed


@dataclass
class RecentPathsState:
    """Recent directories state."""

    version: str = "1.0"
    max_entries: int = 10
    paths: list[RecentPath] = field(default_factory=list)


def get_recent_paths_file() -> Path:
    """Get the path to the recent paths file."""
    home_dir = Path(os.path.expanduser("~")) / ".xpcsviewer"
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir / "recent_paths.json"


class RecentPathsManager:
    """
    Manages recent directory paths.

    Tracks recently accessed directories with timestamps and access counts.
    """

    def __init__(self, max_entries: int = 10) -> None:
        """
        Initialize the RecentPathsManager.

        Args:
            max_entries: Maximum number of recent paths to keep
        """
        self._max_entries = max_entries
        self._state: RecentPathsState | None = None
        self._load()

    def _load(self) -> None:
        """Load recent paths from disk."""
        file_path = get_recent_paths_file()

        if not file_path.exists():
            self._state = RecentPathsState(max_entries=self._max_entries)
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            paths = []
            for entry in data.get("paths", []):
                paths.append(
                    RecentPath(
                        path=entry.get("path", ""),
                        last_accessed=entry.get("last_accessed", ""),
                        access_count=entry.get("access_count", 1),
                    )
                )

            self._state = RecentPathsState(
                version=data.get("version", "1.0"),
                max_entries=data.get("max_entries", self._max_entries),
                paths=paths,
            )
            logger.debug(f"Loaded {len(paths)} recent paths")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load recent paths: {e}")
            self._state = RecentPathsState(max_entries=self._max_entries)

    def _save(self) -> bool:
        """Save recent paths to disk."""
        if self._state is None:
            return False

        file_path = get_recent_paths_file()

        try:
            data = {
                "version": self._state.version,
                "max_entries": self._state.max_entries,
                "paths": [asdict(p) for p in self._state.paths],
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug("Recent paths saved")
            return True

        except OSError as e:
            logger.error(f"Failed to save recent paths: {e}")
            return False

    def add_path(self, path: str) -> None:
        """
        Add or update a path in the recent list.

        Args:
            path: Directory path to add
        """
        if self._state is None:
            self._state = RecentPathsState(max_entries=self._max_entries)

        # Normalize path
        path = str(Path(path).resolve())
        now = datetime.now(UTC).isoformat()

        # Check if path already exists
        for existing in self._state.paths:
            if existing.path == path:
                existing.last_accessed = now
                existing.access_count += 1
                # Move to front
                self._state.paths.remove(existing)
                self._state.paths.insert(0, existing)
                self._save()
                return

        # Add new path
        new_entry = RecentPath(path=path, last_accessed=now, access_count=1)
        self._state.paths.insert(0, new_entry)

        # Trim to max entries
        if len(self._state.paths) > self._state.max_entries:
            self._state.paths = self._state.paths[: self._state.max_entries]

        self._save()

    def get_recent_paths(self) -> list[RecentPath]:
        """
        Get recent paths ordered by last access time.

        Returns:
            List of RecentPath objects, newest first
        """
        if self._state is None:
            return []
        return self._state.paths[:]

    def remove_invalid_path(self, path: str) -> bool:
        """
        Remove a path that no longer exists.

        Args:
            path: Path to remove

        Returns:
            True if path was removed, False if not found
        """
        if self._state is None:
            return False

        path = str(Path(path).resolve())

        for entry in self._state.paths:
            if entry.path == path:
                self._state.paths.remove(entry)
                self._save()
                logger.debug(f"Removed invalid path: {path}")
                return True

        return False

    def clear(self) -> None:
        """Clear all recent paths."""
        if self._state is not None:
            self._state.paths = []
            self._save()
