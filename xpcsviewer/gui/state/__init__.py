"""
State management for XPCS-TOOLKIT GUI.

This module provides state persistence including:
- SessionManager for workspace state
- UserPreferences for user settings
- RecentPathsManager for recent directories
"""

from xpcsviewer.gui.state.preferences import (
    UserPreferences,
    load_preferences,
    save_preferences,
)
from xpcsviewer.gui.state.recent_paths import RecentPath, RecentPathsManager
from xpcsviewer.gui.state.session_manager import (
    AnalysisParameters,
    FileEntry,
    SessionManager,
    SessionState,
    WindowGeometry,
)

__all__ = [
    "AnalysisParameters",
    "FileEntry",
    "RecentPath",
    "RecentPathsManager",
    "SessionManager",
    "SessionState",
    "UserPreferences",
    "WindowGeometry",
    "load_preferences",
    "save_preferences",
]
