"""
State management for XPCS-TOOLKIT GUI.

This module provides state persistence including:
- SessionManager for workspace state
- UserPreferences for user settings
- RecentPathsManager for recent directories
"""

from xpcs_toolkit.gui.state.preferences import (
    UserPreferences,
    load_preferences,
    save_preferences,
)
from xpcs_toolkit.gui.state.recent_paths import RecentPath, RecentPathsManager
from xpcs_toolkit.gui.state.session_manager import (
    AnalysisParameters,
    FileEntry,
    SessionManager,
    SessionState,
    WindowGeometry,
)

__all__ = [
    "SessionManager",
    "SessionState",
    "FileEntry",
    "WindowGeometry",
    "AnalysisParameters",
    "UserPreferences",
    "load_preferences",
    "save_preferences",
    "RecentPathsManager",
    "RecentPath",
]
