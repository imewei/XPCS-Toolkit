"""
User preferences management for XPCS-TOOLKIT GUI.

This module handles loading, saving, and validating user preferences.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from xpcsviewer.utils.atomic_io import safe_json_write

logger = logging.getLogger(__name__)

# Type alias for theme mode
ThemeMode = Literal["light", "dark", "system"]


@dataclass
class UserPreferences:
    """User preferences persisted to disk."""

    # Theme
    theme: ThemeMode = "system"

    # Notifications
    show_completion_toasts: bool = True
    show_error_toasts: bool = True
    toast_duration_ms: int = 3000

    # Window behavior
    restore_session_on_startup: bool = True
    remember_window_geometry: bool = True

    # Version for migrations
    version: str = "1.0"


def get_preferences_path() -> Path:
    """Get the path to the preferences file."""
    home_dir = Path(os.path.expanduser("~")) / ".xpcsviewer"
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir / "preferences.json"


def clamp_preferences(prefs: UserPreferences) -> list[str]:
    """
    Validate and clamp user preferences in-place.

    Invalid fields are corrected to safe defaults rather than
    discarding the entire preferences object.

    Args:
        prefs: UserPreferences to validate and fix

    Returns:
        list of warning messages for fields that were clamped
    """
    warnings: list[str] = []

    # Validate theme
    if prefs.theme not in ("light", "dark", "system"):
        warnings.append(f"Invalid theme '{prefs.theme}', reset to 'system'")
        prefs.theme = "system"

    # Validate and clamp toast duration
    try:
        prefs.toast_duration_ms = int(prefs.toast_duration_ms)
    except (TypeError, ValueError):
        warnings.append("Non-numeric toast_duration_ms, reset to 3000")
        prefs.toast_duration_ms = 3000
    if prefs.toast_duration_ms < 1000:
        warnings.append(
            f"toast_duration_ms {prefs.toast_duration_ms} too low, clamped to 1000"
        )
        prefs.toast_duration_ms = 1000
    elif prefs.toast_duration_ms > 10000:
        warnings.append(
            f"toast_duration_ms {prefs.toast_duration_ms} too high, clamped to 10000"
        )
        prefs.toast_duration_ms = 10000

    return warnings


def load_preferences() -> UserPreferences:
    """
    Load user preferences from disk.

    Returns:
        UserPreferences instance (defaults if file missing or invalid)
    """
    try:
        prefs_path = get_preferences_path()

        if not prefs_path.exists():
            logger.debug("Preferences file not found, using defaults")
            return UserPreferences()

        with open(prefs_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning("Preferences file is not a JSON object, using defaults")
            return UserPreferences()

        # Handle migration from older versions if needed
        version = data.get("version", "1.0")
        if version != "1.0":
            logger.info(f"Migrating preferences from version {version}")

        # Create preferences from loaded data
        prefs = UserPreferences(
            theme=data.get("theme", "system"),
            show_completion_toasts=data.get("show_completion_toasts", True),
            show_error_toasts=data.get("show_error_toasts", True),
            toast_duration_ms=data.get("toast_duration_ms", 3000),
            restore_session_on_startup=data.get("restore_session_on_startup", True),
            remember_window_geometry=data.get("remember_window_geometry", True),
            version=data.get("version", "1.0"),
        )

        # Clamp invalid fields rather than discarding everything
        warnings = clamp_preferences(prefs)
        if warnings:
            logger.warning(f"Preferences clamped: {warnings}")

        logger.debug("Loaded user preferences successfully")
        return prefs

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse preferences.json: {e}")
        return UserPreferences()
    except OSError as e:
        logger.warning(f"Failed to read preferences.json: {e}")
        return UserPreferences()


def save_preferences(prefs: UserPreferences) -> bool:
    """
    Save user preferences to disk.

    Args:
        prefs: UserPreferences to save

    Returns:
        True if save successful, False otherwise
    """
    # Clamp before saving so the on-disk file is always valid
    warnings = clamp_preferences(prefs)
    if warnings:
        logger.warning(f"Preferences clamped before save: {warnings}")

    try:
        prefs_path = get_preferences_path()
        safe_json_write(prefs_path, asdict(prefs))
        logger.debug("Saved user preferences successfully")
        return True

    except OSError as e:
        logger.error(f"Failed to save preferences.json: {e}")
        return False
