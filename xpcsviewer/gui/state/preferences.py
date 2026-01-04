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


def validate_preferences(prefs: UserPreferences) -> list[str]:
    """
    Validate user preferences.

    Args:
        prefs: UserPreferences to validate

    Returns:
        list of validation error messages (empty if valid)
    """
    errors = []

    # Validate theme
    if prefs.theme not in ("light", "dark", "system"):
        errors.append(
            f"Invalid theme '{prefs.theme}', must be 'light', 'dark', or 'system'"
        )

    # Validate toast duration
    if not 1000 <= prefs.toast_duration_ms <= 10000:
        errors.append(
            f"toast_duration_ms must be between 1000 and 10000, got {prefs.toast_duration_ms}"
        )

    return errors


def load_preferences() -> UserPreferences:
    """
    Load user preferences from disk.

    Returns:
        UserPreferences instance (defaults if file missing or invalid)
    """
    prefs_path = get_preferences_path()

    if not prefs_path.exists():
        logger.debug("Preferences file not found, using defaults")
        return UserPreferences()

    try:
        with open(prefs_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle migration from older versions if needed
        version = data.get("version", "1.0")
        if version != "1.0":
            logger.info(f"Migrating preferences from version {version}")
            # Future migration logic would go here

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

        # Validate and fix if needed
        errors = validate_preferences(prefs)
        if errors:
            logger.warning(f"Invalid preferences, using defaults: {errors}")
            return UserPreferences()

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
    # Validate before saving
    errors = validate_preferences(prefs)
    if errors:
        logger.error(f"Cannot save invalid preferences: {errors}")
        return False

    prefs_path = get_preferences_path()

    try:
        data = asdict(prefs)
        with open(prefs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved user preferences successfully")
        return True

    except OSError as e:
        logger.error(f"Failed to save preferences.json: {e}")
        return False
