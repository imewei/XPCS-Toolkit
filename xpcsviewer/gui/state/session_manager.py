"""
Session state management for XPCS-TOOLKIT GUI.

This module handles saving and restoring workspace state between sessions.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FileEntry:
    """A file in the session's target list."""

    path: str  # Absolute file path
    order: int  # Position in list (0-indexed)
    exists: bool = True  # Computed on load, not stored


@dataclass
class WindowGeometry:
    """Main window geometry."""

    x: int = 100
    y: int = 100
    width: int = 1200
    height: int = 800
    maximized: bool = False


@dataclass
class AnalysisParameters:
    """Analysis parameters for each tab."""

    # Tab 0: SAXS 2D
    saxs2d_colormap: str = "viridis"
    saxs2d_auto_level: bool = True
    saxs2d_log_scale: bool = False

    # Tab 1: SAXS 1D
    saxs1d_log_x: bool = False
    saxs1d_log_y: bool = True

    # Tab 2: Stability
    stability_metric: str = "mean"

    # Tab 3: Intensity vs Time
    intensity_t_normalize: bool = False

    # Tab 4: G2 Correlation
    g2_fit_function: str = "single_exp"
    g2_q_index: int = 0
    g2_show_fit: bool = True

    # Tab 5: Diffusion
    diffusion_model: str = "free"

    # Tab 6: Two-time
    twotime_selected_q: int = 0
    twotime_colormap: str = "viridis"
    twotime_symmetric: bool = True

    # Tab 7: Q-map
    qmap_colormap: str = "viridis"
    qmap_show_rings: bool = True


@dataclass
class SessionState:
    """Complete workspace session state."""

    # Metadata
    version: str = "1.0"
    timestamp: str = ""

    # File state
    data_path: str | None = None
    target_files: list[FileEntry] = field(default_factory=list)

    # UI state
    active_tab: int = 0
    window_geometry: WindowGeometry = field(default_factory=WindowGeometry)

    # Analysis state
    analysis_params: AnalysisParameters = field(default_factory=AnalysisParameters)


def get_session_path() -> Path:
    """Get the path to the session file."""
    home_dir = Path(os.path.expanduser("~")) / ".xpcsviewer"
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir / "session.json"


class SessionManager:
    """
    Manages saving and restoring workspace state.

    Handles file validation, corrupted data, and missing files gracefully.
    """

    def __init__(self) -> None:
        """Initialize the SessionManager."""
        self._warnings: list[str] = []

    def save_session(self, session: SessionState) -> bool:
        """
        Save the current session state to disk.

        Args:
            session: Complete session state to persist

        Returns:
            True if save successful, False otherwise
        """
        session_path = get_session_path()

        try:
            # Update timestamp
            session.timestamp = datetime.now(UTC).isoformat()

            # Convert to dict for JSON serialization
            data = {
                "version": session.version,
                "timestamp": session.timestamp,
                "data_path": session.data_path,
                "target_files": [
                    {"path": f.path, "order": f.order} for f in session.target_files
                ],
                "active_tab": session.active_tab,
                "window_geometry": asdict(session.window_geometry),
                "analysis_params": asdict(session.analysis_params),
            }

            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.debug("Session saved successfully")
            return True

        except OSError as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self) -> SessionState | None:
        """
        Load session state from disk.

        Returns:
            SessionState if valid session exists, None otherwise
        """
        self._warnings = []
        session_path = get_session_path()

        if not session_path.exists():
            logger.debug("No session file found")
            return None

        try:
            with open(session_path, encoding="utf-8") as f:
                data = json.load(f)

            # Validate version
            version = data.get("version", "1.0")
            if version != "1.0":
                logger.info(f"Migrating session from version {version}")

            # Parse target files and validate existence
            target_files = []
            for entry in data.get("target_files", []):
                path = entry.get("path", "")
                order = entry.get("order", len(target_files))
                exists = Path(path).exists()

                if not exists:
                    self._warnings.append(f"File not found: {path}")
                else:
                    target_files.append(FileEntry(path=path, order=order, exists=True))

            # Parse window geometry
            geom_data = data.get("window_geometry", {})
            geometry = WindowGeometry(
                x=geom_data.get("x", 100),
                y=geom_data.get("y", 100),
                width=geom_data.get("width", 1200),
                height=geom_data.get("height", 800),
                maximized=geom_data.get("maximized", False),
            )

            # Parse analysis parameters
            params_data = data.get("analysis_params", {})
            params = AnalysisParameters(
                saxs2d_colormap=params_data.get("saxs2d_colormap", "viridis"),
                saxs2d_auto_level=params_data.get("saxs2d_auto_level", True),
                saxs2d_log_scale=params_data.get("saxs2d_log_scale", False),
                saxs1d_log_x=params_data.get("saxs1d_log_x", False),
                saxs1d_log_y=params_data.get("saxs1d_log_y", True),
                stability_metric=params_data.get("stability_metric", "mean"),
                intensity_t_normalize=params_data.get("intensity_t_normalize", False),
                g2_fit_function=params_data.get("g2_fit_function", "single_exp"),
                g2_q_index=params_data.get("g2_q_index", 0),
                g2_show_fit=params_data.get("g2_show_fit", True),
                diffusion_model=params_data.get("diffusion_model", "free"),
                twotime_selected_q=params_data.get("twotime_selected_q", 0),
                twotime_colormap=params_data.get("twotime_colormap", "viridis"),
                twotime_symmetric=params_data.get("twotime_symmetric", True),
                qmap_colormap=params_data.get("qmap_colormap", "viridis"),
                qmap_show_rings=params_data.get("qmap_show_rings", True),
            )

            # Validate active_tab
            active_tab = data.get("active_tab", 0)
            if not 0 <= active_tab <= 9:
                logger.warning(f"Invalid active_tab {active_tab}, using 0")
                active_tab = 0

            session = SessionState(
                version=version,
                timestamp=data.get("timestamp", ""),
                data_path=data.get("data_path"),
                target_files=target_files,
                active_tab=active_tab,
                window_geometry=geometry,
                analysis_params=params,
            )

            logger.debug("Session loaded successfully")
            return session

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse session.json: {e}")
            self._warnings.append(f"Session file corrupted: {e}")
            return None
        except OSError as e:
            logger.warning(f"Failed to read session.json: {e}")
            self._warnings.append(f"Cannot read session file: {e}")
            return None

    def get_warnings(self) -> list[str]:
        """
        Get warnings from last load operation.

        Returns:
            List of warning messages

        Note:
            Warnings are cleared after calling this method.
        """
        warnings = self._warnings[:]
        self._warnings = []
        return warnings

    def clear_session(self) -> None:
        """Clear saved session state."""
        session_path = get_session_path()
        if session_path.exists():
            try:
                session_path.unlink()
                logger.debug("Session cleared")
            except OSError as e:
                logger.warning(f"Failed to clear session: {e}")

    def has_saved_session(self) -> bool:
        """
        Check if a saved session exists.

        Returns:
            True if valid session file exists
        """
        session_path = get_session_path()
        return session_path.exists()
