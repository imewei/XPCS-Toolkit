"""
Icon loader for XPCS-Toolkit GUI.

Loads SVG icons from the resources/icons directory and returns QIcon objects.
Replaces ``currentColor`` in SVGs with the active theme's text color so icons
are visible in both light and dark modes.  Falls back to QStyle standard icons
if SVG files are not found.
"""

from pathlib import Path

from xpcsviewer.gui.qt_compat import QIcon, QPixmap, QStyle

_ICONS_DIR = Path(__file__).parent.parent / "ui" / "resources" / "icons"

# Map icon name -> SVG filename
_ICON_FILES = {
    "folder-open": "folder-open.svg",
    "refresh": "refresh.svg",
    "play-circle": "play-circle.svg",
    "sun": "sun.svg",
    "moon": "moon.svg",
    "activity": "activity.svg",
    "grid-mask": "grid-mask.svg",
    "tab-scattering": "tab-scattering.svg",
    "tab-diagnostics": "tab-diagnostics.svg",
    "tab-correlation": "tab-correlation.svg",
    "tab-analysis": "tab-analysis.svg",
    "tab-setup": "tab-setup.svg",
}

# Fallback standard icons for each name (used if SVG not found)
_FALLBACK_ICONS = {
    "folder-open": QStyle.StandardPixmap.SP_DirOpenIcon,
    "refresh": QStyle.StandardPixmap.SP_BrowserReload,
    "play-circle": QStyle.StandardPixmap.SP_MediaPlay,
    "sun": QStyle.StandardPixmap.SP_ComputerIcon,
    "moon": QStyle.StandardPixmap.SP_ComputerIcon,
    "activity": QStyle.StandardPixmap.SP_ComputerIcon,
    "grid-mask": QStyle.StandardPixmap.SP_FileDialogContentsView,
}

_icon_cache: dict[str, QIcon] = {}

# Current color used to recolor SVGs (updated on theme change)
_current_color: str = "#1A1A1A"  # default: dark text for light theme


def set_icon_color(color: str) -> None:
    """Set the color used to replace ``currentColor`` in SVGs.

    Call this when the theme changes, then call ``clear_icon_cache()``
    so subsequent ``get_icon()`` calls pick up the new color.

    Args:
        color: Hex color string (e.g. '#E8E8E4' for dark theme text).
    """
    global _current_color  # noqa: PLW0603
    _current_color = color


def _load_svg_icon(svg_path: Path) -> QIcon:
    """Load an SVG, replacing ``currentColor`` with the active text color."""
    svg_bytes = svg_path.read_bytes()
    svg_bytes = svg_bytes.replace(b"currentColor", _current_color.encode("ascii"))

    pixmap = QPixmap()
    pixmap.loadFromData(svg_bytes)
    return QIcon(pixmap)


def get_icon(name: str, style: QStyle | None = None) -> QIcon:
    """Return a QIcon for the given icon name.

    Loads from the SVG icons directory with ``currentColor`` replaced by the
    active theme's text color.  Falls back to QStyle standard icons if the
    SVG file is not found.

    Args:
        name: Icon name (e.g. 'folder-open', 'refresh', 'play-circle').
        style: QStyle instance for fallback icons. Required only when SVG
               is missing.

    Returns:
        QIcon instance.
    """
    if name in _icon_cache:
        return _icon_cache[name]

    svg_filename = _ICON_FILES.get(name)
    if svg_filename:
        svg_path = _ICONS_DIR / svg_filename
        if svg_path.exists():
            icon = _load_svg_icon(svg_path)
            _icon_cache[name] = icon
            return icon

    # Fallback to QStyle standard icon
    if style is not None and name in _FALLBACK_ICONS:
        icon = style.standardIcon(_FALLBACK_ICONS[name])
        _icon_cache[name] = icon
        return icon

    # Last resort: empty icon
    return QIcon()


def clear_icon_cache() -> None:
    """Clear the icon cache.

    Call after ``set_icon_color()`` so icons are re-rendered with the new
    theme color on next ``get_icon()`` call.
    """
    _icon_cache.clear()
