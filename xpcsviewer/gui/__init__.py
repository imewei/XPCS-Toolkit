"""
GUI components for XPCS-TOOLKIT.

Submodules:
- theme: Design tokens and ThemeManager
- state: Session, preferences, and recent paths management
- widgets: Reusable custom widgets
- shortcuts: Keyboard shortcut management
"""

from xpcsviewer.gui import shortcuts, state, theme, widgets

__all__ = [
    "shortcuts",
    "state",
    "theme",
    "widgets",
]
