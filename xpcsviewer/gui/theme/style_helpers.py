"""
Style helper utilities for XPCS-TOOLKIT GUI.

This module provides functions to apply dynamic styling properties to widgets,
enabling the use of QSS property selectors defined in base.qss.
"""

from typing import Literal

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import QGroupBox, QPushButton, QWidget

ButtonStyle = Literal["primary", "secondary", "destructive", "icon"]
ButtonSize = Literal["normal", "small"]


def set_button_style(button: QPushButton, style: ButtonStyle) -> None:
    """
    Set the visual style of a button.

    Available styles:
    - "primary": Default filled button (accent color)
    - "secondary": Outlined button with transparent background
    - "destructive": Red outlined button for remove/delete actions
    - "icon": Minimal icon-only button

    Args:
        button: The QPushButton to style
        style: One of "primary", "secondary", "destructive", "icon"
    """
    if style == "primary":
        # Primary is the default, remove any custom property
        button.setProperty("buttonStyle", None)
    else:
        button.setProperty("buttonStyle", style)

    # Force style refresh
    button.style().unpolish(button)
    button.style().polish(button)


def set_button_size(button: QPushButton, size: ButtonSize) -> None:
    """
    Set the size variant of a button.

    Args:
        button: The QPushButton to size
        size: One of "normal", "small"
    """
    if size == "normal":
        button.setProperty("buttonSize", None)
    else:
        button.setProperty("buttonSize", size)

    button.style().unpolish(button)
    button.style().polish(button)


def set_card_style(group_box: QGroupBox, enabled: bool = True) -> None:
    """
    Apply card styling to a QGroupBox.

    Card style provides a elevated, bordered appearance with
    a colored title badge.

    Args:
        group_box: The QGroupBox to style
        enabled: Whether to enable card style
    """
    group_box.setProperty("cardStyle", "true" if enabled else None)
    group_box.style().unpolish(group_box)
    group_box.style().polish(group_box)


def set_control_row(widget: QWidget, enabled: bool = True) -> None:
    """
    Apply control row styling to a widget container.

    Control row style provides a subtle background and border
    for grouping related controls horizontally.

    Args:
        widget: The container widget
        enabled: Whether to enable the style
    """
    widget.setProperty("controlRow", "true" if enabled else None)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_density(widget: QWidget, density: Literal["normal", "compact"]) -> None:
    """
    Set the density variant for a widget and its children.

    Compact density reduces spacing and control heights for
    information-dense interfaces.

    Args:
        widget: The widget (affects all children)
        density: One of "normal", "compact"
    """
    if density == "normal":
        widget.setProperty("density", None)
    else:
        widget.setProperty("density", density)

    widget.style().unpolish(widget)
    widget.style().polish(widget)


def apply_destructive_buttons(widget: QWidget, button_names: list[str]) -> None:
    """
    Apply destructive styling to multiple buttons by object name.

    Useful for bulk-styling remove/delete buttons across a UI.

    Args:
        widget: Parent widget containing the buttons
        button_names: List of button object names to style
    """
    for name in button_names:
        button = widget.findChild(QPushButton, name)
        if button is not None:
            set_button_style(button, "destructive")


def apply_secondary_buttons(widget: QWidget, button_names: list[str]) -> None:
    """
    Apply secondary styling to multiple buttons by object name.

    Args:
        widget: Parent widget containing the buttons
        button_names: List of button object names to style
    """
    for name in button_names:
        button = widget.findChild(QPushButton, name)
        if button is not None:
            set_button_style(button, "secondary")


def set_button_role(button: QPushButton, role: str) -> None:
    """
    Set the semantic role of a button for emphasis styling.

    Args:
        button: The button to style
        role: Role name (e.g., "primary" for emphasized buttons)
    """
    button.setProperty("buttonRole", role)
    button.style().unpolish(button)
    button.style().polish(button)


def apply_primary_buttons(widget: QWidget, button_names: list[str]) -> None:
    """
    Apply primary (emphasized) styling to action buttons.

    Args:
        widget: Parent widget containing the buttons
        button_names: List of button object names to style
    """
    for name in button_names:
        button = widget.findChild(QPushButton, name)
        if button is not None:
            set_button_role(button, "primary")


def set_settings_panel(group_box: QGroupBox) -> None:
    """
    Apply settings panel styling to a QGroupBox.

    Args:
        group_box: The group box to style as a settings panel
    """
    group_box.setProperty("settingsPanel", "true")
    group_box.style().unpolish(group_box)
    group_box.style().polish(group_box)
