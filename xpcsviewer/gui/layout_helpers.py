"""
Layout helper utilities for XPCS-TOOLKIT GUI.

This module provides functions to improve the visual layout and organization
of the application at runtime, complementing the QSS styling.
"""

from typing import Literal

from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QLabel,
    QLayout,
    QPushButton,
    QWidget,
)


def set_panel_margins(widget: QWidget, margins: int = 12) -> None:
    """
    Set consistent margins on a widget's layout.

    Args:
        widget: Widget with a layout to modify
        margins: Margin size in pixels (default 12)
    """
    layout = widget.layout()
    if layout:
        layout.setContentsMargins(margins, margins, margins, margins)


def set_layout_spacing(widget: QWidget, spacing: int = 8) -> None:
    """
    Set consistent spacing on a widget's layout.

    Args:
        widget: Widget with a layout to modify
        spacing: Spacing between items in pixels
    """
    layout = widget.layout()
    if layout:
        layout.setSpacing(spacing)


def apply_group_box_styling(
    group_box: QGroupBox,
    style: Literal["panel", "card", "minimal"] = "panel",
) -> None:
    """
    Apply semantic styling to a QGroupBox.

    Args:
        group_box: The group box to style
        style: Visual style variant
            - "panel": Standard settings panel look
            - "card": Elevated card appearance
            - "minimal": Subtle, borderless appearance
    """
    if style == "panel":
        group_box.setProperty("settingsPanel", "true")
    elif style == "card":
        group_box.setProperty("cardStyle", "true")
    elif style == "minimal":
        group_box.setProperty("minimalStyle", "true")

    group_box.style().unpolish(group_box)
    group_box.style().polish(group_box)


def set_button_role(
    button: QPushButton,
    role: Literal["primary", "secondary", "action"] = "action",
) -> None:
    """
    Set the semantic role of a button for styling.

    Args:
        button: Button to style
        role: Button's role in the UI
            - "primary": Main action (Plot, Fit, etc.)
            - "secondary": Supporting action
            - "action": Default action button
    """
    if role == "primary":
        button.setProperty("buttonRole", "primary")
    elif role == "secondary":
        button.setProperty("buttonRole", None)
    else:
        button.setProperty("buttonRole", None)

    button.style().unpolish(button)
    button.style().polish(button)


def create_separator(
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> QFrame:
    """
    Create a visual separator line.

    Args:
        orientation: Line direction

    Returns:
        QFrame configured as a separator
    """
    separator = QFrame()
    if orientation == "horizontal":
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
    else:
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFixedWidth(1)

    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setStyleSheet("background-color: palette(mid);")
    return separator


def create_section_header(text: str, level: int = 1) -> QLabel:
    """
    Create a styled section header label.

    Args:
        text: Header text
        level: Header level (1 = main, 2 = sub)

    Returns:
        Styled QLabel
    """
    label = QLabel(text)
    if level == 1:
        label.setProperty("role", "sectionTitle")
    else:
        label.setProperty("role", "subTitle")
    return label


def improve_file_panel_layout(parent: QWidget) -> None:
    """
    Improve the file selection panel layout.

    Applies better spacing, margins, and visual organization to
    the source/target file list panels.

    Args:
        parent: Main window or widget containing the file panel
    """
    # Find and improve the file panel components
    box_source = parent.findChild(QGroupBox, "box_source")
    box_target = parent.findChild(QGroupBox, "box_target")

    if box_source:
        layout = box_source.layout()
        if layout:
            layout.setContentsMargins(8, 16, 8, 8)
            layout.setSpacing(4)

    if box_target:
        layout = box_target.layout()
        if layout:
            layout.setContentsMargins(8, 16, 8, 8)
            layout.setSpacing(4)


def improve_control_panel_layout(group_box: QGroupBox) -> None:
    """
    Improve the layout of a control panel group box.

    Args:
        group_box: The control panel to improve
    """
    layout = group_box.layout()
    if layout:
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(8)

    apply_group_box_styling(group_box, "panel")


def mark_primary_action_buttons(parent: QWidget) -> None:
    """
    Mark primary action buttons (Plot, Fit) for emphasis.

    Args:
        parent: Parent widget containing the buttons
    """
    primary_button_names = [
        "pushButton_plot_saxs2d",
        "pushButton_plot_saxs1d",
        "pushButton_plot_stability",
        "pushButton_plot_intt",
        "pushButton_4",  # G2 plot button
        "pushButton_8",  # Diffusion fit button
        "btn_start_avg_job",
    ]

    for name in primary_button_names:
        button = parent.findChild(QPushButton, name)
        if button:
            set_button_role(button, "primary")


def improve_tab_content_spacing(tab_widget: QWidget) -> None:
    """
    Improve spacing within tab content areas.

    Args:
        tab_widget: The QTabWidget to improve
    """
    for i in range(tab_widget.count()):
        page = tab_widget.widget(i)
        if page:
            layout = page.layout()
            if layout:
                layout.setContentsMargins(8, 8, 8, 8)
                layout.setSpacing(8)


def apply_all_layout_improvements(main_window: QWidget) -> None:
    """
    Apply all layout improvements to the main window.

    This is the main entry point for layout enhancements.

    Args:
        main_window: The XpcsViewer main window
    """
    # Improve file selection panel
    improve_file_panel_layout(main_window)

    # Mark primary action buttons
    mark_primary_action_buttons(main_window)

    # Improve control panel group boxes
    control_panels = [
        "groupBox_3",  # SAXS 2D Plot Setting
        "groupBox_6",  # SAXS 1D Plot Setting
        "groupBox_4",  # Stability Plot Setting
        "groupBox_7",  # Intensity-Time Plot Setting
        "groupBox",    # G2 Data Selection
        "groupBox_2",  # G2 Fitting
        "groupBox_8",  # Two-time settings
        "groupBox_9",  # Two-time controls
    ]

    for name in control_panels:
        group_box = main_window.findChild(QGroupBox, name)
        if group_box:
            improve_control_panel_layout(group_box)

    # Improve tab content spacing
    tab_widget = main_window.findChild(QWidget, "tabWidget")
    if tab_widget:
        improve_tab_content_spacing(tab_widget)


def add_visual_separator_before_action(
    layout: QLayout,
    position: int = -1,
) -> None:
    """
    Add a visual separator before action buttons in a layout.

    Args:
        layout: The layout to modify
        position: Position to insert (-1 for end)
    """
    separator = create_separator()
    if position < 0:
        position = layout.count() - 1
    if hasattr(layout, "insertWidget"):
        layout.insertWidget(position, separator)
