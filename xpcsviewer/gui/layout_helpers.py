"""
Layout helper utilities for XPCS-TOOLKIT GUI.

This module provides functions to improve the visual layout and organization
of the application at runtime, complementing the QSS styling.
"""

from typing import Literal

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QScrollArea, QSizePolicy, QSplitter

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import (
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
    group_box.update()


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
    button.update()


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
    separator.setObjectName("themeSeparator")
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
        layout.setContentsMargins(4, 14, 4, 4)
        layout.setSpacing(4)

    # Shrink-to-content vertically so plots get remaining space
    group_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

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
    for i in range(tab_widget.count()):  # type: ignore[attr-defined]
        page = tab_widget.widget(i)  # type: ignore[attr-defined]
        if page:
            layout = page.layout()
            if layout:
                layout.setContentsMargins(2, 2, 2, 2)
                layout.setSpacing(2)


def rearrange_g2_fitting_buttons(main_window: QWidget) -> None:
    """Rearrange g2 fitting Bayesian controls in gridLayout_12.

    Moves 'Fit All Q' below 'Plot Diagnosis' and adds a 'Q-bin:' label
    above 'Fit Bayesian' for clarity.
    """
    grid_12: QGridLayout | None = main_window.findChild(QGridLayout, "gridLayout_12")
    if grid_12 is None:
        return

    # Find widgets by objectName
    btn_bayesian = main_window.findChild(QPushButton, "btn_g2_bayesian")
    btn_diagnosis = main_window.findChild(QPushButton, "btn_g2_diagnosis")
    btn_all_q = main_window.findChild(QPushButton, "btn_g2_bayesian_all")
    sb_qidx = main_window.findChild(QWidget, "sb_g2_bayesian_qidx")
    sb_workers = main_window.findChild(QWidget, "sb_g2_bayesian_workers")

    if not all([btn_bayesian, btn_diagnosis, sb_qidx]):
        return

    # Remove widgets from layout without detaching parent.
    # Using removeWidget (not setParent(None)) preserves the native
    # window handle and input-method context — critical for QSpinBox.
    widgets_to_move = [
        btn_bayesian,
        btn_diagnosis,
        sb_qidx,
        btn_all_q,
        sb_workers,
    ]
    for w in widgets_to_move:
        if w is not None:
            grid_12.removeWidget(w)

    # Create Q-bin label
    q_label = QLabel("Q-bin:")
    q_label.setObjectName("label_g2_bayesian_qbin")

    # Re-add in new arrangement (cols 11-12)
    # Row 0: Q-bin label + spinbox
    grid_12.addWidget(q_label, 0, 11, 1, 1)
    grid_12.addWidget(sb_qidx, 0, 12, 1, 1)
    # Row 1: Fit Bayesian (full width)
    grid_12.addWidget(btn_bayesian, 1, 11, 1, 2)
    # Row 2: Plot Diagnosis (full width)
    grid_12.addWidget(btn_diagnosis, 2, 11, 1, 2)
    # Row 3: Fit All Q + workers spinbox
    if btn_all_q is not None:
        grid_12.addWidget(btn_all_q, 3, 11, 1, 1)
    if sb_workers is not None:
        grid_12.addWidget(sb_workers, 3, 12, 1, 1)


def rearrange_diffusion_tab(main_window: QWidget) -> None:
    """Rearrange the Diffusion tab for a compact, plot-dominant layout.

    1. Inside groupBox_5: params and controls side-by-side in one row.
    2. Move tauq_msg below groupBox_5 at full width with thin scrollbar.
    3. Maximise plot area, minimise control/metadata chrome.
    """
    _rearrange_diffusion_controls(main_window)
    _rearrange_diffusion_grid(main_window)
    _style_tauq_msg(main_window)


def _rearrange_diffusion_controls(main_window: QWidget) -> None:
    """Compact groupBox_5: params col 0, buttons col 1, options col 2."""
    grid_38: QGridLayout | None = main_window.findChild(QGridLayout, "gridLayout_38")
    if grid_38 is None:
        return

    # Drain all items, keyed by objectName
    layouts: dict[str, object] = {}
    widgets: dict[str, QWidget] = {}
    while grid_38.count():
        item = grid_38.takeAt(0)
        if item.layout():
            layouts[item.layout().objectName()] = item.layout()
        elif item.widget():
            widgets[item.widget().objectName()] = item.widget()

    grid_21 = layouts.get("gridLayout_21")  # a / b / q params
    grid_15 = layouts.get("gridLayout_15")  # plot_type + offset
    btn_fit = widgets.get("pushButton_8")
    btn_export = widgets.get("btn_export_diffusion")
    btn_bayesian = widgets.get("btn_diff_bayesian")
    btn_diagnosis = widgets.get("btn_diff_diagnosis")

    # Col 0: params (spans all rows)
    if grid_21:
        grid_38.addLayout(grid_21, 0, 0, 3, 1)

    # Col 1: action buttons stacked vertically
    if btn_fit:
        btn_fit.setMinimumSize(0, 0)
        grid_38.addWidget(btn_fit, 0, 1)
    if btn_export:
        btn_export.setMinimumSize(0, 0)
        grid_38.addWidget(btn_export, 1, 1)
    if btn_bayesian:
        btn_bayesian.setMinimumSize(0, 0)
        grid_38.addWidget(btn_bayesian, 2, 1)

    # Col 2: options + diagnosis
    if grid_15:
        grid_38.addLayout(grid_15, 0, 2, 2, 1)
    if btn_diagnosis:
        btn_diagnosis.setMinimumSize(0, 0)
        grid_38.addWidget(btn_diagnosis, 2, 2)

    grid_38.setColumnStretch(0, 3)
    grid_38.setColumnStretch(1, 1)
    grid_38.setColumnStretch(2, 1)
    grid_38.setContentsMargins(2, 2, 2, 2)
    grid_38.setSpacing(2)

    # Make groupBox_5 as compact as possible vertically
    group_box_5 = main_window.findChild(QGroupBox, "groupBox_5")
    if group_box_5 is not None:
        group_box_5.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        # Tight margins: minimal top (just enough for title), zero elsewhere
        gb_layout = group_box_5.layout()
        if gb_layout is not None:
            gb_layout.setContentsMargins(4, 14, 4, 2)
            gb_layout.setSpacing(2)


def _rearrange_diffusion_grid(main_window: QWidget) -> None:
    """Move tauq_msg below groupBox_5; maximise plot row height."""
    grid_22: QGridLayout | None = main_window.findChild(QGridLayout, "gridLayout_22")
    tauq_msg = main_window.findChild(QWidget, "tauq_msg")
    if grid_22 is None or tauq_msg is None:
        return

    # Remove tauq_msg from gridLayout_39 (inside groupBox_5)
    parent = tauq_msg.parentWidget()
    if parent is not None and parent.layout() is not None:
        parent.layout().removeWidget(tauq_msg)

    # Add to tab grid: row 2, spanning both plot columns
    grid_22.addWidget(tauq_msg, 2, 0, 1, 2)

    # Row stretch: plots take all flexible space; rows 1-2 fixed to content
    grid_22.setRowStretch(0, 1)
    grid_22.setRowStretch(1, 0)
    grid_22.setRowStretch(2, 0)

    # Tighten outer grid spacing
    grid_22.setContentsMargins(4, 4, 4, 4)
    grid_22.setSpacing(4)


def _style_tauq_msg(main_window: QWidget) -> None:
    """Configure tauq_msg with fixed height and thin scrollbar."""
    tauq_msg = main_window.findChild(QWidget, "tauq_msg")
    if tauq_msg is None:
        return

    tauq_msg.setFixedHeight(80)
    tauq_msg.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # Thin scrollbar via stylesheet
    tauq_msg.setStyleSheet(
        "QTreeWidget QScrollBar:vertical {"
        "  width: 6px;"
        "  background: transparent;"
        "}"
        "QTreeWidget QScrollBar::handle:vertical {"
        "  background: rgba(128, 128, 128, 0.4);"
        "  border-radius: 3px;"
        "  min-height: 20px;"
        "}"
        "QTreeWidget QScrollBar::add-line:vertical,"
        "QTreeWidget QScrollBar::sub-line:vertical {"
        "  height: 0px;"
        "}"
        "QScrollBar:vertical {"
        "  width: 6px;"
        "  background: transparent;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: rgba(128, 128, 128, 0.4);"
        "  border-radius: 3px;"
        "  min-height: 20px;"
        "}"
        "QScrollBar::add-line:vertical,"
        "QScrollBar::sub-line:vertical {"
        "  height: 0px;"
        "}"
    )

    # Enable horizontal scrollbar only when needed
    tree = tauq_msg
    if hasattr(tree, "setHorizontalScrollBarPolicy"):
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    if hasattr(tree, "setVerticalScrollBarPolicy"):
        tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


def optimize_g2map_tab(main_window: QWidget) -> None:
    """Optimize the G2 Map tab for equal panel sizing and tight layout.

    Forces the three splitter panels (G2 Map, Q-map, G2 Profile) to share
    space equally and minimises internal padding.
    """
    splitter: QSplitter | None = main_window.findChild(QSplitter, "splitter_g2map")
    if splitter is None:
        return

    # Tighten each panel's internal layout margins
    for i in range(splitter.count()):
        child = splitter.widget(i)
        if child and child.layout():
            child.layout().setContentsMargins(0, 0, 0, 0)
            child.layout().setSpacing(2)

    # Force equal initial sizes (Qt normalises proportionally)
    splitter.setSizes([10000, 10000, 10000])

    # Constrain the controls groupbox to shrink-to-content
    tab_g2map = main_window.findChild(QWidget, "tab_g2map")
    if tab_g2map:
        for gb in tab_g2map.findChildren(QGroupBox):
            gb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            gb_layout = gb.layout()
            if gb_layout:
                gb_layout.setContentsMargins(4, 14, 4, 2)
                gb_layout.setSpacing(4)


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

    # Improve control panel group boxes (apply settingsPanel property)
    control_panels = [
        "groupBox_3",  # SAXS 2D Plot Setting
        "groupBox_6",  # SAXS 1D Plot Setting
        "groupBox_4",  # Stability Plot Setting
        "groupBox_7",  # Intensity-Time Plot Setting
        "groupBox",  # G2 Data Selection
        "groupBox_2",  # G2 Fitting
        "groupBox_8",  # Two-time settings
        "groupBox_9",  # Two-time controls
        "groupBox_5",  # Q-Map settings
        "groupBox_10",  # Average settings
    ]

    for name in control_panels:
        group_box = main_window.findChild(QGroupBox, name)
        if group_box:
            improve_control_panel_layout(group_box)

    # Rearrange G2 Fitting tab Bayesian controls
    rearrange_g2_fitting_buttons(main_window)

    # Rearrange Diffusion tab layout
    rearrange_diffusion_tab(main_window)

    # Optimize G2 Map tab: equal panel sizes, tight margins
    optimize_g2map_tab(main_window)

    # Apply compact density to sidebar panels
    _apply_compact_density_to_sidebars(main_window)

    # Improve tab content spacing
    tab_widget = main_window.findChild(QWidget, "tabWidget")
    if tab_widget:
        improve_tab_content_spacing(tab_widget)

    # Final pass: tighten scroll areas, splitters, inner layouts
    _tighten_all_containers(main_window)


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


def _apply_compact_density_to_sidebars(main_window: QWidget) -> None:
    """Apply compact density to sidebar control panel widgets.

    Compact density reduces padding on form controls in panels that
    contain many stacked settings, making them easier to scan.

    Args:
        main_window: The XpcsViewer main window
    """
    # Widget names for sidebar splitter panels (left/right control areas)
    sidebar_splitter_names = [
        "splitter_3",
        "splitter_2",
        "widget_saxs2d_controls",
        "widget_g2_controls",
    ]

    for name in sidebar_splitter_names:
        widget = main_window.findChild(QWidget, name)
        if widget:
            widget.setProperty("density", "compact")
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()


def _tighten_all_containers(main_window: QWidget) -> None:
    """Remove padding from scroll areas, splitters, and inner layouts.

    This is a final pass that catches all remaining whitespace sources
    that tab-level and groupbox-level optimizations miss.
    """
    # 1. Remove frames from all QScrollArea widgets (removes 1-2px border)
    for sa in main_window.findChildren(QScrollArea):
        sa.setFrameShape(QFrame.Shape.NoFrame)
        # Tighten the scroll area's own content margins
        inner = sa.widget()
        if inner and inner.layout():
            inner.layout().setContentsMargins(0, 0, 0, 0)
            inner.layout().setSpacing(0)

    # 2. Reduce splitter handle widths (default ~5px → 2px)
    for sp in main_window.findChildren(QSplitter):
        sp.setHandleWidth(2)

    # 3. Tighten the QTabWidget's own internal margins
    tab_widget = main_window.findChild(QWidget, "tabWidget")
    if tab_widget:
        tab_widget.setContentsMargins(0, 0, 0, 0)  # type: ignore[arg-type]
