"""SimpleMask window UI setup.

This module provides the setupUi function for SimpleMaskWindow layout.
"""

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import (
    QAction,
    QButtonGroup,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    Qt,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from xpcsviewer.simplemask.drawing_tools import DRAWING_TOOLS, ERASER_TOOL
from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI


def setup_ui(window) -> None:  # noqa: PLR0915 - UI setup requires many statements
    """Set up the SimpleMask window UI.

    Creates a layout with:
    - Toolbar: Drawing tools, mask actions
    - Left panel: ImageView for detector display with ROI support
    - Right panel: Control panels for tools and parameters
    - Status bar for messages

    Args:
        window: SimpleMaskWindow instance to set up
    """
    # Central widget
    central_widget = QWidget()
    window.setCentralWidget(central_widget)

    # Main layout with splitter for resizable panels
    main_layout = QHBoxLayout(central_widget)
    main_layout.setContentsMargins(4, 4, 4, 4)
    main_layout.setSpacing(4)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    main_layout.addWidget(splitter)

    # Left panel: Image display
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0)

    # PyQtGraph ImageView with ROI support
    window.image_view = ImageViewROI()
    window.image_view.ui.histogram.hide()  # Hide histogram by default
    window.image_view.ui.roiBtn.hide()  # Hide ROI button (we use our own tools)
    window.image_view.ui.menuBtn.hide()  # Hide menu button
    left_layout.addWidget(window.image_view)

    splitter.addWidget(left_panel)

    # Right panel: Controls
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(4, 4, 4, 4)

    # Info label at top
    window.info_label = QLabel("Load data from XPCS Viewer to begin")
    window.info_label.setWordWrap(True)
    right_layout.addWidget(window.info_label)

    # Drawing Tools Group
    drawing_group = QGroupBox("Drawing Tools")
    drawing_layout = QVBoxLayout(drawing_group)

    # Tool buttons
    tool_buttons_layout = QHBoxLayout()
    window.tool_buttons = {}
    window.tool_button_group = QButtonGroup(window)
    window.tool_button_group.setExclusive(True)

    for tool_key, tool in DRAWING_TOOLS.items():
        btn = QToolButton()
        btn.setText(tool.name)
        btn.setToolTip(tool.tooltip)
        btn.setCheckable(True)
        btn.setMinimumWidth(60)
        window.tool_buttons[tool_key] = btn
        window.tool_button_group.addButton(btn)
        tool_buttons_layout.addWidget(btn)

    drawing_layout.addLayout(tool_buttons_layout)

    # Eraser button (separate, not in exclusive group for drawing)
    eraser_layout = QHBoxLayout()
    window.btn_eraser = QToolButton()
    window.btn_eraser.setText(ERASER_TOOL.name)
    window.btn_eraser.setToolTip(ERASER_TOOL.tooltip)
    window.btn_eraser.setCheckable(True)
    window.btn_eraser.setMinimumWidth(60)
    eraser_layout.addWidget(window.btn_eraser)
    eraser_layout.addStretch()
    drawing_layout.addLayout(eraser_layout)

    # Apply drawing button
    apply_layout = QHBoxLayout()
    window.btn_apply_drawing = QPushButton("Apply Drawing")
    window.btn_apply_drawing.setToolTip("Apply all ROIs to create mask")
    apply_layout.addWidget(window.btn_apply_drawing)
    apply_layout.addStretch()
    drawing_layout.addLayout(apply_layout)

    right_layout.addWidget(drawing_group)

    # Mask Actions Group
    mask_group = QGroupBox("Mask Actions")
    mask_layout = QVBoxLayout(mask_group)

    mask_buttons_layout = QHBoxLayout()

    window.btn_undo = QPushButton("Undo")
    window.btn_undo.setToolTip("Undo last mask change (Ctrl+Z)")
    window.btn_undo.setShortcut(QKeySequence.StandardKey.Undo)
    mask_buttons_layout.addWidget(window.btn_undo)

    window.btn_redo = QPushButton("Redo")
    window.btn_redo.setToolTip("Redo mask change (Ctrl+Y)")
    window.btn_redo.setShortcut(QKeySequence.StandardKey.Redo)
    mask_buttons_layout.addWidget(window.btn_redo)

    window.btn_reset = QPushButton("Reset")
    window.btn_reset.setToolTip("Reset mask to initial state")
    mask_buttons_layout.addWidget(window.btn_reset)

    mask_layout.addLayout(mask_buttons_layout)

    # Show/hide mask overlay toggle
    overlay_layout = QHBoxLayout()
    window.btn_toggle_mask = QPushButton("Show Mask")
    window.btn_toggle_mask.setCheckable(True)
    window.btn_toggle_mask.setToolTip("Toggle mask overlay display")
    overlay_layout.addWidget(window.btn_toggle_mask)
    overlay_layout.addStretch()
    mask_layout.addLayout(overlay_layout)

    right_layout.addWidget(mask_group)

    # Geometry Parameters Group
    geometry_group = QGroupBox("Geometry Parameters")
    geometry_layout = QFormLayout(geometry_group)

    # Beam center X
    window.spin_bcx = QDoubleSpinBox()
    window.spin_bcx.setRange(0, 10000)
    window.spin_bcx.setDecimals(2)
    window.spin_bcx.setSuffix(" px")
    window.spin_bcx.setToolTip("Beam center X (column)")
    geometry_layout.addRow("Beam X:", window.spin_bcx)

    # Beam center Y
    window.spin_bcy = QDoubleSpinBox()
    window.spin_bcy.setRange(0, 10000)
    window.spin_bcy.setDecimals(2)
    window.spin_bcy.setSuffix(" px")
    window.spin_bcy.setToolTip("Beam center Y (row)")
    geometry_layout.addRow("Beam Y:", window.spin_bcy)

    # Detector distance
    window.spin_det_dist = QDoubleSpinBox()
    window.spin_det_dist.setRange(1, 100000)
    window.spin_det_dist.setDecimals(1)
    window.spin_det_dist.setSuffix(" mm")
    window.spin_det_dist.setValue(5000.0)
    window.spin_det_dist.setToolTip("Sample-to-detector distance")
    geometry_layout.addRow("Distance:", window.spin_det_dist)

    # Pixel size
    window.spin_pix_dim = QDoubleSpinBox()
    window.spin_pix_dim.setRange(0.001, 10)
    window.spin_pix_dim.setDecimals(4)
    window.spin_pix_dim.setSuffix(" mm")
    window.spin_pix_dim.setValue(0.075)
    window.spin_pix_dim.setToolTip("Pixel dimension")
    geometry_layout.addRow("Pixel Size:", window.spin_pix_dim)

    # X-ray energy
    window.spin_energy = QDoubleSpinBox()
    window.spin_energy.setRange(0.1, 100)
    window.spin_energy.setDecimals(3)
    window.spin_energy.setSuffix(" keV")
    window.spin_energy.setValue(10.0)
    window.spin_energy.setToolTip("X-ray energy")
    geometry_layout.addRow("Energy:", window.spin_energy)

    right_layout.addWidget(geometry_group)

    # Q-Map Actions Group
    qmap_group = QGroupBox("Q-Map")
    qmap_layout = QVBoxLayout(qmap_group)

    # Generate Q-Map button
    qmap_buttons_layout = QHBoxLayout()
    window.btn_generate_qmap = QPushButton("Generate Q-Map")
    window.btn_generate_qmap.setToolTip("Compute Q-map from geometry parameters")
    qmap_buttons_layout.addWidget(window.btn_generate_qmap)
    qmap_buttons_layout.addStretch()
    qmap_layout.addLayout(qmap_buttons_layout)

    # Show Q-Map overlay toggle
    qmap_overlay_layout = QHBoxLayout()
    window.btn_toggle_qmap = QPushButton("Show Q-Map")
    window.btn_toggle_qmap.setCheckable(True)
    window.btn_toggle_qmap.setToolTip("Toggle Q-map overlay display")
    qmap_overlay_layout.addWidget(window.btn_toggle_qmap)
    qmap_overlay_layout.addStretch()
    qmap_layout.addLayout(qmap_overlay_layout)

    right_layout.addWidget(qmap_group)

    # Partition (Q-Binning) Group
    partition_group = QGroupBox("Partition (Q-Binning)")
    partition_layout = QFormLayout(partition_group)

    # Dynamic Q bins
    window.spin_dq_num = QSpinBox()
    window.spin_dq_num.setRange(1, 1000)
    window.spin_dq_num.setValue(10)
    window.spin_dq_num.setToolTip("Number of dynamic Q-bins")
    partition_layout.addRow("Dynamic Q:", window.spin_dq_num)

    # Static Q bins
    window.spin_sq_num = QSpinBox()
    window.spin_sq_num.setRange(1, 1000)
    window.spin_sq_num.setValue(100)
    window.spin_sq_num.setToolTip("Number of static Q-bins")
    partition_layout.addRow("Static Q:", window.spin_sq_num)

    # Dynamic phi bins
    window.spin_dp_num = QSpinBox()
    window.spin_dp_num.setRange(1, 360)
    window.spin_dp_num.setValue(36)
    window.spin_dp_num.setToolTip("Number of dynamic phi-bins")
    partition_layout.addRow("Dynamic φ:", window.spin_dp_num)

    # Static phi bins
    window.spin_sp_num = QSpinBox()
    window.spin_sp_num.setRange(1, 360)
    window.spin_sp_num.setValue(360)
    window.spin_sp_num.setToolTip("Number of static phi-bins")
    partition_layout.addRow("Static φ:", window.spin_sp_num)

    # Compute Partition button
    partition_buttons_layout = QHBoxLayout()
    window.btn_compute_partition = QPushButton("Compute Partition")
    window.btn_compute_partition.setToolTip(
        "Compute Q-space partition from current settings"
    )
    partition_buttons_layout.addWidget(window.btn_compute_partition)
    partition_layout.addRow(partition_buttons_layout)

    # Show partition overlay toggle
    partition_overlay_layout = QHBoxLayout()
    window.btn_toggle_partition = QPushButton("Show Partition")
    window.btn_toggle_partition.setCheckable(True)
    window.btn_toggle_partition.setToolTip("Toggle partition overlay display")
    partition_overlay_layout.addWidget(window.btn_toggle_partition)
    partition_layout.addRow(partition_overlay_layout)

    # Export partition button
    export_layout = QHBoxLayout()
    window.btn_export_partition = QPushButton("Export to Viewer")
    window.btn_export_partition.setToolTip("Export partition to XPCS Viewer")
    export_layout.addWidget(window.btn_export_partition)
    partition_layout.addRow(export_layout)

    right_layout.addWidget(partition_group)

    # Stretch to push remaining controls to bottom
    right_layout.addStretch()

    splitter.addWidget(right_panel)

    # Set splitter sizes (70% image, 30% controls)
    splitter.setSizes([700, 300])
    splitter.setStretchFactor(0, 7)
    splitter.setStretchFactor(1, 3)

    # Store references
    window.splitter = splitter
    window.left_panel = left_panel
    window.right_panel = right_panel
    window.right_layout = right_layout

    # Create menu bar
    _setup_menubar(window)

    # Create toolbar
    _setup_toolbar(window)

    # Status bar
    window.status_bar = QStatusBar()
    window.setStatusBar(window.status_bar)
    window.status_bar.showMessage("Ready")

    # Set window properties
    window.setMinimumSize(1000, 700)
    window.resize(1200, 800)


def _setup_toolbar(window) -> None:
    """Set up the main toolbar with drawing tools and actions.

    Args:
        window: SimpleMaskWindow instance
    """
    toolbar = QToolBar("Main Toolbar")
    toolbar.setMovable(False)
    window.addToolBar(toolbar)

    # Drawing tool actions
    window.tool_actions = {}
    for tool_key, tool in DRAWING_TOOLS.items():
        action = QAction(tool.name, window)
        action.setToolTip(tool.tooltip)
        action.setCheckable(True)
        if tool.shortcut:
            action.setShortcut(QKeySequence(tool.shortcut))
        window.tool_actions[tool_key] = action
        toolbar.addAction(action)

    toolbar.addSeparator()

    # Eraser action
    window.action_eraser = QAction(ERASER_TOOL.name, window)
    window.action_eraser.setToolTip(ERASER_TOOL.tooltip)
    window.action_eraser.setCheckable(True)
    if ERASER_TOOL.shortcut:
        window.action_eraser.setShortcut(QKeySequence(ERASER_TOOL.shortcut))
    toolbar.addAction(window.action_eraser)

    toolbar.addSeparator()

    # Apply drawing action
    window.action_apply = QAction("Apply", window)
    window.action_apply.setToolTip("Apply all drawings to mask")
    toolbar.addAction(window.action_apply)

    toolbar.addSeparator()

    # Mask actions
    window.action_undo = QAction("Undo", window)
    window.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
    window.action_undo.setToolTip("Undo last mask change (Ctrl+Z)")
    toolbar.addAction(window.action_undo)

    window.action_redo = QAction("Redo", window)
    window.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
    window.action_redo.setToolTip("Redo mask change (Ctrl+Y)")
    toolbar.addAction(window.action_redo)

    toolbar.addSeparator()

    # Apply to Viewer button
    window.action_apply_viewer = QAction("Apply to Viewer", window)
    window.action_apply_viewer.setToolTip(
        "Export current mask to XPCS Viewer for analysis"
    )
    toolbar.addAction(window.action_apply_viewer)

    toolbar.addSeparator()

    # Generate Q-Map action
    window.action_generate_qmap = QAction("Generate Q-Map", window)
    window.action_generate_qmap.setToolTip("Compute Q-map from geometry parameters")
    toolbar.addAction(window.action_generate_qmap)

    # Toggle Q-Map overlay
    window.action_toggle_qmap = QAction("Show Q-Map", window)
    window.action_toggle_qmap.setCheckable(True)
    window.action_toggle_qmap.setToolTip("Toggle Q-map overlay display")
    toolbar.addAction(window.action_toggle_qmap)

    toolbar.addSeparator()

    # Toggle mask overlay
    window.action_toggle_mask = QAction("Show Mask", window)
    window.action_toggle_mask.setCheckable(True)
    window.action_toggle_mask.setToolTip("Toggle mask overlay display")
    toolbar.addAction(window.action_toggle_mask)

    window.toolbar = toolbar


def _setup_menubar(window) -> None:
    """Set up the menu bar with File menu.

    Args:
        window: SimpleMaskWindow instance
    """
    menubar = window.menuBar()

    # File menu
    file_menu = menubar.addMenu("&File")

    # Save Mask action
    window.action_save_mask = QAction("&Save Mask...", window)
    window.action_save_mask.setShortcut(QKeySequence.StandardKey.Save)
    window.action_save_mask.setToolTip("Save mask to HDF5 file (Ctrl+S)")
    file_menu.addAction(window.action_save_mask)

    # Load Mask action
    window.action_load_mask = QAction("&Load Mask...", window)
    window.action_load_mask.setShortcut(QKeySequence.StandardKey.Open)
    window.action_load_mask.setToolTip("Load mask from HDF5 file (Ctrl+O)")
    file_menu.addAction(window.action_load_mask)

    file_menu.addSeparator()

    # Apply to Viewer action
    window.action_apply_to_viewer = QAction("&Apply to Viewer", window)
    window.action_apply_to_viewer.setToolTip("Export current mask to XPCS Viewer")
    file_menu.addAction(window.action_apply_to_viewer)

    file_menu.addSeparator()

    # Close action
    window.action_close = QAction("&Close", window)
    window.action_close.setShortcut(QKeySequence.StandardKey.Close)
    window.action_close.setToolTip("Close mask editor window")
    file_menu.addAction(window.action_close)

    window.file_menu = file_menu
