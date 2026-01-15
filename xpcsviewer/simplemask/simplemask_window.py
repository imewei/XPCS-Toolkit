"""SimpleMask window for mask editing.

This module provides the main QMainWindow for SimpleMask integration
with XPCS Viewer.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray
# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import (
    QAction,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    Signal,
    QWidget,
)

from xpcsviewer.simplemask.drawing_tools import (
    DRAWING_TOOLS,
    ERASER_TOOL,
    get_tool_color,
)
from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel
from xpcsviewer.simplemask.ui.simplemask_ui import setup_ui

logger = logging.getLogger(__name__)


class SimpleMaskWindow(QMainWindow):
    """Main window for SimpleMask mask editor.

    This window provides mask creation and editing tools integrated
    with XPCS Viewer.

    Signals:
        mask_exported: Emitted with mask array when user clicks "Apply to Viewer"
        qmap_exported: Emitted with partition dict when user exports Q-map
    """

    mask_exported = Signal(np.ndarray)
    qmap_exported = Signal(dict)

    # UI Components (injected by setup_ui)
    status_bar: QStatusBar
    image_view: ImageViewROI
    info_label: QLabel
    tool_buttons: dict[str, QToolButton]
    tool_button_group: QButtonGroup
    tool_actions: dict[str, QAction]
    toolbar: QToolBar
    splitter: QSplitter
    left_panel: QWidget
    right_panel: QWidget
    right_layout: QVBoxLayout
    file_menu: QMenu

    # Buttons
    btn_eraser: QToolButton
    btn_apply_drawing: QPushButton
    btn_undo: QPushButton
    btn_redo: QPushButton
    btn_reset: QPushButton
    btn_toggle_mask: QPushButton
    btn_generate_qmap: QPushButton
    btn_toggle_qmap: QPushButton
    btn_compute_partition: QPushButton
    btn_toggle_partition: QPushButton
    btn_export_partition: QPushButton

    # SpinBoxes
    spin_bcx: QDoubleSpinBox
    spin_bcy: QDoubleSpinBox
    spin_det_dist: QDoubleSpinBox
    spin_pix_dim: QDoubleSpinBox
    spin_energy: QDoubleSpinBox
    spin_dq_num: QSpinBox
    spin_sq_num: QSpinBox
    spin_dp_num: QSpinBox
    spin_sp_num: QSpinBox

    # Actions
    action_eraser: QAction
    action_apply: QAction
    action_undo: QAction
    action_redo: QAction
    action_apply_viewer: QAction
    action_generate_qmap: QAction
    action_toggle_qmap: QAction
    action_toggle_mask: QAction
    action_save_mask: QAction
    action_load_mask: QAction
    action_apply_to_viewer: QAction
    action_close: QAction

    def __init__(self, parent_viewer: Any = None):
        """Initialize SimpleMask window.

        Args:
            parent_viewer: Reference to parent XPCS Viewer instance
        """
        super().__init__()
        self.parent_viewer = parent_viewer
        self.kernel: SimpleMaskKernel | None = None
        self._unsaved_changes = False
        self._detector_image: NDArray[np.float64] | None = None
        self._metadata: dict | None = None
        self._current_tool: str | None = None
        self._show_mask_overlay = False
        self._show_qmap_overlay = False
        self._show_partition_overlay = False
        self._last_save_path: str | None = None

        self.setWindowTitle("Mask Editor")

        # Set up the UI
        setup_ui(self)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Connect UI signals to handlers."""
        # Drawing tool actions (toolbar)
        for tool_key, action in self.tool_actions.items():
            action.triggered.connect(
                lambda checked, k=tool_key: self._on_tool_selected(k)  # noqa: ARG005
            )

        # Drawing tool buttons (side panel)
        for tool_key, btn in self.tool_buttons.items():
            btn.clicked.connect(
                lambda checked, k=tool_key: self._on_tool_selected(k)  # noqa: ARG005
            )

        # Eraser
        self.action_eraser.triggered.connect(lambda: self._on_eraser_selected())
        self.btn_eraser.clicked.connect(lambda: self._on_eraser_selected())

        # Apply drawing
        self.action_apply.triggered.connect(self._on_apply_drawing)
        self.btn_apply_drawing.clicked.connect(self._on_apply_drawing)

        # Mask actions
        self.action_undo.triggered.connect(lambda: self._on_mask_action("undo"))
        self.action_redo.triggered.connect(lambda: self._on_mask_action("redo"))
        self.btn_undo.clicked.connect(lambda: self._on_mask_action("undo"))
        self.btn_redo.clicked.connect(lambda: self._on_mask_action("redo"))
        self.btn_reset.clicked.connect(lambda: self._on_mask_action("reset"))

        # Toggle mask overlay
        self.action_toggle_mask.triggered.connect(self._on_toggle_mask)
        self.btn_toggle_mask.clicked.connect(self._on_toggle_mask)

        # File menu actions
        self.action_save_mask.triggered.connect(self._on_save_mask)
        self.action_load_mask.triggered.connect(self._on_load_mask)
        self.action_apply_to_viewer.triggered.connect(self.export_mask_to_viewer)
        self.action_close.triggered.connect(self.close)

        # Toolbar Apply to Viewer button
        self.action_apply_viewer.triggered.connect(self.export_mask_to_viewer)

        # Q-Map controls
        self.btn_generate_qmap.clicked.connect(self._on_generate_qmap)
        self.action_generate_qmap.triggered.connect(self._on_generate_qmap)
        self.btn_toggle_qmap.clicked.connect(self._on_toggle_qmap)
        self.action_toggle_qmap.triggered.connect(self._on_toggle_qmap)

        # Geometry spinbox changes update kernel
        self.spin_bcx.valueChanged.connect(self._on_geometry_changed)
        self.spin_bcy.valueChanged.connect(self._on_geometry_changed)
        self.spin_det_dist.valueChanged.connect(self._on_geometry_changed)
        self.spin_pix_dim.valueChanged.connect(self._on_geometry_changed)
        self.spin_energy.valueChanged.connect(self._on_geometry_changed)

        # Partition controls
        self.btn_compute_partition.clicked.connect(self._on_compute_partition)
        self.btn_toggle_partition.clicked.connect(self._on_toggle_partition)
        self.btn_export_partition.clicked.connect(self.export_partition_to_viewer)

    def _on_tool_selected(self, tool_key: str) -> None:
        """Handle drawing tool selection.

        Args:
            tool_key: Key of selected tool from DRAWING_TOOLS
        """
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first before drawing")
            return

        tool = DRAWING_TOOLS.get(tool_key)
        if tool is None:
            return

        self._current_tool = tool_key
        color = get_tool_color(tool.default_mode)

        # Add ROI to the view
        roi = self.kernel.add_drawing(
            sl_type=tool.tool_type,
            sl_mode=tool.default_mode,
            color=color,
        )

        if roi is not None:
            self.status_bar.showMessage(
                f"Added {tool.name} - drag to position, click Apply when done"
            )
            logger.debug(f"Added {tool.name} ROI")

        # Sync button states
        self._sync_tool_buttons(tool_key)

    def _on_eraser_selected(self) -> None:
        """Handle eraser tool selection."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first before drawing")
            return

        self._current_tool = "eraser"
        color = get_tool_color(ERASER_TOOL.default_mode)

        # Add inclusive ROI (eraser)
        roi = self.kernel.add_drawing(
            sl_type=ERASER_TOOL.tool_type,
            sl_mode=ERASER_TOOL.default_mode,
            color=color,
        )

        if roi is not None:
            self.status_bar.showMessage(
                "Added Eraser region - drag to position, click Apply to remove from mask"
            )
            logger.debug("Added Eraser ROI")

        # Sync button states
        self._sync_tool_buttons("eraser")

    def _sync_tool_buttons(self, selected_tool: str | None) -> None:
        """Synchronize tool button checked states.

        Args:
            selected_tool: Currently selected tool key, or None
        """
        # Uncheck all tool buttons
        for btn in self.tool_buttons.values():
            btn.setChecked(False)
        for action in self.tool_actions.values():
            action.setChecked(False)

        self.btn_eraser.setChecked(False)
        self.action_eraser.setChecked(False)

        # Check the selected tool
        if selected_tool == "eraser":
            self.btn_eraser.setChecked(True)
            self.action_eraser.setChecked(True)
        elif selected_tool in self.tool_buttons:
            self.tool_buttons[selected_tool].setChecked(True)
            if selected_tool in self.tool_actions:
                self.tool_actions[selected_tool].setChecked(True)

    def _on_apply_drawing(self) -> None:
        """Apply all pending drawings to the mask."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("No data loaded")
            return

        # Apply drawings from ROIs
        mask_from_drawing = self.kernel.apply_drawing()

        # Update mask via mask assembly
        if self.kernel.mask_kernel is not None:
            self.kernel.mask_kernel.evaluate("mask_draw", arr=~mask_from_drawing)
            self.kernel.mask_apply("mask_draw")

        self.mark_unsaved()
        self._refresh_mask_display()
        self.status_bar.showMessage("Applied drawings to mask")
        logger.info("Applied drawings to mask")

    def _on_mask_action(self, action: Literal["undo", "redo", "reset"]) -> None:
        """Handle undo/redo/reset mask action.

        Args:
            action: Action type
        """
        if self.kernel is None:
            logger.debug(f"_on_mask_action({action}): No kernel loaded")
            return

        logger.debug(f"Mask action: {action}")
        self.kernel.mask_action(action)
        self._refresh_mask_display()

        if action == "reset":
            self.mark_saved()
            self.status_bar.showMessage("Mask reset to initial state")
            logger.info("Mask reset to initial state")
        else:
            self.status_bar.showMessage(f"Mask {action}")

    def _on_save_mask(self) -> None:
        """Handle save mask action with file dialog."""
        if self.kernel is None or self.kernel.mask is None:
            self.status_bar.showMessage("No mask to save")
            return

        # Suggest default filename
        default_name = "mask.h5"
        default_path = self._last_save_path or str(Path.home() / default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            default_path,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not file_path:
            return

        # Ensure .h5 extension
        if not file_path.endswith((".h5", ".hdf5")):
            file_path += ".h5"

        self.kernel.save_mask(file_path)
        self._last_save_path = file_path
        self.mark_saved()
        self.status_bar.showMessage(f"Mask saved to {Path(file_path).name}")
        logger.info(f"Saved mask to {file_path}")

    def _on_load_mask(self) -> None:
        """Handle load mask action with file dialog and validation."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first before loading a mask")
            return

        # Start from last save location or home
        start_path = self._last_save_path or str(Path.home())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            start_path,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not file_path:
            return

        # Validate mask dimensions before loading
        if not self._validate_mask_dimensions(file_path):
            return

        # Load the mask
        success = self.kernel.load_mask(file_path)
        if success:
            self._last_save_path = file_path
            self._refresh_mask_display()
            self.status_bar.showMessage(f"Loaded mask from {Path(file_path).name}")
            logger.info(f"Loaded mask from {file_path}")
        else:
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load mask from:\n{file_path}",
            )

    def _validate_mask_dimensions(self, file_path: str) -> bool:
        """Validate that mask file dimensions match current detector shape.

        Args:
            file_path: Path to HDF5 mask file

        Returns:
            True if dimensions match or user confirms override
        """
        if self.kernel is None or self.kernel.shape is None:
            return False

        try:
            with h5py.File(file_path, "r") as hf:
                if "mask" not in hf:
                    QMessageBox.warning(
                        self,
                        "Invalid Mask File",
                        f"No 'mask' dataset found in:\n{file_path}",
                    )
                    return False

                mask_shape = hf["mask"].shape
                detector_shape = self.kernel.shape

                if mask_shape != detector_shape:
                    reply = QMessageBox.warning(
                        self,
                        "Dimension Mismatch",
                        f"Mask dimensions {mask_shape} do not match\n"
                        f"detector dimensions {detector_shape}.\n\n"
                        "Load anyway? (mask will be cropped or padded)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.No:
                        return False

        except OSError as e:
            QMessageBox.critical(
                self,
                "File Error",
                f"Could not read mask file:\n{file_path}\n\nError: {e}",
            )
            return False

        return True

    def _on_toggle_mask(self) -> None:
        """Toggle mask overlay visibility."""
        self._show_mask_overlay = not self._show_mask_overlay
        if self._show_mask_overlay:
            self._show_qmap_overlay = False  # Can only show one overlay at a time

        # Sync button states
        self.btn_toggle_mask.setChecked(self._show_mask_overlay)
        self.action_toggle_mask.setChecked(self._show_mask_overlay)
        self.btn_toggle_qmap.setChecked(False)
        self.action_toggle_qmap.setChecked(False)

        # Update button text
        text = "Hide Mask" if self._show_mask_overlay else "Show Mask"
        self.btn_toggle_mask.setText(text)
        self.action_toggle_mask.setText(text)
        self.btn_toggle_qmap.setText("Show Q-Map")
        self.action_toggle_qmap.setText("Show Q-Map")

        self._refresh_display()

    def _on_toggle_qmap(self) -> None:
        """Toggle Q-map overlay visibility."""
        if self.kernel is None or self.kernel.qmap is None:
            self.status_bar.showMessage("Generate Q-map first")
            return

        self._show_qmap_overlay = not self._show_qmap_overlay
        if self._show_qmap_overlay:
            self._show_mask_overlay = False  # Can only show one overlay at a time

        # Sync button states
        self.btn_toggle_qmap.setChecked(self._show_qmap_overlay)
        self.action_toggle_qmap.setChecked(self._show_qmap_overlay)
        self.btn_toggle_mask.setChecked(False)
        self.action_toggle_mask.setChecked(False)

        # Update button text
        text = "Hide Q-Map" if self._show_qmap_overlay else "Show Q-Map"
        self.btn_toggle_qmap.setText(text)
        self.action_toggle_qmap.setText(text)
        self.btn_toggle_mask.setText("Show Mask")
        self.action_toggle_mask.setText("Show Mask")

        self._refresh_display()

    def _on_generate_qmap(self) -> None:
        """Generate Q-map from current geometry parameters."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first")
            return

        # Get geometry from spinboxes
        new_metadata = {
            "bcx": self.spin_bcx.value(),
            "bcy": self.spin_bcy.value(),
            "det_dist": self.spin_det_dist.value(),
            "pix_dim": self.spin_pix_dim.value(),
            "energy": self.spin_energy.value(),
        }

        # Update kernel parameters
        self.kernel.update_parameters(new_metadata)

        try:
            qmap, _units = self.kernel.compute_qmap()
        except ValueError as e:
            # Handle missing/invalid parameters
            logger.warning(f"Q-map generation failed: {e}")
            self.status_bar.showMessage(f"Q-map failed: check geometry values")
            QMessageBox.warning(
                self,
                "Q-Map Generation Failed",
                f"Failed to generate Q-map:\n\n{e}",
            )
            return

        # Initialize mask_kernel if this is the first successful Q-map computation
        # (happens when initial load failed due to missing geometry)
        # Initialize mask_kernel if this is the first successful Q-map computation
        # (happens when initial load failed due to missing geometry)
        if (
            self.kernel.mask_kernel is None
            and self.kernel.detector_image is not None
            and self.kernel.shape is not None
        ):
            from xpcsviewer.simplemask.area_mask import MaskAssemble

            self.kernel.mask_kernel = MaskAssemble(
                self.kernel.shape, self.kernel.detector_image
            )
            if self.kernel.qmap is not None:
                self.kernel.mask_kernel.update_qmap(self.kernel.qmap)
            logger.info("Initialized mask kernel after successful Q-map generation")

        if qmap is not None and "q" in qmap:
            q_min = np.nanmin(qmap["q"])
            q_max = np.nanmax(qmap["q"])
            self.status_bar.showMessage(
                f"Q-map generated: {q_min:.4f} - {q_max:.4f} Å⁻¹"
            )
            logger.info(f"Generated Q-map: q range [{q_min:.4f}, {q_max:.4f}] Å⁻¹")

            # Update info label to show successful state
            if self._detector_image is not None:
                shape = self._detector_image.shape
                bcx = new_metadata["bcx"]
                bcy = new_metadata["bcy"]
                self.info_label.setText(
                    f"Image: {shape[1]} x {shape[0]} pixels\n"
                    f"Beam center: ({bcx:.1f}, {bcy:.1f})"
                )
        else:
            self.status_bar.showMessage("Q-map generation failed")

    def _on_geometry_changed(self) -> None:
        """Handle geometry parameter change."""
        # Mark that geometry has been modified
        # Q-map will be regenerated on demand

    def _on_compute_partition(self) -> None:
        """Compute partition from current Q-map and binning parameters."""
        if self.kernel is None or not self.kernel.is_ready():
            self.status_bar.showMessage("Load data first")
            return

        if self.kernel.qmap is None:
            self.status_bar.showMessage("Generate Q-map first")
            return

        # Get binning parameters from spinboxes
        dq_num = self.spin_dq_num.value()
        sq_num = self.spin_sq_num.value()
        dp_num = self.spin_dp_num.value()
        sp_num = self.spin_sp_num.value()

        # Compute partition
        partition = self.kernel.compute_partition(
            mode="q-phi",
            dq_num=dq_num,
            sq_num=sq_num,
            dp_num=dp_num,
            sp_num=sp_num,
        )

        if partition is not None:
            dq_count = (
                partition.get("dynamic_roi_map", np.array([])).max()
                if "dynamic_roi_map" in partition
                else 0
            )
            self.status_bar.showMessage(
                f"Partition computed: {dq_count} dynamic Q-bins"
            )
            logger.info(
                f"Computed partition: {dq_count} dynamic Q-bins, {sq_num} static Q-bins"
            )
        else:
            self.status_bar.showMessage("Partition computation failed")

    def _on_toggle_partition(self) -> None:
        """Toggle partition overlay visibility."""
        if self.kernel is None or self.kernel.new_partition is None:
            self.status_bar.showMessage("Compute partition first")
            self.btn_toggle_partition.setChecked(False)
            return

        self._show_partition_overlay = not self._show_partition_overlay
        if self._show_partition_overlay:
            self._show_mask_overlay = False
            self._show_qmap_overlay = False

        # Sync button states
        self.btn_toggle_partition.setChecked(self._show_partition_overlay)
        self.btn_toggle_mask.setChecked(False)
        self.action_toggle_mask.setChecked(False)
        self.btn_toggle_qmap.setChecked(False)
        self.action_toggle_qmap.setChecked(False)

        # Update button text
        text = "Hide Partition" if self._show_partition_overlay else "Show Partition"
        self.btn_toggle_partition.setText(text)
        self.btn_toggle_mask.setText("Show Mask")
        self.action_toggle_mask.setText("Show Mask")
        self.btn_toggle_qmap.setText("Show Q-Map")
        self.action_toggle_qmap.setText("Show Q-Map")

        self._refresh_display()

    def _update_geometry_spinboxes(self) -> None:
        """Update geometry spinboxes from metadata."""
        if self._metadata is None:
            return

        # Block signals during update
        self.spin_bcx.blockSignals(True)
        self.spin_bcy.blockSignals(True)
        self.spin_det_dist.blockSignals(True)
        self.spin_pix_dim.blockSignals(True)
        self.spin_energy.blockSignals(True)

        try:
            bcx = self._metadata.get("bcx")
            if bcx is not None:
                self.spin_bcx.setValue(float(bcx))

            bcy = self._metadata.get("bcy")
            if bcy is not None:
                self.spin_bcy.setValue(float(bcy))

            det_dist = self._metadata.get("det_dist")
            if det_dist is not None:
                self.spin_det_dist.setValue(float(det_dist))

            pix_dim = self._metadata.get("pix_dim")
            if pix_dim is not None:
                self.spin_pix_dim.setValue(float(pix_dim))

            energy = self._metadata.get("energy")
            if energy is not None:
                self.spin_energy.setValue(float(energy))
        finally:
            self.spin_bcx.blockSignals(False)
            self.spin_bcy.blockSignals(False)
            self.spin_det_dist.blockSignals(False)
            self.spin_pix_dim.blockSignals(False)
            self.spin_energy.blockSignals(False)

    def _refresh_mask_display(self) -> None:
        """Refresh display with optional mask overlay (legacy method)."""
        self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh display with optional overlays (mask, Q-map, or partition)."""
        if self._detector_image is None:
            return

        # Base image (log scale)
        log_image = np.log10(self._detector_image + 1)

        if (
            self._show_partition_overlay
            and self.kernel is not None
            and self.kernel.new_partition is not None
        ):
            # Show partition overlay
            display_image = self._create_partition_display(self.kernel.new_partition)
            self.image_view.setImage(display_image, autoLevels=True, autoRange=False)
        elif (
            self._show_qmap_overlay
            and self.kernel is not None
            and self.kernel.qmap is not None
        ):
            # Show Q-map overlay
            display_image = self._create_qmap_display(log_image, self.kernel.qmap)
            self.image_view.setImage(display_image, autoLevels=True, autoRange=False)
        elif (
            self._show_mask_overlay
            and self.kernel is not None
            and self.kernel.mask is not None
        ):
            # Create image with mask overlay
            # Masked pixels (False in mask) shown darker
            display_image = self._create_masked_display(log_image, self.kernel.mask)
            self.image_view.setImage(display_image, autoLevels=False, autoRange=False)
        else:
            self.image_view.setImage(log_image, autoLevels=True, autoRange=False)

    def _create_qmap_display(self, image: NDArray, qmap: dict[str, NDArray]) -> NDArray:
        """Create display image with Q-map overlay.

        Args:
            image: Base image (2D)
            qmap: Q-map dictionary with 'q' key

        Returns:
            Q-map visualization
        """
        if "q" not in qmap:
            return image

        q_array = qmap["q"]

        # Return Q-map values directly for visualization
        # Apply mask if available
        if self.kernel is not None and self.kernel.mask is not None:
            display = q_array.copy()
            display[~self.kernel.mask] = np.nan
            return display

        return q_array

    def _create_partition_display(self, partition: dict) -> NDArray:
        """Create display image with partition overlay.

        Args:
            partition: Partition dictionary from kernel

        Returns:
            Partition visualization (color-coded by Q-bin)
        """
        # Use dynamic ROI map for visualization
        if "dynamic_roi_map" in partition:
            roi_map = partition["dynamic_roi_map"].astype(np.float64)
        elif "static_roi_map" in partition:
            roi_map = partition["static_roi_map"].astype(np.float64)
        else:
            return (
                np.zeros(self._detector_image.shape)
                if self._detector_image is not None
                else np.array([])
            )

        # Apply mask if available
        if self.kernel is not None and self.kernel.mask is not None:
            display = roi_map.copy()
            display[~self.kernel.mask] = np.nan
            return display

        return roi_map

    def _create_masked_display(self, image: NDArray, mask: NDArray) -> NDArray:
        """Create display image with mask overlay.

        Args:
            image: Base image (2D)
            mask: Boolean mask array (True = keep, False = masked)

        Returns:
            Image with mask overlay visualization
        """
        # Normalize image to 0-1 range
        vmin, vmax = np.nanmin(image), np.nanmax(image)
        if vmax > vmin:
            normalized = (image - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(image)

        # Create display - masked pixels shown darker/red-tinted
        display = normalized.copy()

        # Darken masked pixels
        masked_pixels = ~mask
        display[masked_pixels] = display[masked_pixels] * 0.3

        return display

    def load_from_viewer(
        self,
        detector_image: NDArray[np.float64] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Load data from XPCS Viewer.

        Args:
            detector_image: 2D detector image array, or None for empty canvas
            metadata: Geometry metadata dict containing:
                - bcx, bcy: Beam center in pixels
                - det_dist: Sample-to-detector distance in mm
                - pix_dim: Pixel dimension in mm
                - energy: X-ray energy in keV
                - shape: Detector shape (height, width)
        """
        if detector_image is None:
            logger.info("No detector image provided, starting with empty canvas")
            self.info_label.setText(
                "No data loaded.\n\n"
                "Open an HDF5 file in XPCS Viewer first,\n"
                "then click the Mask Editor tab again."
            )
            self.status_bar.showMessage("No data - load file in XPCS Viewer first")
            return

        self._detector_image = detector_image
        self._metadata = metadata or {}

        # Initialize kernel with image view handle
        if self.kernel is None:
            self.kernel = SimpleMaskKernel(
                pg_hdl=self.image_view, infobar=self.status_bar
            )

        try:
            success = self.kernel.read_data(detector_image, self._metadata)
        except ValueError as e:
            # Handle missing geometry parameters gracefully
            logger.warning(f"Failed to initialize Q-map: {e}")

            # Store image without Q-map for now
            self.kernel.detector_image = detector_image.copy()
            self.kernel.shape = detector_image.shape
            self.kernel.metadata = self._metadata.copy() if self._metadata else {}
            self.kernel.metadata["shape"] = self.kernel.shape
            self.kernel.mask = np.ones(self.kernel.shape, dtype=bool)

            # Display the image anyway
            self._display_image(detector_image)

            # Show info about missing parameters
            shape = detector_image.shape
            self.info_label.setText(
                f"Image: {shape[1]} x {shape[0]} pixels\n\n"
                f"⚠️ Missing geometry parameters.\n"
                f"Set values in Geometry panel,\n"
                f"then click 'Generate Q-Map'."
            )
            self.status_bar.showMessage(
                "Loaded image - set geometry parameters manually"
            )

            # Populate spinboxes with any available values
            self._update_geometry_spinboxes()

            logger.info(
                f"Loaded detector image {shape} without Q-map (missing geometry)"
            )
            return

        if success:
            logger.info(f"Loaded detector image with shape {detector_image.shape}")
            self._unsaved_changes = False

            # Display the image
            self._display_image(detector_image)

            # Update info label
            shape = detector_image.shape
            bcx = self._metadata.get("bcx", "N/A")
            bcy = self._metadata.get("bcy", "N/A")
            self.info_label.setText(
                f"Image: {shape[1]} x {shape[0]} pixels\nBeam center: ({bcx}, {bcy})"
            )
            self.status_bar.showMessage(f"Loaded {shape[1]}x{shape[0]} detector image")

            # Populate geometry spinboxes from metadata
            self._update_geometry_spinboxes()

    def _display_image(self, image: NDArray[np.float64]) -> None:
        """Display detector image in the ImageView.

        Args:
            image: 2D detector image array
        """
        # Use log scale for better visualization of scattering data
        # Add small offset to avoid log(0)
        log_image = np.log10(image + 1)

        self.image_view.setImage(
            log_image,
            autoLevels=True,
            autoRange=True,
        )

        # Set aspect ratio to 1:1 (square pixels)
        self.image_view.getView().setAspectLocked(True)

    def refresh_display(self) -> None:
        """Refresh the image display with current mask overlay."""
        self._refresh_mask_display()

    def export_mask_to_viewer(self) -> None:
        """Emit current mask via mask_exported signal."""
        if self.kernel is not None and self.kernel.mask is not None:
            self.mask_exported.emit(self.kernel.mask)
            logger.info("Mask exported to viewer")
            self.status_bar.showMessage("Mask exported to XPCS Viewer")

    def export_partition_to_viewer(self) -> None:
        """Emit current partition via qmap_exported signal."""
        if self.kernel is not None and self.kernel.new_partition is not None:
            self.qmap_exported.emit(self.kernel.new_partition)
            logger.info("Partition exported to viewer")
            self.status_bar.showMessage("Partition exported to XPCS Viewer")

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved mask modifications.

        Returns:
            True if there are unsaved changes
        """
        return self._unsaved_changes

    def mark_unsaved(self) -> None:
        """Mark that there are unsaved changes."""
        self._unsaved_changes = True
        if not self.windowTitle().endswith("*"):
            self.setWindowTitle(self.windowTitle() + " *")

    def mark_saved(self) -> None:
        """Mark that changes have been saved."""
        self._unsaved_changes = False
        title = self.windowTitle()
        if title.endswith(" *"):
            self.setWindowTitle(title[:-2])

    def closeEvent(self, event) -> None:
        """Handle window close with unsaved changes prompt.

        Args:
            event: Close event
        """
        # Skip confirmation dialog in test/headless mode to prevent blocking
        import os

        is_test_mode = os.environ.get("QT_QPA_PLATFORM") == "offscreen"

        if self._unsaved_changes and not is_test_mode:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved mask changes.\n\nClose without saving?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Clear parent reference to allow garbage collection
        if self.parent_viewer is not None:
            self.parent_viewer._simplemask_window = None

        event.accept()
