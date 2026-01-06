"""Core kernel for SimpleMask operations.

This module provides the SimpleMaskKernel class that handles all mask operations,
Q-map computation, partition generation, and ROI management.

Adapted from pySimpleMask for integration with XPCS Viewer.
"""

import logging
from typing import Any, Literal

import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from xpcsviewer.simplemask.area_mask import MaskAssemble
from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI, LineROI
from xpcsviewer.simplemask.qmap import compute_qmap
from xpcsviewer.simplemask.utils import (
    check_consistency,
    combine_partitions,
    generate_partition,
    hash_numpy_dict,
    optimize_integer_array,
)

pg.setConfigOptions(imageAxisOrder="row-major")

logger = logging.getLogger(__name__)

# Compression threshold for HDF5 datasets (elements)
HDF5_COMPRESSION_THRESHOLD = 1024

# Version for partition file metadata
__version__ = "1.0.0"


class SimpleMaskKernel:
    """Core kernel for SimpleMask operations.

    This class manages mask creation, editing, Q-map computation,
    and partition generation for XPCS analysis.

    Attributes:
        shape: Detector shape as (height, width)
        detector_image: 2D array of detector data
        mask: Current combined mask array
        qmap: Dictionary of Q-map arrays
        qmap_unit: Dictionary of Q-map units
        metadata: Geometry metadata dictionary
    """

    def __init__(
        self,
        pg_hdl: ImageViewROI | None = None,
        infobar: Any | None = None,
    ):
        """Initialize SimpleMask kernel.

        Args:
            pg_hdl: PyQtGraph ImageView handle for display
            infobar: Status bar widget for messages
        """
        self.detector_image: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None
        self.qmap: dict[str, np.ndarray] | None = None
        self.qmap_unit: dict[str, str] | None = None
        self.mask: np.ndarray | None = None
        self.mask_kernel: MaskAssemble | None = None
        self.new_partition: dict | None = None
        self.metadata: dict[str, Any] = {}

        self.hdl = pg_hdl
        self.infobar = infobar

        if self.hdl is not None:
            self.hdl.scene.sigMouseMoved.connect(self.show_location)

    def is_ready(self) -> bool:
        """Check if kernel has data loaded.

        Returns:
            True if detector image is loaded
        """
        return self.detector_image is not None

    def read_data(
        self,
        detector_image: np.ndarray,
        metadata: dict[str, Any],
    ) -> bool:
        """Load detector data and initialize mask.

        Args:
            detector_image: 2D array of detector counts
            metadata: Dictionary with geometry parameters:
                - bcx: Beam center X (column)
                - bcy: Beam center Y (row)
                - det_dist: Detector distance (mm)
                - pix_dim: Pixel size (mm)
                - energy: X-ray energy (keV)
                - shape: Detector shape (height, width)

        Returns:
            True if successful
        """
        self.detector_image = detector_image.copy()
        self.shape = detector_image.shape
        self.metadata = metadata.copy()
        self.metadata["shape"] = self.shape

        # Initialize mask
        self.mask = np.ones(self.shape, dtype=bool)

        # Compute Q-map
        stype = self.metadata.get("stype", "Transmission")
        self.qmap, self.qmap_unit = compute_qmap(stype, self.metadata)

        # Initialize mask assembly
        self.mask_kernel = MaskAssemble(self.shape, self.detector_image)
        self.mask_kernel.update_qmap(self.qmap)

        return True

    def compute_qmap(self) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        """Recompute Q-map from current metadata.

        Returns:
            Tuple of (qmap_dict, units_dict)
        """
        stype = self.metadata.get("stype", "Transmission")
        self.qmap, self.qmap_unit = compute_qmap(stype, self.metadata)
        if self.mask_kernel is not None:
            self.mask_kernel.update_qmap(self.qmap)
        return self.qmap, self.qmap_unit

    def mask_evaluate(self, target: str, **kwargs) -> str:
        """Evaluate a mask type without applying.

        Args:
            target: Mask type key
            **kwargs: Parameters for mask evaluation

        Returns:
            Description string from the mask worker
        """
        if self.mask_kernel is None:
            return "Mask kernel not initialized"
        return self.mask_kernel.evaluate(target, **kwargs)

    def mask_action(self, action: Literal["undo", "redo", "reset"] = "undo") -> None:
        """Execute undo/redo/reset on mask history.

        Args:
            action: One of "undo", "redo", or "reset"
        """
        if self.mask_kernel is None:
            return
        self.mask_kernel.redo_undo(action=action)
        self.mask_apply()

    def mask_apply(self, target: str | None = None) -> np.ndarray:
        """Apply a mask type and update current mask.

        Args:
            target: Mask type to apply, or None to get current mask

        Returns:
            Current combined mask array
        """
        if self.mask_kernel is None:
            return self.mask

        self.mask = self.mask_kernel.apply(target)
        return self.mask

    def save_mask(self, save_name: str) -> None:
        """Save mask to HDF5 file.

        Args:
            save_name: Output file path (.h5/.hdf5)
        """
        if self.mask is None:
            logger.warning("No mask to save")
            return

        mask = self.mask.astype(np.uint8)

        with h5py.File(save_name, "w") as hf:
            hf.create_dataset("mask", data=mask, compression="lzf")
            hf.attrs["shape"] = self.shape
            hf.attrs["version"] = __version__

    def load_mask(self, fname: str, key: str = "mask") -> bool:
        """Load mask from HDF5 file.

        Args:
            fname: Path to HDF5 file
            key: Dataset key for mask data

        Returns:
            True if successful
        """
        if self.mask_kernel is None:
            return False

        self.mask_kernel.evaluate("mask_file", fname=fname, key=key)
        self.mask_apply("mask_file")
        return True

    def save_partition(self, save_fname: str, root: str = "/qmap") -> None:
        """Save partition to HDF5 file.

        Args:
            save_fname: Output file path (.h5/.hdf5)
            root: HDF5 group path for partition data
        """
        if self.new_partition is None:
            logger.warning("No partition to save")
            return

        # Optimize integer arrays
        for key, val in self.new_partition.items():
            self.new_partition[key] = optimize_integer_array(val)

        hash_val = hash_numpy_dict(self.new_partition)
        logger.info(f"Hash value of the partition: {hash_val}")

        def optimize_save(group_handle, key, val):
            compression = (
                "lzf"
                if isinstance(val, np.ndarray) and val.size > HDF5_COMPRESSION_THRESHOLD
                else None
            )
            return group_handle.create_dataset(key, data=val, compression=compression)

        with h5py.File(save_fname, "w") as hf:
            if root in hf:
                del hf[root]
            group_handle = hf.create_group(root)

            for key, val in self.new_partition.items():
                dset = optimize_save(group_handle, key, val)
                if "_v_list_dim" in key:
                    dim = int(key[-1])
                    dset.attrs["unit"] = self.new_partition["map_units"][dim]
                    dset.attrs["name"] = self.new_partition["map_names"][dim]
                    dset.attrs["size"] = val.size

            group_handle.attrs["hash"] = hash_val
            group_handle.attrs["version"] = __version__

    def show_location(self, pos) -> None:
        """Display pixel coordinates and Q values at mouse position.

        Args:
            pos: Mouse scene position
        """
        if self.hdl is None or self.shape is None:
            return
        if not self.hdl.scene.itemsBoundingRect().contains(pos):
            return

        mouse_point = self.hdl.getView().mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())

        if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
            msg = f"x={col}, y={row}"
            if self.qmap is not None and "q" in self.qmap:
                q_val = self.qmap["q"][row, col]
                msg += f", q={q_val:.4f} Å⁻¹"
            if self.detector_image is not None:
                intensity = self.detector_image[row, col]
                msg += f", I={intensity:.1f}"
            if self.infobar is not None:
                self.infobar.clear()
                self.infobar.setText(msg)

    def show_saxs(
        self,
        cmap: str = "jet",
        log: bool = True,  # noqa: ARG002 - kept for API compatibility
        plot_center: bool = True,
    ) -> None:
        """Display detector image with optional beam center marker.

        Args:
            cmap: Matplotlib colormap name
            log: Whether to use log scale (unused, kept for API compatibility)
            plot_center: Whether to display beam center marker
        """
        if self.detector_image is None or self.hdl is None:
            return

        self.hdl.clear()
        center = self.get_center()

        # Create display data (original + mask overlay)
        data = self.detector_image.copy()
        self.hdl.setImage(data)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        if plot_center and center[0] is not None:
            t = pg.ScatterPlotItem()
            t.addPoints(x=[center[0]], y=[center[1]], symbol="+", size=15)
            self.hdl.add_item(t, label="center")

    def apply_drawing(self) -> np.ndarray:
        """Apply all pending drawing ROIs to create a mask.

        Returns:
            Mask array from drawing operations
        """
        if self.detector_image is None or self.hdl is None:
            return np.ones(self.shape, dtype=bool) if self.shape else np.array([])

        shape = self.shape
        ones = np.ones((shape[0] + 1, shape[1] + 1), dtype=bool)
        mask_e = np.zeros_like(ones, dtype=bool)
        mask_i = np.zeros_like(mask_e)

        for k, x in self.hdl.roi.items():
            if not k.startswith("roi_"):
                continue

            mask_temp = np.zeros_like(ones, dtype=bool)
            sl, _ = x.getArraySlice(self.detector_image, self.hdl.imageItem)
            y = x.getArrayRegion(ones, self.hdl.imageItem)

            nz_idx = np.nonzero(y)
            if len(nz_idx[0]) == 0:
                continue

            h_beg = np.min(nz_idx[1])
            h_end = np.max(nz_idx[1]) + 1
            v_beg = np.min(nz_idx[0])
            v_end = np.max(nz_idx[0]) + 1

            sl_v = slice(sl[0].start, sl[0].start + v_end - v_beg)
            sl_h = slice(sl[1].start, sl[1].start + h_end - h_beg)
            mask_temp[sl_v, sl_h] = y[v_beg:v_end, h_beg:h_end]

            if hasattr(x, "sl_mode"):
                if x.sl_mode == "exclusive":
                    mask_e[mask_temp] = 1
                elif x.sl_mode == "inclusive":
                    mask_i[mask_temp] = 1

        self.hdl.remove_rois(filter_str="roi_")

        if np.sum(mask_i) == 0:
            mask_i = 1

        mask_p = np.logical_not(mask_e) * mask_i
        mask_p = mask_p[:-1, :-1]

        return mask_p

    def add_drawing(  # noqa: PLR0912 - branches needed for different shape types
        self,
        sl_type: Literal[
            "Rectangle", "Circle", "Ellipse", "Polygon", "Line"
        ] = "Polygon",
        sl_mode: Literal["exclusive", "inclusive"] = "exclusive",
        num_edges: int | None = None,
        radius: float = 60,
        color: str = "r",
        width: int = 3,
        second_point: tuple[float, float] | None = None,
        label: str | None = None,
        movable: bool = True,
    ) -> pg.ROI | None:
        """Add a drawing ROI to the view.

        Args:
            sl_type: Shape type - Rectangle, Circle, Ellipse, Polygon, or Line
            sl_mode: "exclusive" masks pixels, "inclusive" preserves pixels
            num_edges: Number of edges for Polygon (random 6-10 if None)
            radius: Default radius for Circle
            color: Pen color
            width: Pen width
            second_point: End point for Line ROI
            label: ROI label (auto-generated if None)
            movable: Whether ROI can be moved

        Returns:
            Created ROI object, or None if creation failed
        """
        if self.hdl is None or self.shape is None:
            return None

        if label is not None and label in self.hdl.roi:
            self.hdl.remove_item(label)

        cen = self.get_center()
        if cen[0] is None or cen[0] < 0 or cen[1] < 0:
            cen = (self.shape[1] // 2, self.shape[0] // 2)
        elif cen[0] > self.shape[1] or cen[1] > self.shape[0]:
            logger.warning("Beam center out of range, using image center")
            cen = (self.shape[1] // 2, self.shape[0] // 2)

        if sl_mode == "inclusive":
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        handle_pen = pg.mkPen(color=color, width=width)

        kwargs = {
            "pen": pen,
            "removable": True,
            "hoverPen": pen,
            "handlePen": handle_pen,
            "movable": movable,
        }

        new_roi: pg.ROI | None = None

        if sl_type == "Ellipse":
            new_roi = pg.EllipseROI(cen, [60, 80], **kwargs)
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])

        elif sl_type == "Circle":
            if second_point is not None:
                radius = np.sqrt(
                    (second_point[1] - cen[1]) ** 2 + (second_point[0] - cen[0]) ** 2
                )
            new_roi = pg.CircleROI(
                pos=[cen[0] - radius, cen[1] - radius], radius=radius, **kwargs
            )

        elif sl_type == "Polygon":
            if num_edges is None:
                num_edges = np.random.randint(6, 11)
            offset = np.random.randint(0, 360)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + np.deg2rad(offset)
            x = radius * np.cos(theta) + cen[0]
            y = radius * np.sin(theta) + cen[1]
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, **kwargs)

        elif sl_type == "Rectangle":
            new_roi = pg.RectROI(cen, [30, 150], **kwargs)
            new_roi.addScaleHandle([0, 0], [1, 1])

        elif sl_type == "Line":
            if second_point is None:
                return None
            line_width = kwargs.pop("width", 1) if "width" in kwargs else 1
            new_roi = LineROI(cen, second_point, line_width, **kwargs)

        else:
            raise TypeError(f"ROI type not implemented: {sl_type}")

        if new_roi is not None:
            new_roi.sl_mode = sl_mode
            roi_key = self.hdl.add_item(new_roi, label)
            new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(roi_key))

        return new_roi

    def remove_roi(self, roi_key: str) -> None:
        """Remove an ROI by key.

        Args:
            roi_key: Label of the ROI to remove
        """
        if self.hdl is not None:
            self.hdl.remove_item(roi_key)

    def compute_partition(
        self,
        mode: str = "q-phi",
        dq_num: int = 10,
        sq_num: int = 100,
        dp_num: int = 36,
        sp_num: int = 360,
        style: str = "linear",
        phi_offset: float = 0.0,
        symmetry_fold: int = 1,
    ) -> dict | None:
        """Compute Q-space partition for XPCS analysis.

        Args:
            mode: Partition mode (e.g., "q-phi", "x-y")
            dq_num: Number of dynamic Q bins
            sq_num: Number of static Q bins
            dp_num: Number of dynamic phi bins
            sp_num: Number of static phi bins
            style: Binning style - "linear" or "logarithmic"
            phi_offset: Phi angle offset
            symmetry_fold: Symmetry fold for phi

        Returns:
            Partition dictionary or None if no data
        """
        if self.detector_image is None or self.qmap is None or self.mask is None:
            return None

        map_names = mode.split("-")
        logger.info(f"Computing partition with mode {mode}: map_names {map_names}")

        name0, name1 = map_names

        # Generate dynamic partition
        pack_dq = generate_partition(
            name0, self.mask, self.qmap[name0], dq_num, style=style
        )
        pack_dp = generate_partition(
            name1,
            self.mask,
            self.qmap[name1],
            dp_num,
            style=style,
            phi_offset=phi_offset,
            symmetry_fold=symmetry_fold,
        )
        dynamic_map = combine_partitions(pack_dq, pack_dp, prefix="dynamic")

        # Generate static partition
        pack_sq = generate_partition(
            name0, self.mask, self.qmap[name0], sq_num, style=style
        )
        pack_sp = generate_partition(
            name1,
            self.mask,
            self.qmap[name1],
            sp_num,
            style=style,
            phi_offset=phi_offset,
            symmetry_fold=symmetry_fold,
        )
        static_map = combine_partitions(pack_sq, pack_sp, prefix="static")

        # Check consistency
        flag_consistency = check_consistency(
            dynamic_map["dynamic_roi_map"], static_map["static_roi_map"], self.mask
        )
        logger.info(f"dqmap/sqmap consistency check: {flag_consistency}")

        center = self.get_center()
        partition = {
            "beam_center_x": center[0],
            "beam_center_y": center[1],
            "pixel_size": self.metadata.get("pix_dim", 0.075),
            "mask": self.mask,
            "energy": self.metadata.get("energy", 10.0),
            "detector_distance": self.metadata.get("det_dist", 5000.0),
            "map_names": list(map_names),
            "map_units": [self.qmap_unit[name0], self.qmap_unit[name1]],
            "source_file": self.metadata.get("source_file", ""),
        }
        partition.update(dynamic_map)
        partition.update(static_map)

        self.new_partition = partition
        return partition

    def update_parameters(self, new_metadata: dict[str, Any]) -> None:
        """Update geometry parameters and recompute Q-map.

        Args:
            new_metadata: Dictionary with updated geometry values
        """
        self.metadata.update(new_metadata)
        self.compute_qmap()

    def get_center(
        self, mode: Literal["xy", "vh"] = "xy"
    ) -> tuple[float | None, float | None]:
        """Get beam center coordinates.

        Args:
            mode: "xy" for (x, y) = (column, row), "vh" for (vertical, horizontal)

        Returns:
            Beam center coordinates
        """
        if not self.metadata:
            return (None, None)

        bcx = self.metadata.get("bcx")
        bcy = self.metadata.get("bcy")

        if bcx is None or bcy is None:
            return (None, None)

        if mode == "xy":
            return (bcx, bcy)
        if mode == "vh":
            return (bcy, bcx)
        return (bcx, bcy)
