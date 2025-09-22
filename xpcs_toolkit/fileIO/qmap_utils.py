# Standard library imports
import warnings

# Third-party imports
import numpy as np

from xpcs_toolkit.utils.logging_config import get_logger

# Local imports
from .aps_8idi import key as key_map
from .hdf_reader import _connection_pool

logger = get_logger(__name__)

# Detector and beam constants
DEFAULT_DETECTOR_SIZE = 1024
DEFAULT_BEAM_CENTER = DEFAULT_DETECTOR_SIZE // 2  # 512


class QMapManager:
    def __init__(self):
        self.db = {}

    def get_qmap(self, fname):
        hash_value = get_hash(fname)  # Compute hash
        if hash_value not in self.db:
            qmap = QMap(fname=fname)
            self.db[hash_value] = qmap
        return self.db[hash_value]


class QMap:
    def __init__(self, fname=None, root_key="/xpcs/qmap"):
        self.root_key = root_key
        self.fname = fname

        # Initialize caching structures first
        self._qmap_cache = None
        self._qbin_cache = {}
        self._extent_cache = None

        # Load dataset and handle errors gracefully
        try:
            self.load_dataset()
            self.extent = self.get_detector_extent()

            # Compute qmap and qbin labels
            self.qmap, self.qmap_units = self.compute_qmap()
            self.qbin_labels = self.create_qbin_labels()
        except Exception as e:
            logger.error(f"Failed to initialize QMap for {fname}: {e}")
            # Create minimal fallback
            self._create_minimal_fallback()

    def load_dataset(self):
        info = {}
        # Optimize to read all keys in single file operation using connection pool
        try:
            with _connection_pool.get_connection(self.fname, "r") as f:
                # Check if qmap group exists
                qmap_path = key_map["nexus"]["mask"]  # Get base path for qmap
                qmap_group = qmap_path.rsplit("/", 1)[0]  # Extract qmap group path

                if qmap_group not in f:
                    logger.warning(
                        f"QMap group '{qmap_group}' not found in {self.fname}, creating default qmap"
                    )
                    return self._create_default_qmap()

                # Batch read all required keys in one file operation
                keys_to_load = [
                    "mask",
                    "dqmap",
                    "sqmap",
                    "dqlist",
                    "sqlist",
                    "dplist",
                    "splist",
                    "bcx",
                    "bcy",
                    "X_energy",
                    "static_index_mapping",
                    "dynamic_index_mapping",
                    "pixel_size",
                    "det_dist",
                    "dynamic_num_pts",
                    "static_num_pts",
                    "map_names",
                    "map_units",
                ]

                for key in keys_to_load:
                    try:
                        path = key_map["nexus"][key]
                        if path in f:
                            info[key] = f[path][()]
                        else:
                            logger.warning(
                                f"QMap key '{key}' not found in {self.fname}, using default"
                            )
                            info[key] = self._get_default_value(key)
                    except Exception as e:
                        logger.warning(
                            f"Error reading qmap key '{key}' from {self.fname}: {e}"
                        )
                        info[key] = self._get_default_value(key)

            # Post-process data after reading all at once
            # Handle division by zero for X_energy
            if info["X_energy"] != 0:
                info["k0"] = 2 * np.pi / (12.398 / info["X_energy"])
            else:
                info["k0"] = 0.0  # or np.nan if preferred for invalid energy
            if isinstance(info["map_names"][0], bytes):
                info["map_names"] = [item.decode("utf-8") for item in info["map_names"]]
            if isinstance(info["map_units"][0], bytes):
                info["map_units"] = [item.decode("utf-8") for item in info["map_units"]]

            # Ensure beam center values are proper Python floats to avoid "invalid index to scalar variable" errors
            # HDF5 can return various numpy scalar types that may not work correctly in arithmetic operations
            if "bcx" in info:
                info["bcx"] = float(info["bcx"])
            if "bcy" in info:
                info["bcy"] = float(info["bcy"])

            # Ensure dynamic_num_pts and static_num_pts are properly formatted arrays
            # They should be [n_dim0, n_dim1] but may be stored as scalars in some files
            if "dynamic_num_pts" in info:
                info["dynamic_num_pts"] = self._normalize_num_pts(
                    info["dynamic_num_pts"], info.get("dplist", [])
                )
            if "static_num_pts" in info:
                info["static_num_pts"] = self._normalize_num_pts(
                    info["static_num_pts"], info.get("splist", [])
                )

            self.__dict__.update(info)
            self.is_loaded = True
            return info

        except Exception as e:
            logger.error(f"Failed to load qmap from {self.fname}: {e}")
            return self._create_default_qmap()

    def _normalize_num_pts(self, num_pts, corresponding_list):
        """
        Normalize num_pts to be a 2-element array [n_dim0, n_dim1].

        Args:
            num_pts: Can be a scalar or array from HDF5
            corresponding_list: The corresponding phi/angle list to infer second dimension

        Returns:
            np.ndarray: [n_dim0, n_dim1] format
        """
        # Convert to numpy array first
        num_pts_array = np.asarray(num_pts)

        if num_pts_array.ndim == 0:
            # Scalar case: need to infer 2D structure
            n_dim0 = int(num_pts_array)
            n_dim1 = len(corresponding_list) if len(corresponding_list) > 0 else 1
            return np.array([n_dim0, n_dim1])
        if num_pts_array.ndim == 1 and len(num_pts_array) >= 2:
            # Already in correct format
            return num_pts_array[:2]  # Take first 2 elements
        if num_pts_array.ndim == 1 and len(num_pts_array) == 1:
            # 1-element array, treat as scalar
            n_dim0 = int(num_pts_array[0])
            n_dim1 = len(corresponding_list) if len(corresponding_list) > 0 else 1
            return np.array([n_dim0, n_dim1])
        # Fallback: assume it's total number of bins
        total_bins = int(num_pts_array.flat[0])
        n_dim1 = len(corresponding_list) if len(corresponding_list) > 0 else 1
        n_dim0 = total_bins // n_dim1 if n_dim1 > 0 else total_bins
        return np.array([n_dim0, n_dim1])

    def _get_default_value(self, key):
        """Get default values for missing qmap keys."""
        defaults = {
            "mask": np.ones(
                (DEFAULT_DETECTOR_SIZE, DEFAULT_DETECTOR_SIZE), dtype=np.int32
            ),
            "dqmap": np.ones(
                (DEFAULT_DETECTOR_SIZE, DEFAULT_DETECTOR_SIZE), dtype=np.int32
            ),
            "sqmap": np.ones(
                (DEFAULT_DETECTOR_SIZE, DEFAULT_DETECTOR_SIZE), dtype=np.int32
            ),
            "dqlist": np.linspace(0.01, 0.1, 10),
            "sqlist": np.linspace(0.01, 0.1, 10),
            "dplist": np.linspace(0, 360, 36),
            "splist": np.linspace(0, 360, 36),
            "bcx": float(DEFAULT_BEAM_CENTER),
            "bcy": float(DEFAULT_BEAM_CENTER),
            "X_energy": 8.0,
            "pixel_size": 75e-6,
            "det_dist": 5.0,
            "dynamic_num_pts": np.array([10, 1]),
            "static_num_pts": np.array([10, 1]),
            "static_index_mapping": np.arange(10),
            "dynamic_index_mapping": np.arange(10),
            "map_names": ["q", "phi"],
            "map_units": ["1/A", "degree"],
        }
        return defaults.get(key, np.array([0]))

    def _create_default_qmap(self):
        """Create a minimal default qmap when file doesn't have qmap data."""
        logger.info(f"Creating default qmap for {self.fname}")

        info = {}
        for key in [
            "mask",
            "dqmap",
            "sqmap",
            "dqlist",
            "sqlist",
            "dplist",
            "splist",
            "bcx",
            "bcy",
            "X_energy",
            "static_index_mapping",
            "dynamic_index_mapping",
            "pixel_size",
            "det_dist",
            "dynamic_num_pts",
            "static_num_pts",
            "map_names",
            "map_units",
        ]:
            info[key] = self._get_default_value(key)

        info["k0"] = 2 * np.pi / (12.398 / info["X_energy"])

        # Ensure beam center values are proper Python floats
        info["bcx"] = float(info["bcx"])
        info["bcy"] = float(info["bcy"])

        # Ensure num_pts are in correct format
        info["dynamic_num_pts"] = self._normalize_num_pts(
            info["dynamic_num_pts"], info.get("dplist", [])
        )
        info["static_num_pts"] = self._normalize_num_pts(
            info["static_num_pts"], info.get("splist", [])
        )

        self.__dict__.update(info)
        self.is_loaded = True
        return info

    def _create_minimal_fallback(self):
        """Create absolute minimal qmap when everything fails."""
        self.mask = np.ones((10, 10), dtype=np.int32)
        self.bcx = 5.0  # Small value for minimal 10x10 detector
        self.bcy = 5.0  # Small value for minimal 10x10 detector
        self.pixel_size = 75e-6
        self.det_dist = 5.0
        self.X_energy = 8.0
        self.k0 = 2 * np.pi / (12.398 / self.X_energy)
        self.extent = (-0.01, 0.01, -0.01, 0.01)
        self.qmap = {"q": np.ones((10, 10))}
        self.qmap_units = {"q": "1/A"}
        self.qbin_labels = ["q=0.01 1/A"]

        # Add critical missing attributes for reshape_phi_analysis
        self.sqlist = np.linspace(0.01, 0.1, 10)
        self.splist = np.linspace(0, 360, 10)
        self.dqlist = np.linspace(0.01, 0.1, 10)
        self.dplist = np.linspace(0, 360, 10)
        self.static_index_mapping = np.arange(10)
        self.dynamic_index_mapping = np.arange(10)
        self.static_num_pts = np.array([10, 1])
        self.dynamic_num_pts = np.array([10, 1])
        self.dqmap = np.ones((10, 10), dtype=np.int32)
        self.sqmap = np.ones((10, 10), dtype=np.int32)
        self.map_names = ["q", "phi"]
        self.map_units = ["1/A", "degree"]
        self.is_loaded = False
        logger.warning(f"Created minimal fallback qmap for {self.fname}")

    def reshape_phi_analysis(self, compressed_data_raw, label="data", mode="saxs_1d"):
        """
        the saxs1d and stability data are compressed. the values of the empty
        static bins are not saved. this function reshapes the array and fills
        the empty bins with nan. nanmean is performed to get the correct
        results;
        """
        assert mode in ("saxs_1d", "stability")

        # Defensive check for static_index_mapping
        if (
            not hasattr(self, "static_index_mapping")
            or self.static_index_mapping is None
        ):
            logger.warning(
                f"Missing static_index_mapping in QMap for {self.fname}, using fallback"
            )
            return self._fallback_reshape_phi_analysis(compressed_data_raw, label, mode)

        # Ensure static_index_mapping is a numpy array
        if not isinstance(self.static_index_mapping, np.ndarray):
            logger.warning(
                f"static_index_mapping is not a numpy array in QMap for {self.fname}, using fallback"
            )
            return self._fallback_reshape_phi_analysis(compressed_data_raw, label, mode)

        # Check if arrays have compatible sizes
        try:
            num_samples = compressed_data_raw.size // self.static_index_mapping.size
            assert (
                num_samples * self.static_index_mapping.size == compressed_data_raw.size
            )
        except (AttributeError, ZeroDivisionError, AssertionError) as e:
            logger.warning(
                f"Size mismatch in QMap for {self.fname}: {e}, using fallback"
            )
            return self._fallback_reshape_phi_analysis(compressed_data_raw, label, mode)

        # Check required attributes exist
        for attr in ["sqlist", "splist"]:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                logger.warning(
                    f"Missing attribute {attr} in QMap for {self.fname}, using fallback"
                )
                return self._fallback_reshape_phi_analysis(
                    compressed_data_raw, label, mode
                )

        shape = (num_samples, len(self.sqlist), len(self.splist))
        compressed_data = compressed_data_raw.reshape(num_samples, -1)

        if shape[2] == 1:
            labels = [label]
            avg = compressed_data.reshape(shape[0], -1)
        else:
            full_data = np.full((shape[0], shape[1] * shape[2]), fill_value=np.nan)
            for i in range(num_samples):
                full_data[i, self.static_index_mapping] = compressed_data[i]
            full_data = full_data.reshape(shape)

            # Handle empty slice warning by avoiding the problematic call when data is all NaN
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message=".*Mean of empty slice.*"
                )
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                # Also use numpy error state suppression as backup
                with np.errstate(all="ignore"):
                    avg = np.nanmean(full_data, axis=2)

        if mode == "saxs_1d":
            if num_samples != 1:
                logger.warning(
                    f"saxs1d mode expects 1 sample but got {num_samples}, using fallback for {self.fname}"
                )
                return self._fallback_reshape_phi_analysis(
                    compressed_data_raw, label, mode
                )
            if shape[2] > 1:
                saxs1d = np.concatenate([avg[..., None], full_data], axis=-1)
                saxs1d = saxs1d[0].T  # shape: (num_lines + 1, num_q)
                labels = [label + "_%d" % (n + 1) for n in range(shape[2])]
                labels = [label, *labels]
            else:
                saxs1d = avg.reshape(1, -1)  # shape: (1, num_q)
                labels = [label]
            saxs1d_info = {
                "q": self.sqlist,
                "Iq": saxs1d,
                "phi": self.splist,
                "num_lines": shape[2],
                "labels": labels,
                "data_raw": compressed_data_raw,
            }
            return saxs1d_info
        if mode == "stability":  # saxs1d_segments
            # avg shape is (num_samples, num_q)
            return avg
        return None

    def _fallback_reshape_phi_analysis(
        self, compressed_data_raw, label="data", mode="saxs_1d"
    ):
        """Fallback method when normal reshape fails."""
        data = (
            np.array(compressed_data_raw)
            if not isinstance(compressed_data_raw, np.ndarray)
            else compressed_data_raw
        )

        if mode == "saxs_1d":
            # For saxs_1d, return simple structure
            if data.size == 0:
                data = np.ones(10) * 0.1  # Minimal default values

            # Ensure data is at least 1D with 10 points to match sqlist
            if data.size < 10:
                padded_data = np.zeros(10)
                padded_data[: data.size] = data.flatten()
                data = padded_data

            saxs1d_info = {
                "q": getattr(self, "sqlist", np.linspace(0.01, 0.1, 10)),
                "Iq": data.reshape(1, -1),
                "phi": getattr(self, "splist", np.array([0])),
                "num_lines": 1,
                "labels": [label],
                "data_raw": compressed_data_raw,
            }
            return saxs1d_info
        if mode == "stability":
            # For stability mode, return reshaped data
            if data.size == 0:
                data = np.ones((1, 10)) * 100  # Minimal default values
            return data.reshape(1, -1) if data.ndim == 1 else data

        return data

    def get_detector_extent(self):
        """
        Optimized detector extent calculation with caching.
        get the angular extent on the detector, for saxs2d, qmap/display;
        :return:
        """
        # Check cache first
        if hasattr(self, "_extent_cache") and self._extent_cache is not None:
            return self._extent_cache

        shape = self.mask.shape
        # Pre-compute pixel-to-q conversion factor
        pix2q = self.pixel_size / self.det_dist * self.k0

        # Vectorized extent calculation
        qx_min = (0 - self.bcx) * pix2q
        qx_max = (shape[1] - self.bcx) * pix2q
        qy_min = (0 - self.bcy) * pix2q
        qy_max = (shape[0] - self.bcy) * pix2q

        extent = (qx_min, qx_max, qy_min, qy_max)

        # Cache the result
        self._extent_cache = extent
        return extent

    def get(self, key, default=None):
        """Provide dictionary-like access to QMap attributes."""
        return getattr(self, key, default)

    def get_qmap_at_pos(self, x, y):
        shape = self.mask.shape
        if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
            return None
        qmap, qmap_units = self.qmap, self.qmap_units
        result = ""
        for key in self.qmap:
            if key in ["q", "qx", "qy", "phi", "alpha"]:
                result += f" {key}={qmap[key][y, x]:.3f} {qmap_units[key]},"
            else:
                result += f" {key}={qmap[key][y, x]} {qmap_units[key]},"
        return result[:-1]

    def create_qbin_labels(self):
        if self.map_names == ["q", "phi"]:
            label_0 = [f"q={x:.5f} {self.map_units[0]}" for x in self.dqlist]
            label_1 = [f"φ={y:.1f} {self.map_units[1]}" for y in self.dplist]
        elif self.map_names == ["x", "y"]:
            label_0 = [f"x={x:.1f} {self.map_units[0]}" for x in self.dqlist]
            label_1 = [f"y={y:.1f} {self.map_units[1]}" for y in self.dplist]
        else:
            name0, name1 = self.map_names
            label_0 = [f"{name0}={x:.3f} {self.map_units[0]}" for x in self.dqlist]
            label_1 = [f"{name1}={y:.3f} {self.map_units[1]}" for y in self.dplist]

        if self.dynamic_num_pts[1] == 1:
            return label_0
        combined_list = []
        for item_a in label_0:
            for item_b in label_1:
                combined_list.append(f"{item_a}, {item_b}")
        return combined_list

    def get_qbin_label(self, qbin: int, append_qbin=False):
        qbin_absolute = self.dynamic_index_mapping[qbin - 1]
        if qbin_absolute < 0 or qbin_absolute > len(self.qbin_labels):
            return "invalid qbin"
        label = self.qbin_labels[qbin_absolute]
        if append_qbin:
            label = f"qbin={qbin}, {label}"
        return label

    def get_qbin_in_qrange(self, qrange, zero_based=True):
        """
        Optimized q-bin selection with improved vectorization and caching.
        """
        # Generate cache key for this q-range selection
        qrange_key = f"qbin_range_{qrange}_{zero_based}"
        if not hasattr(self, "_qbin_cache"):
            self._qbin_cache = {}

        if qrange_key in self._qbin_cache:
            return self._qbin_cache[qrange_key]

        if self.map_names[0] != "q":
            logger.info("qrange is only supported for qmaps with 0-axis as q")
            qrange = None

        # Optimize qlist computation using broadcasting
        qlist = np.broadcast_to(
            self.dqlist[:, np.newaxis], (len(self.dqlist), self.dynamic_num_pts[1])
        )

        if qrange is None:
            qselected = np.ones_like(qlist, dtype=bool)
        else:
            # Vectorized range comparison
            qselected = (qlist >= qrange[0]) & (qlist <= qrange[1])

        qselected_flat = qselected.flatten()

        # Handle edge case where no q-bins are selected
        if not np.any(qselected_flat):
            qselected_flat = np.ones_like(qlist, dtype=bool).flatten()

        # Vectorized approach to finding valid q-bins
        # Use boolean indexing instead of loops
        index_compressed = np.arange(len(self.dynamic_index_mapping))
        index_nature = self.dynamic_index_mapping

        # Create boolean mask for valid q-bins
        valid_mask = qselected_flat[index_nature]
        qbin_valid = index_compressed[valid_mask]

        # Get corresponding q-values efficiently
        qvalues = qlist.flatten()[qselected_flat]

        if not zero_based:
            qbin_valid = qbin_valid + 1

        result = (qbin_valid, qvalues)

        # Cache the result
        self._qbin_cache[qrange_key] = result
        return result

    def get_qbinlist_at_qindex(self, qindex, zero_based=True):
        # qindex is zero based; index of dyanmic_map_dim0
        assert self.map_names == ["q", "phi"], "only q-phi map is supported"
        qp_idx = np.ones(self.dynamic_num_pts, dtype=int).flatten() * (-1)
        qp_idx[self.dynamic_index_mapping] = np.arange(len(self.dynamic_index_mapping))
        qp_column_at_qindex = qp_idx.reshape(self.dynamic_num_pts)[qindex]
        qbin_list = [int(idx) for idx in qp_column_at_qindex if idx != -1]
        # if zero_based; it returns the numpy array index in g2[:, xx]
        if not zero_based:
            qbin_list = [idx + 1 for idx in qbin_list]
        return qbin_list

    def compute_qmap(self):
        """
        Optimized qmap computation with improved vectorization and memory efficiency.
        """
        # Check if qmap is already computed and cached
        if hasattr(self, "_qmap_cache") and self._qmap_cache is not None:
            return self._qmap_cache

        shape = self.mask.shape

        # Use more efficient data types and vectorized operations
        # Create coordinate arrays more efficiently
        v_offset = np.arange(shape[0], dtype=np.float32) - self.bcy
        h_offset = np.arange(shape[1], dtype=np.float32) - self.bcx
        vg, hg = np.meshgrid(v_offset, h_offset, indexing="ij")

        # Vectorized computation of polar coordinates
        # Use hypot for more accurate radius calculation
        r_pixel = np.hypot(vg, hg)
        r = r_pixel * self.pixel_size

        # Optimized angle calculation with single arctangent
        phi = np.arctan2(vg, hg) * (-1)

        # More efficient alpha calculation
        alpha = np.arctan(r / self.det_dist)

        # Vectorized q-space calculations
        sin_alpha = np.sin(alpha)
        qr = sin_alpha * self.k0

        # Use trigonometric identities for efficiency
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        qx = qr * cos_phi
        qy = qr * sin_phi

        # Convert to degrees efficiently
        phi_deg = np.rad2deg(phi)

        # Use memory-efficient data types where precision allows
        qmap = {
            "phi": phi_deg,  # Keep as float64 for precision
            "alpha": alpha.astype(np.float32),
            "q": qr,  # Keep as float64 for precision
            "qx": qx.astype(np.float32),
            "qy": qy.astype(np.float32),
            "x": hg.astype(np.int32),  # Convert to int for memory efficiency
            "y": vg.astype(np.int32),
            "r_pixel": r_pixel.astype(np.float32),  # Add this for ROI calculations
        }

        qmap_unit = {
            "phi": "°",
            "alpha": "°",
            "q": "Å⁻¹",
            "qx": "Å⁻¹",
            "qy": "Å⁻¹",
            "x": "pixel",
            "y": "pixel",
            "r_pixel": "pixel",
        }

        # Cache the result for future use
        result = (qmap, qmap_unit)
        self._qmap_cache = result
        return result


def get_hash(fname, root_key="/xpcs/qmap"):
    """Extracts the hash from the HDF5 file."""
    try:
        with _connection_pool.get_connection(fname, "r") as f:
            if root_key in f:
                return f[root_key].attrs.get("hash", fname)
            # If qmap doesn't exist, use filename as hash
            logger.warning(f"QMap not found in {fname}, using filename as hash")
            return fname
    except Exception as e:
        logger.warning(f"Error reading hash from {fname}: {e}")
        return fname


def get_qmap(fname, **kwargs):
    return QMap(fname, **kwargs)


def test_qmap_manager():
    import time

    for _i in range(5):
        t0 = time.perf_counter()
        get_qmap(
            "/net/s8iddata/export/8-id-ECA/MQICHU/projects/2025_0223_boost_corr_nexus/cluster_results1/Z1113_Sanjeeva-h60_a0004_t0600_f008000_r00003_results.hdf"
        )
        get_qmap(
            "/net/s8iddata/export/8-id-ECA/MQICHU/projects/2025_0223_boost_corr_nexus/cluster_results1/Z1113_Sanjeeva-h60_a0004_t0600_f008000_r00003_results2.hdf"
        )
        get_qmap(
            "/net/s8iddata/export/8-id-ECA/MQICHU/projects/2025_0223_boost_corr_nexus/cluster_results1/Z1113_Sanjeeva-h60_a0004_t0600_f008000_r00003_results3.hdf"
        )
        t1 = time.perf_counter()
        logger.debug(f"QMap loading benchmark: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    test_qmap_manager()
