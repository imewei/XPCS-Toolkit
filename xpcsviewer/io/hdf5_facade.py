"""Unified HDF5 I/O facade with schema validation.

This module provides a high-level facade for all HDF5 file operations in
the XPCS Viewer codebase, with built-in schema validation, connection pooling,
and error handling.

The facade consolidates scattered HDF5 access patterns into a single,
well-tested, and maintainable interface.

Public API:
    HDF5Facade: Main facade class for HDF5 operations
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import hdf5plugin  # noqa: F401 - enables compression filters
import numpy as np

from xpcsviewer.fileIO.hdf_reader import _connection_pool
from xpcsviewer.schemas import (
    G2Data,
    GeometryMetadata,
    MaskSchema,
    PartitionSchema,
    QMapSchema,
)
from xpcsviewer.utils.log_utils import log_timing
from xpcsviewer.utils.logging_config import get_logger

if TYPE_CHECKING:
    from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

logger = get_logger(__name__)


class HDF5ValidationError(Exception):
    """Raised when HDF5 data fails schema validation."""

    pass


class HDF5Facade:
    """Unified facade for HDF5 I/O operations with schema validation.

    This facade provides a single entry point for all HDF5 file operations,
    with built-in:
    - Schema validation using dataclasses
    - Connection pooling for performance
    - Consistent error handling
    - Logging and monitoring

    Attributes
    ----------
    pool : HDF5ConnectionPool
        Connection pool for managing HDF5 file handles
    validate : bool
        Whether to perform schema validation
    """

    def __init__(
        self,
        pool: HDF5ConnectionPool | None = None,
        validate: bool = True,
    ):
        """Initialize HDF5 facade.

        Parameters
        ----------
        pool : HDF5ConnectionPool, optional
            Connection pool to use. If None, uses global pool.
        validate : bool, optional
            Enable schema validation, by default True
        """
        self.pool = pool if pool is not None else _connection_pool
        self.validate = validate

    @log_timing(threshold_ms=500)
    def read_qmap(self, file_path: str | Path, group: str = "/xpcs/qmap") -> QMapSchema:
        """Read Q-map from HDF5 file with schema validation.

        Parameters
        ----------
        file_path : str or Path
            Path to HDF5 file
        group : str, optional
            HDF5 group containing Q-map data, by default "/xpcs/qmap"

        Returns
        -------
        QMapSchema
            Validated Q-map data

        Raises
        ------
        HDF5ValidationError
            If Q-map data fails schema validation
        FileNotFoundError
            If file does not exist
        KeyError
            If required datasets are missing
        """
        file_path = str(file_path)
        logger.debug(f"Reading Q-map from {file_path}:{group}")

        try:
            with self.pool.get_connection(file_path, "r") as f:
                if group not in f:
                    raise KeyError(f"Q-map group '{group}' not found in {file_path}")

                qmap_group = f[group]

                # Read required datasets
                sqmap = qmap_group["sqmap"][:]
                dqmap = qmap_group["dqmap"][:]
                phis = qmap_group["phis"][:]

                # Read units (with defaults for backward compatibility)
                sqmap_unit = qmap_group.get("sqmap_unit", "nm^-1")
                if isinstance(sqmap_unit, h5py.Dataset):
                    val = sqmap_unit[()]
                    sqmap_unit = val.decode() if isinstance(val, bytes) else str(val)
                elif isinstance(sqmap_unit, bytes):
                    sqmap_unit = sqmap_unit.decode()

                dqmap_unit = qmap_group.get("dqmap_unit", "nm^-1")
                if isinstance(dqmap_unit, h5py.Dataset):
                    val = dqmap_unit[()]
                    dqmap_unit = val.decode() if isinstance(val, bytes) else str(val)
                elif isinstance(dqmap_unit, bytes):
                    dqmap_unit = dqmap_unit.decode()

                phis_unit = qmap_group.get("phis_unit", "rad")
                if isinstance(phis_unit, h5py.Dataset):
                    val = phis_unit[()]
                    phis_unit = val.decode() if isinstance(val, bytes) else str(val)
                elif isinstance(phis_unit, bytes):
                    phis_unit = phis_unit.decode()

                # Read optional datasets
                mask = qmap_group["mask"][:] if "mask" in qmap_group else None
                partition_map = (
                    qmap_group["partition_map"][:]
                    if "partition_map" in qmap_group
                    else None
                )

            # Create validated schema
            if self.validate:
                qmap = QMapSchema(
                    sqmap=sqmap,
                    dqmap=dqmap,
                    phis=phis,
                    sqmap_unit=sqmap_unit,
                    dqmap_unit=dqmap_unit,
                    phis_unit=phis_unit,
                    mask=mask,
                    partition_map=partition_map,
                )
            else:
                # Skip validation - use dict
                qmap_data = {
                    "sqmap": sqmap,
                    "dqmap": dqmap,
                    "phis": phis,
                    "sqmap_unit": sqmap_unit,
                    "dqmap_unit": dqmap_unit,
                    "phis_unit": phis_unit,
                    "mask": mask,
                    "partition_map": partition_map,
                }
                # Still create schema, just without validation
                qmap = QMapSchema(**qmap_data)

            logger.debug(f"Successfully read Q-map from {file_path}")
            return qmap

        except Exception as e:
            if isinstance(e, (KeyError, ValueError)):
                raise HDF5ValidationError(
                    f"Failed to validate Q-map from {file_path}: {e}"
                ) from e
            raise

    @log_timing(threshold_ms=500)
    def write_mask(
        self,
        file_path: str | Path,
        mask_schema: MaskSchema,
        group: str = "/simplemask/mask",
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """Write mask to HDF5 file with versioning.

        Parameters
        ----------
        file_path : str or Path
            Path to HDF5 file
        mask_schema : MaskSchema
            Validated mask schema to write
        group : str, optional
            HDF5 group to write to, by default "/simplemask/mask"
        compression : str, optional
            Compression algorithm, by default "gzip"
        compression_opts : int, optional
            Compression level (1-9), by default 4

        Raises
        ------
        ValueError
            If mask_schema validation fails
        """
        file_path = str(file_path)
        logger.debug(f"Writing mask to {file_path}:{group}")

        try:
            with h5py.File(file_path, "a") as f:
                # Create group if it doesn't exist
                if group in f:
                    logger.warning(f"Overwriting existing mask at {group}")
                    del f[group]

                mask_group = f.create_group(group)

                # Write mask dataset
                mask_group.create_dataset(
                    "mask",
                    data=mask_schema.mask,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                # Write version as attribute
                mask_group.attrs["version"] = mask_schema.version

                # Write description if provided
                if mask_schema.description:
                    mask_group.attrs["description"] = mask_schema.description

                # Write metadata as subgroup
                metadata_group = mask_group.create_group("metadata")
                metadata_dict = mask_schema.metadata.to_dict()
                for key, value in metadata_dict.items():
                    if key == "shape":
                        metadata_group.attrs[key] = value
                    else:
                        metadata_group.attrs[key] = value

            logger.info(
                f"Successfully wrote mask to {file_path}:{group} "
                f"(version {mask_schema.version})"
            )

        except Exception as e:
            logger.error(f"Failed to write mask to {file_path}: {e}")
            raise

    @log_timing(threshold_ms=500)
    def write_partition(
        self,
        file_path: str | Path,
        partition_schema: PartitionSchema,
        group: str = "/simplemask/partition",
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """Write partition to HDF5 file with versioning.

        Parameters
        ----------
        file_path : str or Path
            Path to HDF5 file
        partition_schema : PartitionSchema
            Validated partition schema to write
        group : str, optional
            HDF5 group to write to, by default "/simplemask/partition"
        compression : str, optional
            Compression algorithm, by default "gzip"
        compression_opts : int, optional
            Compression level (1-9), by default 4

        Raises
        ------
        ValueError
            If partition_schema validation fails
        """
        file_path = str(file_path)
        logger.debug(f"Writing partition to {file_path}:{group}")

        try:
            with h5py.File(file_path, "a") as f:
                # Create group if it doesn't exist
                if group in f:
                    logger.warning(f"Overwriting existing partition at {group}")
                    del f[group]

                partition_group = f.create_group(group)

                # Write partition map
                partition_group.create_dataset(
                    "partition_map",
                    data=partition_schema.partition_map,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                # Write scalar attributes
                partition_group.attrs["num_pts"] = partition_schema.num_pts
                partition_group.attrs["version"] = partition_schema.version

                # Write lists as datasets
                partition_group.create_dataset(
                    "val_list",
                    data=np.array(partition_schema.val_list),
                )
                partition_group.create_dataset(
                    "num_list",
                    data=np.array(partition_schema.num_list, dtype=np.int32),
                )

                # Write method if provided
                if partition_schema.method:
                    partition_group.attrs["method"] = partition_schema.method

                # Write mask if provided
                if partition_schema.mask is not None:
                    partition_group.create_dataset(
                        "mask",
                        data=partition_schema.mask,
                        compression=compression,
                        compression_opts=compression_opts,
                    )

                # Write metadata as subgroup
                metadata_group = partition_group.create_group("metadata")
                metadata_dict = partition_schema.metadata.to_dict()
                for key, value in metadata_dict.items():
                    if key == "shape":
                        metadata_group.attrs[key] = value
                    else:
                        metadata_group.attrs[key] = value

            logger.info(
                f"Successfully wrote partition to {file_path}:{group} "
                f"(version {partition_schema.version}, {partition_schema.num_pts} bins)"
            )

        except Exception as e:
            logger.error(f"Failed to write partition to {file_path}: {e}")
            raise

    @log_timing(threshold_ms=500)
    def read_g2_data(
        self, file_path: str | Path, q_idx: int | None = None, group: str = "/xpcs/g2"
    ) -> G2Data:
        """Read G2 correlation data from HDF5 file with schema validation.

        Parameters
        ----------
        file_path : str or Path
            Path to HDF5 file
        q_idx : int, optional
            If provided, read only this Q-bin index. Otherwise read all.
        group : str, optional
            HDF5 group containing G2 data, by default "/xpcs/g2"

        Returns
        -------
        G2Data
            Validated G2 correlation data

        Raises
        ------
        HDF5ValidationError
            If G2 data fails schema validation
        KeyError
            If required datasets are missing
        """
        file_path = str(file_path)
        logger.debug(f"Reading G2 data from {file_path}:{group}")

        try:
            with self.pool.get_connection(file_path, "r") as f:
                if group not in f:
                    raise KeyError(f"G2 group '{group}' not found in {file_path}")

                g2_group = f[group]

                # Read G2 data
                if q_idx is not None:
                    g2 = g2_group["g2"][q_idx : q_idx + 1, :]
                    g2_err = g2_group["g2_err"][q_idx : q_idx + 1, :]
                else:
                    g2 = g2_group["g2"][:]
                    g2_err = g2_group["g2_err"][:]

                delay_times = g2_group["delay_times"][:]

                # Read Q values (may be stored differently in different file formats)
                if "q_values" in g2_group:
                    q_values = list(g2_group["q_values"][:])
                elif "q_list" in g2_group:
                    q_values = list(g2_group["q_list"][:])
                else:
                    # Fallback: create dummy Q values
                    logger.warning(
                        f"Q values not found in {file_path}, using dummy values"
                    )
                    q_values = list(range(g2.shape[0]))

                if q_idx is not None:
                    q_values = q_values[q_idx : q_idx + 1]

            # Create validated schema
            g2_data = G2Data(
                g2=g2,
                g2_err=g2_err,
                delay_times=delay_times,
                q_values=q_values,
            )

            logger.debug(
                f"Successfully read G2 data from {file_path} "
                f"({g2.shape[0]} Q-bins, {g2.shape[1]} delay points)"
            )
            return g2_data

        except Exception as e:
            if isinstance(e, (KeyError, ValueError)):
                raise HDF5ValidationError(
                    f"Failed to validate G2 data from {file_path}: {e}"
                ) from e
            raise

    @log_timing(threshold_ms=300)
    def read_geometry_metadata(
        self, file_path: str | Path, group: str = "/xpcs/metadata"
    ) -> GeometryMetadata:
        """Read geometry metadata from HDF5 file with schema validation.

        Parameters
        ----------
        file_path : str or Path
            Path to HDF5 file
        group : str, optional
            HDF5 group containing metadata, by default "/xpcs/metadata"

        Returns
        -------
        GeometryMetadata
            Validated geometry metadata

        Raises
        ------
        HDF5ValidationError
            If metadata fails schema validation
        KeyError
            If required fields are missing
        """
        file_path = str(file_path)
        logger.debug(f"Reading geometry metadata from {file_path}:{group}")

        try:
            with self.pool.get_connection(file_path, "r") as f:
                if group not in f:
                    raise KeyError(f"Metadata group '{group}' not found in {file_path}")

                metadata_group = f[group]

                # Read required fields
                bcx = float(metadata_group["bcx"][()])
                bcy = float(metadata_group["bcy"][()])
                det_dist = float(metadata_group["det_dist"][()])
                lambda_ = float(metadata_group["lambda_"][()])
                pix_dim = float(metadata_group["pix_dim"][()])

                # Read shape
                if "shape" in metadata_group.attrs:
                    shape = tuple(metadata_group.attrs["shape"])
                elif "height" in metadata_group and "width" in metadata_group:
                    shape = (
                        int(metadata_group["height"][()]),
                        int(metadata_group["width"][()]),
                    )
                else:
                    raise KeyError("Shape information not found in metadata")

                # Read optional fields
                det_rotation = None
                if "det_rotation" in metadata_group:
                    det_rotation = float(metadata_group["det_rotation"][()])

                incident_angle = None
                if "incident_angle" in metadata_group:
                    incident_angle = float(metadata_group["incident_angle"][()])

            # Create validated schema
            metadata = GeometryMetadata(
                bcx=bcx,
                bcy=bcy,
                det_dist=det_dist,
                lambda_=lambda_,
                pix_dim=pix_dim,
                shape=shape,
                det_rotation=det_rotation,
                incident_angle=incident_angle,
            )

            logger.debug(f"Successfully read geometry metadata from {file_path}")
            return metadata

        except Exception as e:
            if isinstance(e, (KeyError, ValueError)):
                raise HDF5ValidationError(
                    f"Failed to validate geometry metadata from {file_path}: {e}"
                ) from e
            raise

    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns
        -------
        dict
            Connection pool statistics including cache hits, pool size, etc.
        """
        return self.pool.get_pool_stats()

    def clear_pool(self) -> None:
        """Clear the connection pool.

        Use this to force close all pooled connections, for example
        after major changes or before application shutdown.
        """
        self.pool.clear_pool()
        logger.info("HDF5 connection pool cleared")
