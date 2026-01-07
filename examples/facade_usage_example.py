"""Example usage of HDF5 Facade and Schema infrastructure.

This script demonstrates how to use the new facade and schema components
for type-safe, validated HDF5 I/O operations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from xpcsviewer.backends import create_adapters, get_backend
from xpcsviewer.io import HDF5Facade
from xpcsviewer.schemas import GeometryMetadata, MaskSchema, PartitionSchema, QMapSchema


def example_schema_validation():
    """Demonstrate schema validation with type safety."""
    print("=== Schema Validation Example ===\n")

    # Create geometry metadata with validation
    metadata = GeometryMetadata(
        bcx=512.5,
        bcy=512.5,
        det_dist=5000.0,
        lambda_=1.54,
        pix_dim=0.075,
        shape=(1024, 1024),
    )
    print(f"Created metadata: {metadata.shape} detector")

    # Create Q-map schema with validation
    sqmap = np.random.rand(100, 100)
    dqmap = np.random.rand(100, 100)
    phis = np.random.rand(100, 100)

    qmap = QMapSchema(
        sqmap=sqmap,
        dqmap=dqmap,
        phis=phis,
        sqmap_unit="nm^-1",
        dqmap_unit="nm^-1",
        phis_unit="rad",
    )
    print(f"Created Q-map: {qmap.sqmap.shape} with unit {qmap.sqmap_unit}")

    # Demonstrate validation - this will raise ValueError
    try:
        invalid_qmap = QMapSchema(
            sqmap=sqmap,
            dqmap=dqmap[:50, :50],  # Wrong shape!
            phis=phis,
            sqmap_unit="nm^-1",
            dqmap_unit="nm^-1",
            phis_unit="rad",
        )
    except ValueError as e:
        print(f"✓ Validation caught shape mismatch: {e}")

    print()


def example_facade_io():
    """Demonstrate HDF5 facade for reading and writing."""
    print("=== HDF5 Facade I/O Example ===\n")

    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Initialize facade
        facade = HDF5Facade()

        # Create test data
        metadata = GeometryMetadata(
            bcx=512.5,
            bcy=512.5,
            det_dist=5000.0,
            lambda_=1.54,
            pix_dim=0.075,
            shape=(1024, 1024),
        )

        mask = MaskSchema(
            mask=np.random.randint(0, 2, size=(1024, 1024), dtype=np.int32),
            metadata=metadata,
            version="1.0.0",
            description="Example mask for demonstration",
        )

        # Write mask using facade
        print(f"Writing mask to {tmp_path}")
        facade.write_mask(tmp_path, mask)

        # Create partition data
        partition_map = np.random.randint(0, 50, size=(1024, 1024), dtype=np.int32)
        partition = PartitionSchema(
            partition_map=partition_map,
            num_pts=50,
            val_list=[0.01 * i for i in range(50)],
            num_list=[np.count_nonzero(partition_map == i) for i in range(50)],
            metadata=metadata,
            version="1.0.0",
            method="linear",
        )

        # Write partition using facade
        print(f"Writing partition to {tmp_path}")
        facade.write_partition(tmp_path, partition)

        # Verify data was written
        with h5py.File(tmp_path, "r") as f:
            print(f"\nHDF5 file structure:")
            print(f"  Groups: {list(f.keys())}")
            if "simplemask" in f:
                print(f"  SimpleMask groups: {list(f['simplemask'].keys())}")

        # Get connection pool stats
        stats = facade.get_pool_stats()
        print(f"\nConnection pool stats:")
        print(f"  Pool size: {stats['pool_size']}/{stats['max_pool_size']}")
        print(f"  Cache hit ratio: {stats['cache_hit_ratio']:.2%}")

    finally:
        # Cleanup
        tmp_path.unlink()
        print(f"\n✓ Cleaned up {tmp_path}")

    print()


def example_io_adapters():
    """Demonstrate I/O adapters for backend conversions."""
    print("=== I/O Adapters Example ===\n")

    # Get backend and create adapters
    backend = get_backend()
    print(f"Using backend: {backend.__class__.__name__}")

    pyqt_adapter, hdf5_adapter, mpl_adapter = create_adapters(
        backend, enable_monitoring=True
    )

    # Example: Convert data for PyQtGraph plotting
    x = backend.linspace(0, 10, 100)
    y = backend.sin(x)

    print(f"Backend arrays: {type(x).__name__}, {type(y).__name__}")

    # Convert for PyQtGraph
    x_pyqt = pyqt_adapter.to_pyqtgraph(x)
    y_pyqt = pyqt_adapter.to_pyqtgraph(y)
    print(f"PyQtGraph arrays: {type(x_pyqt).__name__}, {type(y_pyqt).__name__}")

    # Convert for HDF5
    data_hdf5 = hdf5_adapter.to_hdf5(y)
    print(f"HDF5 array: {type(data_hdf5).__name__}")

    # Convert for Matplotlib (can handle multiple arrays)
    x_mpl, y_mpl = mpl_adapter.to_matplotlib(x, y)
    print(f"Matplotlib arrays: {type(x_mpl).__name__}, {type(y_mpl).__name__}")

    # Get performance stats
    print(f"\nPerformance stats:")
    pyqt_stats = pyqt_adapter.get_stats()
    print(
        f"  PyQtGraph: {pyqt_stats['conversion_count']} conversions, "
        f"avg {pyqt_stats['average_conversion_time_ms']:.3f}ms"
    )

    hdf5_stats = hdf5_adapter.get_stats()
    print(
        f"  HDF5: {hdf5_stats['write_count']} writes, "
        f"avg {hdf5_stats['average_write_time_ms']:.3f}ms"
    )

    print()


def example_legacy_compatibility():
    """Demonstrate backward compatibility with legacy dict-based code."""
    print("=== Legacy Compatibility Example ===\n")

    # Create schema
    qmap = QMapSchema(
        sqmap=np.random.rand(50, 50),
        dqmap=np.random.rand(50, 50),
        phis=np.random.rand(50, 50),
        sqmap_unit="nm^-1",
        dqmap_unit="nm^-1",
        phis_unit="rad",
    )

    # Convert to dict for legacy code
    qmap_dict = qmap.to_dict()
    print(f"Converted to dict with keys: {list(qmap_dict.keys())}")

    # Legacy code can use the dict
    legacy_sqmap = qmap_dict["sqmap"]
    print(f"Legacy code accesses sqmap: shape={legacy_sqmap.shape}")

    # Convert back from dict to schema
    qmap_restored = QMapSchema.from_dict(qmap_dict)
    print(f"Restored schema: {qmap_restored.sqmap.shape}")
    print(f"Data preserved: {np.allclose(qmap.sqmap, qmap_restored.sqmap)}")

    print()


if __name__ == "__main__":
    print("XPCS Viewer Facade and Schema Infrastructure Examples")
    print("=" * 60)
    print()

    example_schema_validation()
    example_facade_io()
    example_io_adapters()
    example_legacy_compatibility()

    print("=" * 60)
    print("All examples completed successfully!")
