I/O Facade
==========

High-level HDF5 I/O with schema validation, connection pooling, and
consistent error handling.

.. currentmodule:: xpcsviewer.io

Overview
--------

The ``xpcsviewer.io`` module provides the ``HDF5Facade`` class, a unified
interface for reading and writing validated XPCS data. It sits above the
lower-level ``xpcsviewer.fileIO`` module and returns schema-validated objects
from :mod:`xpcsviewer.schemas`.

Quick Start
-----------

.. code-block:: python

   from xpcsviewer.io import HDF5Facade

   facade = HDF5Facade()

   # Read with schema validation
   qmap = facade.read_qmap("data.hdf5")
   print(f"Q-map shape: {qmap.sqmap.shape}")
   print(f"Units: {qmap.sqmap_unit}")

   # Read G2 correlation data
   g2_data = facade.read_g2_data("data.hdf5")
   print(f"Delay points: {g2_data.g2.shape[0]}, Q-bins: {g2_data.g2.shape[1]}")

   # Read geometry metadata
   geometry = facade.read_geometry_metadata("data.hdf5")
   print(f"Beam center: ({geometry.bcx}, {geometry.bcy})")

HDF5Facade
----------

.. autoclass:: HDF5Facade
   :members:
   :no-index:

HDF5ValidationError
-------------------

.. autoclass:: HDF5ValidationError
   :no-index:

   Raised when HDF5 data fails schema validation during read operations.

   .. code-block:: python

      from xpcsviewer.io import HDF5Facade, HDF5ValidationError

      facade = HDF5Facade()
      try:
          qmap = facade.read_qmap("corrupted.hdf5")
      except HDF5ValidationError as e:
          print(f"Validation failed: {e}")

Writing Data
------------

The facade provides write methods for masks and partitions:

.. code-block:: python

   import numpy as np
   from xpcsviewer.schemas import MaskSchema, GeometryMetadata

   metadata = GeometryMetadata(
       bcx=256.0, bcy=256.0,
       det_dist=5000.0, lambda_=1.0,
       pix_dim=0.075, shape=(512, 512),
   )

   mask = MaskSchema(
       mask=np.ones((512, 512), dtype=np.int32),
       metadata=metadata,
       description="Clean mask",
   )

   facade = HDF5Facade()
   facade.write_mask("output.hdf5", mask)

Connection Pool
---------------

The facade uses an internal connection pool for HDF5 file handles, reducing
overhead for repeated reads from the same file.

.. code-block:: python

   # Check pool statistics
   stats = facade.get_pool_stats()
   print(f"Pool size: {stats}")

   # Clear the pool (e.g., before shutdown)
   facade.clear_pool()

See Also
--------

- :doc:`schemas` - Data structure schemas returned by the facade
- :doc:`fileio` - Lower-level HDF5 reading utilities
- :doc:`backends` - Backend-aware I/O adapters
