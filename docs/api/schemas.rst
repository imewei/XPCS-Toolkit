Schemas
=======

Type-safe dataclass schemas with built-in validation for all shared data
structures in XPCS Viewer.

.. currentmodule:: xpcsviewer.schemas

Overview
--------

The schemas module provides frozen dataclasses with validation at construction
time, ensuring data integrity at I/O boundaries. Each schema validates shape,
dtype, NaN presence, and physical constraints.

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from xpcsviewer.schemas import QMapSchema, GeometryMetadata

   # Create a validated Q-map
   shape = (512, 512)
   qmap = QMapSchema(
       sqmap=np.random.rand(*shape).astype(np.float64),
       dqmap=np.zeros(shape, dtype=np.float64),
       phis=np.random.rand(*shape).astype(np.float64),
       sqmap_unit="nm^-1",
       dqmap_unit="nm^-1",
       phis_unit="rad",
   )

   # Convert from legacy dictionary
   qmap = QMapSchema.from_dict(legacy_dict)

   # Convert back for backward compatibility
   d = qmap.to_dict()

Schema Classes
--------------

QMapSchema
~~~~~~~~~~

.. autoclass:: QMapSchema
   :members:
   :no-index:

   **Validation Rules:**

   - All arrays must be 2D with identical shapes
   - Arrays must be ``float64`` dtype
   - No NaN values allowed in Q-map arrays
   - Unit strings must be ``'nm^-1'`` or ``'A^-1'`` for Q, ``'rad'`` or ``'deg'`` for phi
   - Mask values (if provided) must be 0 or 1

GeometryMetadata
~~~~~~~~~~~~~~~~

.. autoclass:: GeometryMetadata
   :members:
   :no-index:

   **Validation Rules:**

   - ``det_dist``, ``lambda_``, ``pix_dim`` must be positive
   - ``shape`` must be a 2-tuple of positive integers
   - Beam center is warned (not rejected) if outside detector bounds

G2Data
~~~~~~

.. autoclass:: G2Data
   :members:
   :no-index:

   **Validation Rules:**

   - ``g2`` and ``g2_err`` must have identical 2D shapes ``(n_delay, n_q)``
   - All arrays must be ``float64`` dtype
   - ``delay_times`` must be strictly monotonically increasing
   - ``g2_err`` values must be non-negative
   - ``q_values`` length must match number of Q-bins

PartitionSchema
~~~~~~~~~~~~~~~

.. autoclass:: PartitionSchema
   :members:
   :no-index:

   **Validation Rules:**

   - ``partition_map`` must be 2D integer array
   - ``num_pts`` must be positive
   - ``val_list`` and ``num_list`` lengths must equal ``num_pts``
   - ``method`` (if provided) must be ``'linear'`` or ``'log'``

MaskSchema
~~~~~~~~~~

.. autoclass:: MaskSchema
   :members:
   :no-index:

   **Validation Rules:**

   - ``mask`` must be 2D integer array with values 0 or 1
   - Shape is warned (not rejected) if different from metadata shape

Immutability
------------

All schemas are frozen dataclasses. Array fields are defensively copied at
construction to prevent external mutation. List fields (e.g., ``q_values``)
are converted to tuples internally.

.. code-block:: python

   # This raises dataclasses.FrozenInstanceError:
   qmap.sqmap = new_array

   # Use to_dict() / from_dict() to create modified copies:
   d = qmap.to_dict()
   d["sqmap"] = modified_array
   new_qmap = QMapSchema.from_dict(d)

Validators
----------

Validation helper functions used by schema ``__post_init__`` methods.

.. automodule:: xpcsviewer.schemas.validators
   :members:
   :no-index:

See Also
--------

- :doc:`io` - HDF5 facade that returns validated schema objects
- :doc:`backends` - Backend array conversion at I/O boundaries
