File I/O
========

The file I/O subsystem handles reading XPCS data from HDF5 files and provides
utilities for Q-space mapping and detector geometry calculations.

.. currentmodule:: xpcs_toolkit.fileIO

HDF5 Reader
-----------

.. automodule:: xpcs_toolkit.fileIO.hdf_reader
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: get
.. autofunction:: get_analysis_type
.. autofunction:: get_file_info
.. autofunction:: batch_read_fields

HDF5 Connection Pool
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: HDF5ConnectionPool
   :members:
   :undoc-members:
   :show-inheritance:

Enhanced HDF5 Reader
--------------------

.. automodule:: xpcs_toolkit.fileIO.hdf_reader_enhanced
   :members:
   :undoc-members:
   :show-inheritance:

Q-space Mapping
---------------

.. automodule:: xpcs_toolkit.fileIO.qmap_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autofunction:: get_qmap
.. autofunction:: get_pixel_mask
.. autofunction:: get_q_val_data

APS 8-IDI Beamline Support
--------------------------

.. automodule:: xpcs_toolkit.fileIO.aps_8idi
   :members:
   :undoc-members:
   :show-inheritance:

File Type Utilities
-------------------

.. automodule:: xpcs_toolkit.fileIO.ftype_utils
   :members:
   :undoc-members:
   :show-inheritance:
