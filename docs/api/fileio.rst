File I/O
========

HDF5 data reading and Q-space mapping utilities for XPCS datasets.

.. currentmodule:: xpcsviewer.fileIO

HDF5 Reader
-----------

Optimized HDF5 file reading with connection pooling and batch operations.
Supports both synchronous and asynchronous reading patterns.

.. automodule:: xpcsviewer.fileIO.hdf_reader
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Enhanced HDF5 Reader
--------------------

Advanced HDF5 reader with additional caching and performance optimizations.
Built on top of the base HDF5 reader with extended functionality.

.. automodule:: xpcsviewer.fileIO.hdf_reader_enhanced
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Q-space Mapping
---------------

Detector geometry calculations and Q-space coordinate transformations.
Essential for converting pixel coordinates to reciprocal space.

.. automodule:: xpcsviewer.fileIO.qmap_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

APS 8-IDI Beamline Support
--------------------------

Beamline-specific data structure definitions and format handlers.
Supports both "nexus" and legacy data formats from APS-8IDI.

.. automodule:: xpcsviewer.fileIO.aps_8idi
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
