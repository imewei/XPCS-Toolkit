File I/O
========

HDF5 data reading and Q-space mapping utilities for XPCS datasets.

.. note::
   For complete API documentation of all file I/O modules, see :doc:`../xpcs_toolkit.fileIO`.

.. currentmodule:: xpcs_toolkit.fileIO

HDF5 Reader
-----------

Optimized HDF5 file reading with connection pooling and batch operations.
Supports both synchronous and asynchronous reading patterns.

See :mod:`xpcs_toolkit.fileIO.hdf_reader` for complete API documentation.

Enhanced HDF5 Reader
--------------------

Advanced HDF5 reader with additional caching and performance optimizations.
Built on top of the base HDF5 reader with extended functionality.

See :mod:`xpcs_toolkit.fileIO.hdf_reader_enhanced` for complete API documentation.

Q-space Mapping
---------------

Detector geometry calculations and Q-space coordinate transformations.
Essential for converting pixel coordinates to reciprocal space.

See :mod:`xpcs_toolkit.fileIO.qmap_utils` for complete API documentation.

APS 8-IDI Beamline Support
--------------------------

Beamline-specific data structure definitions and format handlers.
Supports both "nexus" and legacy data formats from APS-8IDI.

See :mod:`xpcs_toolkit.fileIO.aps_8idi` for complete API documentation.

File Type Utilities
-------------------

File format detection and validation utilities for XPCS data files.

See :mod:`xpcs_toolkit.fileIO.ftype_utils` for complete API documentation.
