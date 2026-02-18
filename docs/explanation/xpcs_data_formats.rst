XPCS Data Formats and Conventions
==================================

This page explains the data formats and file conventions used by XPCS Viewer
for X-ray Photon Correlation Spectroscopy data. Understanding these formats
is important for interoperability with beamline data pipelines and third-party
analysis tools.

What is XPCS Data?
-------------------

X-ray Photon Correlation Spectroscopy (XPCS) measures temporal fluctuations
of coherent X-ray speckle patterns scattered from a sample. The raw data
consists of a time series of 2D detector images, from which correlation
functions are computed to extract sample dynamics.

The primary data products of an XPCS experiment are:

- **G2 correlation functions**: :math:`g_2(\tau)` curves for each Q-bin,
  encoding the sample dynamics (see :doc:`g2_theory`).
- **Q-maps**: Pixel-to-Q mappings derived from detector geometry
  (see :doc:`mask_qmap_internals`).
- **Two-time correlation matrices**: :math:`C(t_1, t_2)` for detecting
  non-equilibrium dynamics (see :doc:`twotime_analysis`).
- **Masks**: Boolean arrays identifying valid detector pixels.
- **Metadata**: Experimental geometry, beam parameters, and acquisition
  settings.

HDF5 File Structure
--------------------

XPCS Viewer stores and reads data in HDF5 files. The standard hierarchy
follows conventions from the APS 8-ID-I beamline, with extensions for
mask and partition data from the SimpleMask module.

.. code-block:: text

   /
   +-- xpcs/                      # Main XPCS data group
   |   +-- qmap/                  # Q-space mapping
   |   |   +-- sqmap              # float64, (H, W) -- static Q magnitude
   |   |   +-- dqmap              # float64, (H, W) -- dynamic Q magnitude
   |   |   +-- phis               # float64, (H, W) -- azimuthal angle
   |   |   +-- mask               # int32, (H, W)   -- pixel mask (0/1)
   |   |   +-- partition_map      # int32, (H, W)   -- Q-bin index per pixel
   |   |
   |   +-- g2/                    # G2 correlation results
   |   |   +-- g2                 # float64, (n_delay, n_q)
   |   |   +-- g2_err             # float64, (n_delay, n_q)
   |   |   +-- delay_times        # float64, (n_delay,)
   |   |   +-- q_values           # float64, (n_q,)
   |   |
   |   +-- metadata/              # Geometry and experimental parameters
   |       +-- bcx                # float -- beam center X (column, pixels)
   |       +-- bcy                # float -- beam center Y (row, pixels)
   |       +-- det_dist           # float -- sample-to-detector distance (mm)
   |       +-- lambda_            # float -- X-ray wavelength (Angstrom)
   |       +-- pix_dim            # float -- pixel size (mm)
   |       +-- shape              # (int, int) -- detector (height, width)
   |
   +-- simplemask/                # SimpleMask-generated data
   |   +-- mask/                  # Mask with metadata and version
   |   +-- partition/             # Q-bin partition with geometry
   |
   +-- exchange/                  # Two-time correlation data
       +-- C2T_all                # float64, (n_frames, n_frames) -- C(t1, t2)

Beamline Key Mapping
^^^^^^^^^^^^^^^^^^^^^

Different beamlines store XPCS results under different HDF5 paths. XPCS Viewer
uses a **key map** (``fileIO/aps_8idi.py``) to translate logical data names
to beamline-specific paths. For example, the G2 correlation data might reside
at ``/xpcs/g2`` at one beamline and ``/exchange/norm-0-g2`` at another.

This abstraction allows XPCS Viewer to read data from multiple sources without
modifying the analysis modules.

Schema Validation
------------------

Raw HDF5 data is untyped beyond basic array shape and dtype. To enforce
correctness, XPCS Viewer uses **schema validators** -- frozen dataclasses
that validate data at I/O boundaries.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Schema
     - Validates
     - Key Constraints
   * - ``QMapSchema``
     - Q-map arrays and units
     - Matching shapes, float64 dtype, no NaN, valid unit strings
   * - ``GeometryMetadata``
     - Detector geometry
     - Positive distances and wavelength, beam center within shape
   * - ``G2Data``
     - Correlation data
     - Shape consistency, float64, non-negative errors, monotonic delay times
   * - ``PartitionSchema``
     - Q-bin partition
     - Integer partition map, matching list lengths, non-negative counts
   * - ``MaskSchema``
     - Pixel mask
     - 2D integer array, values only 0 or 1

Schemas are constructed at the point of data ingestion (the I/O boundary),
so invalid data is rejected immediately with a clear error message rather
than causing subtle failures downstream.

Unit Conventions
-----------------

XPCS Viewer follows these unit conventions:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Quantity
     - Unit
     - Notes
   * - Q magnitude
     - nm\ :sup:`-1` (preferred) or A\ :sup:`-1`
     - Internal default is ``nm^-1``; legacy files may use ``A^-1``
   * - Azimuthal angle
     - radians or degrees
     - Internal computation uses radians; display may use degrees
   * - Delay times
     - seconds
     - Logarithmically spaced in most experiments
   * - Detector distance
     - mm
     - Sample-to-detector distance
   * - Pixel size
     - mm
     - Physical size of one detector pixel
   * - Wavelength
     - Angstrom
     - Converted from energy via :math:`\lambda = 12.398 / E_{\text{keV}}`
   * - X-ray energy
     - keV
     - Input parameter for Q-map computation

Array Conventions
------------------

- **Mask values**: ``0`` = masked (invalid), ``1`` = valid pixel.
- **Array dtype**: float64 for all scientific data; int32 for masks and
  partition maps.
- **Array orientation**: ``(height, width)`` -- row-major (C order), with
  ``(0, 0)`` at the top-left corner of the detector image.
- **Beam center**: ``bcx`` is the column (X) index, ``bcy`` is the row (Y)
  index, both in pixel coordinates (0-indexed).

Connection Pooling and the HDF5 Facade
----------------------------------------

HDF5 files are accessed through the ``HDF5Facade`` class, which combines
schema validation with connection pooling. The connection pool caches open
file handles keyed by ``(file_path, mode)`` to avoid repeated open/close
overhead during interactive analysis where the same file is read many times.

The facade provides a unified interface for all data types:

- ``read_qmap()`` returns a validated ``QMapSchema``
- ``read_g2_data()`` returns a validated ``G2Data``
- ``read_geometry_metadata()`` returns a validated ``GeometryMetadata``
- ``write_mask()`` and ``write_partition()`` accept schema objects and
  write with compression and version metadata

This design ensures that all HDF5 access goes through a single, validated
code path rather than being scattered across modules.
