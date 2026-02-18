Mask Editor and Q-Map Tutorial
===============================

This tutorial shows how to use the SimpleMask module to create detector masks,
compute Q-maps from geometry metadata, and generate Q-space partitions for
XPCS analysis.

Overview
--------

The mask editor pipeline has three stages:

1. **Load detector data** -- Provide a 2D detector image and geometry metadata.
2. **Create a mask** -- Mark bad pixels, beamstop regions, and detector gaps.
3. **Compute partition** -- Bin the detector into Q-phi regions for correlation.

The kernel class :class:`~xpcsviewer.simplemask.simplemask_kernel.SimpleMaskKernel`
handles all computation. The GUI-level ``SimpleMaskWindow`` wraps it with
interactive drawing tools.

Step 1 -- Initialize the Kernel
---------------------------------

.. code-block:: python

   import numpy as np
   from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

   # Create kernel without GUI handles (headless mode)
   kernel = SimpleMaskKernel(pg_hdl=None, infobar=None)

Step 2 -- Load Detector Data
------------------------------

Provide a detector image and a metadata dictionary describing the geometry:

.. code-block:: python

   # Example: 256x256 detector with synthetic data
   detector_image = np.random.poisson(lam=100, size=(256, 256)).astype(np.float64)

   metadata = {
       "bcx": 128.0,        # Beam center X (column), pixels
       "bcy": 128.0,        # Beam center Y (row), pixels
       "det_dist": 5000.0,  # Detector distance, mm
       "pix_dim": 0.075,    # Pixel size, mm
       "energy": 10.0,      # X-ray energy, keV
       "stype": "Transmission",
   }

   success = kernel.read_data(detector_image, metadata)
   print(f"Data loaded: {success}")
   print(f"Detector shape: {kernel.shape}")
   print(f"Q-map keys: {list(kernel.qmap.keys())}")

The :meth:`~xpcsviewer.simplemask.simplemask_kernel.SimpleMaskKernel.read_data`
method automatically computes the Q-map from the geometry and initializes a
mask of all ones (all pixels valid).

Step 3 -- Inspect the Q-Map
-----------------------------

The ``qmap`` dictionary contains momentum-transfer and angle maps:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(14, 4))

   # Q magnitude map
   im0 = axes[0].imshow(kernel.qmap["q"], origin="lower")
   axes[0].set_title(f"Q ({kernel.qmap_unit['q']})")
   plt.colorbar(im0, ax=axes[0])

   # Azimuthal angle map
   im1 = axes[1].imshow(kernel.qmap["phi"], origin="lower")
   axes[1].set_title(f"phi ({kernel.qmap_unit['phi']})")
   plt.colorbar(im1, ax=axes[1])

   # Detector image
   im2 = axes[2].imshow(np.log1p(detector_image), origin="lower")
   axes[2].set_title("log(1 + I)")
   plt.colorbar(im2, ax=axes[2])

   plt.tight_layout()
   plt.show()

Step 4 -- Modify the Mask
---------------------------

The mask is a boolean array where ``True`` means valid (unmasked):

.. code-block:: python

   # Mask a rectangular beamstop region
   kernel.mask[120:136, 120:136] = False

   # Mask pixels with zero counts
   kernel.mask[detector_image == 0] = False

   # Mask hot pixels (intensity > threshold)
   hot_threshold = np.percentile(detector_image, 99.9)
   kernel.mask[detector_image > hot_threshold] = False

   valid_pixels = kernel.mask.sum()
   total_pixels = kernel.mask.size
   print(f"Valid pixels: {valid_pixels}/{total_pixels} "
         f"({100 * valid_pixels / total_pixels:.1f}%)")

Step 5 -- Save and Load Masks
-------------------------------

Masks are saved to HDF5 files for reuse:

.. code-block:: python

   # Save mask
   kernel.save_mask("my_mask.h5")

   # Load mask in a new kernel
   kernel2 = SimpleMaskKernel(pg_hdl=None, infobar=None)
   kernel2.read_data(detector_image, metadata)
   kernel2.load_mask("my_mask.h5", key="mask")

Step 6 -- Recompute Q-Map
---------------------------

If geometry parameters change, recompute the Q-map:

.. code-block:: python

   # Update beam center
   kernel.metadata["bcx"] = 130.0
   kernel.metadata["bcy"] = 126.0

   qmap, qmap_unit = kernel.compute_qmap()
   print(f"Recomputed Q range: [{qmap['q'].min():.4f}, {qmap['q'].max():.4f}] {qmap_unit['q']}")

Step 7 -- Compute a Q-Phi Partition
-------------------------------------

The partition divides the detector into Q-phi bins for correlation analysis.
There are two levels: **dynamic** (fewer bins, used for G2 correlation) and
**static** (finer bins, used for SAXS averaging).

.. code-block:: python

   partition = kernel.compute_partition(
       mode="q-phi",       # Partition in Q and phi
       dq_num=10,          # 10 dynamic Q bins
       sq_num=100,         # 100 static Q bins
       dp_num=1,           # 1 dynamic phi bin (full azimuthal average)
       sp_num=1,           # 1 static phi bin
       style="linear",     # Linear Q spacing ("logarithmic" also supported)
   )

   if partition is not None:
       print(f"Partition keys: {list(partition.keys())}")
       print(f"Dynamic ROI map shape: {partition['dynamic_roi_map'].shape}")
       print(f"Number of dynamic bins: {partition['dynamic_roi_map'].max()}")

       # Visualize the dynamic partition
       fig, ax = plt.subplots()
       roi = partition["dynamic_roi_map"].astype(float)
       roi[roi == 0] = np.nan  # Hide masked regions
       ax.imshow(roi, origin="lower", cmap="tab20")
       ax.set_title("Dynamic Q-Phi Partition")
       plt.tight_layout()
       plt.show()

Step 8 -- Validate with Schemas
---------------------------------

Use :class:`~xpcsviewer.schemas.QMapSchema` to validate Q-map data at
I/O boundaries:

.. code-block:: python

   from xpcsviewer.schemas import QMapSchema

   # Create a validated QMapSchema from the compute_qmap output
   qmap_schema = QMapSchema.from_compute_qmap(kernel.qmap, kernel.qmap_unit)

   print(f"Schema sqmap shape: {qmap_schema.sqmap.shape}")
   print(f"Schema unit: {qmap_schema.sqmap_unit}")

   # Convert back to dict for legacy interfaces
   qmap_dict = qmap_schema.to_dict()

.. note::

   The :class:`~xpcsviewer.schemas.QMapSchema` is a frozen dataclass. It
   validates shapes, dtypes, NaN values, and units at construction time.
   Arrays are defensively copied to prevent external mutation.

GUI Workflow
--------------

In the full GUI application, the mask editor is launched from the
**Mask Editor** tab. It provides:

- **Drawing tools**: Rectangle, Circle, Polygon, Line, Ellipse, Eraser
- **Undo/redo history** via the ``MaskAssemble`` class
- **Export signals**: ``mask_exported(np.ndarray)`` and ``qmap_exported(dict)``

The GUI calls the same :class:`~xpcsviewer.simplemask.simplemask_kernel.SimpleMaskKernel`
methods demonstrated above.

Next Steps
----------

- :doc:`getting_started` -- Load data and perform basic G2 analysis
- :doc:`fitting_guide` -- Fit G2 curves with NLSQ and Bayesian inference
- :doc:`cookbook` -- Batch processing and advanced patterns
