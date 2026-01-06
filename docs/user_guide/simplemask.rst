================
Mask Editor Guide
================

The Mask Editor (SimpleMask) provides tools for creating, editing, and managing detector masks
for XPCS analysis. It also supports Q-map generation and Q-binning (partition) for correlation analysis.

Overview
========

The Mask Editor is launched from the **Mask Editor** tab in XPCS Viewer. It opens a separate window
that allows you to:

* Create masks using drawing tools (Rectangle, Circle, Polygon, Line, Ellipse)
* Remove masked regions with the Eraser tool
* Save and load masks in HDF5 format
* Generate Q-maps from detector geometry
* Define Q-bins (partition) for XPCS correlation analysis
* Export masks and partitions back to XPCS Viewer

Launching the Mask Editor
=========================

1. Load an HDF5 file in XPCS Viewer
2. Click the **Mask Editor** tab (tab 11)
3. The SimpleMask window opens with the detector image

If no file is loaded, the Mask Editor opens with an empty canvas.

Drawing Tools
=============

The toolbar provides several drawing tools for creating mask regions:

.. list-table::
   :widths: 15 10 40
   :header-rows: 1

   * - Tool
     - Shortcut
     - Description
   * - Rectangle
     - ``R``
     - Draw rectangular mask regions
   * - Circle
     - ``C``
     - Draw circular mask regions
   * - Polygon
     - ``P``
     - Draw polygonal mask regions (click to add vertices)
   * - Line
     - ``L``
     - Draw line mask regions with adjustable width
   * - Ellipse
     - ``E``
     - Draw elliptical mask regions
   * - Eraser
     - ``X``
     - Remove (include) previously masked regions

Tool Modes
----------

* **Exclusive mode** (default): Drawn regions are excluded from analysis (masked out)
* **Inclusive mode** (Eraser): Drawn regions are included in analysis (unmask)

Creating Masks
==============

Basic Workflow
--------------

1. Select a drawing tool from the toolbar
2. Draw on the detector image to define mask regions
3. Click **Apply Drawing** to apply the ROIs to the mask
4. Toggle **Show Mask** to visualize masked regions (semi-transparent red overlay)

The mask shows excluded regions in red. Pixels in masked regions will be ignored during
XPCS correlation analysis.

Undo/Redo
---------

* **Undo** (``Ctrl+Z``): Revert the last mask change
* **Redo** (``Ctrl+Y``): Restore an undone change
* **Reset**: Clear all mask changes and return to initial state

Saving and Loading Masks
========================

Save Mask
---------

1. Click **File > Save Mask...** (``Ctrl+S``)
2. Choose a location and filename (HDF5 format)
3. The mask is saved with shape information and version metadata

Load Mask
---------

1. Click **File > Load Mask...** (``Ctrl+O``)
2. Select a mask file (HDF5 format)
3. The mask dimensions are validated against the current detector shape
4. If dimensions match, the mask is loaded and displayed

.. note::
   Loading a mask with different dimensions than the current detector will show
   an error dialog. Ensure the mask was created for the same detector geometry.

Geometry Parameters
===================

The right panel provides geometry parameter inputs for Q-map computation:

.. list-table::
   :widths: 20 15 40
   :header-rows: 1

   * - Parameter
     - Unit
     - Description
   * - Beam X
     - pixels
     - Beam center X coordinate (column)
   * - Beam Y
     - pixels
     - Beam center Y coordinate (row)
   * - Distance
     - mm
     - Sample-to-detector distance
   * - Pixel Size
     - mm
     - Pixel dimension
   * - Energy
     - keV
     - X-ray energy

These values are automatically populated from the loaded HDF5 file metadata when available.

Q-Map Generation
================

The Q-map shows the momentum transfer (Q) value at each detector pixel.

Generate Q-Map
--------------

1. Verify or adjust the geometry parameters
2. Click **Generate Q-Map**
3. Toggle **Show Q-Map** to display the Q-map overlay

The Q-map is color-coded from low Q (dark) to high Q (bright).

Q-Binning (Partition)
=====================

Partition divides the detector into Q-bins for correlation analysis.

Partition Parameters
--------------------

.. list-table::
   :widths: 20 15 40
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - Dynamic Q
     - 10
     - Number of dynamic Q-bins (for g2 analysis)
   * - Static Q
     - 100
     - Number of static Q-bins (finer resolution)
   * - Dynamic Phi
     - 36
     - Number of azimuthal bins for dynamic analysis
   * - Static Phi
     - 360
     - Number of azimuthal bins for static analysis

Compute Partition
-----------------

1. Set the partition parameters
2. Click **Compute Partition**
3. Toggle **Show Partition** to display the partition overlay

The partition overlay shows color-coded Q-bins.

Exporting to XPCS Viewer
========================

Export Mask
-----------

Click **Apply to Viewer** in the toolbar to send the current mask to XPCS Viewer.
The mask will be used for subsequent analysis operations.

Export Partition
----------------

Click **Export to Viewer** in the Partition panel to send the Q-map partition to XPCS Viewer.
This includes:

* Dynamic ROI map (for g2 correlation)
* Static ROI map (for static analysis)
* Beam center coordinates
* Mask array

Keyboard Shortcuts
==================

.. list-table::
   :widths: 20 40
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``Ctrl+S``
     - Save mask
   * - ``Ctrl+O``
     - Load mask
   * - ``Ctrl+Z``
     - Undo
   * - ``Ctrl+Y``
     - Redo
   * - ``Ctrl+W``
     - Close window
   * - ``R``
     - Rectangle tool
   * - ``C``
     - Circle tool
   * - ``P``
     - Polygon tool
   * - ``L``
     - Line tool
   * - ``E``
     - Ellipse tool
   * - ``X``
     - Eraser tool

Troubleshooting
===============

Mask dimension mismatch
-----------------------

If you see "Mask dimensions do not match" when loading a mask:

* The mask was created for a different detector size
* Recreate the mask with the current detector geometry

Q-map not displaying
--------------------

* Ensure data is loaded (not empty canvas)
* Click **Generate Q-Map** first before toggling display
* Verify geometry parameters are reasonable values

Performance with large detectors
--------------------------------

For detectors larger than 2048x2048:

* Partition computation may take several seconds
* The status bar shows progress during computation
* Performance is optimized for up to 4096x4096 images
