==========
SimpleMask
==========

The ``xpcsviewer.simplemask`` module provides mask editing and Q-map functionality
for XPCS analysis.

Module Overview
===============

.. automodule:: xpcsviewer.simplemask
   :members:
   :undoc-members:
   :show-inheritance:

SimpleMaskWindow
================

The main window class for mask editing.

.. autoclass:: xpcsviewer.simplemask.SimpleMaskWindow
   :members:
   :undoc-members:
   :show-inheritance:

   **Signals**

   .. py:attribute:: mask_exported
      :type: Signal(np.ndarray)

      Emitted when mask is exported to XPCS Viewer.

   .. py:attribute:: qmap_exported
      :type: Signal(dict)

      Emitted when partition is exported to XPCS Viewer.

   **Key Methods**

   .. py:method:: load_from_viewer(detector_image, metadata)

      Load detector image and metadata from XPCS Viewer.

      :param detector_image: 2D numpy array of detector data
      :param metadata: Dictionary with geometry parameters (bcx, bcy, det_dist, pix_dim, energy)

   .. py:method:: export_mask_to_viewer()

      Export current mask via mask_exported signal.

   .. py:method:: export_partition_to_viewer()

      Export current partition via qmap_exported signal.

SimpleMaskKernel
================

Core computation logic for mask operations.

.. autoclass:: xpcsviewer.simplemask.simplemask_kernel.SimpleMaskKernel
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Methods**

   .. py:method:: read_data(det_image, parameters)

      Initialize kernel with detector data and geometry.

   .. py:method:: add_drawing(drawing_type, params)

      Add a drawing ROI to the kernel.

      :param drawing_type: One of 'rectangle', 'circle', 'polygon', 'line', 'ellipse'
      :param params: Dictionary with ROI parameters

   .. py:method:: apply_drawing(inclusive=False)

      Apply all drawings to create/update the mask.

   .. py:method:: mask_action(action='apply')

      Perform mask action (apply, undo, redo, reset).

   .. py:method:: compute_partition(dq_num=10, sq_num=100, dp_num=36, sp_num=360)

      Compute Q-space partition for correlation analysis.

      :param dq_num: Number of dynamic Q-bins
      :param sq_num: Number of static Q-bins
      :param dp_num: Number of dynamic phi-bins
      :param sp_num: Number of static phi-bins
      :returns: Dictionary with partition data

   .. py:method:: save_mask(file_path)

      Save mask to HDF5 file.

   .. py:method:: load_mask(file_path)

      Load mask from HDF5 file.

Q-Map Module
============

Q-map computation from detector geometry.

.. automodule:: xpcsviewer.simplemask.qmap
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Functions**

   .. py:function:: compute_qmap(shape, bcx, bcy, det_dist, pix_dim, energy, orientation='N')

      Compute Q-map for detector geometry.

      :param shape: Detector shape (rows, cols)
      :param bcx: Beam center X (column)
      :param bcy: Beam center Y (row)
      :param det_dist: Sample-to-detector distance (mm)
      :param pix_dim: Pixel dimension (mm)
      :param energy: X-ray energy (keV)
      :param orientation: Detector orientation ('N', 'E', 'S', 'W')
      :returns: Dictionary with 'q', 'phi', 'r' arrays

Area Mask Module
================

Mask assembly with undo/redo history.

.. automodule:: xpcsviewer.simplemask.area_mask
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Classes**

   .. py:class:: MaskAssemble

      Manages mask with undo/redo history.

      .. py:method:: apply(mask_component)

         Apply a mask component and add to history.

      .. py:method:: undo()

         Undo the last mask change.

      .. py:method:: redo()

         Redo a previously undone change.

      .. py:method:: reset()

         Reset mask to initial state.

      .. py:method:: get_mask()

         Get the current combined mask.

Drawing Tools Module
====================

Tool definitions for mask drawing.

.. automodule:: xpcsviewer.simplemask.drawing_tools
   :members:
   :undoc-members:
   :show-inheritance:

   **Available Tools**

   * ``DRAWING_TOOLS['rectangle']`` - Rectangle ROI
   * ``DRAWING_TOOLS['circle']`` - Circle ROI
   * ``DRAWING_TOOLS['polygon']`` - Polygon ROI
   * ``DRAWING_TOOLS['line']`` - Line ROI with width
   * ``DRAWING_TOOLS['ellipse']`` - Ellipse ROI
   * ``ERASER_TOOL`` - Eraser (inclusive mode)

Utils Module
============

Partition utilities for Q-binning.

.. automodule:: xpcsviewer.simplemask.utils
   :members:
   :undoc-members:
   :show-inheritance:

PyQtGraph Modifications
=======================

Custom ROI classes for drawing tools.

.. automodule:: xpcsviewer.simplemask.pyqtgraph_mod
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Classes**

   .. py:class:: LineROI

      Custom line ROI with adjustable width for mask drawing.
