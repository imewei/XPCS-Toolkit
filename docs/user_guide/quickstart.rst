Quick Start
===========

This guide helps you get started with XPCS Viewer quickly.

Launch GUI
----------

Use ``xpcsviewer-gui`` to launch the graphical interface:

.. code-block:: bash

   # Launch GUI in current directory
   xpcsviewer-gui

   # Launch GUI with data path
   xpcsviewer-gui /path/to/hdf/data

   # With debug logging
   xpcsviewer-gui --log-level DEBUG

CLI Batch Processing
--------------------

Use ``xpcsviewer`` for command-line batch processing:

.. code-block:: bash

   # Show available commands
   xpcsviewer --help

   # Generate twotime plots for all phi angles at q=0.05
   xpcsviewer twotime --input /data --output /results --q 0.05

   # Generate high-resolution PDF plots
   xpcsviewer twotime -i /data -o /results --phi 45 --dpi 300 --format pdf

Load Data (Python API)
----------------------

.. code-block:: python

   from xpcsviewer import XpcsFile

   # Load XPCS dataset
   xf = XpcsFile('data.hdf')
   print(f"Analysis type: {xf.atype}")

Basic Analysis
--------------

G2 Correlation
~~~~~~~~~~~~~~

.. code-block:: python

   from xpcsviewer.module import g2mod

   # Get G2 data from multiple files
   xf_list = [XpcsFile(f) for f in ['file1.h5', 'file2.h5']]
   q, tel, g2, g2_err, labels = g2mod.get_data(xf_list)

SAXS Data
~~~~~~~~~

.. code-block:: python

   from xpcsviewer import XpcsFile

   xf = XpcsFile('data.h5')

   # 1D SAXS
   q_values = xf.saxs_1d_q
   intensities = xf.saxs_1d

   # 2D SAXS
   saxs_2d = xf.saxs_2d

GUI Workflow
------------

1. Launch: ``xpcsviewer-gui /path/to/data``
2. Select files from Source list (left panel)
3. Add to Target with ``Ctrl+Shift+A`` or drag-and-drop
4. Choose analysis tab (SAXS 2D, G2, Twotime, etc.)
5. Results display in interactive plot area

Key Shortcuts
~~~~~~~~~~~~~

- ``Ctrl+O``: Open folder
- ``Ctrl+R``: Reload data
- ``Ctrl+P``: Command Palette
- ``Ctrl+Shift+A``: Add to target
- ``Ctrl+L``: View logs

Next Steps
----------

- See :doc:`/usage` for detailed CLI and GUI documentation
- See :doc:`examples` for more usage examples
- See :doc:`/api/index` for the Python API reference
