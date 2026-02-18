Quick Start
===========

A 5-minute introduction to the main ways you can use XPCS Viewer: the GUI,
the CLI, and the Python API.

.. admonition:: What you'll learn

   - Launch the graphical interface and navigate the main window
   - Run command-line batch processing
   - Load an XPCS dataset with the Python API
   - Extract G2 and SAXS data from a file

Launch the GUI
--------------

XPCS Viewer includes a PySide6 GUI for interactive exploration. Launch it
from the terminal:

.. code-block:: bash

   # Launch GUI in current directory
   xpcsviewer-gui

   # Launch GUI with data path
   xpcsviewer-gui /path/to/hdf/data

   # With debug logging
   xpcsviewer-gui --log-level DEBUG

The GUI opens a main window with a source file list on the left and analysis
tabs on the right. The typical workflow is:

1. Select files from the Source list (left panel)
2. Add to Target with ``Ctrl+Shift+A`` or drag-and-drop
3. Choose an analysis tab (SAXS 2D, G2, Twotime, etc.)
4. Results display in the interactive plot area

CLI Batch Processing
--------------------

For scripted or automated pipelines, use the ``xpcsviewer`` command:

.. code-block:: bash

   # Show available commands
   xpcsviewer --help

   # Generate twotime plots for all phi angles at q=0.05
   xpcsviewer twotime --input /data --output /results --q 0.05

   # Generate high-resolution PDF plots
   xpcsviewer twotime -i /data -o /results --phi 45 --dpi 300 --format pdf

Load Data with the Python API
------------------------------

The :class:`~xpcsviewer.xpcs_file.XpcsFile` class is the main entry point
for programmatic access. It reads HDF5 files produced by XPCS correlators:

.. code-block:: python

   from xpcsviewer import XpcsFile

   # Load XPCS dataset
   xf = XpcsFile("data.hdf")
   print(f"Analysis type: {xf.atype}")

Extract G2 and SAXS Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xpcsviewer.module import g2mod

   # Get G2 data from multiple files
   xf_list = [XpcsFile(f) for f in ["file1.h5", "file2.h5"]]
   q, tel, g2, g2_err, labels = g2mod.get_data(xf_list)

.. code-block:: python

   from xpcsviewer import XpcsFile

   xf = XpcsFile("data.h5")

   # 1D SAXS (returns q, Iq, xlabel, ylabel)
   q, Iq, xlabel, ylabel = xf.get_saxs1d_data()

   # Or access the raw dictionary
   q_values = xf.saxs_1d["q"]
   intensities = xf.saxs_1d["Iq"]

   # 2D SAXS
   saxs_2d = xf.saxs_2d

Key Shortcuts
-------------

- ``Ctrl+O``: Open folder
- ``Ctrl+R``: Reload data
- ``Ctrl+P``: Command Palette
- ``Ctrl+Shift+A``: Add to target
- ``Ctrl+L``: View logs

Next Steps
----------

- See :doc:`getting_started` for a full walkthrough of loading and analyzing data
- See :doc:`mask_editor` for creating masks and Q-maps
- See :doc:`fitting_guide` for G2 curve fitting
- See :doc:`/how-to/quickstart` for a concise task-oriented reference
