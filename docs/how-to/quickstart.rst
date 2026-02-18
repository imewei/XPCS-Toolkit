Quick Start
===========

Concise recipes for getting up and running with XPCS Viewer. For a guided
walkthrough with explanations, see :doc:`/tutorials/getting_started`.

Launch the GUI
--------------

.. code-block:: bash

   # Launch GUI in the current directory
   xpcsviewer-gui

   # Launch GUI pointed at a data folder
   xpcsviewer-gui /path/to/hdf/data

   # Launch with debug logging
   xpcsviewer-gui --log-level DEBUG

GUI Workflow
~~~~~~~~~~~~

1. Launch: ``xpcsviewer-gui /path/to/data``
2. Select files from the Source list (left panel)
3. Add to Target with ``Ctrl+Shift+A`` or drag-and-drop
4. Choose an analysis tab (SAXS 2D, G2, Twotime, etc.)
5. Results display in the interactive plot area

Key Shortcuts
~~~~~~~~~~~~~

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``Ctrl+O``
     - Open folder
   * - ``Ctrl+R``
     - Reload data
   * - ``Ctrl+P``
     - Command Palette
   * - ``Ctrl+Shift+A``
     - Add to target
   * - ``Ctrl+L``
     - View logs

CLI Batch Processing
--------------------

.. code-block:: bash

   # Show available commands
   xpcsviewer --help

   # Generate twotime plots for all phi angles at q=0.05
   xpcsviewer twotime --input /data --output /results --q 0.05

   # Generate high-resolution PDF plots
   xpcsviewer twotime -i /data -o /results --phi 45 --dpi 300 --format pdf

Load Data (Python API)
----------------------

.. include:: /_includes/xpcsfile_basics.rst

Access SAXS Data
~~~~~~~~~~~~~~~~

.. code-block:: python

   from xpcsviewer.xpcs_file import XpcsFile

   xf = XpcsFile("data.hdf")

   # 1D SAXS
   q, Iq, xlabel, ylabel = xf.get_saxs1d_data()

   # 2D SAXS
   saxs_2d = xf.saxs_2d

Multi-File G2 Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xpcsviewer.module import g2mod

   xf_list = [XpcsFile(f) for f in ["file1.h5", "file2.h5"]]
   q, tel, g2, g2_err, labels = g2mod.get_data(xf_list)

See Also
--------

- :doc:`/tutorials/getting_started` -- Step-by-step tutorial with explanations
- :doc:`/tutorials/fitting_guide` -- G2 fitting workflow
- :doc:`/api/index` -- Python API reference
