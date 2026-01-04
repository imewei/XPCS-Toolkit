Command-Line Interface
======================

Entry points and command-line utilities for XPCS Viewer.

.. currentmodule:: xpcsviewer

Entry Points
------------

The package provides separate entry points for CLI and GUI usage.

GUI Entry Point
~~~~~~~~~~~~~~~

Launch the graphical interface with ``xpcsviewer-gui``.

The ``main_gui()`` function is the primary GUI entry point that provides
interactive XPCS data analysis.

**Usage:**

.. code-block:: bash

   # Launch GUI in current directory
   xpcsviewer-gui

   # Launch with data path
   xpcsviewer-gui /path/to/hdf/data

   # With debug logging
   xpcsviewer-gui --log-level DEBUG

CLI Entry Point
~~~~~~~~~~~~~~~

Access batch processing commands with ``xpcsviewer``.

The ``main()`` function is the CLI entry point for batch processing operations.

**Usage:**

.. code-block:: bash

   # Show available commands
   xpcsviewer --help

   # Run twotime batch processing
   xpcsviewer twotime --input /data --output /results --q 0.05

Available Commands
------------------

Twotime Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~

Generate twotime correlation images from HDF5 files.

.. automodule:: xpcsviewer.cli.twotime_batch
   :members:
   :undoc-members:
   :show-inheritance:

**Command syntax:**

.. code-block:: bash

   xpcsviewer twotime --input INPUT --output OUTPUT (--q Q | --phi PHI | --q-phi Q,PHI)
                     [--dpi DPI] [--format {png,jpg,pdf,svg}]

**Options:**

- ``--input, -i``: HDF file path or directory containing HDF files
- ``--output, -o``: Output directory for generated images
- ``--q``: Q-value to process (generates all phi angles)
- ``--phi``: Phi-value to process (generates all q values)
- ``--q-phi``: Specific q-phi pair as 'q,phi' (single image)
- ``--dpi``: Image resolution in DPI (default: 300)
- ``--format``: Image format: png, jpg, pdf, svg (default: png)

**Examples:**

.. code-block:: bash

   # Process all phi angles at a specific q-value
   xpcsviewer twotime --input /path/to/data --output /results --q 0.05

   # Process all q values at a specific phi angle
   xpcsviewer twotime --input /path/to/data --output /results --phi 45

   # Generate high-resolution PDF plots
   xpcsviewer twotime -i /data -o /results --q 0.05 --dpi 300 --format pdf

Utility Functions
-----------------

safe_shutdown
~~~~~~~~~~~~~

Gracefully shuts down the application, cleaning up resources and thread pools.
Called automatically on exit or when handling termination signals.

signal_handler
~~~~~~~~~~~~~~

Handles termination signals (SIGTERM, SIGINT) for graceful shutdown.
Registered automatically when the CLI or GUI starts.
