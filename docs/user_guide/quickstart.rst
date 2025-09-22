Quick Start Guide
=================

This guide covers basic usage of XPCS Toolkit for common analysis workflows.

Basic Data Loading
------------------

Load an XPCS dataset:

.. code-block:: python

   from xpcs_toolkit import XpcsFile

   # Load data file
   xf = XpcsFile('path/to/your/data.hdf')

   # Check analysis type
   print(f"Analysis type: {xf.atype}")
   print(f"Data shape: {xf.saxs_2d.shape}")

Launching the GUI
-----------------

Start the interactive viewer:

.. code-block:: python

   from xpcs_toolkit.xpcs_viewer import main

   # Launch GUI
   main()

Or from command line:

.. code-block:: bash

   # Launch from directory containing HDF files
   xpcs-toolkit /path/to/data/directory

   # Launch from current directory
   xpcs-toolkit

Common Analysis Workflows
--------------------------

G2 Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xpcs_toolkit.module import g2mod

   # Load data files
   xf_list = [XpcsFile(f) for f in file_list]

   # Get G2 data
   success, g2_data, tau_data, q_data, labels = g2mod.get_data(
       xf_list,
       q_range=[0, 10],
       t_range=[0, 100]
   )

   if success:
       print(f"G2 data shape: {g2_data[0].shape}")

SAXS Analysis
~~~~~~~~~~~~~

.. code-block:: python

   from xpcs_toolkit.module import saxs1d

   # Plot SAXS data
   saxs1d.pg_plot(
       plot_handle,
       xf_list,
       plot_type='single',
       plot_norm='log'
   )

Two-time Correlation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xpcs_toolkit.module import twotime

   # Display two-time correlation
   twotime.pg_plot(
       plot_handles,
       xf,
       selection=None,
       plot_type='default'
   )

File Management
---------------

Working with Multiple Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from xpcs_toolkit.file_locator import FileLocator

   # Initialize file locator
   locator = FileLocator('/path/to/data')

   # Get file list with filters
   file_info = locator.get_hdf_info(
       target_list=['sample1', 'sample2'],
       atype_filter=['Multitau']
   )

Data Access Patterns
--------------------

Lazy Loading
~~~~~~~~~~~~

XPCS Toolkit uses lazy loading for memory efficiency:

.. code-block:: python

   # Data is loaded on first access
   xf = XpcsFile('large_dataset.hdf')

   # This triggers actual data loading
   saxs_data = xf.saxs_2d  # Large array loaded here

   # Subsequent access uses cached data
   same_data = xf.saxs_2d  # No additional loading

Memory Management
~~~~~~~~~~~~~~~~~

For large datasets:

.. code-block:: python

   from xpcs_toolkit.utils import MemoryManager

   # Configure memory limits
   MemoryManager.set_max_memory("8GB")

   # Enable automatic cleanup
   MemoryManager.enable_pressure_detection()

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Suppress Qt warnings
   export PYXPCS_SUPPRESS_QT_WARNINGS=1

   # Set memory limit
   export PYXPCS_MAX_MEMORY=16GB

   # Enable debug logging
   export PYXPCS_LOG_LEVEL=DEBUG

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**File not found errors**:
   Ensure HDF5 files contain required XPCS data structure

**Memory errors with large datasets**:
   Increase system memory or reduce data size

**GUI not responding**:
   Use async workers for long computations

**Qt platform errors**:
   Set appropriate QT_QPA_PLATFORM environment variable

Getting Help
------------

- Check the :doc:`../api/index` for detailed function documentation
- See :doc:`examples` for more complex workflows
- Review :doc:`../developer/index` for development information
