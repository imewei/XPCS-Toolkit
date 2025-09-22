Installation
============

Requirements
------------

- Python 3.10 or higher
- 64-bit operating system (Windows, macOS, or Linux)
- Minimum 8 GB RAM (16 GB recommended for large datasets)
- Graphics card with OpenGL support (for 3D visualization)

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- numpy >= 1.21.0
- scipy >= 1.7.0
- h5py >= 3.1.0
- PySide6 >= 6.2.0
- pyqtgraph >= 0.13.0
- matplotlib >= 3.5.0
- psutil >= 5.8.0

Install Methods
---------------

Method 1: pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install xpcs-toolkit

Method 2: Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create dedicated environment
   conda create -n xpcs-toolkit python=3.10
   conda activate xpcs-toolkit

   # Install package
   pip install xpcs-toolkit

Method 3: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For developers or users wanting the latest features:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/AZjk/xpcs-toolkit.git
   cd xpcs-toolkit

   # Create environment
   conda create -n xpcs-toolkit python=3.10
   conda activate xpcs-toolkit

   # Install in development mode
   pip install -e .

Verification
------------

Test the installation:

.. code-block:: python

   import xpcs_toolkit
   print(f"XPCS Toolkit version: {xpcs_toolkit.__version__}")

   # Test GUI (optional)
   from xpcs_toolkit import XpcsViewer
   # Should open GUI window

Common Issues
-------------

Qt Platform Plugin Error
~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter Qt platform plugin errors:

.. code-block:: bash

   # Linux
   export QT_QPA_PLATFORM=xcb

   # macOS (if issues with native plugin)
   export QT_QPA_PLATFORM=cocoa

Memory Issues
~~~~~~~~~~~~~

For large datasets, increase available memory:

.. code-block:: bash

   # Set environment variable for larger memory pools
   export PYXPCS_MAX_MEMORY=16GB

OpenGL Issues
~~~~~~~~~~~~~

For older graphics hardware:

.. code-block:: bash

   # Use software rendering
   export PYXPCS_FORCE_SOFTWARE_RENDERING=1

Configuration
-------------

Optional configuration file at ``~/.xpcs_toolkit/config.yaml``:

.. code-block:: yaml

   # Default data directory
   data_directory: "/path/to/xpcs/data"

   # Memory management
   max_memory_usage: "8GB"
   enable_caching: true

   # GUI settings
   default_theme: "dark"
   plot_backend: "pyqtgraph"

   # Logging
   log_level: "INFO"
   log_to_file: true
