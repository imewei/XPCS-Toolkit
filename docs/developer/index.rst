Developer Guide
===============

.. toctree::
   :maxdepth: 2

   optimization
   /architecture/dependency_analysis
   /architecture/integration_catalog
   /architecture/dependency_diagram
   /architecture/FACADE_INFRASTRUCTURE

Setup
-----

.. code-block:: bash

   git clone https://github.com/imewei/XPCSViewer.git
   cd XPCSViewer
   pip install -e .

Development
-----------

.. code-block:: bash

   # Run tests
   make test

   # Lint code
   make lint

   # Build docs
   make docs

Architecture
------------

- **Core**: ``xpcs_file.py`` data container
- **Modules**: ``module/`` analysis algorithms
- **File I/O**: ``fileIO/`` HDF5 access
- **GUI**: ``xpcs_viewer.py`` interface
- **Threading**: ``threading/`` concurrency
- **Utils**: ``utils/`` support functions
