Developer Documentation
========================

This section contains technical documentation for developers working on
XPCS Toolkit or integrating it into other software.

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   contributing
   testing
   optimization
   logging
   qt_compliance
   deployment

Development Setup
-----------------

Clone and Setup
~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/AZjk/xpcs-toolkit.git
   cd xpcs-toolkit

   # Create development environment
   conda create -n xpcs-dev python=3.10
   conda activate xpcs-dev

   # Install in development mode with dev dependencies
   pip install -e ".[dev]"

Development Tools
~~~~~~~~~~~~~~~~~

The project uses:

- **Testing**: pytest with extensive test coverage
- **Linting**: ruff for code formatting and style
- **Type checking**: mypy for static type analysis
- **Documentation**: Sphinx for documentation generation
- **Version control**: Git with conventional commits

Development Commands
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run tests
   pytest tests/ -v

   # Run with coverage
   pytest tests/ --cov=xpcs_toolkit

   # Lint code
   ruff check xpcs_toolkit/
   ruff format xpcs_toolkit/

   # Type checking
   mypy xpcs_toolkit/

   # Build documentation
   cd docs
   make html

Code Standards
--------------

- Follow PEP 8 style guidelines
- Use type hints for all public APIs
- Write docstrings in NumPy format
- Maintain test coverage above 80%
- Use conventional commits for git messages
- Keep functions focused and modular

Architecture Overview
---------------------

The codebase is organized into several key subsystems:

- **Core Data** (``xpcs_file.py``): Main data container
- **Analysis Modules** (``module/``): Scientific algorithms
- **File I/O** (``fileIO/``): HDF5 data access
- **GUI** (``xpcs_viewer.py``): User interface
- **Threading** (``threading/``): Concurrency management
- **Utilities** (``utils/``): Support functions
