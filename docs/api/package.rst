xpcsviewer package
====================

Main Package
------------

.. automodule:: xpcsviewer
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: XpcsFile

Core Classes
------------

XpcsFile
~~~~~~~~

.. autoclass:: xpcsviewer.XpcsFile
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   The main data container class for XPCS datasets. Provides lazy loading
   of large data arrays and built-in analysis capabilities.

   .. note::
      XpcsFile automatically detects the analysis type (Multitau, Twotime, etc.)
      and loads appropriate data fields.

XpcsFile Submodules
-------------------

The ``xpcsviewer.xpcs_file`` package splits internal concerns into submodules:

Cache
~~~~~

LRU data cache with memory-pressure awareness.

.. automodule:: xpcsviewer.xpcs_file.cache
   :members:
   :no-index:

Fitting Helpers
~~~~~~~~~~~~~~~

File-level fitting convenience functions used internally by
:meth:`~xpcsviewer.XpcsFile.fit_g2`.

.. automodule:: xpcsviewer.xpcs_file.fitting
   :members:
   :no-index:

Memory Monitor
~~~~~~~~~~~~~~

Virtual-memory monitoring for adaptive cache eviction.

.. automodule:: xpcsviewer.xpcs_file.memory
   :members:
   :no-index:

Package Information
-------------------

.. autodata:: xpcsviewer.__version__
   :annotation: = version string

.. autodata:: xpcsviewer.__author__
   :annotation: = "Miaoqi Chu & Wei Chen"

.. autodata:: xpcsviewer.__credits__
   :annotation: = "Argonne National Laboratory"
