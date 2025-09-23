xpcs_toolkit package
====================

Main Package
------------

.. automodule:: xpcs_toolkit
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: XpcsFile

Core Classes
------------

XpcsFile
~~~~~~~~

.. autoclass:: xpcs_toolkit.XpcsFile
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   The main data container class for XPCS datasets. Provides lazy loading
   of large data arrays and built-in analysis capabilities.

   .. note::
      XpcsFile automatically detects the analysis type (Multitau, Twotime, etc.)
      and loads appropriate data fields.

Package Information
-------------------

.. autodata:: xpcs_toolkit.__version__
   :annotation: = version string

.. autodata:: xpcs_toolkit.__author__
   :annotation: = "Miaoqi Chu"

.. autodata:: xpcs_toolkit.__credits__
   :annotation: = "Argonne National Laboratory"
