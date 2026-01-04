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

Package Information
-------------------

.. autodata:: xpcsviewer.__version__
   :annotation: = version string

.. autodata:: xpcsviewer.__author__
   :annotation: = "Miaoqi Chu"

.. autodata:: xpcsviewer.__credits__
   :annotation: = "Argonne National Laboratory"
