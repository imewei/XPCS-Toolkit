Installation
============

Requirements
------------

- Python 3.12+
- 8 GB RAM minimum

Install
-------

.. include:: /_includes/installation_snippet.rst

Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/imewei/XPCSViewer.git
   cd XPCSViewer
   pip install -e .

Verify
------

.. code-block:: python

   import xpcsviewer
   print(xpcsviewer.__version__)
