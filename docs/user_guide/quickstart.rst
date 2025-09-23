Quick Start
===========

Launch GUI
----------

.. code-block:: bash

   # From HDF directory
   xpcs-toolkit /path/to/data

   # From current directory
   xpcs-toolkit

Load Data
---------

.. code-block:: python

   from xpcs_toolkit import XpcsFile

   # Load XPCS dataset
   xf = XpcsFile('data.hdf')
   print(f"Analysis type: {xf.atype}")

Basic Analysis
--------------

G2 Correlation
~~~~~~~~~~~~~~

.. code-block:: python

   from xpcs_toolkit.module import g2mod

   # Get G2 data
   success, g2_data, tau_data, q_data, labels = g2mod.get_data(
       [xf], q_range=[0, 10], t_range=[0, 100]
   )

SAXS Plot
~~~~~~~~~

.. code-block:: python

   from xpcs_toolkit.module import saxs1d

   # Plot SAXS data
   saxs1d.pg_plot(plot_handle, [xf], plot_type='single')
