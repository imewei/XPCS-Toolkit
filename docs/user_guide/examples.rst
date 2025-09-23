Examples
========

G2 Analysis
-----------

.. code-block:: python

   from xpcs_toolkit import XpcsFile
   from xpcs_toolkit.module import g2mod

   # Load data
   xf = XpcsFile('data.hdf')

   # Get G2 data
   success, g2_data, tau_data, q_data, labels = g2mod.get_data(
       [xf], q_range=[1, 5], t_range=[1e-6, 1]
   )

SAXS Processing
---------------

.. code-block:: python

   from xpcs_toolkit import XpcsFile

   # Load SAXS data
   xf = XpcsFile('saxs_data.hdf')
   saxs_2d_data = xf.saxs_2d

Batch Processing
----------------

.. code-block:: python

   from pathlib import Path
   from xpcs_toolkit import XpcsFile

   # Process multiple files
   data_dir = Path('/path/to/data')
   for hdf_file in data_dir.glob('*.hdf'):
       xf = XpcsFile(str(hdf_file))
       # Process data...
