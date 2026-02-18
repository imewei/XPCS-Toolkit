.. code-block:: python

   from xpcsviewer.xpcs_file import XpcsFile

   # Open a multi-tau XPCS result file
   xf = XpcsFile("/path/to/A001_Multitau_result.hdf")

   # Display a summary of loaded attributes
   print(xf)

   # Access analysis type
   print(f"Analysis type: {xf.atype}")

   # Extract G2 data for all Q bins
   q_values, t_el, g2, g2_err, labels = xf.get_g2_data()

   # Restrict by Q and delay-time range
   q_values, t_el, g2, g2_err, labels = xf.get_g2_data(
       qrange=(0.01, 0.05),
       trange=(0.1, 100),
   )
