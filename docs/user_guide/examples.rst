Examples
========

This section provides practical examples for common XPCS analysis workflows.

Example 1: Basic G2 Analysis
-----------------------------

Complete workflow for G2 correlation analysis:

.. code-block:: python

   import numpy as np
   from xpcs_toolkit import XpcsFile
   from xpcs_toolkit.module import g2mod
   import matplotlib.pyplot as plt

   # Load data files
   file_paths = ['data_001.hdf', 'data_002.hdf', 'data_003.hdf']
   xf_list = [XpcsFile(f) for f in file_paths]

   # Extract G2 data
   success, g2_data, tau_data, q_data, labels = g2mod.get_data(
       xf_list,
       q_range=[1, 5],    # Q range selection
       t_range=[1e-6, 1]  # Time range in seconds
   )

   if success:
       # Plot results
       for i, (g2, tau, label) in enumerate(zip(g2_data, tau_data, labels)):
           plt.figure(figsize=(10, 6))
           for q_idx in range(g2.shape[1]):
               plt.semilogx(tau, g2[:, q_idx],
                           label=f'Q={q_data[q_idx]:.3f}')

           plt.xlabel('Time (s)')
           plt.ylabel('G2(τ)')
           plt.title(f'G2 Correlation - {label}')
           plt.legend()
           plt.grid(True)
           plt.show()

Example 2: SAXS Data Processing
-------------------------------

Process and visualize SAXS data:

.. code-block:: python

   from xpcs_toolkit import XpcsFile
   from xpcs_toolkit.module import saxs1d, saxs2d
   import numpy as np

   # Load SAXS data
   xf = XpcsFile('saxs_data.hdf')

   # Check if it's SAXS analysis type
   if 'Saxs' in xf.atype:
       print(f"SAXS data shape: {xf.saxs_2d.shape}")

       # Get 2D SAXS data
       saxs_2d_data = xf.saxs_2d

       # Calculate radial average for 1D profile
       from xpcs_toolkit.fileIO.qmap_utils import get_qmap
       qmap = get_qmap(xf.fname)

       # Create Q bins
       q_min, q_max = qmap.min(), qmap.max()
       q_bins = np.linspace(q_min, q_max, 100)

       # Radial averaging (simplified)
       q_centers = (q_bins[1:] + q_bins[:-1]) / 2
       intensity_1d = []

       for i in range(len(q_bins)-1):
           mask = (qmap >= q_bins[i]) & (qmap < q_bins[i+1])
           intensity_1d.append(np.mean(saxs_2d_data[mask]))

       # Plot 1D profile
       plt.figure(figsize=(8, 6))
       plt.loglog(q_centers, intensity_1d, 'o-')
       plt.xlabel('Q (Å⁻¹)')
       plt.ylabel('Intensity')
       plt.title('SAXS 1D Profile')
       plt.grid(True)
       plt.show()

Example 3: Batch Processing
---------------------------

Process multiple files in batch:

.. code-block:: python

   import os
   from pathlib import Path
   from xpcs_toolkit import XpcsFile
   from xpcs_toolkit.module import g2mod

   # Define data directory
   data_dir = Path('/path/to/xpcs/data')
   output_dir = Path('./analysis_results')
   output_dir.mkdir(exist_ok=True)

   # Find all HDF files
   hdf_files = list(data_dir.glob('*.hdf'))

   # Process each file
   results = {}

   for hdf_file in hdf_files:
       try:
           # Load file
           xf = XpcsFile(str(hdf_file))

           # Check if it's correlation analysis
           if any(atype in ['Multitau', 'Twotime'] for atype in xf.atype):
               # Extract G2 data
               success, g2_data, tau_data, q_data, labels = g2mod.get_data(
                   [xf],
                   q_range=[0.5, 2.0],
                   t_range=[1e-6, 10]
               )

               if success:
                   # Store results
                   results[hdf_file.stem] = {
                       'g2': g2_data[0],
                       'tau': tau_data[0],
                       'q': q_data,
                       'label': labels[0]
                   }

                   # Save data
                   output_file = output_dir / f"{hdf_file.stem}_g2.npz"
                   np.savez(output_file,
                           g2=g2_data[0],
                           tau=tau_data[0],
                           q=q_data)

                   print(f"Processed: {hdf_file.name}")

       except Exception as e:
           print(f"Error processing {hdf_file.name}: {e}")

   print(f"Processed {len(results)} files successfully")

Example 4: Memory-Efficient Processing
--------------------------------------

Handle large datasets with memory management:

.. code-block:: python

   from xpcs_toolkit import XpcsFile
   from xpcs_toolkit.utils import MemoryManager
   import gc

   # Configure memory management
   MemoryManager.set_max_memory("8GB")
   MemoryManager.enable_pressure_detection()

   # Process large files sequentially
   large_files = ['large_data_1.hdf', 'large_data_2.hdf']

   for file_path in large_files:
       # Load file
       xf = XpcsFile(file_path)

       # Process data in chunks if needed
       if hasattr(xf, 'saxs_2d') and xf.saxs_2d.nbytes > 1e9:  # > 1GB
           print(f"Large dataset detected: {xf.saxs_2d.nbytes/1e9:.1f} GB")

           # Process in chunks
           chunk_size = 1000
           total_frames = xf.saxs_2d.shape[0]

           for start in range(0, total_frames, chunk_size):
               end = min(start + chunk_size, total_frames)
               chunk = xf.saxs_2d[start:end]

               # Process chunk
               result = np.mean(chunk, axis=0)

               # Save or accumulate results
               # ... processing code ...

       # Explicit cleanup
       del xf
       gc.collect()

       print(f"Memory usage: {MemoryManager.get_memory_usage():.1f}%")

Example 5: Custom ROI Analysis
------------------------------

Define and analyze custom regions of interest:

.. code-block:: python

   from xpcs_toolkit import XpcsFile
   from xpcs_toolkit.utils.vectorized_roi import VectorizedROI
   import numpy as np

   # Load data
   xf = XpcsFile('roi_analysis.hdf')

   # Define custom ROI
   roi_params = {
       'center_x': 512,
       'center_y': 512,
       'radius_inner': 50,
       'radius_outer': 100,
       'angle_start': 0,
       'angle_end': 90  # First quadrant
   }

   # Create ROI mask
   roi_processor = VectorizedROI()
   mask = roi_processor.create_annular_mask(
       shape=xf.saxs_2d.shape[-2:],
       **roi_params
   )

   # Apply ROI to data
   if len(xf.saxs_2d.shape) == 3:  # Time series
       roi_intensity = []
       for frame in xf.saxs_2d:
           roi_intensity.append(np.mean(frame[mask]))

       # Plot time series
       plt.figure(figsize=(10, 6))
       plt.plot(roi_intensity)
       plt.xlabel('Frame Number')
       plt.ylabel('ROI Intensity')
       plt.title('Custom ROI Time Series')
       plt.grid(True)
       plt.show()

   else:  # Single frame
       roi_value = np.mean(xf.saxs_2d[mask])
       print(f"ROI average intensity: {roi_value:.2f}")

Running the Examples
--------------------

To run these examples:

1. Ensure you have XPCS data files in the correct format
2. Update file paths to match your data location
3. Install required dependencies (matplotlib for plotting)
4. Run Python scripts or use in Jupyter notebooks

For more examples, see the `examples/` directory in the source repository.
