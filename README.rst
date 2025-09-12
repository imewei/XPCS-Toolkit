============
XPCS Toolkit
============

A python-based interactive visualization tool to view XPCS dataset. This is forked from AdvancedPhotonSource/pyXpcsViewer.

**Requirements:** Python 3.12 or higher.

To cite XPCS Toolkit:  

Chu et al., *"pyXPCSviewer: an open-source interactive tool for X-ray photon correlation spectroscopy visualization and analysis"*, 
`Journal of Synchrotron Radiation, (2022) 29, 1122–1129 <https://onlinelibrary.wiley.com/doi/epdf/10.1107/S1600577522004830>`_.

Supported Format
----------------

This tools supports the customized nexus fileformat developed at APS-8IDI's XPCS data format for both multi-tau and two-time correlation. 

Install and Uninstall
---------------------
Updated 03/11/2025

It is highly recommended to set up a new `virtual environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
to isolate XPCS Toolkit, so it does not interfere with dependencies of your existing applications.

0. Install conda following the instructions at `link <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

1. Create a brand-new environment with conda:

   .. code-block:: bash

      conda create -n your_env_name python>=3.12

   Replace **your_env_name** with your preferred environment name. **Note:** Python 3.12 or higher is required.

2. Activate the new environment:

   .. code-block:: bash

      conda activate your_env_name

3. Install XPCS Toolkit:

   .. code-block:: bash

      pip install xpcs-toolkit

   **Note:** Running conda and pip commands together is generally not recommended. XPCS Toolkit will only use pip or conda once compatibility issues are resolved.

4. Launch XPCS Toolkit:

   1. Activate your environment if you have not already.
   2. Run:

      .. code-block:: bash

         xpcs-toolkit path_to_hdf_directory   # Run the viewer from the hdf directory
         xpcs-toolkit                         # Run in the current directory
         pyxpcsviewer path_to_hdf_directory   # Alternative command (legacy)
         run_viewer                           # Alternative command (alias)

    run_viewer and pyxpcsviewer are aliases to xpcs-toolkit and can also be used to launch the viewer.

5. To upgrade:

   1. Activate your environment if you have not already.
   2. Run:

      .. code-block:: bash

         pip install -U xpcs-toolkit

6. To uninstall:

   1. Activate your environment if you have not already.
   2. Run:

      .. code-block:: bash

         pip uninstall xpcs-toolkit

   3. If you want to remove the environment altogether, first deactivate it:

      .. code-block:: bash

         conda deactivate

      Then remove it:

      .. code-block:: bash

         conda remove -n your_env_name --all

Performance Optimizations
-------------------------

The XPCS Toolkit includes comprehensive performance optimizations that provide **25-40% overall performance improvement**:

* **Threading System Optimizations**: Signal batching, enhanced thread pools, optimized workers
* **Memory Management**: Advanced caching, optimized cleanup, pressure monitoring  
* **I/O Optimizations**: HDF5 connection pooling, batch operations
* **Scientific Computing**: Vectorized algorithms, parallel processing
* **Monitoring Ecosystem**: Real-time performance monitoring, bottleneck detection

**Quick Setup** (one-line optimization activation):

.. code-block:: python

   from xpcs_toolkit.utils import setup_complete_optimization_ecosystem
   setup_complete_optimization_ecosystem()  # Enables all optimizations

For complete optimization documentation, see `docs/OPTIMIZATION_GUIDE.md <docs/OPTIMIZATION_GUIDE.md>`_.

Documentation
-------------

* **📖 Complete Documentation**: `docs/DOCUMENTATION_INDEX.md <docs/DOCUMENTATION_INDEX.md>`_ - Navigation guide for all documentation
* **🎯 Performance Guide**: `docs/OPTIMIZATION_GUIDE.md <docs/OPTIMIZATION_GUIDE.md>`_ - Complete optimization reference
* **🔍 Logging Guide**: `docs/LOGGING_SYSTEM.md <docs/LOGGING_SYSTEM.md>`_ - Logging infrastructure and best practices
* **🧪 Testing Guide**: `docs/TESTING.md <docs/TESTING.md>`_ - Testing framework and validation
* **🛠️ Development Guide**: `CLAUDE.md <CLAUDE.md>`_ - Development workflows and architecture
* **📋 Production Guide**: `docs/PRODUCTION_READINESS_FINAL_REPORT.md <docs/PRODUCTION_READINESS_FINAL_REPORT.md>`_ - Production deployment guidance

Gallery
-------

1. The integrated scattering pattern over the whole time series.

   .. image:: docs/images/saxs2d.png

2. The reduced one-dimensional small-angle scattering data.

   .. image:: docs/images/saxs1d.png

3. The sample's stability against X-ray beam damage. The time series is divided into 10 sections. The SAXS-1D curve is plotted for each section.

   .. image:: docs/images/stability.png

4. Intensity fluctuation vs. Time.

   .. image:: docs/images/intt.png

5. Average Tool box.

   .. image:: docs/images/average.png

6. g2 plot for multitau analysis. Users can fit the time scale using a single exponential function, with options to specify the fitting range and fitting flags (fix or fit).

   .. image:: docs/images/g2mod.png

7. Diffusion analysis. g2 fitting in the previous panel is required to plot :math:`\tau \mbox{vs.} q`.

   .. image:: docs/images/diffusion.png

8. Two-time correlation. Users can select two q indexes either on the q-map or on the SAXS-2D image.

   .. image:: docs/images/twotime.png

9. Experiment condition viewer. It reads the file structure and string entries of the selected HDF file.

   .. image:: docs/images/hdf_info.png