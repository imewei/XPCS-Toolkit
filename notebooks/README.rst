.. _notebooks:

Jupyter Notebooks
=================

Interactive notebooks for learning xpcsviewer.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - Description
   * - ``01_loading_data.ipynb``
     - Load HDF5 data, inspect metadata, extract G2 and SAXS profiles
   * - ``02_g2_analysis.ipynb``
     - G2 correlation analysis: fitting, tau(q), SAXS, heatmaps
   * - ``03_fitting_workflow.ipynb``
     - Complete fitting pipeline: NLSQ, Bayesian NUTS, model comparison

Running
-------

.. code-block:: bash

   cd notebooks/
   jupyter lab  # or jupyter notebook

Requirements: ``pip install jupyterlab xpcsviewer matplotlib``
