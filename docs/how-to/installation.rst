Installation
============

Requirements
------------

- Python 3.12+
- 8 GB RAM minimum

Install
-------

.. include:: /_includes/installation_snippet.rst

All scientific dependencies (JAX, NumPyro, ArviZ, interpax, etc.) are included
automatically — no extras needed.

GPU Acceleration (Optional)
---------------------------

**Performance Impact:** 20-100x speedup for large datasets (>1M points)

**Prerequisites:**

- NVIDIA GPU with SM >= 5.2 (Maxwell or newer)
- System CUDA 12.x or 13.x installed (``nvcc`` in PATH)

**Quick Install (Recommended):**

.. code-block:: bash

   # Auto-detect system CUDA version and install matching JAX
   make install-jax-gpu

This detects your system CUDA version, validates GPU compatibility,
removes any conflicting packages, installs the correct JAX GPU package,
and verifies GPU detection.

**Manual Install:**

.. code-block:: bash

   # 1. Check your CUDA version
   nvcc --version    # Note: release 12.x or 13.x

   # 2. Remove ALL existing JAX/CUDA packages (prevents plugin conflicts)
   pip uninstall -y jax jaxlib \
       jax-cuda13-plugin jax-cuda13-pjrt \
       jax-cuda12-plugin jax-cuda12-pjrt

   # 3. Install matching package
   pip install "jax[cuda13-local]"   # For CUDA 13.x (SM >= 7.5)
   pip install "jax[cuda12-local]"   # For CUDA 12.x (SM >= 5.2)

   # 4. Verify
   python -c "import jax; print(jax.default_backend(), jax.devices())"
   # Expected: gpu [CudaDevice(id=0)]

Or install via project extras (ensures matching versions):

.. code-block:: bash

   pip install "xpcsviewer-gui[gpu_cuda13]"   # CUDA 13
   pip install "xpcsviewer-gui[gpu_cuda12]"   # CUDA 12

**Troubleshooting:**

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Symptom
     - Cause
     - Fix
   * - ``plugin version X is not compatible with jaxlib Y``
     - Plugin/jaxlib version mismatch
     - Uninstall all, reinstall: ``make install-jax-gpu``
   * - ``PJRT_Api already exists for device type cuda``
     - Both cuda12 and cuda13 plugins installed
     - Uninstall all, reinstall only ONE
   * - ``nvcc not found``
     - CUDA toolkit missing or not in PATH
     - ``sudo apt install nvidia-cuda-toolkit``
   * - ``Backend: cpu`` (GPU exists)
     - GPU JAX packages not installed
     - ``make install-jax-gpu``

**Diagnostics:**

.. code-block:: bash

   make gpu-check      # Verify backend, devices, SVD
   make gpu-diagnose   # Check plugins, versions, conflicts

Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/imewei/XPCSViewer.git
   cd XPCSViewer

   # With uv (recommended) — installs package + dev/test dependencies
   uv sync
   make install-hooks      # install pre-commit + commit-msg hooks

   # Or with pip (editable install, no dev dependencies)
   pip install -e .

Verify
------

.. code-block:: python

   import xpcsviewer
   print(xpcsviewer.__version__)
