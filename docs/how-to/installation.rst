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

Default installation uses CPU-only JAX. For GPU acceleration on Linux with an
NVIDIA GPU and system CUDA 12+ or 13+:

.. code-block:: bash

   # Auto-detect system CUDA version (from repo checkout)
   make install-jax-gpu

   # Or manually
   pip install jax[cuda12-local]    # System CUDA 12.x
   pip install jax[cuda13-local]    # System CUDA 13.x

Verify GPU detection:

.. code-block:: bash

   python -c "import jax; print(jax.devices())"
   # Expected: [CudaDevice(id=0), ...]

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
