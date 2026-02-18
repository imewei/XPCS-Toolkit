API Reference
=============

.. toctree::
   :maxdepth: 2

   package
   backends
   schemas
   io
   fitting
   modules
   simplemask
   fileio
   cli
   plotting
   threading
   utils
   constants
   gui
   helper

Core Components
---------------

- **XpcsFile**: Data container and analysis (:doc:`package`)
- **Backends**: JAX/NumPy abstraction with device management (:doc:`backends`)
- **Schemas**: Type-safe validated data structures (:doc:`schemas`)
- **I/O Facade**: HDF5 operations with schema validation (:doc:`io`)
- **Fitting**: Bayesian fitting with NLSQ warm-start (:doc:`fitting`)
- **Analysis Modules**: G2, SAXS, two-time, stability (:doc:`modules`)
- **SimpleMask**: Mask editing and Q-map generation (:doc:`simplemask`)
- **File I/O**: Low-level HDF5 reading and Q-space mapping (:doc:`fileio`)
- **CLI**: Command-line interface for batch processing (:doc:`cli`)
- **Plotting**: PyQtGraph and Matplotlib integration (:doc:`plotting`)
- **Threading**: Asynchronous workers and progress management (:doc:`threading`)
- **Constants**: Application-wide configuration values (:doc:`constants`)
- **GUI**: PySide6 interactive visualization (:doc:`gui`)
- **Utils**: Logging, memory management, optimization (:doc:`utils`)
- **Helper**: Internal data models and utilities (:doc:`helper`)
