"""JAX migration tests for xpcsviewer.

Test categories:
    backend/: Backend detection, device transfer, initialization
    numerical/: NumPy vs JAX equivalence, CPU vs GPU equivalence
    precision/: Float32 vs float64, angular computation accuracy
    performance/: Benchmark tests
    autograd/: Gradient-based calibration tests
    visualization/: Fitting visualization tests (FR-013 to FR-021)
    integration/: Qt-JAX interop, HDF5-JAX I/O, memory limits
"""
