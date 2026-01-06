"""Tests for Qt/JAX interoperability (T074).

Tests that Qt widgets work correctly with JAX backend (US5).
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQtJaxDataConversion:
    """Tests for data conversion between Qt and JAX."""

    def test_jax_array_to_numpy_for_pyqtgraph(self, monkeypatch) -> None:
        """Test JAX arrays convert to NumPy for PyQtGraph."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Create JAX array
        jax_array = backend.linspace(0, 1, 100)

        # Convert to NumPy (required for PyQtGraph)
        numpy_array = ensure_numpy(jax_array)

        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (100,)

    def test_qmap_returns_numpy_for_display(self, monkeypatch) -> None:
        """Test Q-map returns NumPy arrays suitable for Qt display."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # All returned arrays should be NumPy (I/O boundary)
        for key, array in qmap.items():
            assert isinstance(array, np.ndarray), f"{key} is not NumPy array"

    def test_partition_returns_numpy_for_display(self, monkeypatch) -> None:
        """Test partition returns NumPy arrays suitable for Qt display."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.utils import generate_partition

        qmap = np.random.random((128, 128)).astype(np.float64)
        mask = np.ones((128, 128), dtype=bool)

        result = generate_partition(
            map_name="q", mask=mask, xmap=qmap, num_pts=36, style="linear"
        )
        # generate_partition returns a dict with 'partition' key
        partition = result["partition"]

        assert isinstance(partition, np.ndarray)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQtJaxSignals:
    """Tests for Qt signal handling with JAX arrays."""

    def test_numpy_conversion_at_signal_boundaries(self, monkeypatch) -> None:
        """Test arrays are converted to NumPy at signal boundaries."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Simulate computation that would be emitted via signal
        result = backend.sin(backend.linspace(0, 2 * backend.pi, 100))

        # Before emitting via Qt signal, convert to NumPy
        signal_data = ensure_numpy(result)

        assert isinstance(signal_data, np.ndarray)
        # Should be copyable (needed for Qt signals)
        copied = signal_data.copy()
        assert copied is not signal_data

    def test_mask_array_conversion(self, monkeypatch) -> None:
        """Test mask arrays convert properly for Qt display."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Create boolean mask (JAX style)
        x = backend.linspace(-1, 1, 100)
        y = backend.linspace(-1, 1, 100)
        xx, yy = backend.meshgrid(x, y, indexing="ij")
        mask = (xx**2 + yy**2) < 0.5

        # Convert for display
        numpy_mask = ensure_numpy(mask)

        assert isinstance(numpy_mask, np.ndarray)
        assert numpy_mask.dtype == np.bool_ or numpy_mask.dtype == bool


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQtJaxThreading:
    """Tests for Qt/JAX threading compatibility."""

    def test_backend_access_from_main_thread(self, monkeypatch) -> None:
        """Test backend can be accessed from main thread."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        # Should work from main thread
        backend = get_backend()
        result = backend.array([1, 2, 3])

        assert result is not None

    def test_multiple_computations_no_crash(self, monkeypatch) -> None:
        """Test multiple rapid computations don't cause issues."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        results = []
        for i in range(10):
            x = backend.linspace(0, i + 1, 100)
            y = backend.sin(x)
            results.append(ensure_numpy(y))

        assert len(results) == 10
        for r in results:
            assert isinstance(r, np.ndarray)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestImageDataConversion:
    """Tests for image data conversion for Qt display."""

    def test_image_array_for_display(self, monkeypatch) -> None:
        """Test image arrays convert correctly for display."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Simulate detector image data
        shape = (256, 256)
        image = backend.linspace(0, 1, shape[0] * shape[1])
        image = image.reshape(shape)

        # Convert for Qt ImageItem
        numpy_image = ensure_numpy(image)

        assert isinstance(numpy_image, np.ndarray)
        assert numpy_image.shape == shape
        # Should be C-contiguous for Qt
        assert numpy_image.flags["C_CONTIGUOUS"]

    def test_float_to_uint8_conversion(self, monkeypatch) -> None:
        """Test float images can be converted to uint8 for display."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Float image in [0, 1]
        float_image = backend.linspace(0, 1, 100).reshape(10, 10)

        # Convert to NumPy then to uint8
        numpy_float = ensure_numpy(float_image)
        uint8_image = (numpy_float * 255).astype(np.uint8)

        assert uint8_image.dtype == np.uint8
        assert uint8_image.min() >= 0
        assert uint8_image.max() <= 255
