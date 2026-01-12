"""Pytest fixtures for JAX performance tests.

Provides fixtures that ensure JAX backend is active for JIT tests.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def jax_backend_active(monkeypatch):
    """Ensure JAX backend is active for the test.

    Sets environment variables and resets the backend to use JAX.
    Restores NumPy backend after the test.
    """
    pytest.importorskip("jax")

    # Set environment to use JAX
    monkeypatch.setenv("XPCS_USE_JAX", "1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    from xpcsviewer.backends import _reset_backend, get_backend, set_backend

    # Reset and switch to JAX
    _reset_backend()
    set_backend("jax")

    backend = get_backend()
    assert backend.name == "jax", f"Expected JAX backend, got {backend.name}"

    yield backend

    # Cleanup: reset to NumPy
    _reset_backend()
    set_backend("numpy")
