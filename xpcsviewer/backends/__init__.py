"""Backend abstraction layer for JAX/NumPy array operations.

This module provides a unified interface for array computations that can
run on either JAX (with GPU support) or NumPy (CPU fallback).

Public API:
    get_backend() -> BackendProtocol
    set_backend(name: str) -> None
    BackendProtocol
    ensure_numpy(array) -> np.ndarray
    ensure_backend_array(array) -> BackendArray

Environment Variables:
    XPCS_USE_JAX: 'true', 'false', or 'auto' (default: 'auto')
    XPCS_USE_GPU: 'true' or 'false' (default: 'false')
    XPCS_GPU_FALLBACK: 'true' or 'false' (default: 'true')
    XPCS_GPU_MEMORY_FRACTION: float 0.0-1.0 (default: 0.9)
"""

from __future__ import annotations

import os

# Ensure float64 is enabled before any JAX import elsewhere in the codebase.
# This must happen at module scope (import time) so that lazy JIT decoration
# in other modules picks up the correct precision setting.
os.environ.setdefault("JAX_ENABLE_X64", "true")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import BackendProtocol

# Module-level state
_current_backend: BackendProtocol | None = None
_jax_configured: bool = False


def _configure_jax() -> None:
    """Configure JAX settings (float64, memory) on first use."""
    global _jax_configured
    if _jax_configured:
        return

    try:
        # Enable float64 for scientific computing precision
        os.environ.setdefault("JAX_ENABLE_X64", "true")

        import jax

        jax.config.update("jax_enable_x64", True)

        # Configure GPU memory fraction if specified
        try:
            memory_fraction = float(os.environ.get("XPCS_GPU_MEMORY_FRACTION", "0.9"))
            if 0.0 < memory_fraction < 1.0:
                os.environ.setdefault(
                    "XLA_PYTHON_CLIENT_MEM_FRACTION", str(memory_fraction)
                )
        except ValueError:
            # Handle invalid float gracefully (e.g. if set to "invalid" in tests)
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Invalid XPCS_GPU_MEMORY_FRACTION value: '{os.environ.get('XPCS_GPU_MEMORY_FRACTION')}', defaulting to 0.9"
            )
            os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

        _jax_configured = True
    except ImportError:
        pass  # JAX not available


def _detect_backend() -> str:
    """Detect which backend to use based on environment and availability."""
    use_jax = os.environ.get("XPCS_USE_JAX", "auto").lower()

    if use_jax == "false":
        return "numpy"

    if use_jax == "true":
        try:
            _configure_jax()
            import jax

            return "jax"
        except ImportError as e:
            raise ImportError(
                "JAX requested but not installed. "
                "Install with: pip install 'xpcsviewer-gui[jax]'"
            ) from e

    # Auto-detect
    try:
        _configure_jax()
        import jax

        return "jax"
    except ImportError:
        return "numpy"


def get_backend() -> BackendProtocol:
    """Get the current computation backend.

    Returns the JAX backend if available and configured, otherwise
    falls back to NumPy.

    Returns
    -------
    BackendProtocol
        The active backend instance.
    """
    global _current_backend

    if _current_backend is None:
        backend_name = _detect_backend()
        set_backend(backend_name)

    return _current_backend  # type: ignore[return-value]


def set_backend(name: str) -> None:
    """Set the computation backend.

    Parameters
    ----------
    name : str
        Backend name: 'jax' or 'numpy'

    Raises
    ------
    ValueError
        If backend name is not recognized.
    ImportError
        If JAX backend is requested but not available.
    """
    global _current_backend

    name = name.lower()

    if name == "jax":
        _configure_jax()
        from ._jax_backend import JAXBackend

        _current_backend = JAXBackend()  # type: ignore[assignment]
    elif name == "numpy":
        from ._numpy_backend import NumPyBackend

        _current_backend = NumPyBackend()  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'jax' or 'numpy'.")


def reset_backend() -> None:
    """Reset backend to trigger re-detection on next get_backend() call.

    Also resets legacy fitting closures that capture a stale backend (JAX-N-07).
    """
    global _current_backend, _jax_configured
    _current_backend = None
    _jax_configured = False

    # Invalidate legacy closures that captured the old backend (JAX-N-07).
    # Import is deferred to avoid circular import at module load time.
    try:
        from xpcsviewer.fitting.legacy import reset_legacy_closures

        reset_legacy_closures()
    except ImportError:
        pass  # fitting module may not be installed


# Alias for testing
_reset_backend = reset_backend


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse boolean environment variable.

    Accepts: 'true', '1', 'yes' (case-insensitive) for True
             'false', '0', 'no' (case-insensitive) for False

    Parameters
    ----------
    name : str
        Environment variable name
    default : bool
        Default value if not set or invalid

    Returns
    -------
    bool
        Parsed boolean value
    """
    value = os.environ.get(name, "").lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return default


# Convenience re-exports
from ._base import BackendProtocol
from ._conversions import ensure_backend_array, ensure_numpy

__all__ = [
    "BackendProtocol",
    "ensure_backend_array",
    "ensure_numpy",
    "get_backend",
    "reset_backend",
    "set_backend",
]
