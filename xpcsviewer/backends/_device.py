"""Device management for JAX CPU/GPU selection.

This module provides the DeviceManager singleton for configuring and
managing compute devices, along with DeviceConfig and DeviceType.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Enumeration of supported device types."""

    CPU = "cpu"
    GPU = "gpu"


@dataclass
class DeviceConfig:
    """Configuration for device selection.

    Attributes
    ----------
    preferred_device : DeviceType
        Preferred compute device (default: CPU)
    allow_gpu_fallback : bool
        Allow fallback to CPU if GPU is unavailable (default: True)
    memory_fraction : float
        Maximum fraction of GPU memory to use (default: 0.9)
    """

    preferred_device: DeviceType = DeviceType.CPU
    allow_gpu_fallback: bool = True
    memory_fraction: float = 0.9

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
            )

    @classmethod
    def from_environment(cls) -> DeviceConfig:
        """Create configuration from environment variables.

        Environment Variables
        ---------------------
        XPCS_USE_GPU : str
            'true' or 'false' (default: 'false')
        XPCS_GPU_FALLBACK : str
            'true' or 'false' (default: 'true')
        XPCS_GPU_MEMORY_FRACTION : str
            Float value 0.0-1.0 (default: '0.9')
        """
        use_gpu = os.environ.get("XPCS_USE_GPU", "false").lower() == "true"
        allow_fallback = os.environ.get("XPCS_GPU_FALLBACK", "true").lower() == "true"
        memory_fraction = float(os.environ.get("XPCS_GPU_MEMORY_FRACTION", "0.9"))

        return cls(
            preferred_device=DeviceType.GPU if use_gpu else DeviceType.CPU,
            allow_gpu_fallback=allow_fallback,
            memory_fraction=memory_fraction,
        )


@dataclass
class DeviceInfo:
    """Information about a compute device.

    Attributes
    ----------
    device_type : DeviceType
        Type of device (CPU or GPU)
    device_id : int
        Device ID (0 for CPU, GPU index for GPU)
    name : str
        Human-readable device name
    memory_total : int | None
        Total memory in bytes (GPU only)
    memory_available : int | None
        Available memory in bytes (GPU only)
    """

    device_type: DeviceType
    device_id: int = 0
    name: str = ""
    memory_total: int | None = None
    memory_available: int | None = None


class DeviceManager:
    """Singleton manager for compute device selection and placement.

    This class provides centralized management of device selection,
    including automatic fallback from GPU to CPU when needed.

    Examples
    --------
    >>> manager = DeviceManager()
    >>> manager.configure(DeviceConfig(preferred_device=DeviceType.GPU))
    >>> if manager.is_gpu_enabled:
    ...     print("Using GPU")
    """

    _instance: DeviceManager | None = None
    _lock = threading.RLock()
    _initialized: bool

    def __new__(cls) -> DeviceManager:
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize device manager (only runs once)."""
        if self._initialized:
            return

        self._config: DeviceConfig = DeviceConfig.from_environment()
        self._current_device: DeviceInfo | None = None
        self._jax_available: bool | None = None
        self._gpu_available: bool | None = None
        self._initialized = True

        # Auto-configure based on environment
        self._auto_configure()

    def _auto_configure(self) -> None:
        """Auto-configure device based on environment settings."""
        try:
            self.configure(self._config)
        except RuntimeError as e:
            logger.warning(f"Device auto-configuration failed: {e}")

    def configure(self, config: DeviceConfig) -> None:
        """Configure device manager with specified settings.

        Parameters
        ----------
        config : DeviceConfig
            Device configuration

        Raises
        ------
        RuntimeError
            If GPU is requested but unavailable and fallback is disabled
        """
        self._config = config

        if config.preferred_device == DeviceType.GPU:
            if self.gpu_available:
                self._setup_gpu(config.memory_fraction)
                self._current_device = DeviceInfo(
                    device_type=DeviceType.GPU,
                    device_id=0,
                    name=self._get_gpu_name(),
                )
                logger.info(f"Using GPU: {self._current_device.name}")
            elif config.allow_gpu_fallback:
                logger.warning("GPU requested but not available, falling back to CPU")
                self._current_device = DeviceInfo(
                    device_type=DeviceType.CPU,
                    device_id=0,
                    name="CPU",
                )
            else:
                raise RuntimeError(
                    "GPU requested but not available, and fallback is disabled. "
                    "Install JAX GPU support with: make install-jax-gpu"
                )
        else:
            self._current_device = DeviceInfo(
                device_type=DeviceType.CPU,
                device_id=0,
                name="CPU",
            )
            logger.info("Using CPU")

    def _setup_gpu(self, memory_fraction: float) -> None:
        """Configure GPU memory settings.

        Parameters
        ----------
        memory_fraction : float
            Maximum fraction of GPU memory to use
        """
        if not self.jax_available:
            return

        # Set memory fraction via environment variable
        # This must be done before JAX initializes the GPU
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

    def _get_gpu_name(self) -> str:
        """Get the name of the first available GPU."""
        if not self.jax_available:
            return "Unknown GPU"

        try:
            import jax

            devices = jax.devices("gpu")
            if devices:
                return str(devices[0])
            return "Unknown GPU"
        except RuntimeError:
            return "Unknown GPU"

    @property
    def jax_available(self) -> bool:
        """Check if JAX is installed and available."""
        if self._jax_available is None:
            try:
                import jax  # noqa: F401

                self._jax_available = True
            except ImportError:
                self._jax_available = False
        return self._jax_available

    @property
    def gpu_available(self) -> bool:
        """Check if GPU devices are available."""
        if self._gpu_available is None:
            if not self.jax_available:
                self._gpu_available = False
            else:
                try:
                    import jax

                    devices = jax.devices("gpu")
                    self._gpu_available = len(devices) > 0
                except RuntimeError:
                    self._gpu_available = False
        return self._gpu_available

    @property
    def is_gpu_enabled(self) -> bool:
        """Check if GPU is currently enabled."""
        if self._current_device is None:
            return False
        return self._current_device.device_type == DeviceType.GPU

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available (alias for gpu_available)."""
        return self.gpu_available

    @property
    def available_devices(self) -> list:
        """Get list of available compute devices.

        Returns
        -------
        list
            List of JAX device objects, or empty list if JAX unavailable
        """
        if not self.jax_available:
            return []

        try:
            import jax

            return list(jax.devices())
        except Exception:
            return []

    @property
    def config(self) -> DeviceConfig:
        """Get current device configuration."""
        return self._config

    @property
    def current_device(self) -> DeviceInfo | None:
        """Get current device info."""
        return self._current_device

    def get_device(self) -> Any | None:
        """Get the current JAX device object.

        Returns
        -------
        jax.Device or None
            JAX device object, or None if JAX is not available
        """
        if not self.jax_available:
            return None

        import jax

        if self.is_gpu_enabled:
            devices = jax.devices("gpu")
            return devices[0] if devices else jax.devices("cpu")[0]
        return jax.devices("cpu")[0]

    def place_on_device(self, array: Any) -> Any:
        """Place array on the current device.

        Parameters
        ----------
        array : array-like
            Array to place on device

        Returns
        -------
        array
            Array on the appropriate device
        """
        if not self.jax_available:
            return array

        import jax

        # Log array info at entry if DEBUG enabled
        if logger.isEnabledFor(logging.DEBUG):
            shape = getattr(array, "shape", "N/A")
            dtype = getattr(array, "dtype", "N/A")
            logger.debug(f"place_on_device: shape={shape}, dtype={dtype}")

        device = self.get_device()
        if device is not None:
            result = jax.device_put(array, device)
            logger.debug(f"place_on_device: placed on {device}")
            return result
        return array

    def get_memory_info(self) -> dict[str, int | None]:
        """Get GPU memory information.

        Returns
        -------
        dict
            Dictionary with 'total' and 'available' memory in bytes,
            or None values if not on GPU
        """
        if not self.is_gpu_enabled or not self.jax_available:
            return {"total": None, "available": None}

        try:
            import jax

            devices = jax.devices("gpu")
            if devices:
                # JAX doesn't provide direct memory querying,
                # but we can get device info
                device = devices[0]
                logger.debug("GPU device found: %s (memory info unavailable)", device)
                # Memory info would require platform-specific APIs
                return {"total": None, "available": None}
        except Exception as e:
            logger.debug("Failed to query GPU memory info: %s", e)

        return {"total": None, "available": None}

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
                cls._instance = None
