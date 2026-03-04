"""GPU detection and warning utilities (System CUDA version).

This module provides utilities for detecting GPU hardware, system CUDA installation,
and JAX backend configuration. It helps users identify when GPU acceleration is
available but not being used.

Example usage:
    >>> from xpcsviewer.utils.device import check_gpu_availability, get_device_info
    >>> check_gpu_availability()  # Prints warning if GPU available but not used
    >>> info = get_device_info()  # Get comprehensive device information
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


def get_system_cuda_version() -> tuple[str | None, int | None]:
    """Detect system CUDA version from nvcc.

    Returns
    -------
    tuple[str | None, int | None]
        Tuple of (full_version, major_version) or (None, None) if not found.
        Example: ("12.6", 12) or ("13.0", 13)

    Examples
    --------
    >>> version, major = get_system_cuda_version()
    >>> if major is not None:
    ...     print(f"CUDA {version} detected")
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # Parse "release X.Y" from output
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # Extract version like "12.6" from "release 12.6, V12.6.77"
                    parts = line.split("release")[-1].strip()
                    version = parts.split(",")[0].strip()
                    major = int(version.split(".")[0])
                    return version, major

    except subprocess.TimeoutExpired:
        _logger.debug("nvcc timed out")
    except FileNotFoundError:
        _logger.debug("nvcc not found")
    except (ValueError, IndexError) as e:
        _logger.debug(f"Failed to parse CUDA version: {e}")
    except Exception as e:
        _logger.debug(f"CUDA detection failed: {e}")

    return None, None


def get_gpu_info() -> tuple[str | None, float | None]:
    """Detect GPU name and SM version.

    Returns
    -------
    tuple[str | None, float | None]
        Tuple of (gpu_name, sm_version) or (None, None) if not found.
        Example: ("NVIDIA GeForce RTX 4090", 8.9)

    Examples
    --------
    >>> name, sm = get_gpu_info()
    >>> if name is not None:
    ...     print(f"GPU: {name} (SM {sm})")
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            if len(parts) >= 2:
                gpu_name = parts[0]
                sm_version = float(parts[1])
                return gpu_name, sm_version

    except subprocess.TimeoutExpired:
        _logger.debug("nvidia-smi timed out")
    except FileNotFoundError:
        _logger.debug("nvidia-smi not found")
    except (ValueError, IndexError) as e:
        _logger.debug(f"Failed to parse GPU info: {e}")
    except Exception as e:
        _logger.debug(f"GPU detection failed: {e}")

    return None, None


def get_recommended_package() -> str | None:
    """Get recommended JAX package based on system CUDA.

    Returns
    -------
    str | None
        Package name like "jax[cuda12-local]" or "jax[cuda13-local]",
        or None if no compatible setup found.

    Examples
    --------
    >>> pkg = get_recommended_package()
    >>> if pkg:
    ...     print(f"Install with: pip install {pkg}")
    """
    cuda_version, cuda_major = get_system_cuda_version()
    gpu_name, sm_version = get_gpu_info()

    if cuda_major is None:
        _logger.debug("No system CUDA detected")
        return None

    if sm_version is None:
        _logger.debug("No GPU detected")
        return None

    # Check compatibility
    if cuda_major == 13:
        if sm_version >= 7.5:
            return "jax[cuda13-local]"
        _logger.debug(f"GPU SM {sm_version} doesn't support CUDA 13")
        return None
    if cuda_major == 12:
        if sm_version >= 5.2:
            return "jax[cuda12-local]"
        _logger.debug(f"GPU SM {sm_version} too old for CUDA 12")
        return None
    _logger.debug(f"CUDA {cuda_major} not supported")
    return None


def check_plugin_conflicts() -> list[str]:
    """Check for known JAX CUDA plugin conflicts.

    Returns
    -------
    list[str]
        List of issue descriptions (empty = no issues).

    Examples
    --------
    >>> issues = check_plugin_conflicts()
    >>> for issue in issues:
    ...     print(f"WARNING: {issue}")
    """
    issues = []
    try:
        import importlib.metadata as md

        jaxlib_v = md.version("jaxlib")

        cuda12 = cuda13 = None
        try:
            cuda12 = md.version("jax-cuda12-plugin")
        except md.PackageNotFoundError:
            pass
        try:
            cuda13 = md.version("jax-cuda13-plugin")
        except md.PackageNotFoundError:
            pass

        # Check for dual plugin conflict
        if cuda12 and cuda13:
            issues.append(
                f"Both cuda12 ({cuda12}) and cuda13 ({cuda13}) plugins installed. "
                "Only ONE can be active — this causes PJRT registration conflicts."
            )

        # Check for version mismatch
        for name, version in [("cuda12", cuda12), ("cuda13", cuda13)]:
            if version and version != jaxlib_v:
                issues.append(
                    f"jax-{name}-plugin {version} != jaxlib {jaxlib_v}. "
                    "Plugin version must exactly match jaxlib."
                )

    except Exception as e:
        _logger.debug(f"Plugin conflict check failed: {e}")

    return issues


def check_gpu_availability(warn: bool = True) -> bool:
    """Check if GPU is available but not being used by JAX.

    Prints a helpful warning if GPU hardware and system CUDA are detected
    but JAX is running in CPU-only mode.

    Parameters
    ----------
    warn : bool, optional
        If True, print warning when GPU available but not used, by default True.

    Returns
    -------
    bool
        True if GPU is being used by JAX, False otherwise.

    Examples
    --------
    >>> if not check_gpu_availability():
    ...     print("Consider enabling GPU acceleration")
    """
    try:
        gpu_name, sm_version = get_gpu_info()
        cuda_version, cuda_major = get_system_cuda_version()

        if gpu_name is None:
            _logger.debug("No GPU hardware detected")
            return False

        # Check if JAX is using GPU
        import jax

        devices = jax.devices()
        using_gpu = any("cuda" in str(d).lower() for d in devices)

        if using_gpu:
            _logger.debug(f"JAX is using GPU: {devices}")
            # Check for plugin issues even when GPU works
            issues = check_plugin_conflicts()
            for issue in issues:
                _logger.warning(f"Plugin issue: {issue}")
            return True

        # GPU available but not being used
        if warn:
            _print_gpu_warning(gpu_name, sm_version, cuda_version, cuda_major)

        return False

    except ImportError:
        _logger.debug("JAX not installed")
        return False
    except Exception as e:
        _logger.debug(f"GPU check failed: {e}")
        return False


def _print_gpu_warning(
    gpu_name: str,
    sm_version: float | None,
    cuda_version: str | None,
    cuda_major: int | None,
) -> None:
    """Print warning about GPU acceleration availability."""
    print("\nGPU ACCELERATION AVAILABLE")
    print("===========================")
    print(f"GPU: {gpu_name} (SM {sm_version})")
    print(f"System CUDA: {cuda_version or 'Not found'}")
    print("JAX backend: CPU-only")

    issues = check_plugin_conflicts()
    if issues:
        print("\nIssues detected:")
        for issue in issues:
            print(f"  - {issue}")

    print()

    if cuda_major is None:
        print("To enable GPU acceleration:")
        print("  1. Install CUDA toolkit (12.x or 13.x)")
        print("  2. Ensure nvcc is in PATH")
        print("  3. Run: make install-jax-gpu")
    else:
        pkg = f"jax[cuda{cuda_major}-local]"
        print("Enable 20-100x speedup:")
        print("  make install-jax-gpu")
        print()
        print("Or manually:")
        print(
            "  pip uninstall -y jax jaxlib "
            "jax-cuda13-plugin jax-cuda13-pjrt "
            "jax-cuda12-plugin jax-cuda12-pjrt"
        )
        print(f'  pip install "{pkg}"')

    print("\nSee README.rst for details.\n")


def get_device_info() -> dict:
    """Get comprehensive device information.

    Returns
    -------
    dict
        Dictionary containing:
        - jax_version: JAX version string
        - jax_backend: Current backend (cpu, gpu)
        - devices: List of device strings
        - gpu_count: Number of GPU devices
        - using_gpu: Boolean
        - gpu_hardware: GPU name
        - gpu_sm_version: SM version (float)
        - system_cuda_version: System CUDA version string
        - system_cuda_major: System CUDA major version (int)
        - recommended_package: Recommended JAX package
        - plugin_issues: List of detected plugin conflict descriptions

    Examples
    --------
    >>> info = get_device_info()
    >>> print(f"JAX backend: {info['jax_backend']}")
    >>> if info['using_gpu']:
    ...     print(f"Using {info['gpu_count']} GPU(s)")
    """
    info: dict = {
        "jax_version": None,
        "jax_backend": None,
        "devices": [],
        "gpu_count": 0,
        "using_gpu": False,
        "gpu_hardware": None,
        "gpu_sm_version": None,
        "system_cuda_version": None,
        "system_cuda_major": None,
        "recommended_package": None,
        "plugin_issues": [],
    }

    # JAX info
    try:
        import jax

        info["jax_version"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        devices = jax.devices()
        info["devices"] = [str(d) for d in devices]
        info["gpu_count"] = sum(1 for d in devices if "cuda" in str(d).lower())
        info["using_gpu"] = info["gpu_count"] > 0
    except ImportError:
        pass

    # GPU hardware info
    gpu_name, sm_version = get_gpu_info()
    info["gpu_hardware"] = gpu_name
    info["gpu_sm_version"] = sm_version

    # System CUDA info
    cuda_version, cuda_major = get_system_cuda_version()
    info["system_cuda_version"] = cuda_version
    info["system_cuda_major"] = cuda_major

    # Recommended package and plugin health
    info["recommended_package"] = get_recommended_package()
    info["plugin_issues"] = check_plugin_conflicts()

    return info
