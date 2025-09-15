"""
Specialized caching for computation results in XPCS Toolkit.

This module provides optimized caching for expensive computations like G2 fitting,
SAXS analysis, and two-time correlation calculations.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .advanced_cache import get_global_cache
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class G2FitResult:
    """G2 fitting result with metadata."""

    fit_func: str
    fit_val: np.ndarray
    t_el: np.ndarray
    q_val: np.ndarray
    q_range: str
    t_range: str
    bounds: list
    fit_flag: str
    fit_line: list
    label: list
    computation_time_ms: float
    cache_key: str


@dataclass
class SAXSResult:
    """SAXS analysis result with metadata."""

    q: np.ndarray
    intensity: np.ndarray
    xlabel: str
    ylabel: str
    processing_params: dict
    computation_time_ms: float
    cache_key: str


@dataclass
class TwoTimeResult:
    """Two-time correlation result with metadata."""

    c2_data: np.ndarray
    t0: float
    selection: int
    max_size: int
    correct_diag: bool
    computation_time_ms: float
    cache_key: str


class ComputationCache:
    """
    Specialized cache for expensive XPCS computations with parameter tracking.
    """

    def __init__(self, max_entries_per_type: int = 100):
        self._cache = get_global_cache()
        self._max_entries_per_type = max_entries_per_type
        self._lock = threading.RLock()

        # Track computation types and their keys
        self._computation_keys: dict[str, list[str]] = {
            "g2_fitting": [],
            "saxs_analysis": [],
            "twotime_correlation": [],
            "qmap_calculation": [],
            "metadata_extraction": [],
        }

    def _generate_parameter_key(self, computation_type: str, **params) -> str:
        """Generate cache key from computation parameters."""
        key_parts = [computation_type]

        # Sort parameters for consistent keys
        for param_name, param_value in sorted(params.items()):
            if isinstance(param_value, np.ndarray):
                # For arrays, use shape, dtype, and hash of sample values
                if param_value.size > 1000:
                    sample = param_value.flat[
                        :: param_value.size // 100
                    ]  # Sample 100 values
                else:
                    sample = param_value.flatten()
                array_key = f"{param_name}:{param_value.shape}:{param_value.dtype}:{hash(sample.tobytes())}"
                key_parts.append(array_key)
            elif isinstance(param_value, (list, tuple)):
                key_parts.append(
                    f"{param_name}:{sorted(param_value) if isinstance(param_value, list) else param_value!s}"
                )
            elif isinstance(param_value, dict):
                key_parts.append(f"{param_name}:{sorted(param_value.items())!s}")
            else:
                key_parts.append(f"{param_name}:{param_value}")

        # Create hash from concatenated parts
        cache_string = "|".join(key_parts)
        return hashlib.sha256(cache_string.encode()).hexdigest()[
            :24
        ]  # Use 24 chars for readability

    def _manage_computation_keys(self, computation_type: str, key: str):
        """Manage computation keys and enforce limits."""
        with self._lock:
            if key not in self._computation_keys[computation_type]:
                self._computation_keys[computation_type].append(key)

                # Enforce maximum entries per type
                while (
                    len(self._computation_keys[computation_type])
                    > self._max_entries_per_type
                ):
                    old_key = self._computation_keys[computation_type].pop(0)
                    self._cache.invalidate(old_key)
                    logger.debug(
                        f"Evicted old {computation_type} cache entry: {old_key}"
                    )

    def cache_g2_fitting(
        self,
        file_path: str,
        q_range: tuple[float, float] | None = None,
        t_range: tuple[float, float] | None = None,
        bounds: list | None = None,
        fit_flag: list | None = None,
        fit_func: str = "single",
    ) -> Callable:
        """
        Cache decorator for G2 fitting computations.

        Parameters
        ----------
        file_path : str
            Path to XPCS file
        q_range : tuple, optional
            Q range for fitting
        t_range : tuple, optional
            Time range for fitting
        bounds : list, optional
            Fitting bounds
        fit_flag : list, optional
            Fitting flags
        fit_func : str
            Fitting function type

        Returns
        -------
        callable
            Cached function decorator
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()

                # Generate cache key from parameters
                cache_key = self._generate_parameter_key(
                    "g2_fitting",
                    file_path=file_path,
                    q_range=q_range,
                    t_range=t_range,
                    bounds=bounds,
                    fit_flag=fit_flag,
                    fit_func=fit_func,
                    **kwargs,
                )

                # Try to get from cache
                cached_result, found = self._cache.get(cache_key)
                if found and isinstance(cached_result, G2FitResult):
                    logger.debug(f"G2 fitting cache hit: {cache_key}")
                    return cached_result

                # Compute result
                result = func(*args, **kwargs)

                # Wrap result with metadata
                computation_time_ms = (time.time() - start_time) * 1000.0
                cached_result = G2FitResult(
                    fit_func=result.get("fit_func", fit_func),
                    fit_val=result.get("fit_val"),
                    t_el=result.get("t_el"),
                    q_val=result.get("q_val"),
                    q_range=str(q_range),
                    t_range=str(t_range),
                    bounds=bounds,
                    fit_flag=str(fit_flag),
                    fit_line=result.get("fit_line"),
                    label=result.get("label"),
                    computation_time_ms=computation_time_ms,
                    cache_key=cache_key,
                )

                # Cache result with appropriate TTL based on computation time
                ttl_seconds = max(
                    3600, computation_time_ms * 10
                )  # At least 1 hour, longer for expensive computations
                self._cache.put(cache_key, cached_result, ttl_seconds=ttl_seconds)
                self._manage_computation_keys("g2_fitting", cache_key)

                logger.debug(
                    f"G2 fitting result cached: {cache_key} (computation: {computation_time_ms:.1f}ms)"
                )
                return cached_result

            return wrapper

        return decorator

    def cache_saxs_analysis(self, file_path: str, processing_params: dict) -> Callable:
        """
        Cache decorator for SAXS analysis computations.

        Parameters
        ----------
        file_path : str
            Path to XPCS file
        processing_params : dict
            SAXS processing parameters

        Returns
        -------
        callable
            Cached function decorator
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()

                # Generate cache key from parameters
                cache_key = self._generate_parameter_key(
                    "saxs_analysis",
                    file_path=file_path,
                    processing_params=processing_params,
                    **kwargs,
                )

                # Try to get from cache
                cached_result, found = self._cache.get(cache_key)
                if found and isinstance(cached_result, SAXSResult):
                    logger.debug(f"SAXS analysis cache hit: {cache_key}")
                    return cached_result

                # Compute result
                q, intensity, xlabel, ylabel = func(*args, **kwargs)

                # Wrap result with metadata
                computation_time_ms = (time.time() - start_time) * 1000.0
                cached_result = SAXSResult(
                    q=q,
                    intensity=intensity,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    processing_params=processing_params,
                    computation_time_ms=computation_time_ms,
                    cache_key=cache_key,
                )

                # Cache result with TTL based on data size and computation time
                data_size_mb = (q.nbytes + intensity.nbytes) / (1024 * 1024)
                ttl_seconds = max(
                    1800, min(7200, data_size_mb * 100)
                )  # 30min to 2 hours based on size
                self._cache.put(cache_key, cached_result, ttl_seconds=ttl_seconds)
                self._manage_computation_keys("saxs_analysis", cache_key)

                logger.debug(
                    f"SAXS analysis result cached: {cache_key} (computation: {computation_time_ms:.1f}ms)"
                )
                return cached_result

            return wrapper

        return decorator

    def cache_twotime_correlation(
        self, file_path: str, selection: int, max_size: int, correct_diag: bool = True
    ) -> Callable:
        """
        Cache decorator for two-time correlation computations.

        Parameters
        ----------
        file_path : str
            Path to XPCS file
        selection : int
            Q-bin selection
        max_size : int
            Maximum correlation matrix size
        correct_diag : bool
            Whether to correct diagonal

        Returns
        -------
        callable
            Cached function decorator
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()

                # Generate cache key from parameters
                cache_key = self._generate_parameter_key(
                    "twotime_correlation",
                    file_path=file_path,
                    selection=selection,
                    max_size=max_size,
                    correct_diag=correct_diag,
                    **kwargs,
                )

                # Try to get from cache
                cached_result, found = self._cache.get(cache_key)
                if found and isinstance(cached_result, TwoTimeResult):
                    logger.debug(f"Two-time correlation cache hit: {cache_key}")
                    return cached_result

                # Compute result
                result = func(*args, **kwargs)

                # Wrap result with metadata
                computation_time_ms = (time.time() - start_time) * 1000.0
                cached_result = TwoTimeResult(
                    c2_data=result,
                    t0=kwargs.get("t0", 0.0),
                    selection=selection,
                    max_size=max_size,
                    correct_diag=correct_diag,
                    computation_time_ms=computation_time_ms,
                    cache_key=cache_key,
                )

                # Two-time correlations are expensive and should be cached longer
                data_size_mb = (
                    result.nbytes / (1024 * 1024) if hasattr(result, "nbytes") else 0
                )
                ttl_seconds = max(
                    7200, computation_time_ms * 20
                )  # At least 2 hours, longer for expensive computations
                self._cache.put(cache_key, cached_result, ttl_seconds=ttl_seconds)
                self._manage_computation_keys("twotime_correlation", cache_key)

                logger.info(
                    f"Two-time correlation cached: {cache_key} (computation: {computation_time_ms:.1f}ms, size: {data_size_mb:.1f}MB)"
                )
                return cached_result

            return wrapper

        return decorator

    def get_cached_g2_fitting(self, file_path: str, **params) -> G2FitResult | None:
        """Get cached G2 fitting result if available."""
        cache_key = self._generate_parameter_key(
            "g2_fitting", file_path=file_path, **params
        )
        result, found = self._cache.get(cache_key)
        return result if found and isinstance(result, G2FitResult) else None

    def get_cached_saxs_analysis(self, file_path: str, **params) -> SAXSResult | None:
        """Get cached SAXS analysis result if available."""
        cache_key = self._generate_parameter_key(
            "saxs_analysis", file_path=file_path, **params
        )
        result, found = self._cache.get(cache_key)
        return result if found and isinstance(result, SAXSResult) else None

    def get_cached_twotime_correlation(
        self, file_path: str, **params
    ) -> TwoTimeResult | None:
        """Get cached two-time correlation result if available."""
        cache_key = self._generate_parameter_key(
            "twotime_correlation", file_path=file_path, **params
        )
        result, found = self._cache.get(cache_key)
        return result if found and isinstance(result, TwoTimeResult) else None

    def invalidate_file_cache(self, file_path: str):
        """Invalidate all cached computations for a specific file."""
        invalidated_count = 0

        with self._lock:
            for computation_type, keys in self._computation_keys.items():
                keys_to_remove = []
                for key in keys:
                    # Check if this key belongs to the file (simple string matching)
                    test_key = self._generate_parameter_key(
                        computation_type, file_path=file_path
                    )
                    if key.startswith(test_key[:16]):  # Match file portion of key
                        if self._cache.invalidate(key):
                            invalidated_count += 1
                            keys_to_remove.append(key)

                # Remove invalidated keys from tracking
                for key in keys_to_remove:
                    keys.remove(key)

        if invalidated_count > 0:
            logger.info(
                f"Invalidated {invalidated_count} cached computations for {file_path}"
            )

    def get_computation_stats(self) -> dict[str, Any]:
        """Get statistics about cached computations."""
        with self._lock:
            stats = {
                "computation_counts": {
                    comp_type: len(keys)
                    for comp_type, keys in self._computation_keys.items()
                },
                "total_computations": sum(
                    len(keys) for keys in self._computation_keys.values()
                ),
                "cache_stats": self._cache.get_stats(),
            }

            # Add performance estimates
            cache_stats = stats["cache_stats"]
            hit_rates = cache_stats.get("hit_rates", {})
            overall_hit_rate = hit_rates.get("overall_hit_rate", 0.0)

            # Estimate time savings based on average computation times
            avg_g2_time_ms = 5000  # Typical G2 fitting time
            avg_saxs_time_ms = 1000  # Typical SAXS analysis time
            avg_twotime_time_ms = 15000  # Typical two-time correlation time

            estimated_time_saved_ms = (
                len(self._computation_keys["g2_fitting"])
                * avg_g2_time_ms
                * overall_hit_rate
                + len(self._computation_keys["saxs_analysis"])
                * avg_saxs_time_ms
                * overall_hit_rate
                + len(self._computation_keys["twotime_correlation"])
                * avg_twotime_time_ms
                * overall_hit_rate
            )

            stats["performance_estimates"] = {
                "estimated_time_saved_seconds": estimated_time_saved_ms / 1000.0,
                "estimated_time_saved_hours": estimated_time_saved_ms
                / (1000.0 * 3600.0),
            }

            return stats

    def cleanup_old_computations(self, max_age_hours: float = 24.0):
        """Clean up old cached computations."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        removed_count = 0
        with self._lock:
            for _computation_type, keys in self._computation_keys.items():
                keys_to_remove = []

                for key in keys:
                    cached_result, found = self._cache.get(key)
                    if found:
                        # Check creation time based on result type
                        created_time = None
                        if hasattr(cached_result, "computation_time_ms"):
                            # Estimate creation time (this is approximate)
                            created_time = current_time - (
                                cached_result.computation_time_ms / 1000.0
                            )

                        if created_time and created_time < cutoff_time:
                            if self._cache.invalidate(key):
                                removed_count += 1
                                keys_to_remove.append(key)
                    else:
                        # Key not found in cache, remove from tracking
                        keys_to_remove.append(key)

                # Remove cleaned keys from tracking
                for key in keys_to_remove:
                    keys.remove(key)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cached computations")

        return removed_count


# Global computation cache instance
_global_computation_cache: ComputationCache | None = None


def get_computation_cache() -> ComputationCache:
    """Get global computation cache instance."""
    global _global_computation_cache

    if _global_computation_cache is None:
        _global_computation_cache = ComputationCache()

    return _global_computation_cache


# Convenience decorators using global cache
def cache_g2_fitting(**params):
    """Convenience decorator for G2 fitting caching."""
    return get_computation_cache().cache_g2_fitting(**params)


def cache_saxs_analysis(**params):
    """Convenience decorator for SAXS analysis caching."""
    return get_computation_cache().cache_saxs_analysis(**params)


def cache_twotime_correlation(**params):
    """Convenience decorator for two-time correlation caching."""
    return get_computation_cache().cache_twotime_correlation(**params)
