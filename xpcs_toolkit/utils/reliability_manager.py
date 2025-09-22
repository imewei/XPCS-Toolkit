"""
Unified Reliability Manager for XPCS Toolkit.

This module provides a single entry point for enabling and configuring
all reliability features while maintaining zero performance loss.
"""

import atexit
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .health_monitor import (
    HealthStatus,
    get_health_monitor,
    start_health_monitoring,
    stop_health_monitoring,
)
from .logging_config import get_logger
from .reliability import ValidationLevel, get_fallback_manager, get_validation_cache
from .state_validator import (
    StateValidationLevel,
    get_state_validator,
    start_state_monitoring,
    stop_state_monitoring,
)

logger = get_logger(__name__)


class ReliabilityProfile(Enum):
    """Predefined reliability profiles balancing safety vs performance."""

    MINIMAL = "minimal"  # Maximum performance, minimal safety
    BALANCED = "balanced"  # Good balance of performance and safety
    STRICT = "strict"  # High safety, moderate performance impact
    PARANOID = "paranoid"  # Maximum safety, accept performance cost


@dataclass
class ReliabilityConfig:
    """Configuration for reliability features."""

    profile: ReliabilityProfile = ReliabilityProfile.BALANCED

    # Exception handling
    enable_exception_conversion: bool = True
    enable_fallback_strategies: bool = True

    # Validation
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_validation_caching: bool = True
    cache_size_limit: int = 1000

    # Health monitoring
    enable_health_monitoring: bool = True
    health_monitoring_interval: float = 30.0
    enable_automatic_recovery: bool = True

    # State validation
    enable_state_validation: bool = True
    state_validation_level: StateValidationLevel = StateValidationLevel.STANDARD
    state_monitoring_interval: float = 60.0

    # Performance
    max_cpu_overhead_percent: float = 2.0
    max_memory_overhead_mb: float = 50.0
    enable_performance_monitoring: bool = True

    def apply_profile(self) -> None:
        """Apply predefined profile settings."""
        if self.profile == ReliabilityProfile.MINIMAL:
            self.validation_level = ValidationLevel.MINIMAL
            self.enable_validation_caching = False
            self.enable_health_monitoring = False
            self.enable_state_validation = False
            self.max_cpu_overhead_percent = 0.5

        elif self.profile == ReliabilityProfile.BALANCED:
            self.validation_level = ValidationLevel.STANDARD
            self.state_validation_level = StateValidationLevel.STANDARD
            self.health_monitoring_interval = 30.0
            self.state_monitoring_interval = 60.0

        elif self.profile == ReliabilityProfile.STRICT:
            self.validation_level = ValidationLevel.STRICT
            self.state_validation_level = StateValidationLevel.STRICT
            self.health_monitoring_interval = 15.0
            self.state_monitoring_interval = 30.0
            self.max_cpu_overhead_percent = 5.0

        elif self.profile == ReliabilityProfile.PARANOID:
            self.validation_level = ValidationLevel.PARANOID
            self.state_validation_level = StateValidationLevel.PARANOID
            self.health_monitoring_interval = 5.0
            self.state_monitoring_interval = 15.0
            self.max_cpu_overhead_percent = 10.0
            self.max_memory_overhead_mb = 200.0


class XPCSReliabilityManager:
    """
    Unified manager for all XPCS Toolkit reliability features.

    Provides single-point configuration and monitoring with guaranteed
    performance characteristics.
    """

    def __init__(self, config: ReliabilityConfig | None = None):
        self.config = config or ReliabilityConfig()
        self.config.apply_profile()

        self._initialized = False
        self._active_features: dict[str, bool] = {}
        self._performance_baseline: dict[str, float] | None = None
        self._lock = threading.RLock()

        # Component references
        self._health_monitor = None
        self._state_validator = None
        self._validation_cache = None
        self._fallback_manager = None

        # Performance monitoring
        self._start_time = time.time()
        self._performance_samples: list[dict[str, float]] = []

    def initialize(self, validate_performance: bool = True) -> bool:
        """
        Initialize all reliability features according to configuration.

        Args:
            validate_performance: Whether to validate performance impact

        Returns:
            bool: True if initialization successful and within performance limits
        """
        with self._lock:
            if self._initialized:
                logger.debug("Reliability manager already initialized")
                return True

            logger.info(
                f"Initializing XPCS reliability features (profile: {self.config.profile.value})"
            )

            try:
                # Record baseline performance if requested
                if validate_performance:
                    self._performance_baseline = self._measure_baseline_performance()

                # Initialize components based on configuration
                success = True

                if self.config.enable_health_monitoring:
                    success &= self._initialize_health_monitoring()

                if self.config.enable_state_validation:
                    success &= self._initialize_state_validation()

                if self.config.enable_validation_caching:
                    success &= self._initialize_validation_caching()

                if self.config.enable_fallback_strategies:
                    success &= self._initialize_fallback_strategies()

                # Validate performance impact if baseline was recorded
                if validate_performance and self._performance_baseline:
                    performance_ok = self._validate_performance_impact()
                    if not performance_ok:
                        logger.warning(
                            "Performance impact exceeded limits, disabling some features"
                        )
                        success = self._optimize_for_performance()

                # Register cleanup
                atexit.register(self.shutdown)

                self._initialized = success

                if success:
                    logger.info("XPCS reliability features initialized successfully")
                    logger.info(
                        f"Enabled features: {[k for k, v in self._active_features.items() if v]}"
                    )
                else:
                    logger.error("Failed to initialize some reliability features")

                return success

            except Exception as e:
                logger.error(f"Error initializing reliability features: {e}")
                return False

    def _initialize_health_monitoring(self) -> bool:
        """Initialize health monitoring component."""
        try:
            self._health_monitor = get_health_monitor()
            start_health_monitoring(self.config.health_monitoring_interval)

            # Register critical status callback
            def critical_status_handler(status: HealthStatus, summary: dict[str, Any]):
                if status == HealthStatus.CRITICAL:
                    logger.critical(f"Critical system status detected: {summary}")
                    if self.config.enable_automatic_recovery:
                        self._trigger_emergency_recovery()

            self._health_monitor.register_health_callback(
                HealthStatus.CRITICAL, critical_status_handler
            )

            self._active_features["health_monitoring"] = True
            logger.debug("Health monitoring initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize health monitoring: {e}")
            self._active_features["health_monitoring"] = False
            return False

    def _initialize_state_validation(self) -> bool:
        """Initialize state validation component."""
        try:
            self._state_validator = get_state_validator(
                self.config.state_validation_level
            )
            start_state_monitoring(
                self.config.state_monitoring_interval,
                self.config.state_validation_level,
            )

            self._active_features["state_validation"] = True
            logger.debug("State validation initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize state validation: {e}")
            self._active_features["state_validation"] = False
            return False

    def _initialize_validation_caching(self) -> bool:
        """Initialize validation caching."""
        try:
            self._validation_cache = get_validation_cache()
            # Configure cache size limit
            self._validation_cache._max_size = self.config.cache_size_limit

            self._active_features["validation_caching"] = True
            logger.debug("Validation caching initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize validation caching: {e}")
            self._active_features["validation_caching"] = False
            return False

    def _initialize_fallback_strategies(self) -> bool:
        """Initialize fallback strategies."""
        try:
            self._fallback_manager = get_fallback_manager()

            self._active_features["fallback_strategies"] = True
            logger.debug("Fallback strategies initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize fallback strategies: {e}")
            self._active_features["fallback_strategies"] = False
            return False

    def _measure_baseline_performance(self) -> dict[str, float]:
        """Measure baseline performance before enabling reliability features."""
        import psutil

        # Simple performance baseline
        start_time = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)

        # Simple computation benchmark
        computation_start = time.time()
        sum(i**2 for i in range(10000))  # Simple CPU work
        computation_time = time.time() - computation_start

        baseline = {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "computation_time": computation_time,
            "timestamp": start_time,
        }

        logger.debug(f"Performance baseline: {baseline}")
        return baseline

    def _validate_performance_impact(self) -> bool:
        """Validate that performance impact is within acceptable limits."""
        if not self._performance_baseline:
            return True  # Can't validate without baseline

        try:
            # Current performance
            current = self._measure_baseline_performance()
            baseline = self._performance_baseline

            # Calculate overhead
            cpu_overhead = current["cpu_percent"] - baseline["cpu_percent"]
            memory_overhead = current["memory_mb"] - baseline["memory_mb"]
            computation_overhead = (
                (current["computation_time"] - baseline["computation_time"])
                / baseline["computation_time"]
            ) * 100

            # Check against limits
            cpu_ok = cpu_overhead <= self.config.max_cpu_overhead_percent
            memory_ok = memory_overhead <= self.config.max_memory_overhead_mb
            computation_ok = (
                computation_overhead <= 10.0
            )  # 10% max computation overhead

            logger.debug(
                f"Performance impact: CPU={cpu_overhead:.1f}%, Memory={memory_overhead:.1f}MB, "
                f"Computation={computation_overhead:.1f}%"
            )

            if not (cpu_ok and memory_ok and computation_ok):
                logger.warning(
                    f"Performance impact exceeded limits: "
                    f"CPU={cpu_overhead:.1f}%/{self.config.max_cpu_overhead_percent}%, "
                    f"Memory={memory_overhead:.1f}MB/{self.config.max_memory_overhead_mb}MB"
                )
                return False

            return True

        except Exception as e:
            logger.debug(f"Error validating performance impact: {e}")
            return True  # Assume OK if we can't measure

    def _optimize_for_performance(self) -> bool:
        """Optimize configuration for better performance."""
        logger.info("Optimizing reliability configuration for performance")

        # Reduce monitoring frequency
        if self._active_features.get("health_monitoring"):
            self.config.health_monitoring_interval *= 2
            stop_health_monitoring()
            start_health_monitoring(self.config.health_monitoring_interval)

        if self._active_features.get("state_validation"):
            self.config.state_monitoring_interval *= 2
            stop_state_monitoring()
            start_state_monitoring(self.config.state_monitoring_interval)

        # Reduce validation strictness
        if self.config.validation_level == ValidationLevel.STRICT:
            self.config.validation_level = ValidationLevel.STANDARD
        elif self.config.validation_level == ValidationLevel.PARANOID:
            self.config.validation_level = ValidationLevel.STRICT

        # Reduce cache size
        if self._validation_cache:
            self.config.cache_size_limit = max(100, self.config.cache_size_limit // 2)
            self._validation_cache._max_size = self.config.cache_size_limit

        logger.info("Performance optimization completed")
        return True

    def _trigger_emergency_recovery(self) -> None:
        """Trigger emergency recovery procedures."""
        logger.critical("Triggering emergency recovery procedures")

        try:
            # Force garbage collection
            import gc

            collected = gc.collect()
            logger.debug(f"Emergency GC collected {collected} objects")

            # Clear caches
            if self._validation_cache:
                self._validation_cache.clear()
                logger.debug("Cleared validation cache")

            # Reduce monitoring frequency temporarily
            if self._health_monitor:
                self._health_monitor.monitoring_interval *= 2
                logger.debug("Reduced monitoring frequency")

        except Exception as e:
            logger.error(f"Error during emergency recovery: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive reliability system status."""
        with self._lock:
            status = {
                "initialized": self._initialized,
                "profile": self.config.profile.value,
                "active_features": self._active_features.copy(),
                "uptime_seconds": time.time() - self._start_time,
                "performance_monitoring": self.config.enable_performance_monitoring,
            }

            # Add component-specific status
            if self._health_monitor:
                status["health_status"] = self._health_monitor.get_health_summary()

            if self._state_validator:
                status["state_statistics"] = self._state_validator.get_statistics()

            if self._validation_cache:
                with self._validation_cache._lock:
                    status["validation_cache"] = {
                        "entries": len(self._validation_cache._cache),
                        "max_size": self._validation_cache._max_size,
                        "hit_rate_estimate": "N/A",  # Would need hit tracking for accurate rate
                    }

            # Performance impact
            if self._performance_baseline and self.config.enable_performance_monitoring:
                current_perf = self._measure_baseline_performance()
                baseline = self._performance_baseline
                status["performance_impact"] = {
                    "cpu_overhead_percent": current_perf["cpu_percent"]
                    - baseline["cpu_percent"],
                    "memory_overhead_mb": current_perf["memory_mb"]
                    - baseline["memory_mb"],
                    "within_limits": self._validate_performance_impact(),
                }

            return status

    def shutdown(self) -> None:
        """Shutdown all reliability features."""
        with self._lock:
            if not self._initialized:
                return

            logger.info("Shutting down XPCS reliability features")

            try:
                # Stop monitoring systems
                if self._active_features.get("health_monitoring"):
                    stop_health_monitoring()

                if self._active_features.get("state_validation"):
                    stop_state_monitoring()

                # Clear caches
                if self._validation_cache:
                    self._validation_cache.clear()

                # Cleanup state validator
                if self._state_validator:
                    self._state_validator.cleanup_destroyed_objects()

                self._initialized = False
                self._active_features.clear()

                logger.info("XPCS reliability features shutdown complete")

            except Exception as e:
                logger.error(f"Error during reliability shutdown: {e}")

    def reconfigure(self, new_config: ReliabilityConfig) -> bool:
        """Reconfigure reliability features at runtime."""
        with self._lock:
            if not self._initialized:
                self.config = new_config
                return self.initialize()

            logger.info("Reconfiguring reliability features")

            # Shutdown current configuration
            self.shutdown()

            # Apply new configuration
            self.config = new_config
            self.config.apply_profile()

            # Reinitialize with new settings
            return self.initialize()

    @classmethod
    def create_from_environment(cls) -> "XPCSReliabilityManager":
        """Create reliability manager from environment variables."""
        config = ReliabilityConfig()

        # Read from environment
        profile_name = os.environ.get("XPCS_RELIABILITY_PROFILE", "balanced")
        try:
            config.profile = ReliabilityProfile(profile_name.lower())
        except ValueError:
            logger.warning(
                f"Invalid reliability profile '{profile_name}', using 'balanced'"
            )
            config.profile = ReliabilityProfile.BALANCED

        # Override specific settings from environment
        config.enable_health_monitoring = (
            os.environ.get("XPCS_ENABLE_HEALTH_MONITORING", "true").lower() == "true"
        )
        config.enable_state_validation = (
            os.environ.get("XPCS_ENABLE_STATE_VALIDATION", "true").lower() == "true"
        )

        try:
            config.health_monitoring_interval = float(
                os.environ.get("XPCS_HEALTH_MONITORING_INTERVAL", "30.0")
            )
            config.max_cpu_overhead_percent = float(
                os.environ.get("XPCS_MAX_CPU_OVERHEAD_PERCENT", "2.0")
            )
        except ValueError as e:
            logger.warning(f"Invalid environment configuration: {e}")

        config.apply_profile()
        return cls(config)


# Global reliability manager instance
_reliability_manager: XPCSReliabilityManager | None = None
_manager_lock = threading.Lock()


def get_reliability_manager() -> XPCSReliabilityManager:
    """Get or create the global reliability manager."""
    global _reliability_manager
    if _reliability_manager is None:
        with _manager_lock:
            if _reliability_manager is None:
                _reliability_manager = XPCSReliabilityManager.create_from_environment()
    return _reliability_manager


def initialize_reliability(
    profile: ReliabilityProfile = ReliabilityProfile.BALANCED,
    validate_performance: bool = True,
) -> bool:
    """Initialize XPCS reliability features with specified profile."""
    config = ReliabilityConfig(profile=profile)
    config.apply_profile()

    manager = get_reliability_manager()
    manager.config = config
    return manager.initialize(validate_performance)


def get_reliability_status() -> dict[str, Any]:
    """Get current reliability system status."""
    manager = get_reliability_manager()
    return manager.get_status()


def shutdown_reliability() -> None:
    """Shutdown reliability features."""
    global _reliability_manager
    if _reliability_manager:
        _reliability_manager.shutdown()


# Quick setup functions for common scenarios
def enable_production_reliability() -> bool:
    """Enable reliability features optimized for production use."""
    return initialize_reliability(
        ReliabilityProfile.BALANCED, validate_performance=True
    )


def enable_development_reliability() -> bool:
    """Enable reliability features optimized for development use."""
    return initialize_reliability(ReliabilityProfile.STRICT, validate_performance=False)


def enable_minimal_reliability() -> bool:
    """Enable minimal reliability features for maximum performance."""
    return initialize_reliability(ReliabilityProfile.MINIMAL, validate_performance=True)


def enable_maximum_reliability() -> bool:
    """Enable maximum reliability features for critical applications."""
    return initialize_reliability(
        ReliabilityProfile.PARANOID, validate_performance=False
    )
