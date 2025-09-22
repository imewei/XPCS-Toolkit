"""
Predictive Memory Pressure Detection for XPCS Toolkit

This module provides advanced memory pressure prediction specifically tailored
for XPCS data analysis workflows, enabling proactive memory management.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import psutil

from .logging_config import get_logger
from .memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


@dataclass
class MemoryPrediction:
    """Container for memory usage predictions."""

    predicted_mb: float
    confidence: float  # 0.0 to 1.0
    time_horizon_seconds: float
    pressure_level: MemoryPressure
    recommended_actions: list[str]


@dataclass
class OperationProfile:
    """Memory profile for a specific operation type."""

    operation_type: str
    avg_memory_mb: float
    peak_memory_mb: float
    duration_seconds: float
    input_size_correlation: float  # How much memory scales with input size


class XPCSMemoryPredictor:
    """
    Advanced memory pressure predictor for XPCS workflows.

    This predictor learns from historical memory usage patterns and
    provides proactive warnings before memory pressure situations.
    """

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size

        # Memory usage history
        self.memory_history: deque = deque(maxlen=history_size)
        self.operation_history: deque = deque(maxlen=history_size)

        # Operation profiles for different XPCS operations
        self.operation_profiles: dict[str, OperationProfile] = {
            "load_saxs_2d": OperationProfile("load_saxs_2d", 150.0, 300.0, 2.0, 1.2),
            "compute_saxs_log": OperationProfile(
                "compute_saxs_log", 100.0, 200.0, 1.5, 1.0
            ),
            "fit_g2": OperationProfile("fit_g2", 50.0, 100.0, 5.0, 0.3),
            "fit_g2_parallel": OperationProfile(
                "fit_g2_parallel", 200.0, 400.0, 3.0, 0.3
            ),
            "load_twotime_c2": OperationProfile(
                "load_twotime_c2", 500.0, 1000.0, 3.0, 2.0
            ),
            "average_files": OperationProfile("average_files", 100.0, 300.0, 10.0, 0.8),
            "roi_calculation": OperationProfile(
                "roi_calculation", 30.0, 60.0, 1.0, 0.5
            ),
        }

        # Memory manager for coordinated cleanup
        self.memory_manager = get_memory_manager()

        # Prediction models
        self._trend_window = 10  # Number of recent samples for trend analysis
        self._peak_detection_window = 5  # Window for peak detection

        logger.info("XPCSMemoryPredictor initialized")

    def record_operation(
        self,
        operation_type: str,
        input_size_mb: float,
        memory_before_mb: float,
        memory_after_mb: float,
        duration_seconds: float,
    ):
        """
        Record memory usage for a completed operation.

        Parameters
        ----------
        operation_type : str
            Type of operation ('load_saxs_2d', 'fit_g2', etc.)
        input_size_mb : float
            Size of input data in MB
        memory_before_mb : float
            Memory usage before operation
        memory_after_mb : float
            Memory usage after operation
        duration_seconds : float
            How long the operation took
        """
        memory_delta = memory_after_mb - memory_before_mb
        timestamp = time.time()

        # Record the operation
        operation_record = {
            "timestamp": timestamp,
            "operation_type": operation_type,
            "input_size_mb": input_size_mb,
            "memory_before_mb": memory_before_mb,
            "memory_after_mb": memory_after_mb,
            "memory_delta_mb": memory_delta,
            "duration_seconds": duration_seconds,
        }

        self.operation_history.append(operation_record)

        # Update operation profile with exponential moving average
        if operation_type in self.operation_profiles:
            profile = self.operation_profiles[operation_type]
            alpha = 0.1  # Learning rate

            # Update average memory usage
            profile.avg_memory_mb = (
                alpha * abs(memory_delta) + (1 - alpha) * profile.avg_memory_mb
            )

            # Update peak memory (max of recent observations)
            profile.peak_memory_mb = max(
                abs(memory_delta) * 1.5,  # Safety factor
                profile.peak_memory_mb * 0.95,  # Gradual decay
            )

            # Update duration
            profile.duration_seconds = (
                alpha * duration_seconds + (1 - alpha) * profile.duration_seconds
            )

            # Update size correlation if we have meaningful input size
            if input_size_mb > 10:  # Only for operations with significant data
                size_correlation = abs(memory_delta) / input_size_mb
                profile.input_size_correlation = (
                    alpha * size_correlation
                    + (1 - alpha) * profile.input_size_correlation
                )

            logger.debug(
                f"Updated profile for {operation_type}: "
                f"avg={profile.avg_memory_mb:.1f}MB, "
                f"peak={profile.peak_memory_mb:.1f}MB"
            )

    def predict_operation_memory(
        self, operation_type: str, input_size_mb: float = 0.0
    ) -> MemoryPrediction:
        """
        Predict memory usage for a planned operation.

        Parameters
        ----------
        operation_type : str
            Type of operation to predict
        input_size_mb : float
            Expected input data size

        Returns
        -------
        MemoryPrediction
            Prediction with memory estimate and recommendations
        """
        current_memory = psutil.virtual_memory()
        current_usage_mb = (current_memory.total - current_memory.available) / (
            1024 * 1024
        )

        if operation_type not in self.operation_profiles:
            # Create default profile for unknown operations
            predicted_mb = max(50.0, input_size_mb * 0.5)
            confidence = 0.3
        else:
            profile = self.operation_profiles[operation_type]

            # Base prediction from profile
            base_prediction = profile.avg_memory_mb

            # Scale by input size if relevant
            if input_size_mb > 10:
                size_scaling = input_size_mb * profile.input_size_correlation
                predicted_mb = base_prediction + size_scaling
            else:
                predicted_mb = base_prediction

            # Add safety buffer based on historical variance
            safety_factor = 1.3  # 30% safety buffer
            predicted_mb *= safety_factor

            # Confidence based on amount of historical data
            num_observations = len(
                [
                    op
                    for op in self.operation_history
                    if op["operation_type"] == operation_type
                ]
            )
            confidence = min(0.9, 0.3 + 0.1 * num_observations)

        # Predict resulting memory usage
        predicted_total_mb = current_usage_mb + predicted_mb

        # Determine pressure level
        memory_total_mb = current_memory.total / (1024 * 1024)
        pressure_ratio = predicted_total_mb / memory_total_mb

        if pressure_ratio >= 0.9:
            pressure_level = MemoryPressure.CRITICAL
        elif pressure_ratio >= 0.8:
            pressure_level = MemoryPressure.HIGH
        elif pressure_ratio >= 0.65:
            pressure_level = MemoryPressure.MODERATE
        else:
            pressure_level = MemoryPressure.LOW

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pressure_level, operation_type, predicted_mb, current_usage_mb
        )

        return MemoryPrediction(
            predicted_mb=predicted_mb,
            confidence=confidence,
            time_horizon_seconds=profile.duration_seconds
            if operation_type in self.operation_profiles
            else 5.0,
            pressure_level=pressure_level,
            recommended_actions=recommendations,
        )

    def _generate_recommendations(
        self,
        pressure_level: MemoryPressure,
        operation_type: str,
        predicted_mb: float,
        current_usage_mb: float,
    ) -> list[str]:
        """Generate memory management recommendations."""
        recommendations = []

        if pressure_level == MemoryPressure.CRITICAL:
            recommendations.extend(
                [
                    "CRITICAL: Clear all non-essential caches immediately",
                    "Consider breaking operation into smaller chunks",
                    "Close other applications if possible",
                    f"Operation will require {predicted_mb:.0f}MB additional memory",
                ]
            )

        elif pressure_level == MemoryPressure.HIGH:
            recommendations.extend(
                [
                    "Clear computation and plot caches",
                    "Consider using streaming/chunked processing",
                    "Monitor memory usage closely during operation",
                ]
            )

        elif pressure_level == MemoryPressure.MODERATE:
            recommendations.extend(
                [
                    "Consider clearing old cached data",
                    "Monitor for memory leaks during operation",
                ]
            )

        # Operation-specific recommendations
        if operation_type == "load_saxs_2d" and predicted_mb > 200:
            recommendations.append("Consider using memory-mapped file access")

        elif operation_type == "fit_g2" and predicted_mb > 100:
            recommendations.append(
                "Consider using sequential fitting instead of parallel"
            )

        elif operation_type == "load_twotime_c2" and predicted_mb > 500:
            recommendations.extend(
                ["Use chunked C2 loading", "Clear SAXS data cache before loading C2"]
            )

        return recommendations

    def detect_memory_trends(self) -> dict[str, float]:
        """
        Analyze memory usage trends to predict future pressure.

        Returns
        -------
        dict[str, float]
            Trend metrics including growth rate and volatility
        """
        if len(self.memory_history) < self._trend_window:
            return {"growth_rate_mb_per_hour": 0.0, "volatility": 0.0}

        # Get recent memory samples
        recent_samples = list(self.memory_history)[-self._trend_window :]
        timestamps = [sample["timestamp"] for sample in recent_samples]
        memory_values = [sample["memory_mb"] for sample in recent_samples]

        # Calculate trend using linear regression
        if len(memory_values) >= 3:
            # Simple linear regression
            x = np.array(timestamps) - timestamps[0]  # Relative time
            y = np.array(memory_values)

            if len(x) > 1 and np.std(x) > 0:
                slope = np.polyfit(x, y, 1)[0]
                growth_rate_mb_per_hour = slope * 3600  # Convert to per hour

                # Calculate volatility as standard deviation of residuals
                y_pred = np.polyfit(x, y, 1)[0] * x + np.polyfit(x, y, 1)[1]
                volatility = np.std(y - y_pred)
            else:
                growth_rate_mb_per_hour = 0.0
                volatility = 0.0
        else:
            growth_rate_mb_per_hour = 0.0
            volatility = 0.0

        return {
            "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
            "volatility": volatility,
            "samples_analyzed": len(memory_values),
        }

    def check_proactive_cleanup_needed(self) -> tuple[bool, list[str]]:
        """
        Check if proactive cleanup is recommended based on trends.

        Returns
        -------
        tuple[bool, list[str]]
            (cleanup_needed, reasons)
        """
        trends = self.detect_memory_trends()
        current_pressure = self.memory_manager.get_memory_pressure()

        cleanup_needed = False
        reasons = []

        # Check for rapid memory growth
        if trends["growth_rate_mb_per_hour"] > 100:  # Growing > 100MB/hour
            cleanup_needed = True
            reasons.append(
                f"Rapid memory growth: {trends['growth_rate_mb_per_hour']:.0f}MB/hour"
            )

        # Check for high volatility (possible memory leaks)
        if trends["volatility"] > 50:  # High volatility
            cleanup_needed = True
            reasons.append(f"High memory volatility: {trends['volatility']:.0f}MB")

        # Check current pressure level
        if current_pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            cleanup_needed = True
            reasons.append(f"Current memory pressure: {current_pressure.value}")

        # Check cache efficiency
        cache_stats = self.memory_manager.get_cache_stats()
        if cache_stats.get("cache_efficiency", 1.0) < 0.3:  # Low hit rate
            cleanup_needed = True
            reasons.append("Low cache efficiency suggests wasted memory")

        return cleanup_needed, reasons

    def update_memory_snapshot(self):
        """Update the memory usage snapshot for trend analysis."""
        memory = psutil.virtual_memory()
        timestamp = time.time()

        memory_snapshot = {
            "timestamp": timestamp,
            "memory_mb": (memory.total - memory.available) / (1024 * 1024),
            "memory_percent": memory.percent,
            "available_mb": memory.available / (1024 * 1024),
        }

        self.memory_history.append(memory_snapshot)

    def get_prediction_summary(self) -> dict[str, any]:
        """Get a summary of current memory predictions and trends."""
        trends = self.detect_memory_trends()
        cleanup_needed, cleanup_reasons = self.check_proactive_cleanup_needed()
        current_memory = psutil.virtual_memory()

        return {
            "current_memory_mb": (current_memory.total - current_memory.available)
            / (1024 * 1024),
            "current_pressure": self.memory_manager.get_memory_pressure().value,
            "memory_trends": trends,
            "proactive_cleanup_needed": cleanup_needed,
            "cleanup_reasons": cleanup_reasons,
            "operation_profiles": {
                name: {
                    "avg_memory_mb": profile.avg_memory_mb,
                    "peak_memory_mb": profile.peak_memory_mb,
                    "duration_seconds": profile.duration_seconds,
                }
                for name, profile in self.operation_profiles.items()
            },
            "cache_stats": self.memory_manager.get_cache_stats(),
        }


# Global predictor instance
_global_predictor: XPCSMemoryPredictor | None = None


def get_memory_predictor() -> XPCSMemoryPredictor:
    """Get or create the global memory predictor instance."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = XPCSMemoryPredictor()
    return _global_predictor


def predict_operation_memory(
    operation_type: str, input_size_mb: float = 0.0
) -> MemoryPrediction:
    """Convenience function for memory prediction."""
    return get_memory_predictor().predict_operation_memory(
        operation_type, input_size_mb
    )


def record_operation_memory(
    operation_type: str,
    input_size_mb: float,
    memory_before_mb: float,
    memory_after_mb: float,
    duration_seconds: float,
):
    """Convenience function for recording operation memory usage."""
    return get_memory_predictor().record_operation(
        operation_type,
        input_size_mb,
        memory_before_mb,
        memory_after_mb,
        duration_seconds,
    )
