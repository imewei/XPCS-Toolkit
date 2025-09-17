"""
Qt Scalability Testing and Validation System.

This module provides comprehensive scalability testing for the Qt compliance
system, validating performance under varying loads, data volumes, and
concurrent operations.
"""

import asyncio
import gc
import math
import statistics
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal, QThread
from PySide6.QtWidgets import QApplication, QWidget

from ..monitoring import (
    get_qt_error_detector,
    get_integrated_monitoring_system,
    initialize_integrated_monitoring,
    shutdown_integrated_monitoring
)
from ..threading import get_qt_compliant_thread_manager, SafeWorkerBase
from .qt_performance_profiler import get_qt_performance_profiler
from .qt_optimization_engine import get_qt_optimization_engine
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ScalabilityDimension(Enum):
    """Dimensions for scalability testing."""

    DATA_VOLUME = "data_volume"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    THREAD_COUNT = "thread_count"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_LOAD = "cpu_load"
    USER_COUNT = "user_count"
    REQUEST_RATE = "request_rate"


class LoadPattern(Enum):
    """Load patterns for scalability testing."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    SPIKE = "spike"
    SUSTAINED = "sustained"
    BURST = "burst"


@dataclass
class ScalabilityConfiguration:
    """Configuration for scalability testing."""

    # Test dimensions
    dimensions: List[ScalabilityDimension] = field(default_factory=lambda: [
        ScalabilityDimension.DATA_VOLUME,
        ScalabilityDimension.CONCURRENT_OPERATIONS
    ])

    # Load patterns
    load_pattern: LoadPattern = LoadPattern.LINEAR

    # Scale parameters
    min_scale: int = 1
    max_scale: int = 1000
    scale_steps: int = 10
    scale_multiplier: float = 2.0  # For exponential scaling

    # Test duration
    warmup_duration_seconds: float = 5.0
    test_duration_seconds: float = 30.0
    cooldown_duration_seconds: float = 5.0

    # Performance thresholds
    max_response_time_ms: float = 1000.0
    min_throughput_ops_per_second: float = 100.0
    max_memory_growth_mb: float = 500.0
    max_cpu_utilization_percent: float = 80.0

    # Failure criteria
    max_error_rate: float = 0.05  # 5%
    max_consecutive_failures: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScalabilityMetrics:
    """Metrics collected during scalability testing."""

    timestamp: float
    scale_level: int
    dimension: ScalabilityDimension

    # Performance metrics
    response_time_ms: float = 0.0
    throughput_ops_per_second: float = 0.0
    latency_percentiles: Dict[str, float] = field(default_factory=dict)

    # Resource metrics
    memory_usage_mb: float = 0.0
    memory_growth_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    thread_count: int = 0
    file_descriptors: int = 0

    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    error_count: int = 0

    # Qt-specific metrics
    qt_errors: int = 0
    qt_warnings_processed: int = 0
    qt_operations_per_second: float = 0.0

    # Scalability indicators
    efficiency_score: float = 0.0  # Performance per resource unit
    scalability_index: float = 0.0  # How well performance scales with load

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ScalabilityTestResult:
    """Result of scalability testing."""

    test_name: str
    timestamp: float
    configuration: ScalabilityConfiguration
    success: bool

    # Metrics by scale level
    metrics_by_scale: Dict[int, ScalabilityMetrics] = field(default_factory=dict)

    # Scalability analysis
    max_sustainable_scale: int = 0
    breaking_point_scale: Optional[int] = None
    linear_scalability_range: Tuple[int, int] = (0, 0)
    scalability_coefficient: float = 0.0

    # Performance characteristics
    baseline_performance: Optional[ScalabilityMetrics] = None
    peak_performance: Optional[ScalabilityMetrics] = None
    degradation_point: Optional[int] = None

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['metrics_by_scale'] = {
            str(k): v.to_dict() for k, v in self.metrics_by_scale.items()
        }
        if self.baseline_performance:
            result['baseline_performance'] = self.baseline_performance.to_dict()
        if self.peak_performance:
            result['peak_performance'] = self.peak_performance.to_dict()
        return result


class QtScalabilityTester(QObject):
    """
    Comprehensive Qt scalability testing system.

    Provides:
    - Multi-dimensional scalability testing
    - Load pattern simulation
    - Breaking point detection
    - Performance degradation analysis
    - Resource efficiency measurement
    - Scalability recommendations
    """

    # Signals
    test_started = Signal(str)  # test_name
    scale_level_completed = Signal(str, int, object)  # test_name, scale_level, metrics
    test_completed = Signal(str, object)  # test_name, result
    breaking_point_detected = Signal(str, int)  # test_name, scale_level

    def __init__(self, config: Optional[ScalabilityConfiguration] = None, parent: QObject = None):
        """Initialize scalability tester."""
        super().__init__(parent)

        self.config = config or ScalabilityConfiguration()

        # Test registry
        self._scalability_tests: Dict[str, Callable] = {}
        self._register_standard_tests()

        # Results storage
        self._test_results: Dict[str, ScalabilityTestResult] = {}

        # Component references
        self._profiler = get_qt_performance_profiler()
        self._optimization_engine = get_qt_optimization_engine()

        logger.info("Qt scalability tester initialized")

    def _register_standard_tests(self):
        """Register standard scalability tests."""
        self._scalability_tests.update({
            "qt_error_detection_scalability": self._test_qt_error_detection_scalability,
            "widget_creation_scalability": self._test_widget_creation_scalability,
            "timer_management_scalability": self._test_timer_management_scalability,
            "thread_pool_scalability": self._test_thread_pool_scalability,
            "signal_slot_scalability": self._test_signal_slot_scalability,
            "memory_usage_scalability": self._test_memory_usage_scalability,
            "monitoring_system_scalability": self._test_monitoring_system_scalability,
            "filtering_algorithm_scalability": self._test_filtering_algorithm_scalability,
            "concurrent_operations_scalability": self._test_concurrent_operations_scalability,
            "data_volume_scalability": self._test_data_volume_scalability
        })

    def run_scalability_test(self, test_name: str, config: Optional[ScalabilityConfiguration] = None) -> ScalabilityTestResult:
        """Run a specific scalability test."""
        if test_name not in self._scalability_tests:
            raise ValueError(f"Unknown scalability test: {test_name}")

        test_config = config or self.config
        logger.info(f"Running scalability test: {test_name}")

        self.test_started.emit(test_name)

        # Create result object
        result = ScalabilityTestResult(
            test_name=test_name,
            timestamp=time.perf_counter(),
            configuration=test_config,
            success=False
        )

        try:
            # Generate scale levels
            scale_levels = self._generate_scale_levels(test_config)

            # Run test at each scale level
            for scale_level in scale_levels:
                logger.debug(f"Testing {test_name} at scale level {scale_level}")

                # Run test function
                test_func = self._scalability_tests[test_name]
                metrics = test_func(scale_level, test_config)

                # Store metrics
                result.metrics_by_scale[scale_level] = metrics

                # Emit progress signal
                self.scale_level_completed.emit(test_name, scale_level, metrics)

                # Check for breaking point
                if self._is_breaking_point(metrics, test_config):
                    result.breaking_point_scale = scale_level
                    self.breaking_point_detected.emit(test_name, scale_level)
                    logger.warning(f"Breaking point detected at scale level {scale_level}")
                    break

                # Brief pause between scale levels
                time.sleep(0.5)

            # Analyze results
            self._analyze_scalability_results(result)

            result.success = True
            logger.info(f"Scalability test completed: {test_name}")

        except Exception as e:
            logger.error(f"Scalability test failed: {test_name} - {e}")
            logger.debug(traceback.format_exc())

        # Store result
        self._test_results[test_name] = result
        self.test_completed.emit(test_name, result)

        return result

    def _generate_scale_levels(self, config: ScalabilityConfiguration) -> List[int]:
        """Generate scale levels based on configuration."""
        scale_levels = []

        if config.load_pattern == LoadPattern.LINEAR:
            step_size = (config.max_scale - config.min_scale) / config.scale_steps
            scale_levels = [int(config.min_scale + i * step_size) for i in range(config.scale_steps + 1)]

        elif config.load_pattern == LoadPattern.EXPONENTIAL:
            for i in range(config.scale_steps + 1):
                scale = config.min_scale * (config.scale_multiplier ** i)
                if scale <= config.max_scale:
                    scale_levels.append(int(scale))
                else:
                    break

        elif config.load_pattern == LoadPattern.STEP:
            step_size = (config.max_scale - config.min_scale) // config.scale_steps
            for i in range(config.scale_steps + 1):
                scale_levels.append(config.min_scale + i * step_size)

        # Ensure we don't exceed max_scale and remove duplicates
        scale_levels = sorted(list(set([s for s in scale_levels if s <= config.max_scale])))

        return scale_levels

    def _is_breaking_point(self, metrics: ScalabilityMetrics, config: ScalabilityConfiguration) -> bool:
        """Check if current metrics indicate a breaking point."""
        # Response time threshold
        if metrics.response_time_ms > config.max_response_time_ms:
            return True

        # Throughput threshold
        if metrics.throughput_ops_per_second < config.min_throughput_ops_per_second:
            return True

        # Memory threshold
        if metrics.memory_growth_mb > config.max_memory_growth_mb:
            return True

        # CPU threshold
        if metrics.cpu_utilization_percent > config.max_cpu_utilization_percent:
            return True

        # Error rate threshold
        if metrics.error_rate > config.max_error_rate:
            return True

        return False

    def _test_qt_error_detection_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test Qt error detection scalability."""
        qt_detector = get_qt_error_detector()
        qt_detector.clear_error_history()

        # Generate test messages based on scale level
        message_count = scale_level * 10  # 10 messages per scale unit
        test_messages = self._generate_test_messages(message_count)

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        errors_detected = 0

        # Process messages
        for msg in test_messages:
            error = qt_detector._process_qt_message(msg["msg_type"], msg["message"])
            if error:
                errors_detected += 1

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = end_time - start_time
        response_time_ms = duration * 1000
        throughput = message_count / duration if duration > 0 else 0
        memory_growth = end_memory - start_memory

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=response_time_ms,
            throughput_ops_per_second=throughput,
            memory_usage_mb=end_memory,
            memory_growth_mb=memory_growth,
            qt_errors=errors_detected,
            qt_warnings_processed=message_count,
            qt_operations_per_second=throughput,
            success_rate=1.0,  # Assume success if no exceptions
            error_rate=0.0
        )

    def _test_widget_creation_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test widget creation scalability."""
        widget_count = scale_level * 50  # 50 widgets per scale unit

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        widgets = []
        successful_creations = 0

        try:
            # Create widgets
            for i in range(widget_count):
                widget = QWidget()
                widget.setObjectName(f"scale_widget_{i}")
                widgets.append(widget)
                successful_creations += 1

        except Exception as e:
            logger.warning(f"Widget creation failed at count {successful_creations}: {e}")

        # Process events
        QCoreApplication.processEvents()

        mid_time = time.perf_counter()
        peak_memory = self._get_memory_usage()

        # Cleanup widgets
        for widget in widgets:
            widget.deleteLater()

        widgets.clear()
        gc.collect()

        # Process cleanup events
        for _ in range(10):
            QCoreApplication.processEvents()
            time.sleep(0.001)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        total_duration = end_time - start_time
        creation_duration = mid_time - start_time
        throughput = successful_creations / creation_duration if creation_duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=total_duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=peak_memory,
            memory_growth_mb=peak_memory - start_memory,
            success_rate=successful_creations / widget_count if widget_count > 0 else 0,
            error_rate=(widget_count - successful_creations) / widget_count if widget_count > 0 else 0
        )

    def _test_timer_management_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test timer management scalability."""
        timer_count = scale_level * 20  # 20 timers per scale unit

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        timers = []
        successful_timers = 0

        try:
            # Create and start timers
            for i in range(timer_count):
                timer = QTimer()
                timer.setObjectName(f"scale_timer_{i}")
                timer.timeout.connect(lambda: None)
                timer.start(1000)  # 1 second interval
                timers.append(timer)
                successful_timers += 1

            # Let timers run briefly
            time.sleep(0.1)

        except Exception as e:
            logger.warning(f"Timer creation failed at count {successful_timers}: {e}")

        peak_memory = self._get_memory_usage()

        # Stop and cleanup timers
        for timer in timers:
            timer.stop()
            timer.deleteLater()

        timers.clear()
        QCoreApplication.processEvents()

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = end_time - start_time
        throughput = successful_timers / duration if duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=peak_memory,
            memory_growth_mb=peak_memory - start_memory,
            success_rate=successful_timers / timer_count if timer_count > 0 else 0,
            error_rate=(timer_count - successful_timers) / timer_count if timer_count > 0 else 0
        )

    def _test_thread_pool_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test thread pool scalability."""
        thread_manager = get_qt_compliant_thread_manager()

        class ScalabilityWorker(SafeWorkerBase):
            def __init__(self, work_duration=0.01):
                super().__init__()
                self.work_duration = work_duration
                self.completed = False

            def do_work(self):
                time.sleep(self.work_duration)
                self.completed = True

        worker_count = scale_level * 10  # 10 workers per scale unit

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        workers = []
        successful_workers = 0

        try:
            # Create and start workers
            for i in range(worker_count):
                worker = ScalabilityWorker()
                workers.append(worker)
                thread_manager.start_worker(worker, f"scale_worker_{i}")
                successful_workers += 1

        except Exception as e:
            logger.warning(f"Worker creation failed at count {successful_workers}: {e}")

        # Wait for completion with timeout
        timeout = 30.0
        while time.perf_counter() - start_time < timeout:
            completed_count = sum(1 for w in workers if w.completed)
            if completed_count == len(workers):
                break
            time.sleep(0.01)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Count completed workers
        completed_workers = sum(1 for w in workers if w.completed)

        # Cleanup
        thread_manager.cleanup_finished_threads()

        # Calculate metrics
        duration = end_time - start_time
        throughput = completed_workers / duration if duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.CONCURRENT_OPERATIONS,
            response_time_ms=duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=end_memory,
            memory_growth_mb=end_memory - start_memory,
            success_rate=completed_workers / worker_count if worker_count > 0 else 0,
            error_rate=(worker_count - completed_workers) / worker_count if worker_count > 0 else 0
        )

    def _test_signal_slot_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test signal/slot scalability."""
        from PySide6.QtCore import QObject, Signal

        class ScalabilitySignalObject(QObject):
            test_signal = Signal(int)

            def __init__(self):
                super().__init__()
                self.signal_count = 0

            def slot_handler(self, value):
                self.signal_count += 1

        connection_count = scale_level * 5  # 5 connections per scale unit
        signal_count = scale_level * 100  # 100 signals per scale unit

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        objects = []
        successful_connections = 0

        try:
            # Create objects and connections
            for i in range(connection_count):
                obj = ScalabilitySignalObject()
                obj.test_signal.connect(obj.slot_handler)
                objects.append(obj)
                successful_connections += 1

            # Emit signals
            total_emissions = 0
            for i in range(signal_count):
                obj_index = i % len(objects)
                objects[obj_index].test_signal.emit(i)
                total_emissions += 1

            QCoreApplication.processEvents()

        except Exception as e:
            logger.warning(f"Signal/slot operation failed: {e}")

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Cleanup
        for obj in objects:
            obj.deleteLater()

        # Calculate metrics
        duration = end_time - start_time
        throughput = total_emissions / duration if duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=end_memory,
            memory_growth_mb=end_memory - start_memory,
            success_rate=successful_connections / connection_count if connection_count > 0 else 0,
            error_rate=0.0
        )

    def _test_memory_usage_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test memory usage scalability."""
        object_count = scale_level * 100  # 100 objects per scale unit

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        objects = []
        peak_memory = start_memory

        try:
            # Create objects in batches to monitor memory growth
            batch_size = 50
            for batch_start in range(0, object_count, batch_size):
                batch_end = min(batch_start + batch_size, object_count)

                # Create batch of objects
                for i in range(batch_start, batch_end):
                    widget = QWidget()
                    timer = QTimer(widget)
                    timer.timeout.connect(lambda: None)
                    objects.append((widget, timer))

                # Monitor memory
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)

                # Brief pause to allow memory measurement
                time.sleep(0.01)

        except Exception as e:
            logger.warning(f"Memory test failed at {len(objects)} objects: {e}")

        mid_time = time.perf_counter()

        # Cleanup objects
        for widget, timer in objects:
            widget.deleteLater()

        objects.clear()
        gc.collect()

        # Process cleanup events
        for _ in range(20):
            QCoreApplication.processEvents()
            time.sleep(0.001)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        creation_duration = mid_time - start_time
        total_duration = end_time - start_time
        throughput = len(objects) / creation_duration if creation_duration > 0 else 0

        memory_growth = peak_memory - start_memory
        memory_leaked = max(0, end_memory - start_memory)

        # Memory efficiency score (objects per MB)
        efficiency = len(objects) / max(memory_growth, 0.1)

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.MEMORY_PRESSURE,
            response_time_ms=total_duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=peak_memory,
            memory_growth_mb=memory_growth,
            efficiency_score=efficiency,
            success_rate=1.0,
            error_rate=0.0
        )

    def _test_monitoring_system_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test monitoring system scalability."""
        operation_count = scale_level * 20  # 20 operations per scale unit

        # Initialize monitoring
        monitoring_system = initialize_integrated_monitoring()

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        successful_operations = 0

        try:
            # Perform operations that trigger monitoring
            for i in range(operation_count):
                # Create widget (triggers monitoring)
                widget = QWidget()
                widget.setObjectName(f"monitoring_test_{i}")

                # Get monitoring status (triggers collection)
                status = monitoring_system.get_system_status()

                # Create timer (triggers monitoring)
                timer = QTimer(widget)
                timer.timeout.connect(lambda: None)
                timer.start(100)
                timer.stop()

                widget.deleteLater()
                successful_operations += 1

                if i % 10 == 0:
                    QCoreApplication.processEvents()

        except Exception as e:
            logger.warning(f"Monitoring test failed at operation {successful_operations}: {e}")

        finally:
            shutdown_integrated_monitoring()

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = end_time - start_time
        throughput = successful_operations / duration if duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=end_memory,
            memory_growth_mb=end_memory - start_memory,
            success_rate=successful_operations / operation_count if operation_count > 0 else 0,
            error_rate=(operation_count - successful_operations) / operation_count if operation_count > 0 else 0
        )

    def _test_filtering_algorithm_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test filtering algorithm scalability."""
        message_count = scale_level * 100  # 100 messages per scale unit

        optimization_engine = get_qt_optimization_engine()
        message_filter = optimization_engine.message_filter

        # Generate test messages
        test_messages = self._generate_test_messages(message_count)
        message_texts = [msg["message"] for msg in test_messages]

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Filter messages
        filtered_results = message_filter.filter_messages_batch(message_texts)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = end_time - start_time
        throughput = message_count / duration if duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=end_memory,
            memory_growth_mb=end_memory - start_memory,
            qt_warnings_processed=message_count,
            success_rate=1.0,
            error_rate=0.0
        )

    def _test_concurrent_operations_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test concurrent operations scalability."""
        concurrency_level = scale_level  # Direct mapping

        def concurrent_operation(operation_id: int) -> Tuple[bool, float]:
            """Single concurrent operation."""
            try:
                start = time.perf_counter()

                # Perform Qt operations
                widget = QWidget()
                widget.setObjectName(f"concurrent_widget_{operation_id}")

                timer = QTimer(widget)
                timer.timeout.connect(lambda: None)
                timer.start(10)
                timer.stop()

                widget.deleteLater()

                duration = time.perf_counter() - start
                return True, duration

            except Exception as e:
                logger.warning(f"Concurrent operation {operation_id} failed: {e}")
                return False, 0.0

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=min(concurrency_level, 32)) as executor:
            futures = [
                executor.submit(concurrent_operation, i)
                for i in range(concurrency_level * 5)  # 5 operations per concurrency unit
            ]

            successful_operations = 0
            operation_durations = []

            for future in as_completed(futures):
                success, duration = future.result()
                if success:
                    successful_operations += 1
                    operation_durations.append(duration)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Process cleanup
        QCoreApplication.processEvents()

        # Calculate metrics
        total_duration = end_time - start_time
        total_operations = concurrency_level * 5
        throughput = successful_operations / total_duration if total_duration > 0 else 0

        # Calculate latency percentiles
        latency_percentiles = {}
        if operation_durations:
            operation_durations_ms = [d * 1000 for d in operation_durations]
            latency_percentiles = {
                "p50": statistics.median(operation_durations_ms),
                "p95": statistics.quantiles(operation_durations_ms, n=20)[18] if len(operation_durations_ms) >= 20 else max(operation_durations_ms),
                "p99": statistics.quantiles(operation_durations_ms, n=100)[98] if len(operation_durations_ms) >= 100 else max(operation_durations_ms)
            }

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.CONCURRENT_OPERATIONS,
            response_time_ms=total_duration * 1000,
            throughput_ops_per_second=throughput,
            latency_percentiles=latency_percentiles,
            memory_usage_mb=end_memory,
            memory_growth_mb=end_memory - start_memory,
            success_rate=successful_operations / total_operations if total_operations > 0 else 0,
            error_rate=(total_operations - successful_operations) / total_operations if total_operations > 0 else 0
        )

    def _test_data_volume_scalability(self, scale_level: int, config: ScalabilityConfiguration) -> ScalabilityMetrics:
        """Test data volume scalability."""
        data_size = scale_level * 1000  # 1000 data points per scale unit

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        # Generate and process large data volumes
        data_points = []
        successful_processing = 0

        try:
            # Generate data
            for i in range(data_size):
                data_point = {
                    "id": i,
                    "widget": QWidget(),
                    "timestamp": time.perf_counter(),
                    "metadata": f"data_point_{i}"
                }
                data_points.append(data_point)

            # Process data
            for data_point in data_points:
                widget = data_point["widget"]
                widget.setObjectName(f"data_widget_{data_point['id']}")
                successful_processing += 1

                if successful_processing % 100 == 0:
                    QCoreApplication.processEvents()

        except Exception as e:
            logger.warning(f"Data processing failed at {successful_processing} items: {e}")

        peak_memory = self._get_memory_usage()

        # Cleanup
        for data_point in data_points:
            data_point["widget"].deleteLater()

        data_points.clear()
        gc.collect()

        # Process cleanup
        for _ in range(10):
            QCoreApplication.processEvents()
            time.sleep(0.001)

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        # Calculate metrics
        duration = end_time - start_time
        throughput = successful_processing / duration if duration > 0 else 0

        return ScalabilityMetrics(
            timestamp=time.perf_counter(),
            scale_level=scale_level,
            dimension=ScalabilityDimension.DATA_VOLUME,
            response_time_ms=duration * 1000,
            throughput_ops_per_second=throughput,
            memory_usage_mb=peak_memory,
            memory_growth_mb=peak_memory - start_memory,
            success_rate=successful_processing / data_size if data_size > 0 else 0,
            error_rate=(data_size - successful_processing) / data_size if data_size > 0 else 0
        )

    def _generate_test_messages(self, count: int) -> List[Dict[str, Any]]:
        """Generate test Qt messages."""
        from PySide6.QtCore import QtMsgType

        message_templates = [
            "QStyleHints::colorSchemeChanged() connection warning #{i}",
            "QObject::connect: Unique connection requires a pointer to member function #{i}",
            "qt.svg renderer warning about invalid element #{i}",
            "QTimer::start: Timers can only be used with threads started with QThread #{i}",
            "QWidget: Cannot create a QWidget when no GUI thread #{i}"
        ]

        messages = []
        for i in range(count):
            template = message_templates[i % len(message_templates)]
            messages.append({
                "msg_type": QtMsgType.QtWarningMsg,
                "message": template.format(i=i)
            })

        return messages

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _analyze_scalability_results(self, result: ScalabilityTestResult):
        """Analyze scalability test results."""
        if not result.metrics_by_scale:
            return

        # Find baseline (smallest scale)
        min_scale = min(result.metrics_by_scale.keys())
        result.baseline_performance = result.metrics_by_scale[min_scale]

        # Find peak performance
        peak_throughput = 0
        peak_scale = 0
        for scale, metrics in result.metrics_by_scale.items():
            if metrics.throughput_ops_per_second > peak_throughput:
                peak_throughput = metrics.throughput_ops_per_second
                peak_scale = scale

        if peak_scale > 0:
            result.peak_performance = result.metrics_by_scale[peak_scale]

        # Calculate maximum sustainable scale
        result.max_sustainable_scale = self._find_max_sustainable_scale(result)

        # Calculate scalability coefficient
        result.scalability_coefficient = self._calculate_scalability_coefficient(result)

        # Find linear scalability range
        result.linear_scalability_range = self._find_linear_range(result)

        # Find degradation point
        result.degradation_point = self._find_degradation_point(result)

        # Generate recommendations
        result.recommendations = self._generate_scalability_recommendations(result)

    def _find_max_sustainable_scale(self, result: ScalabilityTestResult) -> int:
        """Find maximum sustainable scale level."""
        max_scale = 0

        for scale, metrics in result.metrics_by_scale.items():
            # Check if performance is still acceptable
            if (metrics.response_time_ms <= result.configuration.max_response_time_ms and
                metrics.throughput_ops_per_second >= result.configuration.min_throughput_ops_per_second and
                metrics.error_rate <= result.configuration.max_error_rate):
                max_scale = max(max_scale, scale)

        return max_scale

    def _calculate_scalability_coefficient(self, result: ScalabilityTestResult) -> float:
        """Calculate scalability coefficient (how well performance scales with load)."""
        if len(result.metrics_by_scale) < 2:
            return 0.0

        # Use throughput as the primary metric
        scales = sorted(result.metrics_by_scale.keys())
        throughputs = [result.metrics_by_scale[s].throughput_ops_per_second for s in scales]

        # Calculate correlation between scale and throughput
        if len(scales) < 2:
            return 0.0

        # Simple linear correlation
        n = len(scales)
        sum_x = sum(scales)
        sum_y = sum(throughputs)
        sum_xy = sum(x * y for x, y in zip(scales, throughputs))
        sum_x2 = sum(x * x for x in scales)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        correlation = (n * sum_xy - sum_x * sum_y) / math.sqrt(denominator * (n * sum(y * y for y in throughputs) - sum_y * sum_y))

        # Convert correlation to scalability coefficient (0-1 scale)
        return max(0, correlation)

    def _find_linear_range(self, result: ScalabilityTestResult) -> Tuple[int, int]:
        """Find the range where scaling is approximately linear."""
        scales = sorted(result.metrics_by_scale.keys())
        if len(scales) < 3:
            return (0, 0)

        # Find the longest subsequence with approximately linear scaling
        best_start = 0
        best_end = 0
        best_length = 0

        for start in range(len(scales)):
            for end in range(start + 2, len(scales)):
                if self._is_linear_range(result, scales[start:end+1]):
                    length = end - start + 1
                    if length > best_length:
                        best_length = length
                        best_start = start
                        best_end = end

        if best_length >= 3:
            return (scales[best_start], scales[best_end])
        else:
            return (0, 0)

    def _is_linear_range(self, result: ScalabilityTestResult, scale_range: List[int]) -> bool:
        """Check if a range exhibits linear scaling."""
        if len(scale_range) < 3:
            return False

        throughputs = [result.metrics_by_scale[s].throughput_ops_per_second for s in scale_range]

        # Calculate R-squared for linear fit
        # Simplified implementation
        mean_throughput = statistics.mean(throughputs)
        total_variance = sum((t - mean_throughput) ** 2 for t in throughputs)

        if total_variance == 0:
            return True

        # Linear regression
        n = len(scale_range)
        sum_x = sum(scale_range)
        sum_y = sum(throughputs)
        sum_xy = sum(x * y for x, y in zip(scale_range, throughputs))
        sum_x2 = sum(x * x for x in scale_range)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        predicted = [slope * x + intercept for x in scale_range]
        residual_variance = sum((actual - pred) ** 2 for actual, pred in zip(throughputs, predicted))

        r_squared = 1 - (residual_variance / total_variance)

        # Consider linear if R-squared > 0.8
        return r_squared > 0.8

    def _find_degradation_point(self, result: ScalabilityTestResult) -> Optional[int]:
        """Find the point where performance starts to degrade significantly."""
        scales = sorted(result.metrics_by_scale.keys())
        if len(scales) < 2:
            return None

        # Look for significant throughput decrease
        for i in range(1, len(scales)):
            current_throughput = result.metrics_by_scale[scales[i]].throughput_ops_per_second
            previous_throughput = result.metrics_by_scale[scales[i-1]].throughput_ops_per_second

            if previous_throughput > 0:
                degradation = (previous_throughput - current_throughput) / previous_throughput
                if degradation > 0.2:  # 20% degradation
                    return scales[i]

        return None

    def _generate_scalability_recommendations(self, result: ScalabilityTestResult) -> List[str]:
        """Generate scalability recommendations."""
        recommendations = []

        # Max sustainable scale recommendations
        if result.max_sustainable_scale < result.configuration.max_scale * 0.5:
            recommendations.append("Low maximum sustainable scale - consider system optimization")

        # Breaking point recommendations
        if result.breaking_point_scale:
            recommendations.append(f"Breaking point detected at scale {result.breaking_point_scale} - investigate resource bottlenecks")

        # Scalability coefficient recommendations
        if result.scalability_coefficient < 0.7:
            recommendations.append("Poor scalability coefficient - system doesn't scale well with load")

        # Linear range recommendations
        linear_start, linear_end = result.linear_scalability_range
        if linear_end - linear_start < result.configuration.max_scale * 0.3:
            recommendations.append("Short linear scalability range - consider architectural improvements")

        # Memory growth recommendations
        peak_memory_growth = max(
            (metrics.memory_growth_mb for metrics in result.metrics_by_scale.values()),
            default=0
        )
        if peak_memory_growth > 200:  # More than 200MB growth
            recommendations.append("High memory growth - implement memory optimization strategies")

        # Error rate recommendations
        max_error_rate = max(
            (metrics.error_rate for metrics in result.metrics_by_scale.values()),
            default=0
        )
        if max_error_rate > 0.1:  # More than 10% error rate
            recommendations.append("High error rate at scale - improve error handling and stability")

        return recommendations

    def run_comprehensive_scalability_suite(self, config: Optional[ScalabilityConfiguration] = None) -> Dict[str, ScalabilityTestResult]:
        """Run all scalability tests."""
        logger.info("Running comprehensive Qt scalability test suite")

        test_config = config or self.config
        results = {}

        for test_name in self._scalability_tests.keys():
            try:
                logger.info(f"Running scalability test: {test_name}")
                result = self.run_scalability_test(test_name, test_config)
                results[test_name] = result

                # Brief pause between tests
                time.sleep(2.0)

            except Exception as e:
                logger.error(f"Scalability test {test_name} failed: {e}")

        logger.info(f"Scalability test suite completed: {len(results)} tests executed")
        return results

    def generate_scalability_report(self, results: Optional[Dict[str, ScalabilityTestResult]] = None) -> Dict[str, Any]:
        """Generate comprehensive scalability report."""
        if results is None:
            results = self._test_results

        successful_tests = [r for r in results.values() if r.success]

        # Calculate summary statistics
        avg_max_sustainable_scale = statistics.mean([r.max_sustainable_scale for r in successful_tests]) if successful_tests else 0
        avg_scalability_coefficient = statistics.mean([r.scalability_coefficient for r in successful_tests]) if successful_tests else 0

        # Find overall breaking points
        breaking_points = [r.breaking_point_scale for r in successful_tests if r.breaking_point_scale]
        min_breaking_point = min(breaking_points) if breaking_points else None

        return {
            "summary": {
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "success_rate": len(successful_tests) / len(results) if results else 0,
                "avg_max_sustainable_scale": avg_max_sustainable_scale,
                "avg_scalability_coefficient": avg_scalability_coefficient,
                "min_breaking_point": min_breaking_point
            },
            "detailed_results": {name: result.to_dict() for name, result in results.items()},
            "scalability_analysis": {
                "best_scaling_test": self._find_best_scaling_test(successful_tests),
                "worst_scaling_test": self._find_worst_scaling_test(successful_tests),
                "bottleneck_analysis": self._analyze_bottlenecks(successful_tests)
            },
            "recommendations": self._generate_suite_scalability_recommendations(successful_tests)
        }

    def _find_best_scaling_test(self, results: List[ScalabilityTestResult]) -> Optional[str]:
        """Find the test with the best scalability characteristics."""
        if not results:
            return None

        best_test = max(results, key=lambda r: r.scalability_coefficient)
        return best_test.test_name

    def _find_worst_scaling_test(self, results: List[ScalabilityTestResult]) -> Optional[str]:
        """Find the test with the worst scalability characteristics."""
        if not results:
            return None

        worst_test = min(results, key=lambda r: r.scalability_coefficient)
        return worst_test.test_name

    def _analyze_bottlenecks(self, results: List[ScalabilityTestResult]) -> Dict[str, Any]:
        """Analyze system bottlenecks based on scalability results."""
        bottlenecks = {
            "memory": 0,
            "cpu": 0,
            "io": 0,
            "threading": 0
        }

        for result in results:
            # Analyze failure modes
            if result.breaking_point_scale:
                # Check what caused the breaking point
                breaking_metrics = result.metrics_by_scale.get(result.breaking_point_scale)
                if breaking_metrics:
                    if breaking_metrics.memory_growth_mb > result.configuration.max_memory_growth_mb:
                        bottlenecks["memory"] += 1
                    if breaking_metrics.cpu_utilization_percent > result.configuration.max_cpu_utilization_percent:
                        bottlenecks["cpu"] += 1
                    if breaking_metrics.response_time_ms > result.configuration.max_response_time_ms:
                        bottlenecks["io"] += 1
                    if "thread" in result.test_name and breaking_metrics.error_rate > 0:
                        bottlenecks["threading"] += 1

        # Find primary bottleneck
        primary_bottleneck = max(bottlenecks, key=bottlenecks.get) if any(bottlenecks.values()) else "none"

        return {
            "bottleneck_counts": bottlenecks,
            "primary_bottleneck": primary_bottleneck
        }

    def _generate_suite_scalability_recommendations(self, results: List[ScalabilityTestResult]) -> List[str]:
        """Generate recommendations for the entire scalability suite."""
        recommendations = []

        # Collect all individual recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)

        # Count recommendation frequency
        recommendation_counts = defaultdict(int)
        for rec in all_recommendations:
            recommendation_counts[rec] += 1

        # Prioritize frequent recommendations
        frequent_recommendations = [
            rec for rec, count in recommendation_counts.items()
            if count >= len(results) * 0.3  # Appears in 30% or more of tests
        ]

        recommendations.extend(frequent_recommendations)

        # Add suite-level recommendations
        avg_coefficient = statistics.mean([r.scalability_coefficient for r in results]) if results else 0
        if avg_coefficient < 0.6:
            recommendations.append("Overall poor scalability - consider architectural redesign")

        avg_max_scale = statistics.mean([r.max_sustainable_scale for r in results]) if results else 0
        if avg_max_scale < 100:
            recommendations.append("Low sustainable scale across tests - investigate system limitations")

        return recommendations


# Global instance
_qt_scalability_tester: Optional[QtScalabilityTester] = None


def get_qt_scalability_tester(config: ScalabilityConfiguration = None) -> QtScalabilityTester:
    """Get the global Qt scalability tester instance."""
    global _qt_scalability_tester

    if _qt_scalability_tester is None:
        _qt_scalability_tester = QtScalabilityTester(config)

    return _qt_scalability_tester


def run_qt_scalability_tests(config: Optional[ScalabilityConfiguration] = None) -> Dict[str, Any]:
    """Run complete Qt scalability test suite."""
    tester = get_qt_scalability_tester(config)

    try:
        # Run comprehensive scalability tests
        results = tester.run_comprehensive_scalability_suite(config)

        # Generate report
        report = tester.generate_scalability_report(results)

        return report

    except Exception as e:
        logger.error(f"Scalability test suite execution failed: {e}")
        return {"error": str(e), "success": False}


if __name__ == "__main__":
    # Run scalability tests
    config = ScalabilityConfiguration(
        max_scale=500,
        scale_steps=10,
        test_duration_seconds=20.0
    )

    report = run_qt_scalability_tests(config)

    print("Qt Scalability Test Report")
    print("=" * 50)
    print(f"Success Rate: {report.get('summary', {}).get('success_rate', 0):.1%}")
    print(f"Average Max Sustainable Scale: {report.get('summary', {}).get('avg_max_sustainable_scale', 0):.0f}")
    print(f"Average Scalability Coefficient: {report.get('summary', {}).get('avg_scalability_coefficient', 0):.2f}")

    bottleneck_analysis = report.get('scalability_analysis', {}).get('bottleneck_analysis', {})
    primary_bottleneck = bottleneck_analysis.get('primary_bottleneck', 'unknown')
    print(f"Primary Bottleneck: {primary_bottleneck}")

    best_test = report.get('scalability_analysis', {}).get('best_scaling_test')
    worst_test = report.get('scalability_analysis', {}).get('worst_scaling_test')
    if best_test:
        print(f"Best Scaling Test: {best_test}")
    if worst_test:
        print(f"Worst Scaling Test: {worst_test}")