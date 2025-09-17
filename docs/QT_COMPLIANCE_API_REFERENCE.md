# Qt Compliance System API Reference

## Overview

This document provides comprehensive API documentation for the Qt Compliance System implemented in the XPCS Toolkit. The system ensures Qt threading compliance, prevents resource leaks, and provides robust error handling for GUI applications.

## Table of Contents

1. [Signal/Slot Safety API](#signalslot-safety-api)
2. [Thread Management API](#thread-management-api)
3. [Worker Safety API](#worker-safety-api)
4. [Health Monitoring API](#health-monitoring-api)
5. [Utilities and Context Managers](#utilities-and-context-managers)
6. [Examples and Usage Patterns](#examples-and-usage-patterns)

## Signal/Slot Safety API

### Module: `xpcs_toolkit.plothandler.qt_signal_fixes`

#### Core Classes

##### `QtConnectionFixer`

Static utility class for fixing and preventing Qt signal/slot connection issues.

**Methods:**

```python
@staticmethod
@contextmanager
def suppress_qt_warnings():
    pass
```
Context manager to temporarily suppress Qt connection warnings.

**Returns:** Context manager for warning suppression

**Example:**
```python
with QtConnectionFixer.suppress_qt_warnings():
    widget = pg.ImageView()  # No QStyleHints warnings
```

---

```python
@staticmethod
def validate_signal_connection(signal: Signal, slot: Callable,
                             connection_type: Qt.ConnectionType = Qt.ConnectionType.AutoConnection) -> bool:
    pass
```
Validate and establish a Qt5+ compliant signal/slot connection.

**Args:**
- `signal`: Qt signal object
- `slot`: Callable slot function or method
- `connection_type`: Qt connection type (default: AutoConnection)

**Returns:** `True` if connection successful, `False` otherwise

**Example:**
```python
success = QtConnectionFixer.validate_signal_connection(
    button.clicked, my_handler, Qt.ConnectionType.QueuedConnection
)
```

##### `PyQtGraphWrapper`

Wrapper for PyQtGraph components that minimizes Qt connection warnings.

```python
@staticmethod
def create_imageview_dev(*args, **kwargs):
    pass
```
Create ImageViewDev with Qt connection issue mitigation.

**Returns:** ImageViewDev instance with warning suppression

#### Utility Functions

```python
def safe_connect(signal: Signal, slot: Callable,
                connection_type: Qt.ConnectionType = Qt.ConnectionType.AutoConnection) -> bool:
    pass
```
Safely connect a signal to a slot with validation.

**Args:**
- `signal`: Qt signal
- `slot`: Callable slot
- `connection_type`: Connection type

**Returns:** `True` if successful

---

```python
def safe_disconnect(signal: Signal, slot: Optional[Callable] = None) -> bool:
    pass
```
Safely disconnect a signal from a slot.

**Args:**
- `signal`: Qt signal
- `slot`: Optional specific slot to disconnect

**Returns:** `True` if successful

---

```python
@contextmanager
def qt_connection_context():
    pass
```
Context manager for Qt operations that may generate connection warnings.

**Usage:**
```python
with qt_connection_context():
    widget = pg.ImageView()
    widget.sigTimeChanged.connect(handler)
```

## Thread Management API

### Module: `xpcs_toolkit.threading.qt_compliant_thread_manager`

#### Core Classes

##### `QtCompliantThreadManager`

Main Qt-compliant thread lifecycle manager.

**Constructor:**
```python
def __init__(self, parent: QObject = None):
    pass
```

**Methods:**

```python
def create_managed_thread(self, thread_id: str, target: Callable = None,
                         thread_purpose: str = "general") -> Optional[QThread]:
    pass
```
Create a managed Qt thread with proper lifecycle tracking.

**Args:**
- `thread_id`: Unique identifier for the thread
- `target`: Optional target function to run in thread
- `thread_purpose`: Description of thread purpose

**Returns:** QThread instance or None if creation failed

---

```python
def start_thread(self, thread_id: str) -> bool
def stop_thread(self, thread_id: str, timeout_ms: int = 5000) -> bool
def get_thread_info(self, thread_id: str) -> Optional[ThreadInfo]
```

##### `QtThreadSafetyValidator`

Validates Qt thread safety for operations and objects.

**Methods:**

```python
def is_main_thread(self) -> bool
def is_qt_thread(self) -> bool
def ensure_main_thread(self, operation_name: str)
def validate_timer_creation(self, parent: QObject = None) -> bool
```

##### `QtCompliantTimerManager`

Manager for Qt-compliant timer creation and lifecycle.

**Methods:**

```python
def create_timer(self, timer_id: str, parent: QObject = None,
                interval: int = 1000, single_shot: bool = False) -> Optional[QTimer]
def start_timer(self, timer_id: str, interval: Optional[int] = None) -> bool
def stop_timer(self, timer_id: str) -> bool
def destroy_timer(self, timer_id: str) -> bool
```

#### Global Functions

```python
def get_qt_compliant_thread_manager() -> QtCompliantThreadManager
def initialize_qt_thread_management() -> QtCompliantThreadManager
def shutdown_qt_thread_management():
    pass
```

## Worker Safety API

### Module: `xpcs_toolkit.threading.enhanced_worker_safety`

#### Core Classes

##### `SafeWorkerBase(QRunnable)`

Enhanced base class for safe worker operations with comprehensive safety features.

**Constructor:**
```python
def __init__(self, worker_id: str = None, max_retries: int = 3,
             timeout_seconds: float = 300.0)
```

**Key Methods:**

```python
def acquire_resource(self, resource_id: str, resource_type: ResourceType,
                    cleanup_callback: Callable = None,
                    auto_cleanup_timeout: float = 300.0,
                    metadata: Dict[str, Any] = None) -> bool
```
Acquire a resource lease for the worker.

**Args:**
- `resource_id`: Unique identifier for the resource
- `resource_type`: Type of resource (ResourceType enum)
- `cleanup_callback`: Optional cleanup function
- `auto_cleanup_timeout`: Timeout for automatic cleanup
- `metadata`: Additional resource metadata

**Returns:** `True` if resource was successfully acquired

---

```python
def emit_progress_safe(self, current: int, total: int, message: str = "",
                      additional_stats: Dict[str, Any] = None)
```
Emit progress signal safely with enhanced information.

---

```python
def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                severity: WorkerErrorSeverity = WorkerErrorSeverity.MEDIUM) -> WorkerRecoveryAction
```
Handle an error with automatic recovery determination.

**Abstract Method:**
```python
def do_work(self) -> Any:
    pass
```
Method to be implemented by subclasses containing the actual work.

##### `SafeWorkerPool`

Enhanced worker pool with automatic safety management.

**Constructor:**
```python
def __init__(self, thread_pool: QtCore.QThreadPool, pool_id: str = "safe_pool")
```

**Methods:**

```python
def submit_safe_worker(self, worker: SafeWorkerBase) -> str
def cancel_all_workers(self, reason: str = "Pool shutdown")
def get_pool_stats(self) -> Dict[str, Any]
def shutdown():
    pass
```

#### Enums and Data Classes

```python
class WorkerErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class WorkerRecoveryAction(Enum):
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK = "fallback"
    FAIL_GRACEFULLY = "fail_gracefully"
    ESCALATE = "escalate"

class ResourceType(Enum):
    MEMORY = "memory"
    FILE_HANDLE = "file_handle"
    NETWORK_CONNECTION = "network_connection"
    TEMPORARY_FILE = "temporary_file"
    CACHE_ENTRY = "cache_entry"
    GUI_OBJECT = "gui_object"
```

#### Factory Functions

```python
def create_safe_worker(work_function: Callable, worker_id: str = None,
                      max_retries: int = 3, timeout_seconds: float = 300.0,
                      **kwargs) -> SafeWorkerBase
```
Factory function to create a safe worker from a work function.

## Health Monitoring API

### Module: `xpcs_toolkit.threading.thread_pool_integration_validator`

#### Core Classes

##### `ThreadPoolIntegrationValidator`

Comprehensive validator for thread pool integration and resource management.

**Methods:**

```python
def register_thread_pool(self, pool_id: str, thread_pool: QThreadPool)
def validate_thread_pool_health(self, pool_id: str) -> Dict[str, Any]
def track_resource_allocation(self, pool_id: str, resource_id: str,
                            resource_type: str, worker_id: str = None)
def track_resource_deallocation(self, pool_id: str, resource_id: str)
def get_health_summary(self) -> Dict[str, Any]
def generate_validation_report(self) -> str:
    pass
```

##### `ThreadPoolHealthMetrics`

Comprehensive thread pool health metrics dataclass.

**Key Attributes:**
- Thread counts: `active_threads`, `max_threads`, `queued_tasks`
- Performance: `avg_task_duration`, `cpu_utilization`, `throughput_tasks_per_second`
- Health indicators: `resource_leak_count`, `qt_violations`, `task_timeout_count`

**Methods:**
```python
def calculate_health_score(self) -> float
def get_health_status(self) -> PoolHealthStatus:
    pass
```

#### Global Functions

```python
def get_thread_pool_validator() -> ThreadPoolIntegrationValidator
def initialize_thread_pool_validation() -> ThreadPoolIntegrationValidator
@contextmanager
def thread_pool_validation_context():
    pass
```

## Utilities and Context Managers

### Context Managers

```python
@contextmanager
def qt_compliant_context(thread_manager: QtCompliantThreadManager = None)
```
Context manager for Qt-compliant thread operations.

```python
@contextmanager
def safe_worker_context(thread_pool: QtCore.QThreadPool = None, pool_id: str = "temp_pool")
```
Context manager for safe worker operations.

### Configuration Functions

```python
def configure_pyqtgraph_for_qt_compatibility():
    pass
```
Configure PyQtGraph settings to minimize Qt connection warnings.

## Examples and Usage Patterns

### Basic Qt-Compliant Threading

```python
from xpcs_toolkit.threading.qt_compliant_thread_manager import (
    get_qt_compliant_thread_manager
)

# Initialize thread manager
manager = get_qt_compliant_thread_manager()

# Create and start a managed thread
thread = manager.create_managed_thread("data_processor")
if thread and manager.start_thread("data_processor"):
    print("Thread started successfully")

# Clean up
manager.stop_thread("data_processor")
manager.cleanup_finished_threads()
```

### Safe Worker Implementation

```python
from xpcs_toolkit.threading.enhanced_worker_safety import (
    SafeWorkerBase, ResourceType
)

class DataProcessor(SafeWorkerBase):
    def do_work(self):
        # Acquire resources safely
        self.acquire_resource("temp_file", ResourceType.TEMPORARY_FILE)

        # Report progress
        self.emit_progress_safe(0, 100, "Starting processing")

        # Simulate work with progress updates
        for i in range(100):
            self.check_cancelled_safe()  # Check for cancellation
            self.emit_progress_safe(i+1, 100, f"Processing item {i+1}")
            time.sleep(0.01)

        return "Processing complete"

# Use the worker
worker = DataProcessor("data_proc_1", max_retries=2)
# Worker automatically handles errors, retries, and cleanup
```

### PyQtGraph Widget Creation

```python
from xpcs_toolkit.plothandler.qt_signal_fixes import qt_connection_context
import pyqtgraph as pg

# Create PyQtGraph widgets without warnings
with qt_connection_context():
    image_view = pg.ImageView()
    plot_widget = pg.PlotWidget()

    # Make signal connections safely
    image_view.sigTimeChanged.connect(my_time_handler)
```

### Health Monitoring Setup

```python
from xpcs_toolkit.threading.thread_pool_integration_validator import (
    get_thread_pool_validator
)
from PySide6.QtCore import QThreadPool

# Initialize validator
validator = get_thread_pool_validator()

# Register thread pools for monitoring
main_pool = QThreadPool.globalInstance()
validator.register_thread_pool("main_pool", main_pool)

# Get health reports
health_summary = validator.get_health_summary()
validation_report = validator.generate_validation_report()
print(validation_report)
```

### Error Handling and Recovery

```python
from xpcs_toolkit.threading.enhanced_worker_safety import (
    SafeWorkerBase, WorkerErrorSeverity
)

class RobustWorker(SafeWorkerBase):
    def do_work(self):
        try:
            # Risky operation
            result = risky_operation()
            return result
        except SpecificError as e:
            # Handle specific error with custom recovery
            recovery_action = self.handle_error(
                e,
                context={"operation": "risky_operation"},
                severity=WorkerErrorSeverity.HIGH
            )

            if recovery_action == WorkerRecoveryAction.RETRY:
                # Retry logic is handled automatically
                return self.do_work()
            else:
                # Fallback operation
                return fallback_operation()
```

## Best Practices

### Thread Safety
1. Always use `QtCompliantThreadManager` for thread creation
2. Validate thread context before Qt operations
3. Use queued connections for cross-thread signals

### Resource Management
1. Always acquire resources through `SafeWorkerBase.acquire_resource()`
2. Provide cleanup callbacks for complex resources
3. Monitor resource usage with `ThreadPoolIntegrationValidator`

### Error Handling
1. Use `SafeWorkerBase` for automatic error recovery
2. Implement custom error handling for domain-specific errors
3. Monitor error rates through health metrics

### Performance
1. Use `qt_connection_context()` for PyQtGraph widget creation
2. Monitor thread pool health regularly
3. Implement proper progress reporting for long-running operations

This API reference provides comprehensive coverage of the Qt Compliance System's functionality. For implementation details and examples, refer to the source code and test files.