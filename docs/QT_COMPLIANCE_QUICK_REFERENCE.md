# Qt Compliance Quick Reference Card

## 🚨 Emergency Reference - Qt Compliance System

> **Goal**: Maintain ZERO Qt errors in XPCS Toolkit

## ⚡ Quick Fixes

### QTimer Threading Violation
```python
# ❌ WRONG
timer = QTimer()

# ✅ CORRECT
from xpcs_toolkit.threading.qt_compliant_thread_manager import get_qt_compliant_thread_manager
manager = get_qt_compliant_thread_manager()
timer = manager.timer_manager.create_timer("my_timer")
```

### QStyleHints Warnings (PyQtGraph)
```python
# ❌ WRONG
image_view = pg.ImageView()

# ✅ CORRECT
from xpcs_toolkit.plothandler.qt_signal_fixes import qt_connection_context
with qt_connection_context():
    image_view = pg.ImageView()
```

### Signal/Slot Connection Issues
```python
# ❌ WRONG (Qt4 style)
button.clicked["bool"].connect(handler)

# ✅ CORRECT (Qt5+ with safety)
from xpcs_toolkit.plothandler.qt_signal_fixes import safe_connect
safe_connect(button.clicked, handler)
```

## 🏗️ Worker Creation Template

```python
from xpcs_toolkit.threading.enhanced_worker_safety import SafeWorkerBase, ResourceType

class MyWorker(SafeWorkerBase):
    def __init__(self, data):
        super().__init__(
            worker_id=f"my_worker_{id(data)}",
            max_retries=3,
            timeout_seconds=300.0
        )
        self.data = data

    def do_work(self):
        # 1. Acquire resources
        self.acquire_resource("temp_file", ResourceType.TEMPORARY_FILE)

        # 2. Report progress
        self.emit_progress_safe(0, 100, "Starting")

        # 3. Check cancellation
        self.check_cancelled_safe()

        # 4. Do work
        result = process_data(self.data)

        # 5. Auto cleanup
        return result
```

## 🔧 Essential Imports

```python
# Thread management
from xpcs_toolkit.threading.qt_compliant_thread_manager import (
    get_qt_compliant_thread_manager, QtThreadSafetyValidator
)

# Worker safety
from xpcs_toolkit.threading.enhanced_worker_safety import (
    SafeWorkerBase, ResourceType, create_safe_worker
)

# Signal/slot safety
from xpcs_toolkit.plothandler.qt_signal_fixes import (
    qt_connection_context, safe_connect, safe_disconnect
)

# Health monitoring
from xpcs_toolkit.threading.thread_pool_integration_validator import (
    get_thread_pool_validator
)
```

## 🧪 Quick Tests

```bash
# Validate Qt compliance
python test_thread_management_quick.py

# Full validation
python test_background_thread_management.py

# Documentation tests
python -m pytest tests/documentation/test_qt_compliance_documentation.py -v
```

## 📊 Health Check

```python
# Quick health check
validator = get_thread_pool_validator()
summary = validator.get_health_summary()
print(f"Issues: {summary['total_issues']}")

# Full report
report = validator.generate_validation_report()
print(report)
```

## 🎯 Thread Safety Rules

1. **GUI operations** → Main thread only
2. **QTimer creation** → Qt threads only
3. **PyQtGraph widgets** → Use `qt_connection_context()`
4. **Worker threads** → Inherit from `SafeWorkerBase`
5. **Resources** → Always use `acquire_resource()`

## 🚀 Performance Tips

- Batch PyQtGraph operations in `qt_connection_context()`
- Use `QueuedConnection` for cross-thread signals
- Monitor with `ThreadPoolIntegrationValidator`
- Implement progress reporting for long operations

## 🆘 Troubleshooting

| Error | Solution |
|-------|----------|
| Timer threading violation | Use `QtCompliantTimerManager` |
| QStyleHints warnings | Use `qt_connection_context()` |
| Signal connection errors | Use `safe_connect()` |
| Resource leaks | Use `SafeWorkerBase.acquire_resource()` |
| Worker crashes | Use `SafeWorkerBase` error handling |

## 📈 Success Metrics

- **Qt Errors**: 0 (was 8+)
- **Thread Safety**: 100% compliant
- **Resource Leaks**: 0 detected
- **Performance Impact**: < 5%

---

**Remember**: Every Qt operation should be thread-safe and compliant. When in doubt, use the safety utilities! 🛡️
