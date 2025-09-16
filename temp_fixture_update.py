@pytest.fixture
def memory_limited_environment():
    """Mock environment with limited memory."""
    # Patch both the direct psutil import and the module-specific import
    with patch("psutil.virtual_memory") as mock_memory, \
         patch("xpcs_toolkit.utils.memory_utils.psutil.virtual_memory") as mock_memory_utils:

        memory_mock = Mock(
            total=1024 * 1024 * 1024,  # 1 GB
            available=128 * 1024 * 1024,  # 128 MB available
            percent=87.5,  # 87.5% memory usage
        )

        mock_memory.return_value = memory_mock
        mock_memory_utils.return_value = memory_mock

        # Clear the cached memory monitor to ensure fresh data
        try:
            from xpcs_toolkit.utils.memory_utils import get_cached_memory_monitor
            monitor = get_cached_memory_monitor()
            with monitor._lock:
                monitor._cached_status = None  # Force fresh read
        except (ImportError, AttributeError):
            pass  # Monitor might not be available or have different structure

        yield mock_memory
