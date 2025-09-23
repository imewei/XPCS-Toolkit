"""
Qt Signal/Slot Connection Fixes for PyQtGraph Integration.

This module provides fixes and workarounds for Qt signal/slot connection issues
that occur with PyQtGraph components, particularly QStyleHints warnings.
"""

import logging
import warnings
from collections.abc import Callable
from contextlib import contextmanager

from PySide6 import QtWidgets
from PySide6.QtCore import QMetaObject, QObject, Qt, Signal

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class QtConnectionFixer:
    """
    Utility class to fix and prevent Qt signal/slot connection issues.

    This class provides methods to safely create PyQtGraph widgets with
    minimal Qt connection warnings, and to upgrade deprecated Qt4-style
    connections to modern Qt5+ syntax.
    """

    @staticmethod
    @contextmanager
    def suppress_qt_warnings():
        """
        Context manager to temporarily suppress Qt connection warnings.

        This is useful when creating PyQtGraph widgets that may generate
        QStyleHints connection warnings due to internal PyQtGraph implementation.
        """
        # Store original warning settings
        original_filters = warnings.filters.copy()

        # Suppress specific Qt warnings
        warnings.filterwarnings("ignore", category=UserWarning, module=".*qt.*")
        warnings.filterwarnings("ignore", message=".*QStyleHints.*")
        warnings.filterwarnings("ignore", message=".*unique connections require.*")
        warnings.filterwarnings("ignore", message=".*QObject::connect.*")

        # Also suppress Qt logging messages temporarily
        qt_logger = logging.getLogger("qt")
        qt_core_logger = logging.getLogger("qt.core")
        qt_qobject_logger = logging.getLogger("qt.core.qobject")

        original_level = qt_logger.level
        original_core_level = qt_core_logger.level
        original_qobject_level = qt_qobject_logger.level

        qt_logger.setLevel(logging.CRITICAL)
        qt_core_logger.setLevel(logging.CRITICAL)
        qt_qobject_logger.setLevel(logging.CRITICAL)

        # Set environment variable to suppress Qt debug output
        import os

        original_qt_logging = os.environ.get("QT_LOGGING_RULES", "")
        os.environ["QT_LOGGING_RULES"] = "qt.core.qobject.connect=false"

        try:
            yield
        finally:
            # Restore original settings
            warnings.filters[:] = original_filters
            qt_logger.setLevel(original_level)
            qt_core_logger.setLevel(original_core_level)
            qt_qobject_logger.setLevel(original_qobject_level)

            # Restore Qt logging rules
            if original_qt_logging:
                os.environ["QT_LOGGING_RULES"] = original_qt_logging
            else:
                os.environ.pop("QT_LOGGING_RULES", None)

    @staticmethod
    def create_safe_imageview(*args, **kwargs):
        """
        Create PyQtGraph ImageView with suppressed connection warnings.

        Args:
            *args: Arguments to pass to ImageView constructor
            **kwargs: Keyword arguments to pass to ImageView constructor

        Returns:
            PyQtGraph ImageView instance
        """
        try:
            import pyqtgraph as pg

            with QtConnectionFixer.suppress_qt_warnings():
                imageview = pg.ImageView(*args, **kwargs)

            logger.debug("Created ImageView with suppressed Qt warnings")
            return imageview

        except ImportError:
            logger.error("PyQtGraph not available")
            raise
        except Exception as e:
            logger.warning(f"Failed to create ImageView: {e}")
            raise

    @staticmethod
    def validate_signal_connection(
        signal: Signal,
        slot: Callable,
        connection_type: Qt.ConnectionType = Qt.ConnectionType.AutoConnection,
    ) -> bool:
        """
        Validate and establish a Qt5+ compliant signal/slot connection.

        Args:
            signal: Qt signal object
            slot: Callable slot function or method
            connection_type: Qt connection type

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Validate signal
            if not hasattr(signal, "connect"):
                logger.warning("Invalid signal object - missing connect method")
                return False

            # Validate slot
            if not callable(slot):
                logger.warning("Invalid slot - not callable")
                return False

            # Check for Qt4-style string connections (should be avoided)
            if isinstance(signal, str) or isinstance(slot, str):
                logger.warning(
                    "Detected Qt4-style string-based connection - should be updated"
                )
                return False

            # Establish connection with Qt5+ syntax
            signal.connect(slot, connection_type)
            logger.debug(
                f"Successfully connected signal to {slot.__name__ if hasattr(slot, '__name__') else 'slot'}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to establish signal/slot connection: {e}")
            return False

    @staticmethod
    def upgrade_qt4_connection(
        obj: QObject,
        signal_name: str,
        slot: Callable,
        signal_signature: str | None = None,
    ) -> bool:
        """
        Upgrade a Qt4-style connection to Qt5+ syntax.

        Args:
            obj: QObject containing the signal
            signal_name: Name of the signal (e.g., 'clicked')
            slot: Slot function to connect
            signal_signature: Optional signal signature for overloaded signals

        Returns:
            True if upgrade successful, False otherwise
        """
        try:
            # Get the signal object
            signal = getattr(obj, signal_name)

            if signal is None:
                logger.warning(f"Signal '{signal_name}' not found on object {obj}")
                return False

            # Handle overloaded signals if signature is specified
            if signal_signature is not None:
                # This handles cases like clicked["bool"] -> clicked
                signal = signal  # In Qt5+, we typically use the base signal
                logger.debug(
                    f"Handling overloaded signal {signal_name} with signature {signal_signature}"
                )

            # Establish Qt5+ connection
            return QtConnectionFixer.validate_signal_connection(signal, slot)

        except Exception as e:
            logger.error(f"Failed to upgrade Qt4 connection: {e}")
            return False


class PyQtGraphWrapper:
    """
    Wrapper for PyQtGraph components that minimizes Qt connection warnings.

    This class provides static methods to create PyQtGraph widgets with proper
    Qt signal/slot connection handling, reducing or eliminating connection warnings
    that commonly occur with PyQtGraph components like ImageView widgets.

    The wrapper uses the QtConnectionFixer context manager to suppress cosmetic
    Qt warnings during widget creation while maintaining full functionality.
    """

    @staticmethod
    def create_imageview_dev(*args, **kwargs):
        """
        Create ImageViewDev with Qt connection issue mitigation.

        Args:
            *args: Arguments for ImageViewDev
            **kwargs: Keyword arguments for ImageViewDev

        Returns:
            ImageViewDev instance
        """
        try:
            from .pyqtgraph_handler import ImageViewDev

            with QtConnectionFixer.suppress_qt_warnings():
                imageview = ImageViewDev(*args, **kwargs)

            logger.debug("Created ImageViewDev with Qt warning suppression")
            return imageview

        except Exception as e:
            logger.warning(f"Failed to create ImageViewDev: {e}")
            raise

    @staticmethod
    def create_imageview_plotitem(*args, **kwargs):
        """
        Create ImageViewPlotItem with Qt connection issue mitigation.

        Args:
            *args: Arguments for ImageViewPlotItem
            **kwargs: Keyword arguments for ImageViewPlotItem

        Returns:
            ImageViewPlotItem instance
        """
        try:
            from .pyqtgraph_handler import ImageViewPlotItem

            with QtConnectionFixer.suppress_qt_warnings():
                imageview = ImageViewPlotItem(*args, **kwargs)

            logger.debug("Created ImageViewPlotItem with Qt warning suppression")
            return imageview

        except Exception as e:
            logger.warning(f"Failed to create ImageViewPlotItem: {e}")
            raise


# Utility functions for common signal/slot operations
def safe_connect(
    signal: Signal,
    slot: Callable,
    connection_type: Qt.ConnectionType = Qt.ConnectionType.AutoConnection,
) -> bool:
    """
    Safely connect a signal to a slot with validation.

    Args:
        signal: Qt signal
        slot: Callable slot
        connection_type: Connection type

    Returns:
        True if successful
    """
    return QtConnectionFixer.validate_signal_connection(signal, slot, connection_type)


def safe_disconnect(signal: Signal, slot: Callable | None = None) -> bool:
    """
    Safely disconnect a signal from a slot.

    Args:
        signal: Qt signal
        slot: Optional specific slot to disconnect

    Returns:
        True if successful
    """
    try:
        if slot is not None:
            signal.disconnect(slot)
        else:
            signal.disconnect()
        return True
    except (TypeError, RuntimeError) as e:
        logger.debug(f"Signal disconnect info: {e}")
        return True  # Already disconnected is not an error
    except Exception as e:
        logger.warning(f"Failed to disconnect signal: {e}")
        return False


@contextmanager
def qt_connection_context():
    """
    Context manager for Qt operations that may generate connection warnings.

    This context manager temporarily suppresses Qt connection warnings that are
    generated by PyQtGraph widgets during creation. It's particularly useful for
    eliminating QStyleHints connection warnings when creating ImageView widgets.

    Args:
        None

    Yields:
        None: Context for executing Qt operations with warning suppression

    Returns:
        None

    Example:
        >>> with qt_connection_context():
        ...     # Create PyQtGraph widgets without warnings
        ...     widget = pg.ImageView()
        ...     # Make signal connections safely
        ...     widget.sigTimeChanged.connect(my_handler)

    Note:
        This context manager only suppresses cosmetic warnings and does not
        affect the functionality of Qt operations. All signal/slot connections
        remain fully functional.
    """
    with QtConnectionFixer.suppress_qt_warnings():
        yield


# Decorator for functions that create Qt widgets
def suppress_qt_connection_warnings(func):
    """
    Decorator to suppress Qt connection warnings for widget creation functions.

    This decorator wraps a function to suppress Qt connection warnings during
    execution, particularly useful for functions that create PyQtGraph widgets
    which may generate cosmetic QStyleHints warnings.

    Args:
        func: The function to wrap with Qt warning suppression

    Returns:
        Wrapped function that executes with Qt warnings suppressed

    Usage::

        @suppress_qt_connection_warnings
        def create_widget():
            return pg.ImageView()
    """

    def wrapper(*args, **kwargs):
        with QtConnectionFixer.suppress_qt_warnings():
            return func(*args, **kwargs)

    return wrapper


# Configuration function to set up PyQtGraph for minimal warnings
def configure_pyqtgraph_for_qt_compatibility():
    """
    Configure PyQtGraph settings to minimize Qt connection warnings.

    This should be called early in the application initialization to set up
    PyQtGraph with optimal configuration options that reduce Qt connection warnings.

    Args:
        None

    Returns:
        None
    """
    try:
        import os

        import pyqtgraph as pg

        # Set environment variable to suppress Qt connection warnings globally
        os.environ["QT_LOGGING_RULES"] = "qt.core.qobject.connect=false"

        # Set PyQtGraph configuration options that may reduce warnings
        pg.setConfigOptions(
            imageAxisOrder="row-major",  # This was already set
            useNumba=False,  # Disable numba to avoid potential issues
            enableExperimental=False,  # Disable experimental features
        )

        # Configure Qt application attributes for better compatibility
        app = QtWidgets.QApplication.instance()
        if app is not None:
            # Set attributes that may reduce connection warnings
            app.setAttribute(
                Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True
            )

        logger.info("Configured PyQtGraph for Qt compatibility")

    except ImportError:
        logger.warning("PyQtGraph not available for configuration")
    except Exception as e:
        logger.warning(f"Failed to configure PyQtGraph: {e}")


# Legacy connection detector and upgrader
class LegacyConnectionDetector:
    """
    Detects and helps upgrade legacy Qt4-style connections.
    """

    @staticmethod
    def scan_for_legacy_connections(obj: QObject) -> list[str]:
        """
        Scan an object for potential legacy signal connections.

        Args:
            obj: QObject to scan

        Returns:
            List of potential legacy connection patterns found
        """
        issues = []

        # This is a basic scan - in practice, legacy connections are
        # found through code review rather than runtime detection
        try:
            # Check object's signal list
            meta = obj.metaObject()
            for i in range(meta.methodCount()):
                method = meta.method(i)
                if method.methodType() == QMetaObject.MethodType.Signal:
                    signal_name = method.name().data().decode()
                    # Check if signal has been connected with old syntax
                    # (This is difficult to detect at runtime)
                    logger.debug(f"Found signal: {signal_name}")

        except Exception as e:
            logger.debug(f"Legacy connection scan failed: {e}")

        return issues

    @staticmethod
    def suggest_modern_syntax(legacy_pattern: str) -> str:
        """
        Suggest modern Qt5+ syntax for a legacy connection pattern.

        Args:
            legacy_pattern: Legacy connection pattern

        Returns:
            Suggested modern syntax
        """
        suggestions = {
            'clicked["bool"]': "clicked",
            'valueChanged["int"]': "valueChanged",
            'currentIndexChanged["int"]': "currentIndexChanged",
            'stateChanged["int"]': "stateChanged",
            'textChanged["QString"]': "textChanged",
        }

        return suggestions.get(legacy_pattern, legacy_pattern)
