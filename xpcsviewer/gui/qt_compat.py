"""Qt binding compatibility layer.

This module provides a unified import interface for Qt classes,
supporting both PySide6 and PyQt6 via the qtpy abstraction layer.

Usage:
    from xpcsviewer.gui.qt_compat import QtCore, QtWidgets, Signal

Environment Variables:
    QT_API: Set to 'pyside6' or 'pyqt6' before importing.
            Defaults to 'pyside6' if not set.

Example:
    # Import Qt classes through the compatibility layer
    from xpcsviewer.gui.qt_compat import (
        QtCore, QtWidgets, QtGui,
        Signal, Slot, QObject, Property,
        QMainWindow, QWidget, QApplication,
    )
"""

import os

# Set default Qt binding to PySide6 for backward compatibility
# This must be set BEFORE importing qtpy
os.environ.setdefault("QT_API", "pyside6")

# Import Qt modules through qtpy
from qtpy import QtCore, QtGui, QtWidgets

# Import commonly used Qt classes from QtCore
from qtpy.QtCore import (
    Property,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QSize,
    QStringListModel,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
    Slot,
)

# Import commonly used Qt classes from QtWidgets
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QStyleFactory,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Import commonly used Qt classes from QtGui
from qtpy.QtGui import (
    QAction as QGuiAction,
    QColor,
    QDesktopServices,
    QFont,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPalette,
    QPixmap,
    QShortcut,
)

# Import QMetaMethod and QMetaObject for signal inspection
from qtpy.QtCore import QMetaMethod, QMetaObject

# Re-export __all__ for explicit public API
__all__ = [
    # Qt Modules
    "QtCore",
    "QtGui",
    "QtWidgets",
    # QtCore classes
    "Property",
    "QMetaMethod",
    "QMetaObject",
    "QModelIndex",
    "QObject",
    "QPoint",
    "QRect",
    "QSize",
    "QStringListModel",
    "Qt",
    "QThread",
    "QThreadPool",
    "QTimer",
    "QUrl",
    "Signal",
    "Slot",
    # QtWidgets classes
    "QAbstractItemView",
    "QAction",
    "QActionGroup",
    "QApplication",
    "QButtonGroup",
    "QCheckBox",
    "QComboBox",
    "QDialog",
    "QDockWidget",
    "QDoubleSpinBox",
    "QFileDialog",
    "QFormLayout",
    "QFrame",
    "QGroupBox",
    "QHBoxLayout",
    "QLabel",
    "QLayout",
    "QLineEdit",
    "QListView",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMenuBar",
    "QMessageBox",
    "QProgressBar",
    "QPushButton",
    "QSizePolicy",
    "QSpinBox",
    "QSplitter",
    "QStatusBar",
    "QStyle",
    "QStyleFactory",
    "QTabWidget",
    "QTextEdit",
    "QToolBar",
    "QToolButton",
    "QTreeWidget",
    "QTreeWidgetItem",
    "QVBoxLayout",
    "QWidget",
    # QtGui classes
    "QColor",
    "QDesktopServices",
    "QFont",
    "QGuiAction",
    "QGuiApplication",
    "QIcon",
    "QKeySequence",
    "QPalette",
    "QPixmap",
    "QShortcut",
]
