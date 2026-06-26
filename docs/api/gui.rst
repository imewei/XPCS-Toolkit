GUI Components
==============

Interactive XPCS data visualization interface with modern theming and user experience features.

.. currentmodule:: xpcsviewer

Main Application
----------------

The main GUI application window built with PySide6. Provides tab-based
interface for different analysis modes (SAXS 2D/1D, G2, stability, two-time).

The main application module (``xpcsviewer.xpcs_viewer``) provides the entry point
for the GUI.

.. note::
   The GUI components have limited automated testing due to their interactive
   nature. Manual testing and user feedback are primary validation methods.

Viewer Kernel
-------------

Backend kernel that bridges GUI and data processing operations.
Manages file collections, averaging operations, and plot state.

.. automodule:: xpcsviewer.viewer_kernel
   :members:
   :no-index:

File Locator
------------

File discovery and management utilities for XPCS datasets.
Handles file system navigation and dataset validation.

.. automodule:: xpcsviewer.file_locator
   :members:
   :no-index:

Default Settings
----------------

Application default settings (window size, etc.).

.. automodule:: xpcsviewer.default_setting
   :members:
   :no-index:

Command Line Interface
----------------------

The GUI is launched via the ``xpcsviewer-gui`` command. For CLI batch processing,
use the ``xpcsviewer`` command with subcommands.

See :doc:`cli` for complete CLI and entry point documentation.

GUI Modernization Components
----------------------------

The following modules provide modern UI/UX capabilities.

Theme System
~~~~~~~~~~~~

Light/dark mode theming with consistent visual styling.

**Modules:**

- :mod:`xpcsviewer.gui.theme` - Theme management and color tokens
- :mod:`xpcsviewer.gui.theme.manager` - Theme switching and application
- :mod:`xpcsviewer.gui.theme.tokens` - Design tokens for colors, spacing, typography
- :mod:`xpcsviewer.gui.theme.plot_themes` - Theme integration for PyQtGraph and Matplotlib

**Features:**

- Automatic system theme detection
- Persistent theme preferences
- QSS stylesheets for consistent widget styling
- Plot backend theme synchronization

State Management
~~~~~~~~~~~~~~~~

Session persistence and preferences management.

**Modules:**

- :mod:`xpcsviewer.gui.state` - State management utilities
- :mod:`xpcsviewer.gui.state.session_manager` - Session save/restore functionality
- :mod:`xpcsviewer.gui.state.preferences` - User preferences storage
- :mod:`xpcsviewer.gui.state.recent_paths` - Recently opened files tracking

**Features:**

- Automatic session persistence across restarts
- Window geometry and state restoration
- Recent files management with validation
- Type-safe preference access

Keyboard Shortcuts
~~~~~~~~~~~~~~~~~~

Customizable keyboard shortcut management.

**Modules:**

- :mod:`xpcsviewer.gui.shortcuts` - Shortcut management system
- :mod:`xpcsviewer.gui.shortcuts.shortcut_manager` - Shortcut registration and handling

**Features:**

- Centralized shortcut registry
- Conflict detection and resolution
- User-customizable keybindings
- Context-aware shortcut activation

Modern Widgets
~~~~~~~~~~~~~~

Enhanced UI components for improved user experience.

**Modules:**

- :mod:`xpcsviewer.gui.widgets` - Modern UI widgets
- :mod:`xpcsviewer.gui.widgets.command_palette` - VS Code-style command palette (Ctrl+Shift+P)
- :mod:`xpcsviewer.gui.widgets.toast_notification` - Non-intrusive status notifications
- :mod:`xpcsviewer.gui.widgets.drag_drop_list` - Enhanced drag-and-drop file handling

**Features:**

- Fuzzy search command palette
- Animated toast notifications with auto-dismiss
- Drag-and-drop support with visual feedback
- Theme-aware styling

Plot Handler Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Theme-aware plotting backends.

**Modules:**

- :mod:`xpcsviewer.plothandler` - Plot rendering backends
- :mod:`xpcsviewer.plothandler.plot_constants` - Theme-aware plot colors and styles
- :mod:`xpcsviewer.plothandler.matplot_qt` - Matplotlib Qt integration with theming
- :mod:`xpcsviewer.plothandler.pyqtgraph_handler` - PyQtGraph backend with theming

**Features:**

- Automatic plot theme switching with application theme
- Consistent color palettes across backends
- High-contrast modes for accessibility

Qt Compatibility Layer
~~~~~~~~~~~~~~~~~~~~~~

Unified import interface for Qt classes supporting both PySide6 and PyQt6.

.. automodule:: xpcsviewer.gui.qt_compat
   :members:
   :no-index:

Layout Helpers
~~~~~~~~~~~~~~

Convenience functions for building Qt layouts programmatically.

.. automodule:: xpcsviewer.gui.layout_helpers
   :members:
   :no-index:

SVG Icon System
~~~~~~~~~~~~~~~

Scalable SVG icons with theme-aware color replacement.

The icon system provides runtime SVG loading with automatic ``currentColor``
replacement, ensuring icons match the active light or dark theme.
Icons are cached after first load for efficient reuse across the UI.

.. automodule:: xpcsviewer.gui.icons
   :members:
   :no-index:

Category Tab Bar
~~~~~~~~~~~~~~~~

Visual tab grouping with category separators.

Overlays painted separator lines between logical tab groups (Scattering,
Correlation, Utilities) so users can visually parse the 12-tab interface.

.. automodule:: xpcsviewer.gui.widgets.category_tab_bar
   :members:
   :no-index:

