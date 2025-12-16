GUI Components
==============

Interactive XPCS data visualization interface with modern theming and user experience features.

.. note::
   For complete API documentation of all GUI modules, see :doc:`../xpcs_toolkit`.

.. currentmodule:: xpcs_toolkit

Main Application
----------------

The main GUI application window built with PySide6. Provides tab-based
interface for different analysis modes (SAXS 2D/1D, G2, stability, two-time).

See :mod:`xpcs_toolkit.xpcs_viewer` for complete API documentation.

.. note::
   The GUI components have limited automated testing due to their interactive
   nature. Manual testing and user feedback are primary validation methods.

Viewer Kernel
-------------

Backend kernel that bridges GUI and data processing operations.
Manages file collections, averaging operations, and plot state.

See :mod:`xpcs_toolkit.viewer_kernel` for complete API documentation.

File Locator
------------

File discovery and management utilities for XPCS datasets.
Handles file system navigation and dataset validation.

See :mod:`xpcs_toolkit.file_locator` for complete API documentation.

Command Line Interface
----------------------

Command-line entry points for launching the GUI application.
Supports various startup configurations and directory specifications.

See :mod:`xpcs_toolkit.cli` for complete API documentation.

GUI Modernization Components
----------------------------

The following modules provide modern UI/UX capabilities added in v1.1.0.

Theme System
~~~~~~~~~~~~

Light/dark mode theming with consistent visual styling.

**Modules:**

- :mod:`xpcs_toolkit.gui.theme` - Theme management and color tokens
- :mod:`xpcs_toolkit.gui.theme.manager` - Theme switching and application
- :mod:`xpcs_toolkit.gui.theme.tokens` - Design tokens for colors, spacing, typography
- :mod:`xpcs_toolkit.gui.theme.plot_themes` - Theme integration for PyQtGraph and Matplotlib

**Features:**

- Automatic system theme detection
- Persistent theme preferences
- QSS stylesheets for consistent widget styling
- Plot backend theme synchronization

State Management
~~~~~~~~~~~~~~~~

Session persistence and preferences management.

**Modules:**

- :mod:`xpcs_toolkit.gui.state` - State management utilities
- :mod:`xpcs_toolkit.gui.state.session_manager` - Session save/restore functionality
- :mod:`xpcs_toolkit.gui.state.preferences` - User preferences storage
- :mod:`xpcs_toolkit.gui.state.recent_paths` - Recently opened files tracking

**Features:**

- Automatic session persistence across restarts
- Window geometry and state restoration
- Recent files management with validation
- Type-safe preference access

Keyboard Shortcuts
~~~~~~~~~~~~~~~~~~

Customizable keyboard shortcut management.

**Modules:**

- :mod:`xpcs_toolkit.gui.shortcuts` - Shortcut management system
- :mod:`xpcs_toolkit.gui.shortcuts.shortcut_manager` - Shortcut registration and handling

**Features:**

- Centralized shortcut registry
- Conflict detection and resolution
- User-customizable keybindings
- Context-aware shortcut activation

Modern Widgets
~~~~~~~~~~~~~~

Enhanced UI components for improved user experience.

**Modules:**

- :mod:`xpcs_toolkit.gui.widgets` - Modern UI widgets
- :mod:`xpcs_toolkit.gui.widgets.command_palette` - VS Code-style command palette (Ctrl+Shift+P)
- :mod:`xpcs_toolkit.gui.widgets.toast_notification` - Non-intrusive status notifications
- :mod:`xpcs_toolkit.gui.widgets.drag_drop_list` - Enhanced drag-and-drop file handling

**Features:**

- Fuzzy search command palette
- Animated toast notifications with auto-dismiss
- Drag-and-drop support with visual feedback
- Theme-aware styling

Plot Handler Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Theme-aware plotting backends.

**Modules:**

- :mod:`xpcs_toolkit.plothandler` - Plot rendering backends
- :mod:`xpcs_toolkit.plothandler.plot_constants` - Theme-aware plot colors and styles
- :mod:`xpcs_toolkit.plothandler.matplot_qt` - Matplotlib Qt integration with theming
- :mod:`xpcs_toolkit.plothandler.pyqtgraph_handler` - PyQtGraph backend with theming

**Features:**

- Automatic plot theme switching with application theme
- Consistent color palettes across backends
- High-contrast modes for accessibility
