GUI Components
==============

Interactive XPCS data visualization interface with modern theming and user experience features.

.. note::
   For complete API documentation of all GUI modules, see :doc:`xpcs_toolkit`.

.. currentmodule:: xpcsviewer

Main Application
----------------

The main GUI application window built with PySide6. Provides tab-based
interface for different analysis modes (SAXS 2D/1D, G2, stability, two-time).

See :mod:`xpcsviewer.xpcs_viewer` for complete API documentation.

.. note::
   The GUI components have limited automated testing due to their interactive
   nature. Manual testing and user feedback are primary validation methods.

Viewer Kernel
-------------

Backend kernel that bridges GUI and data processing operations.
Manages file collections, averaging operations, and plot state.

See :mod:`xpcsviewer.viewer_kernel` for complete API documentation.

File Locator
------------

File discovery and management utilities for XPCS datasets.
Handles file system navigation and dataset validation.

See :mod:`xpcsviewer.file_locator` for complete API documentation.

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
- :mod:`xpcsviewer.gui.widgets.error_dialog` - Actionable error display with copy-to-clipboard
- :mod:`xpcsviewer.gui.widgets.empty_state` - Empty state placeholders with action buttons

**Features:**

- Fuzzy search command palette
- Animated toast notifications with auto-dismiss
- Drag-and-drop support with visual feedback
- Theme-aware styling
- Actionable error dialogs with technical details

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
