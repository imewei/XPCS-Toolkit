=====
Usage
=====

XPCS Viewer provides both a graphical user interface (GUI) and a command-line interface (CLI)
for analyzing X-ray Photon Correlation Spectroscopy data.

Command-Line Entry Points
=========================

The package provides several entry points:

.. list-table:: Entry Points
   :widths: 25 15 60
   :header-rows: 1

   * - Command
     - Type
     - Description
   * - ``xpcsviewer-gui``
     - GUI
     - Launch the graphical interface
   * - ``xpcsviewer``
     - CLI
     - Command-line interface (requires subcommand)

GUI Usage
=========

Launch the GUI
--------------

.. code-block:: bash

   # Launch GUI in current directory
   xpcsviewer-gui

   # Launch GUI with specific data path
   xpcsviewer-gui /path/to/hdf/data

   # Using --path option
   xpcsviewer-gui --path /path/to/data

   # With custom label style
   xpcsviewer-gui --path /data --label_style "custom_style"

   # Enable debug logging
   xpcsviewer-gui --log-level DEBUG

GUI Options
-----------

.. code-block:: text

   usage: xpcsviewer-gui [-h] [--version] [--path PATH]
                         [--label_style LABEL_STYLE]
                         [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                         [positional_path]

   positional arguments:
     positional_path       Path to the result folder

   options:
     -h, --help            Show help message and exit
     --version             Show program's version number and exit
     --path PATH           Path to the result folder (default: ./)
     --label_style STYLE   Custom label style for file identification
     --log-level LEVEL     Set logging level (default: INFO)

GUI Workflow
------------

1. **Load Data**: Launch with a directory containing HDF5 files
2. **Select Files**: Files appear in the Source list; select and add to Target
3. **Choose Analysis**: Select an analysis tab (SAXS 2D, G2, Twotime, etc.)
4. **View Results**: Interactive plots display in the main area

Keyboard Shortcuts
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 40
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``Ctrl+O``
     - Open folder
   * - ``Ctrl+R``
     - Reload data
   * - ``Ctrl+P``
     - Open Command Palette
   * - ``Ctrl+Shift+A``
     - Add to target selection
   * - ``Ctrl+Tab``
     - Next tab
   * - ``Ctrl+Shift+Tab``
     - Previous tab
   * - ``Ctrl+L``
     - View logs
   * - ``Esc``
     - Clear selection
   * - ``F1``
     - Show documentation

Analysis Tabs
~~~~~~~~~~~~~

.. list-table::
   :widths: 20 50
   :header-rows: 1

   * - Tab
     - Description
   * - SAXS 2D
     - Integrated 2D scattering patterns with ROI support
   * - SAXS 1D
     - Radial averaging of 2D patterns
   * - Stability
     - Sample stability monitoring over time
   * - Intensity vs Time
     - Time-series intensity analysis
   * - G2
     - Multi-tau correlation analysis with fitting
   * - Diffusion
     - Diffusion analysis and fitting
   * - Twotime
     - Two-time correlation maps (q-phi selection)
   * - Qmap
     - Q-space mapping visualization
   * - Average
     - File averaging tools for batch processing
   * - Metadata
     - Dataset information and parameters

CLI Usage
=========

The CLI provides batch processing capabilities without launching the GUI.

CLI Help
--------

.. code-block:: bash

   # Show available commands
   xpcsviewer --help

   # Show version
   xpcsviewer --version

CLI Output:

.. code-block:: text

   usage: xpcsviewer [-h] [--version]
                     [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                     {twotime} ...

   XPCS Viewer CLI: Command-line tools for XPCS data analysis

   positional arguments:
     {twotime}             Available commands
       twotime             Batch process twotime correlation data

   options:
     -h, --help            Show help message and exit
     --version             Show program's version number and exit
     --log-level LEVEL     Set logging level (default: INFO)

   Use 'xpcsviewer-gui' to launch the graphical interface.

Twotime Batch Processing
------------------------

Generate twotime correlation images from HDF5 files.

.. code-block:: bash

   # Process all phi angles at a specific q-value
   xpcsviewer twotime --input /path/to/data --output /results --q 0.05

   # Process all q values at a specific phi angle
   xpcsviewer twotime --input /path/to/data --output /results --phi 45

   # Process a specific q-phi pair (single image)
   xpcsviewer twotime --input file.h5 --output /results --q-phi "0.05,45"

   # Full example with all options
   xpcsviewer twotime \
       --input /experiment_data \
       --output /results/plots \
       --q 0.05 \
       --dpi 300 \
       --format pdf \
       --log-level INFO

Twotime Options
~~~~~~~~~~~~~~~

.. code-block:: text

   usage: xpcsviewer twotime [-h] --input INPUT --output OUTPUT
                             (--q Q | --phi PHI | --q-phi Q_PHI)
                             [--dpi DPI] [--format {png,jpg,jpeg,pdf,svg}]

   options:
     -h, --help            Show help message and exit
     --input, -i INPUT     HDF file path or directory containing HDF files
     --output, -o OUTPUT   Output directory for generated images
     --q Q                 Q-value to process (generates all phi angles)
     --phi PHI             Phi-value to process (generates all q values)
     --q-phi Q_PHI         Specific q-phi pair as 'q,phi' (single image)
     --dpi DPI             Image resolution in DPI (default: 300)
     --format FORMAT       Image format: png, jpg, pdf, svg (default: png)

Python API
==========

Import the Package
------------------

.. code-block:: python

   import xpcsviewer

Load XPCS Data
--------------

.. code-block:: python

   from xpcsviewer import XpcsFile

   # Load a single HDF5 file
   xf = XpcsFile('/path/to/data.h5')

   # Check analysis type
   print(f"Analysis type: {xf.atype}")

   # Access data attributes
   print(f"G2 shape: {xf.g2.shape}")
   print(f"Delay times: {xf.tel}")

G2 Analysis
-----------

.. code-block:: python

   from xpcsviewer.module import g2mod

   # Load multiple files
   xf_list = [XpcsFile(f) for f in ['file1.h5', 'file2.h5']]

   # Get G2 correlation data
   q, tel, g2, g2_err, labels = g2mod.get_data(xf_list)

SAXS Analysis
-------------

.. code-block:: python

   from xpcsviewer.module import saxs1d

   # Get SAXS 1D data
   q_values = xf.saxs_1d_q
   intensities = xf.saxs_1d

Configuration
=============

Configuration files are stored in ``~/.xpcsviewer/``:

- **Theme preferences**: Light/dark/system mode
- **Session state**: Workspace persistence
- **Recent paths**: Recently opened directories
- **Keyboard shortcuts**: Custom key bindings
- **Logs**: Application logs in ``~/.xpcsviewer/logs/``

Logging
-------

Control logging verbosity:

.. code-block:: bash

   # Debug level (most verbose)
   xpcsviewer-gui --log-level DEBUG

   # Warning level (errors and warnings only)
   xpcsviewer twotime -i data -o out --q 0.05 --log-level WARNING

View logs:

.. code-block:: bash

   # From GUI: Help → View Logs (Ctrl+L)
   # Or directly:
   ls ~/.xpcsviewer/logs/

Exit Codes
----------

.. list-table::
   :widths: 10 50
   :header-rows: 1

   * - Code
     - Meaning
   * - 0
     - Success
   * - 1
     - General failure
   * - 2
     - Missing GUI dependencies
