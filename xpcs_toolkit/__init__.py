# Handle imports gracefully for documentation builds
import os
from importlib.metadata import PackageNotFoundError, version

if os.environ.get("BUILDING_DOCS"):
    # Provide placeholder classes for documentation builds
    class XpcsFile:
        """Placeholder XpcsFile class for documentation builds."""

        pass

    class FileLocator:
        """Placeholder FileLocator class for documentation builds."""

        pass

    class ViewerKernel:
        """Placeholder ViewerKernel class for documentation builds."""

        pass

    class ViewerUI:
        """Placeholder ViewerUI class for documentation builds."""

        pass

    class XpcsViewer:
        """Placeholder XpcsViewer class for documentation builds."""

        pass

    # Create module-like objects for subpackages
    module = type("module", (), {})()
    plothandler = type("plothandler", (), {})()
    utils = type("utils", (), {})()

else:
    try:
        from xpcs_toolkit import module, plothandler, utils
        from xpcs_toolkit.file_locator import FileLocator
        from xpcs_toolkit.viewer_kernel import ViewerKernel
        from xpcs_toolkit.viewer_ui import ViewerUI
        from xpcs_toolkit.xpcs_file import XpcsFile as XpcsFile  # Explicit re-export
        from xpcs_toolkit.xpcs_viewer import XpcsViewer
    except ImportError:
        # For documentation builds where dependencies may not be available
        class XpcsFile:
            """Placeholder XpcsFile class for documentation builds."""

            pass

        class FileLocator:
            """Placeholder FileLocator class for documentation builds."""

            pass

        class ViewerKernel:
            """Placeholder ViewerKernel class for documentation builds."""

            pass

        class ViewerUI:
            """Placeholder ViewerUI class for documentation builds."""

            pass

        class XpcsViewer:
            """Placeholder XpcsViewer class for documentation builds."""

            pass


# Version handling
try:
    __version__ = version("xpcs-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback if package is not installed

__author__ = "Miaoqi Chu"
__credits__ = "Argonne National Laboratory"
