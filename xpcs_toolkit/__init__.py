from importlib.metadata import PackageNotFoundError, version

# Handle imports gracefully for documentation builds
try:
    from xpcs_toolkit.xpcs_file import XpcsFile as XpcsFile  # Explicit re-export
except ImportError:
    # For documentation builds where dependencies may not be available
    class XpcsFile:
        """Placeholder XpcsFile class for documentation builds."""

        pass


# Version handling
try:
    __version__ = version("xpcs-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback if package is not installed

__author__ = "Miaoqi Chu"
__credits__ = "Argonne National Laboratory"
