from importlib.metadata import version, PackageNotFoundError
from xpcs_toolkit.xpcs_file import XpcsFile as XpcsFile  # Explicit re-export

# Version handling
try:
    __version__ = version("xpcs-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback if package is not installed

__author__ = "Miaoqi Chu"
__credits__ = "Argonne National Laboratory"
