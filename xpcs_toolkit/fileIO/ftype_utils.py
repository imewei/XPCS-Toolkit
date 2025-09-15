# Standard library imports
import os

# Third-party imports
import h5py

# Local imports
from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


def isNeXusFile(fname):
    logger.debug(f"Checking if {fname} is NeXus file")
    try:
        with h5py.File(fname, "r") as f:
            if "/entry/instrument/bluesky/metadata/" in f:
                logger.debug(f"{fname} identified as NeXus file")
                return True
        logger.debug(f"{fname} is not a NeXus file")
        return False
    except Exception as e:
        logger.error(f"Error checking NeXus file {fname}: {e}")
        return False


def isLegacyFile(fname):
    logger.debug(f"Checking if {fname} is legacy file")
    try:
        with h5py.File(fname, "r") as f:
            if "/xpcs/Version" in f:
                logger.debug(f"{fname} identified as legacy file")
                return True
    except Exception as e:
        logger.error(f"Error checking legacy file {fname}: {e}")
        return False


def get_ftype(fname: str):
    logger.debug(f"Determining file type for {fname}")

    if not os.path.isfile(fname):
        logger.warning(f"File does not exist: {fname}")
        return False

    try:
        if isLegacyFile(fname):
            logger.info(f"File {fname} identified as legacy type")
            return "legacy"
        if isNeXusFile(fname):
            logger.info(f"File {fname} identified as nexus type")
            return "nexus"
        logger.warning(f"Unknown file type for {fname}")
        return False
    except Exception as e:
        logger.error(f"Error determining file type for {fname}: {e}")
        return False
