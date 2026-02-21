"""Atomic file I/O utilities.

Provides ``safe_json_write`` which writes JSON data to a temporary file
then atomically replaces the target — preventing data corruption if the
process is interrupted mid-write.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_json_write(path: str | Path, data: dict, *, indent: int = 2) -> None:
    """Write *data* as JSON to *path* atomically.

    Creates the parent directory if needed, writes to a temporary file in the
    same directory, then calls :func:`os.replace` to swap the temp file into
    place (atomic on POSIX).

    Parameters
    ----------
    path : str | Path
        Destination file path.
    data : dict
        JSON-serialisable dictionary.
    indent : int
        JSON indentation (default 2).

    Raises
    ------
    OSError
        If the write or replace fails.
    TypeError
        If *data* is not JSON-serialisable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
