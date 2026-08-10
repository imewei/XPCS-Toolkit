"""Exceptions raised by the raw reader stack."""

from __future__ import annotations


class RawDataReadError(Exception):
    """Uniform error the GUI catches, wrapping whatever a parser raised.

    Every format loader and beamline reader in this package can fail with a
    different underlying exception type (IOError, ValueError,
    FileNotFoundError, ...). Callers outside this package only need to
    catch this one type.
    """

    def __init__(self, path: str, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"failed to read raw data file {path!r}: {cause}")
