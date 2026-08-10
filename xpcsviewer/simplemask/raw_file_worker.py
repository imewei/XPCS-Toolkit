"""Qt worker that loads a raw detector file off the GUI thread.

Runs in a QThreadPool via xpcsviewer.threading.async_workers.BaseAsyncWorker
-- see xpcsviewer/threading/async_workers.py:152-398 for the run()/signals
contract this relies on (signals.finished / signals.error are emitted
automatically by the base class; this file only implements do_work()).
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from xpcsviewer.simplemask.reader import get_reader
from xpcsviewer.simplemask.reader.adapter import to_kernel_metadata
from xpcsviewer.simplemask.reader.exceptions import RawDataReadError
from xpcsviewer.threading.async_workers import BaseAsyncWorker


class RawFileLoadResult(TypedDict):
    """Shape of the dict RawFileLoadWorker.do_work() returns on success."""

    scattering: np.ndarray
    metadata: dict[str, Any]
    metadata_is_placeholder: bool


class RawFileLoadWorker(BaseAsyncWorker):
    """Loads a raw detector file and adapts it to SimpleMaskKernel's schema.

    On success, ``signals.finished`` carries a :class:`RawFileLoadResult`.
    On failure, ``signals.error`` carries the wrapped :class:`RawDataReadError`
    message and traceback (see ``BaseAsyncWorker.run()``).
    """

    def __init__(self, path: str, beamline: str, worker_id: str | None = None) -> None:
        super().__init__(worker_id)
        self.path = path
        self.beamline = beamline

    def do_work(self) -> RawFileLoadResult:
        """Load and adapt raw detector file data.

        Returns
        -------
        RawFileLoadResult
            Dictionary with keys:
            - "scattering": np.ndarray of shape (detector_shape_y, detector_shape_x)
            - "metadata": dict with kernel-adapted schema
            - "metadata_is_placeholder": bool

        Raises
        ------
        RawDataReadError
            On any failure (reader construction, data prep, or adaptation)
        """
        self.emit_status(f"Reading {self.path}...")
        try:
            reader = get_reader(self.beamline, self.path)
            reader.prepare_data()
            # After prepare_data(), metadata/scat are guaranteed to be
            # non-None -- narrow types for mypy. Both asserts live inside
            # this try block so that if either ever fired (e.g. a future
            # reader subclass bug), it still crosses the worker boundary as
            # RawDataReadError like every other failure here, not a bare
            # AssertionError.
            assert reader.metadata is not None
            assert isinstance(reader.scat, np.ndarray)
            # Adaptation errors (missing field, unsupported Reflection
            # geometry) belong inside this same try block -- every failure
            # from this worker must cross the boundary as RawDataReadError,
            # not just the reader-construction/read failures.
            metadata = to_kernel_metadata(reader.metadata, reader.stype)
        except Exception as exc:
            raise RawDataReadError(self.path, exc) from exc

        self.emit_status("Read complete")
        return {
            "scattering": reader.scat,
            "metadata": metadata,
            "metadata_is_placeholder": reader.metadata_is_placeholder,
        }
