"""Unit tests for raw_file_worker.py -- run do_work() directly, no QThreadPool."""

from unittest.mock import patch

import numpy as np
import pytest

from xpcsviewer.simplemask.raw_file_worker import RawFileLoadWorker
from xpcsviewer.simplemask.reader.exceptions import RawDataReadError


class _FakeReader:
    """Stand-in for a FileReader subclass, avoiding real file IO in this test."""

    def __init__(self, fname):
        self.fname = fname
        self.stype = "Transmission"
        self.metadata_is_placeholder = False
        self.scat = None
        self.metadata = None

    def prepare_data(self):
        self.scat = np.full((4, 4), 3.0, dtype=np.float32)
        self.metadata = {
            "energy": 10.0,
            "beam_center_x": 2.0,
            "beam_center_y": 2.0,
            "detector_shape_x": 4,
            "detector_shape_y": 4,
            "pixel_size": 75e-6,
            "detector_distance": 5.0,
        }


class _FakeFailingReader(_FakeReader):
    def prepare_data(self):
        raise RuntimeError("simulated read failure")


def test_do_work_returns_scattering_and_adapted_metadata():
    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_FakeReader("scan.h5"),
    ):
        worker = RawFileLoadWorker("scan.h5", "APS_8IDI")
        result = worker.do_work()

    assert result["scattering"].shape == (4, 4)
    assert result["metadata"]["bcx"] == pytest.approx(2.0)
    assert result["metadata"]["pix_dim"] == pytest.approx(0.075)
    assert result["metadata_is_placeholder"] is False


def test_do_work_wraps_failure_as_raw_data_read_error():
    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_FakeFailingReader("bad.h5"),
    ):
        worker = RawFileLoadWorker("bad.h5", "APS_8IDI")
        with pytest.raises(RawDataReadError) as exc_info:
            worker.do_work()

    assert exc_info.value.path == "bad.h5"


def test_do_work_wraps_adapter_failure_as_raw_data_read_error():
    # to_kernel_metadata() raising (e.g. a missing required field) must also
    # cross the worker boundary as RawDataReadError, not an unwrapped
    # KeyError -- this was a real bug caught in review: adaptation used to
    # run outside the try block.
    class _IncompleteMetadataReader(_FakeReader):
        def prepare_data(self):
            self.scat = np.zeros((2, 2), dtype=np.float32)
            self.metadata = {"energy": 10.0}  # missing every other required field

    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_IncompleteMetadataReader("incomplete.h5"),
    ):
        worker = RawFileLoadWorker("incomplete.h5", "APS_8IDI")
        with pytest.raises(RawDataReadError) as exc_info:
            worker.do_work()

    assert exc_info.value.path == "incomplete.h5"
    assert isinstance(exc_info.value.cause, KeyError)


def test_do_work_propagates_placeholder_flag():
    reader = _FakeReader("scan.h5")

    class _PlaceholderReader(_FakeReader):
        def prepare_data(self):
            super().prepare_data()
            self.metadata_is_placeholder = True

    with patch(
        "xpcsviewer.simplemask.raw_file_worker.get_reader",
        return_value=_PlaceholderReader("scan.h5"),
    ):
        worker = RawFileLoadWorker("scan.h5", "APS_8IDI")
        result = worker.do_work()

    assert result["metadata_is_placeholder"] is True
