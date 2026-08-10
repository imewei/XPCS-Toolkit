"""Unit tests for reader/exceptions.py."""

from xpcsviewer.simplemask.reader.exceptions import RawDataReadError


def test_raw_data_read_error_carries_path_and_cause():
    cause = ValueError("bad frame count")
    err = RawDataReadError("/data/scan.h5", cause)
    assert err.path == "/data/scan.h5"
    assert err.cause is cause
    assert "/data/scan.h5" in str(err)
    assert "bad frame count" in str(err)
