"""Unit tests for reader/adapter.py metadata schema translation.

This is the one place a silent bug would hide (spec's revision note): a
wrong rename or a missed/extra unit conversion here produces a wrong q-map
without raising anything. Every conversion is asserted with an exact
expected value, not just "did it run."
"""

import pytest

from xpcsviewer.simplemask.reader.adapter import to_kernel_metadata

BASE_READER_METADATA = {
    "energy": 11.0,
    "beam_center_x": 512.0,
    "beam_center_y": 256.0,
    "detector_shape_x": 1024,
    "detector_shape_y": 512,
    "pixel_size": 75e-6,  # meters
    "detector_distance": 5.0,  # meters
}


class TestToKernelMetadata:
    def test_beam_center_is_renamed_without_conversion(self):
        result = to_kernel_metadata(BASE_READER_METADATA, stype="Transmission")
        assert result["bcx"] == pytest.approx(512.0)
        assert result["bcy"] == pytest.approx(256.0)

    def test_pixel_size_converts_meters_to_millimeters(self):
        result = to_kernel_metadata(BASE_READER_METADATA, stype="Transmission")
        # 75e-6 m * 1000 = 0.075 mm
        assert result["pix_dim"] == pytest.approx(0.075)

    def test_detector_distance_converts_meters_to_millimeters(self):
        result = to_kernel_metadata(BASE_READER_METADATA, stype="Transmission")
        # 5.0 m * 1000 = 5000.0 mm
        assert result["det_dist"] == pytest.approx(5000.0)

    def test_energy_passes_through_unchanged(self):
        result = to_kernel_metadata(BASE_READER_METADATA, stype="Transmission")
        assert result["energy"] == pytest.approx(11.0)

    def test_shape_is_height_width_tuple(self):
        result = to_kernel_metadata(BASE_READER_METADATA, stype="Transmission")
        assert result["shape"] == (512, 1024)  # (detector_shape_y, detector_shape_x)

    def test_stype_passed_through(self):
        result = to_kernel_metadata(BASE_READER_METADATA, stype="Transmission")
        assert result["stype"] == "Transmission"

    def test_missing_required_field_raises_keyerror(self):
        incomplete = dict(BASE_READER_METADATA)
        del incomplete["energy"]
        with pytest.raises(KeyError):
            to_kernel_metadata(incomplete, stype="Transmission")

    def test_reflection_not_yet_supported_raises(self):
        # 9-ID-D orientation mapping lands in PR2; fail loud, not silently
        # wrong, if Reflection is requested before then.
        with pytest.raises(NotImplementedError):
            to_kernel_metadata(BASE_READER_METADATA, stype="Reflection")
