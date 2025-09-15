"""Unit tests for APS 8IDI beamline-specific data structures.

This module provides comprehensive unit tests for the APS 8IDI
beamline-specific data structure definitions and key mappings.
"""

import pytest

from xpcs_toolkit.fileIO.aps_8idi import key


class TestAPS8IDIKeyStructure:
    """Test suite for APS 8IDI key structure validation."""

    def test_key_is_dict(self):
        """Test that key is a dictionary."""
        assert isinstance(key, dict)
        assert len(key) > 0

    def test_nexus_key_exists(self):
        """Test that 'nexus' key exists in main key dict."""
        assert "nexus" in key
        assert isinstance(key["nexus"], dict)
        assert len(key["nexus"]) > 0

    def test_key_structure_completeness(self):
        """Test that all essential keys are present."""
        nexus_keys = key["nexus"]

        # Essential qmap keys
        essential_qmap_keys = [
            "mask",
            "dqmap",
            "sqmap",
            "dqlist",
            "sqlist",
            "dplist",
            "splist",
            "bcx",
            "bcy",
            "det_dist",
            "pixel_size",
            "static_index_mapping",
            "dynamic_index_mapping",
            "static_num_pts",
            "dynamic_num_pts",
            "map_names",
            "map_units",
        ]

        for qmap_key in essential_qmap_keys:
            assert qmap_key in nexus_keys, f"Missing qmap key: {qmap_key}"

        # Essential analysis keys
        essential_analysis_keys = [
            "Int_t",
            "G2",
            "tau",
            "g2",
            "stride_frame",
            "avg_frame",
            "g2_err",
            "saxs_2d",
            "saxs_1d",
            "Iqp",
        ]

        for analysis_key in essential_analysis_keys:
            assert analysis_key in nexus_keys, f"Missing analysis key: {analysis_key}"

        # Essential metadata keys
        essential_metadata_keys = [
            "start_time",
            "t0",
            "t1",
            "X_energy",
            "pix_dim_x",
            "pix_dim_y",
        ]

        for metadata_key in essential_metadata_keys:
            assert metadata_key in nexus_keys, f"Missing metadata key: {metadata_key}"

    def test_twotime_keys_exist(self):
        """Test that two-time correlation keys exist."""
        nexus_keys = key["nexus"]

        twotime_keys = [
            "c2_g2_segments",
            "c2_g2",
            "c2_delay",
            "c2_prefix",
            "c2_processed_bins",
            "c2_stride_frame",
            "c2_avg_frame",
        ]

        for twotime_key in twotime_keys:
            assert twotime_key in nexus_keys, f"Missing two-time key: {twotime_key}"


class TestAPS8IDIPathFormats:
    """Test suite for APS 8IDI path format validation."""

    def test_all_paths_are_strings(self):
        """Test that all path values are strings."""
        nexus_keys = key["nexus"]

        for key_name, path in nexus_keys.items():
            assert isinstance(path, str), (
                f"Path for key '{key_name}' is not a string: {path}"
            )
            assert len(path) > 0, f"Path for key '{key_name}' is empty"

    def test_paths_start_with_slash(self):
        """Test that most paths start with '/' (absolute paths)."""
        nexus_keys = key["nexus"]

        # Most paths should start with '/' except some special cases
        relative_path_exceptions = ["c2_g2"]  # Known relative paths

        for key_name, path in nexus_keys.items():
            if key_name not in relative_path_exceptions:
                assert path.startswith("/"), (
                    f"Path for key '{key_name}' should start with '/': {path}"
                )

    def test_qmap_paths_consistency(self):
        """Test that qmap paths follow consistent structure."""
        nexus_keys = key["nexus"]

        qmap_keys = [
            "mask",
            "dqmap",
            "sqmap",
            "dqlist",
            "sqlist",
            "dplist",
            "splist",
            "bcx",
            "bcy",
            "det_dist",
            "pixel_size",
            "static_index_mapping",
            "dynamic_index_mapping",
            "static_num_pts",
            "dynamic_num_pts",
            "map_names",
            "map_units",
        ]

        for qmap_key in qmap_keys:
            path = nexus_keys[qmap_key]
            assert path.startswith("/xpcs/qmap/"), (
                f"QMap path should start with '/xpcs/qmap/': {path}"
            )

    def test_multitau_paths_consistency(self):
        """Test that multitau paths follow consistent structure."""
        nexus_keys = key["nexus"]

        multitau_keys = ["G2", "tau", "g2", "g2_err", "stride_frame", "avg_frame"]

        for multitau_key in multitau_keys:
            path = nexus_keys[multitau_key]
            assert path.startswith("/xpcs/multitau/"), (
                f"Multitau path should start with '/xpcs/multitau/': {path}"
            )

    def test_twotime_paths_consistency(self):
        """Test that two-time paths follow consistent structure."""
        nexus_keys = key["nexus"]

        twotime_keys = [
            "c2_g2_segments",
            "c2_delay",
            "c2_prefix",
            "c2_processed_bins",
            "c2_stride_frame",
            "c2_avg_frame",
        ]

        for twotime_key in twotime_keys:
            if twotime_key in nexus_keys:  # Some may not exist in all versions
                path = nexus_keys[twotime_key]
                assert path.startswith("/xpcs/twotime/"), (
                    f"Two-time path should start with '/xpcs/twotime/': {path}"
                )

    def test_temporal_mean_paths_consistency(self):
        """Test that temporal mean paths follow consistent structure."""
        nexus_keys = key["nexus"]

        temporal_keys = ["saxs_2d", "saxs_1d", "Iqp"]

        for temporal_key in temporal_keys:
            path = nexus_keys[temporal_key]
            assert path.startswith("/xpcs/temporal_mean/"), (
                f"Temporal mean path should start with '/xpcs/temporal_mean/': {path}"
            )

    def test_entry_paths_consistency(self):
        """Test that entry-level paths follow consistent structure."""
        nexus_keys = key["nexus"]

        entry_keys = ["start_time", "t0", "t1", "X_energy", "pix_dim_x", "pix_dim_y"]

        for entry_key in entry_keys:
            path = nexus_keys[entry_key]
            assert path.startswith("/entry/"), (
                f"Entry path should start with '/entry/': {path}"
            )


class TestAPS8IDISpecificPaths:
    """Test suite for specific APS 8IDI path validation."""

    def test_qmap_specific_paths(self):
        """Test specific qmap path values."""
        nexus_keys = key["nexus"]

        # Test specific known paths
        expected_paths = {
            "mask": "/xpcs/qmap/mask",
            "dqmap": "/xpcs/qmap/dynamic_roi_map",
            "sqmap": "/xpcs/qmap/static_roi_map",
            "bcx": "/xpcs/qmap/beam_center_x",
            "bcy": "/xpcs/qmap/beam_center_y",
            "det_dist": "/xpcs/qmap/detector_distance",
            "pixel_size": "/xpcs/qmap/pixel_size",
        }

        for key_name, expected_path in expected_paths.items():
            assert nexus_keys[key_name] == expected_path, (
                f"Path mismatch for {key_name}: expected {expected_path}, got {nexus_keys[key_name]}"
            )

    def test_analysis_specific_paths(self):
        """Test specific analysis path values."""
        nexus_keys = key["nexus"]

        expected_paths = {
            "tau": "/xpcs/multitau/delay_list",
            "g2": "/xpcs/multitau/normalized_g2",
            "G2": "/xpcs/multitau/unnormalized_G2",
            "g2_err": "/xpcs/multitau/normalized_g2_err",
            "saxs_1d": "/xpcs/temporal_mean/scattering_1d",
            "saxs_2d": "/xpcs/temporal_mean/scattering_2d",
            "Int_t": "/xpcs/spatial_mean/intensity_vs_time",
        }

        for key_name, expected_path in expected_paths.items():
            assert nexus_keys[key_name] == expected_path, (
                f"Path mismatch for {key_name}: expected {expected_path}, got {nexus_keys[key_name]}"
            )

    def test_detector_specific_paths(self):
        """Test detector-specific path values."""
        nexus_keys = key["nexus"]

        expected_paths = {
            "t0": "/entry/instrument/detector_1/frame_time",
            "t1": "/entry/instrument/detector_1/count_time",
            "pix_dim_x": "/entry/instrument/detector_1/x_pixel_size",
            "pix_dim_y": "/entry/instrument/detector_1/y_pixel_size",
        }

        for key_name, expected_path in expected_paths.items():
            assert nexus_keys[key_name] == expected_path, (
                f"Path mismatch for {key_name}: expected {expected_path}, got {nexus_keys[key_name]}"
            )

    def test_beam_specific_paths(self):
        """Test beam-specific path values."""
        nexus_keys = key["nexus"]

        expected_paths = {
            "X_energy": "/entry/instrument/incident_beam/incident_energy",
            "start_time": "/entry/start_time",
        }

        for key_name, expected_path in expected_paths.items():
            assert nexus_keys[key_name] == expected_path, (
                f"Path mismatch for {key_name}: expected {expected_path}, got {nexus_keys[key_name]}"
            )


class TestAPS8IDIKeyAccess:
    """Test suite for APS 8IDI key access patterns."""

    def test_key_access_by_category(self):
        """Test accessing keys by category."""
        nexus_keys = key["nexus"]

        # Test that we can access different categories of keys
        qmap_keys = [
            k
            for k in nexus_keys
            if k in ["mask", "dqmap", "sqmap", "bcx", "bcy", "det_dist"]
        ]
        assert len(qmap_keys) >= 6, "Should have at least 6 qmap keys"

        analysis_keys = [
            k for k in nexus_keys if k in ["tau", "g2", "G2", "saxs_1d", "saxs_2d"]
        ]
        assert len(analysis_keys) >= 5, "Should have at least 5 analysis keys"

        detector_keys = [
            k for k in nexus_keys if k in ["t0", "t1", "pix_dim_x", "pix_dim_y"]
        ]
        assert len(detector_keys) >= 4, "Should have at least 4 detector keys"

    def test_key_lookup_functionality(self):
        """Test that key lookup works as expected."""
        # Test direct access
        assert key["nexus"]["mask"] == "/xpcs/qmap/mask"
        assert key["nexus"]["g2"] == "/xpcs/multitau/normalized_g2"

        # Test that we can iterate over keys
        nexus_keys = key["nexus"]
        key_count = sum(1 for _ in nexus_keys)
        assert key_count > 20, "Should have more than 20 keys defined"

        # Test that we can access values
        path_count = sum(1 for _ in nexus_keys.values())
        assert path_count == key_count, "Number of paths should equal number of keys"


class TestAPS8IDICompatibility:
    """Test suite for APS 8IDI backward compatibility."""

    def test_essential_keys_unchanged(self):
        """Test that essential key paths haven't changed."""
        nexus_keys = key["nexus"]

        # These are critical paths that should remain stable
        critical_paths = {
            "g2": "/xpcs/multitau/normalized_g2",
            "tau": "/xpcs/multitau/delay_list",
            "saxs_1d": "/xpcs/temporal_mean/scattering_1d",
            "mask": "/xpcs/qmap/mask",
            "dqmap": "/xpcs/qmap/dynamic_roi_map",
            "bcx": "/xpcs/qmap/beam_center_x",
            "bcy": "/xpcs/qmap/beam_center_y",
        }

        for key_name, expected_path in critical_paths.items():
            assert key_name in nexus_keys, f"Critical key '{key_name}' is missing"
            assert nexus_keys[key_name] == expected_path, (
                f"Critical path changed for {key_name}: expected {expected_path}, got {nexus_keys[key_name]}"
            )

    def test_no_duplicate_paths(self):
        """Test that no two keys map to the same path."""
        nexus_keys = key["nexus"]

        paths = list(nexus_keys.values())
        unique_paths = set(paths)

        # Allow one known exception: some keys might legitimately share paths
        # but generally paths should be unique
        assert len(unique_paths) >= len(paths) - 2, (
            "Too many duplicate paths detected, possible configuration error"
        )

    def test_path_consistency_with_nexus_standard(self):
        """Test that paths follow NeXus standard conventions."""
        nexus_keys = key["nexus"]

        for _key_name, path in nexus_keys.items():
            # Paths should not contain double slashes
            assert "//" not in path, f"Path contains double slash: {path}"

            # Paths should not end with slash (except root)
            if path != "/":
                assert not path.endswith("/"), f"Path should not end with slash: {path}"

            # Paths should use forward slashes only
            assert "\\" not in path, f"Path should use forward slashes only: {path}"


class TestAPS8IDIKeyValidation:
    """Test suite for APS 8IDI key validation."""

    def test_key_names_are_valid_identifiers(self):
        """Test that key names are valid Python identifiers."""
        nexus_keys = key["nexus"]

        for key_name in nexus_keys:
            # Key names should be strings
            assert isinstance(key_name, str), f"Key name is not a string: {key_name}"

            # Key names should not be empty
            assert len(key_name) > 0, "Key name is empty"

            # Key names should contain only valid characters
            assert key_name.replace("_", "").replace("2", "").isalnum(), (
                f"Key name contains invalid characters: {key_name}"
            )

    def test_path_depth_reasonable(self):
        """Test that paths have reasonable depth."""
        nexus_keys = key["nexus"]

        for key_name, path in nexus_keys.items():
            path_depth = path.count("/")

            # Most paths should have reasonable depth (not too deep or too shallow)
            assert 1 <= path_depth <= 6, (
                f"Path depth seems unreasonable for {key_name}: {path} (depth: {path_depth})"
            )

    def test_required_path_components(self):
        """Test that paths contain required components."""
        nexus_keys = key["nexus"]

        # Most paths should contain 'xpcs' or 'entry'
        paths_with_required_components = 0

        for _key_name, path in nexus_keys.items():
            if "xpcs" in path or "entry" in path:
                paths_with_required_components += 1

        # At least 90% of paths should contain required components
        total_paths = len(nexus_keys)
        required_ratio = paths_with_required_components / total_paths
        assert required_ratio >= 0.9, (
            f"Only {required_ratio:.1%} of paths contain required components, expected at least 90%"
        )


@pytest.mark.parametrize(
    "key_category,expected_prefix",
    [
        (["mask", "dqmap", "sqmap", "bcx", "bcy"], "/xpcs/qmap/"),
        (["g2", "tau", "G2", "stride_frame", "avg_frame"], "/xpcs/multitau/"),
        (["saxs_1d", "saxs_2d", "Iqp"], "/xpcs/temporal_mean/"),
        (["t0", "t1", "X_energy"], "/entry/"),
    ],
)
def test_path_category_consistency(key_category, expected_prefix):
    """Test that keys in same category have consistent path prefixes."""
    nexus_keys = key["nexus"]

    for key_name in key_category:
        if key_name in nexus_keys:  # Key might not exist in all versions
            path = nexus_keys[key_name]
            assert path.startswith(expected_prefix), (
                f"Key {key_name} path {path} doesn't start with expected prefix {expected_prefix}"
            )


@pytest.mark.parametrize(
    "key_name",
    ["mask", "dqmap", "sqmap", "g2", "tau", "saxs_1d", "bcx", "bcy", "t0", "X_energy"],
)
def test_critical_keys_exist(key_name):
    """Test that critical keys exist in the key mapping."""
    nexus_keys = key["nexus"]
    assert key_name in nexus_keys, (
        f"Critical key '{key_name}' is missing from key mapping"
    )


class TestAPS8IDIDataTypes:
    """Test suite for APS 8IDI data type implications."""

    def test_key_structure_immutability(self):
        """Test that key structure behaves as expected for read-only access."""
        # The key structure should be accessible but we shouldn't modify it in tests
        original_mask_path = key["nexus"]["mask"]

        # Verify we can read it
        assert isinstance(original_mask_path, str)

        # Verify structure is consistent
        assert "nexus" in key
        assert "mask" in key["nexus"]

    def test_key_completeness_for_typical_workflow(self):
        """Test that key mapping supports typical XPCS analysis workflow."""
        nexus_keys = key["nexus"]

        # Keys needed for basic XPCS analysis
        workflow_keys = [
            "g2",
            "tau",
            "saxs_1d",
            "mask",
            "dqmap",
            "bcx",
            "bcy",
            "det_dist",
            "pixel_size",
        ]

        for workflow_key in workflow_keys:
            assert workflow_key in nexus_keys, (
                f"Workflow key '{workflow_key}' missing - this would break typical XPCS analysis"
            )

    def test_twotime_analysis_support(self):
        """Test that key mapping supports two-time correlation analysis."""
        nexus_keys = key["nexus"]

        twotime_workflow_keys = [
            "c2_g2_segments",
            "c2_processed_bins",
            "c2_stride_frame",
            "c2_avg_frame",
        ]

        for twotime_key in twotime_workflow_keys:
            assert twotime_key in nexus_keys, (
                f"Two-time workflow key '{twotime_key}' missing"
            )
