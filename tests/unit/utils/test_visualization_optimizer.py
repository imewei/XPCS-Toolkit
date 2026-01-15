"""Tests for visualization optimizer data integrity features.

Tests for Technical Guidelines compliance:
- T017: VisualizationConfig defaults
- T018: Downsampling disabled by default
"""

import numpy as np
import pytest


class TestVisualizationConfig:
    """Test VisualizationConfig dataclass defaults (T017)."""

    def test_default_allow_downsampling_is_false(self):
        """Verify allow_downsampling defaults to False per Technical Guidelines."""
        from xpcsviewer.utils.visualization_optimizer import VisualizationConfig

        config = VisualizationConfig()
        assert config.allow_downsampling is False

    def test_default_downsample_threshold(self):
        """Verify default threshold is 100_000."""
        from xpcsviewer.utils.visualization_optimizer import VisualizationConfig

        config = VisualizationConfig()
        assert config.downsample_threshold == 100_000

    def test_default_log_downsampling_is_true(self):
        """Verify logging is enabled by default."""
        from xpcsviewer.utils.visualization_optimizer import VisualizationConfig

        config = VisualizationConfig()
        assert config.log_downsampling is True

    def test_get_visualization_config_returns_default(self):
        """Verify get_visualization_config returns config with downsampling OFF."""
        from xpcsviewer.utils.visualization_optimizer import get_visualization_config

        config = get_visualization_config()
        assert config.allow_downsampling is False

    def test_set_visualization_config(self):
        """Verify set_visualization_config updates module config."""
        from xpcsviewer.utils.visualization_optimizer import (
            VisualizationConfig,
            get_visualization_config,
            set_visualization_config,
        )

        # Enable downsampling
        new_config = VisualizationConfig(allow_downsampling=True)
        set_visualization_config(new_config)

        config = get_visualization_config()
        assert config.allow_downsampling is True

        # Restore default for other tests
        set_visualization_config(VisualizationConfig())


class TestDownsamplingDisabledByDefault:
    """Test that downsampling is disabled by default (T018)."""

    def test_image_optimizer_preserves_data_by_default(self):
        """Verify ImageDisplayOptimizer returns input unchanged when downsampling OFF."""
        from xpcsviewer.utils.visualization_optimizer import (
            ImageDisplayOptimizer,
            VisualizationConfig,
            set_visualization_config,
        )

        # Ensure downsampling is OFF
        set_visualization_config(VisualizationConfig(allow_downsampling=False))

        optimizer = ImageDisplayOptimizer(max_display_size=(100, 100))

        # Create large image that would normally be downsampled
        large_image = np.random.rand(4096, 4096)

        # Should return unchanged
        result = optimizer.optimize_image_for_display(large_image)

        # Verify exact same array (not just same shape)
        assert result is large_image
        assert result.shape == large_image.shape
        assert np.array_equal(result, large_image)

    def test_image_optimizer_downsamples_when_enabled(self):
        """Verify ImageDisplayOptimizer downsamples when explicitly enabled."""
        from xpcsviewer.utils.visualization_optimizer import (
            ImageDisplayOptimizer,
            VisualizationConfig,
            set_visualization_config,
        )

        # Enable downsampling
        set_visualization_config(VisualizationConfig(allow_downsampling=True))

        optimizer = ImageDisplayOptimizer(max_display_size=(100, 100))

        # Create large image
        large_image = np.random.rand(4096, 4096)

        # Should downsample
        result = optimizer.optimize_image_for_display(large_image)

        # Verify downsampled
        assert result.shape != large_image.shape
        assert result.size < large_image.size

        # Restore default
        set_visualization_config(VisualizationConfig())

    def test_large_dataset_display_preserves_data_by_default(self):
        """Verify AdvancedGUIRenderer preserves data when downsampling OFF."""
        from xpcsviewer.utils.visualization_optimizer import (
            AdvancedGUIRenderer,
            VisualizationConfig,
            set_visualization_config,
        )

        # Ensure downsampling is OFF
        set_visualization_config(VisualizationConfig(allow_downsampling=False))

        renderer = AdvancedGUIRenderer()

        # Create large 1D dataset
        large_data = np.random.rand(500_000)

        # Should return unchanged
        result = renderer.optimize_large_dataset_display(large_data, max_points=10000)

        # Verify exact same array
        assert result is large_data
        assert len(result) == 500_000

    def test_500k_point_dataset_preserved(self):
        """SC-001: Load 500k+ point dataset, verify all points preserved."""
        from xpcsviewer.utils.visualization_optimizer import (
            ImageDisplayOptimizer,
            VisualizationConfig,
            set_visualization_config,
        )

        # Ensure downsampling is OFF (default behavior)
        set_visualization_config(VisualizationConfig(allow_downsampling=False))

        optimizer = ImageDisplayOptimizer()

        # Create 500k+ point dataset (717x717 = 514,089 points)
        dataset = np.random.rand(717, 717)
        original_size = dataset.size

        # Process through optimizer
        result = optimizer.optimize_image_for_display(dataset)

        # Verify ALL points preserved
        assert result.size == original_size
        assert np.array_equal(result, dataset)
