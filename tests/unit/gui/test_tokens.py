"""Unit tests for theme design tokens."""

import pytest

from xpcsviewer.gui.theme.tokens import (
    DARK_COLORS,
    DARK_TOKENS,
    LIGHT_COLORS,
    LIGHT_TOKENS,
    SPACING_TOKENS,
    TYPOGRAPHY_TOKENS,
    ColorTokens,
    SpacingTokens,
    ThemeDefinition,
    TypographyTokens,
)


class TestColorTokens:
    """Tests for ColorTokens dataclass."""

    def test_light_colors_are_valid_hex(self):
        """All light theme colors should be valid hex strings."""
        for field_name in ColorTokens.__dataclass_fields__:
            value = getattr(LIGHT_COLORS, field_name)
            assert isinstance(value, str)
            assert value.startswith("#"), f"{field_name} should start with #"
            assert len(value) == 7, f"{field_name} should be 7 chars (#RRGGBB)"

    def test_dark_colors_are_valid_hex(self):
        """All dark theme colors should be valid hex strings."""
        for field_name in ColorTokens.__dataclass_fields__:
            value = getattr(DARK_COLORS, field_name)
            assert isinstance(value, str)
            assert value.startswith("#"), f"{field_name} should start with #"
            assert len(value) == 7, f"{field_name} should be 7 chars (#RRGGBB)"

    def test_color_tokens_are_immutable(self):
        """ColorTokens should be immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            LIGHT_COLORS.background_primary = "#000000"

    def test_light_and_dark_are_different(self):
        """Light and dark themes should have different color values."""
        assert LIGHT_COLORS.background_primary != DARK_COLORS.background_primary
        assert LIGHT_COLORS.text_primary != DARK_COLORS.text_primary

    def test_wcag_contrast_light_text(self):
        """Light theme text on background should have sufficient contrast."""
        # Basic check: primary text should be dark on light background
        assert (
            LIGHT_COLORS.text_primary.lower() < LIGHT_COLORS.background_primary.lower()
        )

    def test_wcag_contrast_dark_text(self):
        """Dark theme text on background should have sufficient contrast."""
        # Basic check: primary text should be light on dark background
        assert DARK_COLORS.text_primary.lower() > DARK_COLORS.background_primary.lower()


class TestSpacingTokens:
    """Tests for SpacingTokens dataclass."""

    def test_spacing_follows_8px_grid(self):
        """Spacing values should follow 8px base grid."""
        assert SPACING_TOKENS.xs == 4  # 0.5x
        assert SPACING_TOKENS.sm == 8  # 1x
        assert SPACING_TOKENS.md == 16  # 2x
        assert SPACING_TOKENS.lg == 24  # 3x
        assert SPACING_TOKENS.xl == 32  # 4x
        assert SPACING_TOKENS.xxl == 48  # 6x

    def test_spacing_tokens_are_immutable(self):
        """SpacingTokens should be immutable."""
        with pytest.raises(AttributeError):
            SPACING_TOKENS.sm = 10

    def test_spacing_values_are_integers(self):
        """All spacing values should be integers."""
        for field_name in SpacingTokens.__dataclass_fields__:
            value = getattr(SPACING_TOKENS, field_name)
            assert isinstance(value, int), f"{field_name} should be int"

    def test_spacing_values_are_positive(self):
        """All spacing values should be positive."""
        for field_name in SpacingTokens.__dataclass_fields__:
            value = getattr(SPACING_TOKENS, field_name)
            assert value > 0, f"{field_name} should be positive"


class TestTypographyTokens:
    """Tests for TypographyTokens dataclass."""

    def test_default_font_family(self):
        """Default font family should use system fonts."""
        # Font stack should include system fonts for cross-platform compatibility
        font_family = TYPOGRAPHY_TOKENS.font_family
        # Check for at least one of the common system font names
        has_system_font = any(
            f in font_family
            for f in ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "system-ui"]
        )
        assert has_system_font, f"Expected system font in: {font_family}"
        assert "sans-serif" in font_family

    def test_font_sizes_are_reasonable(self):
        """Font sizes should be in reasonable range (8-36pt)."""
        assert 8 <= TYPOGRAPHY_TOKENS.size_xs <= 12
        assert 10 <= TYPOGRAPHY_TOKENS.size_sm <= 14
        assert 12 <= TYPOGRAPHY_TOKENS.size_base <= 16
        assert 14 <= TYPOGRAPHY_TOKENS.size_lg <= 18
        assert 16 <= TYPOGRAPHY_TOKENS.size_xl <= 24
        assert 20 <= TYPOGRAPHY_TOKENS.size_xxl <= 36

    def test_font_sizes_are_ascending(self):
        """Font sizes should increase from xs to xxl."""
        sizes = [
            TYPOGRAPHY_TOKENS.size_xs,
            TYPOGRAPHY_TOKENS.size_sm,
            TYPOGRAPHY_TOKENS.size_base,
            TYPOGRAPHY_TOKENS.size_lg,
            TYPOGRAPHY_TOKENS.size_xl,
            TYPOGRAPHY_TOKENS.size_xxl,
        ]
        assert sizes == sorted(sizes), "Font sizes should be in ascending order"

    def test_font_weights_are_valid(self):
        """Font weights should be valid CSS font-weight values."""
        assert 100 <= TYPOGRAPHY_TOKENS.weight_normal <= 400
        assert 400 <= TYPOGRAPHY_TOKENS.weight_medium <= 600
        assert 600 <= TYPOGRAPHY_TOKENS.weight_bold <= 900


class TestThemeDefinition:
    """Tests for ThemeDefinition dataclass."""

    def test_light_theme_name(self):
        """Light theme should have correct name."""
        assert LIGHT_TOKENS.name == "light"
        assert LIGHT_TOKENS.display_name == "Light Mode"

    def test_dark_theme_name(self):
        """Dark theme should have correct name."""
        assert DARK_TOKENS.name == "dark"
        assert DARK_TOKENS.display_name == "Dark Mode"

    def test_themes_share_spacing(self):
        """Light and dark themes should share spacing tokens."""
        assert LIGHT_TOKENS.spacing == DARK_TOKENS.spacing

    def test_themes_share_typography(self):
        """Light and dark themes should share typography tokens."""
        assert LIGHT_TOKENS.typography == DARK_TOKENS.typography

    def test_theme_has_all_components(self):
        """Theme definition should have all required components."""
        for theme in [LIGHT_TOKENS, DARK_TOKENS]:
            assert isinstance(theme.colors, ColorTokens)
            assert isinstance(theme.spacing, SpacingTokens)
            assert isinstance(theme.typography, TypographyTokens)

    def test_theme_definitions_are_immutable(self):
        """ThemeDefinition should be immutable."""
        with pytest.raises(AttributeError):
            LIGHT_TOKENS.name = "modified"
