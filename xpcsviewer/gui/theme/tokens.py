"""
Design tokens for XPCS-TOOLKIT GUI theming.

This module defines the color palette, spacing, and typography tokens
that form the foundation of the theme system.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorTokens:
    """Immutable color token set for a theme."""

    # Surface colors
    background_primary: str  # Main window background
    background_secondary: str  # Panel/sidebar background
    background_elevated: str  # Popups, tooltips, floating elements
    background_input: str  # Text input fields

    # Text colors
    text_primary: str  # Main text
    text_secondary: str  # Subdued text, labels
    text_disabled: str  # Disabled elements
    text_placeholder: str  # Input placeholders

    # Accent colors
    accent_primary: str  # Primary action buttons, links
    accent_hover: str  # Hover state
    accent_pressed: str  # Active/pressed state
    accent_focus: str  # Focus ring color

    # Semantic colors
    success: str  # Success states, confirmations
    warning: str  # Warnings, cautions
    error: str  # Errors, destructive actions
    info: str  # Informational elements

    # Border colors
    border_subtle: str  # Light borders, dividers
    border_strong: str  # Emphasized borders
    border_focus: str  # Focus indicators

    # Plot-specific (synced with matplotlib/pyqtgraph)
    plot_background: str  # Plot area background
    plot_grid: str  # Grid lines
    plot_axis: str  # Axis lines and ticks


@dataclass(frozen=True)
class SpacingTokens:
    """Spacing tokens based on 8px grid system."""

    xs: int = 4  # Extra small (0.5x)
    sm: int = 8  # Small (1x base)
    md: int = 16  # Medium (2x)
    lg: int = 24  # Large (3x)
    xl: int = 32  # Extra large (4x)
    xxl: int = 48  # Extra extra large (6x)


@dataclass(frozen=True)
class TypographyTokens:
    """Typography tokens for consistent text styling."""

    font_family: str = (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    )

    # Font sizes (in points)
    size_xs: int = 10
    size_sm: int = 11
    size_base: int = 13
    size_lg: int = 15
    size_xl: int = 18
    size_xxl: int = 24

    # Font weights
    weight_normal: int = 400
    weight_medium: int = 500
    weight_bold: int = 700


@dataclass(frozen=True)
class ThemeDefinition:
    """Complete theme definition."""

    name: str  # "light" or "dark"
    display_name: str  # "Light Mode" or "Dark Mode"
    colors: ColorTokens
    spacing: SpacingTokens
    typography: TypographyTokens


# Light theme colors - optimized for readability and WCAG AA compliance
LIGHT_COLORS = ColorTokens(
    # Surfaces
    background_primary="#FFFFFF",
    background_secondary="#F5F5F7",
    background_elevated="#FFFFFF",
    background_input="#FFFFFF",
    # Text
    text_primary="#1D1D1F",
    text_secondary="#6E6E73",
    text_disabled="#AEAEB2",
    text_placeholder="#8E8E93",
    # Accent (blue)
    accent_primary="#007AFF",
    accent_hover="#0056B3",
    accent_pressed="#004494",
    accent_focus="#007AFF",
    # Semantic
    success="#34C759",
    warning="#FF9500",
    error="#FF3B30",
    info="#5AC8FA",
    # Borders
    border_subtle="#E5E5EA",
    border_strong="#C7C7CC",
    border_focus="#007AFF",
    # Plot
    plot_background="#FFFFFF",
    plot_grid="#E5E5EA",
    plot_axis="#1D1D1F",
)

# Dark theme colors - optimized for reduced eye strain
DARK_COLORS = ColorTokens(
    # Surfaces
    background_primary="#1C1C1E",
    background_secondary="#2C2C2E",
    background_elevated="#3A3A3C",
    background_input="#2C2C2E",
    # Text
    text_primary="#F5F5F7",
    text_secondary="#AEAEB2",
    text_disabled="#636366",
    text_placeholder="#8E8E93",
    # Accent (blue - adjusted for dark background)
    accent_primary="#0A84FF",
    accent_hover="#409CFF",
    accent_pressed="#0066CC",
    accent_focus="#0A84FF",
    # Semantic (adjusted for dark background)
    success="#30D158",
    warning="#FF9F0A",
    error="#FF453A",
    info="#64D2FF",
    # Borders
    border_subtle="#38383A",
    border_strong="#545456",
    border_focus="#0A84FF",
    # Plot
    plot_background="#1C1C1E",
    plot_grid="#38383A",
    plot_axis="#F5F5F7",
)

# Shared spacing and typography tokens
SPACING_TOKENS = SpacingTokens()
TYPOGRAPHY_TOKENS = TypographyTokens()

# Complete theme definitions
LIGHT_TOKENS = ThemeDefinition(
    name="light",
    display_name="Light Mode",
    colors=LIGHT_COLORS,
    spacing=SPACING_TOKENS,
    typography=TYPOGRAPHY_TOKENS,
)

DARK_TOKENS = ThemeDefinition(
    name="dark",
    display_name="Dark Mode",
    colors=DARK_COLORS,
    spacing=SPACING_TOKENS,
    typography=TYPOGRAPHY_TOKENS,
)
