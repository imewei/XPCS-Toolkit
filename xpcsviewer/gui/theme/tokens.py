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

    font_family: str = "'Helvetica Neue', 'Segoe UI', 'Ubuntu', 'Roboto', sans-serif"
    font_family_mono: str = (
        "'SF Mono', 'Menlo', 'Consolas', 'DejaVu Sans Mono', 'Ubuntu Mono', monospace"
    )

    # Font sizes (in points)
    size_xs: int = 10
    size_sm: int = 11
    size_base: int = 13
    size_lg: int = 15
    size_xl: int = 18
    size_xxl: int = 24



@dataclass(frozen=True)
class ThemeDefinition:
    """Complete theme definition."""

    name: str  # "light" or "dark"
    display_name: str  # "Light Mode" or "Dark Mode"
    colors: ColorTokens
    spacing: SpacingTokens
    typography: TypographyTokens


# Light theme colors - warm off-white surfaces with teal instrument accents
LIGHT_COLORS = ColorTokens(
    # Surfaces — warm off-white like fine paper
    background_primary="#FAFAF8",
    background_secondary="#F0EFEC",
    background_elevated="#FAFAF8",
    background_input="#FAFAF8",
    # Text — warm near-black
    text_primary="#1A1A1A",
    text_secondary="#64645F",
    text_disabled="#A3A39E",
    text_placeholder="#8A8A84",
    # Accent — teal (oscilloscope phosphor)
    accent_primary="#0D9488",
    accent_hover="#0F766E",
    accent_pressed="#115E59",
    accent_focus="#0D9488",
    # Semantic — natural tones
    success="#16A34A",
    warning="#D97706",
    error="#DC2626",
    info="#0891B2",
    # Borders — warm
    border_subtle="#E2E1DD",
    border_strong="#C4C3BE",
    border_focus="#0D9488",
    # Plot
    plot_background="#FAFAF8",
    plot_grid="#E2E1DD",
    plot_axis="#1A1A1A",
)

# Dark theme colors - deep blue-gray panels with bright teal phosphor accents
DARK_COLORS = ColorTokens(
    # Surfaces — deep blue-black instrument panel
    background_primary="#191B1F",
    background_secondary="#23262B",
    background_elevated="#2D3139",
    background_input="#23262B",
    # Text — warm off-white
    text_primary="#E8E8E4",
    text_secondary="#A3A3A0",
    text_disabled="#5A5D63",
    text_placeholder="#7A7D83",
    # Accent — teal-400 bright phosphor
    accent_primary="#2DD4BF",
    accent_hover="#5EEAD4",
    accent_pressed="#14B8A6",
    accent_focus="#2DD4BF",
    # Semantic (adjusted for dark background)
    success="#4ADE80",
    warning="#FBBF24",
    error="#F87171",
    info="#22D3EE",
    # Borders — blue-gray
    border_subtle="#2D3139",
    border_strong="#3D4149",
    border_focus="#2DD4BF",
    # Plot
    plot_background="#191B1F",
    plot_grid="#2D3139",
    plot_axis="#E8E8E4",
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
