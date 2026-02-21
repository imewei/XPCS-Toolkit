"""
Category-aware tab bar overlay with visual group separators.

Installs an event filter on the existing QTabBar to paint subtle vertical
separator lines at category group boundaries.  Does NOT replace the tab bar
(QTabWidget.setTabBar destroys existing tabs).
"""

from xpcsviewer.gui.qt_compat import QColor, Qt, QtCore, QtGui, QtWidgets

# Map tab index -> category name
# Indices must match the order set by _apply_tab_labels():
#   0: SAXS 2D, 1: SAXS 1D, 2: Stability, 3: I(t),
#   4: g2, 5: g2 Fit, 6: g2 Map, 7: Diffusion,
#   8: Two-Time, 9: Q-Map, 10: Average, 11: Metadata
TAB_INDEX_CATEGORY = {
    0: "Scattering",
    1: "Scattering",
    2: "Diagnostics",
    3: "Diagnostics",
    4: "Correlation",
    5: "Correlation",
    6: "Correlation",
    7: "Analysis",
    8: "Analysis",  # Two-Time (grouped with Diffusion)
    9: "Setup",
    10: "Setup",
    11: "Setup",
}

# Derive category boundaries from TAB_INDEX_CATEGORY
_CATEGORY_START_INDICES: set[int] = set()
_prev_cat = None
for _idx in sorted(TAB_INDEX_CATEGORY):
    _cat = TAB_INDEX_CATEGORY[_idx]
    if _cat != _prev_cat and _prev_cat is not None:
        _CATEGORY_START_INDICES.add(_idx)
    _prev_cat = _cat

# Separator visual parameters
_SEP_WIDTH = 2
_SEP_MARGIN_TOP = 6
_SEP_MARGIN_BOTTOM = 6
_SEP_COLOR_LIGHT = QColor(180, 180, 175, 200)
_SEP_COLOR_DARK = QColor(70, 74, 82, 220)


class CategorySeparatorFilter(QtCore.QObject):
    """Event filter that paints category separators on an existing QTabBar.

    Install on a QTabBar via ``tab_bar.installEventFilter(filter_obj)``.
    This avoids the destructive ``QTabWidget.setTabBar()`` call.
    """

    def __init__(self, parent: QtWidgets.QTabBar | None = None):
        super().__init__(parent)
        self._dark_mode = False

    def set_dark_mode(self, dark: bool) -> None:
        """Update separator color for the current theme."""
        self._dark_mode = dark
        if self.parent():
            self.parent().update()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Paint category separators after the tab bar paints itself."""
        if event.type() == QtCore.QEvent.Type.Paint:
            tab_bar = obj
            if not isinstance(tab_bar, QtWidgets.QTabBar):
                return False

            # Let the tab bar paint itself via the C++ virtual (avoids
            # re-entering this event filter, which would cause recursion).
            type(tab_bar).paintEvent(tab_bar, event)

            # Overlay separators
            painter = QtGui.QPainter(tab_bar)
            try:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

                sep_color = _SEP_COLOR_DARK if self._dark_mode else _SEP_COLOR_LIGHT
                pen = QtGui.QPen(sep_color, _SEP_WIDTH)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)

                for idx in _CATEGORY_START_INDICES:
                    if idx >= tab_bar.count():
                        continue
                    rect = tab_bar.tabRect(idx)
                    if rect.isNull():
                        continue
                    x = rect.left()
                    top = rect.top() + _SEP_MARGIN_TOP
                    bottom = rect.bottom() - _SEP_MARGIN_BOTTOM
                    painter.drawLine(x, top, x, bottom)
            finally:
                painter.end()
            return True  # event handled (we already called paintEvent)

        return False
