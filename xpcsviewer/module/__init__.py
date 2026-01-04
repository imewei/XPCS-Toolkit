# Import modules gracefully for documentation builds
import os

if os.environ.get("BUILDING_DOCS"):
    # Provide placeholder modules for documentation
    class PlaceholderModule:
        """Placeholder module for documentation builds."""

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    g2mod = PlaceholderModule()
    intt = PlaceholderModule()
    saxs1d = PlaceholderModule()
    saxs2d = PlaceholderModule()
    stability = PlaceholderModule()
    average_toolbox = PlaceholderModule()
    tauq = PlaceholderModule()
    twotime = PlaceholderModule()
    twotime_utils = PlaceholderModule()
else:
    try:
        from . import (
            average_toolbox,
            g2mod,
            intt,
            saxs1d,
            saxs2d,
            stability,
            tauq,
            twotime,
            twotime_utils,
        )
    except ImportError:
        # Fallback for missing dependencies
        pass

__all__ = [
    "average_toolbox",
    "g2mod",
    "intt",
    "saxs1d",
    "saxs2d",
    "stability",
    "tauq",
    "twotime",
    "twotime_utils",
]
