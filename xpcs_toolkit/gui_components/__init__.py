"""
GUI Components Package for XPCS Toolkit

This package contains reusable GUI components for the XPCS Toolkit,
including robust fitting controls, diagnostic widgets, and enhanced
plotting components.
"""

from .robust_fitting_controls import RobustFittingControlPanel
from .diagnostic_widgets import DiagnosticDashboard, RealTimeDiagnosticWidget
from .interactive_parameter_widgets import ParameterAnalysisWidget, ConfidenceIntervalWidget
from .enhanced_plotting import EnhancedG2PlotWidget, UncertaintyVisualizationMixin

__all__ = [
    'RobustFittingControlPanel',
    'DiagnosticDashboard',
    'RealTimeDiagnosticWidget',
    'ParameterAnalysisWidget',
    'ConfidenceIntervalWidget',
    'EnhancedG2PlotWidget',
    'UncertaintyVisualizationMixin'
]