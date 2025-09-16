"""
Enhanced Plotting Components with Uncertainty Visualization

This module provides enhanced G2 plotting capabilities including uncertainty bands,
outlier highlighting, multi-model overlays, and advanced visualization features.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel
from PySide6.QtGui import QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from scipy import stats

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class UncertaintyVisualizationMixin:
    """
    Mixin class providing uncertainty visualization capabilities for plots.

    Features:
    - Confidence bands
    - Prediction intervals
    - Bootstrap confidence regions
    - Error bar enhancements
    """

    def setup_uncertainty_visualization(self):
        """Initialize uncertainty visualization components."""
        self.uncertainty_items = {}
        self.confidence_level = 0.95
        self.show_confidence_bands = True
        self.show_prediction_intervals = False
        self.show_bootstrap_regions = False

    def add_confidence_bands(self, x_data, y_lower, y_upper, color='red', alpha=0.3, name='Confidence'):
        """Add confidence bands to the plot."""
        if name in self.uncertainty_items:
            self.removeItem(self.uncertainty_items[name])

        # Create fill between curves
        fill_item = pg.FillBetweenItem(
            pg.PlotCurveItem(x_data, y_lower),
            pg.PlotCurveItem(x_data, y_upper),
            brush=pg.mkBrush(*pg.colorTuple(pg.mkColor(color)), int(255 * alpha))
        )

        self.addItem(fill_item)
        self.uncertainty_items[name] = fill_item

        # Add boundary lines
        lower_line = self.plot(
            x_data, y_lower,
            pen=pg.mkPen(color=color, style=Qt.DashLine, width=1),
            name=f'{name} Lower'
        )
        upper_line = self.plot(
            x_data, y_upper,
            pen=pg.mkPen(color=color, style=Qt.DashLine, width=1),
            name=f'{name} Upper'
        )

        self.uncertainty_items[f'{name}_lower'] = lower_line
        self.uncertainty_items[f'{name}_upper'] = upper_line

    def add_prediction_intervals(self, x_data, y_lower, y_upper, color='green', name='Prediction'):
        """Add prediction intervals to the plot."""
        if name in self.uncertainty_items:
            self.removeItem(self.uncertainty_items[name])

        # Prediction intervals shown as dotted lines only
        lower_line = self.plot(
            x_data, y_lower,
            pen=pg.mkPen(color=color, style=Qt.DotLine, width=2),
            name=f'{name} Lower'
        )
        upper_line = self.plot(
            x_data, y_upper,
            pen=pg.mkPen(color=color, style=Qt.DotLine, width=2),
            name=f'{name} Upper'
        )

        self.uncertainty_items[f'{name}_lower'] = lower_line
        self.uncertainty_items[f'{name}_upper'] = upper_line

    def add_bootstrap_regions(self, x_data, y_percentiles, colors=None, name='Bootstrap'):
        """Add bootstrap confidence regions."""
        if colors is None:
            colors = ['lightblue', 'blue', 'darkblue']

        # y_percentiles should be a dict with keys like 'p5', 'p25', 'p50', 'p75', 'p95'
        if 'p50' in y_percentiles:
            # Median line
            median_line = self.plot(
                x_data, y_percentiles['p50'],
                pen=pg.mkPen(color='blue', width=3),
                name=f'{name} Median'
            )
            self.uncertainty_items[f'{name}_median'] = median_line

        # 90% confidence region (5th to 95th percentile)
        if 'p5' in y_percentiles and 'p95' in y_percentiles:
            self.add_confidence_bands(
                x_data, y_percentiles['p5'], y_percentiles['p95'],
                color=colors[0], alpha=0.2, name=f'{name}_90'
            )

        # 50% confidence region (25th to 75th percentile)
        if 'p25' in y_percentiles and 'p75' in y_percentiles:
            self.add_confidence_bands(
                x_data, y_percentiles['p25'], y_percentiles['p75'],
                color=colors[1], alpha=0.3, name=f'{name}_50'
            )

    def highlight_outliers(self, x_data, y_data, outlier_mask, symbol='x', color='red', size=8):
        """Highlight outlier points on the plot."""
        outlier_name = 'outliers'
        if outlier_name in self.uncertainty_items:
            self.removeItem(self.uncertainty_items[outlier_name])

        if np.any(outlier_mask):
            outlier_plot = self.plot(
                x_data[outlier_mask],
                y_data[outlier_mask],
                pen=None,
                symbol=symbol,
                symbolSize=size,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(color, width=2),
                name='Outliers'
            )
            self.uncertainty_items[outlier_name] = outlier_plot

    def add_error_bars_enhanced(self, x_data, y_data, y_err, x_err=None,
                               color='black', alpha=0.7, cap_size=3):
        """Add enhanced error bars with customizable appearance."""
        error_name = 'error_bars'
        if error_name in self.uncertainty_items:
            self.removeItem(self.uncertainty_items[error_name])

        # Create error bar item
        error_item = pg.ErrorBarItem(
            x=x_data,
            y=y_data,
            top=y_err,
            bottom=y_err,
            left=x_err if x_err is not None else None,
            right=x_err if x_err is not None else None,
            pen=pg.mkPen(color=color, width=1),
            beam=cap_size / 100.0  # Convert to relative size
        )

        self.addItem(error_item)
        self.uncertainty_items[error_name] = error_item

    def clear_uncertainty_visualization(self):
        """Clear all uncertainty visualization elements."""
        for item in self.uncertainty_items.values():
            if hasattr(item, 'scene') and item.scene() is not None:
                self.removeItem(item)
        self.uncertainty_items.clear()


class EnhancedG2PlotWidget(PlotWidget, UncertaintyVisualizationMixin):
    """
    Enhanced G2 plot widget with advanced visualization capabilities.

    Features:
    - Uncertainty visualization
    - Multi-model comparison
    - Interactive outlier detection
    - Advanced error bar display
    - Residual analysis integration
    """

    # Signals
    outlier_selection_changed = Signal(list)
    model_comparison_requested = Signal(dict)
    plot_settings_changed = Signal(dict)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.logger = logger

        # Initialize components
        self.setup_enhanced_plot()
        self.setup_uncertainty_visualization()
        self.setup_model_comparison()

        # Data storage
        self.fitted_models = {}
        self.outlier_data = {}
        self.current_data = {}

    def setup_enhanced_plot(self):
        """Setup enhanced plotting features."""
        # Configure plot appearance
        self.setBackground('w')
        self.showGrid(x=True, y=True, alpha=0.3)

        # Set axis labels with enhanced formatting
        self.setLabel('left', 'g₂(τ)', units='', **{'font-size': '12pt'})
        self.setLabel('bottom', 'τ', units='s', **{'font-size': '12pt'})

        # Enable crosshair cursor
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.addItem(self.crosshair_v, ignoreBounds=True)
        self.addItem(self.crosshair_h, ignoreBounds=True)
        self.crosshair_v.hide()
        self.crosshair_h.hide()

        # Mouse tracking for crosshair
        self.scene().sigMouseMoved.connect(self.mouse_moved)

        # Enable legend with enhanced styling
        self.legend = self.addLegend(offset=(-10, 10))
        self.legend.setParentItem(self.plotItem)

        # Set log scale for x-axis (time)
        self.setLogMode(x=True, y=False)

        # Enhanced grid
        self.getPlotItem().getAxis('left').setGrid(128)
        self.getPlotItem().getAxis('bottom').setGrid(128)

    def setup_model_comparison(self):
        """Setup multi-model comparison features."""
        self.model_colors = [
            'blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'
        ]
        self.model_styles = [
            Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine
        ]

    def mouse_moved(self, pos):
        """Handle mouse movement for crosshair display."""
        if self.plotItem.sceneBoundingRect().contains(pos):
            mouse_point = self.plotItem.vb.mapSceneToView(pos)
            self.crosshair_v.setPos(mouse_point.x())
            self.crosshair_h.setPos(mouse_point.y())
            self.crosshair_v.show()
            self.crosshair_h.show()
        else:
            self.crosshair_v.hide()
            self.crosshair_h.hide()

    def plot_g2_with_uncertainty(self, x_data, y_data, y_err=None,
                                 fit_result=None, uncertainty_data=None,
                                 outlier_mask=None, label='Data', **kwargs):
        """
        Plot G2 data with comprehensive uncertainty visualization.

        Parameters:
        -----------
        x_data : array-like
            Time delay values
        y_data : array-like
            G2 correlation values
        y_err : array-like, optional
            Error bars for y_data
        fit_result : dict, optional
            Fit results containing fitted curve
        uncertainty_data : dict, optional
            Uncertainty quantification data
        outlier_mask : array-like, optional
            Boolean mask indicating outliers
        label : str
            Label for the data series
        """
        # Clear previous uncertainty visualization
        self.clear_uncertainty_visualization()

        # Plot data points
        symbol_brush = pg.mkBrush(100, 100, 100, 150)
        if outlier_mask is not None:
            # Plot normal points
            normal_mask = ~outlier_mask
            if np.any(normal_mask):
                self.plot(
                    x_data[normal_mask], y_data[normal_mask],
                    pen=None, symbol='o', symbolSize=4,
                    symbolBrush=symbol_brush,
                    name=f'{label} (Normal)'
                )

            # Highlight outliers
            self.highlight_outliers(x_data, y_data, outlier_mask)
        else:
            self.plot(
                x_data, y_data,
                pen=None, symbol='o', symbolSize=4,
                symbolBrush=symbol_brush,
                name=label
            )

        # Add error bars if provided
        if y_err is not None:
            self.add_error_bars_enhanced(x_data, y_data, y_err)

        # Plot fit curve if provided
        if fit_result is not None and 'x_fit' in fit_result and 'y_fit' in fit_result:
            fit_color = kwargs.get('fit_color', 'blue')
            self.plot(
                fit_result['x_fit'], fit_result['y_fit'],
                pen=pg.mkPen(color=fit_color, width=2),
                name=f'{label} Fit'
            )

        # Add uncertainty visualization
        if uncertainty_data is not None:
            self.add_uncertainty_visualization(uncertainty_data, fit_result)

        # Store current data for analysis
        self.current_data = {
            'x_data': x_data,
            'y_data': y_data,
            'y_err': y_err,
            'fit_result': fit_result,
            'uncertainty_data': uncertainty_data,
            'outlier_mask': outlier_mask
        }

    def add_uncertainty_visualization(self, uncertainty_data, fit_result):
        """Add various types of uncertainty visualization."""
        if fit_result is None or 'x_fit' not in fit_result:
            return

        x_fit = fit_result['x_fit']

        # Confidence bands
        if ('confidence_lower' in uncertainty_data and
            'confidence_upper' in uncertainty_data and
            self.show_confidence_bands):

            self.add_confidence_bands(
                x_fit,
                uncertainty_data['confidence_lower'],
                uncertainty_data['confidence_upper'],
                color='red',
                alpha=0.3,
                name='Confidence'
            )

        # Prediction intervals
        if ('prediction_lower' in uncertainty_data and
            'prediction_upper' in uncertainty_data and
            self.show_prediction_intervals):

            self.add_prediction_intervals(
                x_fit,
                uncertainty_data['prediction_lower'],
                uncertainty_data['prediction_upper'],
                color='green',
                name='Prediction'
            )

        # Bootstrap regions
        if ('bootstrap_percentiles' in uncertainty_data and
            self.show_bootstrap_regions):

            self.add_bootstrap_regions(
                x_fit,
                uncertainty_data['bootstrap_percentiles'],
                name='Bootstrap'
            )

    def add_model_comparison(self, models_data, reference_model=None):
        """
        Add multiple model overlays for comparison.

        Parameters:
        -----------
        models_data : dict
            Dictionary of model results {model_name: fit_result}
        reference_model : str, optional
            Name of reference model to highlight
        """
        for i, (model_name, fit_result) in enumerate(models_data.items()):
            if 'x_fit' not in fit_result or 'y_fit' not in fit_result:
                continue

            color = self.model_colors[i % len(self.model_colors)]
            style = self.model_styles[i % len(self.model_styles)]

            # Highlight reference model
            width = 3 if model_name == reference_model else 2
            alpha = 255 if model_name == reference_model else 180

            pen = pg.mkPen(color=color, width=width, style=style)

            self.plot(
                fit_result['x_fit'], fit_result['y_fit'],
                pen=pen,
                name=f'{model_name} Fit'
            )

        # Store models for analysis
        self.fitted_models = models_data

    def add_residual_analysis_overlay(self, residual_data):
        """Add residual analysis overlay."""
        if 'x_data' not in residual_data or 'residuals' not in residual_data:
            return

        x_data = residual_data['x_data']
        residuals = residual_data['residuals']

        # Scale residuals for overlay (show as small bars)
        residual_scale = 0.1 * (np.max(self.current_data.get('y_data', [1])) -
                               np.min(self.current_data.get('y_data', [0])))

        # Create residual bars
        for i, (x, r) in enumerate(zip(x_data, residuals)):
            y_base = np.min(self.current_data.get('y_data', [0])) - 0.1
            y_top = y_base + residual_scale * r / np.max(np.abs(residuals))

            line = pg.PlotCurveItem(
                [x, x], [y_base, y_top],
                pen=pg.mkPen('gray', width=1, alpha=128)
            )
            self.addItem(line)

    def update_outlier_detection(self, threshold=2.5, method='zscore'):
        """Update outlier detection with specified method."""
        if 'y_data' not in self.current_data:
            return

        y_data = self.current_data['y_data']

        if method == 'zscore':
            z_scores = np.abs(stats.zscore(y_data))
            outlier_mask = z_scores > threshold
        elif method == 'iqr':
            q1, q3 = np.percentile(y_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (y_data < lower_bound) | (y_data > upper_bound)
        elif method == 'modified_zscore':
            median = np.median(y_data)
            mad = np.median(np.abs(y_data - median))
            modified_z_scores = 0.6745 * (y_data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
        else:
            outlier_mask = np.zeros(len(y_data), dtype=bool)

        # Update visualization
        if np.any(outlier_mask):
            self.highlight_outliers(
                self.current_data['x_data'],
                self.current_data['y_data'],
                outlier_mask
            )

        # Emit signal
        outlier_indices = np.where(outlier_mask)[0].tolist()
        self.outlier_selection_changed.emit(outlier_indices)

        return outlier_mask

    def set_uncertainty_display_options(self, show_confidence=True,
                                       show_prediction=False,
                                       show_bootstrap=False):
        """Set which uncertainty visualizations to display."""
        self.show_confidence_bands = show_confidence
        self.show_prediction_intervals = show_prediction
        self.show_bootstrap_regions = show_bootstrap

        # Refresh display if data is available
        if self.current_data:
            self.refresh_uncertainty_display()

    def refresh_uncertainty_display(self):
        """Refresh uncertainty display with current settings."""
        if 'uncertainty_data' in self.current_data and self.current_data['uncertainty_data']:
            self.clear_uncertainty_visualization()
            self.add_uncertainty_visualization(
                self.current_data['uncertainty_data'],
                self.current_data['fit_result']
            )

    def export_plot_data(self):
        """Export current plot data for analysis."""
        return {
            'current_data': self.current_data,
            'fitted_models': self.fitted_models,
            'outlier_data': self.outlier_data,
            'uncertainty_settings': {
                'show_confidence_bands': self.show_confidence_bands,
                'show_prediction_intervals': self.show_prediction_intervals,
                'show_bootstrap_regions': self.show_bootstrap_regions
            }
        }

    def import_plot_data(self, plot_data):
        """Import plot data for display."""
        if 'current_data' in plot_data:
            current_data = plot_data['current_data']
            self.plot_g2_with_uncertainty(**current_data)

        if 'fitted_models' in plot_data:
            self.add_model_comparison(plot_data['fitted_models'])

        if 'uncertainty_settings' in plot_data:
            settings = plot_data['uncertainty_settings']
            self.set_uncertainty_display_options(**settings)


class G2PlotControlWidget(QWidget):
    """
    Control widget for G2 plot display options.

    Features:
    - Uncertainty display toggles
    - Outlier detection controls
    - Model comparison options
    - Plot export functionality
    """

    # Signals
    settings_changed = Signal(dict)
    outlier_detection_changed = Signal(str, float)
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Initialize the control UI."""
        layout = QVBoxLayout(self)

        # Uncertainty display controls
        uncertainty_layout = QHBoxLayout()

        self.show_confidence_cb = QCheckBox("Confidence Bands")
        self.show_confidence_cb.setChecked(True)
        uncertainty_layout.addWidget(self.show_confidence_cb)

        self.show_prediction_cb = QCheckBox("Prediction Intervals")
        uncertainty_layout.addWidget(self.show_prediction_cb)

        self.show_bootstrap_cb = QCheckBox("Bootstrap Regions")
        uncertainty_layout.addWidget(self.show_bootstrap_cb)

        layout.addLayout(uncertainty_layout)

        # Outlier detection controls
        outlier_layout = QHBoxLayout()

        outlier_layout.addWidget(QLabel("Outlier Method:"))
        self.outlier_method = QComboBox()
        self.outlier_method.addItems(["zscore", "iqr", "modified_zscore"])
        outlier_layout.addWidget(self.outlier_method)

        outlier_layout.addWidget(QLabel("Threshold:"))
        self.outlier_threshold = QComboBox()
        self.outlier_threshold.addItems(["1.5", "2.0", "2.5", "3.0"])
        self.outlier_threshold.setCurrentText("2.5")
        outlier_layout.addWidget(self.outlier_threshold)

        layout.addLayout(outlier_layout)

    def setup_connections(self):
        """Setup signal connections."""
        # Uncertainty display toggles
        for cb in [self.show_confidence_cb, self.show_prediction_cb, self.show_bootstrap_cb]:
            cb.toggled.connect(self.emit_settings_changed)

        # Outlier detection
        self.outlier_method.currentTextChanged.connect(self.emit_outlier_changed)
        self.outlier_threshold.currentTextChanged.connect(self.emit_outlier_changed)

    def emit_settings_changed(self):
        """Emit settings changed signal."""
        settings = {
            'show_confidence': self.show_confidence_cb.isChecked(),
            'show_prediction': self.show_prediction_cb.isChecked(),
            'show_bootstrap': self.show_bootstrap_cb.isChecked()
        }
        self.settings_changed.emit(settings)

    def emit_outlier_changed(self):
        """Emit outlier detection changed signal."""
        method = self.outlier_method.currentText()
        threshold = float(self.outlier_threshold.currentText())
        self.outlier_detection_changed.emit(method, threshold)

    def get_current_settings(self):
        """Get current control settings."""
        return {
            'uncertainty': {
                'show_confidence': self.show_confidence_cb.isChecked(),
                'show_prediction': self.show_prediction_cb.isChecked(),
                'show_bootstrap': self.show_bootstrap_cb.isChecked()
            },
            'outlier_detection': {
                'method': self.outlier_method.currentText(),
                'threshold': float(self.outlier_threshold.currentText())
            }
        }