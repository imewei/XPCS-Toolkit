# Robust Fitting GUI Integration Guide

## Overview

The XPCS Toolkit now includes advanced robust fitting capabilities with a comprehensive GUI interface. This enhancement provides researchers with powerful tools for G2 correlation analysis while maintaining full compatibility with existing workflows.

## Key Features

### 🔬 Advanced Fitting Algorithms
- **Multi-strategy optimization**: Automatic fallback between Trust Region Reflective → Levenberg-Marquardt → Differential Evolution
- **Outlier detection**: Automatic identification and handling of outlier data points
- **Robust parameter estimation**: Intelligent initial parameter guessing with multiple heuristics
- **Bootstrap analysis**: Statistical validation of fit parameters with confidence intervals

### 📊 Real-time Diagnostics
- **Live residual plots**: Real-time visualization of fitting residuals
- **Parameter convergence tracking**: Monitor parameter evolution during optimization
- **Goodness-of-fit metrics**: R², χ², AIC, BIC, RMSE with color-coded quality indicators
- **Statistical tests**: Shapiro-Wilk normality test, Durbin-Watson autocorrelation test

### 🎯 Interactive Analysis
- **Parameter sensitivity analysis**: Explore how parameters affect the fit
- **Confidence interval visualization**: Bootstrap confidence bands and prediction intervals
- **Outlier highlighting**: Visual identification of problematic data points
- **Multi-model comparison**: Compare different fitting models side-by-side

### 🚀 Enhanced Visualization
- **Uncertainty bands**: Confidence intervals and prediction bands
- **Error bar enhancements**: Improved error visualization with customizable appearance
- **Outlier highlighting**: Clear marking of detected outliers
- **Interactive crosshairs**: Mouse tracking with coordinate display

## Getting Started

### Installation and Setup

The robust fitting GUI is automatically available if the enhanced components are installed. No additional setup is required.

### Basic Usage

1. **Load your XPCS data** as usual in the main XPCS Toolkit interface
2. **Navigate to the G2 tab**
3. **Plot your G2 data** using the traditional interface
4. **Click "🔬 Advanced Fitting"** to open the robust fitting interface

### Interface Modes

#### Traditional Mode (Default)
- Uses the existing G2 fitting interface
- Fast, reliable fitting for well-behaved data
- Minimal computational overhead
- Ideal for routine analysis

#### Robust Mode (Advanced)
- Full robust fitting capabilities
- Comprehensive diagnostics and visualization
- Interactive parameter analysis
- Best for challenging datasets or publication-quality analysis

## Detailed Feature Guide

### Robust Fitting Control Panel

#### Basic Controls Tab
- **Optimization Method**: Choose from automatic multi-strategy, single methods, or Bayesian MCMC
- **Convergence Tolerances**: Adjust fitting precision and maximum iterations
- **Robustness Options**: Enable outlier detection and adaptive weight adjustment

#### Advanced Controls Tab
- **Strategy Configuration**: Performance tracking, caching, and timeout settings
- **Parameter Estimation**: Multiple initial parameter estimation methods
- **Error Handling**: Fallback methods and retry logic

#### Diagnostics Tab
- **Diagnostic Options**: Comprehensive analysis including residuals, correlations, and goodness-of-fit
- **Visualization Options**: Confidence bands, prediction intervals, outlier highlighting
- **Confidence Level**: Adjustable confidence level for statistical analysis

#### Bootstrap Tab
- **Bootstrap Analysis**: Statistical validation with multiple bootstrap methods
- **Cross-Validation**: Model validation with k-fold cross-validation
- **Parallel Processing**: Utilize multiple CPU cores for faster analysis

### Real-time Diagnostic Visualization

#### Residual Analysis
- **Residuals vs Fitted Values**: Detect heteroscedasticity and patterns
- **Q-Q Plot**: Test for normality of residuals
- **Outlier Detection**: Configurable threshold for outlier identification

#### Parameter Convergence
- **Convergence History**: Track parameter evolution during optimization
- **Fit Quality Evolution**: Monitor R², χ², and AIC during fitting

#### Statistical Metrics
- **Goodness-of-fit**: R², Adjusted R², χ², AIC, BIC, RMSE
- **Statistical Tests**: Normality, autocorrelation, heteroscedasticity tests
- **Outlier Statistics**: Count, percentage, and indices of detected outliers

### Interactive Parameter Analysis

#### Parameter Adjustment
- **Real-time Sliders**: Adjust parameters and see immediate effects
- **Sensitivity Analysis**: Quantify parameter importance using Sobol indices
- **Correlation Matrix**: Visualize parameter correlations and identify collinearity

#### Confidence Intervals
- **Confidence Bands**: Visualize parameter uncertainty
- **Prediction Intervals**: Estimate prediction uncertainty
- **Bootstrap Distributions**: Full parameter distribution visualization

### Enhanced Plotting Features

#### Uncertainty Visualization
- **Confidence Bands**: Shaded regions showing fit uncertainty
- **Prediction Intervals**: Dotted lines showing prediction bounds
- **Error Bars**: Enhanced error bar display with customizable appearance

#### Outlier Handling
- **Automatic Detection**: Multiple algorithms (Z-score, IQR, Modified Z-score)
- **Visual Highlighting**: Clear marking with different symbols and colors
- **Interactive Selection**: Click to inspect individual outliers

#### Multi-model Comparison
- **Model Overlays**: Display multiple fit models simultaneously
- **Color Coding**: Distinguish between different models
- **Statistical Comparison**: AIC/BIC-based model selection

## User Experience Features

### Progressive Disclosure
- **Smart Defaults**: Reasonable default settings for most use cases
- **Contextual Help**: Tooltips and help dialogs explaining features
- **Recommendations**: Intelligent suggestions based on data quality

### Performance Optimization
- **Background Processing**: Non-blocking computations preserve GUI responsiveness
- **Progress Tracking**: Real-time progress bars and status updates
- **Caching**: Intelligent caching of computationally expensive operations

### Export and Integration
- **Result Export**: Save fitting results in multiple formats (NPZ, JSON, CSV)
- **Plot Export**: High-quality figure export for publications
- **Integration**: Seamless integration with existing XPCS workflows

## Best Practices

### When to Use Robust Fitting

✅ **Recommended for:**
- Data with suspected outliers
- Publication-quality analysis requiring uncertainty estimates
- Challenging datasets where traditional fitting fails
- Parameter sensitivity studies
- Model comparison and validation

❌ **May be overkill for:**
- High-quality, clean datasets
- Routine analysis where speed is critical
- Simple exploratory analysis

### Optimization Tips

1. **Start with Traditional Mode** for initial exploration
2. **Use Robust Mode** when you need detailed analysis
3. **Enable Diagnostics** for publication-quality fits
4. **Check Residual Plots** to validate model assumptions
5. **Use Bootstrap Analysis** for confidence intervals
6. **Compare Multiple Models** using AIC/BIC criteria

### Common Workflows

#### Publication-Quality Analysis
1. Load data and perform initial traditional fit
2. Switch to robust mode with full diagnostics enabled
3. Run bootstrap analysis for confidence intervals
4. Check residual plots and statistical tests
5. Export results with uncertainty estimates

#### Outlier Investigation
1. Enable outlier detection in robust mode
2. Examine residual plots for patterns
3. Use interactive tools to inspect outlier points
4. Decide whether to exclude or handle outliers
5. Re-fit with appropriate outlier handling

#### Model Comparison
1. Fit multiple models (single vs double exponential)
2. Compare AIC/BIC values
3. Examine residual patterns for each model
4. Use cross-validation for model selection
5. Report best model with statistical justification

## Troubleshooting

### Common Issues

#### "No Data Available"
- **Cause**: Robust fitting opened before plotting G2 data
- **Solution**: Plot G2 data first, then open robust fitting interface

#### "Optimization Failed"
- **Cause**: Poor initial parameters or inappropriate bounds
- **Solution**: Try different parameter estimation methods or adjust bounds

#### "Poor Fit Quality"
- **Cause**: Model mismatch or systematic errors in data
- **Solution**: Try different models, check for outliers, examine residual plots

#### Performance Issues
- **Cause**: Large datasets or complex analysis
- **Solution**: Reduce bootstrap samples, disable real-time updates, use faster methods

### Getting Help

1. **Tooltips**: Hover over controls for quick explanations
2. **Help Dialogs**: Click "?" buttons for detailed information
3. **Documentation**: Refer to this guide and API documentation
4. **Logs**: Check application logs for detailed error messages

## Technical Details

### Implementation Architecture

The robust fitting GUI is built on a modular architecture:

- **Control Panels**: Reusable widgets for parameter configuration
- **Diagnostic Widgets**: Real-time visualization components
- **Enhanced Plotting**: Advanced plotting with uncertainty visualization
- **Integration Layer**: Seamless connection to existing XPCS infrastructure

### Performance Characteristics

- **Memory Usage**: Optimized for large datasets with lazy loading
- **CPU Usage**: Multi-threaded operations where appropriate
- **Responsiveness**: Background processing maintains GUI responsiveness
- **Caching**: Intelligent caching reduces redundant computations

### Extensibility

The modular design allows for easy extension:

- **New Optimization Methods**: Add custom optimization algorithms
- **Additional Diagnostics**: Implement new diagnostic visualizations
- **Custom Models**: Support for user-defined fitting functions
- **Export Formats**: Add new export formats as needed

## Future Enhancements

Planned future improvements include:

- **Machine Learning Integration**: Automated parameter estimation using ML
- **Advanced Model Selection**: Information-theoretic model selection
- **Parallel Processing**: GPU acceleration for large datasets
- **Cloud Integration**: Remote computation for intensive analysis
- **Interactive Tutorials**: Built-in tutorials for new users

## Contributing

The robust fitting GUI is designed to be extensible and welcomes contributions:

1. **Bug Reports**: Submit issues with detailed reproduction steps
2. **Feature Requests**: Suggest new features with use cases
3. **Code Contributions**: Follow the established coding standards
4. **Documentation**: Help improve documentation and examples

## License and Citation

This enhancement maintains the same license as the main XPCS Toolkit. When using robust fitting features in publications, please cite both the main XPCS Toolkit and acknowledge the robust fitting enhancements.

---

*For technical support or questions about the robust fitting GUI, please refer to the main XPCS Toolkit documentation or contact the development team.*
