"""Result dataclasses for Bayesian fitting.

This module defines the data structures for fit results, diagnostics,
and sampler configuration.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd
    from nlsq.diagnostics import ModelHealthReport
    from nlsq.result import CurveFitResult
    from numpy.typing import ArrayLike


def safe_version(package_name: str) -> str:
    """Safely retrieve package version for reproducibility tracking.

    Per Technical Guidelines, fit artifacts must include software versions.
    This function provides robust version retrieval that never raises exceptions.

    Parameters
    ----------
    package_name : str
        Name of the package to get version for

    Returns
    -------
    str
        Version string, or "unknown" if version cannot be determined
    """
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        # Fallback: try __version__ attribute
        try:
            import importlib

            module = importlib.import_module(package_name)
            return getattr(module, "__version__", "unknown")
        except Exception:
            return "unknown"


@dataclass
class SamplerConfig:
    """Configuration for NUTS sampler.

    Attributes
    ----------
    num_warmup : int
        Number of warmup (burn-in) samples (default: 500)
    num_samples : int
        Number of posterior samples per chain (default: 1000)
    num_chains : int
        Number of MCMC chains (default: 4)
    target_accept_prob : float
        Target acceptance probability for NUTS (default: 0.8)
    max_tree_depth : int
        Maximum tree depth for NUTS (default: 10)
    random_seed : int or None
        Random seed for reproducibility (default: None)
    """

    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 4
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    random_seed: int | None = None


@dataclass
class FitDiagnostics:
    """Convergence diagnostics for MCMC sampling.

    Attributes
    ----------
    r_hat : dict[str, float]
        Gelman-Rubin statistic per parameter
    ess_bulk : dict[str, int]
        Bulk ESS per parameter
    ess_tail : dict[str, int]
        Tail ESS per parameter
    divergences : int
        Number of divergent transitions
    max_treedepth_reached : int
        Count of max treedepth events
    bfmi : float | None
        Bayesian Fraction of Missing Information (mean across chains).
        Added per Technical Guidelines for Bayesian inference compliance.
    converged : bool
        True if all diagnostics pass thresholds

    Convergence Thresholds
    ----------------------
    r_hat < 1.01 : All parameters must converge
    ess_bulk > 400 : Sufficient effective samples
    ess_tail > 400 : Sufficient tail samples
    divergences == 0 : No divergent transitions
    bfmi >= 0.2 : Adequate exploration (if computed)
    """

    r_hat: dict[str, float] = field(default_factory=dict)
    ess_bulk: dict[str, int] = field(default_factory=dict)
    ess_tail: dict[str, int] = field(default_factory=dict)
    divergences: int = 0
    max_treedepth_reached: int = 0
    bfmi: float | None = None  # NEW: Per Technical Guidelines

    @property
    def converged(self) -> bool:
        """Check if all diagnostics pass thresholds."""
        # R-hat threshold
        if any(r > 1.01 for r in self.r_hat.values()):
            return False

        # ESS thresholds
        if any(e < 400 for e in self.ess_bulk.values()):
            return False
        if any(e < 400 for e in self.ess_tail.values()):
            return False

        # No divergences
        if self.divergences > 0:
            return False

        # BFMI check (per Technical Guidelines)
        if self.bfmi is not None and self.bfmi < 0.2:
            return False

        return True


@dataclass
class NLSQResult:
    """Result from nonlinear least squares fitting.

    This class wraps NLSQ 0.6.0's CurveFitResult and delegates statistical
    properties to the native result for accuracy and consistency.

    Attributes
    ----------
    params : dict[str, float]
        Point estimates for each parameter
    converged : bool
        Whether optimization converged
    chi_squared : float
        Reduced chi-squared statistic
    pcov_valid : bool
        Covariance validity flag (FR-021)
    pcov_message : str
        Validation message describing covariance status
    native_result : CurveFitResult, optional
        NLSQ 0.6.0 native result object for property delegation
    _param_names : list[str]
        Ordered parameter names for covariance indexing

    Properties (delegated to native_result when available)
    ------------------------------------------------------
    r_squared : float
        Coefficient of determination (R²). Range: (-∞, 1], where 1 is perfect fit.
    adj_r_squared : float
        Adjusted R² accounting for number of parameters.
    rmse : float
        Root mean squared error. Lower is better.
    mae : float
        Mean absolute error. Robust to outliers.
    aic : float
        Akaike Information Criterion. Lower is better for model selection.
    bic : float
        Bayesian Information Criterion. Penalizes complexity more than AIC.
    residuals : ndarray
        Fit residuals as numpy array.
    predictions : ndarray
        Model predictions at input x values.
    covariance : ndarray
        Parameter covariance matrix (n_params x n_params).
    confidence_intervals : dict[str, tuple[float, float]]
        Parameter confidence intervals at 95% level.
    diagnostics : ModelHealthReport or None
        NLSQ model health diagnostics (if compute_diagnostics=True).
    is_healthy : bool
        Whether the fit passes all health checks.
    health_score : int
        Health score (0-100).
    condition_number : float
        Condition number from identifiability diagnostics.
    """

    # Core fields (required)
    params: dict[str, float]
    chi_squared: float
    converged: bool

    # Covariance validation
    pcov_valid: bool = True
    pcov_message: str = ""

    # Native result for delegation (NEW - T018)
    native_result: CurveFitResult | None = None

    # Parameter names for covariance indexing (NEW - T019)
    _param_names: list[str] = field(default_factory=list)

    # Legacy storage fields (used when native_result is None for backward compat)
    _covariance: np.ndarray | None = field(default=None, repr=False)
    _residuals: np.ndarray | None = field(default=None, repr=False)
    _r_squared: float = field(default=0.0, repr=False)
    _adj_r_squared: float = field(default=0.0, repr=False)
    _rmse: float = field(default=0.0, repr=False)
    _mae: float = field(default=0.0, repr=False)
    _aic: float = field(default=0.0, repr=False)
    _bic: float = field(default=0.0, repr=False)
    _confidence_intervals: dict[str, tuple[float, float]] = field(
        default_factory=dict, repr=False
    )
    _predictions: np.ndarray | None = field(default=None, repr=False)

    # Backward compatibility aliases for __init__
    def __post_init__(self) -> None:
        """Initialize param names from params dict if not provided."""
        if not self._param_names:
            self._param_names = list(self.params.keys())

    # T020: r_squared property delegation
    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²)."""
        if self.native_result is not None:
            return self.native_result.r_squared
        return self._r_squared

    @r_squared.setter
    def r_squared(self, value: float) -> None:
        """Set r_squared (for backward compat initialization)."""
        self._r_squared = value

    # T021: adj_r_squared property delegation
    @property
    def adj_r_squared(self) -> float:
        """Adjusted R² accounting for number of parameters."""
        if self.native_result is not None:
            return self.native_result.adj_r_squared
        return self._adj_r_squared

    @adj_r_squared.setter
    def adj_r_squared(self, value: float) -> None:
        """Set adj_r_squared (for backward compat initialization)."""
        self._adj_r_squared = value

    # T022: rmse property delegation
    @property
    def rmse(self) -> float:
        """Root mean squared error."""
        if self.native_result is not None:
            return self.native_result.rmse
        return self._rmse

    @rmse.setter
    def rmse(self, value: float) -> None:
        """Set rmse (for backward compat initialization)."""
        self._rmse = value

    # T023: mae property delegation
    @property
    def mae(self) -> float:
        """Mean absolute error."""
        if self.native_result is not None:
            return self.native_result.mae
        return self._mae

    @mae.setter
    def mae(self, value: float) -> None:
        """Set mae (for backward compat initialization)."""
        self._mae = value

    # T024: aic property delegation
    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        if self.native_result is not None:
            return self.native_result.aic
        return self._aic

    @aic.setter
    def aic(self, value: float) -> None:
        """Set aic (for backward compat initialization)."""
        self._aic = value

    # T025: bic property delegation
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        if self.native_result is not None:
            return self.native_result.bic
        return self._bic

    @bic.setter
    def bic(self, value: float) -> None:
        """Set bic (for backward compat initialization)."""
        self._bic = value

    # T026: residuals property delegation
    @property
    def residuals(self) -> np.ndarray:
        """Fit residuals as numpy array."""
        if self.native_result is not None:
            return np.asarray(self.native_result.residuals)
        if self._residuals is not None:
            return self._residuals
        return np.array([])

    @residuals.setter
    def residuals(self, value: np.ndarray) -> None:
        """Set residuals (for backward compat initialization)."""
        self._residuals = value

    # T027: predictions property delegation
    @property
    def predictions(self) -> np.ndarray | None:
        """Model predictions at input x values."""
        if self.native_result is not None:
            return np.asarray(self.native_result.predictions)
        return self._predictions

    @predictions.setter
    def predictions(self, value: np.ndarray | None) -> None:
        """Set predictions (for backward compat initialization)."""
        self._predictions = value

    # T035: covariance property delegation
    @property
    def covariance(self) -> np.ndarray:
        """Parameter covariance matrix."""
        if self.native_result is not None:
            return np.asarray(self.native_result.pcov)
        if self._covariance is not None:
            return self._covariance
        n = len(self.params)
        return np.zeros((n, n))

    @covariance.setter
    def covariance(self, value: np.ndarray) -> None:
        """Set covariance (for backward compat initialization)."""
        self._covariance = value

    # T029: confidence_intervals property delegation
    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        """Parameter confidence intervals at 95% level."""
        if self.native_result is not None:
            # Cast to dict in case it's a method or other type in strict mypy
            from typing import cast

            return cast(
                dict[str, tuple[float, float]], self.native_result.confidence_intervals
            )
        return self._confidence_intervals

    @confidence_intervals.setter
    def confidence_intervals(self, value: dict[str, tuple[float, float]]) -> None:
        """Set confidence_intervals (for backward compat initialization)."""
        self._confidence_intervals = value

    # T31: diagnostics property delegation
    @property
    def diagnostics(self) -> ModelHealthReport | None:
        """NLSQ model health diagnostics."""
        if self.native_result is not None:
            return self.native_result.diagnostics
        return None

    # T032: is_healthy property
    @property
    def is_healthy(self) -> bool:
        """Whether the fit passes all health checks."""
        if self.diagnostics is not None:
            return str(self.diagnostics.status) == "healthy"
        return True  # Default to healthy when no diagnostics

    # T033: health_score property
    @property
    def health_score(self) -> int:
        """Health score (0-100)."""
        if self.diagnostics is not None:
            return int(self.diagnostics.health_score)
        return 100  # Default to perfect health when no diagnostics

    # T034: condition_number property
    @property
    def condition_number(self) -> float:
        """Condition number from identifiability diagnostics."""
        if (
            self.diagnostics is not None
            and self.diagnostics.identifiability is not None
        ):
            return self.diagnostics.identifiability.condition_number
        return 1.0  # Default to well-conditioned

    def get_param_uncertainty(self, param: str) -> float:
        """Get standard error for a parameter from the covariance matrix.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        float
            Standard error (sqrt of diagonal covariance element)
        """
        param_names = list(self.params.keys())
        if param not in param_names:
            raise KeyError(f"Parameter '{param}' not found in params")
        idx = param_names.index(param)
        return float(np.sqrt(self.covariance[idx, idx]))

    def get_confidence_interval(
        self, param: str, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Get confidence interval for a parameter.

        Parameters
        ----------
        param : str
            Parameter name
        alpha : float
            Significance level (default: 0.05 for 95% CI)

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds of confidence interval
        """
        if param in self.confidence_intervals:
            return self.confidence_intervals[param]

        # Compute from covariance if not cached
        from scipy import stats

        param_value = self.params[param]
        std_err = self.get_param_uncertainty(param)
        z = stats.norm.ppf(1 - alpha / 2)
        return (param_value - z * std_err, param_value + z * std_err)

    # T030: get_prediction_interval method delegation
    def get_prediction_interval(
        self, x: ArrayLike, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get prediction interval at new x values.

        Delegates to native_result.prediction_interval() when available.

        Parameters
        ----------
        x : array_like
            X values at which to compute prediction intervals
        alpha : float
            Significance level (default: 0.05 for 95% PI)

        Returns
        -------
        tuple[ndarray, ndarray]
            (lower, upper) bounds of prediction interval as numpy arrays
        """
        x = np.asarray(x)
        if self.native_result is not None:
            lower, upper = self.native_result.prediction_interval(x=x, alpha=alpha)
            return np.asarray(lower), np.asarray(upper)

        # Fallback: return simple prediction +/- 2*rmse (rough approximation)
        predictions = self.predictions
        if predictions is not None and len(predictions) == len(x):
            margin = 2 * self.rmse
            return predictions - margin, predictions + margin

        # No prediction interval available
        return np.full_like(x, np.nan), np.full_like(x, np.nan)

    def summary(self) -> str:
        """Generate a formatted summary of the fit results.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = ["NLSQ Fit Results", "=" * 50]

        # Convergence status
        status = "Converged" if self.converged else "Did not converge"
        lines.append(f"Status: {status}")
        lines.append("")

        # Fit quality metrics
        lines.append("Fit Quality:")
        lines.append(f"  R²:            {self.r_squared:.6f}")
        lines.append(f"  Adjusted R²:   {self.adj_r_squared:.6f}")
        lines.append(f"  RMSE:          {self.rmse:.6e}")
        lines.append(f"  MAE:           {self.mae:.6e}")
        lines.append(f"  χ² (reduced):  {self.chi_squared:.4f}")
        lines.append("")

        # Model selection criteria
        lines.append("Model Selection:")
        lines.append(f"  AIC:           {self.aic:.2f}")
        lines.append(f"  BIC:           {self.bic:.2f}")
        lines.append("")

        # Parameters with uncertainties
        lines.append("Parameters:")
        for name, value in self.params.items():
            std_err = self.get_param_uncertainty(name)
            ci = self.get_confidence_interval(name)
            lines.append(f"  {name}: {value:.6e} ± {std_err:.6e}")
            lines.append(f"    95% CI: [{ci[0]:.6e}, {ci[1]:.6e}]")

        # Covariance validity
        lines.append("")
        if self.pcov_valid:
            lines.append("Covariance: Valid")
        else:
            lines.append(f"Covariance: Invalid - {self.pcov_message}")

        return "\n".join(lines)

    def plot(self, model, x_data, y_data, **kwargs):
        """Plot fit with uncertainty band.

        Parameters
        ----------
        model : callable
            Model function
        x_data, y_data : array-like
            Original data
        **kwargs
            Additional arguments for visualization

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .visualization import plot_nlsq_fit

        return plot_nlsq_fit(self, model, x_data, y_data, **kwargs)


@dataclass
class FitResult:
    """Container for Bayesian fitting results.

    Attributes
    ----------
    samples : dict[str, ndarray]
        Posterior samples {param_name: (n_samples,)}
    summary : DataFrame
        Summary statistics per parameter
    diagnostics : FitDiagnostics
        Convergence diagnostics
    nlsq_init : dict[str, float]
        NLSQ warm-start point estimates
    arviz_data : InferenceData
        ArviZ-compatible data for plotting (FR-015)
    config : SamplerConfig | None
        Sampler configuration used for this fit (per Technical Guidelines)
    x : ndarray | None
        Input x data (for reproducibility metadata)
    """

    samples: dict[str, np.ndarray]
    summary: pd.DataFrame | None = None
    diagnostics: FitDiagnostics = field(default_factory=FitDiagnostics)
    nlsq_init: dict[str, float] = field(default_factory=dict)
    arviz_data: az.InferenceData | None = None
    config: SamplerConfig | None = None  # Per Technical Guidelines
    x: np.ndarray | None = None  # For data_metadata

    def get_mean(self, param: str) -> float:
        """Get posterior mean for parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        float
            Posterior mean
        """
        if param not in self.samples:
            raise KeyError(f"Parameter '{param}' not found in samples")
        return float(np.mean(self.samples[param]))

    def get_std(self, param: str) -> float:
        """Get posterior standard deviation for parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        float
            Posterior standard deviation
        """
        if param not in self.samples:
            raise KeyError(f"Parameter '{param}' not found in samples")
        return float(np.std(self.samples[param]))

    def get_hdi(self, param: str, prob: float = 0.94) -> tuple[float, float]:
        """Get highest density interval for parameter.

        Parameters
        ----------
        param : str
            Parameter name
        prob : float
            Probability mass for HDI (default: 0.94)

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds of HDI
        """
        if param not in self.samples:
            raise KeyError(f"Parameter '{param}' not found in samples")

        samples = self.samples[param]

        # Simple percentile-based HDI approximation
        alpha = 1 - prob
        lower_pct = 100 * (alpha / 2)
        upper_pct = 100 * (1 - alpha / 2)

        return (
            float(np.percentile(samples, lower_pct)),
            float(np.percentile(samples, upper_pct)),
        )

    def get_samples(self, param: str) -> np.ndarray:
        """Get posterior samples for parameter.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        ndarray
            Posterior samples array
        """
        if param not in self.samples:
            raise KeyError(f"Parameter '{param}' not found in samples")
        return self.samples[param]

    def predict(self, x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Generate posterior predictive samples.

        Note: This requires the model function to be stored or passed.
        Default implementation returns zeros. Override in subclass or
        use the visualization module directly.

        Parameters
        ----------
        x : array-like
            X values for prediction

        Returns
        -------
        tuple[ndarray, ndarray]
            (mean prediction, std prediction)
        """
        x = np.asarray(x)
        # Placeholder - actual implementation requires model function
        return np.zeros_like(x), np.zeros_like(x)

    def to_dict(self) -> dict:
        """Convert to serializable dictionary.

        Per Technical Guidelines, exports include:
        - versions: Package versions for reproducibility
        - sampler_config: Sampler parameters used
        - data_metadata: Data characteristics
        - diagnostics: Including BFMI

        Returns
        -------
        dict
            Dictionary representation with full reproducibility metadata
        """
        # T026: Build versions dictionary for reproducibility
        versions = {
            "xpcsviewer": safe_version("xpcs-toolkit"),
            "numpyro": safe_version("numpyro"),
            "jax": safe_version("jax"),
            "arviz": safe_version("arviz"),
            "nlsq": safe_version("nlsq"),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }

        # T027: Build sampler_config dictionary
        sampler_config: dict[str, Any] = {}
        if self.config is not None:
            sampler_config = {
                "num_warmup": self.config.num_warmup,
                "num_samples": self.config.num_samples,
                "num_chains": self.config.num_chains,
                "target_accept_prob": self.config.target_accept_prob,
                "max_tree_depth": self.config.max_tree_depth,
                "random_seed": self.config.random_seed,
            }

        # T028: Build data_metadata dictionary
        data_metadata: dict[str, Any] = {}
        if self.x is not None:
            x_arr = np.asarray(self.x)
            data_metadata = {
                "n_points": len(x_arr),
                "x_range": [float(x_arr.min()), float(x_arr.max())],
            }

        return {
            "samples": {k: v.tolist() for k, v in self.samples.items()},
            "nlsq_init": self.nlsq_init,
            "diagnostics": {
                "r_hat": self.diagnostics.r_hat,
                "ess_bulk": self.diagnostics.ess_bulk,
                "ess_tail": self.diagnostics.ess_tail,
                "divergences": self.diagnostics.divergences,
                "max_treedepth_reached": self.diagnostics.max_treedepth_reached,
                "bfmi": self.diagnostics.bfmi,  # Per Technical Guidelines
                "converged": self.diagnostics.converged,
            },
            "versions": versions,
            "sampler_config": sampler_config,
            "data_metadata": data_metadata,
        }

    def plot_posterior_predictive(self, model, x_data, y_data, **kwargs):
        """Plot posterior predictive with credible interval.

        Parameters
        ----------
        model : callable
            Model function
        x_data, y_data : array-like
            Original data
        **kwargs
            Additional arguments for visualization

        Returns
        -------
        matplotlib.figure.Figure
        """
        from .visualization import plot_posterior_predictive

        return plot_posterior_predictive(self, model, x_data, y_data, **kwargs)

    def generate_diagnostics(self, output_dir=None, formats=("pdf", "png")) -> dict:
        """Generate ArviZ diagnostic plots.

        Parameters
        ----------
        output_dir : str or Path, optional
            Directory for output files
        formats : tuple
            Output formats

        Returns
        -------
        dict
            Mapping of plot type to figure or file path
        """
        from .visualization import generate_arviz_diagnostics

        return generate_arviz_diagnostics(
            self.arviz_data, output_dir=output_dir, formats=formats
        )
