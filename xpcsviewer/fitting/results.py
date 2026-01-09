"""Result dataclasses for Bayesian fitting.

This module defines the data structures for fit results, diagnostics,
and sampler configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd
    from numpy.typing import ArrayLike


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
    converged : bool
        True if all diagnostics pass thresholds

    Convergence Thresholds
    ----------------------
    r_hat < 1.01 : All parameters must converge
    ess_bulk > 400 : Sufficient effective samples
    ess_tail > 400 : Sufficient tail samples
    divergences == 0 : No divergent transitions
    """

    r_hat: dict[str, float] = field(default_factory=dict)
    ess_bulk: dict[str, int] = field(default_factory=dict)
    ess_tail: dict[str, int] = field(default_factory=dict)
    divergences: int = 0
    max_treedepth_reached: int = 0

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

        return True


@dataclass
class NLSQResult:
    """Result from nonlinear least squares fitting.

    Attributes
    ----------
    params : dict[str, float]
        Point estimates for each parameter
    covariance : ndarray
        Parameter covariance matrix (n_params x n_params)
    residuals : ndarray
        Fit residuals
    chi_squared : float
        Reduced chi-squared statistic
    converged : bool
        Whether optimization converged
    pcov_valid : bool
        Covariance validity flag (FR-021)
    pcov_message : str
        Validation message describing covariance status
    r_squared : float
        Coefficient of determination (R²). Range: (-∞, 1], where 1 is perfect fit.

        Note: When measurement uncertainties (sigma/yerr) are provided, NLSQ 0.6.0
        computes R² using weighted residuals. This can result in negative R² values
        if the weighted model fit is worse than the weighted mean. This is expected
        behavior for weighted least squares - use ``chi_squared`` for fit quality
        assessment when uncertainties are provided.
    adj_r_squared : float
        Adjusted R² accounting for number of parameters. Subject to the same
        weighted residuals behavior as ``r_squared``.
    rmse : float
        Root mean squared error. Lower is better.
    mae : float
        Mean absolute error. Robust to outliers.
    aic : float
        Akaike Information Criterion. Lower is better for model selection.
    bic : float
        Bayesian Information Criterion. Penalizes complexity more than AIC.
    confidence_intervals : dict[str, tuple[float, float]]
        Parameter confidence intervals at 95% level {param: (lower, upper)}.
    predictions : ndarray or None
        Model predictions at input x values.
    model_diagnostics : dict or None
        NLSQ model health diagnostics (if compute_diagnostics=True).
    """

    params: dict[str, float]
    covariance: np.ndarray
    residuals: np.ndarray
    chi_squared: float
    converged: bool
    pcov_valid: bool = True
    pcov_message: str = ""
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    model_diagnostics: dict | None = None

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
        self, param: str, alpha: float = 0.95
    ) -> tuple[float, float]:
        """Get confidence interval for a parameter.

        Parameters
        ----------
        param : str
            Parameter name
        alpha : float
            Confidence level (default: 0.95 for 95% CI)

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
        z = stats.norm.ppf((1 + alpha) / 2)
        return (param_value - z * std_err, param_value + z * std_err)

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
    """

    samples: dict[str, np.ndarray]
    summary: pd.DataFrame | None = None
    diagnostics: FitDiagnostics = field(default_factory=FitDiagnostics)
    nlsq_init: dict[str, float] = field(default_factory=dict)
    arviz_data: az.InferenceData | None = None

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

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            "samples": {k: v.tolist() for k, v in self.samples.items()},
            "nlsq_init": self.nlsq_init,
            "diagnostics": {
                "r_hat": self.diagnostics.r_hat,
                "ess_bulk": self.diagnostics.ess_bulk,
                "ess_tail": self.diagnostics.ess_tail,
                "divergences": self.diagnostics.divergences,
                "max_treedepth_reached": self.diagnostics.max_treedepth_reached,
                "converged": self.diagnostics.converged,
            },
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
