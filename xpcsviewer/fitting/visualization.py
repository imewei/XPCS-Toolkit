"""Fitting visualization module (FR-013 to FR-021).

This module provides visualization functions for NLSQ and Bayesian
fitting results, including uncertainty bands, diagnostic plots,
and publication-quality output.

NLSQ 0.6.0 Enhanced Features:
- Prediction interval visualization
- R², RMSE, AIC/BIC display on plots
- Confidence interval annotations
- Model comparison with information criteria
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    from .results import FitResult, NLSQResult


# Publication style preset (FR-019)
PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_publication_style():
    """Apply publication-quality matplotlib style (FR-019)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(PUBLICATION_STYLE)


def validate_pcov(pcov, param_names=None) -> tuple[bool, str]:
    """Validate covariance matrix before computing uncertainty bands (FR-021).

    Checks:
    - pcov is not None
    - All values are finite (no inf/nan)
    - Matrix is positive semi-definite

    Parameters
    ----------
    pcov : ndarray or None
        Covariance matrix to validate
    param_names : list, optional
        Parameter names for error messages

    Returns
    -------
    is_valid : bool
        True if covariance is valid
    message : str
        Validation message (error description if invalid)
    """
    import numpy as np

    if pcov is None:
        return False, "Covariance matrix is None"

    pcov = np.asarray(pcov)

    if not np.all(np.isfinite(pcov)):
        return False, "Covariance matrix contains inf or nan values"

    # Check positive semi-definite (eigenvalues >= 0)
    try:
        eigenvalues = np.linalg.eigvalsh(pcov)
        if np.any(eigenvalues < -1e-10):  # Small tolerance for numerical issues
            return False, "Covariance matrix is not positive semi-definite"
    except np.linalg.LinAlgError:
        return False, "Failed to compute eigenvalues of covariance matrix"

    return True, "Covariance matrix is valid"


def compute_uncertainty_band(model, x_pred, popt, pcov, confidence=0.95):
    """Compute prediction uncertainty band via error propagation (FR-016).

    Formula: σ_y(x) = sqrt(diag(J @ pcov @ J.T))

    Parameters
    ----------
    model : callable
        Model function: y = model(x, *params)
    x_pred : ndarray
        X values for prediction
    popt : ndarray
        Fitted parameters
    pcov : ndarray
        Parameter covariance matrix (n_params x n_params)
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns
    -------
    y_fit : ndarray
        Fitted curve values
    y_lower : ndarray
        Lower bound of confidence band
    y_upper : ndarray
        Upper bound of confidence band
    """
    import numpy as np
    from scipy import stats

    x_pred = np.asarray(x_pred)
    popt = np.asarray(popt)
    pcov = np.asarray(pcov)

    # Compute fit curve
    y_fit = model(x_pred, *popt)

    # Compute Jacobian via finite differences
    eps = 1e-8
    n_params = len(popt)
    n_points = len(x_pred)
    jacobian = np.zeros((n_points, n_params))

    for i in range(n_params):
        popt_plus = popt.copy()
        popt_plus[i] += eps
        jacobian[:, i] = (model(x_pred, *popt_plus) - y_fit) / eps

    # Variance: diag(J @ pcov @ J.T)
    # Efficient computation: sum((J @ pcov) * J, axis=1)
    variance = np.sum((jacobian @ pcov) * jacobian, axis=1)
    variance = np.maximum(variance, 0)  # Ensure non-negative
    sigma = np.sqrt(variance)

    # Confidence interval
    z = stats.norm.ppf((1 + confidence) / 2)
    y_lower = y_fit - z * sigma
    y_upper = y_fit + z * sigma

    return y_fit, y_lower, y_upper


def compute_prediction_interval(model, x_pred, popt, pcov, residuals, confidence=0.95):
    """Compute prediction interval including residual variance (NLSQ 0.6.0).

    Prediction intervals are wider than confidence intervals because they
    account for both parameter uncertainty AND observation noise.

    Formula: PI = CI ± t * σ_residuals

    Parameters
    ----------
    model : callable
        Model function: y = model(x, *params)
    x_pred : ndarray
        X values for prediction
    popt : ndarray
        Fitted parameters
    pcov : ndarray
        Parameter covariance matrix
    residuals : ndarray
        Fit residuals (y - y_pred) from training data
    confidence : float
        Confidence level (default: 0.95 for 95% PI)

    Returns
    -------
    y_fit : ndarray
        Fitted curve values
    pi_lower : ndarray
        Lower bound of prediction interval
    pi_upper : ndarray
        Upper bound of prediction interval
    """
    import numpy as np
    from scipy import stats

    # Get confidence interval
    y_fit, ci_lower, ci_upper = compute_uncertainty_band(
        model, x_pred, popt, pcov, confidence
    )

    # Estimate residual standard deviation
    n_data = len(residuals)
    n_params = len(popt)
    dof = max(1, n_data - n_params)
    sigma_residuals = np.sqrt(np.sum(residuals**2) / dof)

    # t-value for prediction interval
    t_value = stats.t.ppf((1 + confidence) / 2, dof)

    # Prediction interval = confidence interval + residual variance
    ci_half_width = (ci_upper - ci_lower) / 2
    pi_half_width = np.sqrt(ci_half_width**2 + (t_value * sigma_residuals) ** 2)

    pi_lower = y_fit - pi_half_width
    pi_upper = y_fit + pi_half_width

    return y_fit, pi_lower, pi_upper


def plot_nlsq_fit(
    result: NLSQResult,
    model,
    x_data,
    y_data,
    x_pred=None,
    confidence=0.95,
    ax=None,
    show_metrics: bool = True,
    show_prediction_interval: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
) -> plt.Axes:
    """Plot NLSQ fit with uncertainty band (FR-017, FR-021, NLSQ 0.6.0).

    If covariance is invalid, logs warning and displays
    "Uncertainty unavailable" in legend.

    Parameters
    ----------
    result : NLSQResult
        Output from nlsq_fit()
    model : callable
        Model function
    x_data, y_data : ndarray
        Original data
    x_pred : ndarray, optional
        X values for prediction curve
    confidence : float
        Confidence level for band (default: 0.95)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new if None)
    show_metrics : bool, optional
        Display R², RMSE, and χ² on the plot (default: True).
        Uses NLSQ 0.6.0 enhanced metrics from NLSQResult.
    show_prediction_interval : bool, optional
        Show prediction interval in addition to confidence interval.
        Prediction intervals account for observation noise (default: False).
    xlabel : str, optional
        X-axis label (default: "x")
    ylabel : str, optional
        Y-axis label (default: "y")
    title : str, optional
        Plot title (default: None)

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import logging

    import matplotlib.pyplot as plt
    import numpy as np

    logger = logging.getLogger(__name__)

    if ax is None:
        fig, ax = plt.subplots()

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if x_pred is None:
        x_pred = np.linspace(x_data.min(), x_data.max(), 200)

    # Plot data
    ax.scatter(x_data, y_data, c="k", s=20, alpha=0.7, label="Data", zorder=3)

    # Get parameters as array
    popt = np.array(list(result.params.values()))

    # Compute fit curve
    y_fit = model(x_pred, *popt)

    # Check covariance validity
    if result.pcov_valid:
        try:
            # Prediction interval (wider, includes observation noise)
            if show_prediction_interval:
                _, pi_lower, pi_upper = compute_prediction_interval(
                    model, x_pred, popt, result.covariance, result.residuals, confidence
                )
                ax.fill_between(
                    x_pred,
                    pi_lower,
                    pi_upper,
                    alpha=0.15,
                    color="C0",
                    label=f"{int(confidence * 100)}% PI",
                )

            # Confidence interval (parameter uncertainty only)
            _, y_lower, y_upper = compute_uncertainty_band(
                model, x_pred, popt, result.covariance, confidence
            )
            ax.fill_between(
                x_pred,
                y_lower,
                y_upper,
                alpha=0.3,
                color="C0",
                label=f"{int(confidence * 100)}% CI",
            )
        except Exception as e:
            logger.warning(f"Failed to compute uncertainty band: {e}")
            ax.plot(x_pred, y_fit, "C0-", lw=2, label="Fit (uncertainty unavailable)")
    else:
        logger.warning(f"Covariance invalid: {result.pcov_message}")
        ax.plot(x_pred, y_fit, "C0-", lw=2, label="Fit (uncertainty unavailable)")

    # Plot fit curve on top
    if result.pcov_valid:
        ax.plot(x_pred, y_fit, "C0-", lw=2, label="Fit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    # Add metrics annotation (NLSQ 0.6.0 enhanced)
    if show_metrics:
        metrics_text = (
            f"R² = {result.r_squared:.4f}\n"
            f"RMSE = {result.rmse:.2e}\n"
            f"χ²ᵣ = {result.chi_squared:.3f}"
        )
        # Position in upper right corner with some padding
        ax.text(
            0.97,
            0.97,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    ax.legend(loc="best")

    return ax


def plot_posterior_predictive(
    result: FitResult,
    model,
    x_data,
    y_data,
    x_pred=None,
    credible_level=0.95,
    n_draws=100,
    ax=None,
) -> plt.Axes:
    """Plot Bayesian fit with posterior credible interval (FR-014).

    Parameters
    ----------
    result : FitResult
        Output from Bayesian fitting
    model : callable
        Model function
    x_data, y_data : ndarray
        Original data
    x_pred : ndarray, optional
        X values for prediction (default: smooth range over x_data)
    credible_level : float
        Credible interval level (default: 0.95)
    n_draws : int
        Number of posterior samples for band calculation
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if x_pred is None:
        x_pred = np.linspace(x_data.min(), x_data.max(), 200)

    # Plot data
    ax.scatter(x_data, y_data, c="k", s=20, alpha=0.7, label="Data", zorder=3)

    # Generate posterior predictive samples
    param_names = list(result.samples.keys())
    n_samples = len(result.samples[param_names[0]])
    indices = np.random.choice(n_samples, min(n_draws, n_samples), replace=False)

    predictions = []
    for idx in indices:
        params = [result.samples[name][idx] for name in param_names]
        predictions.append(model(x_pred, *params))

    predictions = np.array(predictions)

    # Compute credible interval
    alpha = 1 - credible_level
    lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
    upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)
    median = np.median(predictions, axis=0)

    # Plot credible interval
    ax.fill_between(
        x_pred,
        lower,
        upper,
        alpha=0.3,
        color="C1",
        label=f"{int(credible_level * 100)}% CI",
    )
    ax.plot(x_pred, median, "C1-", lw=2, label="Median fit")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    return ax


def generate_arviz_diagnostics(
    trace,
    var_names=None,
    output_dir=None,
    formats=("pdf", "png"),
    dpi=300,
    prefix="mcmc",
) -> dict:
    """Generate complete ArviZ diagnostic suite (FR-013).

    Plots generated:
    1. Pair plot (parameter correlations, divergences)
    2. Forest plot (HDI intervals)
    3. Energy plot (NUTS E-BFMI diagnostics)
    4. Autocorrelation plot (chain mixing)
    5. Rank plot (convergence check)
    6. ESS plot (effective sample size)

    Parameters
    ----------
    trace : az.InferenceData
        ArviZ InferenceData object from MCMC
    var_names : list, optional
        Parameter names to plot (default: all)
    output_dir : str or Path, optional
        Directory for output files (None = return figures only)
    formats : tuple
        Output formats (default: ("pdf", "png") per FR-018)
    dpi : int
        Resolution for raster formats (default: 300 per FR-018)
    prefix : str
        Filename prefix

    Returns
    -------
    dict
        Mapping plot_type → figure (if output_dir is None)
        Mapping plot_type_format → file path (if output_dir provided)
    """
    import arviz as az

    if trace is None:
        return {}

    results = {}

    # Define plot functions
    plots = [
        (
            "pair",
            lambda: az.plot_pair(
                trace, var_names=var_names, marginals=True, divergences=True
            ),
        ),
        (
            "forest",
            lambda: az.plot_forest(
                trace, var_names=var_names, combined=True, hdi_prob=0.95
            ),
        ),
        ("energy", lambda: az.plot_energy(trace)),
        ("autocorr", lambda: az.plot_autocorr(trace, var_names=var_names)),
        ("rank", lambda: az.plot_rank(trace, var_names=var_names)),
        ("ess", lambda: az.plot_ess(trace, var_names=var_names, kind="local")),
    ]

    for plot_name, plot_func in plots:
        try:
            axes = plot_func()

            # Get figure from axes
            if hasattr(axes, "figure"):
                fig = axes.figure
            elif hasattr(axes, "__iter__"):
                # Array of axes
                import numpy as np

                axes_flat = np.asarray(axes).flatten()
                fig = axes_flat[0].figure if len(axes_flat) > 0 else None
            else:
                fig = None

            if fig is not None:
                if output_dir is not None:
                    from pathlib import Path

                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    for fmt in formats:
                        filepath = output_dir / f"{prefix}_{plot_name}.{fmt}"
                        fig.savefig(filepath, dpi=dpi if fmt != "pdf" else None)
                        results[f"{plot_name}_{fmt}"] = str(filepath)
                else:
                    results[plot_name] = fig
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Failed to generate {plot_name} plot: {e}"
            )

    return results


def plot_comparison(
    nlsq_result: NLSQResult,
    bayesian_result: FitResult,
    model,
    x_data,
    y_data,
    x_pred=None,
    confidence_level=0.95,
    band_alpha=0.25,
    ax=None,
) -> plt.Axes:
    """Overlay NLSQ confidence band and Bayesian credible interval (FR-020).

    Parameters
    ----------
    nlsq_result : NLSQResult
        NLSQ fitting result
    bayesian_result : FitResult
        Bayesian fitting result
    model : callable
        Model function
    x_data, y_data : ndarray
        Original data
    x_pred : ndarray, optional
        X values for prediction curves
    confidence_level : float
        Confidence/credible level (default: 0.95)
    band_alpha : float
        Transparency for bands (default: 0.25)
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    if x_pred is None:
        x_pred = np.linspace(x_data.min(), x_data.max(), 200)

    # Plot data
    ax.scatter(x_data, y_data, c="k", s=20, alpha=0.7, label="Data", zorder=5)

    # NLSQ fit
    popt = np.array(list(nlsq_result.params.values()))
    y_nlsq = model(x_pred, *popt)

    if nlsq_result.pcov_valid:
        _, y_nlsq_lower, y_nlsq_upper = compute_uncertainty_band(
            model, x_pred, popt, nlsq_result.covariance, confidence_level
        )
        ax.fill_between(
            x_pred,
            y_nlsq_lower,
            y_nlsq_upper,
            alpha=band_alpha,
            color="C0",
            label=f"NLSQ {int(confidence_level * 100)}% CI",
        )
    ax.plot(x_pred, y_nlsq, "C0-", lw=2, label="NLSQ fit")

    # Bayesian fit
    param_names = list(bayesian_result.samples.keys())
    n_samples = len(bayesian_result.samples[param_names[0]])
    indices = np.random.choice(n_samples, min(100, n_samples), replace=False)

    predictions = []
    for idx in indices:
        params = [bayesian_result.samples[name][idx] for name in param_names]
        predictions.append(model(x_pred, *params))

    predictions = np.array(predictions)
    alpha = 1 - confidence_level
    lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
    upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)
    median = np.median(predictions, axis=0)

    ax.fill_between(
        x_pred,
        lower,
        upper,
        alpha=band_alpha,
        color="C1",
        label=f"Bayesian {int(confidence_level * 100)}% CI",
    )
    ax.plot(x_pred, median, "C1--", lw=2, label="Bayesian median")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    return ax


def save_figure(fig, filepath, formats=("pdf", "png"), dpi=300) -> dict:
    """Save figure in multiple formats (FR-018).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str or Path
        Base path (extension will be replaced)
    formats : tuple
        Output formats (default: ("pdf", "png"))
    dpi : int
        Resolution for raster formats

    Returns
    -------
    dict
        Mapping format → saved file path
    """
    from pathlib import Path

    filepath = Path(filepath)
    results = {}

    for fmt in formats:
        output_path = filepath.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi if fmt != "pdf" else None)
        results[fmt] = str(output_path)

    return results
