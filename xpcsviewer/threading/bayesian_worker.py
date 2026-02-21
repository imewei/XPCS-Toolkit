"""Background worker for Bayesian (NumPyro NUTS) fitting.

Runs NLSQ warm-start → NUTS sampling off the main thread so the GUI
remains responsive during long MCMC runs.
"""

from __future__ import annotations

import logging
from typing import Any

from .async_workers import BaseAsyncWorker

logger = logging.getLogger(__name__)


class BayesianFitWorker(BaseAsyncWorker):
    """Run a Bayesian fit function in a background thread.

    Parameters
    ----------
    fit_func : callable
        One of ``fit_single_exp``, ``fit_double_exp``, ``fit_power_law``.
    x, y : array-like
        Independent and dependent data.
    yerr : array-like or None
        Measurement uncertainties (used for G2; ignored for power-law).
    q_index : int
        Q-bin index (for G2) or -1 for diffusion.
    q_value : float
        Q value in inverse Angstroms.
    context : str
        ``"g2"`` or ``"diffusion"`` — used to route callbacks.
    sampler_kwargs : dict
        Extra keyword arguments forwarded to the fit function.
    """

    def __init__(
        self,
        fit_func: Any,
        x: Any,
        y: Any,
        yerr: Any | None = None,
        q_index: int = 0,
        q_value: float = 0.0,
        context: str = "g2",
        sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(worker_id=f"bayesian_{context}_{q_index}")
        self.fit_func = fit_func
        self.x = x
        self.y = y
        self.yerr = yerr
        self.q_index = q_index
        self.q_value = q_value
        self.context = context
        self.sampler_kwargs = sampler_kwargs or {}

    def do_work(self) -> dict[str, Any]:
        """Execute the Bayesian fit and return a result dict."""
        logger.info(
            "BayesianFitWorker started: context=%s, q_index=%d, q_value=%.4g",
            self.context,
            self.q_index,
            self.q_value,
        )
        self.emit_status(f"Running Bayesian fit (Q={self.q_value:.4g})...")

        if self.context == "diffusion":
            # fit_power_law(q, tau, tau_err=..., **kwargs)
            fit_result = self.fit_func(
                self.x, self.y, tau_err=self.yerr, **self.sampler_kwargs
            )
        else:
            # fit_single_exp / fit_double_exp: (x, y, yerr, **kwargs)
            fit_result = self.fit_func(self.x, self.y, self.yerr, **self.sampler_kwargs)

        self.check_cancelled()

        return {
            "fit_result": fit_result,
            "q_index": self.q_index,
            "q_value": self.q_value,
            "context": self.context,
        }
