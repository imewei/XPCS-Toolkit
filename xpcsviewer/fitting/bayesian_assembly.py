"""Assemble Bayesian fitting results into NLSQ-compatible fit_summary format.

Converts a dict of per-Q FitResult objects into the same ``fit_summary``
dict that :meth:`XpcsFile.fit_g2` produces, enabling seamless integration
with the tau-q diffusion pipeline and GUI plotting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .results import FitResult

logger = logging.getLogger(__name__)

# Legacy parameter order for single_exp_all(x, a, b, c, d):
#   idx 0 = a = contrast
#   idx 1 = b = tau
#   idx 2 = c = stretching (fixed 1.0)
#   idx 3 = d = baseline
_SINGLE_EXP_NPARAMS = 4

# Legacy parameter order for double_exp_all(x, a, b, c, d, e, f, g):
#   idx 0 = contrast1
#   idx 1 = tau1
#   idx 2 = stretching1 (fixed 1.0)
#   idx 3 = baseline
#   idx 4 = tau2
#   idx 5 = contrast2
#   idx 6 = stretching2 (fixed 1.0)
_DOUBLE_EXP_NPARAMS = 7


def _extract_single_exp_params(
    fit_result: FitResult,
) -> tuple[NDArray, NDArray]:
    """Extract single-exp params as (values, errors) in legacy order."""
    vals = np.zeros(_SINGLE_EXP_NPARAMS)
    errs = np.zeros(_SINGLE_EXP_NPARAMS)

    vals[0] = fit_result.get_mean("contrast")
    vals[1] = fit_result.get_mean("tau")
    vals[2] = 1.0  # stretching exponent fixed
    vals[3] = fit_result.get_mean("baseline")

    errs[0] = fit_result.get_std("contrast")
    errs[1] = fit_result.get_std("tau")
    errs[2] = 0.0
    errs[3] = fit_result.get_std("baseline")

    return vals, errs


def _extract_double_exp_params(
    fit_result: FitResult,
) -> tuple[NDArray, NDArray]:
    """Extract double-exp params as (values, errors) in legacy order."""
    vals = np.zeros(_DOUBLE_EXP_NPARAMS)
    errs = np.zeros(_DOUBLE_EXP_NPARAMS)

    vals[0] = fit_result.get_mean("contrast1")
    vals[1] = fit_result.get_mean("tau1")
    vals[2] = 1.0  # stretching exponent fixed
    vals[3] = fit_result.get_mean("baseline")
    vals[4] = fit_result.get_mean("tau2")
    vals[5] = fit_result.get_mean("contrast2")
    vals[6] = 1.0  # stretching exponent fixed

    errs[0] = fit_result.get_std("contrast1")
    errs[1] = fit_result.get_std("tau1")
    errs[2] = 0.0
    errs[3] = fit_result.get_std("baseline")
    errs[4] = fit_result.get_std("tau2")
    errs[5] = fit_result.get_std("contrast2")
    errs[6] = 0.0

    return vals, errs


def _compute_fit_line(
    model_func: Any,
    fit_x: NDArray,
    fit_result: FitResult,
    fit_func_name: str,
) -> NDArray:
    """Compute fitted curve from Bayesian posterior means."""
    if fit_func_name == "single":
        return model_func(
            fit_x,
            fit_result.get_mean("tau"),
            fit_result.get_mean("baseline"),
            fit_result.get_mean("contrast"),
        )
    else:
        return model_func(
            fit_x,
            fit_result.get_mean("tau1"),
            fit_result.get_mean("tau2"),
            fit_result.get_mean("baseline"),
            fit_result.get_mean("contrast1"),
            fit_result.get_mean("contrast2"),
        )


def assemble_fit_summary(
    results: dict[int, FitResult | None],
    q_arr: NDArray,
    t_el: NDArray,
    fit_func_name: str,
    model_func: Any,
    *,
    q_range: str = "",
    t_range: str = "",
    bounds: Any = None,
    fit_flag: str = "",
    label: str = "",
) -> dict[str, Any]:
    """Convert per-Q Bayesian results into NLSQ-compatible fit_summary.

    Parameters
    ----------
    results : dict[int, FitResult | None]
        Mapping of Q-index to FitResult (None for failed Q-bins).
    q_arr : ndarray
        Array of Q values (1D).
    t_el : ndarray
        Array of delay times (1D).
    fit_func_name : str
        ``"single"`` or ``"double"``.
    model_func : callable
        Bayesian model function (``single_exp_func`` or ``double_exp_func``).
    q_range, t_range, bounds, fit_flag, label : optional
        Metadata fields matching the NLSQ fit_summary format.

    Returns
    -------
    dict
        fit_summary dict with keys: ``fit_func``, ``fit_val``, ``t_el``,
        ``q_val``, ``q_range``, ``t_range``, ``bounds``, ``fit_flag``,
        ``fit_line``, ``fit_x``, ``label``, ``failed_mask``.
        Failed Q-bins have NaN in ``fit_val``/``fit_line`` and
        ``failed_mask[q_idx] == True``.
    """
    num_q = len(q_arr)
    nparams = _SINGLE_EXP_NPARAMS if fit_func_name == "single" else _DOUBLE_EXP_NPARAMS
    extract_fn = (
        _extract_single_exp_params
        if fit_func_name == "single"
        else _extract_double_exp_params
    )

    # fit_val shape: [num_q, 2, nparams] — dim 1 is [value, error]
    # NaN default: failed Q-bins stay NaN so downstream isfinite() filters skip them
    fit_val = np.full((num_q, 2, nparams), np.nan)

    # Generate fit_x from t_el range (matching NLSQ behavior)
    # Cap at 500 points for visualization — keeps main-thread overhead low
    # even for datasets with thousands of time points and many Q-bins
    _FIT_LINE_POINTS = 500
    fit_x = np.linspace(t_el.min(), t_el.max(), min(_FIT_LINE_POINTS, max(200, len(t_el) * 2)))
    fit_line = np.full((num_q, len(fit_x)), np.nan)

    # Track which Q-bins failed (True = failed)
    failed_mask = np.ones(num_q, dtype=bool)

    succeeded = 0
    failed = 0

    for q_idx in range(num_q):
        fr = results.get(q_idx)
        if fr is None:
            failed += 1
            continue

        try:
            vals, errs = extract_fn(fr)
            fit_val[q_idx, 0, :] = vals
            fit_val[q_idx, 1, :] = errs
            fit_line[q_idx, :] = np.asarray(
                _compute_fit_line(model_func, fit_x, fr, fit_func_name)
            )
            failed_mask[q_idx] = False
            succeeded += 1
        except (KeyError, ValueError) as exc:
            logger.warning("Failed to extract params for Q-index %d: %s", q_idx, exc)
            failed += 1

    logger.info(
        "Bayesian assembly: %d/%d Q-bins succeeded, %d failed",
        succeeded,
        num_q,
        failed,
    )

    return {
        "source": "bayesian",
        "fit_func": fit_func_name,
        "fit_val": fit_val,
        "t_el": t_el,
        "q_val": np.asarray(q_arr),
        "q_range": q_range,
        "t_range": t_range,
        "bounds": bounds,
        "fit_flag": fit_flag,
        "fit_line": fit_line,
        "fit_x": fit_x,
        "label": label,
        "failed_mask": failed_mask,
    }
