"""Fitting functions for XPCS analysis.

This module provides fitting functions for G2 correlation analysis.
"""

from __future__ import annotations

import os
import re

import numpy as np


def single_exp_all(x, a, b, c, d):
    """
    Single exponential fitting for XPCS-multitau analysis.

    Parameters
    ----------
    x : float or ndarray
        Delay in seconds.
    a : float
        Contrast.
    b : float
        Characteristic time (tau).
    c : float
        Restriction factor.
    d : float
        Baseline offset.

    Returns
    -------
    float or ndarray
        Computed value of the single exponential model.
    """
    return a * np.exp(-2 * (x / b) ** c) + d


def double_exp_all(x, a, b1, c1, d, b2, c2, f):
    """
    Double exponential fitting for XPCS-multitau analysis.

    Parameters
    ----------
    x : float or ndarray
        Delay in seconds.
    a : float
        Contrast.
    b1 : float
        Characteristic time (tau) of the first exponential component.
    c1 : float
        Restriction factor for the first component.
    d : float
        Baseline offset.
    b2 : float
        Characteristic time (tau) of the second exponential component.
    c2 : float
        Restriction factor for the second component.
    f : float
        Fractional contribution of the first exponential component (0 ≤ f ≤ 1).

    Returns
    -------
    float or ndarray
        Computed value of the double exponential model.
    """
    t1 = np.exp(-1 * (x / b1) ** c1) * f
    t2 = np.exp(-1 * (x / b2) ** c2) * (1 - f)
    return a * (t1 + t2) ** 2 + d


def power_law(x, a, b):
    """
    Power-law fitting for diffusion behavior.

    Parameters
    ----------
    x : float or ndarray
        Independent variable, typically time delay (tau).
    a : float
        Scaling factor.
    b : float
        Power exponent.

    Returns
    -------
    float or ndarray
        Computed value based on the power-law model.
    """
    return a * x**b


def create_id(fname, label_style=None, simplify_flag=True):
    """
    Generate a simplified or customized ID string from a filename.

    Parameters
    ----------
    fname : str
        Input file name, possibly with path and extension.
    label_style : str or None, optional
        Optional custom regex pattern for extracting a label from the filename.
        If None, uses simplified default label extraction.
    simplify_flag : bool, optional
        If True (default), applies default simplification logic.

    Returns
    -------
    str
        A simplified or customized ID string derived from the input filename.
    """
    # Handle empty or None filename
    if not fname:
        return "unknown"

    # Get basename without path
    basename = os.path.basename(fname)

    # Remove common extensions
    name_no_ext = basename
    for ext in [".hdf5", ".hdf", ".h5", ".nxs"]:
        if name_no_ext.lower().endswith(ext):
            name_no_ext = name_no_ext[: -len(ext)]
            break

    # Apply custom label style if provided
    if label_style is not None:
        match = re.search(label_style, name_no_ext)
        if match:
            return match.group(0)

    # Default simplification: remove common prefixes and suffixes
    if simplify_flag:
        # Remove common prefixes
        for prefix in ["xpcs_", "result_", "analysis_"]:
            if name_no_ext.lower().startswith(prefix):
                name_no_ext = name_no_ext[len(prefix) :]
                break

        # Limit length
        if len(name_no_ext) > 50:
            name_no_ext = name_no_ext[:47] + "..."

    return name_no_ext
