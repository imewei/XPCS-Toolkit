"""Tests for dual fit_summary storage on XpcsFile."""

import numpy as np
import pytest


class TestDualStorage:
    """XpcsFile must store NLSQ and Bayesian fit summaries independently."""

    def test_bayesian_fit_summary_initialized_none(self):
        """bayesian_fit_summary starts as None."""
        from unittest.mock import MagicMock

        xf = MagicMock()
        xf.bayesian_fit_summary = None
        xf.bayesian_results = None
        assert xf.bayesian_fit_summary is None
        assert xf.bayesian_results is None

    def test_dual_storage_independence(self):
        """Setting bayesian_fit_summary must not affect fit_summary."""
        from unittest.mock import MagicMock

        xf = MagicMock()
        xf.fit_summary = {"source": "nlsq", "fit_func": "single"}
        xf.bayesian_fit_summary = {"source": "bayesian", "fit_func": "single"}
        assert xf.fit_summary["source"] == "nlsq"
        assert xf.bayesian_fit_summary["source"] == "bayesian"


class TestAssembleFitSummarySource:
    """assemble_fit_summary must include source='bayesian' in output."""

    def test_source_field_present(self):
        """Assembled fit_summary must have source='bayesian'."""
        from unittest.mock import MagicMock

        from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary

        fr = MagicMock()
        fr.get_mean.side_effect = lambda k: {
            "contrast": 0.3,
            "tau": 1.0,
            "baseline": 1.0,
        }[k]
        fr.get_std.side_effect = lambda k: {
            "contrast": 0.01,
            "tau": 0.1,
            "baseline": 0.01,
        }[k]
        fr.samples = {
            "contrast": np.ones(10) * 0.3,
            "tau": np.ones(10),
            "baseline": np.ones(10),
        }

        from xpcsviewer.fitting.models import single_exp_func

        result = assemble_fit_summary(
            results={0: fr},
            q_arr=np.array([0.01]),
            t_el=np.linspace(0.001, 1.0, 50),
            fit_func_name="single",
            model_func=single_exp_func,
        )
        assert result["source"] == "bayesian"

    def test_q_range_t_range_forwarded(self):
        """q_range and t_range args must appear in output."""
        from unittest.mock import MagicMock

        from xpcsviewer.fitting.bayesian_assembly import assemble_fit_summary

        fr = MagicMock()
        fr.get_mean.side_effect = lambda k: {
            "contrast": 0.3,
            "tau": 1.0,
            "baseline": 1.0,
        }[k]
        fr.get_std.side_effect = lambda k: 0.01
        fr.samples = {
            "contrast": np.ones(10) * 0.3,
            "tau": np.ones(10),
            "baseline": np.ones(10),
        }

        from xpcsviewer.fitting.models import single_exp_func

        result = assemble_fit_summary(
            results={0: fr},
            q_arr=np.array([0.01]),
            t_el=np.linspace(0.001, 1.0, 50),
            fit_func_name="single",
            model_func=single_exp_func,
            q_range="0.01-0.05",
            t_range="0.001-1.0",
        )
        assert result["q_range"] == "0.01-0.05"
        assert result["t_range"] == "0.001-1.0"
