"""Routing checks for XpcsViewer.update_tab_availability.

Exercises the format-presence branching (multitau-only / twotime-only / both)
without constructing the full Qt window: a stub `self` records which
_configure_* / _enable_all_tabs path the method takes.
"""

from types import SimpleNamespace

import pytest

from xpcsviewer.xpcs_viewer import XpcsViewer


class _Recorder:
    """Minimal stand-in for the parts of XpcsViewer the method touches."""

    def __init__(self, atypes):
        # atypes: list of per-file atype lists, e.g. [["Multitau", "Twotime"]]
        self._xf_list = [SimpleNamespace(atype=a) for a in atypes]
        self.vk = SimpleNamespace(
            target=list(self._xf_list),
            get_xf_list=lambda *a, **k: self._xf_list,
        )
        self.statusbar = SimpleNamespace(showMessage=lambda *a, **k: None)
        self.calls = []

    def _enable_all_tabs(self):
        self.calls.append("enable_all")

    def _configure_for_multitau(self):
        self.calls.append("multitau")

    def _configure_for_twotime(self):
        self.calls.append("twotime")


def _run(atypes):
    rec = _Recorder(atypes)
    XpcsViewer.update_tab_availability(rec)
    return rec.calls


@pytest.mark.unit
@pytest.mark.gui
def test_both_file_enables_all_tabs():
    # Single HDF5 file containing both multi-tau and two-time data.
    assert _run([["Multitau", "Twotime"]]) == ["enable_all"]


@pytest.mark.unit
@pytest.mark.gui
def test_mixed_single_type_files_enable_all_tabs():
    assert _run([["Multitau"], ["Twotime"]]) == ["enable_all"]


@pytest.mark.unit
@pytest.mark.gui
def test_multitau_only_disables_twotime():
    assert _run([["Multitau"]]) == ["multitau"]


@pytest.mark.unit
@pytest.mark.gui
def test_twotime_only_disables_g2_group():
    assert _run([["Twotime"]]) == ["twotime"]
