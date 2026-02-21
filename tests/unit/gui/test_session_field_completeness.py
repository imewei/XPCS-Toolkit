"""Test that _collect_session_state covers all AnalysisParameters fields.

Uses AST introspection to ensure new fields added to AnalysisParameters
are also mapped in _collect_session_state, preventing silent data loss
on session save/restore.
"""

import ast
import inspect
from pathlib import Path

import pytest

from xpcsviewer.gui.state.session_manager import AnalysisParameters

# Fields that intentionally have NO widget mapping and rely on dataclass
# defaults.  When a widget is added for one of these, remove it from the
# set and add a mapping in _collect_session_state.
_NO_WIDGET_FIELDS = frozenset(
    {
        "intensity_t_normalize",
        "diffusion_model",
        "twotime_symmetric",
        "qmap_show_rings",
    }
)


def _get_analysis_param_fields() -> set[str]:
    """Return all field names from AnalysisParameters dataclass."""
    import dataclasses

    return {f.name for f in dataclasses.fields(AnalysisParameters)}


def _get_collect_state_keywords() -> set[str]:
    """Parse _collect_session_state and return keyword arg names
    used in the AnalysisParameters(...) constructor call."""
    # Find the source file for XpcsViewer
    from xpcsviewer import xpcs_viewer as mod

    source_path = Path(inspect.getfile(mod))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    # Walk the AST looking for 'AnalysisParameters(...)' calls
    keywords: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Check if the call is to AnalysisParameters
        func = node.func
        if isinstance(func, ast.Name) and func.id == "AnalysisParameters":
            for kw in node.keywords:
                if kw.arg is not None:
                    keywords.add(kw.arg)
        elif isinstance(func, ast.Attribute) and func.attr == "AnalysisParameters":
            for kw in node.keywords:
                if kw.arg is not None:
                    keywords.add(kw.arg)

    return keywords


class TestSessionFieldCompleteness:
    """Ensure _collect_session_state maps all AnalysisParameters fields."""

    def test_all_fields_mapped_or_documented(self):
        """Every AnalysisParameters field must appear as a keyword arg in
        _collect_session_state OR be listed in _NO_WIDGET_FIELDS."""
        all_fields = _get_analysis_param_fields()
        mapped_fields = _get_collect_state_keywords()

        covered = mapped_fields | _NO_WIDGET_FIELDS
        missing = all_fields - covered

        assert not missing, (
            f"AnalysisParameters fields not mapped in _collect_session_state "
            f"and not in _NO_WIDGET_FIELDS: {sorted(missing)}. "
            f"Either add a widget mapping or add to _NO_WIDGET_FIELDS."
        )

    def test_no_stale_no_widget_entries(self):
        """_NO_WIDGET_FIELDS should not contain names that don't exist
        in AnalysisParameters (guards against stale entries)."""
        all_fields = _get_analysis_param_fields()
        stale = _NO_WIDGET_FIELDS - all_fields

        assert not stale, (
            f"_NO_WIDGET_FIELDS contains names not in AnalysisParameters: "
            f"{sorted(stale)}. Remove them."
        )

    def test_no_widget_fields_not_redundantly_mapped(self):
        """Fields in _NO_WIDGET_FIELDS should NOT also be mapped as keyword
        args (that would mean the field now has a widget and should be
        removed from _NO_WIDGET_FIELDS)."""
        mapped_fields = _get_collect_state_keywords()
        overlap = _NO_WIDGET_FIELDS & mapped_fields

        assert not overlap, (
            f"Fields in _NO_WIDGET_FIELDS are also mapped as kwargs: "
            f"{sorted(overlap)}. Remove them from _NO_WIDGET_FIELDS."
        )
