"""Unit tests for TG3 — GUI Critical Fixes: Mask Export and Plot Data.

Tests for BUG-001 (import_mask / import_partition not applied) and
BUG-002 (check_g2_number except block missing val assignment).
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from PySide6 import QtWidgets

# ---------------------------------------------------------------------------
# Helpers / minimal stubs
# ---------------------------------------------------------------------------


def _make_xf(saxs_2d_shape=(512, 512)):
    """Return a minimal XpcsFile-like mock."""
    xf = MagicMock()
    xf.saxs_2d = np.zeros(saxs_2d_shape)
    xf.mask = None
    xf.dqmap = None
    xf.sqmap = None
    return xf


def _make_viewer(xf_list):
    """Return a minimal XpcsViewer-like mock with the methods under test.

    We import the real methods and bind them to the mock so we test the
    actual implementation without constructing the full Qt GUI.
    """
    from xpcsviewer.xpcs_viewer import (  # noqa: PLC0415 (intentional late import)
        XpcsViewer,
    )

    viewer = MagicMock(spec=XpcsViewer)

    # Wire up the vk (viewer kernel) mock
    viewer.vk = MagicMock()
    viewer.vk.target = xf_list  # non-empty → guard passes
    viewer.vk.get_xf_list.return_value = xf_list
    viewer.statusbar = MagicMock()

    # Bind real methods so the code runs, not the mock
    viewer.import_mask = XpcsViewer.import_mask.__get__(viewer, XpcsViewer)
    viewer.import_partition = XpcsViewer.import_partition.__get__(viewer, XpcsViewer)
    viewer.check_g2_number = XpcsViewer.check_g2_number.__get__(viewer, XpcsViewer)

    return viewer


# ---------------------------------------------------------------------------
# Test 1 – import_mask iterates xf_list and sets xf.mask on every XpcsFile
# ---------------------------------------------------------------------------


def test_import_mask_sets_mask_on_all_xf_and_calls_update_plot():
    """import_mask must set xf.mask on every loaded XpcsFile and call update_plot."""
    mask = np.ones((512, 512), dtype=np.int32)
    xf1 = _make_xf()
    xf2 = _make_xf()
    xf_list = [xf1, xf2]

    viewer = _make_viewer(xf_list)

    viewer.import_mask(mask)

    # Each XpcsFile must have its mask attribute set to the exported mask
    assert xf1.mask is mask, "xf1.mask was not set to the exported mask"
    assert xf2.mask is mask, "xf2.mask was not set to the exported mask"

    # update_plot must be called exactly once
    viewer.update_plot.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 – import_partition iterates xf_list and sets partition attrs on all
# ---------------------------------------------------------------------------


def test_import_partition_sets_attrs_on_all_xf_and_calls_update_plot():
    """import_partition must propagate partition data to every XpcsFile and call update_plot."""
    dqmap = np.ones((512, 512), dtype=np.int32) * 2
    sqmap = np.ones((512, 512), dtype=np.int32) * 3
    partition = {
        "dynamic_roi_map": dqmap,
        "static_roi_map": sqmap,
    }

    xf1 = _make_xf()
    xf2 = _make_xf()
    xf_list = [xf1, xf2]

    viewer = _make_viewer(xf_list)

    viewer.import_partition(partition)

    # Each XpcsFile must have dqmap / sqmap set
    assert xf1.dqmap is dqmap, "xf1.dqmap was not set to dynamic_roi_map"
    assert xf1.sqmap is sqmap, "xf1.sqmap was not set to static_roi_map"
    assert xf2.dqmap is dqmap, "xf2.dqmap was not set to dynamic_roi_map"
    assert xf2.sqmap is sqmap, "xf2.sqmap was not set to static_roi_map"

    # update_plot must be called exactly once
    viewer.update_plot.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3 – check_g2_number never returns [None, None] on bad input
# ---------------------------------------------------------------------------


def test_check_g2_number_except_block_assigns_val():
    """check_g2_number must assign val = default_val[n] in the except block.

    When a QLineEdit contains non-numeric text, the except block resets
    the text field but must also assign the default to val so that the
    return list is never [None, None].
    """
    from xpcsviewer.xpcs_viewer import XpcsViewer  # noqa: PLC0415

    viewer = MagicMock(spec=XpcsViewer)
    viewer.statusbar = MagicMock()

    default_val = (0, 0.0092, 1e-8, 1, 0.95, 1.35)

    # Build 6 QLineEdit widgets, some with invalid (non-numeric) text
    line_edits = []
    invalid_indices = {0, 2, 5}  # these will trigger the except branch
    for i in range(6):
        le = MagicMock(spec=QtWidgets.QLineEdit)
        le.text.return_value = (
            "bad_value" if i in invalid_indices else str(default_val[i])
        )
        line_edits.append(le)

    # Assign as instance attributes matching the method's keys tuple
    (
        viewer.g2_qmin,
        viewer.g2_qmax,
        viewer.g2_tmin,
        viewer.g2_tmax,
        viewer.g2_ymin,
        viewer.g2_ymax,
    ) = line_edits

    viewer.check_g2_number = XpcsViewer.check_g2_number.__get__(viewer, XpcsViewer)

    result = viewer.check_g2_number(default_val=default_val)

    # Result must have exactly 6 elements, none of which is None
    assert len(result) == 6, f"Expected 6 values, got {len(result)}"
    for i, v in enumerate(result):
        assert v is not None, (
            f"check_g2_number returned None at index {i}; "
            "the except block must assign val = default_val[n]"
        )
        assert isinstance(v, (int, float)), (
            f"Expected numeric value at index {i}, got {type(v)}"
        )


# ---------------------------------------------------------------------------
# Test 4 – Integration: mask export followed by G2 plot uses new mask values
# ---------------------------------------------------------------------------


def test_mask_export_then_g2_plot_uses_new_mask():
    """Exported mask must be visible on all XpcsFiles before update_plot is called.

    Verifies that the mask assignment to xf_list happens before update_plot
    is triggered, so that any G2 plot refresh will see the correct mask.
    """
    mask = np.ones((256, 256), dtype=np.int32)
    xf1 = _make_xf(saxs_2d_shape=(256, 256))
    xf2 = _make_xf(saxs_2d_shape=(256, 256))
    xf_list = [xf1, xf2]

    viewer = _make_viewer(xf_list)

    # Capture the mask state at the time update_plot is called
    masks_at_update = []

    def capture_masks():
        masks_at_update.append((xf1.mask, xf2.mask))

    viewer.update_plot.side_effect = capture_masks

    viewer.import_mask(mask)

    # update_plot must have been called
    assert viewer.update_plot.called, "update_plot was never called after import_mask"

    # The mask must have been assigned before update_plot fired
    assert len(masks_at_update) == 1, "update_plot was not called exactly once"
    m1, m2 = masks_at_update[0]
    assert m1 is mask, "xf1.mask was not set before update_plot was called"
    assert m2 is mask, "xf2.mask was not set before update_plot was called"
