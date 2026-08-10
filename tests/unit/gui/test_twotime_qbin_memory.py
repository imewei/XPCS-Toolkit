"""Unit tests for per-file twotime qbin-index memory (XpcsViewer)."""

from collections import OrderedDict
from types import SimpleNamespace

from xpcsviewer.xpcs_viewer import XpcsViewer


class _FakeXf:
    def __init__(self, fname):
        self.fname = fname


class _FakeVk:
    def __init__(self, xf_list):
        self._xf_list = xf_list

    def get_xf_list(self, rows, filter_atype=None):
        return self._xf_list

    def get_module(self, name):
        return SimpleNamespace(plot_twotime_g2=lambda hdls, c2_result: None)


class _FakeCombo:
    def __init__(self):
        self._items = []
        self._index = -1

    def blockSignals(self, flag):
        pass

    def clear(self):
        self._items = []
        self._index = -1

    def addItems(self, items):
        self._items.extend(items)

    def setCurrentIndex(self, idx):
        self._index = idx

    def currentIndex(self):
        return self._index

    def currentText(self):
        return self._items[self._index] if 0 <= self._index < len(self._items) else ""

    def count(self):
        return len(self._items)


class _FakeSlider:
    def blockSignals(self, flag):
        pass

    def setMaximum(self, value):
        pass

    def setValue(self, value):
        pass


def _make_stub(xf_list):
    return SimpleNamespace(
        vk=_FakeVk(xf_list),
        get_selected_rows=lambda: [0],
    )


def test_current_twotime_fname_returns_first_file():
    stub = _make_stub([_FakeXf("a.hdf")])
    assert XpcsViewer._current_twotime_fname(stub) == "a.hdf"


def test_current_twotime_fname_none_when_no_files():
    stub = _make_stub([])
    assert XpcsViewer._current_twotime_fname(stub) is None


def test_current_twotime_fname_none_when_no_kernel():
    stub = SimpleNamespace(vk=None, get_selected_rows=lambda: [0])
    assert XpcsViewer._current_twotime_fname(stub) is None


def _make_viewer_stub(current_fname):
    """Stub exercising the real XpcsViewer.apply_twotime_result /
    on_twotime_q_selection_changed methods (unbound, called with the stub
    as `self`), not a reimplementation of their logic."""
    stub = SimpleNamespace(
        vk=_FakeVk([_FakeXf(current_fname)]),
        mp_2t_hdls=None,
        comboBox_twotime_selection=_FakeCombo(),
        horizontalSlider_twotime_selection=_FakeSlider(),
        tabWidget=SimpleNamespace(currentIndex=lambda: 0),
        get_selected_rows=lambda: [0],
        update_plot=lambda: None,
        _twotime_qbin_memory=OrderedDict(),
        _twotime_last_fname=None,
    )
    # on_twotime_q_selection_changed calls self._current_twotime_fname();
    # bind the real implementation so the stub exercises real behavior.
    stub._current_twotime_fname = lambda: XpcsViewer._current_twotime_fname(stub)
    return stub


def test_qbin_memory_restores_index_across_dataset_switch():
    """a.hdf(idx=5) -> b.hdf -> a.hdf must restore idx=5, not carry over b's
    index. Drives the real apply_twotime_result/on_twotime_q_selection_changed
    pipeline in xpcs_viewer.py rather than a standalone reimplementation."""
    stub = _make_viewer_stub("a.hdf")

    # Load a.hdf with 6 qbins; nothing remembered yet -> selection 0.
    XpcsViewer.apply_twotime_result(
        stub,
        {
            "c2_result": None,
            "new_qbin_labels": [f"q{i}" for i in range(6)],
            "xfile": _FakeXf("a.hdf"),
        },
    )
    assert stub.comboBox_twotime_selection.currentIndex() == 0

    # User selects qbin 5 on a.hdf.
    stub.vk = _FakeVk([_FakeXf("a.hdf")])
    XpcsViewer.on_twotime_q_selection_changed(stub, 5)
    assert stub._twotime_qbin_memory["a.hdf"] == 5

    # Switch to b.hdf (3 qbins): must not inherit a.hdf's stale index (5).
    XpcsViewer.apply_twotime_result(
        stub,
        {
            "c2_result": None,
            "new_qbin_labels": [f"q{i}" for i in range(3)],
            "xfile": _FakeXf("b.hdf"),
        },
    )
    assert stub.comboBox_twotime_selection.currentIndex() == 0

    stub.vk = _FakeVk([_FakeXf("b.hdf")])
    XpcsViewer.on_twotime_q_selection_changed(stub, 2)
    assert stub._twotime_qbin_memory["b.hdf"] == 2

    # Switch back to a.hdf: must restore its remembered qbin (5), not b.hdf's.
    XpcsViewer.apply_twotime_result(
        stub,
        {
            "c2_result": None,
            "new_qbin_labels": [f"q{i}" for i in range(6)],
            "xfile": _FakeXf("a.hdf"),
        },
    )
    assert stub.comboBox_twotime_selection.currentIndex() == 5


def test_qbin_memory_evicts_oldest_beyond_cap():
    """The per-file qbin memory must not grow without bound across a long
    session that touches many files."""
    from xpcsviewer.xpcs_viewer import _TWOTIME_QBIN_MEMORY_MAX

    stub = _make_viewer_stub("seed.hdf")
    for i in range(_TWOTIME_QBIN_MEMORY_MAX + 10):
        fname = f"file{i}.hdf"
        stub.vk = _FakeVk([_FakeXf(fname)])
        XpcsViewer.on_twotime_q_selection_changed(stub, 0)

    assert len(stub._twotime_qbin_memory) == _TWOTIME_QBIN_MEMORY_MAX
    # Oldest entries evicted first.
    assert "file0.hdf" not in stub._twotime_qbin_memory
    assert "file9.hdf" not in stub._twotime_qbin_memory
    last_fname = f"file{_TWOTIME_QBIN_MEMORY_MAX + 9}.hdf"
    assert last_fname in stub._twotime_qbin_memory


if __name__ == "__main__":
    test_current_twotime_fname_returns_first_file()
    test_current_twotime_fname_none_when_no_files()
    test_current_twotime_fname_none_when_no_kernel()
    test_qbin_memory_restores_index_across_dataset_switch()
    test_qbin_memory_evicts_oldest_beyond_cap()
    print("ok")
