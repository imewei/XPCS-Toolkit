"""End-to-end regression test for target-file session persistence.

Guards the bug where ``_collect_session_state`` persisted target *filenames*
(relative to ``vk.path``) into ``FileEntry.path`` — documented as an absolute
path. When the app was launched from a directory other than the data
directory, ``SessionManager.load_session`` / ``_restore_session`` resolved
those names against the process cwd, found nothing, and silently dropped the
restored target list.

This test drives the real ``XpcsViewer._collect_session_state`` and
``_restore_session`` methods (via a lightweight stub) against a real
``FileLocator`` and ``SessionManager``, with the working directory
deliberately different from the data directory.
"""

import os
from types import SimpleNamespace

import pytest

from xpcsviewer import file_locator as file_locator_mod
from xpcsviewer.file_locator import FileLocator
from xpcsviewer.gui.state.session_manager import SessionManager
from xpcsviewer.xpcs_viewer import XpcsViewer

pytestmark = [pytest.mark.integration, pytest.mark.gui]


@pytest.fixture
def loadable_files(monkeypatch):
    """Make ``add_target(preload=True)`` accept any existing file.

    Empty ``.h5`` files are not valid XPCS datasets, so stub the dataset
    factory to return a sentinel — the test cares about path bookkeeping,
    not HDF5 parsing.
    """
    monkeypatch.setattr(
        file_locator_mod, "create_xpcs_dataset", lambda *a, **k: object()
    )


def _make_tab_widget():
    return SimpleNamespace(
        currentIndex=lambda: 0,
        count=lambda: 12,
        blockSignals=lambda _b: None,
        setCurrentIndex=lambda _i: None,
    )


def _make_geometry():
    return SimpleNamespace(x=lambda: 100, y=lambda: 120, width=lambda: 1200, height=lambda: 800)


def test_target_files_survive_restart_from_other_cwd(
    tmp_path, monkeypatch, loadable_files
):
    # --- Arrange: data dir with real files, distinct from the working dir ---
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    names = ["alpha.h5", "beta.h5", "gamma.h5"]
    for n in names:
        (data_dir / n).touch()

    work_cwd = tmp_path / "elsewhere"
    work_cwd.mkdir()
    monkeypatch.chdir(work_cwd)  # cwd != data_dir — the condition that broke restore

    # Real session manager backed by a temp session file
    session_file = tmp_path / ".xpcsviewer" / "session.json"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "xpcsviewer.gui.state.session_manager.get_session_path",
        lambda: session_file,
    )
    session_manager = SessionManager()

    # Real kernel holding targets as relative names (production behavior)
    vk = FileLocator(str(data_dir))
    vk.add_target(list(names))
    assert list(vk.target) == names  # stored relative, not absolute

    # --- Act 1: collect + save via the real method ---
    saver = SimpleNamespace(
        vk=vk,
        target_model=vk.target,
        session_manager=session_manager,
        geometry=_make_geometry,
        isMaximized=lambda: False,
        tabWidget=_make_tab_widget(),
        work_dir=SimpleNamespace(text=lambda: str(data_dir)),
    )
    state = XpcsViewer._collect_session_state(saver)

    # Persisted paths must be absolute and actually exist (so load_session keeps them)
    assert [e.path for e in state.target_files] == [
        str(data_dir / n) for n in names
    ]
    assert all(os.path.isabs(e.path) and os.path.isfile(e.path) for e in state.target_files)

    session_manager.save_session(state)

    # --- Act 2: restore into a FRESH kernel via the real method ---
    fresh_vk = FileLocator(str(data_dir))

    def fake_load_path(p):
        restorer.vk = FileLocator(p)

    restorer = SimpleNamespace(
        vk=fresh_vk,
        session_manager=session_manager,
        toast_manager=SimpleNamespace(show_warning=lambda _w: None),
        tabWidget=_make_tab_widget(),
        setGeometry=lambda *a: None,
        load_path=fake_load_path,
    )
    XpcsViewer._restore_session(restorer)

    # --- Assert: all targets restored, none dropped, order preserved ---
    assert list(restorer.vk.target) == names

    # And no "File not found" warnings were emitted by the loader
    assert session_manager.get_warnings() == []
