"""Offscreen screenshot harness for XPCS Toolkit GUI.

Runs the Qt app headlessly (QT_QPA_PLATFORM=offscreen), attempts to load
sample data if available, cycles through tabs, and saves PNGs for review.

Outputs are written to ``screenshots/offscreen`` relative to repo root.
"""

from __future__ import annotations

import os
import pathlib
import sys

from PySide6 import QtCore, QtWidgets

from xpcsviewer.xpcs_viewer import XpcsViewer


def ensure_offscreen() -> None:
    """Force Qt into offscreen mode if not already set."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def find_sample_dir() -> pathlib.Path | None:
    """Return a plausible sample data directory, if one exists."""
    candidates = [
        pathlib.Path("tests/fixtures/reference_data"),
        pathlib.Path("tests/fixtures"),
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def process(app: QtWidgets.QApplication, ms: int = 50) -> None:
    """Pump the event loop briefly."""
    app.processEvents(QtCore.QEventLoop.AllEvents, ms)


def snap(viewer: XpcsViewer, name: str, outdir: pathlib.Path) -> None:
    """Capture a screenshot of the main window."""
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{name}.png"
    pixmap = viewer.grab()
    pixmap.save(str(png_path))
    print(f"Saved {png_path}")


def main() -> int:
    ensure_offscreen()

    app = QtWidgets.QApplication(sys.argv)
    viewer = XpcsViewer()
    viewer.show()
    process(app)

    outdir = pathlib.Path("screenshots/offscreen")
    snap(viewer, "00_initial", outdir)

    sample_dir = find_sample_dir()
    load_error: str | None = None
    if sample_dir:
        try:
            viewer.load_path(str(sample_dir))
            process(app, 200)
        except Exception as exc:
            load_error = f"load_path failed: {type(exc).__name__}: {exc}"
            print(load_error)
    else:
        load_error = "No sample directory found"
        print(load_error)

    # Cycle through tabs and capture each
    tab_widget = viewer.tabWidget
    count = tab_widget.count()
    for idx in range(count):
        tab_widget.setCurrentIndex(idx)
        process(app, 150)
        tab_name = tab_widget.tabText(idx) or f"tab{idx}"
        snap(viewer, f"{idx:02d}_{tab_name.replace(' ', '_')}", outdir)

    # Record load status
    status_path = outdir / "status.txt"
    status_path.write_text(load_error or "load_path succeeded\n", encoding="utf-8")
    print(f"Wrote {status_path}")

    # Allow queued events to flush before exit
    process(app, 100)
    viewer.close()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
