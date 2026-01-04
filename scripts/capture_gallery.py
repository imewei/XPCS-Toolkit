#!/usr/bin/env python3
"""Capture gallery screenshots offscreen for docs/images.

Uses the existing XpcsViewer, switches tabs, and saves window grabs for the
gallery images to reflect the current toolbar-less, maximized UI. Runs in
offscreen mode with no data loaded (plots will be empty but layout/header are
accurate). Adjust paths or delays as needed.
"""

import os
import sys
import time
from pathlib import Path

from PySide6 import QtWidgets

from xpcsviewer.xpcs_viewer import XpcsViewer


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")

    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer = XpcsViewer(path=None)
    viewer.resize(1280, 800)
    viewer.show()
    app.processEvents()
    time.sleep(0.3)

    # Map desired filenames to tab indices (matching docs gallery ordering)
    target_tabs = {
        "saxs2d": 0,
        "saxs1d": 1,
        "stability": 2,
        "intt": 3,
        "g2mod": 4,
        "diffusion": 5,
        "twotime": 6,
        # average tab index is 8 (7 is qmap)
        "average": 8,
        # metadata view (hdf_info)
        "hdf_info": 9,
    }

    for name, idx in target_tabs.items():
        viewer.tabWidget.setCurrentIndex(idx)
        app.processEvents()
        time.sleep(0.2)
        pix = viewer.grab()
        out_path = out_dir / f"{name}.png"
        pix.save(str(out_path), "PNG")
        print(f"Saved {out_path}")

    viewer.close()
    app.quit()


if __name__ == "__main__":
    sys.exit(main())
