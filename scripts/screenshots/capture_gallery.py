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
    viewer._set_theme("dark")
    app.processEvents()
    time.sleep(0.3)

    # Map desired filenames to tab indices (matching current 12-tab layout)
    # Tab order: SAXS_2D(0), SAXS_1D(1), Stability(2), I(t)(3),
    #   g2(4), g2_Fit(5), g2_Map(6), Diffusion(7),
    #   Two-Time(8), Q-Map(9), Average(10), Metadata(11)
    target_tabs = {
        "saxs2d": 0,
        "saxs1d": 1,
        "stability": 2,
        "intt": 3,
        "g2mod": 4,
        "g2_fit": 5,
        "g2_map": 6,
        "diffusion": 7,
        "twotime": 8,
        "average": 10,
        "hdf_info": 11,
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
