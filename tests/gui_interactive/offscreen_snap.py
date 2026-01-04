#!/usr/bin/env python3
"""Offscreen snapshot harness for CI-friendly viewer captures.

Launches the viewer in offscreen mode, waits briefly for the UI to settle,
and writes a deterministic PNG snapshot. Useful for CI smoke checks and
golden-image generation without a display server.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from PySide6 import QtWidgets

# Deterministic defaults for CI runs
DEFAULT_OUTPUT = Path("tests/artifacts/offscreen_snap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture an offscreen viewer snapshot")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the PNG snapshot (default: tests/artifacts/offscreen_snap.png)",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Optional dataset path to open before capturing the snapshot",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.5,
        help="Seconds to wait for UI settle before capture (default: 0.5)",
    )
    return parser.parse_args()


def configure_offscreen_env() -> None:
    """Force deterministic, headless Qt settings."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")


def take_snapshot(output_path: Path, input_path: str | None, wait_time: float) -> Path:
    """Launch the viewer offscreen and capture a snapshot to the given path."""

    configure_offscreen_env()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Local import to keep startup lightweight for CLI tooling
    from xpcsviewer.xpcs_viewer import XpcsViewer

    viewer_kwargs = {"path": input_path} if input_path else {}
    viewer = XpcsViewer(**viewer_kwargs)
    viewer.resize(1280, 800)
    viewer.show()

    app.processEvents()
    if wait_time > 0:
        time.sleep(wait_time)
        app.processEvents()

    screen = app.primaryScreen()
    if screen is None:
        raise RuntimeError("No primary screen available for offscreen capture")

    pixmap = screen.grabWindow(int(viewer.winId()))
    if not pixmap.save(str(output_path), "PNG"):
        raise RuntimeError(f"Failed to write snapshot to {output_path}")

    viewer.close()
    app.quit()

    return output_path


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()

    snapshot_path = take_snapshot(output_path, args.input_path, args.wait)
    print(f"Snapshot saved to {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
