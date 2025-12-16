import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAP_SCRIPT = ROOT / "tests" / "gui_interactive" / "offscreen_snap.py"
GOLDEN = ROOT / "tests" / "gui_interactive" / "goldens" / "offscreen_snap.png"
OUTPUT = ROOT / "tests" / "artifacts" / "offscreen_snap.png"


@pytest.mark.gui
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Golden image comparison skipped in CI due to platform-specific rendering differences",
)
def test_offscreen_snapshot_matches_golden(tmp_path):
    """Generate an offscreen snapshot and compare it byte-for-byte to the golden."""

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    env.setdefault("QT_SCALE_FACTOR", "1")

    out_path = tmp_path / "snap.png"

    subprocess.run(
        ["python", str(SNAP_SCRIPT), "--output", str(out_path)],
        check=True,
        env=env,
    )

    assert GOLDEN.exists(), "Golden snapshot missing; refresh goldens before running."

    generated_bytes = out_path.read_bytes()
    golden_bytes = GOLDEN.read_bytes()

    assert generated_bytes == golden_bytes, (
        "Offscreen snapshot differs from golden image"
    )
