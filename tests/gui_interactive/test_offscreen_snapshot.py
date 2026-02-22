import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAP_SCRIPT = ROOT / "tests" / "gui_interactive" / "offscreen_snap.py"
GOLDEN = ROOT / "tests" / "gui_interactive" / "goldens" / "offscreen_snap.png"
OUTPUT = ROOT / "tests" / "artifacts" / "offscreen_snap.png"

# Pixel match threshold: fraction of pixels that must be within PIXEL_TOLERANCE.
# Offscreen rendering differs across CI environments (font hinting, DPI, Qt
# platform plugin). 80% catches major regressions while tolerating cosmetic
# variation between local and CI runners.
MATCH_THRESHOLD = 0.80
PIXEL_TOLERANCE = 15  # Max per-pixel intensity difference to count as "match"


def pixel_match_rate(img1: np.ndarray, img2: np.ndarray, tolerance: int) -> float:
    """Compute fraction of pixels matching within tolerance.

    More robust than global SSIM for offscreen rendering where localized
    font/cursor differences cause outsized SSIM penalties.
    """
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    return float(np.mean(diff <= tolerance))


@pytest.mark.gui
def test_offscreen_snapshot_matches_golden(tmp_path):
    """Generate an offscreen snapshot and compare it to the golden.

    Uses pixel match rate for comparison — more robust than global SSIM
    for offscreen rendering where font aliasing varies between runs.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow not installed for image comparison")

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
    env.setdefault("QT_SCALE_FACTOR", "1")

    out_path = tmp_path / "snap.png"

    subprocess.run(
        [sys.executable, str(SNAP_SCRIPT), "--output", str(out_path)],
        check=True,
        env=env,
    )

    if not GOLDEN.exists():
        pytest.skip("Golden snapshot missing; run with --update-goldens to create")

    generated_img = np.array(Image.open(out_path).convert("L"))
    golden_img = np.array(Image.open(GOLDEN).convert("L"))

    # Handle size mismatch
    if generated_img.shape != golden_img.shape:
        min_h = min(generated_img.shape[0], golden_img.shape[0])
        min_w = min(generated_img.shape[1], golden_img.shape[1])
        generated_img = generated_img[:min_h, :min_w]
        golden_img = golden_img[:min_h, :min_w]

    match_rate = pixel_match_rate(generated_img, golden_img, PIXEL_TOLERANCE)

    assert match_rate >= MATCH_THRESHOLD, (
        f"Pixel match rate {match_rate:.4f} below threshold {MATCH_THRESHOLD}. "
        f"Only {match_rate * 100:.1f}% of pixels match within ±{PIXEL_TOLERANCE}."
    )
