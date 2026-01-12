import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAP_SCRIPT = ROOT / "tests" / "gui_interactive" / "offscreen_snap.py"
GOLDEN = ROOT / "tests" / "gui_interactive" / "goldens" / "offscreen_snap.png"
OUTPUT = ROOT / "tests" / "artifacts" / "offscreen_snap.png"

# SSIM threshold for perceptual similarity (0.95 = very similar)
SSIM_THRESHOLD = 0.95


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index between two images.

    Uses a simplified SSIM implementation that doesn't require scikit-image.
    Returns value between 0 (different) and 1 (identical).
    """
    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants for numerical stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Mean
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Variance and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


@pytest.mark.gui
def test_offscreen_snapshot_matches_golden(tmp_path):
    """Generate an offscreen snapshot and compare it perceptually to the golden.

    Uses SSIM (structural similarity) for comparison to handle platform-specific
    rendering differences while still catching meaningful visual regressions.
    """
    # Import PIL lazily
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
        ["python", str(SNAP_SCRIPT), "--output", str(out_path)],
        check=True,
        env=env,
    )

    if not GOLDEN.exists():
        pytest.skip("Golden snapshot missing; run with --update-goldens to create")

    # Load images and convert to numpy arrays
    generated_img = np.array(Image.open(out_path).convert("L"))  # Grayscale
    golden_img = np.array(Image.open(GOLDEN).convert("L"))

    # Handle size mismatch by resizing to common dimensions
    if generated_img.shape != golden_img.shape:
        # Resize to smaller of the two
        min_h = min(generated_img.shape[0], golden_img.shape[0])
        min_w = min(generated_img.shape[1], golden_img.shape[1])
        generated_img = generated_img[:min_h, :min_w]
        golden_img = golden_img[:min_h, :min_w]

    ssim_score = compute_ssim(generated_img, golden_img)

    assert ssim_score >= SSIM_THRESHOLD, (
        f"SSIM score {ssim_score:.4f} below threshold {SSIM_THRESHOLD}. "
        f"Image differs significantly from golden."
    )
