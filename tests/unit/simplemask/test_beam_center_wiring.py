"""Unit tests for SimpleMaskKernel.find_beam_center (auto beam-center wiring)."""

import numpy as np
import pytest

from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel


def _make_ring_image(shape, true_center, radius, thickness=1.5, peak=1000.0):
    """Synthetic detector image with a bright ring centered at true_center."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r = np.sqrt((xx - true_center[0]) ** 2 + (yy - true_center[1]) ** 2)
    ring = peak * np.exp(-((r - radius) ** 2) / (2 * thickness**2))
    return ring.astype(np.float64)


class TestFindBeamCenter:
    def test_no_image_loaded_raises(self):
        kernel = SimpleMaskKernel()
        with pytest.raises(RuntimeError):
            kernel.find_beam_center()

    def test_recovers_known_ring_center(self):
        true_center = (130.0, 90.0)
        image = _make_ring_image((200, 250), true_center, radius=60)

        kernel = SimpleMaskKernel()
        kernel.read_data(
            image,
            {
                "bcx": 100.0,  # deliberately off from true_center
                "bcy": 100.0,
                "det_dist": 5000.0,
                "pix_dim": 0.075,
                "energy": 10.0,
            },
        )

        cx, cy, diagnostics = kernel.find_beam_center()

        assert cx == pytest.approx(true_center[0], abs=2.0)
        assert cy == pytest.approx(true_center[1], abs=2.0)
        assert "iterations" in diagnostics

        # kernel.metadata must reflect the refined center (same contract
        # as the manual bcx/bcy spinbox path in SimpleMaskWindow).
        assert kernel.metadata["bcx"] == pytest.approx(cx)
        assert kernel.metadata["bcy"] == pytest.approx(cy)


if __name__ == "__main__":
    TestFindBeamCenter().test_no_image_loaded_raises()
    TestFindBeamCenter().test_recovers_known_ring_center()
    print("ok")
