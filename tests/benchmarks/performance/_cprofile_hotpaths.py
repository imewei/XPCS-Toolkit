"""cProfile analysis script for the 6 hot paths.

Outputs cumulative time breakdown for each hot path.
Run with: uv run python tests/benchmarks/performance/_cprofile_hotpaths.py
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"


def profile_func(name: str, func, *args, **kwargs):
    """Profile a function and print top 15 cumulative callers."""
    print(f"\n{'=' * 70}")
    print(f"  PROFILE: {name}")
    print(f"{'=' * 70}")

    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream)
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    print(stream.getvalue())
    return result


def main():
    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------
    # 1. Q-map computation
    # -------------------------------------------------------------------
    from xpcsviewer.simplemask.qmap import _compute_transmission_qmap_backend

    profile_func(
        "Q-map 512x512 (transmission, with JIT)",
        _compute_transmission_qmap_backend,
        10.0,
        (256.0, 256.0),
        (512, 512),
        0.075,
        5000.0,
    )

    # -------------------------------------------------------------------
    # 2. C2 cleaning
    # -------------------------------------------------------------------
    from xpcsviewer.module.twotime import clean_c2_for_visualization

    c2 = rng.random((500, 500), dtype=np.float64)
    c2[rng.random((500, 500)) < 0.01] = np.nan

    profile_func(
        "C2 clean (nan_to_num, 500x500)",
        clean_c2_for_visualization,
        c2,
        method="nan_to_num",
    )

    # -------------------------------------------------------------------
    # 3. G2 ensemble statistics
    # -------------------------------------------------------------------
    from xpcsviewer.module.g2mod import compute_g2_ensemble_statistics

    g2_list = [rng.random((200, 50)) for _ in range(5)]
    profile_func(
        "G2 ensemble stats (5 x 200x50)",
        compute_g2_ensemble_statistics,
        g2_list,
    )

    # -------------------------------------------------------------------
    # 4. SAXS q-binning (large)
    # -------------------------------------------------------------------
    from xpcsviewer.module.saxs1d import vectorized_q_binning

    q = rng.random(512 * 512) * 0.5
    intensity = rng.random(512 * 512) * 1000
    profile_func(
        "SAXS q-binning (262144 points, 200 bins)",
        vectorized_q_binning,
        q,
        intensity,
        0.01,
        0.5,
        200,
    )

    # -------------------------------------------------------------------
    # 5. NLSQ fitting
    # -------------------------------------------------------------------
    from xpcsviewer.fitting.models import single_exp_func
    from xpcsviewer.fitting.nlsq import nlsq_optimize

    tau = np.logspace(-4, 2, 500)
    g2 = 1.0 + 0.3 * np.exp(-2 * tau / 1.0) + rng.normal(0, 0.01, 500)
    g2_err = np.full(500, 0.01)
    profile_func(
        "NLSQ single_exp (500 points, robust preset)",
        nlsq_optimize,
        single_exp_func,
        tau,
        g2,
        g2_err,
        p0={"tau": 0.5, "baseline": 1.0, "contrast": 0.2},
        bounds={"tau": (0.01, 100.0), "baseline": (0.5, 1.5), "contrast": (0.01, 1.0)},
    )

    # -------------------------------------------------------------------
    # 6. FFT computation
    # -------------------------------------------------------------------
    intensity_t = rng.random(100_000) * 1000

    def fft_pipeline(data):
        fft_vals = np.fft.fft(data)
        mags = np.abs(fft_vals)
        freqs = np.fft.fftfreq(len(data), 1.0)
        mask = freqs > 0
        return freqs[mask], mags[mask]

    profile_func("FFT pipeline (100k frames)", fft_pipeline, intensity_t)

    print("\n" + "=" * 70)
    print("  All profiles complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
