"""Abstract base class for low-level scattering-format loaders.

A format loader turns a raw detector file (HDF5 today; IMM/Rigaku land in
PR3) into a 2-D mean scattering image.
"""

from __future__ import annotations

import abc
import os

import numpy as np


class ScatteringDataset(abc.ABC):
    """Minimal interface every format loader implements."""

    def __init__(self, fname: str) -> None:
        self.fname: str = fname
        # Subclasses set the real detector shape during construction.
        self.det_size: tuple[int, int] = (0, 0)

    @property
    def file_size_mb(self) -> float:
        """Size of the backing file in MiB (0 if it cannot be determined)."""
        try:
            return os.path.getsize(self.fname) / (1024**2)
        except OSError:
            return 0.0

    @abc.abstractmethod
    def get_scattering(
        self, num_frames: int = -1, begin_idx: int = 0, num_processes: int | None = None
    ) -> np.ndarray:
        """Return the per-pixel mean scattering image over a frame range.

        Args:
            num_frames: See ``io_utils.resolve_frame_range`` for semantics.
            begin_idx: First frame index to include.
            num_processes: Optional worker count; ignored by loaders that
                don't parallelize.

        Returns:
            np.ndarray: 2-D image of shape :attr:`det_size`.
        """
        raise NotImplementedError
