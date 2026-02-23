"""Coordinator for batched all-Q Bayesian fitting.

Manages a bounded-parallelism queue of :class:`BayesianFitWorker` instances,
launching at most ``max_concurrent`` workers at a time and collecting results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, Signal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BatchBayesianCoordinator(QObject):
    """Manage parallel Bayesian fits across multiple Q-bins.

    Parameters
    ----------
    q_specs : list[dict]
        Each dict contains keys: ``q_idx``, ``x``, ``y``, ``yerr``,
        ``q_value``, ``fit_func`` (callable), ``sampler_kwargs`` (dict).
    thread_pool : QThreadPool
        Thread pool to submit workers to.
    max_concurrent : int
        Maximum number of workers running simultaneously.
    """

    progress = Signal(int, int, str)  # (completed, total, message)
    all_finished = Signal(dict)  # {q_idx: result_dict | None}
    single_q_finished = Signal(int, object)  # (q_idx, result_dict)
    single_q_error = Signal(int, str)  # (q_idx, error_message)

    def __init__(
        self,
        q_specs: list[dict[str, Any]],
        thread_pool: Any,
        max_concurrent: int = 4,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._q_specs = q_specs
        self._thread_pool = thread_pool
        self._max_concurrent = max(1, max_concurrent)

        self._results: dict[int, dict[str, Any] | None] = {}
        self._active_count = 0
        self._next_index = 0
        self._completed = 0
        self._total = len(q_specs)
        self._cancelled = False
        # Maps q_idx → worker for active workers only; entries are removed
        # on completion to avoid holding references to finished QRunnables.
        self._active_workers: dict[int, Any] = {}

    @property
    def total(self) -> int:
        return self._total

    def start(self) -> None:
        """Launch the initial batch of workers."""
        if self._total == 0:
            self.all_finished.emit({})
            return

        n_launch = min(self._max_concurrent, self._total)
        for _ in range(n_launch):
            self._launch_next()

    def cancel(self) -> None:
        """Stop launching new workers and cancel active ones."""
        self._cancelled = True
        for worker in self._active_workers.values():
            if hasattr(worker, "cancel"):
                worker.cancel()
        logger.info("Batch Bayesian fitting cancelled")

    def _launch_next(self) -> None:
        """Launch the next worker in the queue if available."""
        if self._cancelled or self._next_index >= self._total:
            return

        from .bayesian_worker import BayesianFitWorker

        spec = self._q_specs[self._next_index]
        self._next_index += 1

        q_idx = spec["q_idx"]
        worker = BayesianFitWorker(
            fit_func=spec["fit_func"],
            x=spec["x"],
            y=spec["y"],
            yerr=spec["yerr"],
            q_index=q_idx,
            q_value=spec["q_value"],
            context="g2",
            sampler_kwargs=spec.get("sampler_kwargs"),
        )
        self._active_workers[q_idx] = worker

        # Use default arguments to capture q_idx in closures
        worker.signals.finished.connect(
            lambda result, qi=q_idx: self._on_worker_finished(qi, result)
        )
        worker.signals.error.connect(
            lambda _wid, msg, _tb, _retry, qi=q_idx: self._on_worker_error(qi, msg)
        )
        worker.signals.cancelled.connect(
            lambda _wid, _reason, qi=q_idx: self._on_worker_cancelled(qi)
        )

        self._active_count += 1
        self._thread_pool.start(worker)

    def _on_worker_finished(self, q_idx: int, result: dict[str, Any] | None) -> None:
        """Handle a successful worker completion."""
        self._active_workers.pop(q_idx, None)
        self._active_count -= 1
        self._completed += 1
        self._results[q_idx] = result

        self.single_q_finished.emit(q_idx, result)
        self.progress.emit(
            self._completed,
            self._total,
            f"Q-bin {q_idx} done ({self._completed}/{self._total})",
        )

        if self._completed >= self._total:
            self.all_finished.emit(self._results)
        else:
            self._launch_next()

    def _on_worker_error(self, q_idx: int, error_msg: str) -> None:
        """Handle a worker failure — store None and continue."""
        self._active_workers.pop(q_idx, None)
        self._active_count -= 1
        self._completed += 1
        self._results[q_idx] = None

        logger.warning("Bayesian fit failed for Q-index %d: %s", q_idx, error_msg)
        self.single_q_error.emit(q_idx, error_msg)
        self.progress.emit(
            self._completed,
            self._total,
            f"Q-bin {q_idx} failed ({self._completed}/{self._total})",
        )

        if self._completed >= self._total:
            self.all_finished.emit(self._results)
        else:
            self._launch_next()

    def _on_worker_cancelled(self, q_idx: int) -> None:
        """Handle a cancelled worker — treat as failure to keep accounting correct."""
        self._active_workers.pop(q_idx, None)
        self._active_count -= 1
        self._completed += 1
        self._results[q_idx] = None

        logger.info("Bayesian fit cancelled for Q-index %d", q_idx)
        self.single_q_error.emit(q_idx, "Cancelled")
        self.progress.emit(
            self._completed,
            self._total,
            f"Q-bin {q_idx} cancelled ({self._completed}/{self._total})",
        )

        if self._completed >= self._total:
            self.all_finished.emit(self._results)
        else:
            self._launch_next()
