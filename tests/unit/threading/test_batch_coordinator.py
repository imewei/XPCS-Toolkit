"""Tests for BatchBayesianCoordinator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from xpcsviewer.threading.batch_bayesian_coordinator import (
    BatchBayesianCoordinator,
)


@pytest.fixture()
def mock_thread_pool():
    """Create a mock QThreadPool that runs workers synchronously."""
    pool = MagicMock()
    # start() captures workers but doesn't run them
    pool.started_workers = []

    def capture_start(worker):
        pool.started_workers.append(worker)

    pool.start = capture_start
    return pool


class TestBatchBayesianCoordinator:
    """Test coordinator state management."""

    def _make_specs(self, n: int) -> list[dict]:
        """Create n dummy Q-bin specs."""
        import numpy as np

        return [
            {
                "q_idx": i,
                "x": np.linspace(0.001, 100, 20),
                "y": np.ones(20) + 0.1 * i,
                "yerr": np.ones(20) * 0.01,
                "q_value": 0.001 * (i + 1),
                "fit_func": MagicMock(),
            }
            for i in range(n)
        ]

    def test_empty_specs_finishes_immediately(self, mock_thread_pool, qtbot):
        """Empty spec list should emit all_finished immediately."""
        coord = BatchBayesianCoordinator([], mock_thread_pool)
        with qtbot.waitSignal(coord.all_finished, timeout=1000):
            coord.start()
        assert coord._results == {}
        assert coord._completed == 0

    def test_launches_bounded_workers(self, mock_thread_pool):
        """Should launch min(max_concurrent, total) workers initially."""
        specs = self._make_specs(10)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=3)
        coord.start()
        assert len(mock_thread_pool.started_workers) == 3

    def test_launches_all_if_fewer_than_max(self, mock_thread_pool):
        """If total < max_concurrent, launch all."""
        specs = self._make_specs(2)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=4)
        coord.start()
        assert len(mock_thread_pool.started_workers) == 2

    def test_cancel_stops_launching(self, mock_thread_pool):
        """After cancel(), no more workers should be launched."""
        specs = self._make_specs(10)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=2)
        coord.start()
        initial_count = len(mock_thread_pool.started_workers)

        coord.cancel()

        # Simulate a worker finishing
        coord._on_worker_finished(0, {"fit_result": MagicMock()})

        # Should NOT have launched another worker
        assert len(mock_thread_pool.started_workers) == initial_count

    def test_error_continues_batch(self, mock_thread_pool, qtbot):
        """A failed Q-bin should store None and continue."""
        specs = self._make_specs(2)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=2)
        coord.start()

        # Simulate first worker error, second success
        coord._on_worker_error(0, "test error")
        assert coord._results[0] is None
        assert coord._completed == 1

        with qtbot.waitSignal(coord.all_finished, timeout=1000):
            coord._on_worker_finished(1, {"fit_result": MagicMock()})

        # Check coordinator state directly (PySide6 has issues copying
        # dicts through signals, so we don't rely on blocker.args).
        assert coord._results[0] is None
        assert coord._results[1] is not None

    def test_progress_signal_emitted(self, mock_thread_pool, qtbot):
        """Progress signal should be emitted after each completion."""
        specs = self._make_specs(3)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=3)
        coord.start()

        with qtbot.waitSignal(coord.progress, timeout=1000) as blocker:
            coord._on_worker_finished(0, {"fit_result": MagicMock()})

        completed, total, msg = blocker.args
        assert completed == 1
        assert total == 3

    def test_single_q_error_signal(self, mock_thread_pool, qtbot):
        """single_q_error should be emitted on worker failure."""
        specs = self._make_specs(2)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=2)
        coord.start()

        with qtbot.waitSignal(coord.single_q_error, timeout=1000) as blocker:
            coord._on_worker_error(1, "boom")

        assert blocker.args == [1, "boom"]

    def test_max_concurrent_enforced(self, mock_thread_pool):
        """Should never exceed max_concurrent active workers."""
        specs = self._make_specs(6)
        coord = BatchBayesianCoordinator(specs, mock_thread_pool, max_concurrent=2)
        coord.start()
        assert len(mock_thread_pool.started_workers) == 2

        # Complete first → should launch third
        coord._on_worker_finished(0, {"fit_result": MagicMock()})
        assert len(mock_thread_pool.started_workers) == 3

        # Complete second → should launch fourth
        coord._on_worker_finished(1, {"fit_result": MagicMock()})
        assert len(mock_thread_pool.started_workers) == 4
