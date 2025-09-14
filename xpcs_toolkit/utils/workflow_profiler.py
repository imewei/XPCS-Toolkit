"""
Workflow profiling system for XPCS Toolkit user workflow analysis.

This module provides non-intrusive profiling capabilities to monitor real-world
user workflows and identify CPU bottlenecks during actual XPCS data processing operations.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

from .logging_config import get_logger
from .performance_profiler import global_profiler

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    """Represents a single step in a user workflow."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_time: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    thread_count: Optional[int] = None
    io_wait_time: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    sub_steps: List["WorkflowStep"] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Calculate the duration of the workflow step."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def memory_delta(self) -> Optional[float]:
        """Calculate memory usage delta in MB."""
        if self.memory_before is None or self.memory_after is None:
            return None
        return self.memory_after - self.memory_before


@dataclass
class WorkflowProfile:
    """Complete profile of a user workflow session."""

    session_id: str
    workflow_type: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[WorkflowStep] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    file_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> Optional[float]:
        """Total workflow duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class ThreadingProfiler:
    """Profiles thread utilization during workflows."""

    def __init__(self):
        self.thread_samples: List[
            Tuple[float, int, float]
        ] = []  # (timestamp, thread_count, cpu_percent)
        self.sampling_interval = 0.5  # Sample every 500ms
        self._sampling_active = False
        self._sample_thread = None

    def start_sampling(self):
        """Start thread utilization sampling."""
        if self._sampling_active:
            return

        self._sampling_active = True
        self._sample_thread = threading.Thread(
            target=self._sample_thread_utilization, daemon=True
        )
        self._sample_thread.start()
        logger.debug("Thread utilization sampling started")

    def stop_sampling(self):
        """Stop thread utilization sampling."""
        self._sampling_active = False
        if self._sample_thread and self._sample_thread.is_alive():
            self._sample_thread.join(timeout=2.0)
        logger.debug("Thread utilization sampling stopped")

    def _sample_thread_utilization(self):
        """Sample thread utilization in background."""
        process = psutil.Process()

        while self._sampling_active:
            try:
                timestamp = time.time()
                thread_count = process.num_threads()
                cpu_percent = process.cpu_percent()

                self.thread_samples.append((timestamp, thread_count, cpu_percent))

                # Keep only last 1000 samples to prevent memory growth
                if len(self.thread_samples) > 1000:
                    self.thread_samples = self.thread_samples[-500:]

                time.sleep(self.sampling_interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Error sampling thread utilization: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in thread sampling: {e}")
                break

    def get_utilization_stats(
        self, start_time: float, end_time: float
    ) -> Dict[str, float]:
        """Get thread utilization statistics for a time range."""
        relevant_samples = [
            (ts, threads, cpu)
            for ts, threads, cpu in self.thread_samples
            if start_time <= ts <= end_time
        ]

        if not relevant_samples:
            return {}

        thread_counts = [threads for _, threads, _ in relevant_samples]
        cpu_percentages = [cpu for _, _, cpu in relevant_samples]

        return {
            "avg_thread_count": np.mean(thread_counts),
            "max_thread_count": np.max(thread_counts),
            "min_thread_count": np.min(thread_counts),
            "avg_cpu_percent": np.mean(cpu_percentages),
            "max_cpu_percent": np.max(cpu_percentages),
            "thread_count_std": np.std(thread_counts),
            "cpu_percent_std": np.std(cpu_percentages),
        }


class IOProfiler:
    """Profiles I/O operations during workflows."""

    def __init__(self):
        self.io_operations: List[Dict[str, Any]] = []
        self._io_start_times: Dict[str, float] = {}

    def start_io_operation(
        self, operation_id: str, operation_type: str, file_path: Optional[str] = None
    ):
        """Start tracking an I/O operation."""
        self._io_start_times[operation_id] = time.time()
        logger.debug(f"Started I/O operation: {operation_id} ({operation_type})")

    def end_io_operation(
        self,
        operation_id: str,
        bytes_transferred: Optional[int] = None,
        success: bool = True,
    ):
        """End tracking an I/O operation."""
        if operation_id not in self._io_start_times:
            logger.warning(f"Ending unknown I/O operation: {operation_id}")
            return

        start_time = self._io_start_times.pop(operation_id)
        duration = time.time() - start_time

        operation_record = {
            "operation_id": operation_id,
            "start_time": start_time,
            "duration": duration,
            "bytes_transferred": bytes_transferred,
            "success": success,
            "throughput_mb_s": (bytes_transferred / (1024 * 1024) / duration)
            if bytes_transferred and duration > 0
            else None,
        }

        self.io_operations.append(operation_record)
        logger.debug(f"Completed I/O operation: {operation_id} in {duration:.3f}s")

    def get_io_stats(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get I/O statistics for a time range."""
        relevant_ops = [
            op
            for op in self.io_operations
            if start_time <= op["start_time"] <= end_time
        ]

        if not relevant_ops:
            return {"total_operations": 0}

        durations = [op["duration"] for op in relevant_ops]
        successful_ops = [op for op in relevant_ops if op["success"]]

        total_bytes = sum(
            op["bytes_transferred"] for op in relevant_ops if op["bytes_transferred"]
        )

        return {
            "total_operations": len(relevant_ops),
            "successful_operations": len(successful_ops),
            "success_rate": len(successful_ops) / len(relevant_ops),
            "total_io_time": sum(durations),
            "avg_io_time": np.mean(durations),
            "max_io_time": np.max(durations),
            "total_bytes_transferred": total_bytes,
            "avg_throughput_mb_s": np.mean(
                [
                    op["throughput_mb_s"]
                    for op in relevant_ops
                    if op["throughput_mb_s"] is not None
                ]
            )
            if relevant_ops
            else 0,
        }


class WorkflowProfiler:
    """
    Main workflow profiler that orchestrates all profiling activities.

    This class provides non-intrusive hooks that can be integrated into existing
    XPCS Toolkit workflows to monitor real-world usage patterns and identify
    CPU bottlenecks.
    """

    def __init__(self):
        self.active_profiles: Dict[str, WorkflowProfile] = {}
        self.completed_profiles: List[WorkflowProfile] = []
        self.threading_profiler = ThreadingProfiler()
        self.io_profiler = IOProfiler()

        # Integration with existing performance profiler
        self.performance_profiler = global_profiler

        # Workflow step stack for nested profiling
        self._step_stack: Dict[str, List[WorkflowStep]] = defaultdict(list)

        # System info cache
        self._system_info = self._get_system_info()

        # Maximum profiles to keep in memory
        self.max_completed_profiles = 100

        logger.info("WorkflowProfiler initialized")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for profiling context."""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                "process_id": os.getpid(),
            }
        except Exception as e:
            logger.warning(f"Could not gather system info: {e}")
            return {}

    def start_workflow(
        self, session_id: str, workflow_type: str, **workflow_params
    ) -> str:
        """
        Start profiling a user workflow.

        Args:
            session_id: Unique identifier for the workflow session
            workflow_type: Type of workflow (e.g., 'file_loading', 'g2_analysis', 'plotting')
            **workflow_params: Additional workflow parameters

        Returns:
            Session ID for the started workflow
        """
        if session_id in self.active_profiles:
            logger.warning(
                f"Workflow {session_id} already active, ending previous session"
            )
            self.end_workflow(session_id)

        profile = WorkflowProfile(
            session_id=session_id,
            workflow_type=workflow_type,
            start_time=time.time(),
            system_info=self._system_info.copy(),
        )

        # Add workflow parameters to file info
        profile.file_info.update(workflow_params)

        self.active_profiles[session_id] = profile
        self.threading_profiler.start_sampling()

        logger.info(f"Started workflow profiling: {session_id} ({workflow_type})")
        return session_id

    def end_workflow(self, session_id: str):
        """End profiling a user workflow."""
        if session_id not in self.active_profiles:
            logger.warning(f"Cannot end unknown workflow: {session_id}")
            return

        profile = self.active_profiles.pop(session_id)
        profile.end_time = time.time()

        # Complete any remaining open steps
        if session_id in self._step_stack:
            for step in self._step_stack[session_id]:
                if step.end_time is None:
                    step.end_time = profile.end_time
            del self._step_stack[session_id]

        self.completed_profiles.append(profile)

        # Maintain memory limits
        if len(self.completed_profiles) > self.max_completed_profiles:
            self.completed_profiles = self.completed_profiles[
                -self.max_completed_profiles // 2 :
            ]

        # Stop thread sampling if no active workflows
        if not self.active_profiles:
            self.threading_profiler.stop_sampling()

        logger.info(
            f"Ended workflow profiling: {session_id} "
            f"(duration: {profile.total_duration:.2f}s)"
        )

    def start_step(
        self, session_id: str, step_name: str, **step_params
    ) -> WorkflowStep:
        """
        Start profiling a workflow step.

        Args:
            session_id: Workflow session ID
            step_name: Name of the step
            **step_params: Additional step parameters

        Returns:
            WorkflowStep object for the started step
        """
        if session_id not in self.active_profiles:
            logger.warning(f"Cannot start step for unknown workflow: {session_id}")
            return None

        # Get memory usage before step
        memory_before = None
        thread_count = None
        try:
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            thread_count = process.num_threads()
        except Exception as e:
            logger.debug(f"Could not get process info for step: {e}")

        step = WorkflowStep(
            name=step_name,
            start_time=time.time(),
            memory_before=memory_before,
            thread_count=thread_count,
            parameters=step_params.copy(),
        )

        # Add to step stack for nested tracking
        self._step_stack[session_id].append(step)

        logger.debug(f"Started workflow step: {session_id}.{step_name}")
        return step

    def end_step(
        self, session_id: str, step_name: Optional[str] = None
    ) -> Optional[WorkflowStep]:
        """
        End profiling a workflow step.

        Args:
            session_id: Workflow session ID
            step_name: Name of the step (if None, ends the most recent step)

        Returns:
            Completed WorkflowStep object
        """
        if session_id not in self.active_profiles:
            logger.warning(f"Cannot end step for unknown workflow: {session_id}")
            return None

        if session_id not in self._step_stack or not self._step_stack[session_id]:
            logger.warning(f"No active steps to end for workflow: {session_id}")
            return None

        # Find the step to end
        step = None
        if step_name is None:
            # End the most recent step
            step = self._step_stack[session_id].pop()
        else:
            # Find and remove the named step
            for i, s in enumerate(reversed(self._step_stack[session_id])):
                if s.name == step_name:
                    step = self._step_stack[session_id].pop(-(i + 1))
                    break

        if step is None:
            logger.warning(
                f"Could not find step '{step_name}' to end in workflow: {session_id}"
            )
            return None

        # Complete the step
        step.end_time = time.time()

        # Get memory usage after step
        try:
            process = psutil.Process()
            step.memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        except Exception as e:
            logger.debug(f"Could not get memory info after step: {e}")

        # Add step to the workflow profile
        profile = self.active_profiles[session_id]

        # If there are remaining steps in stack, add as sub-step to parent
        if self._step_stack[session_id]:
            parent_step = self._step_stack[session_id][-1]
            parent_step.sub_steps.append(step)
        else:
            profile.steps.append(step)

        logger.debug(
            f"Ended workflow step: {session_id}.{step_name} "
            f"(duration: {step.duration:.3f}s)"
        )
        return step

    @contextmanager
    def profile_step(self, session_id: str, step_name: str, **step_params):
        """
        Context manager for profiling a workflow step.

        Args:
            session_id: Workflow session ID
            step_name: Name of the step
            **step_params: Additional step parameters
        """
        step = self.start_step(session_id, step_name, **step_params)
        try:
            yield step
        finally:
            self.end_step(session_id, step_name)

    def profile_function_call(self, session_id: str, func_name: str):
        """
        Decorator to profile individual function calls within a workflow.

        Args:
            session_id: Workflow session ID
            func_name: Name of the function being profiled
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_step(
                    session_id,
                    func_name,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys()),
                ):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def add_step_annotation(self, session_id: str, annotation: str, data: Any = None):
        """
        Add an annotation to the current step.

        Args:
            session_id: Workflow session ID
            annotation: Description of the annotation
            data: Optional data to associate with the annotation
        """
        if session_id not in self._step_stack or not self._step_stack[session_id]:
            return

        current_step = self._step_stack[session_id][-1]
        if "annotations" not in current_step.parameters:
            current_step.parameters["annotations"] = []

        current_step.parameters["annotations"].append(
            {"timestamp": time.time(), "annotation": annotation, "data": data}
        )

    def get_active_workflows(self) -> Dict[str, str]:
        """Get currently active workflow sessions."""
        return {
            session_id: profile.workflow_type
            for session_id, profile in self.active_profiles.items()
        }

    def get_workflow_profile(self, session_id: str) -> Optional[WorkflowProfile]:
        """Get a workflow profile by session ID."""
        # Check active profiles first
        if session_id in self.active_profiles:
            return self.active_profiles[session_id]

        # Check completed profiles
        for profile in self.completed_profiles:
            if profile.session_id == session_id:
                return profile

        return None

    def get_recent_profiles(
        self, count: int = 10, workflow_type: Optional[str] = None
    ) -> List[WorkflowProfile]:
        """
        Get recent workflow profiles.

        Args:
            count: Maximum number of profiles to return
            workflow_type: Optional filter by workflow type

        Returns:
            List of recent WorkflowProfile objects
        """
        profiles = self.completed_profiles.copy()

        if workflow_type:
            profiles = [p for p in profiles if p.workflow_type == workflow_type]

        # Sort by start time (most recent first)
        profiles.sort(key=lambda p: p.start_time, reverse=True)

        return profiles[:count]

    def export_profile_data(self, session_id: str, file_path: str):
        """Export workflow profile data to JSON file."""
        profile = self.get_workflow_profile(session_id)
        if not profile:
            raise ValueError(f"Profile not found: {session_id}")

        # Convert profile to serializable format
        profile_data = {
            "session_id": profile.session_id,
            "workflow_type": profile.workflow_type,
            "start_time": profile.start_time,
            "end_time": profile.end_time,
            "total_duration": profile.total_duration,
            "system_info": profile.system_info,
            "file_info": profile.file_info,
            "steps": self._serialize_steps(profile.steps),
        }

        with open(file_path, "w") as f:
            json.dump(profile_data, f, indent=2)

        logger.info(f"Exported profile data to {file_path}")

    def _serialize_steps(self, steps: List[WorkflowStep]) -> List[Dict[str, Any]]:
        """Convert WorkflowStep objects to serializable format."""
        serialized = []
        for step in steps:
            step_data = {
                "name": step.name,
                "start_time": step.start_time,
                "end_time": step.end_time,
                "duration": step.duration,
                "memory_before": step.memory_before,
                "memory_after": step.memory_after,
                "memory_delta": step.memory_delta,
                "thread_count": step.thread_count,
                "parameters": step.parameters,
                "sub_steps": self._serialize_steps(step.sub_steps),
            }
            serialized.append(step_data)
        return serialized

    def clear_completed_profiles(self):
        """Clear all completed workflow profiles."""
        cleared_count = len(self.completed_profiles)
        self.completed_profiles.clear()
        logger.info(f"Cleared {cleared_count} completed workflow profiles")


# Global workflow profiler instance
workflow_profiler = WorkflowProfiler()


def get_workflow_profiler() -> WorkflowProfiler:
    """
    Get the global workflow profiler instance.

    Returns
    -------
    WorkflowProfiler
        Global workflow profiler instance
    """
    return workflow_profiler


# Alias for compatibility
ProfileStep = WorkflowStep


# Convenience functions for easy integration


def start_workflow_profiling(
    workflow_type: str, session_id: Optional[str] = None, **params
) -> str:
    """
    Convenience function to start workflow profiling.

    Args:
        workflow_type: Type of workflow being profiled
        session_id: Optional session ID (generated if not provided)
        **params: Workflow parameters

    Returns:
        Session ID for the started workflow
    """
    if session_id is None:
        session_id = f"{workflow_type}_{int(time.time() * 1000)}"

    return workflow_profiler.start_workflow(session_id, workflow_type, **params)


def end_workflow_profiling(session_id: str):
    """Convenience function to end workflow profiling."""
    workflow_profiler.end_workflow(session_id)


@contextmanager
def profile_workflow(workflow_type: str, session_id: Optional[str] = None, **params):
    """
    Context manager for profiling an entire workflow.

    Args:
        workflow_type: Type of workflow being profiled
        session_id: Optional session ID
        **params: Workflow parameters
    """
    session_id = start_workflow_profiling(workflow_type, session_id, **params)
    try:
        yield session_id
    finally:
        end_workflow_profiling(session_id)


@contextmanager
def profile_workflow_step(session_id: str, step_name: str, **params):
    """
    Context manager for profiling a workflow step.

    Args:
        session_id: Workflow session ID
        step_name: Name of the step
        **params: Step parameters
    """
    with workflow_profiler.profile_step(session_id, step_name, **params):
        yield


def profile_workflow_function(session_id: str, func_name: Optional[str] = None):
    """
    Decorator for profiling individual functions within a workflow.

    Args:
        session_id: Workflow session ID
        func_name: Name of the function (defaults to function.__name__)
    """

    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        return workflow_profiler.profile_function_call(session_id, name)(func)

    return decorator
