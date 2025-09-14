"""
CPU bottleneck analysis system for XPCS Toolkit workflows.

This module analyzes workflow profiles to automatically detect CPU-bound operations,
thread contention issues, and inefficient algorithms that cause performance bottlenecks.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .logging_config import get_logger
from .workflow_profiler import WorkflowProfile, WorkflowStep, workflow_profiler

logger = get_logger(__name__)


class BottleneckSeverity(Enum):
    """Severity levels for performance bottlenecks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BottleneckType(Enum):
    """Types of performance bottlenecks."""

    CPU_BOUND = "cpu_bound"
    THREAD_CONTENTION = "thread_contention"
    MEMORY_ALLOCATION = "memory_allocation"
    ALGORITHMIC_INEFFICIENCY = "algorithmic_inefficiency"
    IO_WAIT = "io_wait"
    SYNCHRONIZATION = "synchronization"
    HOT_PATH = "hot_path"
    CACHE_MISS = "cache_miss"


@dataclass
class BottleneckFinding:
    """Represents a detected performance bottleneck."""

    bottleneck_type: BottleneckType
    severity: BottleneckSeverity
    component: str
    description: str
    metrics: Dict[str, float]
    affected_workflows: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    frequency: int = 1

    def to_dict(self) -> Dict:
        """Convert finding to dictionary representation."""
        return {
            "type": self.bottleneck_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "description": self.description,
            "metrics": self.metrics,
            "affected_workflows": self.affected_workflows,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "frequency": self.frequency,
        }


class HotPathAnalysis(NamedTuple):
    """Results of hot path analysis."""

    path: str
    execution_count: int
    total_time: float
    avg_time: float
    time_percentage: float


class ThreadContentionAnalysis(NamedTuple):
    """Results of thread contention analysis."""

    component: str
    max_threads: int
    avg_threads: float
    contention_score: float
    wait_time_estimate: float


class CPUBottleneckAnalyzer:
    """
    Analyzes workflow profiles to identify CPU performance bottlenecks.

    This analyzer examines multiple workflow profiles to identify patterns
    that indicate CPU performance issues in real-world usage scenarios.
    """

    def __init__(self):
        self.analysis_cache: Dict[str, List[BottleneckFinding]] = {}
        self.pattern_cache: Dict[str, Dict] = {}

        # Thresholds for bottleneck detection
        self.thresholds = {
            "cpu_bound_time": 2.0,  # Seconds
            "memory_growth_mb": 500,  # MB
            "thread_contention_ratio": 0.7,  # Ratio of active to max threads
            "hot_path_percentage": 0.15,  # 15% of total time
            "sync_wait_time": 0.5,  # Seconds
            "algorithmic_complexity": 10,  # O(n) multiplier threshold
        }

        logger.info("CPUBottleneckAnalyzer initialized")

    def analyze_workflow_profiles(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """
        Analyze multiple workflow profiles to identify CPU bottlenecks.

        Args:
            profiles: List of WorkflowProfile objects to analyze

        Returns:
            List of BottleneckFinding objects representing detected bottlenecks
        """
        if not profiles:
            return []

        logger.info(f"Analyzing {len(profiles)} workflow profiles for CPU bottlenecks")

        findings = []

        # Analyze different types of bottlenecks
        findings.extend(self._analyze_cpu_bound_operations(profiles))
        findings.extend(self._analyze_thread_contention(profiles))
        findings.extend(self._analyze_memory_allocation_patterns(profiles))
        findings.extend(self._analyze_algorithmic_inefficiency(profiles))
        findings.extend(self._analyze_hot_paths(profiles))
        findings.extend(self._analyze_synchronization_issues(profiles))

        # Sort findings by severity and confidence
        findings.sort(
            key=lambda f: (self._severity_score(f.severity), f.confidence_score),
            reverse=True,
        )

        logger.info(f"Identified {len(findings)} potential CPU bottlenecks")
        return findings

    def _severity_score(self, severity: BottleneckSeverity) -> int:
        """Convert severity to numeric score for sorting."""
        return {
            BottleneckSeverity.CRITICAL: 4,
            BottleneckSeverity.HIGH: 3,
            BottleneckSeverity.MEDIUM: 2,
            BottleneckSeverity.LOW: 1,
        }[severity]

    def _analyze_cpu_bound_operations(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """Identify CPU-bound operations that could be optimized."""
        findings = []

        # Collect all steps across profiles
        all_steps = []
        for profile in profiles:
            all_steps.extend(self._flatten_steps(profile.steps))

        # Group steps by name to identify patterns
        step_groups = defaultdict(list)
        for step in all_steps:
            step_groups[step.name].append(step)

        for step_name, steps in step_groups.items():
            if len(steps) < 2:  # Need multiple samples
                continue

            durations = [s.duration for s in steps if s.duration is not None]
            if not durations:
                continue

            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            std_duration = statistics.stdev(durations) if len(durations) > 1 else 0

            # Check if this step takes significant CPU time
            if avg_duration > self.thresholds["cpu_bound_time"]:
                severity = self._classify_duration_severity(avg_duration)

                metrics = {
                    "avg_duration": avg_duration,
                    "max_duration": max_duration,
                    "std_duration": std_duration,
                    "sample_count": len(steps),
                    "total_time": sum(durations),
                }

                recommendations = self._generate_cpu_bound_recommendations(
                    step_name, avg_duration, std_duration
                )

                finding = BottleneckFinding(
                    bottleneck_type=BottleneckType.CPU_BOUND,
                    severity=severity,
                    component=step_name,
                    description=f"CPU-bound operation with average duration of {avg_duration:.2f}s",
                    metrics=metrics,
                    affected_workflows=[p.session_id for p in profiles],
                    recommendations=recommendations,
                    confidence_score=min(
                        0.9, len(steps) / 10.0
                    ),  # Higher confidence with more samples
                    frequency=len(steps),
                )

                findings.append(finding)

        return findings

    def _analyze_thread_contention(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """Analyze thread contention issues."""
        findings = []

        contention_data = []

        for profile in profiles:
            # Analyze thread utilization during workflow
            if hasattr(workflow_profiler.threading_profiler, "thread_samples"):
                thread_stats = (
                    workflow_profiler.threading_profiler.get_utilization_stats(
                        profile.start_time, profile.end_time or time.time()
                    )
                )

                if thread_stats:
                    contention_ratio = thread_stats.get(
                        "avg_thread_count", 0
                    ) / thread_stats.get("max_thread_count", 1)

                    contention_data.append(
                        {
                            "workflow_type": profile.workflow_type,
                            "contention_ratio": contention_ratio,
                            "max_threads": thread_stats.get("max_thread_count", 0),
                            "avg_threads": thread_stats.get("avg_thread_count", 0),
                            "cpu_percent": thread_stats.get("avg_cpu_percent", 0),
                        }
                    )

        # Analyze contention patterns by workflow type
        workflow_contention = defaultdict(list)
        for data in contention_data:
            workflow_contention[data["workflow_type"]].append(data)

        for workflow_type, contentions in workflow_contention.items():
            if len(contentions) < 2:
                continue

            avg_contention = statistics.mean(
                [c["contention_ratio"] for c in contentions]
            )
            max_threads = max([c["max_threads"] for c in contentions])
            avg_threads = statistics.mean([c["avg_threads"] for c in contentions])

            if avg_contention > self.thresholds["thread_contention_ratio"]:
                severity = self._classify_contention_severity(avg_contention)

                metrics = {
                    "avg_contention_ratio": avg_contention,
                    "max_threads": max_threads,
                    "avg_threads": avg_threads,
                    "sample_count": len(contentions),
                }

                recommendations = [
                    "Consider reducing thread pool size to match CPU cores",
                    "Analyze thread synchronization points for bottlenecks",
                    "Implement work-stealing thread pool if not already used",
                    "Profile thread idle time to identify synchronization issues",
                ]

                finding = BottleneckFinding(
                    bottleneck_type=BottleneckType.THREAD_CONTENTION,
                    severity=severity,
                    component=workflow_type,
                    description=f"High thread contention detected (ratio: {avg_contention:.2f})",
                    metrics=metrics,
                    affected_workflows=[
                        profile.session_id
                        for profile in profiles
                        if profile.workflow_type == workflow_type
                    ],
                    recommendations=recommendations,
                    confidence_score=min(0.8, len(contentions) / 5.0),
                    frequency=len(contentions),
                )

                findings.append(finding)

        return findings

    def _analyze_memory_allocation_patterns(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """Analyze memory allocation patterns that could cause CPU overhead."""
        findings = []

        # Collect memory deltas for each step type
        memory_patterns = defaultdict(list)

        for profile in profiles:
            for step in self._flatten_steps(profile.steps):
                if step.memory_delta is not None:
                    memory_patterns[step.name].append(step.memory_delta)

        for step_name, memory_deltas in memory_patterns.items():
            if len(memory_deltas) < 3:
                continue

            avg_growth = statistics.mean(memory_deltas)
            max_growth = max(memory_deltas)
            total_growth = sum([d for d in memory_deltas if d > 0])

            # Check for excessive memory allocation
            if (
                avg_growth > self.thresholds["memory_growth_mb"]
                or max_growth > self.thresholds["memory_growth_mb"] * 2
            ):
                severity = self._classify_memory_severity(max_growth)

                metrics = {
                    "avg_memory_growth_mb": avg_growth,
                    "max_memory_growth_mb": max_growth,
                    "total_growth_mb": total_growth,
                    "sample_count": len(memory_deltas),
                }

                recommendations = [
                    "Implement object pooling for frequently allocated objects",
                    "Use memory-mapped files for large data processing",
                    "Consider lazy loading strategies for large datasets",
                    "Profile memory allocations to identify hotspots",
                    "Implement garbage collection optimization strategies",
                ]

                finding = BottleneckFinding(
                    bottleneck_type=BottleneckType.MEMORY_ALLOCATION,
                    severity=severity,
                    component=step_name,
                    description=f"High memory allocation detected (avg: {avg_growth:.1f}MB)",
                    metrics=metrics,
                    affected_workflows=[p.session_id for p in profiles],
                    recommendations=recommendations,
                    confidence_score=min(0.9, len(memory_deltas) / 10.0),
                    frequency=len(memory_deltas),
                )

                findings.append(finding)

        return findings

    def _analyze_algorithmic_inefficiency(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """Detect algorithmic inefficiency patterns."""
        findings = []

        # Look for steps where execution time grows non-linearly with data size
        step_performance = defaultdict(list)

        for profile in profiles:
            for step in self._flatten_steps(profile.steps):
                if step.duration is None:
                    continue

                # Try to extract data size indicators from parameters
                data_size = self._estimate_data_size(step)
                if data_size > 0:
                    step_performance[step.name].append((data_size, step.duration))

        for step_name, performance_data in step_performance.items():
            if len(performance_data) < 5:  # Need enough samples for trend analysis
                continue

            # Analyze time complexity
            complexity_score = self._analyze_time_complexity(performance_data)

            if complexity_score > self.thresholds["algorithmic_complexity"]:
                severity = self._classify_complexity_severity(complexity_score)

                sizes, times = zip(*performance_data)

                metrics = {
                    "complexity_score": complexity_score,
                    "min_data_size": min(sizes),
                    "max_data_size": max(sizes),
                    "min_time": min(times),
                    "max_time": max(times),
                    "sample_count": len(performance_data),
                }

                recommendations = [
                    "Profile algorithm to identify O(n²) or worse complexity",
                    "Consider more efficient data structures (e.g., hash maps vs lists)",
                    "Implement caching for repeated calculations",
                    "Use vectorized operations where possible",
                    "Consider approximate algorithms for large datasets",
                ]

                finding = BottleneckFinding(
                    bottleneck_type=BottleneckType.ALGORITHMIC_INEFFICIENCY,
                    severity=severity,
                    component=step_name,
                    description=f"Algorithmic inefficiency detected (complexity score: {complexity_score:.1f})",
                    metrics=metrics,
                    affected_workflows=[p.session_id for p in profiles],
                    recommendations=recommendations,
                    confidence_score=min(0.8, len(performance_data) / 20.0),
                    frequency=len(performance_data),
                )

                findings.append(finding)

        return findings

    def _analyze_hot_paths(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """Identify frequently executed code paths that consume significant CPU time."""
        findings = []

        # Collect execution statistics for each step
        step_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "times": []})
        total_workflow_time = 0.0

        for profile in profiles:
            workflow_time = profile.total_duration or 0
            total_workflow_time += workflow_time

            for step in self._flatten_steps(profile.steps):
                if step.duration is not None:
                    step_stats[step.name]["count"] += 1
                    step_stats[step.name]["total_time"] += step.duration
                    step_stats[step.name]["times"].append(step.duration)

        # Analyze hot paths
        for step_name, stats in step_stats.items():
            time_percentage = (
                stats["total_time"] / total_workflow_time
                if total_workflow_time > 0
                else 0
            )

            if time_percentage > self.thresholds["hot_path_percentage"]:
                avg_time = stats["total_time"] / stats["count"]
                severity = self._classify_hotpath_severity(time_percentage)

                metrics = {
                    "execution_count": stats["count"],
                    "total_time": stats["total_time"],
                    "avg_time": avg_time,
                    "time_percentage": time_percentage * 100,
                    "frequency_per_workflow": stats["count"] / len(profiles),
                }

                recommendations = [
                    f"Optimize {step_name} as it consumes {time_percentage * 100:.1f}% of total time",
                    "Profile function calls within this component",
                    "Consider caching results if computation is repeated",
                    "Implement parallel processing if not already used",
                    "Look for opportunities to pre-compute or batch operations",
                ]

                finding = BottleneckFinding(
                    bottleneck_type=BottleneckType.HOT_PATH,
                    severity=severity,
                    component=step_name,
                    description=f"Hot path consuming {time_percentage * 100:.1f}% of execution time",
                    metrics=metrics,
                    affected_workflows=[p.session_id for p in profiles],
                    recommendations=recommendations,
                    confidence_score=min(0.95, stats["count"] / 50.0),
                    frequency=stats["count"],
                )

                findings.append(finding)

        return findings

    def _analyze_synchronization_issues(
        self, profiles: List[WorkflowProfile]
    ) -> List[BottleneckFinding]:
        """Detect synchronization-related performance issues."""
        findings = []

        # Look for steps that have high variance in execution time (indicating waiting)
        sync_candidates = []

        for profile in profiles:
            for step in self._flatten_steps(profile.steps):
                # Look for threading-related operations
                if any(
                    keyword in step.name.lower()
                    for keyword in ["lock", "wait", "sync", "thread", "queue", "pool"]
                ):
                    sync_candidates.append(step)

        # Group by step name and analyze variance
        sync_groups = defaultdict(list)
        for step in sync_candidates:
            if step.duration is not None:
                sync_groups[step.name].append(step.duration)

        for step_name, durations in sync_groups.items():
            if len(durations) < 3:
                continue

            avg_duration = statistics.mean(durations)
            std_duration = statistics.stdev(durations)
            coefficient_of_variation = (
                std_duration / avg_duration if avg_duration > 0 else 0
            )

            # High variance suggests synchronization issues
            if (
                avg_duration > self.thresholds["sync_wait_time"]
                and coefficient_of_variation > 0.5
            ):
                severity = self._classify_sync_severity(
                    avg_duration, coefficient_of_variation
                )

                metrics = {
                    "avg_duration": avg_duration,
                    "std_duration": std_duration,
                    "coefficient_of_variation": coefficient_of_variation,
                    "sample_count": len(durations),
                    "max_duration": max(durations),
                }

                recommendations = [
                    "Analyze lock contention in synchronization code",
                    "Consider lock-free data structures where appropriate",
                    "Optimize critical sections to reduce lock hold time",
                    "Use reader-writer locks if applicable",
                    "Profile thread wait times to identify bottlenecks",
                ]

                finding = BottleneckFinding(
                    bottleneck_type=BottleneckType.SYNCHRONIZATION,
                    severity=severity,
                    component=step_name,
                    description=f"Synchronization bottleneck with high variance (CV: {coefficient_of_variation:.2f})",
                    metrics=metrics,
                    affected_workflows=[p.session_id for p in profiles],
                    recommendations=recommendations,
                    confidence_score=min(0.7, len(durations) / 10.0),
                    frequency=len(durations),
                )

                findings.append(finding)

        return findings

    def _flatten_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Flatten nested workflow steps into a single list."""
        flattened = []
        for step in steps:
            flattened.append(step)
            if step.sub_steps:
                flattened.extend(self._flatten_steps(step.sub_steps))
        return flattened

    def _estimate_data_size(self, step: WorkflowStep) -> float:
        """Estimate data size for a workflow step based on parameters."""
        params = step.parameters

        # Look for common size indicators
        size_indicators = [
            "file_size",
            "data_size",
            "array_size",
            "num_points",
            "width",
            "height",
            "length",
            "count",
            "num_files",
        ]

        for indicator in size_indicators:
            if indicator in params:
                try:
                    return float(params[indicator])
                except (ValueError, TypeError):
                    continue

        # If width and height are available, calculate area
        if "width" in params and "height" in params:
            try:
                return float(params["width"]) * float(params["height"])
            except (ValueError, TypeError):
                pass

        return 0.0

    def _analyze_time_complexity(
        self, performance_data: List[Tuple[float, float]]
    ) -> float:
        """
        Analyze time complexity by fitting data to various complexity models.
        Returns a complexity score where higher values indicate worse complexity.
        """
        if len(performance_data) < 3:
            return 0.0

        sizes, times = zip(*performance_data)
        sizes = np.array(sizes)
        times = np.array(times)

        # Normalize data
        size_norm = sizes / np.max(sizes)
        time_norm = times / np.max(times)

        # Test different complexity models
        models = {
            "linear": size_norm,
            "quadratic": size_norm**2,
            "logarithmic": np.log(size_norm + 1),
            "nlogn": size_norm * np.log(size_norm + 1),
        }

        best_fit_score = 0.0
        worst_complexity = 1.0  # Linear complexity baseline

        for model_name, model_data in models.items():
            try:
                # Calculate correlation coefficient
                correlation = np.corrcoef(model_data, time_norm)[0, 1]
                if np.isnan(correlation):
                    continue

                fit_score = abs(correlation)

                # Assign complexity weights
                complexity_weights = {
                    "logarithmic": 0.5,
                    "linear": 1.0,
                    "nlogn": 2.0,
                    "quadratic": 4.0,
                }

                if fit_score > best_fit_score:
                    best_fit_score = fit_score
                    worst_complexity = complexity_weights.get(model_name, 1.0)

            except Exception as e:
                logger.debug(f"Error fitting {model_name} model: {e}")
                continue

        # Return complexity score (higher is worse)
        return worst_complexity * best_fit_score * 10

    # Classification methods for severity

    def _classify_duration_severity(self, duration: float) -> BottleneckSeverity:
        """Classify severity based on operation duration."""
        if duration > 30:
            return BottleneckSeverity.CRITICAL
        elif duration > 10:
            return BottleneckSeverity.HIGH
        elif duration > 5:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _classify_contention_severity(
        self, contention_ratio: float
    ) -> BottleneckSeverity:
        """Classify severity based on thread contention ratio."""
        if contention_ratio > 0.9:
            return BottleneckSeverity.CRITICAL
        elif contention_ratio > 0.8:
            return BottleneckSeverity.HIGH
        elif contention_ratio > 0.7:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _classify_memory_severity(self, memory_mb: float) -> BottleneckSeverity:
        """Classify severity based on memory allocation."""
        if memory_mb > 2000:
            return BottleneckSeverity.CRITICAL
        elif memory_mb > 1000:
            return BottleneckSeverity.HIGH
        elif memory_mb > 500:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _classify_complexity_severity(
        self, complexity_score: float
    ) -> BottleneckSeverity:
        """Classify severity based on algorithmic complexity score."""
        if complexity_score > 30:
            return BottleneckSeverity.CRITICAL
        elif complexity_score > 20:
            return BottleneckSeverity.HIGH
        elif complexity_score > 10:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _classify_hotpath_severity(self, time_percentage: float) -> BottleneckSeverity:
        """Classify severity based on time percentage consumed."""
        if time_percentage > 0.5:  # 50%
            return BottleneckSeverity.CRITICAL
        elif time_percentage > 0.3:  # 30%
            return BottleneckSeverity.HIGH
        elif time_percentage > 0.2:  # 20%
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _classify_sync_severity(self, duration: float, cv: float) -> BottleneckSeverity:
        """Classify severity based on synchronization metrics."""
        combined_score = duration * cv
        if combined_score > 5.0:
            return BottleneckSeverity.CRITICAL
        elif combined_score > 2.0:
            return BottleneckSeverity.HIGH
        elif combined_score > 1.0:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _generate_cpu_bound_recommendations(
        self, step_name: str, avg_duration: float, std_duration: float
    ) -> List[str]:
        """Generate recommendations for CPU-bound operations."""
        recommendations = []

        if "fit" in step_name.lower():
            recommendations.extend(
                [
                    "Use vectorized fitting algorithms (e.g., scipy.optimize)",
                    "Implement parallel fitting for multiple datasets",
                    "Consider approximate fitting methods for large datasets",
                ]
            )
        elif "load" in step_name.lower() or "read" in step_name.lower():
            recommendations.extend(
                [
                    "Use memory-mapped files for large datasets",
                    "Implement lazy loading strategies",
                    "Use HDF5 chunking optimization",
                ]
            )
        elif "plot" in step_name.lower() or "visual" in step_name.lower():
            recommendations.extend(
                [
                    "Use data decimation for large datasets",
                    "Implement progressive rendering",
                    "Cache rendered plots when possible",
                ]
            )
        elif "process" in step_name.lower() or "compute" in step_name.lower():
            recommendations.extend(
                [
                    "Use NumPy vectorized operations",
                    "Implement multiprocessing for CPU-intensive tasks",
                    "Consider using numba JIT compilation",
                ]
            )

        # General recommendations
        recommendations.extend(
            [
                f"Profile {step_name} to identify specific bottlenecks",
                "Consider caching results if computation is repeated",
                "Use appropriate data structures for the operation",
            ]
        )

        return recommendations

    def get_bottleneck_summary(
        self, findings: List[BottleneckFinding]
    ) -> Dict[str, Any]:
        """Generate a summary of bottleneck findings."""
        if not findings:
            return {"total_findings": 0}

        # Count by type and severity
        type_counts = Counter(f.bottleneck_type.value for f in findings)
        severity_counts = Counter(f.severity.value for f in findings)

        # Calculate aggregate metrics
        total_affected_workflows = set()
        for finding in findings:
            total_affected_workflows.update(finding.affected_workflows)

        # Find most problematic components
        component_issues = defaultdict(int)
        for finding in findings:
            component_issues[finding.component] += 1

        top_components = sorted(
            component_issues.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_findings": len(findings),
            "by_type": dict(type_counts),
            "by_severity": dict(severity_counts),
            "affected_workflows": len(total_affected_workflows),
            "top_problematic_components": top_components,
            "critical_count": sum(
                1 for f in findings if f.severity == BottleneckSeverity.CRITICAL
            ),
            "high_priority_count": sum(
                1
                for f in findings
                if f.severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]
            ),
            "avg_confidence_score": statistics.mean(
                f.confidence_score for f in findings
            ),
        }

    def export_findings_report(self, findings: List[BottleneckFinding], file_path: str):
        """Export bottleneck findings to a detailed report file."""
        report_data = {
            "analysis_timestamp": time.time(),
            "summary": self.get_bottleneck_summary(findings),
            "findings": [f.to_dict() for f in findings],
        }

        import json

        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Exported bottleneck analysis report to {file_path}")


# Global analyzer instance
cpu_bottleneck_analyzer = CPUBottleneckAnalyzer()


def get_bottleneck_analyzer() -> CPUBottleneckAnalyzer:
    """
    Get the global CPU bottleneck analyzer instance.

    Returns
    -------
    CPUBottleneckAnalyzer
        Global bottleneck analyzer instance
    """
    return cpu_bottleneck_analyzer


# Convenience functions


def analyze_recent_workflows(
    count: int = 20, workflow_type: Optional[str] = None
) -> List[BottleneckFinding]:
    """
    Analyze recent workflows for CPU bottlenecks.

    Args:
        count: Number of recent workflows to analyze
        workflow_type: Optional filter by workflow type

    Returns:
        List of BottleneckFinding objects
    """
    profiles = workflow_profiler.get_recent_profiles(count, workflow_type)
    return cpu_bottleneck_analyzer.analyze_workflow_profiles(profiles)


def get_bottleneck_report() -> Dict[str, Any]:
    """Get a summary report of recent CPU bottlenecks."""
    findings = analyze_recent_workflows()
    return cpu_bottleneck_analyzer.get_bottleneck_summary(findings)


def find_workflow_hotspots(
    workflow_type: str, count: int = 10
) -> List[BottleneckFinding]:
    """
    Find performance hotspots for a specific workflow type.

    Args:
        workflow_type: Type of workflow to analyze
        count: Number of recent workflows to include

    Returns:
        List of hotspot findings
    """
    findings = analyze_recent_workflows(count, workflow_type)
    return [f for f in findings if f.bottleneck_type == BottleneckType.HOT_PATH]
