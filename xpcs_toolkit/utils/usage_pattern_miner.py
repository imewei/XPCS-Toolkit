"""
Usage pattern mining system for XPCS Toolkit workflow optimization.

This module analyzes user behavior patterns and data access patterns to identify
opportunities for cache optimization, preloading strategies, and automatic
performance tuning.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from .logging_config import get_logger
from .workflow_profiler import WorkflowProfile, WorkflowStep, workflow_profiler

logger = get_logger(__name__)


class AccessPattern(NamedTuple):
    """Represents a data access pattern."""

    resource_id: str
    access_count: int
    first_access: float
    last_access: float
    avg_interval: float
    access_sequence: list[float]


class WorkflowPattern(NamedTuple):
    """Represents a workflow usage pattern."""

    workflow_type: str
    frequency: int
    avg_duration: float
    common_steps: list[str]
    common_parameters: dict[str, Any]
    typical_file_sizes: list[float]
    peak_hours: list[int]


@dataclass
class UsagePattern:
    """General usage pattern discovered from workflow analysis."""

    pattern_type: str
    pattern_id: str
    description: str
    confidence: float
    estimated_improvement: float
    frequency: int
    detection_count: int
    first_detected: float
    last_seen: float
    affected_workflows: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheOptimization:
    """Cache optimization recommendation."""

    resource_type: str
    cache_size_mb: float
    hit_rate_improvement: float
    priority: int
    description: str
    implementation_notes: list[str] = field(default_factory=list)


@dataclass
class PreloadingRecommendation:
    """Preloading recommendation based on usage patterns."""

    resource_pattern: str
    trigger_condition: str
    predicted_benefit: float
    confidence: float
    description: str


@dataclass
class ThreadPoolRecommendation:
    """Thread pool optimization recommendation."""

    workflow_type: str
    recommended_pool_size: int
    current_average_threads: float
    utilization_improvement: float
    reasoning: str


class UsagePatternMiner:
    """
    Analyzes user behavior and data access patterns for optimization opportunities.

    This class examines workflow histories to identify:
    - Data access patterns for cache optimization
    - User behavior patterns for preloading
    - Resource utilization patterns for thread pool tuning
    - Seasonal/temporal usage patterns
    """

    def __init__(self, max_history_days: int = 30):
        self.max_history_days = max_history_days
        self.access_history: dict[str, list[tuple[float, str]]] = defaultdict(
            list
        )  # resource_id -> [(timestamp, workflow_id)]
        self.workflow_sequences: list[
            tuple[str, float, str]
        ] = []  # (workflow_type, timestamp, session_id)
        self.file_access_patterns: dict[str, AccessPattern] = {}
        self.workflow_patterns: dict[str, WorkflowPattern] = {}

        # Pattern analysis cache
        self._pattern_cache = {}
        self._cache_timestamp = 0
        self._cache_expiry = 300  # 5 minutes

        # Lock for thread safety
        self._lock = threading.RLock()

        logger.info("UsagePatternMiner initialized")

    def analyze_usage_patterns(self, profiles: list[WorkflowProfile]) -> dict[str, Any]:
        """
        Analyze usage patterns from workflow profiles.

        Args:
            profiles: List of WorkflowProfile objects to analyze

        Returns:
            Dictionary containing various usage pattern analyses
        """
        with self._lock:
            self._update_access_history(profiles)
            self._update_workflow_sequences(profiles)

            # Perform different pattern analyses
            patterns = {
                "temporal_patterns": self._analyze_temporal_patterns(profiles),
                "data_access_patterns": self._analyze_data_access_patterns(profiles),
                "workflow_sequences": self._analyze_workflow_sequences(profiles),
                "resource_utilization": self._analyze_resource_utilization_patterns(
                    profiles
                ),
                "file_size_patterns": self._analyze_file_size_patterns(profiles),
                "user_behavior_clusters": self._analyze_user_behavior_clusters(
                    profiles
                ),
            }

            logger.info("Completed usage pattern analysis")
            return patterns

    def _update_access_history(self, profiles: list[WorkflowProfile]):
        """Update access history with new workflow profiles."""
        current_time = time.time()
        cutoff_time = current_time - (self.max_history_days * 24 * 3600)

        for profile in profiles:
            # Extract file access information
            for step in self._flatten_steps(profile.steps):
                # Look for file-related parameters
                if "file_path" in step.parameters:
                    file_path = step.parameters["file_path"]
                    self.access_history[file_path].append(
                        (step.start_time, profile.session_id)
                    )

                if "file_list" in step.parameters:
                    for file_path in step.parameters["file_list"]:
                        self.access_history[file_path].append(
                            (step.start_time, profile.session_id)
                        )

        # Clean old access history
        for resource_id in list(self.access_history.keys()):
            self.access_history[resource_id] = [
                (ts, sid)
                for ts, sid in self.access_history[resource_id]
                if ts > cutoff_time
            ]
            if not self.access_history[resource_id]:
                del self.access_history[resource_id]

    def _update_workflow_sequences(self, profiles: list[WorkflowProfile]):
        """Update workflow sequences for pattern analysis."""
        current_time = time.time()
        cutoff_time = current_time - (self.max_history_days * 24 * 3600)

        for profile in profiles:
            self.workflow_sequences.append(
                (profile.workflow_type, profile.start_time, profile.session_id)
            )

        # Clean old sequences
        self.workflow_sequences = [
            (wtype, ts, sid)
            for wtype, ts, sid in self.workflow_sequences
            if ts > cutoff_time
        ]

        # Sort by timestamp
        self.workflow_sequences.sort(key=lambda x: x[1])

    def _analyze_temporal_patterns(
        self, profiles: list[WorkflowProfile]
    ) -> dict[str, Any]:
        """Analyze temporal usage patterns."""
        if not profiles:
            return {}

        # Group by hour of day and day of week
        hourly_usage = defaultdict(int)
        daily_usage = defaultdict(int)
        workflow_hours = defaultdict(list)

        for profile in profiles:
            dt = datetime.fromtimestamp(profile.start_time)
            hour = dt.hour
            day = dt.weekday()  # Monday = 0, Sunday = 6

            hourly_usage[hour] += 1
            daily_usage[day] += 1
            workflow_hours[profile.workflow_type].append(hour)

        # Find peak usage times
        peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_days = sorted(daily_usage.items(), key=lambda x: x[1], reverse=True)[:3]

        # Analyze workflow-specific temporal patterns
        workflow_temporal = {}
        for workflow_type, hours in workflow_hours.items():
            if len(hours) >= 3:
                hour_counter = Counter(hours)
                peak_workflow_hours = [h for h, _ in hour_counter.most_common(3)]
                workflow_temporal[workflow_type] = {
                    "peak_hours": peak_workflow_hours,
                    "avg_hour": statistics.mean(hours),
                    "hour_std": statistics.stdev(hours) if len(hours) > 1 else 0,
                }

        return {
            "peak_hours": [h for h, _ in peak_hours],
            "peak_days": [d for d, _ in peak_days],
            "hourly_distribution": dict(hourly_usage),
            "daily_distribution": dict(daily_usage),
            "workflow_temporal_patterns": workflow_temporal,
            "total_sessions": len(profiles),
        }

    def _analyze_data_access_patterns(
        self, profiles: list[WorkflowProfile]
    ) -> dict[str, Any]:
        """Analyze data access patterns for cache optimization."""
        access_patterns = {}

        # Build access patterns for each resource
        for resource_id, accesses in self.access_history.items():
            if len(accesses) < 2:
                continue

            timestamps = [ts for ts, _ in accesses]
            timestamps.sort()

            # Calculate access intervals
            intervals = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]

            pattern = AccessPattern(
                resource_id=resource_id,
                access_count=len(accesses),
                first_access=timestamps[0],
                last_access=timestamps[-1],
                avg_interval=statistics.mean(intervals) if intervals else 0,
                access_sequence=timestamps,
            )

            access_patterns[resource_id] = pattern

        # Categorize access patterns
        frequent_access = []  # High frequency, short intervals
        bursty_access = []  # High frequency in short time periods

        for resource_id, pattern in access_patterns.items():
            # Frequent access: accessed more than 5 times with avg interval < 1 hour
            if pattern.access_count >= 5 and pattern.avg_interval < 3600:
                frequent_access.append(resource_id)

            # Bursty access: multiple accesses in short time periods
            timestamps = pattern.access_sequence
            if len(timestamps) >= 3:
                # Find time windows with high activity
                window_size = 300  # 5 minutes
                for i in range(len(timestamps) - 2):
                    window_end = timestamps[i] + window_size
                    window_accesses = sum(
                        1 for ts in timestamps[i:] if ts <= window_end
                    )
                    if window_accesses >= 3:
                        if resource_id not in bursty_access:
                            bursty_access.append(resource_id)
                        break

        # Find sequential access patterns
        sequential_patterns = self._find_sequential_access_patterns(profiles)

        return {
            "total_resources": len(access_patterns),
            "frequent_access_resources": frequent_access,
            "bursty_access_resources": bursty_access,
            "sequential_access_patterns": sequential_patterns,
            "access_pattern_details": {
                k: v._asdict() for k, v in access_patterns.items()
            },
        }

    def _find_sequential_access_patterns(
        self, profiles: list[WorkflowProfile]
    ) -> list[dict[str, Any]]:
        """Find patterns where files are accessed in sequence."""
        sequential_patterns = []

        for profile in profiles:
            file_sequences = []
            current_sequence = []

            for step in self._flatten_steps(profile.steps):
                if "file_path" in step.parameters:
                    file_path = step.parameters["file_path"]
                    current_sequence.append((step.start_time, file_path))
                elif "file_list" in step.parameters:
                    # End current sequence and start analyzing
                    if len(current_sequence) >= 2:
                        file_sequences.append(current_sequence)
                    current_sequence = []

                    # Add the file list as a sequence
                    for i, file_path in enumerate(step.parameters["file_list"]):
                        current_sequence.append((step.start_time + i * 0.01, file_path))

            if len(current_sequence) >= 2:
                file_sequences.append(current_sequence)

            # Analyze sequences for patterns
            for sequence in file_sequences:
                if len(sequence) >= 3:
                    files = [fp for _, fp in sequence]
                    pattern_info = {
                        "workflow_session": profile.session_id,
                        "workflow_type": profile.workflow_type,
                        "sequence_length": len(files),
                        "files": files,
                        "time_span": sequence[-1][0] - sequence[0][0],
                    }
                    sequential_patterns.append(pattern_info)

        return sequential_patterns

    def _analyze_workflow_sequences(
        self, profiles: list[WorkflowProfile]
    ) -> dict[str, Any]:
        """Analyze sequences of workflow types for prediction."""
        if len(self.workflow_sequences) < 5:
            return {"insufficient_data": True}

        # Find common workflow transitions
        transitions = defaultdict(int)
        workflow_gaps = defaultdict(list)

        for i in range(len(self.workflow_sequences) - 1):
            current = self.workflow_sequences[i]
            next_workflow = self.workflow_sequences[i + 1]

            transition = (current[0], next_workflow[0])
            transitions[transition] += 1

            # Calculate time gap between workflows
            gap = next_workflow[1] - current[1]
            workflow_gaps[transition].append(gap)

        # Find most common transitions
        common_transitions = sorted(
            transitions.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Calculate average gaps for transitions
        transition_timings = {}
        for (from_wf, to_wf), gaps in workflow_gaps.items():
            if len(gaps) >= 2:
                transition_timings[(from_wf, to_wf)] = {
                    "avg_gap": statistics.mean(gaps),
                    "median_gap": statistics.median(gaps),
                    "min_gap": min(gaps),
                    "max_gap": max(gaps),
                }

        return {
            "common_transitions": common_transitions,
            "transition_timings": transition_timings,
            "total_transitions": len(transitions),
            "unique_transitions": len(transitions),
        }

    def _analyze_resource_utilization_patterns(
        self, profiles: list[WorkflowProfile]
    ) -> dict[str, Any]:
        """Analyze resource utilization patterns for thread pool optimization."""
        utilization_by_workflow = defaultdict(list)

        for profile in profiles:
            # Analyze thread usage patterns from steps
            thread_counts = []
            durations = []

            for step in self._flatten_steps(profile.steps):
                if step.thread_count is not None and step.duration is not None:
                    thread_counts.append(step.thread_count)
                    durations.append(step.duration)

            if thread_counts and durations:
                # Weight thread count by duration
                weighted_avg_threads = sum(
                    tc * dur for tc, dur in zip(thread_counts, durations, strict=False)
                ) / sum(durations)
                max_threads = max(thread_counts)

                utilization_by_workflow[profile.workflow_type].append(
                    {
                        "weighted_avg_threads": weighted_avg_threads,
                        "max_threads": max_threads,
                        "total_duration": sum(durations),
                    }
                )

        # Aggregate utilization statistics
        utilization_stats = {}
        for workflow_type, utilizations in utilization_by_workflow.items():
            if utilizations:
                avg_weighted_threads = statistics.mean(
                    [u["weighted_avg_threads"] for u in utilizations]
                )
                avg_max_threads = statistics.mean(
                    [u["max_threads"] for u in utilizations]
                )
                total_time = sum([u["total_duration"] for u in utilizations])

                utilization_stats[workflow_type] = {
                    "avg_weighted_threads": avg_weighted_threads,
                    "avg_max_threads": avg_max_threads,
                    "total_execution_time": total_time,
                    "sample_count": len(utilizations),
                }

        return utilization_stats

    def _analyze_file_size_patterns(
        self, profiles: list[WorkflowProfile]
    ) -> dict[str, Any]:
        """Analyze file size patterns for memory optimization."""
        file_sizes_by_workflow = defaultdict(list)

        for profile in profiles:
            for step in self._flatten_steps(profile.steps):
                if "file_size" in step.parameters:
                    size_mb = step.parameters["file_size"] / (
                        1024 * 1024
                    )  # Convert to MB
                    file_sizes_by_workflow[profile.workflow_type].append(size_mb)

        size_patterns = {}
        for workflow_type, sizes in file_sizes_by_workflow.items():
            if sizes:
                size_patterns[workflow_type] = {
                    "avg_size_mb": statistics.mean(sizes),
                    "median_size_mb": statistics.median(sizes),
                    "min_size_mb": min(sizes),
                    "max_size_mb": max(sizes),
                    "size_std_mb": statistics.stdev(sizes) if len(sizes) > 1 else 0,
                    "total_data_processed_gb": sum(sizes) / 1024,
                    "file_count": len(sizes),
                }

        return size_patterns

    def _analyze_user_behavior_clusters(
        self, profiles: list[WorkflowProfile]
    ) -> dict[str, Any]:
        """Analyze user behavior to identify common usage clusters."""
        # Group profiles by similar characteristics
        behavior_vectors = []
        profile_info = []

        for profile in profiles:
            # Create behavior vector
            vector = [
                profile.total_duration or 0,
                len(profile.steps),
                datetime.fromtimestamp(profile.start_time).hour,
                len(
                    [
                        s
                        for s in self._flatten_steps(profile.steps)
                        if s.memory_delta and s.memory_delta > 0
                    ]
                ),
            ]

            behavior_vectors.append(vector)
            profile_info.append(
                {
                    "session_id": profile.session_id,
                    "workflow_type": profile.workflow_type,
                    "start_time": profile.start_time,
                }
            )

        if len(behavior_vectors) < 5:
            return {"insufficient_data": True}

        # Simple clustering based on workflow type and duration
        clusters = defaultdict(list)

        for _i, (vector, info) in enumerate(
            zip(behavior_vectors, profile_info, strict=False)
        ):
            # Create cluster key based on workflow type and duration ranges
            duration = vector[0]
            if duration < 10:
                duration_bucket = "fast"
            elif duration < 60:
                duration_bucket = "medium"
            else:
                duration_bucket = "slow"

            cluster_key = f"{info['workflow_type']}_{duration_bucket}"
            clusters[cluster_key].append(
                {
                    "profile_info": info,
                    "behavior_vector": vector,
                }
            )

        # Analyze clusters
        cluster_analysis = {}
        for cluster_key, cluster_profiles in clusters.items():
            if len(cluster_profiles) >= 2:
                vectors = [cp["behavior_vector"] for cp in cluster_profiles]

                cluster_analysis[cluster_key] = {
                    "size": len(cluster_profiles),
                    "avg_duration": statistics.mean([v[0] for v in vectors]),
                    "avg_steps": statistics.mean([v[1] for v in vectors]),
                    "common_hour": round(statistics.mean([v[2] for v in vectors])),
                    "memory_intensive_steps": statistics.mean([v[3] for v in vectors]),
                }

        return {
            "clusters": cluster_analysis,
            "total_clusters": len(cluster_analysis),
            "clustered_profiles": sum(len(cp) for cp in clusters.values()),
        }

    def generate_cache_optimization_recommendations(
        self, patterns: dict[str, Any]
    ) -> list[CacheOptimization]:
        """Generate cache optimization recommendations based on usage patterns."""
        recommendations = []

        data_access = patterns.get("data_access_patterns", {})
        frequent_resources = data_access.get("frequent_access_resources", [])

        if frequent_resources:
            # Recommend metadata caching for frequently accessed files
            total_frequent = len(frequent_resources)
            estimated_cache_size = min(
                100, total_frequent * 0.5
            )  # 0.5MB per file metadata

            recommendations.append(
                CacheOptimization(
                    resource_type="file_metadata",
                    cache_size_mb=estimated_cache_size,
                    hit_rate_improvement=0.6,  # Estimated 60% improvement
                    priority=1,
                    description=f"Cache metadata for {total_frequent} frequently accessed files",
                    implementation_notes=[
                        "Use LRU cache for file metadata (headers, dimensions, etc.)",
                        "Cache should persist across workflow sessions",
                        "Monitor cache hit rates and adjust size accordingly",
                    ],
                )
            )

        # Recommend computation result caching for repeated operations
        if "workflow_sequences" in patterns:
            common_transitions = patterns["workflow_sequences"].get(
                "common_transitions", []
            )
            if common_transitions:
                recommendations.append(
                    CacheOptimization(
                        resource_type="computation_results",
                        cache_size_mb=200,
                        hit_rate_improvement=0.4,
                        priority=2,
                        description="Cache intermediate computation results for repeated workflows",
                        implementation_notes=[
                            "Cache G2 fitting results with parameter-based keys",
                            "Cache SAXS processing results for identical parameters",
                            "Implement cache invalidation for parameter changes",
                        ],
                    )
                )

        return recommendations

    def generate_preloading_recommendations(
        self, patterns: dict[str, Any]
    ) -> list[PreloadingRecommendation]:
        """Generate preloading recommendations based on usage patterns."""
        recommendations = []

        # Sequential access pattern preloading
        data_access = patterns.get("data_access_patterns", {})
        sequential_patterns = data_access.get("sequential_access_patterns", [])

        if sequential_patterns:
            # Find common file sequences
            sequence_counts = defaultdict(int)
            for pattern in sequential_patterns:
                if len(pattern["files"]) >= 3:
                    # Create pattern from first few files
                    file_pattern = tuple(Path(f).suffix for f in pattern["files"][:3])
                    sequence_counts[file_pattern] += 1

            common_sequences = [
                seq for seq, count in sequence_counts.items() if count >= 2
            ]

            if common_sequences:
                recommendations.append(
                    PreloadingRecommendation(
                        resource_pattern="sequential_file_access",
                        trigger_condition="After loading first file in sequence",
                        predicted_benefit=0.3,  # 30% improvement in load times
                        confidence=0.7,
                        description=f"Preload next files in sequence based on {len(common_sequences)} common patterns",
                    )
                )

        # Workflow transition preloading
        if "workflow_sequences" in patterns:
            transitions = patterns["workflow_sequences"].get("common_transitions", [])
            timings = patterns["workflow_sequences"].get("transition_timings", {})

            for (from_wf, to_wf), count in transitions[:5]:  # Top 5 transitions
                if count >= 3 and (from_wf, to_wf) in timings:
                    timing_info = timings[(from_wf, to_wf)]
                    if timing_info["avg_gap"] > 60:  # At least 1 minute gap
                        recommendations.append(
                            PreloadingRecommendation(
                                resource_pattern=f"{from_wf}_to_{to_wf}",
                                trigger_condition=f"Near completion of {from_wf} workflow",
                                predicted_benefit=0.2,
                                confidence=min(0.9, count / 10.0),
                                description=f"Preload resources for {to_wf} during {from_wf} execution",
                            )
                        )

        return recommendations

    def generate_thread_pool_recommendations(
        self, patterns: dict[str, Any]
    ) -> list[ThreadPoolRecommendation]:
        """Generate thread pool optimization recommendations."""
        recommendations = []

        utilization_stats = patterns.get("resource_utilization", {})

        for workflow_type, stats in utilization_stats.items():
            if stats["sample_count"] >= 3:
                current_avg = stats["avg_weighted_threads"]
                max_threads = stats["avg_max_threads"]

                # Recommend optimal thread pool size
                if max_threads > current_avg * 2:
                    # Thread pool might be oversized
                    recommended_size = max(2, int(current_avg * 1.5))
                    recommendations.append(
                        ThreadPoolRecommendation(
                            workflow_type=workflow_type,
                            recommended_pool_size=recommended_size,
                            current_average_threads=current_avg,
                            utilization_improvement=0.15,
                            reasoning=f"Reduce pool size from ~{int(max_threads)} to {recommended_size} based on actual usage",
                        )
                    )
                elif current_avg > max_threads * 0.8:
                    # Thread pool might be undersized
                    recommended_size = int(current_avg * 1.2)
                    recommendations.append(
                        ThreadPoolRecommendation(
                            workflow_type=workflow_type,
                            recommended_pool_size=recommended_size,
                            current_average_threads=current_avg,
                            utilization_improvement=0.25,
                            reasoning=f"Increase pool size to {recommended_size} to reduce thread contention",
                        )
                    )

        return recommendations

    def _flatten_steps(self, steps: list[WorkflowStep]) -> list[WorkflowStep]:
        """Flatten nested workflow steps."""
        flattened = []
        for step in steps:
            flattened.append(step)
            if step.sub_steps:
                flattened.extend(self._flatten_steps(step.sub_steps))
        return flattened

    def get_pattern_summary(self, patterns: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of identified usage patterns."""
        summary = {
            "analysis_timestamp": time.time(),
            "temporal_insights": {},
            "data_access_insights": {},
            "resource_utilization_insights": {},
            "optimization_opportunities": 0,
        }

        # Temporal insights
        temporal = patterns.get("temporal_patterns", {})
        if temporal:
            summary["temporal_insights"] = {
                "peak_usage_hours": temporal.get("peak_hours", []),
                "total_sessions": temporal.get("total_sessions", 0),
                "workflow_temporal_patterns": len(
                    temporal.get("workflow_temporal_patterns", {})
                ),
            }

        # Data access insights
        data_access = patterns.get("data_access_patterns", {})
        if data_access:
            summary["data_access_insights"] = {
                "frequent_access_files": len(
                    data_access.get("frequent_access_resources", [])
                ),
                "bursty_access_files": len(
                    data_access.get("bursty_access_resources", [])
                ),
                "sequential_patterns": len(
                    data_access.get("sequential_access_patterns", [])
                ),
            }

        # Resource utilization insights
        resource_util = patterns.get("resource_utilization", {})
        if resource_util:
            summary["resource_utilization_insights"] = {
                "analyzed_workflow_types": len(resource_util),
                "avg_thread_utilization": statistics.mean(
                    [stats["avg_weighted_threads"] for stats in resource_util.values()]
                )
                if resource_util
                else 0,
            }

        # Count optimization opportunities
        summary["optimization_opportunities"] = (
            len(data_access.get("frequent_access_resources", []))
            + len(data_access.get("sequential_access_patterns", []))
            + len(resource_util)
        )

        return summary

    def clear_old_patterns(self):
        """Clear old pattern data to free memory."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (self.max_history_days * 24 * 3600)

            # Clear old access history
            for resource_id in list(self.access_history.keys()):
                self.access_history[resource_id] = [
                    (ts, sid)
                    for ts, sid in self.access_history[resource_id]
                    if ts > cutoff_time
                ]
                if not self.access_history[resource_id]:
                    del self.access_history[resource_id]

            # Clear old workflow sequences
            self.workflow_sequences = [
                (wtype, ts, sid)
                for wtype, ts, sid in self.workflow_sequences
                if ts > cutoff_time
            ]

            # Clear pattern cache
            self._pattern_cache.clear()

            logger.info("Cleared old usage pattern data")


# Global pattern miner instance
usage_pattern_miner = UsagePatternMiner()


def get_pattern_miner() -> UsagePatternMiner:
    """
    Get the global usage pattern miner instance.

    Returns
    -------
    UsagePatternMiner
        Global usage pattern miner instance
    """
    return usage_pattern_miner


# Convenience functions


def analyze_current_usage_patterns() -> dict[str, Any]:
    """Analyze current usage patterns from recent workflows."""
    profiles = workflow_profiler.get_recent_profiles(50)  # Last 50 workflows
    return usage_pattern_miner.analyze_usage_patterns(profiles)


def get_optimization_recommendations() -> dict[str, list]:
    """Get all optimization recommendations based on current usage patterns."""
    patterns = analyze_current_usage_patterns()

    return {
        "cache_optimizations": usage_pattern_miner.generate_cache_optimization_recommendations(
            patterns
        ),
        "preloading_recommendations": usage_pattern_miner.generate_preloading_recommendations(
            patterns
        ),
        "thread_pool_recommendations": usage_pattern_miner.generate_thread_pool_recommendations(
            patterns
        ),
        "pattern_summary": usage_pattern_miner.get_pattern_summary(patterns),
    }


def get_data_access_insights(workflow_type: str | None = None) -> dict[str, Any]:
    """Get insights about data access patterns for a specific workflow type."""
    profiles = workflow_profiler.get_recent_profiles(30, workflow_type)
    if not profiles:
        return {}

    patterns = usage_pattern_miner.analyze_usage_patterns(profiles)
    return patterns.get("data_access_patterns", {})


def get_temporal_usage_insights() -> dict[str, Any]:
    """Get insights about temporal usage patterns."""
    profiles = workflow_profiler.get_recent_profiles(100)
    if not profiles:
        return {}

    patterns = usage_pattern_miner.analyze_usage_patterns(profiles)
    return patterns.get("temporal_patterns", {})
