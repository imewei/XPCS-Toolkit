#!/usr/bin/env python3
"""
Benchmark Database System for XPCS Toolkit Historical Performance Tracking

This module provides comprehensive historical CPU performance data tracking and storage,
performance trend analysis and visualization, baseline management for different system
configurations, and performance report generation with export capabilities.

Features:
- SQLite-based performance database with efficient indexing
- Historical performance data tracking across multiple dimensions
- Performance trend analysis with statistical modeling
- Baseline management for different system configurations
- Comprehensive reporting and visualization capabilities
- Data export in multiple formats (JSON, CSV, HTML reports)
- Integration with existing monitoring and regression detection systems
- Performance data aggregation and summarization
- Automated data retention and cleanup policies

Integration Points:
- Works with CPU performance test suite for data ingestion
- Integrates with regression detector for baseline management
- Connects to monitoring systems for real-time data
- Supports performance profiler data import/export

Author: Claude Code Performance Testing Generator
Date: 2025-01-11
"""

from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PerformanceRecord:
    """Individual performance measurement record."""

    id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    test_name: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    git_commit: Optional[str] = None
    branch_name: Optional[str] = None
    system_config: Dict[str, Any] = field(default_factory=dict)
    test_config: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfiguration:
    """System configuration for performance measurements."""

    id: Optional[int] = None
    config_name: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    memory_total_gb: float = 0.0
    python_version: str = ""
    os_platform: str = ""
    numpy_version: str = ""
    scipy_version: str = ""
    config_hash: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""

    metric_name: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0-1 scale
    slope: float
    r_squared: float
    p_value: float
    data_points: int
    time_span_days: float
    projected_change_30days: float
    statistical_significance: bool


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark results."""

    metric_name: str
    test_name: str
    time_period: str
    sample_count: int
    mean_value: float
    median_value: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    trend_analysis: Optional[PerformanceTrend] = None


# =============================================================================
# Benchmark Database Class
# =============================================================================


class BenchmarkDatabase:
    """Comprehensive benchmark database for performance tracking."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize benchmark database."""
        self.db_path = Path(db_path) if db_path else Path("benchmark_performance.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()

        # Configure matplotlib for headless environments
        plt.switch_backend("Agg")

        logger.info(f"BenchmarkDatabase initialized with database: {self.db_path}")

    def _initialize_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # System configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_configurations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_name TEXT NOT NULL,
                    cpu_model TEXT,
                    cpu_cores INTEGER,
                    cpu_threads INTEGER,
                    memory_total_gb REAL,
                    python_version TEXT,
                    os_platform TEXT,
                    numpy_version TEXT,
                    scipy_version TEXT,
                    config_hash TEXT UNIQUE,
                    created_at REAL,
                    UNIQUE(config_name, config_hash)
                )
            """)

            # Performance records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    test_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    git_commit TEXT,
                    branch_name TEXT,
                    system_config_id INTEGER,
                    test_config_json TEXT,
                    environment_info_json TEXT,
                    additional_metadata_json TEXT,
                    FOREIGN KEY (system_config_id) REFERENCES system_configurations (id)
                )
            """)

            # Baselines table for regression detection
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    baseline_std REAL,
                    sample_size INTEGER,
                    confidence_interval_lower REAL,
                    confidence_interval_upper REAL,
                    system_config_id INTEGER,
                    created_at REAL,
                    git_commit TEXT,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (system_config_id) REFERENCES system_configurations (id)
                )
            """)

            # Trend analysis cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trend_analysis_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    analysis_period_days INTEGER,
                    trend_direction TEXT,
                    trend_strength REAL,
                    slope REAL,
                    r_squared REAL,
                    p_value REAL,
                    data_points INTEGER,
                    projected_change_30days REAL,
                    calculated_at REAL,
                    expires_at REAL
                )
            """)

            # Create indexes for performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_records (timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_metric ON performance_records (metric_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_test ON performance_records (test_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_commit ON performance_records (git_commit)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_baseline_metric ON performance_baselines (metric_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_baseline_active ON performance_baselines (is_active)"
            )

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def store_system_config(self, config: SystemConfiguration) -> int:
        """Store system configuration and return its ID."""

        # Generate config hash
        config_data = {
            "cpu_model": config.cpu_model,
            "cpu_cores": config.cpu_cores,
            "cpu_threads": config.cpu_threads,
            "memory_total_gb": config.memory_total_gb,
            "python_version": config.python_version,
            "os_platform": config.os_platform,
            "numpy_version": config.numpy_version,
            "scipy_version": config.scipy_version,
        }
        config_hash = hash(str(sorted(config_data.items())))
        config.config_hash = str(config_hash)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if config already exists
            cursor.execute(
                "SELECT id FROM system_configurations WHERE config_hash = ?",
                (config.config_hash,),
            )
            existing = cursor.fetchone()

            if existing:
                return existing["id"]

            # Insert new config
            cursor.execute(
                """
                INSERT INTO system_configurations 
                (config_name, cpu_model, cpu_cores, cpu_threads, memory_total_gb,
                 python_version, os_platform, numpy_version, scipy_version, 
                 config_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    config.config_name,
                    config.cpu_model,
                    config.cpu_cores,
                    config.cpu_threads,
                    config.memory_total_gb,
                    config.python_version,
                    config.os_platform,
                    config.numpy_version,
                    config.scipy_version,
                    config.config_hash,
                    config.created_at,
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def store_performance_record(self, record: PerformanceRecord) -> int:
        """Store performance record and return its ID."""

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO performance_records 
                (timestamp, test_name, metric_name, metric_value, metric_unit,
                 git_commit, branch_name, system_config_id, test_config_json,
                 environment_info_json, additional_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.timestamp,
                    record.test_name,
                    record.metric_name,
                    record.metric_value,
                    record.metric_unit,
                    record.git_commit,
                    record.branch_name,
                    None,  # system_config_id handled separately
                    json.dumps(record.test_config),
                    json.dumps(record.environment_info),
                    json.dumps(record.additional_metadata),
                ),
            )

            conn.commit()
            return cursor.lastrowid

    def batch_store_records(self, records: List[PerformanceRecord]) -> List[int]:
        """Efficiently store multiple performance records."""

        record_ids = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Prepare batch insert data
            insert_data = []
            for record in records:
                insert_data.append(
                    (
                        record.timestamp,
                        record.test_name,
                        record.metric_name,
                        record.metric_value,
                        record.metric_unit,
                        record.git_commit,
                        record.branch_name,
                        None,  # system_config_id
                        json.dumps(record.test_config),
                        json.dumps(record.environment_info),
                        json.dumps(record.additional_metadata),
                    )
                )

            # Batch insert
            cursor.executemany(
                """
                INSERT INTO performance_records 
                (timestamp, test_name, metric_name, metric_value, metric_unit,
                 git_commit, branch_name, system_config_id, test_config_json,
                 environment_info_json, additional_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                insert_data,
            )

            conn.commit()

            # Get the IDs of inserted records
            first_id = cursor.lastrowid - len(records) + 1
            record_ids = list(range(first_id, cursor.lastrowid + 1))

        logger.info(f"Stored {len(records)} performance records")
        return record_ids

    def query_performance_records(
        self,
        metric_name: Optional[str] = None,
        test_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        git_commit: Optional[str] = None,
        branch_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PerformanceRecord]:
        """Query performance records with flexible filtering."""

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query
            query = "SELECT * FROM performance_records WHERE 1=1"
            params = []

            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)

            if test_name:
                query += " AND test_name = ?"
                params.append(test_name)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if git_commit:
                query += " AND git_commit = ?"
                params.append(git_commit)

            if branch_name:
                query += " AND branch_name = ?"
                params.append(branch_name)

            query += " ORDER BY timestamp DESC"

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to PerformanceRecord objects
            records = []
            for row in rows:
                record = PerformanceRecord(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    test_name=row["test_name"],
                    metric_name=row["metric_name"],
                    metric_value=row["metric_value"],
                    metric_unit=row["metric_unit"] or "",
                    git_commit=row["git_commit"],
                    branch_name=row["branch_name"],
                    system_config=json.loads(row["test_config_json"] or "{}"),
                    test_config=json.loads(row["test_config_json"] or "{}"),
                    environment_info=json.loads(row["environment_info_json"] or "{}"),
                    additional_metadata=json.loads(
                        row["additional_metadata_json"] or "{}"
                    ),
                )
                records.append(record)

            return records

    def analyze_performance_trends(
        self, metric_name: str, days_back: int = 30, min_data_points: int = 10
    ) -> Optional[PerformanceTrend]:
        """Analyze performance trends for a specific metric."""

        # Check cache first
        cached_trend = self._get_cached_trend_analysis(metric_name, days_back)
        if cached_trend:
            return cached_trend

        # Get recent data
        start_time = time.time() - (days_back * 24 * 3600)
        records = self.query_performance_records(
            metric_name=metric_name, start_time=start_time
        )

        if len(records) < min_data_points:
            logger.warning(
                f"Insufficient data for trend analysis: {len(records)} < {min_data_points}"
            )
            return None

        # Extract data for analysis
        timestamps = [r.timestamp for r in records]
        values = [r.metric_value for r in records]

        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_data)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            timestamps, values
        )

        # Determine trend direction and strength
        r_squared = r_value**2
        trend_strength = abs(r_value)

        # Determine trend direction based on slope
        if abs(slope) < std_err:  # Slope not significantly different from zero
            trend_direction = "stable"
        elif slope < 0:
            trend_direction = "improving"  # Assuming lower values are better
        else:
            trend_direction = "degrading"

        # Calculate projected change over 30 days
        time_span_seconds = max(timestamps) - min(timestamps)
        projected_change_30days = slope * (30 * 24 * 3600)  # 30 days in seconds

        # Statistical significance test
        statistical_significance = p_value < 0.05

        trend = PerformanceTrend(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            data_points=len(records),
            time_span_days=time_span_seconds / (24 * 3600),
            projected_change_30days=projected_change_30days,
            statistical_significance=statistical_significance,
        )

        # Cache the result
        self._cache_trend_analysis(trend, days_back)

        return trend

    def _get_cached_trend_analysis(
        self, metric_name: str, days_back: int
    ) -> Optional[PerformanceTrend]:
        """Get cached trend analysis if available and not expired."""

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM trend_analysis_cache 
                WHERE metric_name = ? AND analysis_period_days = ? 
                AND expires_at > ?
                ORDER BY calculated_at DESC LIMIT 1
            """,
                (metric_name, days_back, time.time()),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return PerformanceTrend(
                metric_name=row["metric_name"],
                trend_direction=row["trend_direction"],
                trend_strength=row["trend_strength"],
                slope=row["slope"],
                r_squared=row["r_squared"],
                p_value=row["p_value"],
                data_points=row["data_points"],
                time_span_days=days_back,
                projected_change_30days=row["projected_change_30days"],
                statistical_significance=row["p_value"] < 0.05,
            )

    def _cache_trend_analysis(self, trend: PerformanceTrend, days_back: int):
        """Cache trend analysis results."""

        expires_at = time.time() + (3600 * 6)  # Cache for 6 hours

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO trend_analysis_cache
                (metric_name, analysis_period_days, trend_direction, trend_strength,
                 slope, r_squared, p_value, data_points, projected_change_30days,
                 calculated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trend.metric_name,
                    days_back,
                    trend.trend_direction,
                    trend.trend_strength,
                    trend.slope,
                    trend.r_squared,
                    trend.p_value,
                    trend.data_points,
                    trend.projected_change_30days,
                    time.time(),
                    expires_at,
                ),
            )

            conn.commit()

    def generate_benchmark_summary(
        self,
        metric_name: str,
        time_period: str = "30d",
        test_name: Optional[str] = None,
    ) -> BenchmarkSummary:
        """Generate comprehensive benchmark summary."""

        # Parse time period
        if time_period.endswith("d"):
            days_back = int(time_period[:-1])
        elif time_period.endswith("h"):
            days_back = int(time_period[:-1]) / 24
        else:
            days_back = 30  # Default

        start_time = time.time() - (days_back * 24 * 3600)

        # Get data
        records = self.query_performance_records(
            metric_name=metric_name, test_name=test_name, start_time=start_time
        )

        if not records:
            raise ValueError(
                f"No data found for metric {metric_name} in period {time_period}"
            )

        # Calculate statistics
        values = [r.metric_value for r in records]

        summary = BenchmarkSummary(
            metric_name=metric_name,
            test_name=test_name or "All Tests",
            time_period=time_period,
            sample_count=len(values),
            mean_value=statistics.mean(values),
            median_value=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            min_value=min(values),
            max_value=max(values),
            percentile_95=np.percentile(values, 95),
            percentile_99=np.percentile(values, 99),
        )

        # Add trend analysis
        trend = self.analyze_performance_trends(metric_name, int(days_back))
        summary.trend_analysis = trend

        return summary

    def create_performance_visualizations(
        self,
        metric_name: str,
        output_dir: Union[str, Path],
        days_back: int = 30,
        test_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create performance visualization charts."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get data
        start_time = time.time() - (days_back * 24 * 3600)
        records = self.query_performance_records(
            metric_name=metric_name, test_name=test_name, start_time=start_time
        )

        if not records:
            logger.warning(f"No data found for visualization: {metric_name}")
            return {}

        # Prepare data
        timestamps = [datetime.fromtimestamp(r.timestamp) for r in records]
        values = [r.metric_value for r in records]

        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, values))
        timestamps, values = zip(*sorted_data)

        created_files = {}

        # Time series plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, "b-", alpha=0.7, linewidth=1)
        plt.scatter(timestamps, values, alpha=0.5, s=20)

        # Add trend line
        trend = self.analyze_performance_trends(metric_name, days_back)
        if trend and trend.statistical_significance:
            # Calculate trend line
            time_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]
            trend_values = np.array(time_numeric) * trend.slope + values[0]
            plt.plot(
                timestamps,
                trend_values,
                "r--",
                alpha=0.8,
                linewidth=2,
                label=f"Trend: {trend.trend_direction}",
            )
            plt.legend()

        plt.title(f"Performance Trend: {metric_name}")
        plt.xlabel("Time")
        plt.ylabel(f"{metric_name}")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        timeseries_file = output_dir / f"{metric_name}_timeseries.png"
        plt.savefig(timeseries_file, dpi=150, bbox_inches="tight")
        plt.close()
        created_files["timeseries"] = str(timeseries_file)

        # Distribution plot
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(values, bins=30, alpha=0.7, edgecolor="black")
        plt.axvline(
            statistics.mean(values),
            color="red",
            linestyle="--",
            label=f"Mean: {statistics.mean(values):.3f}",
        )
        plt.axvline(
            statistics.median(values),
            color="green",
            linestyle="--",
            label=f"Median: {statistics.median(values):.3f}",
        )
        plt.xlabel(metric_name)
        plt.ylabel("Frequency")
        plt.title(f"Distribution: {metric_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(values, vert=True)
        plt.ylabel(metric_name)
        plt.title(f"Box Plot: {metric_name}")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        distribution_file = output_dir / f"{metric_name}_distribution.png"
        plt.savefig(distribution_file, dpi=150, bbox_inches="tight")
        plt.close()
        created_files["distribution"] = str(distribution_file)

        # Performance over commits (if git commit data available)
        commits_with_data = [
            (r.git_commit, r.metric_value) for r in records if r.git_commit
        ]

        if len(commits_with_data) > 5:  # Need reasonable number of commits
            plt.figure(figsize=(14, 6))

            # Group by commit and calculate statistics
            commit_groups = {}
            for commit, value in commits_with_data:
                if commit not in commit_groups:
                    commit_groups[commit] = []
                commit_groups[commit].append(value)

            commits = list(commit_groups.keys())[-20:]  # Last 20 commits
            commit_means = [statistics.mean(commit_groups[c]) for c in commits]
            commit_stds = [
                statistics.stdev(commit_groups[c]) if len(commit_groups[c]) > 1 else 0
                for c in commits
            ]

            x_pos = range(len(commits))
            plt.errorbar(
                x_pos, commit_means, yerr=commit_stds, fmt="o-", capsize=5, capthick=2
            )

            plt.xlabel("Git Commit")
            plt.ylabel(metric_name)
            plt.title(f"Performance by Git Commit: {metric_name}")
            plt.xticks(x_pos, [c[:8] for c in commits], rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            commits_file = output_dir / f"{metric_name}_by_commits.png"
            plt.savefig(commits_file, dpi=150, bbox_inches="tight")
            plt.close()
            created_files["commits"] = str(commits_file)

        logger.info(
            f"Created {len(created_files)} visualization files for {metric_name}"
        )
        return created_files

    def export_data(
        self,
        output_path: Union[str, Path],
        format: str = "json",
        metric_name: Optional[str] = None,
        days_back: int = 30,
    ) -> str:
        """Export performance data in various formats."""

        output_path = Path(output_path)

        # Get data
        start_time = time.time() - (days_back * 24 * 3600)
        records = self.query_performance_records(
            metric_name=metric_name, start_time=start_time
        )

        if format.lower() == "json":
            data = [asdict(record) for record in records]

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format.lower() == "csv":
            with open(output_path, "w", newline="") as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=asdict(records[0]).keys())
                    writer.writeheader()
                    for record in records:
                        writer.writerow(asdict(record))

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(records)} records to {output_path}")
        return str(output_path)

    def generate_html_report(
        self,
        output_path: Union[str, Path],
        metrics: Optional[List[str]] = None,
        days_back: int = 30,
    ) -> str:
        """Generate comprehensive HTML performance report."""

        output_path = Path(output_path)

        # Get all metrics if not specified
        if not metrics:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT metric_name FROM performance_records")
                metrics = [row[0] for row in cursor.fetchall()]

        # Generate report data
        report_data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_period_days": days_back,
            "metrics": {},
        }

        for metric_name in metrics:
            try:
                summary = self.generate_benchmark_summary(metric_name, f"{days_back}d")
                report_data["metrics"][metric_name] = asdict(summary)
            except ValueError:
                # Skip metrics with no data
                continue

        # Create HTML report
        html_content = self._generate_html_content(report_data)

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)

    def _generate_html_content(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML content for performance report."""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>XPCS Toolkit Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .metric-name {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }}
        .stat {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
        .stat-label {{ font-weight: bold; color: #666; }}
        .stat-value {{ font-size: 1.1em; color: #000; }}
        .trend {{ padding: 10px; border-radius: 3px; margin: 10px 0; }}
        .trend-improving {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
        .trend-degrading {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .trend-stable {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>XPCS Toolkit Performance Report</h1>
        <p><strong>Generated:</strong> {report_data["generated_at"]}</p>
        <p><strong>Analysis Period:</strong> {report_data["analysis_period_days"]} days</p>
        <p><strong>Metrics Analyzed:</strong> {len(report_data["metrics"])}</p>
    </div>
"""

        for metric_name, metric_data in report_data["metrics"].items():
            trend_class = f"trend-{metric_data.get('trend_analysis', {}).get('trend_direction', 'stable')}"

            html += f"""
    <div class="metric">
        <div class="metric-name">{metric_name}</div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Sample Count</div>
                <div class="stat-value">{metric_data["sample_count"]}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Mean</div>
                <div class="stat-value">{metric_data["mean_value"]:.3f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Median</div>
                <div class="stat-value">{metric_data["median_value"]:.3f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Std Dev</div>
                <div class="stat-value">{metric_data["std_dev"]:.3f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">95th Percentile</div>
                <div class="stat-value">{metric_data["percentile_95"]:.3f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">99th Percentile</div>
                <div class="stat-value">{metric_data["percentile_99"]:.3f}</div>
            </div>
        </div>
"""

            if metric_data.get("trend_analysis"):
                trend = metric_data["trend_analysis"]
                html += f"""
        <div class="trend {trend_class}">
            <strong>Trend Analysis:</strong> {trend["trend_direction"].title()}<br>
            <strong>Strength:</strong> {trend["trend_strength"]:.3f} | 
            <strong>R²:</strong> {trend["r_squared"]:.3f} | 
            <strong>P-value:</strong> {trend["p_value"]:.6f}<br>
            <strong>Data Points:</strong> {trend["data_points"]} | 
            <strong>Time Span:</strong> {trend["time_span_days"]:.1f} days<br>
            <strong>30-day Projection:</strong> {trend["projected_change_30days"]:+.3f}
        </div>
"""

            html += "    </div>\n"

        html += """
</body>
</html>
"""

        return html

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old performance data."""

        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Delete old performance records
            cursor.execute(
                "DELETE FROM performance_records WHERE timestamp < ?", (cutoff_time,)
            )
            records_deleted = cursor.rowcount

            # Delete old trend analysis cache
            cursor.execute(
                "DELETE FROM trend_analysis_cache WHERE calculated_at < ?",
                (cutoff_time,),
            )
            cache_deleted = cursor.rowcount

            # Delete orphaned system configurations
            cursor.execute("""
                DELETE FROM system_configurations 
                WHERE id NOT IN (SELECT DISTINCT system_config_id FROM performance_records 
                                WHERE system_config_id IS NOT NULL)
            """)
            configs_deleted = cursor.rowcount

            conn.commit()

        logger.info(
            f"Cleaned up: {records_deleted} records, {cache_deleted} cache entries, {configs_deleted} configs"
        )
        return {
            "records_deleted": records_deleted,
            "cache_deleted": cache_deleted,
            "configs_deleted": configs_deleted,
        }


# =============================================================================
# Command Line Interface
# =============================================================================


def main():
    """Main CLI interface for benchmark database."""

    import argparse

    parser = argparse.ArgumentParser(description="XPCS Toolkit Benchmark Database")
    parser.add_argument("--db-path", type=str, help="Database file path")
    parser.add_argument(
        "--import-data", type=str, help="Import performance data from JSON file"
    )
    parser.add_argument(
        "--export-data", type=str, help="Export performance data to file"
    )
    parser.add_argument(
        "--export-format", choices=["json", "csv"], default="json", help="Export format"
    )
    parser.add_argument("--metric", type=str, help="Specific metric to analyze")
    parser.add_argument(
        "--days-back", type=int, default=30, help="Days back for analysis"
    )
    parser.add_argument(
        "--generate-report", type=str, help="Generate HTML report to file"
    )
    parser.add_argument(
        "--create-visualizations", type=str, help="Create visualizations in directory"
    )
    parser.add_argument(
        "--trend-analysis", action="store_true", help="Perform trend analysis"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate benchmark summary"
    )
    parser.add_argument("--cleanup", action="store_true", help="Clean up old data")
    parser.add_argument(
        "--cleanup-days", type=int, default=90, help="Days to keep for cleanup"
    )

    args = parser.parse_args()

    # Initialize database
    db = BenchmarkDatabase(args.db_path)

    # Import data
    if args.import_data:
        with open(args.import_data, "r") as f:
            data = json.load(f)

        records = []
        for item in data:
            record = PerformanceRecord(**item)
            records.append(record)

        db.batch_store_records(records)
        print(f"Imported {len(records)} performance records")
        return

    # Export data
    if args.export_data:
        output_file = db.export_data(
            args.export_data,
            format=args.export_format,
            metric_name=args.metric,
            days_back=args.days_back,
        )
        print(f"Exported data to {output_file}")
        return

    # Generate HTML report
    if args.generate_report:
        metrics = [args.metric] if args.metric else None
        report_file = db.generate_html_report(
            args.generate_report, metrics=metrics, days_back=args.days_back
        )
        print(f"Generated HTML report: {report_file}")
        return

    # Create visualizations
    if args.create_visualizations and args.metric:
        viz_files = db.create_performance_visualizations(
            args.metric, args.create_visualizations, days_back=args.days_back
        )
        print(f"Created visualizations: {list(viz_files.values())}")
        return

    # Trend analysis
    if args.trend_analysis and args.metric:
        trend = db.analyze_performance_trends(args.metric, args.days_back)
        if trend:
            print(f"Trend Analysis for {args.metric}:")
            print(f"  Direction: {trend.trend_direction}")
            print(f"  Strength: {trend.trend_strength:.3f}")
            print(f"  R²: {trend.r_squared:.3f}")
            print(f"  P-value: {trend.p_value:.6f}")
            print(f"  Data Points: {trend.data_points}")
            print(f"  30-day Projection: {trend.projected_change_30days:+.3f}")
        else:
            print("No trend data available")
        return

    # Benchmark summary
    if args.summary and args.metric:
        summary = db.generate_benchmark_summary(args.metric, f"{args.days_back}d")
        print(f"Benchmark Summary for {args.metric}:")
        print(f"  Sample Count: {summary.sample_count}")
        print(f"  Mean: {summary.mean_value:.3f}")
        print(f"  Median: {summary.median_value:.3f}")
        print(f"  Std Dev: {summary.std_dev:.3f}")
        print(f"  95th Percentile: {summary.percentile_95:.3f}")
        print(f"  99th Percentile: {summary.percentile_99:.3f}")
        return

    # Cleanup
    if args.cleanup:
        cleanup_result = db.cleanup_old_data(args.cleanup_days)
        print(f"Cleanup complete: {cleanup_result}")
        return

    print("Use --help for usage information")


if __name__ == "__main__":
    main()
