"""Test Performance Optimization and Monitoring System for XPCS Toolkit.

This system provides comprehensive monitoring, optimization, and analysis
of test suite performance, helping maintain fast and efficient testing
across all environments and scales.

Features:
- Real-time test execution monitoring
- Performance regression detection
- Automated test optimization recommendations
- Resource usage tracking (CPU, memory, I/O)
- Parallel execution optimization
- Performance baseline management
- Historical trend analysis
"""

import json
import logging
import platform
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil


@dataclass
class TestPerformanceMetrics:
    """Performance metrics for a single test."""

    test_name: str
    test_file: str
    category: str

    # Timing metrics
    setup_time: float
    execution_time: float
    teardown_time: float
    total_time: float

    # Resource metrics
    peak_memory_mb: float
    cpu_percent: float
    io_operations: int

    # Quality metrics
    passed: bool
    assertions_count: int

    # Environment
    timestamp: datetime
    platform: str
    python_version: str


@dataclass
class TestSuitePerformanceReport:
    """Comprehensive performance report for entire test suite."""

    timestamp: datetime
    total_tests: int
    total_duration: float

    # Performance statistics
    fastest_test: Optional[TestPerformanceMetrics]
    slowest_test: Optional[TestPerformanceMetrics]
    median_time: float
    p95_time: float

    # Resource usage
    peak_memory_mb: float
    avg_cpu_percent: float
    total_io_operations: int

    # Categories breakdown
    category_performance: Dict[str, Dict[str, Any]]

    # Trends and regressions
    performance_trend: str  # "improving", "stable", "degrading"
    regressions_detected: List[str]
    optimizations_suggested: List[str]


class TestExecutionMonitor:
    """Monitor for individual test execution with resource tracking."""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = {
            "peak_memory": 0,
            "cpu_samples": [],
            "io_start": None,
            "io_end": None,
        }
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.metrics["peak_memory"] = 0
        self.metrics["cpu_samples"] = []

        # Get initial I/O counters
        try:
            self.metrics["io_start"] = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            self.metrics["io_start"] = None

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        # Get final I/O counters
        try:
            self.metrics["io_end"] = self.process.io_counters()
        except (psutil.AccessDenied, AttributeError):
            self.metrics["io_end"] = None

        # Calculate final metrics
        peak_memory = self.metrics["peak_memory"] / 1024 / 1024  # Convert to MB
        avg_cpu = (
            statistics.mean(self.metrics["cpu_samples"])
            if self.metrics["cpu_samples"]
            else 0
        )

        io_ops = 0
        if self.metrics["io_start"] and self.metrics["io_end"]:
            io_start = self.metrics["io_start"]
            io_end = self.metrics["io_end"]
            io_ops = (io_end.read_count + io_end.write_count) - (
                io_start.read_count + io_start.write_count
            )

        return {
            "peak_memory_mb": peak_memory,
            "avg_cpu_percent": avg_cpu,
            "io_operations": io_ops,
        }

    def _monitor_resources(self):
        """Monitor resources in background thread."""
        while self.monitoring:
            try:
                # Memory usage
                memory = self.process.memory_info().rss
                if memory > self.metrics["peak_memory"]:
                    self.metrics["peak_memory"] = memory

                # CPU usage
                cpu = self.process.cpu_percent()
                self.metrics["cpu_samples"].append(cpu)

                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


class TestPerformanceDatabase:
    """Database for storing and querying test performance history."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the performance database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    category TEXT NOT NULL,
                    setup_time REAL,
                    execution_time REAL,
                    teardown_time REAL,
                    total_time REAL,
                    peak_memory_mb REAL,
                    cpu_percent REAL,
                    io_operations INTEGER,
                    passed BOOLEAN,
                    assertions_count INTEGER,
                    timestamp TEXT,
                    platform TEXT,
                    python_version TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_name_timestamp
                ON test_performance (test_name, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category_timestamp
                ON test_performance (category, timestamp)
            """)

    def store_metrics(self, metrics: TestPerformanceMetrics):
        """Store performance metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO test_performance (
                    test_name, test_file, category, setup_time, execution_time,
                    teardown_time, total_time, peak_memory_mb, cpu_percent,
                    io_operations, passed, assertions_count, timestamp,
                    platform, python_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.test_name,
                    metrics.test_file,
                    metrics.category,
                    metrics.setup_time,
                    metrics.execution_time,
                    metrics.teardown_time,
                    metrics.total_time,
                    metrics.peak_memory_mb,
                    metrics.cpu_percent,
                    metrics.io_operations,
                    metrics.passed,
                    metrics.assertions_count,
                    metrics.timestamp.isoformat(),
                    metrics.platform,
                    metrics.python_version,
                ),
            )

    def get_test_history(
        self, test_name: str, days: int = 30
    ) -> List[TestPerformanceMetrics]:
        """Get performance history for a specific test."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM test_performance
                WHERE test_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """,
                (test_name, cutoff_date),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    TestPerformanceMetrics(
                        test_name=row[1],
                        test_file=row[2],
                        category=row[3],
                        setup_time=row[4],
                        execution_time=row[5],
                        teardown_time=row[6],
                        total_time=row[7],
                        peak_memory_mb=row[8],
                        cpu_percent=row[9],
                        io_operations=row[10],
                        passed=bool(row[11]),
                        assertions_count=row[12],
                        timestamp=datetime.fromisoformat(row[13]),
                        platform=row[14],
                        python_version=row[15],
                    )
                )

            return results

    def get_slowest_tests(
        self, category: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the slowest tests overall or by category."""
        query = """
            SELECT test_name, test_file, category, AVG(total_time) as avg_time,
                   MAX(total_time) as max_time, COUNT(*) as run_count
            FROM test_performance
            WHERE timestamp >= date('now', '-30 days')
        """
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " GROUP BY test_name ORDER BY avg_time DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [
                {
                    "test_name": row[0],
                    "test_file": row[1],
                    "category": row[2],
                    "avg_time": row[3],
                    "max_time": row[4],
                    "run_count": row[5],
                }
                for row in cursor.fetchall()
            ]


class TestPerformanceOptimizer:
    """Analyzer and optimizer for test performance."""

    def __init__(self, db: TestPerformanceDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across the test suite."""
        analysis = {
            "overall_trend": self._analyze_overall_trend(),
            "category_trends": self._analyze_category_trends(),
            "regression_detection": self._detect_regressions(),
            "optimization_opportunities": self._identify_optimization_opportunities(),
        }

        return analysis

    def _analyze_overall_trend(self) -> Dict[str, Any]:
        """Analyze overall test suite performance trend."""
        with sqlite3.connect(self.db.db_path) as conn:
            # Get average performance over time (weekly buckets)
            cursor = conn.execute("""
                SELECT
                    DATE(timestamp, 'weekday 0', '-7 days') as week_start,
                    AVG(total_time) as avg_time,
                    COUNT(*) as test_count,
                    AVG(peak_memory_mb) as avg_memory
                FROM test_performance
                WHERE timestamp >= date('now', '-90 days')
                GROUP BY week_start
                ORDER BY week_start
            """)

            weekly_data = cursor.fetchall()

            if len(weekly_data) < 2:
                return {
                    "trend": "insufficient_data",
                    "weeks_analyzed": len(weekly_data),
                }

            # Calculate trend
            times = [row[1] for row in weekly_data]
            memories = [row[3] for row in weekly_data]

            # Simple linear regression for trend
            x = list(range(len(times)))
            time_trend = np.polyfit(x, times, 1)[0] if len(times) > 1 else 0
            memory_trend = np.polyfit(x, memories, 1)[0] if len(memories) > 1 else 0

            return {
                "trend": "improving"
                if time_trend < -0.01
                else "degrading"
                if time_trend > 0.01
                else "stable",
                "time_trend_slope": time_trend,
                "memory_trend_slope": memory_trend,
                "weeks_analyzed": len(weekly_data),
                "latest_avg_time": times[-1] if times else 0,
                "latest_avg_memory": memories[-1] if memories else 0,
            }

    def _analyze_category_trends(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance trends by test category."""
        categories = ["unit", "integration", "scientific", "performance", "gui"]
        category_trends = {}

        for category in categories:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT AVG(total_time), AVG(peak_memory_mb), COUNT(*)
                    FROM test_performance
                    WHERE category = ? AND timestamp >= date('now', '-30 days')
                """,
                    (category,),
                )

                result = cursor.fetchone()
                if result and result[2] > 0:  # Has data
                    category_trends[category] = {
                        "avg_time": result[0],
                        "avg_memory": result[1],
                        "test_count": result[2],
                    }
                else:
                    category_trends[category] = {
                        "avg_time": 0,
                        "avg_memory": 0,
                        "test_count": 0,
                    }

        return category_trends

    def _detect_regressions(self) -> List[Dict[str, Any]]:
        """Detect performance regressions in individual tests."""
        regressions = []

        with sqlite3.connect(self.db.db_path) as conn:
            # Find tests with significant performance degradation
            cursor = conn.execute("""
                SELECT test_name, test_file, category,
                       AVG(CASE WHEN timestamp >= date('now', '-7 days') THEN total_time END) as recent_avg,
                       AVG(CASE WHEN timestamp < date('now', '-7 days') AND timestamp >= date('now', '-30 days')
                                THEN total_time END) as historical_avg,
                       COUNT(*) as total_runs
                FROM test_performance
                WHERE timestamp >= date('now', '-30 days')
                GROUP BY test_name
                HAVING recent_avg > historical_avg * 1.3  -- 30% slower
                   AND total_runs >= 5  -- Sufficient data
                ORDER BY (recent_avg / historical_avg) DESC
                LIMIT 20
            """)

            for row in cursor.fetchall():
                regression_factor = row[3] / row[4] if row[4] > 0 else 1
                regressions.append(
                    {
                        "test_name": row[0],
                        "test_file": row[1],
                        "category": row[2],
                        "recent_avg_time": row[3],
                        "historical_avg_time": row[4],
                        "regression_factor": regression_factor,
                        "severity": "critical"
                        if regression_factor > 2.0
                        else "moderate",
                    }
                )

        return regressions

    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for test optimization."""
        opportunities = []

        # Find consistently slow tests
        slow_tests = self.db.get_slowest_tests(limit=20)
        for test in slow_tests:
            if test["avg_time"] > 5.0:  # Tests taking more than 5 seconds
                opportunities.append(
                    {
                        "type": "slow_test",
                        "test_name": test["test_name"],
                        "test_file": test["test_file"],
                        "avg_time": test["avg_time"],
                        "recommendation": "Consider optimizing or parallelizing this test",
                    }
                )

        # Find memory-intensive tests
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, test_file, category, AVG(peak_memory_mb) as avg_memory
                FROM test_performance
                WHERE timestamp >= date('now', '-30 days')
                  AND peak_memory_mb > 100  -- More than 100MB
                GROUP BY test_name
                ORDER BY avg_memory DESC
                LIMIT 10
            """)

            for row in cursor.fetchall():
                opportunities.append(
                    {
                        "type": "memory_intensive",
                        "test_name": row[0],
                        "test_file": row[1],
                        "category": row[2],
                        "avg_memory": row[3],
                        "recommendation": "Consider reducing memory usage or using fixtures more efficiently",
                    }
                )

        # Find tests with high I/O
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT test_name, test_file, category, AVG(io_operations) as avg_io
                FROM test_performance
                WHERE timestamp >= date('now', '-30 days')
                  AND io_operations > 1000
                GROUP BY test_name
                ORDER BY avg_io DESC
                LIMIT 10
            """)

            for row in cursor.fetchall():
                opportunities.append(
                    {
                        "type": "io_intensive",
                        "test_name": row[0],
                        "test_file": row[1],
                        "category": row[2],
                        "avg_io": row[3],
                        "recommendation": "Consider mocking I/O operations or using cached test data",
                    }
                )

        return opportunities

    def generate_optimization_recommendations(self) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []

        analysis = self.analyze_performance_trends()

        # Overall trend recommendations
        if analysis["overall_trend"]["trend"] == "degrading":
            recommendations.append(
                f"Test suite performance is degrading. "
                f"Average test time has increased by {analysis['overall_trend']['time_trend_slope']:.3f}s per week."
            )

        # Regression recommendations
        regressions = analysis["regression_detection"]
        critical_regressions = [r for r in regressions if r["severity"] == "critical"]
        if critical_regressions:
            recommendations.append(
                f"Critical performance regressions detected in {len(critical_regressions)} tests. "
                f"Priority fixes needed for: {', '.join([r['test_name'] for r in critical_regressions[:3]])}"
            )

        # Category-specific recommendations
        category_trends = analysis["category_trends"]
        slow_categories = [
            cat for cat, data in category_trends.items() if data["avg_time"] > 2.0
        ]
        if slow_categories:
            recommendations.append(
                f"Categories with slow average execution time: {', '.join(slow_categories)}. "
                f"Consider parallelization or test refactoring."
            )

        # Optimization opportunities
        opportunities = analysis["optimization_opportunities"]
        if len(opportunities) > 5:
            recommendations.append(
                f"{len(opportunities)} optimization opportunities identified. "
                f"Focus on memory-intensive and I/O-heavy tests first."
            )

        return recommendations


class TestPerformanceMonitor:
    """Main performance monitoring and optimization system."""

    def __init__(self, test_directory: Path = None):
        self.test_dir = test_directory or Path(__file__).parent
        self.project_root = self.test_dir.parent

        # Initialize database
        self.db_path = self.test_dir / "performance_data.db"
        self.db = TestPerformanceDatabase(self.db_path)

        # Initialize optimizer
        self.optimizer = TestPerformanceOptimizer(self.db)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for performance monitoring."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run_performance_profiling(
        self, test_pattern: str = None
    ) -> TestSuitePerformanceReport:
        """Run comprehensive performance profiling of test suite."""
        self.logger.info("Starting test suite performance profiling...")

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", "--durations=0", "-v"]
        if test_pattern:
            cmd.append(test_pattern)
        else:
            cmd.append(str(self.test_dir))

        # Monitor overall execution
        overall_monitor = TestExecutionMonitor()
        overall_monitor.start_monitoring()

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800,  # 30 minute timeout
            )

            execution_time = time.time() - start_time
            overall_metrics = overall_monitor.stop_monitoring()

            # Parse test results and create individual metrics
            test_metrics = self._parse_test_performance(result.stdout, result.stderr)

            # Store metrics in database
            for metrics in test_metrics:
                self.db.store_metrics(metrics)

            # Generate comprehensive report
            report = self._generate_performance_report(
                test_metrics, execution_time, overall_metrics
            )

            self.logger.info(
                f"Performance profiling completed in {execution_time:.2f}s"
            )
            return report

        except subprocess.TimeoutExpired:
            self.logger.error("Performance profiling timed out after 30 minutes")
            raise
        except Exception as e:
            self.logger.error(f"Performance profiling failed: {e}")
            raise

    def _parse_test_performance(
        self, stdout: str, stderr: str
    ) -> List[TestPerformanceMetrics]:
        """Parse test performance data from pytest output."""
        metrics = []
        lines = stdout.split("\n")

        # Parse duration data
        in_durations = False
        for line in lines:
            if "slowest durations" in line:
                in_durations = True
                continue
            elif in_durations and line.startswith("="):
                break
            elif in_durations and "s call" in line:
                # Parse line like: "0.12s call tests/unit/core/test_xpcs_file.py::TestMemoryMonitor::test_get_memory_usage"
                try:
                    duration_str = line.split("s ")[0].strip()
                    duration = float(duration_str)

                    test_path = line.split("s call ")[-1].strip()
                    parts = test_path.split("::")

                    if len(parts) >= 2:
                        test_file = parts[0]
                        test_name = parts[-1]

                        # Determine category from file path
                        category = "unknown"
                        if "/unit/" in test_file:
                            category = "unit"
                        elif "/integration/" in test_file:
                            category = "integration"
                        elif "/scientific/" in test_file:
                            category = "scientific"
                        elif "/performance/" in test_file:
                            category = "performance"
                        elif "/gui/" in test_file:
                            category = "gui"

                        # Create metrics object (simplified - in practice would collect more data)
                        test_metrics = TestPerformanceMetrics(
                            test_name=test_name,
                            test_file=test_file,
                            category=category,
                            setup_time=0.0,  # Not available from basic pytest output
                            execution_time=duration,
                            teardown_time=0.0,
                            total_time=duration,
                            peak_memory_mb=0.0,  # Would need separate monitoring
                            cpu_percent=0.0,
                            io_operations=0,
                            passed=True,  # Assume passed if duration reported
                            assertions_count=0,
                            timestamp=datetime.now(),
                            platform=platform.platform(),
                            python_version=sys.version.split()[0],
                        )

                        metrics.append(test_metrics)

                except (ValueError, IndexError):
                    continue

        return metrics

    def _generate_performance_report(
        self,
        test_metrics: List[TestPerformanceMetrics],
        total_duration: float,
        overall_metrics: Dict[str, Any],
    ) -> TestSuitePerformanceReport:
        """Generate comprehensive performance report."""
        if not test_metrics:
            # Return empty report if no metrics
            return TestSuitePerformanceReport(
                timestamp=datetime.now(),
                total_tests=0,
                total_duration=total_duration,
                fastest_test=None,
                slowest_test=None,
                median_time=0.0,
                p95_time=0.0,
                peak_memory_mb=overall_metrics.get("peak_memory_mb", 0),
                avg_cpu_percent=overall_metrics.get("avg_cpu_percent", 0),
                total_io_operations=overall_metrics.get("io_operations", 0),
                category_performance={},
                performance_trend="unknown",
                regressions_detected=[],
                optimizations_suggested=[],
            )

        # Calculate statistics
        times = [m.total_time for m in test_metrics]
        fastest_test = min(test_metrics, key=lambda m: m.total_time)
        slowest_test = max(test_metrics, key=lambda m: m.total_time)
        median_time = statistics.median(times)
        p95_time = np.percentile(times, 95)

        # Calculate category performance
        categories = {}
        for metrics in test_metrics:
            if metrics.category not in categories:
                categories[metrics.category] = []
            categories[metrics.category].append(metrics)

        category_performance = {}
        for category, cat_metrics in categories.items():
            cat_times = [m.total_time for m in cat_metrics]
            category_performance[category] = {
                "test_count": len(cat_metrics),
                "total_time": sum(cat_times),
                "avg_time": statistics.mean(cat_times),
                "median_time": statistics.median(cat_times),
                "slowest_test": max(cat_metrics, key=lambda m: m.total_time).test_name,
            }

        # Analyze trends and regressions
        analysis = self.optimizer.analyze_performance_trends()
        trend = analysis["overall_trend"]["trend"]
        regressions = [r["test_name"] for r in analysis["regression_detection"]]
        optimizations = self.optimizer.generate_optimization_recommendations()

        return TestSuitePerformanceReport(
            timestamp=datetime.now(),
            total_tests=len(test_metrics),
            total_duration=total_duration,
            fastest_test=fastest_test,
            slowest_test=slowest_test,
            median_time=median_time,
            p95_time=p95_time,
            peak_memory_mb=overall_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=overall_metrics.get("avg_cpu_percent", 0),
            total_io_operations=overall_metrics.get("io_operations", 0),
            category_performance=category_performance,
            performance_trend=trend,
            regressions_detected=regressions,
            optimizations_suggested=optimizations,
        )

    def generate_performance_dashboard(self) -> str:
        """Generate a text-based performance dashboard."""
        analysis = self.optimizer.analyze_performance_trends()
        slowest_tests = self.db.get_slowest_tests(limit=10)

        dashboard = []
        dashboard.append("=" * 80)
        dashboard.append("XPCS TOOLKIT TEST PERFORMANCE DASHBOARD")
        dashboard.append("=" * 80)
        dashboard.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dashboard.append(f"Platform: {platform.platform()}")
        dashboard.append("")

        # Overall trend
        trend_data = analysis["overall_trend"]
        dashboard.append("OVERALL PERFORMANCE TREND")
        dashboard.append("-" * 30)
        dashboard.append(f"Trend: {trend_data['trend'].title()}")
        dashboard.append(
            f"Latest Average Time: {trend_data.get('latest_avg_time', 0):.3f}s"
        )
        dashboard.append(
            f"Latest Average Memory: {trend_data.get('latest_avg_memory', 0):.1f}MB"
        )
        dashboard.append("")

        # Category performance
        dashboard.append("CATEGORY PERFORMANCE")
        dashboard.append("-" * 20)
        for category, data in analysis["category_trends"].items():
            if data["test_count"] > 0:
                dashboard.append(
                    f"{category.title()}: {data['avg_time']:.3f}s avg, {data['test_count']} tests"
                )
        dashboard.append("")

        # Slowest tests
        dashboard.append("SLOWEST TESTS (Last 30 Days)")
        dashboard.append("-" * 30)
        for i, test in enumerate(slowest_tests[:10], 1):
            dashboard.append(
                f"{i:2d}. {test['test_name']} ({test['avg_time']:.3f}s avg)"
            )
        dashboard.append("")

        # Regressions
        regressions = analysis["regression_detection"]
        if regressions:
            dashboard.append("PERFORMANCE REGRESSIONS")
            dashboard.append("-" * 25)
            for reg in regressions[:5]:
                factor = reg["regression_factor"]
                dashboard.append(f"⚠️  {reg['test_name']} ({factor:.1f}x slower)")
        else:
            dashboard.append("✅ No significant performance regressions detected")
        dashboard.append("")

        # Optimization opportunities
        opportunities = analysis["optimization_opportunities"]
        if opportunities:
            dashboard.append("OPTIMIZATION OPPORTUNITIES")
            dashboard.append("-" * 28)
            for opp in opportunities[:5]:
                if opp["type"] == "slow_test":
                    dashboard.append(f"🐌 {opp['test_name']} ({opp['avg_time']:.1f}s)")
                elif opp["type"] == "memory_intensive":
                    dashboard.append(
                        f"💾 {opp['test_name']} ({opp['avg_memory']:.1f}MB)"
                    )
                elif opp["type"] == "io_intensive":
                    dashboard.append(
                        f"💿 {opp['test_name']} ({opp['avg_io']:.0f} I/O ops)"
                    )
        dashboard.append("")

        # Recommendations
        recommendations = self.optimizer.generate_optimization_recommendations()
        if recommendations:
            dashboard.append("RECOMMENDATIONS")
            dashboard.append("-" * 15)
            for i, rec in enumerate(recommendations, 1):
                dashboard.append(f"{i}. {rec}")

        return "\n".join(dashboard)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="XPCS Toolkit Test Performance Monitor"
    )
    parser.add_argument(
        "--profile", help="Run performance profiling (optional test pattern)"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Show performance dashboard"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Show optimization recommendations"
    )
    parser.add_argument("--history", help="Show performance history for specific test")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    monitor = TestPerformanceMonitor()

    if args.profile is not None:
        print("Running performance profiling...")
        report = monitor.run_performance_profiling(args.profile or None)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            print(f"Performance report written to {args.output}")
        else:
            # Print summary
            print(f"Total tests: {report.total_tests}")
            print(f"Total duration: {report.total_duration:.2f}s")
            print(f"Median time: {report.median_time:.3f}s")
            print(
                f"Slowest test: {report.slowest_test.test_name} ({report.slowest_test.total_time:.3f}s)"
            )
            print(f"Performance trend: {report.performance_trend}")

    elif args.dashboard:
        dashboard = monitor.generate_performance_dashboard()
        print(dashboard)

        if args.output:
            with open(args.output, "w") as f:
                f.write(dashboard)

    elif args.optimize:
        recommendations = monitor.optimizer.generate_optimization_recommendations()
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 30)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    elif args.history:
        history = monitor.db.get_test_history(args.history)
        print(f"Performance history for {args.history}:")
        print("Date\t\tTime(s)\tMemory(MB)\tPassed")
        print("-" * 50)
        for metrics in history[:20]:  # Show last 20 runs
            print(
                f"{metrics.timestamp.strftime('%Y-%m-%d')}\t{metrics.total_time:.3f}\t{metrics.peak_memory_mb:.1f}\t\t{metrics.passed}"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
