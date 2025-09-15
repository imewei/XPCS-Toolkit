"""
Performance Regression Detection System for XPCS Toolkit.

Provides tools for:
- Automated performance regression detection
- Statistical analysis of benchmark results
- CI/CD integration utilities
- Performance trend analysis
- Alert generation and reporting
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# Performance test configuration
from .. import PERFORMANCE_CONFIG


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""

    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    rounds: int
    timestamp: float = None
    memory_peak: float | None = None
    memory_delta: float | None = None
    extra_info: dict[str, Any] | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.extra_info is None:
            self.extra_info = {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        return cls(**data)

    def get_coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV) = std / mean."""
        if self.mean_time > 0:
            return self.std_time / self.mean_time
        return float("inf")

    def is_reliable(self, max_cv: float = 0.3) -> bool:
        """Check if result is reliable based on coefficient of variation."""
        return self.get_coefficient_of_variation() <= max_cv


@dataclass
class RegressionAnalysis:
    """Container for regression analysis results."""

    benchmark_name: str
    current_result: BenchmarkResult
    baseline_result: BenchmarkResult | None
    performance_change: float
    is_regression: bool
    is_improvement: bool
    confidence_level: float
    p_value: float | None
    effect_size: float | None
    severity: str  # 'critical', 'major', 'minor', 'none'
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceRegressionDetector:
    """Main class for performance regression detection."""

    def __init__(
        self,
        baseline_file: Path | None = None,
        regression_threshold: float | None = None,
        confidence_level: float | None = None,
    ):
        """
        Initialize regression detector.

        Parameters:
        -----------
        baseline_file : Path, optional
            Path to baseline performance data file
        regression_threshold : float, optional
            Threshold for regression detection (default: 20%)
        confidence_level : float, optional
            Statistical confidence level (default: 95%)
        """
        self.baseline_file = baseline_file
        self.regression_threshold = (
            regression_threshold or PERFORMANCE_CONFIG["performance_threshold"]
        )
        self.confidence_level = (
            confidence_level or PERFORMANCE_CONFIG["statistical_confidence"]
        )

        self.baseline_data: dict[str, Any] = {}
        self.current_results: list[BenchmarkResult] = []

        if self.baseline_file and self.baseline_file.exists():
            self.load_baseline()

    def load_baseline(self) -> None:
        """Load baseline performance data."""
        try:
            with open(self.baseline_file) as f:
                self.baseline_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load baseline data: {e}")
            self.baseline_data = {}

    def save_baseline(
        self, results: list[BenchmarkResult], output_file: Path | None = None
    ) -> None:
        """Save benchmark results as new baseline."""
        output_file = output_file or self.baseline_file

        if output_file:
            baseline_data = {
                "_metadata": {
                    "created_date": datetime.now().isoformat(),
                    "xpcs_toolkit_version": "1.0.0",
                    "python_version": "3.12",
                    "total_benchmarks": len(results),
                    "description": "Performance baseline generated from benchmark results",
                }
            }

            # Group results by category
            categories = {}
            for result in results:
                category = self._extract_category(result.name)
                if category not in categories:
                    categories[category] = {}
                categories[category][result.name] = result.to_dict()

            baseline_data.update(categories)

            # Add thresholds
            baseline_data["thresholds"] = {
                "performance_regression_threshold": self.regression_threshold,
                "memory_regression_threshold_mb": PERFORMANCE_CONFIG[
                    "memory_threshold_mb"
                ],
                "statistical_confidence": self.confidence_level,
                "minimum_samples": PERFORMANCE_CONFIG["min_rounds"],
            }

            with open(output_file, "w") as f:
                json.dump(baseline_data, f, indent=2)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result for analysis."""
        self.current_results.append(result)

    def analyze_regression(self, result: BenchmarkResult) -> RegressionAnalysis:
        """
        Analyze a single benchmark result for regression.

        Parameters:
        -----------
        result : BenchmarkResult
            Current benchmark result to analyze

        Returns:
        --------
        RegressionAnalysis
            Analysis results including regression status
        """
        baseline_result = self._get_baseline_result(result.name)

        if baseline_result is None:
            return RegressionAnalysis(
                benchmark_name=result.name,
                current_result=result,
                baseline_result=None,
                performance_change=0.0,
                is_regression=False,
                is_improvement=False,
                confidence_level=self.confidence_level,
                p_value=None,
                effect_size=None,
                severity="none",
                message="No baseline available for comparison",
            )

        # Calculate performance change
        if baseline_result.mean_time > 0:
            performance_change = (
                result.mean_time - baseline_result.mean_time
            ) / baseline_result.mean_time
        else:
            performance_change = 0.0

        # Statistical significance testing
        p_value, effect_size = self._statistical_test(result, baseline_result)

        # Determine regression status
        is_regression = performance_change > self.regression_threshold and (
            p_value is None or p_value < (1 - self.confidence_level)
        )

        is_improvement = (
            performance_change < -0.1  # 10% improvement
            and (p_value is None or p_value < (1 - self.confidence_level))
        )

        # Determine severity
        severity = self._determine_severity(performance_change, is_regression)

        # Generate message
        message = self._generate_message(
            performance_change, is_regression, is_improvement, severity
        )

        return RegressionAnalysis(
            benchmark_name=result.name,
            current_result=result,
            baseline_result=baseline_result,
            performance_change=performance_change,
            is_regression=is_regression,
            is_improvement=is_improvement,
            confidence_level=self.confidence_level,
            p_value=p_value,
            effect_size=effect_size,
            severity=severity,
            message=message,
        )

    def analyze_all_results(self) -> list[RegressionAnalysis]:
        """Analyze all current results for regressions."""
        analyses = []
        for result in self.current_results:
            analysis = self.analyze_regression(result)
            analyses.append(analysis)
        return analyses

    def generate_report(
        self, analyses: list[RegressionAnalysis] | None = None
    ) -> dict[str, Any]:
        """
        Generate comprehensive regression analysis report.

        Parameters:
        -----------
        analyses : List[RegressionAnalysis], optional
            Pre-computed analyses (will compute if not provided)

        Returns:
        --------
        Dict[str, Any]
            Comprehensive report with regression findings
        """
        if analyses is None:
            analyses = self.analyze_all_results()

        # Summary statistics
        total_tests = len(analyses)
        regressions = [a for a in analyses if a.is_regression]
        improvements = [a for a in analyses if a.is_improvement]
        stable = [a for a in analyses if not a.is_regression and not a.is_improvement]

        # Severity breakdown
        severity_counts = {
            "critical": len([a for a in regressions if a.severity == "critical"]),
            "major": len([a for a in regressions if a.severity == "major"]),
            "minor": len([a for a in regressions if a.severity == "minor"]),
        }

        # Performance change statistics
        performance_changes = [
            a.performance_change for a in analyses if a.baseline_result is not None
        ]

        report = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_benchmarks": total_tests,
                "regression_threshold": self.regression_threshold,
                "confidence_level": self.confidence_level,
            },
            "summary": {
                "total_tests": total_tests,
                "regressions": len(regressions),
                "improvements": len(improvements),
                "stable": len(stable),
                "no_baseline": len([a for a in analyses if a.baseline_result is None]),
            },
            "severity_breakdown": severity_counts,
            "performance_statistics": {
                "mean_change": statistics.mean(performance_changes)
                if performance_changes
                else 0,
                "median_change": statistics.median(performance_changes)
                if performance_changes
                else 0,
                "std_change": statistics.stdev(performance_changes)
                if len(performance_changes) > 1
                else 0,
                "max_regression": max(performance_changes)
                if performance_changes
                else 0,
                "max_improvement": min(performance_changes)
                if performance_changes
                else 0,
            },
            "regressions": [a.to_dict() for a in regressions],
            "improvements": [a.to_dict() for a in improvements],
            "all_analyses": [a.to_dict() for a in analyses],
        }

        return report

    def save_report(self, report: dict[str, Any], output_file: Path) -> None:
        """Save regression analysis report to file."""
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

    def check_ci_failure_conditions(
        self, analyses: list[RegressionAnalysis] | None = None
    ) -> tuple[bool, str]:
        """
        Check if CI should fail based on regression analysis.

        Parameters:
        -----------
        analyses : List[RegressionAnalysis], optional
            Pre-computed analyses

        Returns:
        --------
        Tuple[bool, str]
            (should_fail, reason_message)
        """
        if analyses is None:
            analyses = self.analyze_all_results()

        # Check for critical regressions
        critical_regressions = [a for a in analyses if a.severity == "critical"]
        if critical_regressions:
            return (
                True,
                f"Found {len(critical_regressions)} critical performance regressions",
            )

        # Check for too many major regressions
        major_regressions = [a for a in analyses if a.severity == "major"]
        if len(major_regressions) > 3:
            return (
                True,
                f"Found {len(major_regressions)} major performance regressions (threshold: 3)",
            )

        # Check for overall regression trend
        total_regressions = [a for a in analyses if a.is_regression]
        regression_rate = len(total_regressions) / len(analyses) if analyses else 0
        if regression_rate > 0.3:  # More than 30% regressions
            return (
                True,
                f"High regression rate: {regression_rate:.1%} of tests regressed",
            )

        return False, "Performance regression check passed"

    def _get_baseline_result(self, benchmark_name: str) -> BenchmarkResult | None:
        """Get baseline result for a benchmark by name."""
        # Search through all categories in baseline data
        for category_name, category_data in self.baseline_data.items():
            if category_name.startswith("_") or category_name == "thresholds":
                continue

            if benchmark_name in category_data:
                data = category_data[benchmark_name]
                return BenchmarkResult(
                    name=benchmark_name,
                    mean_time=data.get("mean_time", 0),
                    std_time=data.get("std_time", 0),
                    min_time=data.get("min_time", 0),
                    max_time=data.get("max_time", 0),
                    rounds=data.get("rounds", 0),
                    memory_peak=data.get("memory_peak"),
                    memory_delta=data.get("memory_delta"),
                    extra_info=data.get("extra_info", {}),
                )

        return None

    def _statistical_test(
        self, current: BenchmarkResult, baseline: BenchmarkResult
    ) -> tuple[float | None, float | None]:
        """
        Perform statistical test to determine significance of performance change.

        Returns:
        --------
        Tuple[Optional[float], Optional[float]]
            (p_value, effect_size)
        """
        # For single measurements, we can't perform proper statistical tests
        # We'll use a simple heuristic based on coefficient of variation

        # Effect size (Cohen's d approximation)
        if baseline.std_time > 0:
            effect_size = (
                abs(current.mean_time - baseline.mean_time) / baseline.std_time
            )
        else:
            effect_size = None

        # Simple significance test based on confidence intervals
        # If we assume normal distribution, we can estimate significance
        if current.std_time > 0 and baseline.std_time > 0:
            # Welch's t-test approximation
            pooled_std = np.sqrt((current.std_time**2 + baseline.std_time**2) / 2)
            if pooled_std > 0:
                t_stat = abs(current.mean_time - baseline.mean_time) / pooled_std
                # Rough p-value estimation (very approximate)
                p_value = 2 * (1 - stats.norm.cdf(t_stat))
                p_value = max(0.001, min(0.999, p_value))  # Clamp to reasonable range
            else:
                p_value = None
        else:
            p_value = None

        return p_value, effect_size

    def _determine_severity(
        self, performance_change: float, is_regression: bool
    ) -> str:
        """Determine severity level of regression."""
        if not is_regression:
            return "none"

        if performance_change > 1.0:  # >100% slower
            return "critical"
        if performance_change > 0.5:  # >50% slower
            return "major"
        return "minor"

    def _generate_message(
        self,
        performance_change: float,
        is_regression: bool,
        is_improvement: bool,
        severity: str,
    ) -> str:
        """Generate human-readable message for regression analysis."""
        if is_regression:
            return f"{severity.capitalize()} regression: {performance_change:.1%} slower than baseline"
        if is_improvement:
            return f"Performance improvement: {abs(performance_change):.1%} faster than baseline"
        return f"Performance is stable: {performance_change:.1%} change from baseline"

    def _extract_category(self, benchmark_name: str) -> str:
        """Extract category name from benchmark name."""
        # Simple heuristic to categorize benchmarks
        if "hdf5" in benchmark_name.lower():
            return "hdf5_operations"
        if "g2" in benchmark_name.lower():
            return "g2_analysis"
        if "saxs" in benchmark_name.lower():
            return "saxs_analysis"
        if "twotime" in benchmark_name.lower():
            return "twotime_analysis"
        if "memory" in benchmark_name.lower():
            return "memory_management"
        if "thread" in benchmark_name.lower() or "parallel" in benchmark_name.lower():
            return "threading_performance"
        if "scaling" in benchmark_name.lower():
            return "scalability"
        return "general_performance"


class PerformanceTrendAnalyzer:
    """Analyze performance trends over time."""

    def __init__(self, history_file: Path | None = None):
        """
        Initialize trend analyzer.

        Parameters:
        -----------
        history_file : Path, optional
            Path to historical performance data file
        """
        self.history_file = history_file
        self.history_data: list[dict[str, Any]] = []

        if self.history_file and self.history_file.exists():
            self.load_history()

    def load_history(self) -> None:
        """Load historical performance data."""
        try:
            with open(self.history_file) as f:
                self.history_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load history data: {e}")
            self.history_data = []

    def add_results(
        self, results: list[BenchmarkResult], commit_hash: str | None = None
    ) -> None:
        """Add benchmark results to history."""
        entry = {
            "timestamp": time.time(),
            "commit_hash": commit_hash,
            "results": [result.to_dict() for result in results],
        }
        self.history_data.append(entry)

    def save_history(self) -> None:
        """Save history data to file."""
        if self.history_file:
            with open(self.history_file, "w") as f:
                json.dump(self.history_data, f, indent=2)

    def analyze_trends(
        self, benchmark_name: str, days_back: int = 30
    ) -> dict[str, Any]:
        """
        Analyze performance trends for a specific benchmark.

        Parameters:
        -----------
        benchmark_name : str
            Name of benchmark to analyze
        days_back : int
            Number of days of history to analyze

        Returns:
        --------
        Dict[str, Any]
            Trend analysis results
        """
        # Filter data by time range
        cutoff_time = time.time() - (days_back * 24 * 3600)
        recent_data = [
            entry for entry in self.history_data if entry["timestamp"] >= cutoff_time
        ]

        if not recent_data:
            return {"error": "No recent data available"}

        # Extract time series for this benchmark
        timestamps = []
        values = []

        for entry in recent_data:
            for result_data in entry["results"]:
                if result_data["name"] == benchmark_name:
                    timestamps.append(entry["timestamp"])
                    values.append(result_data["mean_time"])
                    break

        if len(values) < 2:
            return {"error": "Insufficient data points for trend analysis"}

        # Perform trend analysis
        timestamps_norm = np.array(timestamps) - min(
            timestamps
        )  # Normalize to start at 0
        values_array = np.array(values)

        # Linear regression
        slope, _intercept, r_value, p_value, _std_err = stats.linregress(
            timestamps_norm, values_array
        )

        # Calculate trend direction
        trend_direction = "increasing" if slope > 0 else "decreasing"
        trend_strength = abs(r_value)

        # Calculate percentage change over period
        if len(values) > 1:
            percent_change = (values[-1] - values[0]) / values[0] * 100
        else:
            percent_change = 0

        return {
            "benchmark_name": benchmark_name,
            "analysis_period_days": days_back,
            "data_points": len(values),
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "percent_change": percent_change,
            "mean_performance": statistics.mean(values),
            "std_performance": statistics.stdev(values) if len(values) > 1 else 0,
            "timestamps": timestamps,
            "values": values,
        }

    def detect_anomalies(
        self, benchmark_name: str, sensitivity: float = 2.0
    ) -> list[dict[str, Any]]:
        """
        Detect performance anomalies using statistical methods.

        Parameters:
        -----------
        benchmark_name : str
            Name of benchmark to analyze
        sensitivity : float
            Number of standard deviations for anomaly threshold

        Returns:
        --------
        List[Dict[str, Any]]
            List of detected anomalies
        """
        # Extract time series
        timestamps = []
        values = []

        for entry in self.history_data:
            for result_data in entry["results"]:
                if result_data["name"] == benchmark_name:
                    timestamps.append(entry["timestamp"])
                    values.append(result_data["mean_time"])
                    break

        if len(values) < 10:  # Need enough data for anomaly detection
            return []

        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)

        anomalies = []

        for i, (timestamp, value) in enumerate(zip(timestamps, values, strict=False)):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0

            if z_score > sensitivity:
                anomalies.append(
                    {
                        "timestamp": timestamp,
                        "value": value,
                        "z_score": z_score,
                        "index": i,
                        "deviation_type": "high" if value > mean_val else "low",
                    }
                )

        return anomalies

    def generate_trend_report(
        self, benchmark_names: list[str] | None = None, days_back: int = 30
    ) -> dict[str, Any]:
        """Generate comprehensive trend analysis report."""
        if benchmark_names is None:
            # Extract all unique benchmark names
            benchmark_names = set()
            for entry in self.history_data:
                for result_data in entry["results"]:
                    benchmark_names.add(result_data["name"])
            benchmark_names = list(benchmark_names)

        report = {
            "report_metadata": {
                "timestamp": datetime.now().isoformat(),
                "analysis_period_days": days_back,
                "benchmarks_analyzed": len(benchmark_names),
                "total_data_entries": len(self.history_data),
            },
            "trend_analyses": {},
            "anomalies": {},
            "summary": {
                "improving_benchmarks": 0,
                "degrading_benchmarks": 0,
                "stable_benchmarks": 0,
                "total_anomalies": 0,
            },
        }

        for benchmark_name in benchmark_names:
            # Trend analysis
            trend_analysis = self.analyze_trends(benchmark_name, days_back)
            if "error" not in trend_analysis:
                report["trend_analyses"][benchmark_name] = trend_analysis

                # Categorize trend
                if trend_analysis["slope"] < -0.001:  # Improving (getting faster)
                    report["summary"]["improving_benchmarks"] += 1
                elif trend_analysis["slope"] > 0.001:  # Degrading (getting slower)
                    report["summary"]["degrading_benchmarks"] += 1
                else:
                    report["summary"]["stable_benchmarks"] += 1

            # Anomaly detection
            anomalies = self.detect_anomalies(benchmark_name)
            if anomalies:
                report["anomalies"][benchmark_name] = anomalies
                report["summary"]["total_anomalies"] += len(anomalies)

        return report


# CI/CD Integration utilities
class CIIntegration:
    """Utilities for CI/CD pipeline integration."""

    @staticmethod
    def generate_github_actions_output(report: dict[str, Any]) -> str:
        """Generate GitHub Actions output format."""
        summary = report["summary"]

        output_lines = [
            f"::set-output name=total_tests::{summary['total_tests']}",
            f"::set-output name=regressions::{summary['regressions']}",
            f"::set-output name=improvements::{summary['improvements']}",
            f"::set-output name=stable::{summary['stable']}",
        ]

        # Add severity counts
        for severity, count in report["severity_breakdown"].items():
            output_lines.append(f"::set-output name={severity}_regressions::{count}")

        return "\n".join(output_lines)

    @staticmethod
    def generate_junit_xml(
        analyses: list[RegressionAnalysis], output_file: Path
    ) -> None:
        """Generate JUnit XML report for CI systems."""
        # Simple JUnit XML generation
        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_content.append(
            f'<testsuite name="Performance Regression Tests" tests="{len(analyses)}" failures="{len([a for a in analyses if a.is_regression])}" errors="0">'
        )

        for analysis in analyses:
            test_name = analysis.benchmark_name.replace(" ", "_")

            if analysis.is_regression:
                xml_content.append(
                    f'  <testcase name="{test_name}" classname="PerformanceRegression">'
                )
                xml_content.append(f'    <failure message="{analysis.message}">')
                xml_content.append(
                    f"      Performance change: {analysis.performance_change:.1%}"
                )
                xml_content.append(f"      Severity: {analysis.severity}")
                xml_content.append("    </failure>")
                xml_content.append("  </testcase>")
            else:
                xml_content.append(
                    f'  <testcase name="{test_name}" classname="PerformanceRegression"/>'
                )

        xml_content.append("</testsuite>")

        with open(output_file, "w") as f:
            f.write("\n".join(xml_content))

    @staticmethod
    def generate_markdown_report(report: dict[str, Any]) -> str:
        """Generate Markdown report for GitHub PR comments."""
        lines = [
            "# Performance Regression Analysis Report",
            "",
            f"**Generated:** {report['report_metadata']['timestamp']}",
            f"**Total Benchmarks:** {report['summary']['total_tests']}",
            "",
            "## Summary",
            "",
            f"- ✅ **Stable:** {report['summary']['stable']} benchmarks",
            f"- ⬆️ **Improvements:** {report['summary']['improvements']} benchmarks",
            f"- ⬇️ **Regressions:** {report['summary']['regressions']} benchmarks",
            "",
        ]

        # Severity breakdown
        if report["severity_breakdown"]["critical"] > 0:
            lines.append(
                f"- 🔴 **Critical:** {report['severity_breakdown']['critical']} regressions"
            )
        if report["severity_breakdown"]["major"] > 0:
            lines.append(
                f"- 🟡 **Major:** {report['severity_breakdown']['major']} regressions"
            )
        if report["severity_breakdown"]["minor"] > 0:
            lines.append(
                f"- 🟢 **Minor:** {report['severity_breakdown']['minor']} regressions"
            )

        # Performance statistics
        perf_stats = report["performance_statistics"]
        if abs(perf_stats["mean_change"]) > 0.01:  # More than 1% change
            lines.extend(
                [
                    "",
                    "## Performance Statistics",
                    "",
                    f"- **Mean Change:** {perf_stats['mean_change']:.1%}",
                    f"- **Median Change:** {perf_stats['median_change']:.1%}",
                    f"- **Worst Regression:** {perf_stats['max_regression']:.1%}",
                    f"- **Best Improvement:** {abs(perf_stats['max_improvement']):.1%}",
                ]
            )

        # Critical regressions
        critical_regressions = [
            a for a in report["all_analyses"] if a["severity"] == "critical"
        ]
        if critical_regressions:
            lines.extend(["", "## 🔴 Critical Regressions", ""])
            for regression in critical_regressions:
                lines.append(
                    f"- **{regression['benchmark_name']}**: {regression['message']}"
                )

        return "\n".join(lines)


# Example usage functions
def run_regression_analysis(
    results: list[BenchmarkResult], baseline_file: Path, output_dir: Path
) -> bool:
    """
    Run complete regression analysis pipeline.

    Returns True if CI should pass, False if CI should fail.
    """
    # Initialize detector
    detector = PerformanceRegressionDetector(baseline_file)

    # Add all results
    for result in results:
        detector.add_result(result)

    # Analyze regressions
    analyses = detector.analyze_all_results()

    # Generate report
    report = detector.generate_report(analyses)

    # Save report
    report_file = output_dir / "regression_report.json"
    detector.save_report(report, report_file)

    # Generate additional formats
    markdown_report = CIIntegration.generate_markdown_report(report)
    with open(output_dir / "regression_report.md", "w") as f:
        f.write(markdown_report)

    # Generate JUnit XML
    CIIntegration.generate_junit_xml(analyses, output_dir / "regression_results.xml")

    # Check CI failure conditions
    should_fail, reason = detector.check_ci_failure_conditions(analyses)

    if should_fail:
        print(f"❌ CI should fail: {reason}")
    else:
        print("✅ CI should pass: No significant regressions detected")

    return not should_fail
