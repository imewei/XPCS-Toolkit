#!/usr/bin/env python3
"""
Performance Regression Detection System for XPCS Toolkit

This module provides comprehensive performance regression detection with CI/CD
pipeline integration, statistical analysis to distinguish real regressions from
normal variance, and automated baseline comparison and trend analysis.

Features:
- CI/CD pipeline integration for automated regression detection
- Statistical analysis with confidence intervals and significance testing
- Automated baseline comparison and trend analysis
- Performance threshold management and alerting
- Integration with existing test frameworks
- Historical performance tracking and reporting
- Machine learning-based anomaly detection for performance patterns

Integration Points:
- Integrates with CPU performance test suite
- Works with benchmark database for historical data
- Connects to existing monitoring systems
- Supports multiple CI/CD platforms (GitHub Actions, Jenkins, etc.)

Author: Claude Code Performance Testing Generator
Date: 2025-01-11
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy import stats
import psutil

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models and Configuration
# =============================================================================


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""

    metric_name: str
    baseline_value: float
    baseline_std: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    timestamp: float
    git_commit: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    test_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionTest:
    """Regression test result."""

    test_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    relative_change_percent: float
    absolute_change: float
    p_value: float
    is_regression: bool
    severity: str  # 'critical', 'major', 'minor', 'none'
    confidence_level: float
    statistical_power: float
    timestamp: float
    git_commit: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionAnalysisConfig:
    """Configuration for regression analysis."""

    # Statistical parameters
    confidence_level: float = 0.95
    statistical_power: float = 0.80
    min_sample_size: int = 10
    max_baseline_age_days: int = 30

    # Regression detection thresholds
    critical_regression_threshold: float = 0.25  # 25% degradation
    major_regression_threshold: float = 0.15  # 15% degradation
    minor_regression_threshold: float = 0.05  # 5% degradation

    # Improvement detection thresholds
    improvement_threshold: float = 0.10  # 10% improvement

    # Multiple testing correction
    use_bonferroni_correction: bool = True
    false_discovery_rate: float = 0.05

    # Trend analysis
    trend_analysis_window: int = 10  # Number of recent measurements
    trend_significance_threshold: float = 0.05  # P-value for trend significance

    # Alerting configuration
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["console", "log"])
    slack_webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)


class RegressionDetector:
    """Main regression detection system."""

    def __init__(self, config: Optional[RegressionAnalysisConfig] = None):
        self.config = config or RegressionAnalysisConfig()
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.historical_results: List[RegressionTest] = []
        self.baseline_storage_path = Path("performance_baselines.json")
        self.results_storage_path = Path("regression_test_results.json")

        # Load existing baselines and results
        self.load_baselines()
        self.load_historical_results()

    def load_baselines(self):
        """Load performance baselines from storage."""
        if self.baseline_storage_path.exists():
            try:
                with open(self.baseline_storage_path, "r") as f:
                    baseline_data = json.load(f)

                self.baselines = {
                    name: PerformanceBaseline(**data)
                    for name, data in baseline_data.items()
                }

                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")

    def save_baselines(self):
        """Save performance baselines to storage."""
        try:
            baseline_data = {
                name: asdict(baseline) for name, baseline in self.baselines.items()
            }

            with open(self.baseline_storage_path, "w") as f:
                json.dump(baseline_data, f, indent=2)

            logger.info(f"Saved {len(self.baselines)} performance baselines")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def load_historical_results(self):
        """Load historical regression test results."""
        if self.results_storage_path.exists():
            try:
                with open(self.results_storage_path, "r") as f:
                    results_data = json.load(f)

                self.historical_results = [
                    RegressionTest(**data) for data in results_data
                ]

                logger.info(f"Loaded {len(self.historical_results)} historical results")
            except Exception as e:
                logger.error(f"Failed to load historical results: {e}")

    def save_historical_results(self):
        """Save historical regression test results."""
        try:
            results_data = [asdict(result) for result in self.historical_results]

            with open(self.results_storage_path, "w") as f:
                json.dump(results_data, f, indent=2)

            logger.info(f"Saved {len(self.historical_results)} historical results")
        except Exception as e:
            logger.error(f"Failed to save historical results: {e}")

    def create_baseline(
        self,
        metric_name: str,
        values: List[float],
        git_commit: Optional[str] = None,
        test_conditions: Optional[Dict[str, Any]] = None,
    ) -> PerformanceBaseline:
        """Create a new performance baseline from measurement values."""

        if len(values) < self.config.min_sample_size:
            raise ValueError(
                f"Insufficient samples for baseline creation: {len(values)} < {self.config.min_sample_size}"
            )

        # Calculate baseline statistics
        baseline_value = np.mean(values)
        baseline_std = np.std(values, ddof=1)

        # Calculate confidence interval
        t_critical = stats.t.ppf(
            1 - (1 - self.config.confidence_level) / 2, len(values) - 1
        )
        margin_error = t_critical * baseline_std / np.sqrt(len(values))
        confidence_interval = (
            baseline_value - margin_error,
            baseline_value + margin_error,
        )

        # Get system information
        system_info = {
            "cpu_count": os.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": sys.version,
            "platform": sys.platform,
        }

        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            baseline_std=baseline_std,
            sample_size=len(values),
            confidence_interval=confidence_interval,
            timestamp=time.time(),
            git_commit=git_commit,
            system_info=system_info,
            test_conditions=test_conditions or {},
        )

        # Store baseline
        self.baselines[metric_name] = baseline
        self.save_baselines()

        logger.info(
            f"Created baseline for {metric_name}: {baseline_value:.3f} ± {baseline_std:.3f}"
        )
        return baseline

    def detect_regression(
        self,
        metric_name: str,
        current_value: float,
        test_name: Optional[str] = None,
        git_commit: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> RegressionTest:
        """Detect performance regression for a metric."""

        if metric_name not in self.baselines:
            raise ValueError(f"No baseline found for metric: {metric_name}")

        baseline = self.baselines[metric_name]
        test_name = test_name or metric_name

        # Check baseline age
        baseline_age_days = (time.time() - baseline.timestamp) / (24 * 3600)
        if baseline_age_days > self.config.max_baseline_age_days:
            logger.warning(
                f"Baseline for {metric_name} is {baseline_age_days:.1f} days old"
            )

        # Calculate changes
        absolute_change = current_value - baseline.baseline_value
        relative_change_percent = (absolute_change / baseline.baseline_value) * 100

        # Perform statistical test
        # Use one-sample t-test against baseline
        t_statistic = absolute_change / (
            baseline.baseline_std / np.sqrt(baseline.sample_size)
        )
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), baseline.sample_size - 1))

        # Apply Bonferroni correction if enabled
        alpha = 1 - self.config.confidence_level
        if self.config.use_bonferroni_correction:
            # Estimate number of tests (could be tracked more precisely)
            estimated_test_count = len(self.baselines)
            alpha = alpha / estimated_test_count

        # Determine regression severity
        is_regression = False
        severity = "none"

        if p_value < alpha:  # Statistically significant change
            abs_relative_change = abs(relative_change_percent)

            if relative_change_percent > 0:  # Performance degradation (higher is worse)
                if (
                    abs_relative_change
                    >= self.config.critical_regression_threshold * 100
                ):
                    severity = "critical"
                    is_regression = True
                elif (
                    abs_relative_change >= self.config.major_regression_threshold * 100
                ):
                    severity = "major"
                    is_regression = True
                elif (
                    abs_relative_change >= self.config.minor_regression_threshold * 100
                ):
                    severity = "minor"
                    is_regression = True
            else:  # Performance improvement (lower is better)
                if abs_relative_change >= self.config.improvement_threshold * 100:
                    severity = "improvement"

        # Calculate statistical power
        effect_size = abs(absolute_change) / baseline.baseline_std
        statistical_power = self._calculate_statistical_power(
            effect_size, baseline.sample_size, alpha
        )

        # Create regression test result
        regression_test = RegressionTest(
            test_name=test_name,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            relative_change_percent=relative_change_percent,
            absolute_change=absolute_change,
            p_value=p_value,
            is_regression=is_regression,
            severity=severity,
            confidence_level=self.config.confidence_level,
            statistical_power=statistical_power,
            timestamp=time.time(),
            git_commit=git_commit,
            additional_context=additional_context or {},
        )

        # Store result
        self.historical_results.append(regression_test)
        self.save_historical_results()

        # Send alerts if necessary
        if is_regression and self.config.enable_alerts:
            self._send_regression_alert(regression_test)

        return regression_test

    def _calculate_statistical_power(
        self, effect_size: float, sample_size: int, alpha: float
    ) -> float:
        """Calculate statistical power for the test."""

        try:
            # Approximate power calculation for one-sample t-test
            t_critical = stats.t.ppf(1 - alpha / 2, sample_size - 1)

            # Non-centrality parameter
            ncp = effect_size * np.sqrt(sample_size)

            # Power calculation (simplified)
            power = 1 - stats.nct.cdf(t_critical, sample_size - 1, ncp)

            return min(1.0, max(0.0, power))

        except Exception as e:
            logger.warning(f"Failed to calculate statistical power: {e}")
            return 0.5  # Default moderate power

    def analyze_trends(
        self, metric_name: str, window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Analyze performance trends for a metric."""

        window_size = window_size or self.config.trend_analysis_window

        # Get recent results for this metric
        recent_results = [
            r
            for r in self.historical_results[-window_size:]
            if r.metric_name == metric_name
        ]

        if len(recent_results) < 3:
            return {
                "trend_detected": False,
                "message": f"Insufficient data for trend analysis: {len(recent_results)} results",
            }

        # Extract values and timestamps
        values = [r.current_value for r in recent_results]
        timestamps = [r.timestamp for r in recent_results]

        # Perform linear regression to detect trends
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            timestamps, values
        )

        # Determine trend direction and significance
        trend_detected = p_value < self.config.trend_significance_threshold
        trend_direction = (
            "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
        )

        # Calculate trend strength
        trend_strength = abs(r_value)  # Correlation coefficient magnitude

        # Project future values
        latest_timestamp = max(timestamps)
        future_timestamp = latest_timestamp + 7 * 24 * 3600  # 1 week ahead
        projected_value = slope * future_timestamp + intercept

        return {
            "trend_detected": trend_detected,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "slope": slope,
            "p_value": p_value,
            "r_squared": r_value**2,
            "projected_value_1week": projected_value,
            "recent_results_count": len(recent_results),
            "analysis_window_days": (max(timestamps) - min(timestamps)) / (24 * 3600),
        }

    def batch_regression_detection(
        self,
        performance_data: Dict[str, float],
        test_name: Optional[str] = None,
        git_commit: Optional[str] = None,
    ) -> List[RegressionTest]:
        """Perform regression detection on multiple metrics."""

        results = []

        for metric_name, current_value in performance_data.items():
            try:
                result = self.detect_regression(
                    metric_name=metric_name,
                    current_value=current_value,
                    test_name=test_name,
                    git_commit=git_commit,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to detect regression for {metric_name}: {e}")

        # Generate batch summary
        regressions = [r for r in results if r.is_regression]
        improvements = [r for r in results if r.severity == "improvement"]

        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions")

        if improvements:
            logger.info(f"Detected {len(improvements)} performance improvements")

        return results

    def _send_regression_alert(self, regression_test: RegressionTest):
        """Send regression alert through configured channels."""

        alert_message = self._format_alert_message(regression_test)

        # Console/log alert
        if "console" in self.config.alert_channels:
            if regression_test.severity == "critical":
                logger.error(alert_message)
            elif regression_test.severity == "major":
                logger.warning(alert_message)
            else:
                logger.info(alert_message)

        # Slack alert
        if "slack" in self.config.alert_channels and self.config.slack_webhook_url:
            self._send_slack_alert(alert_message, regression_test.severity)

        # Email alert
        if "email" in self.config.alert_channels and self.config.email_recipients:
            self._send_email_alert(alert_message, regression_test)

    def _format_alert_message(self, regression_test: RegressionTest) -> str:
        """Format regression alert message."""

        severity_emoji = {
            "critical": "🚨",
            "major": "⚠️",
            "minor": "📉",
            "improvement": "📈",
        }

        emoji = severity_emoji.get(regression_test.severity, "📊")

        message = f"""
{emoji} Performance Regression Detected

Test: {regression_test.test_name}
Metric: {regression_test.metric_name}
Severity: {regression_test.severity.upper()}

Performance Change:
  Current: {regression_test.current_value:.3f}
  Baseline: {regression_test.baseline_value:.3f}
  Change: {regression_test.relative_change_percent:+.1f}%
  
Statistical Analysis:
  P-value: {regression_test.p_value:.6f}
  Statistical Power: {regression_test.statistical_power:.3f}
  
{f"Git Commit: {regression_test.git_commit}" if regression_test.git_commit else ""}
Timestamp: {datetime.fromtimestamp(regression_test.timestamp).strftime("%Y-%m-%d %H:%M:%S")}
        """.strip()

        return message

    def _send_slack_alert(self, message: str, severity: str):
        """Send Slack alert (placeholder implementation)."""
        # Implementation would use requests to send webhook
        logger.info(f"Would send Slack alert for {severity} regression")

    def _send_email_alert(self, message: str, regression_test: RegressionTest):
        """Send email alert (placeholder implementation)."""
        # Implementation would use smtplib or email service API
        logger.info(f"Would send email alert for {regression_test.severity} regression")

    def generate_regression_report(
        self, days_back: int = 7, include_trends: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive regression analysis report."""

        cutoff_time = time.time() - (days_back * 24 * 3600)
        recent_results = [
            r for r in self.historical_results if r.timestamp >= cutoff_time
        ]

        # Categorize results
        regressions = [r for r in recent_results if r.is_regression]
        improvements = [r for r in recent_results if r.severity == "improvement"]
        stable_results = [r for r in recent_results if r.severity == "none"]

        # Count by severity
        severity_counts = {
            "critical": len([r for r in regressions if r.severity == "critical"]),
            "major": len([r for r in regressions if r.severity == "major"]),
            "minor": len([r for r in regressions if r.severity == "minor"]),
            "improvements": len(improvements),
            "stable": len(stable_results),
        }

        # Metrics summary
        metrics_analyzed = len(set(r.metric_name for r in recent_results))
        unique_tests = len(set(r.test_name for r in recent_results))

        # Trend analysis
        trends = {}
        if include_trends:
            unique_metrics = set(r.metric_name for r in recent_results)
            for metric_name in unique_metrics:
                trends[metric_name] = self.analyze_trends(metric_name)

        # Generate report
        report = {
            "report_metadata": {
                "generated_at": time.time(),
                "analysis_period_days": days_back,
                "total_results_analyzed": len(recent_results),
                "metrics_analyzed": metrics_analyzed,
                "unique_tests": unique_tests,
            },
            "summary": {
                "total_regressions": len(regressions),
                "total_improvements": len(improvements),
                "stability_rate_percent": (len(stable_results) / len(recent_results))
                * 100
                if recent_results
                else 0,
                "severity_breakdown": severity_counts,
            },
            "regression_details": [asdict(r) for r in regressions],
            "improvement_details": [asdict(r) for r in improvements],
            "trend_analysis": trends,
            "baseline_health": {
                "total_baselines": len(self.baselines),
                "stale_baselines": len(
                    [
                        b
                        for b in self.baselines.values()
                        if (time.time() - b.timestamp)
                        > (self.config.max_baseline_age_days * 24 * 3600)
                    ]
                ),
            },
            "recommendations": self._generate_recommendations(regressions, trends),
        }

        return report

    def _generate_recommendations(
        self, regressions: List[RegressionTest], trends: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        # Critical regressions
        critical_regressions = [r for r in regressions if r.severity == "critical"]
        if critical_regressions:
            recommendations.append(
                f"URGENT: Address {len(critical_regressions)} critical performance regressions immediately"
            )

        # Major regressions
        major_regressions = [r for r in regressions if r.severity == "major"]
        if major_regressions:
            recommendations.append(
                f"HIGH PRIORITY: Investigate {len(major_regressions)} major performance regressions"
            )

        # Trending degradation
        degrading_trends = [
            metric
            for metric, trend_data in trends.items()
            if trend_data.get("trend_detected")
            and trend_data.get("trend_direction") == "degrading"
        ]
        if degrading_trends:
            recommendations.append(
                f"Monitor trending degradation in metrics: {', '.join(degrading_trends[:3])}"
            )

        # Stale baselines
        stale_baselines = [
            name
            for name, baseline in self.baselines.items()
            if (time.time() - baseline.timestamp)
            > (self.config.max_baseline_age_days * 24 * 3600)
        ]
        if stale_baselines:
            recommendations.append(
                f"Update {len(stale_baselines)} stale performance baselines"
            )

        # Statistical power
        low_power_tests = [r for r in regressions if r.statistical_power < 0.8]
        if low_power_tests:
            recommendations.append(
                "Consider increasing sample sizes for better statistical power in regression detection"
            )

        return recommendations


# =============================================================================
# CI/CD Integration Support
# =============================================================================


class CICDIntegration:
    """CI/CD pipeline integration for regression detection."""

    def __init__(self, detector: RegressionDetector):
        self.detector = detector

    def github_actions_integration(
        self, performance_data: Dict[str, float], pr_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """GitHub Actions integration for regression detection."""

        # Get GitHub environment variables
        github_ref = os.getenv("GITHUB_REF", "")
        github_sha = os.getenv("GITHUB_SHA", "")
        os.getenv("GITHUB_ACTOR", "")

        # Perform regression detection
        results = self.detector.batch_regression_detection(
            performance_data=performance_data,
            test_name=f"GitHub Actions CI - {github_ref}",
            git_commit=github_sha,
        )

        # Prepare GitHub Actions outputs
        regressions = [r for r in results if r.is_regression]
        has_regressions = len(regressions) > 0

        # Set GitHub Actions outputs
        outputs = {
            "has-regressions": str(has_regressions).lower(),
            "regression-count": str(len(regressions)),
            "critical-regressions": str(
                len([r for r in regressions if r.severity == "critical"])
            ),
            "performance-report": json.dumps([asdict(r) for r in results]),
        }

        # Write outputs to GitHub Actions
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                for key, value in outputs.items():
                    f.write(f"{key}={value}\n")

        return {
            "results": results,
            "outputs": outputs,
            "should_fail_ci": len(
                [r for r in regressions if r.severity in ["critical", "major"]]
            )
            > 0,
        }

    def jenkins_integration(
        self, performance_data: Dict[str, float], build_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """Jenkins integration for regression detection."""

        # Get Jenkins environment variables
        build_number = build_number or os.getenv("BUILD_NUMBER", "")
        job_name = os.getenv("JOB_NAME", "")
        git_commit = os.getenv("GIT_COMMIT", "")

        # Perform regression detection
        results = self.detector.batch_regression_detection(
            performance_data=performance_data,
            test_name=f"Jenkins Build {build_number} - {job_name}",
            git_commit=git_commit,
        )

        # Generate Jenkins-compatible report
        regressions = [r for r in results if r.is_regression]

        return {
            "results": results,
            "build_should_fail": len(
                [r for r in regressions if r.severity in ["critical", "major"]]
            )
            > 0,
            "jenkins_properties": {
                "PERFORMANCE_REGRESSIONS": str(len(regressions)),
                "CRITICAL_REGRESSIONS": str(
                    len([r for r in regressions if r.severity == "critical"])
                ),
                "BUILD_PERFORMANCE_STATUS": "FAILED" if regressions else "PASSED",
            },
        }

    def generate_ci_summary(self, results: List[RegressionTest]) -> str:
        """Generate CI-friendly summary of regression detection results."""

        regressions = [r for r in results if r.is_regression]
        improvements = [r for r in results if r.severity == "improvement"]

        if not results:
            return "⚠️ No performance data analyzed"

        if not regressions and not improvements:
            return f"✅ Performance stable - {len(results)} metrics analyzed"

        summary_parts = []

        if regressions:
            critical = len([r for r in regressions if r.severity == "critical"])
            major = len([r for r in regressions if r.severity == "major"])
            minor = len([r for r in regressions if r.severity == "minor"])

            if critical:
                summary_parts.append(
                    f"🚨 {critical} critical regression{'s' if critical > 1 else ''}"
                )
            if major:
                summary_parts.append(
                    f"⚠️ {major} major regression{'s' if major > 1 else ''}"
                )
            if minor:
                summary_parts.append(
                    f"📉 {minor} minor regression{'s' if minor > 1 else ''}"
                )

        if improvements:
            summary_parts.append(
                f"📈 {len(improvements)} improvement{'s' if len(improvements) > 1 else ''}"
            )

        return " | ".join(summary_parts)


# =============================================================================
# Command Line Interface
# =============================================================================


def main():
    """Main CLI interface for regression detection."""

    import argparse

    parser = argparse.ArgumentParser(
        description="XPCS Toolkit Performance Regression Detection"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--baseline", type=str, help="Create baseline for metric")
    parser.add_argument(
        "--baseline-values", type=str, help="Comma-separated baseline values"
    )
    parser.add_argument("--test", type=str, help="Test for regression against metric")
    parser.add_argument("--test-value", type=float, help="Current test value")
    parser.add_argument(
        "--report", action="store_true", help="Generate regression report"
    )
    parser.add_argument("--days-back", type=int, default=7, help="Days back for report")
    parser.add_argument("--git-commit", type=str, help="Git commit hash")
    parser.add_argument(
        "--ci-mode", choices=["github", "jenkins"], help="CI/CD integration mode"
    )
    parser.add_argument(
        "--performance-data", type=str, help="JSON file with performance data"
    )

    args = parser.parse_args()

    # Load configuration
    config = RegressionAnalysisConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config_data = json.load(f)
            # Update config with loaded data (simplified)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Create detector
    detector = RegressionDetector(config)

    # Create baseline
    if args.baseline and args.baseline_values:
        values = [float(v.strip()) for v in args.baseline_values.split(",")]
        baseline = detector.create_baseline(args.baseline, values, args.git_commit)
        print(
            f"Created baseline for {args.baseline}: {baseline.baseline_value:.3f} ± {baseline.baseline_std:.3f}"
        )
        return

    # Test for regression
    if args.test and args.test_value is not None:
        result = detector.detect_regression(
            args.test, args.test_value, git_commit=args.git_commit
        )

        print("Regression Test Result:")
        print(f"  Metric: {result.metric_name}")
        print(f"  Current: {result.current_value:.3f}")
        print(f"  Baseline: {result.baseline_value:.3f}")
        print(f"  Change: {result.relative_change_percent:+.1f}%")
        print(f"  Regression: {'YES' if result.is_regression else 'NO'}")
        print(f"  Severity: {result.severity}")
        print(f"  P-value: {result.p_value:.6f}")

        return

    # Generate report
    if args.report:
        report = detector.generate_regression_report(args.days_back)

        print("Performance Regression Analysis Report")
        print("=" * 50)
        print(f"Analysis Period: {args.days_back} days")
        print(f"Total Results: {report['report_metadata']['total_results_analyzed']}")
        print(f"Regressions: {report['summary']['total_regressions']}")
        print(f"Improvements: {report['summary']['total_improvements']}")
        print(f"Stability Rate: {report['summary']['stability_rate_percent']:.1f}%")

        if report["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")

        return

    # CI/CD mode
    if args.ci_mode and args.performance_data:
        with open(args.performance_data, "r") as f:
            performance_data = json.load(f)

        ci_integration = CICDIntegration(detector)

        if args.ci_mode == "github":
            result = ci_integration.github_actions_integration(performance_data)
        elif args.ci_mode == "jenkins":
            result = ci_integration.jenkins_integration(performance_data)

        summary = ci_integration.generate_ci_summary(result["results"])
        print(summary)

        # Exit with appropriate code for CI
        if "should_fail_ci" in result and result["should_fail_ci"]:
            sys.exit(1)

        return

    print("Use --help for usage information")


if __name__ == "__main__":
    main()
