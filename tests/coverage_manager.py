#!/usr/bin/env python3
"""
Advanced Coverage Management for XPCS Toolkit.

This module provides comprehensive coverage measurement, analysis, and reporting
with support for module-specific targets, trend analysis, and regression detection.
"""

import base64
import io
import json
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class CoverageTarget:
    """Coverage target configuration for a module or package."""

    module_pattern: str
    target_percentage: float
    critical: bool = False
    description: str = ""


@dataclass
class CoverageMetrics:
    """Coverage metrics for a module or the entire project."""

    module_name: str
    statements: int
    missing: int
    branches: int
    partial_branches: int
    coverage_percentage: float
    branch_coverage_percentage: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def lines_covered(self) -> int:
        """Number of lines covered."""
        return self.statements - self.missing

    @property
    def meets_target(self) -> bool:
        """Check if coverage meets the minimum target."""
        # Will be set by CoverageManager based on targets
        return hasattr(self, "_target") and self.coverage_percentage >= self._target


@dataclass
class CoverageReport:
    """Comprehensive coverage report."""

    timestamp: datetime
    overall_coverage: float
    branch_coverage: float
    total_statements: int
    total_missing: int
    module_metrics: list[CoverageMetrics] = field(default_factory=list)
    targets_met: list[str] = field(default_factory=list)
    targets_missed: list[str] = field(default_factory=list)
    regression_alerts: list[str] = field(default_factory=list)


class CoverageManager:
    """Advanced coverage measurement and analysis system."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the coverage manager."""
        self.project_root = project_root or self._find_project_root()
        self.coverage_db = self.project_root / ".coverage_history.db"
        self.coverage_dir = self.project_root / "coverage_reports"
        self.coverage_dir.mkdir(exist_ok=True)

        # Initialize database
        self._init_coverage_db()

        # Define coverage targets for different module types
        self.coverage_targets = [
            # Critical scientific modules (95% minimum)
            CoverageTarget(
                "xpcs_toolkit/module/g2mod.py", 95.0, True, "G2 correlation analysis"
            ),
            CoverageTarget(
                "xpcs_toolkit/module/twotime.py", 95.0, True, "Two-time correlation"
            ),
            CoverageTarget(
                "xpcs_toolkit/module/saxs1d.py", 95.0, True, "SAXS 1D analysis"
            ),
            CoverageTarget(
                "xpcs_toolkit/module/saxs2d.py", 95.0, True, "SAXS 2D analysis"
            ),
            CoverageTarget(
                "xpcs_toolkit/helper/fitting.py",
                95.0,
                True,
                "Scientific fitting algorithms",
            ),
            CoverageTarget(
                "xpcs_toolkit/fileIO/qmap_utils.py", 95.0, True, "Q-map calculations"
            ),
            # Core modules (85% minimum)
            CoverageTarget(
                "xpcs_toolkit/xpcs_file.py", 85.0, True, "Core data container"
            ),
            CoverageTarget(
                "xpcs_toolkit/fileIO/hdf_reader.py", 85.0, True, "HDF5 I/O operations"
            ),
            CoverageTarget(
                "xpcs_toolkit/viewer_kernel.py", 85.0, True, "Application kernel"
            ),
            CoverageTarget(
                "xpcs_toolkit/threading/", 85.0, False, "Threading subsystem"
            ),
            CoverageTarget("xpcs_toolkit/utils/", 85.0, False, "Utility modules"),
            # Supporting modules (75% minimum)
            CoverageTarget(
                "xpcs_toolkit/module/average_toolbox.py", 75.0, False, "Data averaging"
            ),
            CoverageTarget(
                "xpcs_toolkit/module/stability.py", 75.0, False, "Stability analysis"
            ),
            CoverageTarget(
                "xpcs_toolkit/plothandler/", 75.0, False, "Plotting subsystem"
            ),
            CoverageTarget(
                "xpcs_toolkit/file_locator.py", 75.0, False, "File management"
            ),
            # GUI modules (50% minimum - hard to test)
            CoverageTarget(
                "xpcs_toolkit/xpcs_viewer.py", 50.0, False, "Main GUI application"
            ),
            CoverageTarget("xpcs_toolkit/viewer_ui.py", 30.0, False, "UI components"),
            CoverageTarget("xpcs_toolkit/ui/", 30.0, False, "UI resources"),
        ]

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        return Path.cwd()

    def _init_coverage_db(self):
        """Initialize the coverage history database."""
        with sqlite3.connect(self.coverage_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    statements INTEGER NOT NULL,
                    missing INTEGER NOT NULL,
                    branches INTEGER NOT NULL,
                    partial_branches INTEGER NOT NULL,
                    coverage_percentage REAL NOT NULL,
                    branch_coverage_percentage REAL NOT NULL,
                    git_commit TEXT,
                    test_profile TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_coverage REAL NOT NULL,
                    branch_coverage REAL NOT NULL,
                    total_statements INTEGER NOT NULL,
                    total_missing INTEGER NOT NULL,
                    targets_met INTEGER NOT NULL,
                    targets_total INTEGER NOT NULL,
                    git_commit TEXT,
                    test_profile TEXT
                )
            """)

    def run_coverage_analysis(
        self, test_pattern: str = "tests/", test_profile: str = "full"
    ) -> CoverageReport:
        """Run comprehensive coverage analysis."""
        print(f"Running coverage analysis with profile: {test_profile}")

        # Clean previous coverage data
        coverage_file = self.project_root / ".coverage"
        if coverage_file.exists():
            coverage_file.unlink()

        # Run tests with coverage
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_pattern,
            "--cov=xpcs_toolkit",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=json",
            "--cov-branch",
            "--cov-config=pyproject.toml",
            "--tb=short",
            "--disable-warnings",
        ]

        print(f"Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout
            )

            if result.returncode not in [0, 1]:  # 1 is acceptable (some test failures)
                print(f"Coverage analysis warning: return code {result.returncode}")
                print(f"STDERR: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("Coverage analysis timed out")
            return CoverageReport(
                timestamp=datetime.now(),
                overall_coverage=0.0,
                branch_coverage=0.0,
                total_statements=0,
                total_missing=0,
                regression_alerts=["Coverage analysis timed out"],
            )

        # Parse coverage results
        return self._parse_coverage_results(test_profile)

    def _parse_coverage_results(self, test_profile: str) -> CoverageReport:
        """Parse coverage results from XML and JSON reports."""
        xml_path = self.project_root / "coverage.xml"
        self.project_root / "coverage.json"

        if not xml_path.exists():
            return CoverageReport(
                timestamp=datetime.now(),
                overall_coverage=0.0,
                branch_coverage=0.0,
                total_statements=0,
                total_missing=0,
                regression_alerts=["No coverage data generated"],
            )

        # Parse XML coverage report
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Overall metrics
        overall_coverage = float(root.attrib.get("line-rate", 0)) * 100
        branch_coverage = float(root.attrib.get("branch-rate", 0)) * 100

        # Module-specific metrics
        module_metrics = []

        for package in root.findall(".//package"):
            for class_elem in package.findall("classes/class"):
                filename = class_elem.attrib["filename"]

                # Convert to module name
                if filename.startswith("xpcs_toolkit/"):
                    module_name = filename

                    lines = class_elem.find("lines")
                    if lines is not None:
                        statements = len(lines.findall("line"))
                        missing = len(
                            [
                                line
                                for line in lines.findall("line")
                                if line.attrib.get("hits", "0") == "0"
                            ]
                        )

                        # Branch coverage calculation
                        branches = len(
                            [
                                line
                                for line in lines.findall("line")
                                if "branch" in line.attrib
                            ]
                        )
                        partial_branches = len(
                            [
                                line
                                for line in lines.findall("line")
                                if line.attrib.get("branch", "false") == "partial"
                            ]
                        )

                        if statements > 0:
                            coverage_pct = ((statements - missing) / statements) * 100
                            branch_pct = (
                                ((branches - partial_branches) / branches * 100)
                                if branches > 0
                                else 100
                            )

                            metrics = CoverageMetrics(
                                module_name=module_name,
                                statements=statements,
                                missing=missing,
                                branches=branches,
                                partial_branches=partial_branches,
                                coverage_percentage=coverage_pct,
                                branch_coverage_percentage=branch_pct,
                            )

                            module_metrics.append(metrics)

        # Create comprehensive report
        report = CoverageReport(
            timestamp=datetime.now(),
            overall_coverage=overall_coverage,
            branch_coverage=branch_coverage,
            total_statements=sum(m.statements for m in module_metrics),
            total_missing=sum(m.missing for m in module_metrics),
            module_metrics=module_metrics,
        )

        # Check targets and detect regressions
        self._analyze_targets(report)
        self._detect_regressions(report, test_profile)

        # Save to database
        self._save_coverage_data(report, test_profile)

        return report

    def _analyze_targets(self, report: CoverageReport):
        """Analyze coverage against defined targets."""
        for target in self.coverage_targets:
            # Find matching modules
            matching_modules = []

            for metrics in report.module_metrics:
                if (
                    target.module_pattern.endswith("/")
                    and metrics.module_name.startswith(target.module_pattern)
                ) or (metrics.module_name == target.module_pattern):
                    matching_modules.append(metrics)
                    metrics._target = target.target_percentage

            if matching_modules:
                # Calculate average coverage for pattern
                avg_coverage = sum(
                    m.coverage_percentage for m in matching_modules
                ) / len(matching_modules)

                if avg_coverage >= target.target_percentage:
                    report.targets_met.append(
                        f"{target.module_pattern}: {avg_coverage:.1f}% (target: {target.target_percentage}%)"
                    )
                else:
                    status = "CRITICAL" if target.critical else "WARNING"
                    report.targets_missed.append(
                        f"[{status}] {target.module_pattern}: {avg_coverage:.1f}% < {target.target_percentage}% target"
                    )

    def _detect_regressions(self, report: CoverageReport, test_profile: str):
        """Detect coverage regressions compared to previous runs."""
        with sqlite3.connect(self.coverage_db) as conn:
            # Get recent coverage data for comparison
            cursor = conn.execute(
                """
                SELECT module_name, coverage_percentage, timestamp
                FROM coverage_history
                WHERE test_profile = ?
                AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """,
                (test_profile,),
            )

            recent_data = {}
            for row in cursor.fetchall():
                module_name, coverage, _timestamp = row
                if module_name not in recent_data:
                    recent_data[module_name] = coverage

            # Check for regressions
            for metrics in report.module_metrics:
                if metrics.module_name in recent_data:
                    previous_coverage = recent_data[metrics.module_name]
                    regression_threshold = 5.0  # 5% drop is significant

                    if (
                        previous_coverage - metrics.coverage_percentage
                        > regression_threshold
                    ):
                        report.regression_alerts.append(
                            f"REGRESSION: {metrics.module_name} dropped from "
                            f"{previous_coverage:.1f}% to {metrics.coverage_percentage:.1f}%"
                        )

    def _save_coverage_data(self, report: CoverageReport, test_profile: str):
        """Save coverage data to history database."""
        git_commit = self._get_git_commit()

        with sqlite3.connect(self.coverage_db) as conn:
            # Save session data
            conn.execute(
                """
                INSERT INTO coverage_sessions
                (timestamp, overall_coverage, branch_coverage, total_statements,
                 total_missing, targets_met, targets_total, git_commit, test_profile)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    report.timestamp.isoformat(),
                    report.overall_coverage,
                    report.branch_coverage,
                    report.total_statements,
                    report.total_missing,
                    len(report.targets_met),
                    len(report.targets_met) + len(report.targets_missed),
                    git_commit,
                    test_profile,
                ),
            )

            # Save module-specific data
            for metrics in report.module_metrics:
                conn.execute(
                    """
                    INSERT INTO coverage_history
                    (timestamp, module_name, statements, missing, branches,
                     partial_branches, coverage_percentage, branch_coverage_percentage,
                     git_commit, test_profile)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metrics.timestamp.isoformat(),
                        metrics.module_name,
                        metrics.statements,
                        metrics.missing,
                        metrics.branches,
                        metrics.partial_branches,
                        metrics.coverage_percentage,
                        metrics.branch_coverage_percentage,
                        git_commit,
                        test_profile,
                    ),
                )

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None

    def generate_detailed_report(self, report: CoverageReport) -> str:
        """Generate a detailed human-readable coverage report."""
        lines = []
        lines.append("XPCS TOOLKIT COVERAGE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Overall metrics
        lines.append("OVERALL COVERAGE METRICS")
        lines.append("-" * 30)
        lines.append(f"Line Coverage: {report.overall_coverage:.2f}%")
        lines.append(f"Branch Coverage: {report.branch_coverage:.2f}%")
        lines.append(f"Total Statements: {report.total_statements:,}")
        lines.append(f"Missing Statements: {report.total_missing:,}")
        lines.append("")

        # Target analysis
        lines.append("COVERAGE TARGET ANALYSIS")
        lines.append("-" * 30)

        if report.targets_met:
            lines.append("✅ TARGETS MET:")
            for target in report.targets_met:
                lines.append(f"  {target}")
            lines.append("")

        if report.targets_missed:
            lines.append("❌ TARGETS MISSED:")
            for target in report.targets_missed:
                lines.append(f"  {target}")
            lines.append("")

        # Regression alerts
        if report.regression_alerts:
            lines.append("🚨 REGRESSION ALERTS:")
            for alert in report.regression_alerts:
                lines.append(f"  {alert}")
            lines.append("")

        # Module breakdown (top 10 lowest coverage)
        lines.append("MODULES WITH LOWEST COVERAGE")
        lines.append("-" * 30)

        sorted_modules = sorted(
            report.module_metrics, key=lambda x: x.coverage_percentage
        )[:10]

        for metrics in sorted_modules:
            lines.append(
                f"{metrics.module_name:50} {metrics.coverage_percentage:6.1f}%"
            )

        lines.append("")

        # Critical modules status
        lines.append("CRITICAL MODULES STATUS")
        lines.append("-" * 30)

        critical_modules = [
            m
            for m in report.module_metrics
            if any(
                t.critical
                and (
                    t.module_pattern == m.module_name
                    or (
                        t.module_pattern.endswith("/")
                        and m.module_name.startswith(t.module_pattern)
                    )
                )
                for t in self.coverage_targets
            )
        ]

        for metrics in critical_modules:
            status = (
                "✅"
                if hasattr(metrics, "_target")
                and metrics.coverage_percentage >= metrics._target
                else "❌"
            )
            lines.append(
                f"{status} {metrics.module_name:40} {metrics.coverage_percentage:6.1f}%"
            )

        return "\n".join(lines)

    def generate_json_report(self, report: CoverageReport) -> str:
        """Generate coverage report in JSON format."""
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_metrics": {
                "overall_coverage": report.overall_coverage,
                "branch_coverage": report.branch_coverage,
                "total_statements": report.total_statements,
                "total_missing": report.total_missing,
                "lines_covered": report.total_statements - report.total_missing,
            },
            "targets": {
                "total_targets": len(self.coverage_targets),
                "targets_met": len(report.targets_met),
                "targets_missed": len(report.targets_missed),
                "met_list": report.targets_met,
                "missed_list": report.targets_missed,
                "success_rate": (
                    len(report.targets_met) / len(self.coverage_targets) * 100
                )
                if self.coverage_targets
                else 0,
            },
            "regression_alerts": {
                "count": len(report.regression_alerts),
                "alerts": report.regression_alerts,
            },
            "module_metrics": [],
        }

        # Add module-specific metrics
        for metrics in report.module_metrics:
            module_data = asdict(metrics)
            module_data["timestamp"] = module_data["timestamp"].isoformat()

            # Find matching target
            matching_target = None
            for target in self.coverage_targets:
                if target.module_pattern == metrics.module_name or (
                    target.module_pattern.endswith("/")
                    and metrics.module_name.startswith(target.module_pattern)
                ):
                    matching_target = {
                        "target_percentage": target.target_percentage,
                        "critical": target.critical,
                        "description": target.description,
                        "meets_target": metrics.coverage_percentage
                        >= target.target_percentage,
                    }
                    break

            module_data["target_info"] = matching_target
            report_data["module_metrics"].append(module_data)

        return json.dumps(report_data, indent=2, default=str)

    def generate_html_report(self, report: CoverageReport) -> str:
        """Generate coverage report in HTML format with embedded visualization."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XPCS Toolkit Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; }}
        .targets {{ margin: 20px 0; }}
        .target-met {{ color: #27ae60; }}
        .target-missed {{ color: #e74c3c; }}
        .module-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .module-table th, .module-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .module-table th {{ background-color: #f2f2f2; }}
        .critical {{ background-color: #fff3cd; }}
        .low-coverage {{ background-color: #f8d7da; }}
        .good-coverage {{ background-color: #d4edda; }}
        .chart-container {{ margin: 20px 0; }}
        .alert {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>XPCS Toolkit Coverage Report</h1>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{overall_coverage:.1f}%</div>
            <div class="metric-label">Line Coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{branch_coverage:.1f}%</div>
            <div class="metric-label">Branch Coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{lines_covered:,}</div>
            <div class="metric-label">Lines Covered</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{targets_success_rate:.1f}%</div>
            <div class="metric-label">Targets Met</div>
        </div>
    </div>

    {regression_alerts_section}

    <div class="targets">
        <h2>Coverage Targets</h2>
        {targets_met_section}
        {targets_missed_section}
    </div>

    <div class="modules">
        <h2>Module Coverage Details</h2>
        <table class="module-table">
            <thead>
                <tr>
                    <th>Module</th>
                    <th>Coverage</th>
                    <th>Branch Coverage</th>
                    <th>Lines</th>
                    <th>Missing</th>
                    <th>Target</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {module_rows}
            </tbody>
        </table>
    </div>

    {trend_chart}
</body>
</html>
        """

        # Build regression alerts section
        regression_alerts_section = ""
        if report.regression_alerts:
            alerts_html = "".join(
                [
                    f'<div class="alert">{alert}</div>'
                    for alert in report.regression_alerts
                ]
            )
            regression_alerts_section = f"""
            <div class="regression-alerts">
                <h2>🚨 Regression Alerts</h2>
                {alerts_html}
            </div>
            """

        # Build targets sections
        targets_met_section = ""
        if report.targets_met:
            met_items = "".join(
                [
                    f'<li class="target-met">✅ {target}</li>'
                    for target in report.targets_met
                ]
            )
            targets_met_section = f"<h3>Targets Met</h3><ul>{met_items}</ul>"

        targets_missed_section = ""
        if report.targets_missed:
            missed_items = "".join(
                [
                    f'<li class="target-missed">❌ {target}</li>'
                    for target in report.targets_missed
                ]
            )
            targets_missed_section = f"<h3>Targets Missed</h3><ul>{missed_items}</ul>"

        # Build module table rows
        module_rows = []
        for metrics in sorted(
            report.module_metrics, key=lambda x: x.coverage_percentage
        ):
            # Find matching target
            target_info = "N/A"
            row_class = ""
            status = ""

            for target in self.coverage_targets:
                if target.module_pattern == metrics.module_name or (
                    target.module_pattern.endswith("/")
                    and metrics.module_name.startswith(target.module_pattern)
                ):
                    target_info = f"{target.target_percentage:.1f}%"
                    if target.critical:
                        row_class = "critical"

                    if metrics.coverage_percentage >= target.target_percentage:
                        status = "✅"
                        if row_class != "critical":
                            row_class = "good-coverage"
                    else:
                        status = "❌"
                        row_class = "low-coverage"
                    break

            module_rows.append(f"""
            <tr class="{row_class}">
                <td>{metrics.module_name}</td>
                <td>{metrics.coverage_percentage:.1f}%</td>
                <td>{metrics.branch_coverage_percentage:.1f}%</td>
                <td>{metrics.statements}</td>
                <td>{metrics.missing}</td>
                <td>{target_info}</td>
                <td>{status}</td>
            </tr>
            """)

        # Generate trend chart (placeholder for now)
        trend_chart = ""
        if MATPLOTLIB_AVAILABLE:
            trend_chart = self._generate_trend_chart_html()

        return html_template.format(
            timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            overall_coverage=report.overall_coverage,
            branch_coverage=report.branch_coverage,
            lines_covered=report.total_statements - report.total_missing,
            targets_success_rate=(
                len(report.targets_met) / len(self.coverage_targets) * 100
            )
            if self.coverage_targets
            else 0,
            regression_alerts_section=regression_alerts_section,
            targets_met_section=targets_met_section,
            targets_missed_section=targets_missed_section,
            module_rows="".join(module_rows),
            trend_chart=trend_chart,
        )

    def generate_xml_report(self, report: CoverageReport) -> str:
        """Generate coverage report in XML format."""
        root = ET.Element("coverage_report")
        root.set("timestamp", report.timestamp.isoformat())
        root.set("version", "1.0")

        # Overall metrics
        overall = ET.SubElement(root, "overall_metrics")
        ET.SubElement(overall, "overall_coverage").text = str(report.overall_coverage)
        ET.SubElement(overall, "branch_coverage").text = str(report.branch_coverage)
        ET.SubElement(overall, "total_statements").text = str(report.total_statements)
        ET.SubElement(overall, "total_missing").text = str(report.total_missing)

        # Targets
        targets = ET.SubElement(root, "targets")
        targets.set("total", str(len(self.coverage_targets)))
        targets.set("met", str(len(report.targets_met)))
        targets.set("missed", str(len(report.targets_missed)))

        met_elem = ET.SubElement(targets, "targets_met")
        for target in report.targets_met:
            ET.SubElement(met_elem, "target").text = target

        missed_elem = ET.SubElement(targets, "targets_missed")
        for target in report.targets_missed:
            ET.SubElement(missed_elem, "target").text = target

        # Regression alerts
        if report.regression_alerts:
            alerts = ET.SubElement(root, "regression_alerts")
            for alert in report.regression_alerts:
                ET.SubElement(alerts, "alert").text = alert

        # Module metrics
        modules = ET.SubElement(root, "modules")
        for metrics in report.module_metrics:
            module = ET.SubElement(modules, "module")
            module.set("name", metrics.module_name)
            module.set("coverage", str(metrics.coverage_percentage))
            module.set("branch_coverage", str(metrics.branch_coverage_percentage))
            module.set("statements", str(metrics.statements))
            module.set("missing", str(metrics.missing))
            module.set("branches", str(metrics.branches))
            module.set("partial_branches", str(metrics.partial_branches))

        return ET.tostring(root, encoding="unicode", method="xml")

    def _generate_trend_chart_html(self) -> str:
        """Generate embedded trend chart as base64 encoded image."""
        if not MATPLOTLIB_AVAILABLE:
            return "<p>Matplotlib not available for trend charts</p>"

        try:
            # Get trend data for last 30 days
            trends = self.export_coverage_trends(30)

            if not trends["timestamps"]:
                return "<p>No trend data available</p>"

            # Create the plot
            _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Convert timestamps to datetime objects
            timestamps = [
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                for ts in trends["timestamps"]
            ]

            # Coverage trends
            ax1.plot(
                timestamps,
                trends["overall_coverage"],
                label="Line Coverage",
                marker="o",
            )
            ax1.plot(
                timestamps,
                trends["branch_coverage"],
                label="Branch Coverage",
                marker="s",
            )
            ax1.set_ylabel("Coverage %")
            ax1.set_title("Coverage Trends (Last 30 Days)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Target success rate
            ax2.plot(
                timestamps,
                trends["target_success_rate"],
                label="Target Success Rate",
                marker="^",
                color="green",
            )
            ax2.set_ylabel("Success Rate %")
            ax2.set_xlabel("Date")
            ax2.set_title("Target Achievement Rate")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                ax.xaxis.set_major_locator(
                    mdates.DayLocator(interval=max(1, len(timestamps) // 10))
                )
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"""
            <div class="chart-container">
                <h2>Coverage Trends</h2>
                <img src="data:image/png;base64,{image_base64}" alt="Coverage Trends Chart" style="max-width: 100%; height: auto;">
            </div>
            """

        except Exception as e:
            return f"<p>Error generating trend chart: {e}</p>"

    def export_coverage_trends(self, days: int = 30) -> dict[str, Any]:
        """Export coverage trend data for the last N days."""
        with sqlite3.connect(self.coverage_db) as conn:
            cursor = conn.execute(f"""
                SELECT timestamp, overall_coverage, branch_coverage,
                       targets_met, targets_total, test_profile
                FROM coverage_sessions
                WHERE timestamp > datetime('now', '-{days} days')
                ORDER BY timestamp ASC
            """)

            trend_data = {
                "timestamps": [],
                "overall_coverage": [],
                "branch_coverage": [],
                "target_success_rate": [],
                "profiles": [],
            }

            for row in cursor.fetchall():
                timestamp, overall_cov, branch_cov, met, total, profile = row
                trend_data["timestamps"].append(timestamp)
                trend_data["overall_coverage"].append(overall_cov)
                trend_data["branch_coverage"].append(branch_cov)
                trend_data["target_success_rate"].append(
                    (met / total * 100) if total > 0 else 0
                )
                trend_data["profiles"].append(profile)

            return trend_data

    def cleanup_old_data(self, days: int = 90):
        """Clean up old coverage data to keep database size manageable."""
        with sqlite3.connect(self.coverage_db) as conn:
            conn.execute(f"""
                DELETE FROM coverage_history
                WHERE timestamp < datetime('now', '-{days} days')
            """)

            conn.execute(f"""
                DELETE FROM coverage_sessions
                WHERE timestamp < datetime('now', '-{days} days')
            """)

            print(f"Cleaned up coverage data older than {days} days")


if __name__ == "__main__":
    """Command-line interface for coverage management."""
    import argparse

    parser = argparse.ArgumentParser(description="XPCS Toolkit Coverage Manager")
    parser.add_argument("--analyze", action="store_true", help="Run coverage analysis")
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        help="Test profile to use (fast, core, full, etc.)",
    )
    parser.add_argument(
        "--test-pattern", type=str, default="tests/", help="Test pattern to run"
    )
    parser.add_argument(
        "--trends",
        type=int,
        metavar="DAYS",
        help="Export coverage trends for last N days",
    )
    parser.add_argument(
        "--cleanup", type=int, metavar="DAYS", help="Clean up data older than N days"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed coverage report"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "html", "xml"],
        default="text",
        help="Report output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="FILE",
        help="Output file for reports (default: stdout)",
    )
    parser.add_argument(
        "--export-trends", action="store_true", help="Export trend data in JSON format"
    )
    parser.add_argument(
        "--generate-all-reports",
        action="store_true",
        help="Generate reports in all formats",
    )

    args = parser.parse_args()

    manager = CoverageManager()

    def save_report(content: str, filename: str):
        """Helper function to save report content."""
        if args.output:
            output_path = Path(args.output)
            if filename != args.output:
                output_path = output_path.parent / filename

            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(content)
            print(f"Report saved to: {output_path}")
        else:
            print(content)

    if args.analyze or len(sys.argv) == 1:
        report = manager.run_coverage_analysis(args.test_pattern, args.profile)

        if args.generate_all_reports:
            # Generate all format reports
            formats = {
                "text": manager.generate_detailed_report(report),
                "json": manager.generate_json_report(report),
                "html": manager.generate_html_report(report),
                "xml": manager.generate_xml_report(report),
            }

            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            for fmt, content in formats.items():
                if args.output:
                    filename = f"coverage_report_{timestamp}.{fmt}"
                    save_report(content, filename)
                else:
                    print(f"\n{'=' * 50}")
                    print(f"{fmt.upper()} REPORT")
                    print(f"{'=' * 50}")
                    print(content)

        elif args.report:
            # Generate report in specified format
            if args.format == "text":
                content = manager.generate_detailed_report(report)
            elif args.format == "json":
                content = manager.generate_json_report(report)
            elif args.format == "html":
                content = manager.generate_html_report(report)
            elif args.format == "xml":
                content = manager.generate_xml_report(report)

            save_report(content, f"coverage_report.{args.format}")

        else:
            # Basic summary output
            print("Coverage Analysis Complete:")
            print(f"  Overall Coverage: {report.overall_coverage:.2f}%")
            print(f"  Branch Coverage: {report.branch_coverage:.2f}%")
            print(f"  Targets Met: {len(report.targets_met)}")
            print(f"  Targets Missed: {len(report.targets_missed)}")
            print(f"  Regression Alerts: {len(report.regression_alerts)}")

    if args.export_trends or args.trends:
        days = args.trends if args.trends else 30
        trends = manager.export_coverage_trends(days)

        if args.export_trends:
            trend_json = json.dumps(trends, indent=2, default=str)
            if args.output:
                trend_filename = f"coverage_trends_{days}days.json"
                save_report(trend_json, trend_filename)
            else:
                print(trend_json)
        else:
            print(f"Coverage trends exported for last {days} days")
            print(f"Data points: {len(trends['timestamps'])}")

    if args.cleanup:
        manager.cleanup_old_data(args.cleanup)
