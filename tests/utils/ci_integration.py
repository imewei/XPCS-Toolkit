#!/usr/bin/env python3
"""
CI/CD Integration Utilities for XPCS Toolkit

This module provides utilities for enhanced CI/CD integration, including
test result reporting, artifact generation, and environment detection.

Created: 2025-09-16
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


@dataclass
class TestResult:
    """Individual test result for CI reporting."""

    test_id: str
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    message: str | None = None
    traceback: str | None = None
    markers: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class TestSuite:
    """Test suite results for CI reporting."""

    name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    timestamp: float = field(default_factory=time.time)
    results: list[TestResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 1.0
        return self.passed / self.total_tests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "success_rate": self.success_rate,
            "results": [result.to_dict() for result in self.results],
        }


class CIEnvironmentDetector:
    """Detect CI/CD environment and extract relevant information."""

    @staticmethod
    def detect_ci_environment() -> dict[str, Any]:
        """Detect current CI/CD environment."""
        env_info = {
            "is_ci": False,
            "ci_provider": "unknown",
            "branch": None,
            "commit": None,
            "build_number": None,
            "pull_request": None,
            "runner_os": sys.platform,
            "python_version": sys.version,
        }

        # GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            env_info.update(
                {
                    "is_ci": True,
                    "ci_provider": "github_actions",
                    "branch": os.getenv("GITHUB_REF_NAME"),
                    "commit": os.getenv("GITHUB_SHA"),
                    "build_number": os.getenv("GITHUB_RUN_NUMBER"),
                    "pull_request": os.getenv("GITHUB_EVENT_NAME") == "pull_request",
                    "repository": os.getenv("GITHUB_REPOSITORY"),
                    "actor": os.getenv("GITHUB_ACTOR"),
                    "workflow": os.getenv("GITHUB_WORKFLOW"),
                    "runner_os": os.getenv("RUNNER_OS", sys.platform),
                }
            )

        # GitLab CI
        elif os.getenv("GITLAB_CI"):
            env_info.update(
                {
                    "is_ci": True,
                    "ci_provider": "gitlab_ci",
                    "branch": os.getenv("CI_COMMIT_REF_NAME"),
                    "commit": os.getenv("CI_COMMIT_SHA"),
                    "build_number": os.getenv("CI_PIPELINE_ID"),
                    "pull_request": os.getenv("CI_MERGE_REQUEST_ID") is not None,
                    "repository": os.getenv("CI_PROJECT_PATH"),
                    "runner_id": os.getenv("CI_RUNNER_ID"),
                    "job_name": os.getenv("CI_JOB_NAME"),
                }
            )

        # Jenkins
        elif os.getenv("JENKINS_URL"):
            env_info.update(
                {
                    "is_ci": True,
                    "ci_provider": "jenkins",
                    "branch": os.getenv("GIT_BRANCH"),
                    "commit": os.getenv("GIT_COMMIT"),
                    "build_number": os.getenv("BUILD_NUMBER"),
                    "job_name": os.getenv("JOB_NAME"),
                    "build_url": os.getenv("BUILD_URL"),
                }
            )

        # CircleCI
        elif os.getenv("CIRCLECI"):
            env_info.update(
                {
                    "is_ci": True,
                    "ci_provider": "circleci",
                    "branch": os.getenv("CIRCLE_BRANCH"),
                    "commit": os.getenv("CIRCLE_SHA1"),
                    "build_number": os.getenv("CIRCLE_BUILD_NUM"),
                    "pull_request": os.getenv("CIRCLE_PULL_REQUEST") is not None,
                    "repository": os.getenv("CIRCLE_PROJECT_REPONAME"),
                }
            )

        # Travis CI
        elif os.getenv("TRAVIS"):
            env_info.update(
                {
                    "is_ci": True,
                    "ci_provider": "travis_ci",
                    "branch": os.getenv("TRAVIS_BRANCH"),
                    "commit": os.getenv("TRAVIS_COMMIT"),
                    "build_number": os.getenv("TRAVIS_BUILD_NUMBER"),
                    "pull_request": os.getenv("TRAVIS_PULL_REQUEST") != "false",
                    "repository": os.getenv("TRAVIS_REPO_SLUG"),
                }
            )

        # Azure Pipelines
        elif os.getenv("AZURE_HTTP_USER_AGENT"):
            env_info.update(
                {
                    "is_ci": True,
                    "ci_provider": "azure_pipelines",
                    "branch": os.getenv("BUILD_SOURCEBRANCH"),
                    "commit": os.getenv("BUILD_SOURCEVERSION"),
                    "build_number": os.getenv("BUILD_BUILDNUMBER"),
                    "repository": os.getenv("BUILD_REPOSITORY_NAME"),
                }
            )

        # Generic CI detection
        elif any(
            os.getenv(var) for var in ["CI", "CONTINUOUS_INTEGRATION", "BUILD_NUMBER"]
        ):
            env_info["is_ci"] = True
            env_info["ci_provider"] = "generic"

        return env_info


class TestReportGenerator:
    """Generate test reports in various formats for CI/CD systems."""

    def __init__(self, output_dir: Path | None = None):
        if output_dir is None:
            output_dir = Path.cwd() / "test-reports"

        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_junit_xml(
        self, test_suite: TestSuite, filename: str = "test-results.xml"
    ) -> Path:
        """Generate JUnit XML report for CI systems."""
        output_path = self.output_dir / filename

        # Create root element
        testsuites = ET.Element("testsuites")
        testsuites.set("tests", str(test_suite.total_tests))
        testsuites.set("failures", str(test_suite.failed))
        testsuites.set("errors", str(test_suite.errors))
        testsuites.set("time", f"{test_suite.duration:.3f}")

        # Create testsuite element
        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", test_suite.name)
        testsuite.set("tests", str(test_suite.total_tests))
        testsuite.set("failures", str(test_suite.failed))
        testsuite.set("errors", str(test_suite.errors))
        testsuite.set("skipped", str(test_suite.skipped))
        testsuite.set("time", f"{test_suite.duration:.3f}")
        testsuite.set(
            "timestamp",
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(test_suite.timestamp)),
        )

        # Add individual test cases
        for result in test_suite.results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set(
                "classname",
                result.test_id.split("::")[0] if "::" in result.test_id else "unknown",
            )
            testcase.set("name", result.test_name)
            testcase.set("time", f"{result.duration:.3f}")

            # Add failure/error/skipped information
            if result.status == "failed":
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", result.message or "Test failed")
                if result.traceback:
                    failure.text = result.traceback

            elif result.status == "error":
                error = ET.SubElement(testcase, "error")
                error.set("message", result.message or "Test error")
                if result.traceback:
                    error.text = result.traceback

            elif result.status == "skipped":
                skipped = ET.SubElement(testcase, "skipped")
                skipped.set("message", result.message or "Test skipped")

            # Add properties for markers and categories
            if result.markers or result.categories:
                properties = ET.SubElement(testcase, "properties")

                for marker in result.markers:
                    prop = ET.SubElement(properties, "property")
                    prop.set("name", "marker")
                    prop.set("value", marker)

                for category in result.categories:
                    prop = ET.SubElement(properties, "property")
                    prop.set("name", "category")
                    prop.set("value", category)

        # Write to file
        tree = ET.ElementTree(testsuites)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        return output_path

    def generate_json_report(
        self, test_suite: TestSuite, filename: str = "test-results.json"
    ) -> Path:
        """Generate JSON report with detailed test information."""
        output_path = self.output_dir / filename

        # Add CI environment information
        ci_info = CIEnvironmentDetector.detect_ci_environment()
        report_data = {
            "test_suite": test_suite.to_dict(),
            "ci_environment": ci_info,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "report_version": "1.0",
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, sort_keys=True)

        return output_path

    def generate_markdown_summary(
        self, test_suite: TestSuite, filename: str = "test-summary.md"
    ) -> Path:
        """Generate Markdown summary for PR comments or documentation."""
        output_path = self.output_dir / filename

        ci_info = CIEnvironmentDetector.detect_ci_environment()

        # Generate markdown content
        content = []
        content.append("# Test Results Summary")
        content.append("")

        # Overall results
        content.append("## Overall Results")
        content.append("")
        content.append(f"- **Total Tests:** {test_suite.total_tests}")
        content.append(f"- **Passed:** {test_suite.passed} ✅")
        content.append(f"- **Failed:** {test_suite.failed} ❌")
        content.append(f"- **Skipped:** {test_suite.skipped} ⏭️")
        content.append(f"- **Errors:** {test_suite.errors} 💥")
        content.append(f"- **Success Rate:** {test_suite.success_rate:.1%}")
        content.append(f"- **Duration:** {test_suite.duration:.2f}s")
        content.append("")

        # Status badge
        status_emoji = (
            "✅" if test_suite.failed == 0 and test_suite.errors == 0 else "❌"
        )
        content.append(
            f"**Status:** {status_emoji} {'PASSED' if test_suite.failed == 0 and test_suite.errors == 0 else 'FAILED'}"
        )
        content.append("")

        # Failed tests section
        failed_tests = [
            r for r in test_suite.results if r.status in ("failed", "error")
        ]
        if failed_tests:
            content.append("## Failed Tests")
            content.append("")
            for result in failed_tests:
                content.append(f"- **{result.test_name}**")
                if result.message:
                    content.append(f"  - Error: `{result.message}`")
                content.append(f"  - Duration: {result.duration:.3f}s")
                content.append("")

        # Performance insights
        slow_tests = [r for r in test_suite.results if r.duration > 1.0]
        if slow_tests:
            content.append("## Performance Notes")
            content.append("")
            content.append("### Slow Tests (>1s)")
            for result in sorted(slow_tests, key=lambda x: x.duration, reverse=True)[
                :5
            ]:
                content.append(f"- **{result.test_name}:** {result.duration:.2f}s")
            content.append("")

        # Test categories
        categories = {}
        for result in test_suite.results:
            for category in result.categories:
                if category not in categories:
                    categories[category] = {
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "errors": 0,
                    }
                categories[category][result.status] += 1

        if categories:
            content.append("## Test Categories")
            content.append("")
            content.append("| Category | Passed | Failed | Skipped | Errors |")
            content.append("|----------|--------|--------|---------|--------|")
            for category, stats in sorted(categories.items()):
                content.append(
                    f"| {category} | {stats['passed']} | {stats['failed']} | {stats['skipped']} | {stats['errors']} |"
                )
            content.append("")

        # CI Environment info
        if ci_info["is_ci"]:
            content.append("## CI Environment")
            content.append("")
            content.append(f"- **Provider:** {ci_info['ci_provider']}")
            if ci_info.get("branch"):
                content.append(f"- **Branch:** {ci_info['branch']}")
            if ci_info.get("commit"):
                content.append(f"- **Commit:** {ci_info['commit'][:8]}...")
            if ci_info.get("build_number"):
                content.append(f"- **Build:** {ci_info['build_number']}")
            content.append(f"- **OS:** {ci_info['runner_os']}")
            content.append("")

        content.append("---")
        content.append(f"*Generated at {time.strftime('%Y-%m-%d %H:%M:%S UTC')}*")

        with open(output_path, "w") as f:
            f.write("\n".join(content))

        return output_path

    def generate_coverage_badge_data(
        self, coverage_percentage: float, filename: str = "coverage-badge.json"
    ) -> Path:
        """Generate badge data for coverage reporting."""
        output_path = self.output_dir / filename

        # Determine color based on coverage
        if coverage_percentage >= 90:
            color = "brightgreen"
        elif coverage_percentage >= 80:
            color = "green"
        elif coverage_percentage >= 70:
            color = "yellowgreen"
        elif coverage_percentage >= 60:
            color = "yellow"
        elif coverage_percentage >= 50:
            color = "orange"
        else:
            color = "red"

        badge_data = {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{coverage_percentage:.1f}%",
            "color": color,
        }

        with open(output_path, "w") as f:
            json.dump(badge_data, f, indent=2)

        return output_path


class ArtifactManager:
    """Manage test artifacts for CI/CD systems."""

    def __init__(self, artifact_dir: Path | None = None):
        if artifact_dir is None:
            artifact_dir = Path.cwd() / "test-artifacts"

        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(exist_ok=True, parents=True)

    def collect_log_files(self, log_pattern: str = "*.log") -> list[Path]:
        """Collect log files for artifact storage."""
        log_files = list(Path.cwd().rglob(log_pattern))
        collected_files = []

        for log_file in log_files:
            if log_file.is_file():
                # Copy to artifact directory with timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                artifact_name = f"{timestamp}_{log_file.name}"
                artifact_path = self.artifact_dir / artifact_name

                try:
                    import shutil

                    shutil.copy2(log_file, artifact_path)
                    collected_files.append(artifact_path)
                except OSError:
                    pass

        return collected_files

    def collect_test_outputs(self, output_pattern: str = "test_*") -> list[Path]:
        """Collect test output files for artifact storage."""
        output_files = list(Path.cwd().rglob(output_pattern))
        collected_files = []

        for output_file in output_files:
            if output_file.is_file() and output_file.suffix in [
                ".xml",
                ".json",
                ".html",
                ".txt",
            ]:
                # Copy to artifact directory
                artifact_path = self.artifact_dir / output_file.name

                try:
                    import shutil

                    shutil.copy2(output_file, artifact_path)
                    collected_files.append(artifact_path)
                except OSError:
                    pass

        return collected_files

    def create_archive(
        self, archive_name: str = "test-artifacts.tar.gz"
    ) -> Path | None:
        """Create compressed archive of all artifacts."""
        if not any(self.artifact_dir.iterdir()):
            return None

        archive_path = self.artifact_dir.parent / archive_name

        try:
            import tarfile

            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(self.artifact_dir, arcname=archive_name.replace(".tar.gz", ""))
            return archive_path
        except Exception:
            return None


# Global instances
_ci_detector = CIEnvironmentDetector()
_report_generator = TestReportGenerator()
_artifact_manager = ArtifactManager()


def get_ci_environment() -> dict[str, Any]:
    """Get CI environment information."""
    return _ci_detector.detect_ci_environment()


def generate_ci_reports(test_suite: TestSuite) -> dict[str, Path]:
    """Generate all CI reports for a test suite."""
    reports = {}

    try:
        reports["junit"] = _report_generator.generate_junit_xml(test_suite)
        reports["json"] = _report_generator.generate_json_report(test_suite)
        reports["markdown"] = _report_generator.generate_markdown_summary(test_suite)
    except Exception as e:
        print(f"Warning: Failed to generate some CI reports: {e}")

    return reports


def collect_test_artifacts() -> dict[str, list[Path]]:
    """Collect all test artifacts."""
    artifacts = {}

    try:
        artifacts["logs"] = _artifact_manager.collect_log_files()
        artifacts["outputs"] = _artifact_manager.collect_test_outputs()
        archive = _artifact_manager.create_archive()
        if archive:
            artifacts["archive"] = [archive]
    except Exception as e:
        print(f"Warning: Failed to collect some artifacts: {e}")

    return artifacts


# CI-specific pytest hooks (to be used in conftest.py)
def pytest_configure_ci_reporting():
    """Configure pytest for CI reporting."""
    ci_env = get_ci_environment()

    if ci_env["is_ci"]:
        print(f"Running in {ci_env['ci_provider']} CI environment")

        # Configure pytest options for CI
        import sys

        ci_args = [
            "--tb=short",  # Shorter tracebacks for CI logs
            "--strict-markers",  # Enforce marker definitions
            "--strict-config",  # Enforce configuration
            "--disable-warnings",  # Reduce noise in CI
        ]

        # Add JUnit XML output
        ci_args.extend(["--junitxml=test-reports/junit.xml"])

        # Add to pytest args if not already present
        for arg in ci_args:
            if arg not in sys.argv:
                sys.argv.append(arg)


def pytest_sessionfinish_ci_reporting(session, exitstatus):
    """Generate CI reports after test session."""
    ci_env = get_ci_environment()

    if ci_env["is_ci"]:
        # Create simplified test suite from session results
        test_suite = TestSuite(
            name="XPCS Toolkit Test Suite",
            total_tests=session.testscollected,
            passed=session.testscollected - session.testsfailed,
            failed=session.testsfailed,
            skipped=0,  # Simplified
            errors=0,  # Simplified
            duration=time.time() - getattr(session, "start_time", time.time()),
        )

        # Generate reports
        try:
            reports = generate_ci_reports(test_suite)
            artifacts = collect_test_artifacts()

            print("\nCI Reports generated:")
            for report_type, path in reports.items():
                print(f"  {report_type}: {path}")

            if artifacts:
                print("\nArtifacts collected:")
                for artifact_type, paths in artifacts.items():
                    print(f"  {artifact_type}: {len(paths)} files")

        except Exception as e:
            print(f"Warning: Failed to generate CI reports: {e}")


# GitHub Actions specific utilities
def set_github_output(name: str, value: str) -> None:
    """Set GitHub Actions output parameter."""
    if os.getenv("GITHUB_ACTIONS"):
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"{name}={value}\n")


def github_step_summary(content: str) -> None:
    """Add content to GitHub Actions step summary."""
    if os.getenv("GITHUB_ACTIONS"):
        github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
        if github_step_summary:
            with open(github_step_summary, "a") as f:
                f.write(content)
                f.write("\n")


def cleanup_ci_artifacts() -> None:
    """Clean up CI artifact directories if they are empty."""
    import shutil

    # Directories to clean up
    artifact_dirs = [Path.cwd() / "test-artifacts", Path.cwd() / "test-reports"]

    for artifact_dir in artifact_dirs:
        if artifact_dir.exists() and artifact_dir.is_dir():
            try:
                # Check if directory is empty (only contains . and ..)
                if not any(artifact_dir.iterdir()):
                    shutil.rmtree(artifact_dir)
            except (OSError, PermissionError):
                # Ignore errors during cleanup
                pass


# Pytest fixtures for automatic cleanup
import pytest


@pytest.fixture(scope="session", autouse=True)
def ci_cleanup():
    """Automatic cleanup fixture for CI artifacts."""
    # Ensure cleanup happens even if tests fail
    try:
        yield
    finally:
        cleanup_ci_artifacts()
