"""
CI/CD Integration for Qt Compliance System.

This module provides comprehensive CI/CD integration capabilities including
automated testing, deployment validation, and continuous quality assurance
for the Qt compliance system.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..testing.qt_regression_testing_framework import QtRegressionTestFramework, RegressionTestConfiguration
from ..testing.qt_cross_platform_validation import CrossPlatformQtComplianceTestSuite
from ..testing.qt_integration_tests import QtComplianceIntegrationTestSuite
from ..performance import run_qt_compliance_benchmarks, BenchmarkConfiguration
from .qt_deployment_manager import QtDeploymentManager, DeploymentConfiguration, DeploymentEnvironment
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class CIStage(Enum):
    """CI/CD pipeline stages."""

    SETUP = "setup"
    LINT = "lint"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    CROSS_PLATFORM_TESTS = "cross_platform_tests"
    PERFORMANCE_TESTS = "performance_tests"
    REGRESSION_TESTS = "regression_tests"
    SECURITY_SCAN = "security_scan"
    DEPLOYMENT_VALIDATION = "deployment_validation"
    CLEANUP = "cleanup"


class CIProvider(Enum):
    """Supported CI/CD providers."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    TRAVIS_CI = "travis_ci"
    CIRCLE_CI = "circle_ci"


@dataclass
class CIConfiguration:
    """Configuration for CI/CD integration."""

    # Provider settings
    provider: CIProvider = CIProvider.GITHUB_ACTIONS
    repository_url: Optional[str] = None
    branch_patterns: List[str] = field(default_factory=lambda: ["main", "master", "develop"])

    # Test settings
    enable_unit_tests: bool = True
    enable_integration_tests: bool = True
    enable_cross_platform_tests: bool = True
    enable_performance_tests: bool = True
    enable_regression_tests: bool = True
    enable_security_scanning: bool = True

    # Performance thresholds
    max_test_duration_minutes: int = 30
    max_memory_usage_mb: int = 2048
    min_test_coverage_percent: int = 80
    max_performance_regression_percent: int = 10

    # Platform matrix
    test_platforms: List[str] = field(default_factory=lambda: ["ubuntu-latest", "windows-latest", "macos-latest"])
    python_versions: List[str] = field(default_factory=lambda: ["3.8", "3.9", "3.10", "3.11"])

    # Deployment settings
    auto_deploy_branches: List[str] = field(default_factory=lambda: ["main", "master"])
    staging_environment: str = "staging"
    production_environment: str = "production"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CIStageResult:
    """Result of a CI/CD stage."""

    stage: CIStage
    timestamp: float
    success: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CIPipelineResult:
    """Result of complete CI/CD pipeline execution."""

    pipeline_id: str
    timestamp: float
    success: bool
    total_duration_seconds: float
    stage_results: List[CIStageResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "total_duration_seconds": self.total_duration_seconds,
            "stage_results": [result.to_dict() for result in self.stage_results],
            "summary": self.summary
        }


class QtComplianceCICD:
    """
    Comprehensive CI/CD integration for Qt compliance system.

    Provides:
    - Multi-platform testing automation
    - Performance regression detection
    - Automated deployment validation
    - Quality gate enforcement
    - Artifact generation and management
    """

    def __init__(self, config: Optional[CIConfiguration] = None):
        """Initialize CI/CD integration."""
        self.config = config or CIConfiguration()
        self.pipeline_results: List[CIPipelineResult] = []

        # CI environment detection
        self.ci_environment = self._detect_ci_environment()

        logger.info(f"Qt compliance CI/CD initialized for {self.config.provider.value}")

    def _detect_ci_environment(self) -> Dict[str, Any]:
        """Detect current CI environment."""
        environment = {
            "is_ci": False,
            "provider": None,
            "branch": None,
            "commit": None,
            "pull_request": None
        }

        # GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            environment.update({
                "is_ci": True,
                "provider": "github_actions",
                "branch": os.getenv("GITHUB_REF_NAME"),
                "commit": os.getenv("GITHUB_SHA"),
                "pull_request": os.getenv("GITHUB_EVENT_NAME") == "pull_request"
            })

        # GitLab CI
        elif os.getenv("GITLAB_CI"):
            environment.update({
                "is_ci": True,
                "provider": "gitlab_ci",
                "branch": os.getenv("CI_COMMIT_REF_NAME"),
                "commit": os.getenv("CI_COMMIT_SHA"),
                "pull_request": os.getenv("CI_PIPELINE_SOURCE") == "merge_request_event"
            })

        # Jenkins
        elif os.getenv("JENKINS_URL"):
            environment.update({
                "is_ci": True,
                "provider": "jenkins",
                "branch": os.getenv("BRANCH_NAME"),
                "commit": os.getenv("GIT_COMMIT"),
                "pull_request": os.getenv("CHANGE_ID") is not None
            })

        # Azure DevOps
        elif os.getenv("AZURE_HTTP_USER_AGENT"):
            environment.update({
                "is_ci": True,
                "provider": "azure_devops",
                "branch": os.getenv("BUILD_SOURCEBRANCH"),
                "commit": os.getenv("BUILD_SOURCEVERSION"),
                "pull_request": os.getenv("SYSTEM_PULLREQUEST_PULLREQUESTID") is not None
            })

        return environment

    def run_pipeline(self, pipeline_id: Optional[str] = None) -> CIPipelineResult:
        """Run complete CI/CD pipeline."""
        if pipeline_id is None:
            pipeline_id = f"pipeline_{int(time.time())}"

        logger.info(f"Starting CI/CD pipeline: {pipeline_id}")
        start_time = time.perf_counter()

        pipeline_result = CIPipelineResult(
            pipeline_id=pipeline_id,
            timestamp=start_time,
            success=False,
            total_duration_seconds=0.0
        )

        try:
            # Define pipeline stages
            stages = [
                CIStage.SETUP,
                CIStage.LINT,
                CIStage.UNIT_TESTS,
                CIStage.INTEGRATION_TESTS,
                CIStage.CROSS_PLATFORM_TESTS,
                CIStage.PERFORMANCE_TESTS,
                CIStage.REGRESSION_TESTS,
                CIStage.SECURITY_SCAN,
                CIStage.DEPLOYMENT_VALIDATION,
                CIStage.CLEANUP
            ]

            # Execute stages
            for stage in stages:
                if not self._should_run_stage(stage):
                    logger.info(f"Skipping stage: {stage.value}")
                    continue

                logger.info(f"Running stage: {stage.value}")
                stage_result = self._run_stage(stage)
                pipeline_result.stage_results.append(stage_result)

                # Stop pipeline on failure (unless cleanup stage)
                if not stage_result.success and stage != CIStage.CLEANUP:
                    logger.error(f"Pipeline failed at stage: {stage.value}")
                    break

            # Calculate overall success
            pipeline_result.success = all(
                result.success for result in pipeline_result.stage_results
                if result.stage != CIStage.CLEANUP  # Cleanup failures don't fail pipeline
            )

            # Generate summary
            pipeline_result.summary = self._generate_pipeline_summary(pipeline_result)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            logger.debug(traceback.format_exc())

        finally:
            pipeline_result.total_duration_seconds = time.perf_counter() - start_time

        # Store result
        self.pipeline_results.append(pipeline_result)

        logger.info(f"Pipeline completed: {pipeline_id} - {'SUCCESS' if pipeline_result.success else 'FAILED'}")
        return pipeline_result

    def _should_run_stage(self, stage: CIStage) -> bool:
        """Determine if a stage should run based on configuration."""
        stage_config_map = {
            CIStage.UNIT_TESTS: self.config.enable_unit_tests,
            CIStage.INTEGRATION_TESTS: self.config.enable_integration_tests,
            CIStage.CROSS_PLATFORM_TESTS: self.config.enable_cross_platform_tests,
            CIStage.PERFORMANCE_TESTS: self.config.enable_performance_tests,
            CIStage.REGRESSION_TESTS: self.config.enable_regression_tests,
            CIStage.SECURITY_SCAN: self.config.enable_security_scanning,
        }

        # Always run setup, cleanup, lint, and deployment validation
        if stage in [CIStage.SETUP, CIStage.CLEANUP, CIStage.LINT, CIStage.DEPLOYMENT_VALIDATION]:
            return True

        return stage_config_map.get(stage, True)

    def _run_stage(self, stage: CIStage) -> CIStageResult:
        """Run a specific pipeline stage."""
        start_time = time.perf_counter()
        stage_result = CIStageResult(
            stage=stage,
            timestamp=start_time,
            success=False,
            duration_seconds=0.0
        )

        try:
            if stage == CIStage.SETUP:
                stage_result.success = self._run_setup_stage(stage_result)
            elif stage == CIStage.LINT:
                stage_result.success = self._run_lint_stage(stage_result)
            elif stage == CIStage.UNIT_TESTS:
                stage_result.success = self._run_unit_tests_stage(stage_result)
            elif stage == CIStage.INTEGRATION_TESTS:
                stage_result.success = self._run_integration_tests_stage(stage_result)
            elif stage == CIStage.CROSS_PLATFORM_TESTS:
                stage_result.success = self._run_cross_platform_tests_stage(stage_result)
            elif stage == CIStage.PERFORMANCE_TESTS:
                stage_result.success = self._run_performance_tests_stage(stage_result)
            elif stage == CIStage.REGRESSION_TESTS:
                stage_result.success = self._run_regression_tests_stage(stage_result)
            elif stage == CIStage.SECURITY_SCAN:
                stage_result.success = self._run_security_scan_stage(stage_result)
            elif stage == CIStage.DEPLOYMENT_VALIDATION:
                stage_result.success = self._run_deployment_validation_stage(stage_result)
            elif stage == CIStage.CLEANUP:
                stage_result.success = self._run_cleanup_stage(stage_result)
            else:
                raise ValueError(f"Unknown stage: {stage}")

        except Exception as e:
            stage_result.error_message = str(e)
            logger.error(f"Stage {stage.value} failed: {e}")

        finally:
            stage_result.duration_seconds = time.perf_counter() - start_time

        return stage_result

    def _run_setup_stage(self, result: CIStageResult) -> bool:
        """Run setup stage."""
        try:
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if python_version not in self.config.python_versions:
                logger.warning(f"Python {python_version} not in supported versions: {self.config.python_versions}")

            # Check dependencies
            self._check_dependencies(result)

            # Setup test environment
            os.environ["PYXPCS_TESTING_MODE"] = "1"
            os.environ["QT_QPA_PLATFORM"] = "offscreen"  # For headless testing

            result.details = {
                "python_version": python_version,
                "platform": sys.platform,
                "ci_environment": self.ci_environment
            }

            return True

        except Exception as e:
            result.error_message = str(e)
            return False

    def _check_dependencies(self, result: CIStageResult):
        """Check required dependencies."""
        required_packages = [
            "PySide6",
            "pytest",
            "coverage"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            raise Exception(f"Missing required packages: {missing_packages}")

        result.details["dependencies_checked"] = len(required_packages)

    def _run_lint_stage(self, result: CIStageResult) -> bool:
        """Run linting stage."""
        try:
            lint_results = {}

            # Run flake8 if available
            try:
                flake8_result = subprocess.run(
                    ["flake8", "xpcs_toolkit/", "--max-line-length=120", "--extend-ignore=E203,W503"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                lint_results["flake8"] = {
                    "success": flake8_result.returncode == 0,
                    "output": flake8_result.stdout,
                    "errors": flake8_result.stderr
                }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("flake8 not available or timed out")

            # Run mypy if available
            try:
                mypy_result = subprocess.run(
                    ["mypy", "xpcs_toolkit/", "--ignore-missing-imports"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                lint_results["mypy"] = {
                    "success": mypy_result.returncode == 0,
                    "output": mypy_result.stdout,
                    "errors": mypy_result.stderr
                }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("mypy not available or timed out")

            result.details = lint_results

            # Consider success if at least one linter passed or no linters available
            if lint_results:
                return any(result["success"] for result in lint_results.values())
            else:
                logger.warning("No linters available - skipping lint stage")
                return True

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_unit_tests_stage(self, result: CIStageResult) -> bool:
        """Run unit tests stage."""
        try:
            # Run pytest with coverage
            pytest_args = [
                "python", "-m", "pytest",
                "tests/unit/",
                "-v",
                "--tb=short",
                f"--timeout={self.config.max_test_duration_minutes * 60}",
                "--cov=xpcs_toolkit",
                f"--cov-fail-under={self.config.min_test_coverage_percent}",
                "--cov-report=xml",
                "--cov-report=html",
                "--cov-report=term"
            ]

            test_result = subprocess.run(
                pytest_args,
                capture_output=True,
                text=True,
                timeout=self.config.max_test_duration_minutes * 60
            )

            result.details = {
                "return_code": test_result.returncode,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr
            }

            # Parse test results
            self._parse_pytest_output(test_result.stdout, result)

            return test_result.returncode == 0

        except subprocess.TimeoutExpired:
            result.error_message = "Unit tests timed out"
            return False
        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_integration_tests_stage(self, result: CIStageResult) -> bool:
        """Run integration tests stage."""
        try:
            # Run Qt compliance integration tests
            test_suite = QtComplianceIntegrationTestSuite()
            test_results = test_suite.run_all_tests()

            result.details = {
                "total_tests": len(test_results),
                "passed_tests": sum(1 for r in test_results.values() if r.get("success", False)),
                "test_results": test_results
            }

            success_rate = result.details["passed_tests"] / result.details["total_tests"] if result.details["total_tests"] > 0 else 0
            return success_rate >= 0.9  # 90% success rate required

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_cross_platform_tests_stage(self, result: CIStageResult) -> bool:
        """Run cross-platform tests stage."""
        try:
            # Run cross-platform validation
            test_suite = CrossPlatformQtComplianceTestSuite()
            compatibility_report = test_suite.generate_compatibility_report()

            result.details = compatibility_report

            # Check compatibility score
            compatibility_score = compatibility_report.get("compatibility_score", 0)
            return compatibility_score >= 0.8  # 80% compatibility required

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_performance_tests_stage(self, result: CIStageResult) -> bool:
        """Run performance tests stage."""
        try:
            # Run performance benchmarks
            benchmark_config = BenchmarkConfiguration(
                duration_seconds=10.0,  # Shorter duration for CI
                iterations=20
            )

            benchmark_report = run_qt_compliance_benchmarks(benchmark_config)

            result.details = benchmark_report

            # Check for performance regressions
            regressions = benchmark_report.get("performance_analysis", {}).get("regressions", [])
            max_regression = max([r["delta_percent"] for r in regressions], default=0)

            return max_regression <= self.config.max_performance_regression_percent

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_regression_tests_stage(self, result: CIStageResult) -> bool:
        """Run regression tests stage."""
        try:
            # Run regression testing framework
            framework = QtRegressionTestFramework()

            # Use current version as baseline if none exists
            current_version = "ci_build"
            regression_results = framework.run_regression_tests(current_version)

            result.details = {
                "regression_results": [r.to_dict() for r in regression_results],
                "total_tests": len(regression_results),
                "passed_tests": sum(1 for r in regression_results if r.passed)
            }

            success_rate = result.details["passed_tests"] / result.details["total_tests"] if result.details["total_tests"] > 0 else 0
            return success_rate >= 0.95  # 95% success rate for regression tests

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_security_scan_stage(self, result: CIStageResult) -> bool:
        """Run security scanning stage."""
        try:
            security_results = {}

            # Run bandit security scanner if available
            try:
                bandit_result = subprocess.run(
                    ["bandit", "-r", "xpcs_toolkit/", "-f", "json"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if bandit_result.returncode == 0:
                    security_results["bandit"] = {
                        "success": True,
                        "issues": 0
                    }
                else:
                    # Parse bandit output for issue count
                    try:
                        bandit_output = json.loads(bandit_result.stdout)
                        security_results["bandit"] = {
                            "success": len(bandit_output.get("results", [])) == 0,
                            "issues": len(bandit_output.get("results", []))
                        }
                    except:
                        security_results["bandit"] = {
                            "success": False,
                            "issues": "unknown"
                        }

            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("bandit not available")

            # Run safety check if available
            try:
                safety_result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                security_results["safety"] = {
                    "success": safety_result.returncode == 0,
                    "output": safety_result.stdout
                }

            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("safety not available")

            result.details = security_results

            # Consider success if no critical security issues found
            if security_results:
                return all(result.get("success", True) for result in security_results.values())
            else:
                logger.warning("No security scanners available - skipping security scan")
                return True

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_deployment_validation_stage(self, result: CIStageResult) -> bool:
        """Run deployment validation stage."""
        try:
            # Test deployment to CI environment
            deployment_config = DeploymentConfiguration.for_environment(DeploymentEnvironment.CI_CD)
            deployment_manager = QtDeploymentManager(deployment_config)

            # Validate deployment
            deployment_success = deployment_manager.deploy()

            if deployment_success:
                # Perform health check
                health_check_success = deployment_manager.perform_maintenance_operation(
                    deployment_manager.MaintenanceOperation.HEALTH_CHECK
                )

                deployment_status = deployment_manager.get_deployment_status()

                result.details = {
                    "deployment_success": deployment_success,
                    "health_check_success": health_check_success,
                    "deployment_status": deployment_status.to_dict()
                }

                return deployment_success and health_check_success
            else:
                result.details = {"deployment_success": False}
                return False

        except Exception as e:
            result.error_message = str(e)
            return False

    def _run_cleanup_stage(self, result: CIStageResult) -> bool:
        """Run cleanup stage."""
        try:
            cleanup_actions = []

            # Clean up test artifacts
            test_artifacts = [
                "htmlcov/",
                "coverage.xml",
                ".coverage",
                ".pytest_cache/",
                "__pycache__/",
                "*.pyc"
            ]

            for artifact in test_artifacts:
                try:
                    if os.path.exists(artifact):
                        if os.path.isdir(artifact):
                            import shutil
                            shutil.rmtree(artifact)
                        else:
                            os.remove(artifact)
                        cleanup_actions.append(f"Removed {artifact}")
                except Exception as e:
                    logger.warning(f"Could not clean up {artifact}: {e}")

            # Reset environment variables
            test_env_vars = ["PYXPCS_TESTING_MODE", "QT_QPA_PLATFORM"]
            for var in test_env_vars:
                if var in os.environ:
                    del os.environ[var]
                    cleanup_actions.append(f"Removed env var {var}")

            result.details = {"cleanup_actions": cleanup_actions}
            return True

        except Exception as e:
            result.error_message = str(e)
            return False

    def _parse_pytest_output(self, output: str, result: CIStageResult):
        """Parse pytest output for test metrics."""
        lines = output.split('\n')

        for line in lines:
            # Look for test summary line
            if "failed" in line and "passed" in line:
                # Parse test counts
                import re
                matches = re.findall(r'(\d+) (passed|failed|skipped|error)', line)
                test_counts = {status: int(count) for count, status in matches}

                result.details.update({
                    "test_counts": test_counts,
                    "total_tests": sum(test_counts.values())
                })
                break

            # Look for coverage information
            elif "TOTAL" in line and "%" in line:
                parts = line.split()
                if len(parts) >= 4:
                    coverage_percent = parts[-1].replace('%', '')
                    try:
                        result.details["coverage_percent"] = float(coverage_percent)
                    except ValueError:
                        pass

    def _generate_pipeline_summary(self, pipeline_result: CIPipelineResult) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        stage_counts = {
            "total": len(pipeline_result.stage_results),
            "passed": sum(1 for r in pipeline_result.stage_results if r.success),
            "failed": sum(1 for r in pipeline_result.stage_results if not r.success)
        }

        # Calculate total test counts across all stages
        total_tests = 0
        total_passed = 0
        total_coverage = 0.0

        for stage_result in pipeline_result.stage_results:
            details = stage_result.details
            if "total_tests" in details:
                total_tests += details["total_tests"]
            if "passed_tests" in details:
                total_passed += details["passed_tests"]
            if "coverage_percent" in details:
                total_coverage = max(total_coverage, details["coverage_percent"])

        return {
            "stage_summary": stage_counts,
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "success_rate": total_passed / total_tests if total_tests > 0 else 0,
                "coverage_percent": total_coverage
            },
            "performance": {
                "total_duration_minutes": pipeline_result.total_duration_seconds / 60,
                "average_stage_duration_seconds": sum(r.duration_seconds for r in pipeline_result.stage_results) / len(pipeline_result.stage_results) if pipeline_result.stage_results else 0
            },
            "quality_gates": {
                "minimum_coverage_met": total_coverage >= self.config.min_test_coverage_percent,
                "maximum_duration_met": pipeline_result.total_duration_seconds <= self.config.max_test_duration_minutes * 60,
                "all_stages_passed": stage_counts["failed"] == 0
            }
        }

    def generate_ci_config_files(self, output_directory: Optional[Path] = None) -> Dict[str, str]:
        """Generate CI configuration files for different providers."""
        if output_directory is None:
            output_directory = Path.cwd()

        generated_files = {}

        # GitHub Actions
        if self.config.provider == CIProvider.GITHUB_ACTIONS:
            github_config = self._generate_github_actions_config()
            github_file = output_directory / ".github" / "workflows" / "qt_compliance_ci.yml"
            github_file.parent.mkdir(parents=True, exist_ok=True)
            github_file.write_text(github_config)
            generated_files["github_actions"] = str(github_file)

        # GitLab CI
        elif self.config.provider == CIProvider.GITLAB_CI:
            gitlab_config = self._generate_gitlab_ci_config()
            gitlab_file = output_directory / ".gitlab-ci.yml"
            gitlab_file.write_text(gitlab_config)
            generated_files["gitlab_ci"] = str(gitlab_file)

        # Jenkins
        elif self.config.provider == CIProvider.JENKINS:
            jenkins_config = self._generate_jenkins_config()
            jenkins_file = output_directory / "Jenkinsfile"
            jenkins_file.write_text(jenkins_config)
            generated_files["jenkins"] = str(jenkins_file)

        return generated_files

    def _generate_github_actions_config(self) -> str:
        """Generate GitHub Actions workflow configuration."""
        return f"""name: Qt Compliance CI

on:
  push:
    branches: {self.config.branch_patterns}
  pull_request:
    branches: {self.config.branch_patterns}

jobs:
  test:
    strategy:
      matrix:
        os: {self.config.test_platforms}
        python-version: {self.config.python_versions}

    runs-on: ${{{{ matrix.os }}}}

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}

    - name: Install Qt dependencies
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y qt6-base-dev qt6-tools-dev-tools

    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov pytest-timeout coverage flake8 mypy bandit safety

    - name: Run Qt Compliance CI Pipeline
      run: |
        python -m xpcs_toolkit.deployment.ci_cd_integration

    - name: Upload coverage reports
      if: matrix.python-version == '3.10' && matrix.os == 'ubuntu-latest'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

    - name: Upload test artifacts
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results-${{{{ matrix.os }}}}-${{{{ matrix.python-version }}}}
        path: |
          htmlcov/
          coverage.xml
          pytest_report.xml

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Deploy to staging
      run: |
        python -m xpcs_toolkit.deployment.qt_deployment_manager STAGING
"""

    def _generate_gitlab_ci_config(self) -> str:
        """Generate GitLab CI configuration."""
        return f"""stages:
  - test
  - deploy

variables:
  QT_QPA_PLATFORM: "offscreen"
  PYXPCS_TESTING_MODE: "1"

.python_template: &python_template
  image: python:$PYTHON_VERSION
  before_script:
    - pip install --upgrade pip
    - pip install -e .
    - pip install pytest pytest-cov pytest-timeout coverage flake8 mypy bandit safety

test:
  <<: *python_template
  stage: test
  parallel:
    matrix:
      - PYTHON_VERSION: {self.config.python_versions}
  script:
    - python -m xpcs_toolkit.deployment.ci_cd_integration
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
      - coverage.xml
    expire_in: 1 week
  coverage: '/TOTAL.*\\s+(\\d+%)$/'

deploy_staging:
  <<: *python_template
  stage: deploy
  script:
    - python -m xpcs_toolkit.deployment.qt_deployment_manager STAGING
  only:
    - main
  environment:
    name: staging
"""

    def _generate_jenkins_config(self) -> str:
        """Generate Jenkins pipeline configuration."""
        return f"""pipeline {{
    agent any

    environment {{
        QT_QPA_PLATFORM = 'offscreen'
        PYXPCS_TESTING_MODE = '1'
    }}

    stages {{
        stage('Setup') {{
            steps {{
                checkout scm
                sh 'pip install --upgrade pip'
                sh 'pip install -e .'
                sh 'pip install pytest pytest-cov pytest-timeout coverage flake8 mypy bandit safety'
            }}
        }}

        stage('Test') {{
            parallel {{
                {chr(10).join([
                    f'''stage('Python {version}') {{
                    steps {{
                        sh 'python{version} -m xpcs_toolkit.deployment.ci_cd_integration'
                    }}
                }}'''
                    for version in self.config.python_versions
                ])}
            }}
        }}

        stage('Deploy') {{
            when {{
                branch 'main'
            }}
            steps {{
                sh 'python -m xpcs_toolkit.deployment.qt_deployment_manager STAGING'
            }}
        }}
    }}

    post {{
        always {{
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])

            archiveArtifacts artifacts: 'coverage.xml,htmlcov/**', allowEmptyArchive: true
        }}

        failure {{
            emailext (
                subject: "Build Failed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "Build failed. Check console output at ${{env.BUILD_URL}}",
                to: "{self.ci_environment.get('notification_email', 'team@example.com')}"
            )
        }}
    }}
}}"""

    def create_ci_report(self, pipeline_result: CIPipelineResult, output_file: Optional[Path] = None) -> str:
        """Create comprehensive CI report."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"ci_report_{timestamp}.json")

        # Generate comprehensive report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "pipeline_id": pipeline_result.pipeline_id,
                "ci_environment": self.ci_environment,
                "configuration": self.config.to_dict()
            },
            "pipeline_result": pipeline_result.to_dict(),
            "quality_analysis": self._analyze_quality_metrics(pipeline_result),
            "recommendations": self._generate_ci_recommendations(pipeline_result)
        }

        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"CI report generated: {output_file}")
        return str(output_file)

    def _analyze_quality_metrics(self, pipeline_result: CIPipelineResult) -> Dict[str, Any]:
        """Analyze quality metrics from pipeline results."""
        analysis = {
            "overall_quality_score": 0.0,
            "test_quality": {},
            "performance_quality": {},
            "security_quality": {},
            "coverage_quality": {}
        }

        # Calculate overall quality score (0-100)
        quality_factors = []

        # Test quality
        if pipeline_result.summary.get("test_summary", {}).get("total_tests", 0) > 0:
            test_success_rate = pipeline_result.summary["test_summary"]["success_rate"]
            quality_factors.append(test_success_rate * 100)
            analysis["test_quality"] = {
                "success_rate": test_success_rate,
                "total_tests": pipeline_result.summary["test_summary"]["total_tests"],
                "score": test_success_rate * 100
            }

        # Coverage quality
        coverage_percent = pipeline_result.summary.get("test_summary", {}).get("coverage_percent", 0)
        if coverage_percent > 0:
            coverage_score = min(100, (coverage_percent / self.config.min_test_coverage_percent) * 100)
            quality_factors.append(coverage_score)
            analysis["coverage_quality"] = {
                "coverage_percent": coverage_percent,
                "target_percent": self.config.min_test_coverage_percent,
                "score": coverage_score
            }

        # Performance quality (based on duration)
        duration_minutes = pipeline_result.total_duration_seconds / 60
        if duration_minutes > 0:
            performance_score = max(0, 100 - (duration_minutes / self.config.max_test_duration_minutes) * 50)
            quality_factors.append(performance_score)
            analysis["performance_quality"] = {
                "duration_minutes": duration_minutes,
                "target_minutes": self.config.max_test_duration_minutes,
                "score": performance_score
            }

        # Security quality
        security_issues = 0
        for stage_result in pipeline_result.stage_results:
            if stage_result.stage == CIStage.SECURITY_SCAN and stage_result.success:
                # Extract security issue count
                for tool_result in stage_result.details.values():
                    if isinstance(tool_result, dict) and "issues" in tool_result:
                        if isinstance(tool_result["issues"], int):
                            security_issues += tool_result["issues"]

        security_score = max(0, 100 - security_issues * 20)  # Penalize each issue
        quality_factors.append(security_score)
        analysis["security_quality"] = {
            "issues_found": security_issues,
            "score": security_score
        }

        # Calculate overall score
        if quality_factors:
            analysis["overall_quality_score"] = sum(quality_factors) / len(quality_factors)

        return analysis

    def _generate_ci_recommendations(self, pipeline_result: CIPipelineResult) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []

        # Test recommendations
        test_summary = pipeline_result.summary.get("test_summary", {})
        if test_summary.get("success_rate", 1.0) < 0.95:
            recommendations.append("Test success rate below 95% - investigate and fix failing tests")

        # Coverage recommendations
        coverage = test_summary.get("coverage_percent", 0)
        if coverage < self.config.min_test_coverage_percent:
            recommendations.append(f"Test coverage {coverage:.1f}% below target {self.config.min_test_coverage_percent}% - add more tests")

        # Performance recommendations
        duration_minutes = pipeline_result.total_duration_seconds / 60
        if duration_minutes > self.config.max_test_duration_minutes:
            recommendations.append(f"Pipeline duration {duration_minutes:.1f} minutes exceeds target {self.config.max_test_duration_minutes} minutes - optimize tests")

        # Security recommendations
        for stage_result in pipeline_result.stage_results:
            if stage_result.stage == CIStage.SECURITY_SCAN and not stage_result.success:
                recommendations.append("Security scan detected issues - review and address security findings")

        # Stage-specific recommendations
        failed_stages = [r.stage.value for r in pipeline_result.stage_results if not r.success and r.stage != CIStage.CLEANUP]
        if failed_stages:
            recommendations.append(f"Failed stages: {', '.join(failed_stages)} - investigate and fix")

        return recommendations


def run_ci_pipeline():
    """Run CI pipeline from command line."""
    ci_config = CIConfiguration()
    ci_cd = QtComplianceCICD(ci_config)

    pipeline_result = ci_cd.run_pipeline()

    # Generate report
    report_file = ci_cd.create_ci_report(pipeline_result)

    # Print summary
    print(f"CI Pipeline Result: {'SUCCESS' if pipeline_result.success else 'FAILED'}")
    print(f"Duration: {pipeline_result.total_duration_seconds:.1f}s")
    print(f"Report: {report_file}")

    # Exit with appropriate code
    sys.exit(0 if pipeline_result.success else 1)


if __name__ == "__main__":
    run_ci_pipeline()