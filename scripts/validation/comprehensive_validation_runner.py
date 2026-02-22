#!/usr/bin/env python3
"""
Comprehensive Validation Runner for XPCS Toolkit Phase 6 Integration Testing

This module orchestrates all validation and testing components to provide a complete
assessment of the integrated system performance, accuracy, and stability.

Components Orchestrated:
- Performance benchmarking suite
- Scientific accuracy validation
- Integration testing for all phases
- System stability and stress testing
- Report generation and recommendations

Author: Integration and Validation Agent
Created: 2025-09-11
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import validation components
try:
    sys.path.append(str(Path(__file__).parent.parent / "benchmarks" / "integration"))
    from comprehensive_performance_benchmarks import IntegratedBenchmarkSuite

    sys.path.append(str(Path(__file__).parent))
    from scientific_accuracy_validator import ScientificAccuracyValidationFramework

    sys.path.append(str(Path(__file__).parent.parent / "tests" / "integration"))
    from system_stability_tests import SystemStabilityTestSuite

    VALIDATION_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all validation components: {e}")
    VALIDATION_COMPONENTS_AVAILABLE = False


@dataclass
class ValidationSummary:
    """Summary of all validation results"""

    overall_score: float
    performance_score: float
    accuracy_score: float
    integration_score: float
    stability_score: float
    test_counts: dict[str, int]
    duration_seconds: float
    recommendations: list[str]
    critical_issues: list[str]
    phase_status: dict[str, str]


@dataclass
class ComprehensiveValidationReport:
    """Complete validation report"""

    validation_timestamp: str
    system_info: dict[str, Any]
    validation_summary: ValidationSummary
    performance_results: dict[str, Any] | None
    accuracy_results: dict[str, Any] | None
    integration_results: dict[str, Any] | None
    stability_results: dict[str, Any] | None
    deployment_readiness: dict[str, Any]
    detailed_recommendations: dict[str, list[str]]


class ValidationOrchestrator:
    """Orchestrates all validation and testing components"""

    def __init__(
        self,
        output_dir: str = "comprehensive_validation_results",
        quick_mode: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.quick_mode = quick_mode
        self.start_time = time.time()

        # Initialize logging
        self.setup_logging()

        # Results storage
        self.results = {
            "performance": None,
            "accuracy": None,
            "integration": None,
            "stability": None,
        }

        # Phase validation tracking
        self.phase_validation_status = {
            "Phase1_Memory": "pending",
            "Phase2_IO": "pending",
            "Phase3_Vectorization": "pending",
            "Phase4_Threading": "pending",
            "Phase5_Caching": "pending",
            "Phase6_Integration": "pending",
        }

    def setup_logging(self):
        """Set up logging for validation process"""
        log_file = self.output_dir / "validation_process.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger("ValidationOrchestrator")

    def run_performance_benchmarks(self) -> bool:
        """Run comprehensive performance benchmarks"""
        self.logger.info("Starting performance benchmarking...")

        try:
            if not VALIDATION_COMPONENTS_AVAILABLE:
                self.logger.warning(
                    "Performance benchmarking skipped - components not available"
                )
                return False

            # Create benchmark suite
            benchmark_output = self.output_dir / "performance_benchmarks"
            benchmark_suite = IntegratedBenchmarkSuite(str(benchmark_output))

            # Run benchmarks
            self.results["performance"] = benchmark_suite.run_all_benchmarks()

            # Update phase status based on performance results
            if "summary" in self.results["performance"]:
                for phase, metrics in self.results["performance"]["summary"].items():
                    if phase in self.phase_validation_status:
                        # Consider phase successful if average execution time is reasonable
                        if metrics["avg_execution_time"] < 10.0:  # 10 second threshold
                            self.phase_validation_status[phase] = "passed"
                        else:
                            self.phase_validation_status[phase] = "warning"

            self.logger.info("Performance benchmarking completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Performance benchmarking failed: {e}")
            self.phase_validation_status["Phase6_Integration"] = "failed"
            return False

    def run_accuracy_validation(self) -> bool:
        """Run scientific accuracy validation"""
        self.logger.info("Starting scientific accuracy validation...")

        try:
            if not VALIDATION_COMPONENTS_AVAILABLE:
                self.logger.warning(
                    "Accuracy validation skipped - components not available"
                )
                return False

            # Create accuracy validation framework
            accuracy_output = self.output_dir / "accuracy_validation"
            validator = ScientificAccuracyValidationFramework(
                tolerance=1e-10, output_dir=str(accuracy_output)
            )

            # Run validation
            self.results["accuracy"] = validator.run_all_validations()

            # Update phase status based on accuracy results
            if hasattr(self.results["accuracy"], "overall_accuracy_score"):
                if self.results["accuracy"].overall_accuracy_score >= 95:
                    self.phase_validation_status["Phase3_Vectorization"] = "passed"
                elif self.results["accuracy"].overall_accuracy_score >= 80:
                    self.phase_validation_status["Phase3_Vectorization"] = "warning"
                else:
                    self.phase_validation_status["Phase3_Vectorization"] = "failed"

            self.logger.info("Scientific accuracy validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Accuracy validation failed: {e}")
            self.phase_validation_status["Phase3_Vectorization"] = "failed"
            return False

    def run_integration_tests(self) -> bool:
        """Run integration tests for all phases"""
        self.logger.info("Starting integration testing...")

        try:
            # Run integration tests using subprocess to capture unittest output
            test_file = (
                Path(__file__).parent.parent
                / "tests"
                / "integration"
                / "test_phase_integration.py"
            )

            if not test_file.exists():
                self.logger.warning("Integration test file not found")
                return False

            # Run tests with subprocess
            result = subprocess.run(
                [sys.executable, str(test_file)],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )  # 5 minute timeout

            # Parse results
            integration_results = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            self.results["integration"] = integration_results

            # Update phase status based on integration test results
            if result.returncode == 0:
                # All integration tests passed
                for phase in [
                    "Phase1_Memory",
                    "Phase2_IO",
                    "Phase4_Threading",
                    "Phase5_Caching",
                ]:
                    if self.phase_validation_status[phase] == "pending":
                        self.phase_validation_status[phase] = "passed"
            else:
                # Some integration tests failed
                self.logger.warning("Some integration tests failed")
                self.phase_validation_status["Phase6_Integration"] = "warning"

            self.logger.info("Integration testing completed")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("Integration tests timed out")
            self.phase_validation_status["Phase6_Integration"] = "failed"
            return False
        except Exception as e:
            self.logger.error(f"Integration testing failed: {e}")
            self.phase_validation_status["Phase6_Integration"] = "failed"
            return False

    def run_stability_tests(self) -> bool:
        """Run system stability and stress tests"""
        self.logger.info("Starting system stability testing...")

        try:
            if not VALIDATION_COMPONENTS_AVAILABLE:
                self.logger.warning(
                    "Stability testing skipped - components not available"
                )
                return False

            # Create stability test suite
            stability_output = self.output_dir / "stability_tests"
            stability_suite = SystemStabilityTestSuite(str(stability_output))

            # Run stability tests
            self.results["stability"] = stability_suite.run_all_stress_tests(
                quick_mode=self.quick_mode
            )

            # Update phase status based on stability results
            if hasattr(self.results["stability"], "overall_stability_score"):
                if self.results["stability"].overall_stability_score >= 80:
                    self.phase_validation_status["Phase6_Integration"] = "passed"
                elif self.results["stability"].overall_stability_score >= 60:
                    self.phase_validation_status["Phase6_Integration"] = "warning"
                else:
                    self.phase_validation_status["Phase6_Integration"] = "failed"

            self.logger.info("System stability testing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Stability testing failed: {e}")
            self.phase_validation_status["Phase6_Integration"] = "failed"
            return False

    def calculate_overall_scores(self) -> tuple[float, dict[str, float]]:
        """Calculate overall validation scores"""
        scores = {}

        # Performance score
        if self.results["performance"] and "summary" in self.results["performance"]:
            # Score based on average execution times
            phase_times = [
                metrics["avg_execution_time"]
                for metrics in self.results["performance"]["summary"].values()
            ]
            avg_time = sum(phase_times) / len(phase_times) if phase_times else 10
            scores["performance"] = max(
                0, min(100, (10 - avg_time) * 10)
            )  # Scale to 0-100
        else:
            scores["performance"] = 0

        # Accuracy score
        if self.results["accuracy"] and hasattr(
            self.results["accuracy"], "overall_accuracy_score"
        ):
            scores["accuracy"] = self.results["accuracy"].overall_accuracy_score
        else:
            scores["accuracy"] = 0

        # Integration score
        if self.results["integration"]:
            if self.results["integration"]["success"]:
                scores["integration"] = 100
            else:
                # Parse output for partial success
                stdout = self.results["integration"].get("stdout", "")
                if "FAIL" in stdout or "ERROR" in stdout:
                    scores["integration"] = 50  # Partial success
                else:
                    scores["integration"] = 0
        else:
            scores["integration"] = 0

        # Stability score
        if self.results["stability"] and hasattr(
            self.results["stability"], "overall_stability_score"
        ):
            scores["stability"] = self.results["stability"].overall_stability_score
        else:
            scores["stability"] = 0

        # Overall score (weighted average)
        weights = {
            "performance": 0.25,
            "accuracy": 0.30,  # Higher weight for accuracy
            "integration": 0.25,
            "stability": 0.20,
        }

        overall_score = sum(scores[key] * weights[key] for key in scores)

        return overall_score, scores

    def generate_recommendations(
        self, overall_score: float, individual_scores: dict[str, float]
    ) -> tuple[list[str], list[str]]:
        """Generate recommendations and identify critical issues"""
        recommendations = []
        critical_issues = []

        # Overall assessment
        if overall_score >= 90:
            recommendations.append(
                "Excellent integration - system is ready for production deployment"
            )
        elif overall_score >= 80:
            recommendations.append(
                "Good integration - minor optimizations recommended before deployment"
            )
        elif overall_score >= 70:
            recommendations.append(
                "Acceptable integration - address identified issues before deployment"
            )
        elif overall_score >= 50:
            recommendations.append(
                "Poor integration - significant work needed before deployment"
            )
        else:
            recommendations.append(
                "Critical integration issues - major revision required"
            )
            critical_issues.append("Overall system integration score is critically low")

        # Performance recommendations
        if individual_scores["performance"] < 70:
            recommendations.append(
                "Review and optimize performance-critical code paths"
            )
            if individual_scores["performance"] < 40:
                critical_issues.append(
                    "Performance is significantly below acceptable levels"
                )

        # Accuracy recommendations
        if individual_scores["accuracy"] < 95:
            recommendations.append(
                "Review numerical algorithms for potential precision issues"
            )
            if individual_scores["accuracy"] < 80:
                critical_issues.append(
                    "Scientific accuracy is compromised - immediate review required"
                )

        # Integration recommendations
        if individual_scores["integration"] < 80:
            recommendations.append(
                "Address failing integration tests before deployment"
            )
            if individual_scores["integration"] < 50:
                critical_issues.append("Major integration failures detected")

        # Stability recommendations
        if individual_scores["stability"] < 70:
            recommendations.append("Improve system stability under stress conditions")
            if individual_scores["stability"] < 50:
                critical_issues.append(
                    "System is unstable under load - not suitable for production"
                )

        # Phase-specific recommendations
        failed_phases = [
            phase
            for phase, status in self.phase_validation_status.items()
            if status == "failed"
        ]
        warning_phases = [
            phase
            for phase, status in self.phase_validation_status.items()
            if status == "warning"
        ]

        if failed_phases:
            critical_issues.append(
                f"The following phases have critical failures: {', '.join(failed_phases)}"
            )
            recommendations.append(
                "Review and fix all failing optimization phases before deployment"
            )

        if warning_phases:
            recommendations.append(
                f"Review and optimize the following phases: {', '.join(warning_phases)}"
            )

        # Specific technical recommendations
        if self.results.get("performance"):
            recommendations.append(
                "Monitor memory usage patterns in production deployment"
            )
            recommendations.append("Implement performance monitoring and alerting")

        if self.results.get("accuracy"):
            recommendations.append(
                "Establish numerical precision monitoring in production"
            )
            recommendations.append(
                "Implement regression testing for scientific accuracy"
            )

        recommendations.append(
            "Consider implementing gradual deployment with monitoring"
        )
        recommendations.append(
            "Establish performance baselines and regression detection"
        )

        return recommendations, critical_issues

    def assess_deployment_readiness(
        self, overall_score: float, critical_issues: list[str]
    ) -> dict[str, Any]:
        """Assess readiness for production deployment"""
        readiness_assessment = {
            "ready_for_production": False,
            "ready_for_beta": False,
            "ready_for_alpha": False,
            "recommended_action": "",
            "deployment_blockers": [],
            "deployment_risks": [],
            "monitoring_requirements": [],
        }

        # Determine deployment readiness
        if overall_score >= 85 and len(critical_issues) == 0:
            readiness_assessment["ready_for_production"] = True
            readiness_assessment["recommended_action"] = (
                "Deploy to production with monitoring"
            )
        elif overall_score >= 75 and len(critical_issues) <= 1:
            readiness_assessment["ready_for_beta"] = True
            readiness_assessment["recommended_action"] = (
                "Deploy to beta testing environment"
            )
        elif overall_score >= 60:
            readiness_assessment["ready_for_alpha"] = True
            readiness_assessment["recommended_action"] = (
                "Deploy to alpha testing with close monitoring"
            )
        else:
            readiness_assessment["recommended_action"] = (
                "Continue development - not ready for deployment"
            )

        # Identify deployment blockers
        if len(critical_issues) > 0:
            readiness_assessment["deployment_blockers"] = critical_issues

        # Identify risks
        if self.phase_validation_status.get("Phase1_Memory") != "passed":
            readiness_assessment["deployment_risks"].append(
                "Memory management issues may cause instability"
            )

        if self.phase_validation_status.get("Phase3_Vectorization") != "passed":
            readiness_assessment["deployment_risks"].append(
                "Numerical accuracy may be compromised"
            )

        if self.phase_validation_status.get("Phase6_Integration") != "passed":
            readiness_assessment["deployment_risks"].append(
                "System integration issues may affect reliability"
            )

        # Monitoring requirements
        readiness_assessment["monitoring_requirements"] = [
            "Memory usage and leak detection",
            "Performance metrics and regression detection",
            "Scientific accuracy validation",
            "System stability and error rates",
            "User experience and responsiveness",
        ]

        return readiness_assessment

    def generate_comprehensive_report(self) -> ComprehensiveValidationReport:
        """Generate comprehensive validation report"""
        self.logger.info("Generating comprehensive validation report...")

        # Calculate scores
        overall_score, individual_scores = self.calculate_overall_scores()

        # Generate recommendations
        recommendations, critical_issues = self.generate_recommendations(
            overall_score, individual_scores
        )

        # Count tests
        test_counts = {
            "performance_tests": len(
                self.results.get("performance", {}).get("measurements", [])
            ),
            "accuracy_tests": getattr(self.results.get("accuracy"), "tests_passed", 0)
            + getattr(self.results.get("accuracy"), "tests_failed", 0),
            "integration_tests": self._count_integration_tests(),
            "stability_tests": len(
                getattr(self.results.get("stability"), "stress_test_results", [])
            ),
        }

        # Create validation summary
        validation_summary = ValidationSummary(
            overall_score=overall_score,
            performance_score=individual_scores["performance"],
            accuracy_score=individual_scores["accuracy"],
            integration_score=individual_scores["integration"],
            stability_score=individual_scores["stability"],
            test_counts=test_counts,
            duration_seconds=time.time() - self.start_time,
            recommendations=recommendations,
            critical_issues=critical_issues,
            phase_status=self.phase_validation_status.copy(),
        )

        # Assess deployment readiness
        deployment_readiness = self.assess_deployment_readiness(
            overall_score, critical_issues
        )

        # Detailed recommendations by category
        detailed_recommendations = {
            "performance": [
                r
                for r in recommendations
                if "performance" in r.lower() or "memory" in r.lower()
            ],
            "accuracy": [
                r
                for r in recommendations
                if "accuracy" in r.lower() or "precision" in r.lower()
            ],
            "integration": [
                r
                for r in recommendations
                if "integration" in r.lower() or "phase" in r.lower()
            ],
            "deployment": [
                r
                for r in recommendations
                if "deploy" in r.lower() or "production" in r.lower()
            ],
            "monitoring": [
                r
                for r in recommendations
                if "monitor" in r.lower() or "baseline" in r.lower()
            ],
        }

        # System information
        import platform

        import psutil

        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "validation_components_available": VALIDATION_COMPONENTS_AVAILABLE,
            "quick_mode": self.quick_mode,
        }

        # Create comprehensive report
        report = ComprehensiveValidationReport(
            validation_timestamp=datetime.now().isoformat(),
            system_info=system_info,
            validation_summary=validation_summary,
            performance_results=self.results["performance"],
            accuracy_results=asdict(self.results["accuracy"])
            if self.results["accuracy"]
            else None,
            integration_results=self.results["integration"],
            stability_results=asdict(self.results["stability"])
            if self.results["stability"]
            else None,
            deployment_readiness=deployment_readiness,
            detailed_recommendations=detailed_recommendations,
        )

        return report

    def _count_integration_tests(self) -> int:
        """Count integration tests from results"""
        if not self.results.get("integration"):
            return 0

        stdout = self.results["integration"].get("stdout", "")

        # Try to extract test count from unittest output
        import re

        test_match = re.search(r"Ran (\d+) test", stdout)
        if test_match:
            return int(test_match.group(1))

        # Fallback: count test methods or classes mentioned
        test_lines = [line for line in stdout.split("\n") if "test_" in line]
        return max(len(test_lines), 1)

    def save_report(self, report: ComprehensiveValidationReport):
        """Save comprehensive validation report"""
        timestamp = int(time.time())

        # Save JSON report
        json_file = (
            self.output_dir / f"comprehensive_validation_report_{timestamp}.json"
        )
        with open(json_file, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Save executive summary
        summary_file = self.output_dir / f"executive_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            self._write_executive_summary(f, report)

        # Save detailed technical report
        technical_file = self.output_dir / f"technical_report_{timestamp}.txt"
        with open(technical_file, "w") as f:
            self._write_technical_report(f, report)

        self.logger.info("Comprehensive validation report saved:")
        self.logger.info(f"  JSON Report: {json_file}")
        self.logger.info(f"  Executive Summary: {summary_file}")
        self.logger.info(f"  Technical Report: {technical_file}")

    def _write_executive_summary(self, f, report: ComprehensiveValidationReport):
        """Write executive summary"""
        f.write("XPCS Toolkit Phase 6 Integration Validation\n")
        f.write("Executive Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Validation Date: {report.validation_timestamp}\n")
        f.write(f"Overall Score: {report.validation_summary.overall_score:.1f}%\n\n")

        # Deployment recommendation
        f.write("DEPLOYMENT RECOMMENDATION:\n")
        f.write(f"  {report.deployment_readiness['recommended_action']}\n\n")

        if report.deployment_readiness["ready_for_production"]:
            f.write("✓ READY FOR PRODUCTION DEPLOYMENT\n")
        elif report.deployment_readiness["ready_for_beta"]:
            f.write("✓ READY FOR BETA TESTING\n")
        elif report.deployment_readiness["ready_for_alpha"]:
            f.write("⚠ READY FOR ALPHA TESTING ONLY\n")
        else:
            f.write("✗ NOT READY FOR DEPLOYMENT\n")

        f.write("\n")

        # Score breakdown
        f.write("VALIDATION SCORES:\n")
        f.write(
            f"  Performance:     {report.validation_summary.performance_score:.1f}%\n"
        )
        f.write(f"  Accuracy:        {report.validation_summary.accuracy_score:.1f}%\n")
        f.write(
            f"  Integration:     {report.validation_summary.integration_score:.1f}%\n"
        )
        f.write(
            f"  Stability:       {report.validation_summary.stability_score:.1f}%\n\n"
        )

        # Phase status
        f.write("OPTIMIZATION PHASE STATUS:\n")
        for phase, status in report.validation_summary.phase_status.items():
            status_symbol = {
                "passed": "✓",
                "warning": "⚠",
                "failed": "✗",
                "pending": "○",
            }
            f.write(f"  {status_symbol.get(status, '?')} {phase}: {status.upper()}\n")

        f.write("\n")

        # Critical issues
        if report.validation_summary.critical_issues:
            f.write("CRITICAL ISSUES:\n")
            for issue in report.validation_summary.critical_issues:
                f.write(f"  - {issue}\n")
            f.write("\n")

        # Top recommendations
        f.write("KEY RECOMMENDATIONS:\n")
        for rec in report.validation_summary.recommendations[:5]:
            f.write(f"  - {rec}\n")

        f.write("\n")

        # Test summary
        total_tests = sum(report.validation_summary.test_counts.values())
        f.write("TESTING SUMMARY:\n")
        f.write(f"  Total Tests Run: {total_tests}\n")
        f.write(
            f"  Validation Duration: {report.validation_summary.duration_seconds / 60:.1f} minutes\n"
        )

    def _write_technical_report(self, f, report: ComprehensiveValidationReport):
        """Write detailed technical report"""
        f.write("XPCS Toolkit Phase 6 Integration Validation\n")
        f.write("Detailed Technical Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Report Generated: {report.validation_timestamp}\n")
        f.write(f"System: {report.system_info['platform']}\n")
        f.write(f"Python: {report.system_info['python_version'].split()[0]}\n")
        f.write(f"CPU Cores: {report.system_info['cpu_count']}\n")
        f.write(f"Memory: {report.system_info['memory_total_gb']:.1f}GB\n\n")

        # Detailed results for each validation category
        f.write("PERFORMANCE VALIDATION:\n")
        f.write("-" * 30 + "\n")
        if report.performance_results:
            f.write(f"Score: {report.validation_summary.performance_score:.1f}%\n")
            if "summary" in report.performance_results:
                for phase, metrics in report.performance_results["summary"].items():
                    f.write(f"  {phase}:\n")
                    f.write(f"    Avg Time: {metrics['avg_execution_time']:.3f}s\n")
                    f.write(f"    Max Memory: {metrics['max_memory_usage_mb']:.1f}MB\n")
        else:
            f.write("Performance validation not completed\n")
        f.write("\n")

        f.write("ACCURACY VALIDATION:\n")
        f.write("-" * 30 + "\n")
        if report.accuracy_results:
            f.write(f"Score: {report.validation_summary.accuracy_score:.1f}%\n")
            f.write(f"Tests Passed: {report.accuracy_results.get('tests_passed', 0)}\n")
            f.write(f"Tests Failed: {report.accuracy_results.get('tests_failed', 0)}\n")
        else:
            f.write("Accuracy validation not completed\n")
        f.write("\n")

        f.write("INTEGRATION TESTING:\n")
        f.write("-" * 30 + "\n")
        if report.integration_results:
            f.write(f"Score: {report.validation_summary.integration_score:.1f}%\n")
            f.write(f"Success: {report.integration_results['success']}\n")
            f.write(f"Return Code: {report.integration_results['return_code']}\n")
        else:
            f.write("Integration testing not completed\n")
        f.write("\n")

        f.write("STABILITY TESTING:\n")
        f.write("-" * 30 + "\n")
        if report.stability_results:
            f.write(f"Score: {report.validation_summary.stability_score:.1f}%\n")
            stress_tests = report.stability_results.get("stress_test_results", [])
            for test in stress_tests:
                f.write(
                    f"  {test['test_name']}: {test['system_stability_score']:.1f}%\n"
                )
        else:
            f.write("Stability testing not completed\n")
        f.write("\n")

        # Detailed recommendations by category
        f.write("DETAILED RECOMMENDATIONS:\n")
        f.write("-" * 30 + "\n")
        for category, recs in report.detailed_recommendations.items():
            if recs:
                f.write(f"{category.upper()}:\n")
                for rec in recs:
                    f.write(f"  - {rec}\n")
                f.write("\n")

        # Deployment guidance
        f.write("DEPLOYMENT GUIDANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Recommended Action: {report.deployment_readiness['recommended_action']}\n"
        )

        if report.deployment_readiness["deployment_blockers"]:
            f.write("\nDeployment Blockers:\n")
            for blocker in report.deployment_readiness["deployment_blockers"]:
                f.write(f"  - {blocker}\n")

        if report.deployment_readiness["deployment_risks"]:
            f.write("\nDeployment Risks:\n")
            for risk in report.deployment_readiness["deployment_risks"]:
                f.write(f"  - {risk}\n")

        f.write("\nRequired Monitoring:\n")
        for requirement in report.deployment_readiness["monitoring_requirements"]:
            f.write(f"  - {requirement}\n")

    def run_comprehensive_validation(self) -> ComprehensiveValidationReport:
        """Run complete validation suite"""
        self.logger.info("Starting comprehensive XPCS Toolkit validation")
        self.logger.info("=" * 60)

        try:
            # Run all validation components
            validation_steps = [
                ("Performance Benchmarks", self.run_performance_benchmarks),
                ("Scientific Accuracy", self.run_accuracy_validation),
                ("Integration Tests", self.run_integration_tests),
                ("Stability Tests", self.run_stability_tests),
            ]

            for step_name, step_func in validation_steps:
                self.logger.info(f"Starting {step_name}...")
                success = step_func()

                if success:
                    self.logger.info(f"{step_name} completed successfully")
                else:
                    self.logger.warning(f"{step_name} completed with issues")

                # Brief pause between validation steps
                time.sleep(1)

            # Generate comprehensive report
            report = self.generate_comprehensive_report()

            # Save report
            self.save_report(report)

            return report

        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            import traceback

            traceback.print_exc()
            raise


def main():
    """Main function to run comprehensive validation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="XPCS Toolkit Comprehensive Validation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run validation in quick mode (reduced test duration)",
    )
    parser.add_argument(
        "--output-dir",
        default="comprehensive_validation_results",
        help="Output directory for validation results",
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = ValidationOrchestrator(
        output_dir=args.output_dir, quick_mode=args.quick
    )

    try:
        # Run comprehensive validation
        report = orchestrator.run_comprehensive_validation()

        # Print final summary
        print("\n" + "=" * 80)
        print("XPCS TOOLKIT PHASE 6 INTEGRATION VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Overall Score: {report.validation_summary.overall_score:.1f}%")
        print(
            f"Duration: {report.validation_summary.duration_seconds / 60:.1f} minutes"
        )
        print()

        # Deployment recommendation
        print("DEPLOYMENT RECOMMENDATION:")
        print(f"  {report.deployment_readiness['recommended_action']}")
        print()

        # Score breakdown
        print("VALIDATION SCORES:")
        print(f"  Performance:  {report.validation_summary.performance_score:5.1f}%")
        print(f"  Accuracy:     {report.validation_summary.accuracy_score:5.1f}%")
        print(f"  Integration:  {report.validation_summary.integration_score:5.1f}%")
        print(f"  Stability:    {report.validation_summary.stability_score:5.1f}%")
        print()

        # Overall assessment
        if report.validation_summary.overall_score >= 90:
            print(
                "🎉 EXCELLENT: System integration is outstanding - ready for production!"
            )
        elif report.validation_summary.overall_score >= 80:
            print("✅ GOOD: System integration is solid - ready for deployment")
        elif report.validation_summary.overall_score >= 70:
            print(
                "⚠️  ACCEPTABLE: System integration is adequate - address issues before deployment"
            )
        elif report.validation_summary.overall_score >= 50:
            print("❌ POOR: System integration needs significant improvement")
        else:
            print(
                "🚨 CRITICAL: System integration has major issues - extensive work required"
            )

        # Key recommendations
        if report.validation_summary.recommendations:
            print("\nKEY RECOMMENDATIONS:")
            for i, rec in enumerate(report.validation_summary.recommendations[:5], 1):
                print(f"  {i}. {rec}")

        return report.validation_summary.overall_score >= 70  # Pass threshold

    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return False
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
