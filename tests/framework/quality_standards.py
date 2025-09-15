"""Test Quality Standards and Automated Quality Checking for XPCS Toolkit.

This module provides automated quality checking for test code, ensuring consistent
patterns, maintainability, and scientific rigor across all test modules.

Features:
- Test code pattern validation
- Quality metrics calculation
- Anti-pattern detection
- Style consistency checking
- Scientific testing standards enforcement
- Automated quality reporting

Usage:
    python tests/quality_standards.py --check-all
    python tests/quality_standards.py --fix-issues
    python tests/quality_standards.py --report
"""

import ast
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class QualityLevel(Enum):
    """Test quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class IssueType(Enum):
    """Types of quality issues that can be detected."""

    MISSING_DOCSTRING = "missing_docstring"
    WEAK_ASSERTION = "weak_assertion"
    NO_TEARDOWN = "no_teardown"
    MAGIC_NUMBERS = "magic_numbers"
    POOR_NAMING = "poor_naming"
    MISSING_FIXTURE = "missing_fixture"
    EXCESSIVE_MOCKING = "excessive_mocking"
    NO_ERROR_TESTING = "no_error_testing"
    SCIENTIFIC_TOLERANCE = "scientific_tolerance"
    PERFORMANCE_ISSUE = "performance_issue"


@dataclass
class QualityIssue:
    """Represents a quality issue in test code."""

    file_path: Path
    line_number: int
    issue_type: IssueType
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str | None = None
    auto_fixable: bool = False


@dataclass
class TestQualityMetrics:
    """Comprehensive quality metrics for a test file or suite."""

    file_path: Path

    # Basic metrics
    total_tests: int
    total_lines: int
    test_classes: int

    # Docstring coverage
    documented_tests: int
    documented_classes: int
    docstring_coverage: float

    # Assertion quality
    total_assertions: int
    assertion_density: float  # assertions per test
    weak_assertions: int  # assertTrue, assertNotNone, etc.

    # Scientific rigor
    numerical_comparisons: int
    tolerance_specified: int
    array_comparisons: int
    statistical_tests: int

    # Test patterns
    setup_methods: int
    teardown_methods: int
    fixture_usage: int
    mock_usage: int
    parametrized_tests: int

    # Error testing
    exception_tests: int
    edge_case_tests: int

    # Quality score (0-100)
    quality_score: float
    quality_level: QualityLevel

    # Issues found
    issues: list[QualityIssue]


class TestCodeAnalyzer(ast.NodeVisitor):
    """AST-based analyzer for test code quality assessment."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.metrics = TestQualityMetrics(
            file_path=file_path,
            total_tests=0,
            total_lines=0,
            test_classes=0,
            documented_tests=0,
            documented_classes=0,
            docstring_coverage=0.0,
            total_assertions=0,
            assertion_density=0.0,
            weak_assertions=0,
            numerical_comparisons=0,
            tolerance_specified=0,
            array_comparisons=0,
            statistical_tests=0,
            setup_methods=0,
            teardown_methods=0,
            fixture_usage=0,
            mock_usage=0,
            parametrized_tests=0,
            exception_tests=0,
            edge_case_tests=0,
            quality_score=0.0,
            quality_level=QualityLevel.POOR,
            issues=[],
        )
        self.current_class = None
        self.current_function = None
        self.in_test_function = False

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions to analyze test classes."""
        if self._is_test_class(node.name):
            self.metrics.test_classes += 1
            self.current_class = node

            # Check class docstring
            if ast.get_docstring(node):
                self.metrics.documented_classes += 1
            else:
                self._add_issue(
                    node.lineno,
                    IssueType.MISSING_DOCSTRING,
                    "warning",
                    f"Test class {node.name} missing docstring",
                    "Add descriptive docstring explaining test purpose",
                )

        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions to analyze test methods."""
        if self._is_test_function(node.name):
            self.metrics.total_tests += 1
            self.current_function = node
            self.in_test_function = True

            # Check test docstring
            if ast.get_docstring(node):
                self.metrics.documented_tests += 1
            else:
                self._add_issue(
                    node.lineno,
                    IssueType.MISSING_DOCSTRING,
                    "warning",
                    f"Test function {node.name} missing docstring",
                    "Add docstring explaining what the test validates",
                )

            # Check for parametrization
            if any(
                isinstance(d, ast.Name) and d.id == "parametrize"
                for d in node.decorator_list
            ):
                self.metrics.parametrized_tests += 1

            # Analyze test body
            self._analyze_test_body(node)

        elif self._is_setup_method(node.name):
            self.metrics.setup_methods += 1
        elif self._is_teardown_method(node.name):
            self.metrics.teardown_methods += 1

        self.generic_visit(node)
        self.current_function = None
        self.in_test_function = False

    def visit_Call(self, node: ast.Call):
        """Visit function calls to analyze assertions and patterns."""
        if self.in_test_function:
            # Count assertions
            if self._is_assertion_call(node):
                self.metrics.total_assertions += 1

                # Check for weak assertions
                if self._is_weak_assertion(node):
                    self.metrics.weak_assertions += 1
                    self._add_issue(
                        node.lineno,
                        IssueType.WEAK_ASSERTION,
                        "warning",
                        f"Weak assertion: {self._get_call_name(node)}",
                        "Use more specific assertions like assertEqual, assertIn, etc.",
                    )

                # Check for numerical comparisons
                if self._is_numerical_assertion(node):
                    self.metrics.numerical_comparisons += 1
                    if not self._has_tolerance_parameter(node):
                        self._add_issue(
                            node.lineno,
                            IssueType.SCIENTIFIC_TOLERANCE,
                            "warning",
                            "Numerical comparison without explicit tolerance",
                            "Use assertAlmostEqual or np.testing.assert_allclose with explicit tolerance",
                        )
                    else:
                        self.metrics.tolerance_specified += 1

                # Check for array comparisons
                if self._is_array_assertion(node):
                    self.metrics.array_comparisons += 1

            # Check for mock usage
            if self._is_mock_call(node):
                self.metrics.mock_usage += 1

            # Check for fixture usage
            if self._is_fixture_call(node):
                self.metrics.fixture_usage += 1

        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        """Visit with statements to find exception testing."""
        if self.in_test_function:
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Attribute)
                    and item.context_expr.func.attr == "assertRaises"
                ):
                    self.metrics.exception_tests += 1

        self.generic_visit(node)

    def _analyze_test_body(self, node: ast.FunctionDef):
        """Analyze the body of a test function for patterns and issues."""
        # Check for magic numbers
        for child in ast.walk(node):
            if isinstance(child, ast.Num) and isinstance(child.n, (int, float)):
                if child.n not in [0, 1, -1] and child.n > 100:
                    self._add_issue(
                        child.lineno,
                        IssueType.MAGIC_NUMBERS,
                        "info",
                        f"Magic number {child.n} found",
                        "Consider using named constants for clarity",
                    )

    def _is_test_class(self, name: str) -> bool:
        """Check if a class name follows test class naming conventions."""
        return name.startswith("Test") or name.endswith("Test") or "Test" in name

    def _is_test_function(self, name: str) -> bool:
        """Check if a function name follows test function naming conventions."""
        return name.startswith("test_")

    def _is_setup_method(self, name: str) -> bool:
        """Check if a method is a setup method."""
        return name in ["setUp", "setup_method", "setup_class"]

    def _is_teardown_method(self, name: str) -> bool:
        """Check if a method is a teardown method."""
        return name in ["tearDown", "teardown_method", "teardown_class"]

    def _is_assertion_call(self, node: ast.Call) -> bool:
        """Check if a call is an assertion."""
        name = self._get_call_name(node)
        return name and (
            name.startswith(("assert", "self.assert")) or "assert_" in name
        )

    def _is_weak_assertion(self, node: ast.Call) -> bool:
        """Check if an assertion is considered weak."""
        name = self._get_call_name(node)
        weak_assertions = [
            "assertTrue",
            "assertFalse",
            "assertIsNotNone",
            "assertIsNone",
        ]
        return any(weak in name for weak in weak_assertions)

    def _is_numerical_assertion(self, node: ast.Call) -> bool:
        """Check if assertion involves numerical comparison."""
        name = self._get_call_name(node)
        return any(
            num_assert in name
            for num_assert in [
                "assertEqual",
                "assertAlmostEqual",
                "assertGreater",
                "assertLess",
                "assert_allclose",
                "assert_array_equal",
            ]
        )

    def _has_tolerance_parameter(self, node: ast.Call) -> bool:
        """Check if assertion has tolerance parameter."""
        for keyword in node.keywords:
            if keyword.arg in ["places", "delta", "rtol", "atol"]:
                return True
        return False

    def _is_array_assertion(self, node: ast.Call) -> bool:
        """Check if assertion is for array comparison."""
        name = self._get_call_name(node)
        return any(
            array_assert in name
            for array_assert in ["assert_array_", "assert_allclose", "assertArrayEqual"]
        )

    def _is_mock_call(self, node: ast.Call) -> bool:
        """Check if call involves mocking."""
        name = self._get_call_name(node)
        return name and ("mock" in name.lower() or "Mock" in name)

    def _is_fixture_call(self, node: ast.Call) -> bool:
        """Check if call uses pytest fixtures."""
        # This is simplified - in practice would need more sophisticated detection
        return False

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _add_issue(
        self,
        line_number: int,
        issue_type: IssueType,
        severity: str,
        message: str,
        suggestion: str | None = None,
    ):
        """Add a quality issue to the metrics."""
        issue = QualityIssue(
            file_path=self.file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
        )
        self.metrics.issues.append(issue)

    def finalize_metrics(self):
        """Calculate final quality metrics and score."""
        # Calculate coverage percentages
        if self.metrics.total_tests > 0:
            self.metrics.docstring_coverage = (
                self.metrics.documented_tests / self.metrics.total_tests
            )
            self.metrics.assertion_density = (
                self.metrics.total_assertions / self.metrics.total_tests
            )

        # Calculate quality score (0-100)
        score = 0

        # Docstring coverage (20 points)
        score += self.metrics.docstring_coverage * 20

        # Assertion quality (25 points)
        if self.metrics.total_assertions > 0:
            assertion_quality = 1 - (
                self.metrics.weak_assertions / self.metrics.total_assertions
            )
            score += assertion_quality * 25

        # Scientific rigor (20 points)
        if self.metrics.numerical_comparisons > 0:
            scientific_score = (
                self.metrics.tolerance_specified / self.metrics.numerical_comparisons
            )
            score += scientific_score * 20

        # Test patterns (20 points)
        pattern_score = 0
        if self.metrics.setup_methods > 0:
            pattern_score += 5
        if self.metrics.teardown_methods > 0:
            pattern_score += 5
        if self.metrics.exception_tests > 0:
            pattern_score += 10
        score += pattern_score

        # Penalty for issues (15 points deduction max)
        error_count = sum(
            1 for issue in self.metrics.issues if issue.severity == "error"
        )
        warning_count = sum(
            1 for issue in self.metrics.issues if issue.severity == "warning"
        )
        penalty = min(15, error_count * 5 + warning_count * 2)
        score = max(0, score - penalty)

        self.metrics.quality_score = score

        # Determine quality level
        if score >= 85:
            self.metrics.quality_level = QualityLevel.EXCELLENT
        elif score >= 70:
            self.metrics.quality_level = QualityLevel.GOOD
        elif score >= 50:
            self.metrics.quality_level = QualityLevel.FAIR
        else:
            self.metrics.quality_level = QualityLevel.POOR


class TestQualityChecker:
    """Main class for test quality assessment and reporting."""

    def __init__(self, test_directory: Path | None = None):
        self.test_dir = test_directory or Path(__file__).parent
        self.metrics_by_file: dict[Path, TestQualityMetrics] = {}

    def check_all_tests(self) -> dict[Path, TestQualityMetrics]:
        """Check quality of all test files in the test directory."""
        test_files = self._find_test_files()

        for test_file in test_files:
            try:
                metrics = self._analyze_file(test_file)
                self.metrics_by_file[test_file] = metrics
            except Exception as e:
                print(f"Error analyzing {test_file}: {e}")

        return self.metrics_by_file

    def _find_test_files(self) -> list[Path]:
        """Find all Python test files in the test directory."""
        test_files = []
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend(self.test_dir.rglob(pattern))
        return test_files

    def _analyze_file(self, file_path: Path) -> TestQualityMetrics:
        """Analyze a single test file for quality metrics."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content, filename=str(file_path))

        # Analyze with visitor
        analyzer = TestCodeAnalyzer(file_path)
        analyzer.visit(tree)

        # Set line count
        analyzer.metrics.total_lines = len(content.splitlines())

        # Finalize metrics
        analyzer.finalize_metrics()

        return analyzer.metrics

    def generate_report(self, output_format: str = "text") -> str:
        """Generate comprehensive quality report."""
        if output_format == "json":
            return self._generate_json_report()
        return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        if not self.metrics_by_file:
            return "No test files analyzed."

        report = ["=" * 80]
        report.append("XPCS TOOLKIT TEST QUALITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        total_tests = sum(m.total_tests for m in self.metrics_by_file.values())
        total_files = len(self.metrics_by_file)
        avg_score = (
            sum(m.quality_score for m in self.metrics_by_file.values()) / total_files
        )

        report.append(f"Files Analyzed: {total_files}")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Average Quality Score: {avg_score:.1f}/100")
        report.append("")

        # Quality distribution
        quality_dist = {}
        for level in QualityLevel:
            quality_dist[level] = sum(
                1 for m in self.metrics_by_file.values() if m.quality_level == level
            )

        report.append("Quality Distribution:")
        for level, count in quality_dist.items():
            percentage = (count / total_files) * 100
            report.append(f"  {level.value.title()}: {count} files ({percentage:.1f}%)")
        report.append("")

        # File-by-file analysis
        report.append("INDIVIDUAL FILE ANALYSIS")
        report.append("-" * 40)

        for file_path, metrics in sorted(self.metrics_by_file.items()):
            report.append(f"\nFile: {file_path.name}")
            report.append(
                f"Quality Score: {metrics.quality_score:.1f}/100 ({metrics.quality_level.value})"
            )
            report.append(
                f"Tests: {metrics.total_tests}, Classes: {metrics.test_classes}"
            )
            report.append(f"Docstring Coverage: {metrics.docstring_coverage:.1%}")
            report.append(f"Assertion Density: {metrics.assertion_density:.1f}")

            if metrics.issues:
                report.append(f"Issues ({len(metrics.issues)}):")
                for issue in metrics.issues[:5]:  # Show top 5 issues
                    report.append(f"  Line {issue.line_number}: {issue.message}")
                if len(metrics.issues) > 5:
                    report.append(f"  ... and {len(metrics.issues) - 5} more")

        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS FOR IMPROVEMENT")
        report.append("=" * 80)

        all_issues = []
        for metrics in self.metrics_by_file.values():
            all_issues.extend(metrics.issues)

        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1

        for issue_type, count in sorted(
            issue_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count >= 3:  # Only show common issues
                report.append(
                    f"\n{issue_type.value.replace('_', ' ').title()}: {count} occurrences"
                )
                # Add specific recommendations based on issue type
                if issue_type == IssueType.MISSING_DOCSTRING:
                    report.append(
                        "  - Add descriptive docstrings to all test classes and methods"
                    )
                    report.append(
                        "  - Explain what each test validates and why it's important"
                    )
                elif issue_type == IssueType.WEAK_ASSERTION:
                    report.append(
                        "  - Use specific assertions like assertEqual, assertIn instead of assertTrue"
                    )
                    report.append("  - Provide meaningful assertion messages")
                elif issue_type == IssueType.SCIENTIFIC_TOLERANCE:
                    report.append(
                        "  - Always specify tolerance for numerical comparisons"
                    )
                    report.append(
                        "  - Use np.testing.assert_allclose for array comparisons"
                    )

        return "\n".join(report)

    def _generate_json_report(self) -> str:
        """Generate machine-readable JSON report."""
        report_data = {
            "summary": {
                "files_analyzed": len(self.metrics_by_file),
                "total_tests": sum(
                    m.total_tests for m in self.metrics_by_file.values()
                ),
                "average_score": sum(
                    m.quality_score for m in self.metrics_by_file.values()
                )
                / len(self.metrics_by_file)
                if self.metrics_by_file
                else 0,
                "timestamp": "2025-01-13T00:00:00Z",  # In practice, use actual timestamp
            },
            "files": {},
        }

        for file_path, metrics in self.metrics_by_file.items():
            report_data["files"][str(file_path)] = {
                "quality_score": metrics.quality_score,
                "quality_level": metrics.quality_level.value,
                "total_tests": metrics.total_tests,
                "docstring_coverage": metrics.docstring_coverage,
                "assertion_density": metrics.assertion_density,
                "issues": [
                    {
                        "line": issue.line_number,
                        "type": issue.issue_type.value,
                        "severity": issue.severity,
                        "message": issue.message,
                        "suggestion": issue.suggestion,
                    }
                    for issue in metrics.issues
                ],
            }

        return json.dumps(report_data, indent=2)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="XPCS Toolkit Test Quality Checker")
    parser.add_argument("--check-all", action="store_true", help="Check all test files")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if args.check_all:
        checker = TestQualityChecker()
        checker.check_all_tests()
        report = checker.generate_report(args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Quality report written to {args.output}")
        else:
            print(report)
    else:
        print("Use --check-all to run quality assessment")


if __name__ == "__main__":
    main()
