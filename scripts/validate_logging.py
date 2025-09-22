#!/usr/bin/env python3
"""
Logging Standards Validation Script for XPCS Toolkit

This script validates that the codebase follows logging standards and best practices.
It can be run manually or integrated into CI/CD pipelines.

Features:
- Scan for missing logging imports in Python files
- Check for remaining print statements
- Validate logging patterns and standards
- Generate logging coverage reports
- Check for proper logger naming conventions
- Verify structured logging patterns

Usage:
    python scripts/validate_logging.py
    python scripts/validate_logging.py --fix-imports
    python scripts/validate_logging.py --report-format json
    python scripts/validate_logging.py --exclude tests/
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


# ANSI color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


class LoggingVisitor(ast.NodeVisitor):
    """AST visitor for analyzing logging patterns in Python code."""

    def __init__(self, validator, file_path):
        self.validator = validator
        self.file_path = file_path
        self.logger_names = set()
        self.has_logging_import = False
        self.has_get_logger_call = False
        self.functions_with_logging = set()
        self.classes_with_logging = set()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "logging":
                self.has_logging_import = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and "logging_config" in node.module:
            for alias in node.names:
                if alias.name == "get_logger":
                    self.has_logging_import = True
        self.generic_visit(node)

    def visit_Call(self, node):
        self._check_get_logger_call(node)
        self._check_print_call(node)
        self.generic_visit(node)

    def _check_get_logger_call(self, node):
        """Check for get_logger calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "get_logger":
            self.has_get_logger_call = True

    def _check_print_call(self, node):
        """Check for print calls that should be logging."""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            line_no = node.lineno
            if not self._is_allowed_print_context():
                self.validator.results["print_statements"].append(
                    {
                        "file": str(self.file_path),
                        "line": line_no,
                        "type": "print_call",
                    }
                )

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and "logger" in target.id.lower():
                self.logger_names.add(target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self._has_logging_calls(node):
            self.functions_with_logging.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if self._has_logging_calls(node):
            self.classes_with_logging.add(node.name)
        self.generic_visit(node)

    def _has_logging_calls(self, node):
        """Check if a node contains logging calls."""
        logging_methods = ["debug", "info", "warning", "error", "critical", "exception"]
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and "logger" in child.value.id.lower()
                and child.attr in logging_methods
            ):
                return True
        return False

    def _is_allowed_print_context(self):
        """Check if print statement is in allowed context."""
        return False


class LoggingValidator:
    """Main validator class for logging standards."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.test_dir = project_root / "tests"

        # Configuration
        self.exclude_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "venv",
            ".venv",
            "node_modules",
            ".tox",
        ]

        # Expected logging imports
        self.expected_imports = [
            "from xpcs_toolkit.utils.logging_config import get_logger",
            "from xpcs_toolkit.utils.log_templates import",
        ]

        # Results storage
        self.results = {
            "files_checked": 0,
            "files_with_issues": 0,
            "missing_imports": [],
            "print_statements": [],
            "improper_logger_names": [],
            "missing_exception_logging": [],
            "performance_issues": [],
            "summary": {},
        }

    def run_validation(
        self, exclude_dirs: list[str] | None = None, fix_imports: bool = False
    ) -> dict[str, Any]:
        """Run complete validation of logging standards."""
        print(f"{Colors.BOLD}XPCS Toolkit Logging Standards Validation{Colors.RESET}")
        print(f"Scanning project: {self.project_root}")
        print()

        # Get all Python files
        python_files = self._get_python_files(exclude_dirs or [])

        print(f"Found {len(python_files)} Python files to analyze")
        print()

        # Analyze each file
        for file_path in python_files:
            self._analyze_file(file_path, fix_imports)

        # Generate summary
        self._generate_summary()

        # Print results
        self._print_results()

        return self.results

    def _get_python_files(self, exclude_dirs: list[str]) -> list[Path]:
        """Get all Python files in the project."""
        python_files = []
        exclude_set = set(self.exclude_patterns + exclude_dirs)

        for py_file in self.project_root.rglob("*.py"):
            # Check if file should be excluded
            if any(exclude_part in str(py_file) for exclude_part in exclude_set):
                continue

            # Skip files in excluded directories
            skip_file = False
            for part in py_file.parts:
                if part in exclude_set:
                    skip_file = True
                    break

            if not skip_file:
                python_files.append(py_file)

        return sorted(python_files)

    def _analyze_file(self, file_path: Path, fix_imports: bool = False):
        """Analyze a single Python file for logging standards."""
        self.results["files_checked"] += 1

        try:
            content = file_path.read_text(encoding="utf-8")

            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(file_path, tree)
            except SyntaxError as e:
                print(
                    f"{Colors.YELLOW}Warning: Syntax error in {file_path}: {e}{Colors.RESET}"
                )

            # Check for print statements
            self._check_print_statements(file_path, content)

            # Check for proper logger imports and usage
            self._check_logging_imports(file_path, content, fix_imports)

            # Check for proper exception handling
            self._check_exception_handling(file_path, content)

        except Exception as e:
            print(f"{Colors.RED}Error analyzing {file_path}: {e}{Colors.RESET}")

    def _analyze_ast(self, file_path: Path, tree: ast.AST):
        """Analyze AST for logging patterns."""
        visitor = self._create_logging_visitor(file_path)
        visitor.visit(tree)
        self._check_logger_naming(file_path, visitor.logger_names)

    def _create_logging_visitor(self, file_path: Path):
        """Create and configure the logging visitor."""
        return LoggingVisitor(self, file_path)

    def _check_print_statements(self, file_path: Path, content: str):
        """Check for print statements that should be replaced with logging."""
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments and docstrings
            stripped_line = line.strip()
            if stripped_line.startswith(("#", '"""', "'''")):
                continue

            # Look for print statements
            if re.search(r"\bprint\s*\(", line):
                # Allow prints in specific contexts
                if self._is_allowed_print_context(file_path, i, lines):
                    continue

                self.results["print_statements"].append(
                    {
                        "file": str(file_path),
                        "line": i,
                        "content": line.strip(),
                        "suggestion": "Replace with logger.info() or appropriate log level",
                    }
                )

    def _is_allowed_print_context(
        self, file_path: Path, line_no: int, lines: list[str]
    ) -> bool:
        """Check if print statement is in an allowed context."""
        # Allow in __main__ blocks
        for line in lines[max(0, line_no - 10) : line_no + 5]:
            if 'if __name__ == "__main__"' in line:
                return True

        # Allow in CLI scripts that are not modules
        if "cli" in str(file_path).lower() or "script" in str(file_path).lower():
            return True

        # Allow in specific debug sections
        context_lines = lines[max(0, line_no - 3) : line_no + 2]
        for line in context_lines:
            if any(
                debug_marker in line.lower()
                for debug_marker in ["debug", "temporary", "todo", "fixme"]
            ):
                return True

        return False

    def _check_logging_imports(
        self, file_path: Path, content: str, fix_imports: bool = False
    ):
        """Check for proper logging imports."""
        # Skip non-module files
        if self._should_skip_logging_check(file_path):
            return

        has_get_logger_import = (
            "from xpcs_toolkit.utils.logging_config import get_logger" in content
        )
        has_logger_usage = re.search(
            r"logger\.(debug|info|warning|error|critical|exception)", content
        )

        if has_logger_usage and not has_get_logger_import:
            issue = {
                "file": str(file_path),
                "type": "missing_import",
                "message": "File uses logger but missing get_logger import",
                "suggestion": "Add: from xpcs_toolkit.utils.logging_config import get_logger",
            }

            self.results["missing_imports"].append(issue)

            if fix_imports:
                self._fix_missing_import(file_path, content)

    def _should_skip_logging_check(self, file_path: Path) -> bool:
        """Check if file should be skipped for logging checks."""
        # Skip certain file types
        skip_patterns = [
            "__init__.py",
            "setup.py",
            "conftest.py",
            "conf.py",
            "_rc.py",  # Resource files
        ]

        file_name = file_path.name
        return any(pattern in file_name for pattern in skip_patterns)

    def _fix_missing_import(self, file_path: Path, content: str):
        """Automatically fix missing logging imports."""
        lines = content.split("\n")

        # Find the best place to insert the import
        insert_index = self._find_import_insertion_point(lines)

        # Insert the import
        new_import = "from xpcs_toolkit.utils.logging_config import get_logger"
        lines.insert(insert_index, new_import)

        # Also add logger initialization if not present
        if "logger = get_logger(" not in content:
            lines.insert(insert_index + 1, "")
            lines.insert(insert_index + 2, "logger = get_logger(__name__)")

        # Write back to file
        new_content = "\n".join(lines)
        file_path.write_text(new_content, encoding="utf-8")

        print(f"{Colors.GREEN}Fixed imports in: {file_path}{Colors.RESET}")

    def _find_import_insertion_point(self, lines: list[str]) -> int:
        """Find the best place to insert logging imports."""
        last_import_index = -1
        in_docstring = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Track docstrings
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue

            if in_docstring:
                continue

            # Skip comments and empty lines at the top
            if not stripped or stripped.startswith("#"):
                continue

            # Found an import
            if stripped.startswith("import ") or (
                stripped.startswith("from ") and "import" in stripped
            ):
                last_import_index = i
            elif last_import_index >= 0:
                # No more imports, insert after the last one
                return last_import_index + 1

        # If no imports found, insert after docstring/comments
        return max(0, last_import_index + 1) if last_import_index >= 0 else 0

    def _check_exception_handling(self, file_path: Path, content: str):
        """Check for proper exception handling with logging."""
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Look for try/except blocks
            if re.search(r"^\s*except\s+\w+", line):
                # Check if the except block has logging
                except_block_lines = self._get_except_block_lines(lines, i)
                has_logging = any(
                    re.search(r"logger\.(error|exception|critical)", block_line)
                    for block_line in except_block_lines
                )

                if not has_logging:
                    self.results["missing_exception_logging"].append(
                        {
                            "file": str(file_path),
                            "line": i + 1,
                            "suggestion": "Add logger.exception() or logger.error() in except block",
                        }
                    )

    def _get_except_block_lines(
        self, lines: list[str], except_line_index: int
    ) -> list[str]:
        """Get all lines in an except block."""
        block_lines = []

        # Find the indentation level of the except statement
        except_line = lines[except_line_index]
        except_indent = len(except_line) - len(except_line.lstrip())

        # Collect lines in the except block
        for i in range(except_line_index + 1, len(lines)):
            line = lines[i]

            # Empty lines are part of the block
            if not line.strip():
                block_lines.append(line)
                continue

            # Check indentation
            line_indent = len(line) - len(line.lstrip())

            # If indentation is same or less, we've left the block
            if line_indent <= except_indent:
                break

            block_lines.append(line)

        return block_lines

    def _check_logger_naming(self, file_path: Path, logger_names: set[str]):
        """Check for proper logger naming conventions."""
        for name in logger_names:
            if name != "logger":
                # Allow specific patterns
                allowed_patterns = [
                    "logger",
                    "class_logger",
                    "thread_logger",
                    "perf_logger",
                ]
                if not any(pattern in name for pattern in allowed_patterns):
                    self.results["improper_logger_names"].append(
                        {
                            "file": str(file_path),
                            "logger_name": name,
                            "suggestion": 'Use "logger" as the standard name',
                        }
                    )

    def _generate_summary(self):
        """Generate validation summary."""
        total_issues = (
            len(self.results["missing_imports"])
            + len(self.results["print_statements"])
            + len(self.results["improper_logger_names"])
            + len(self.results["missing_exception_logging"])
        )

        self.results["files_with_issues"] = len(
            {
                issue["file"]
                for issues in [
                    self.results["missing_imports"],
                    self.results["print_statements"],
                    self.results["improper_logger_names"],
                    self.results["missing_exception_logging"],
                ]
                for issue in issues
            }
        )

        self.results["summary"] = {
            "total_files": self.results["files_checked"],
            "files_with_issues": self.results["files_with_issues"],
            "total_issues": total_issues,
            "issue_breakdown": {
                "missing_imports": len(self.results["missing_imports"]),
                "print_statements": len(self.results["print_statements"]),
                "improper_logger_names": len(self.results["improper_logger_names"]),
                "missing_exception_logging": len(
                    self.results["missing_exception_logging"]
                ),
            },
        }

    def _print_results(self):
        """Print validation results to console."""
        summary = self.results["summary"]

        # Print summary
        print(f"{Colors.BOLD}Validation Summary{Colors.RESET}")
        print(f"Files checked: {summary['total_files']}")
        print(f"Files with issues: {summary['files_with_issues']}")
        print(f"Total issues: {summary['total_issues']}")
        print()

        # Print issue breakdown
        if summary["total_issues"] > 0:
            print(f"{Colors.BOLD}Issue Breakdown:{Colors.RESET}")
            for issue_type, count in summary["issue_breakdown"].items():
                if count > 0:
                    color = Colors.RED if count > 0 else Colors.GREEN
                    print(
                        f"  {color}{issue_type.replace('_', ' ').title()}: {count}{Colors.RESET}"
                    )
            print()

        # Print detailed issues
        self._print_detailed_issues()

        # Print overall result
        if summary["total_issues"] == 0:
            print(
                f"{Colors.GREEN}{Colors.BOLD}✓ All files follow logging standards!{Colors.RESET}"
            )
        else:
            print(
                f"{Colors.YELLOW}{Colors.BOLD}! Found {summary['total_issues']} issues that need attention{Colors.RESET}"
            )

    def _print_detailed_issues(self):
        """Print detailed issues by category."""
        self._print_missing_imports()
        self._print_print_statements()
        self._print_improper_logger_names()
        self._print_missing_exception_logging()

    def _print_missing_imports(self):
        """Print missing import issues."""
        if self.results["missing_imports"]:
            print(f"{Colors.RED}{Colors.BOLD}Missing Logging Imports:{Colors.RESET}")
            for issue in self.results["missing_imports"]:
                print(f"  {issue['file']}")
                print(f"    {Colors.YELLOW}{issue['message']}{Colors.RESET}")
                print(f"    Suggestion: {issue['suggestion']}")
            print()

    def _print_print_statements(self):
        """Print print statement issues."""
        if self.results["print_statements"]:
            print(f"{Colors.YELLOW}{Colors.BOLD}Print Statements Found:{Colors.RESET}")
            for issue in self.results["print_statements"]:
                print(f"  {issue['file']}:{issue['line']}")
                if "content" in issue:
                    print(f"    {Colors.CYAN}{issue['content']}{Colors.RESET}")
                if "suggestion" in issue:
                    print(f"    Suggestion: {issue['suggestion']}")
            print()

    def _print_improper_logger_names(self):
        """Print improper logger name issues."""
        if self.results["improper_logger_names"]:
            print(f"{Colors.MAGENTA}{Colors.BOLD}Improper Logger Names:{Colors.RESET}")
            for issue in self.results["improper_logger_names"]:
                print(f"  {issue['file']}")
                print(f"    Logger name: '{issue['logger_name']}'")
                print(f"    Suggestion: {issue['suggestion']}")
            print()

    def _print_missing_exception_logging(self):
        """Print missing exception logging issues."""
        if self.results["missing_exception_logging"]:
            print(f"{Colors.BLUE}{Colors.BOLD}Missing Exception Logging:{Colors.RESET}")
            for issue in self.results["missing_exception_logging"]:
                print(f"  {issue['file']}:{issue['line']}")
                print(f"    Suggestion: {issue['suggestion']}")
            print()

    def generate_report(self, format_type: str = "text") -> str:
        """Generate a report in the specified format."""
        if format_type == "json":
            return json.dumps(self.results, indent=2)
        if format_type == "text":
            return self._generate_text_report()
        raise ValueError(f"Unknown report format: {format_type}")

    def _generate_text_report(self) -> str:
        """Generate a text-based report."""
        lines = []
        lines.append("XPCS Toolkit Logging Standards Validation Report")
        lines.append("=" * 50)
        lines.append("")

        summary = self.results["summary"]
        lines.append(f"Files checked: {summary['total_files']}")
        lines.append(f"Files with issues: {summary['files_with_issues']}")
        lines.append(f"Total issues: {summary['total_issues']}")
        lines.append("")

        # Add detailed issues
        for issue_type, issues in self.results.items():
            if isinstance(issues, list) and issues:
                lines.append(f"{issue_type.replace('_', ' ').title()}:")
                for issue in issues:
                    lines.append(f"  - {issue}")
                lines.append("")

        return "\n".join(lines)


def create_pre_commit_hook():
    """Create a pre-commit hook configuration."""
    hook_config = """
# Add to your .pre-commit-config.yaml file:

repos:
  - repo: local
    hooks:
      - id: validate-logging
        name: Validate Logging Standards
        entry: python scripts/validate_logging.py
        language: system
        types: [python]
        pass_filenames: false
"""
    return hook_config


def main():
    """Main function for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate logging standards in XPCS Toolkit"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Directories to exclude from validation",
    )
    parser.add_argument(
        "--fix-imports", action="store_true", help="Automatically fix missing imports"
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "json"],
        default="text",
        help="Report output format",
    )
    parser.add_argument("--output-file", type=Path, help="Save report to file")
    parser.add_argument(
        "--pre-commit-hook",
        action="store_true",
        help="Generate pre-commit hook configuration",
    )

    args = parser.parse_args()

    if args.pre_commit_hook:
        print(create_pre_commit_hook())
        return 0

    # Run validation
    validator = LoggingValidator(args.project_root)
    results = validator.run_validation(
        exclude_dirs=args.exclude, fix_imports=args.fix_imports
    )

    # Generate report
    if args.output_file:
        report = validator.generate_report(args.report_format)
        args.output_file.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {args.output_file}")

    # Return exit code based on results
    total_issues = results["summary"]["total_issues"]
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
