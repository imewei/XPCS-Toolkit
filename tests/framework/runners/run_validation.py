#!/usr/bin/env python3
"""
Test Suite Validation Runner

Utility script to easily run the comprehensive test suite validation
with different configurations and output formats.

Usage:
    python run_validation.py [options]

    --config CONFIG_FILE    Use custom configuration file
    --format FORMAT         Output format: json, yaml, console (default: console)
    --output OUTPUT_FILE    Output file path (default: auto-generated)
    --verbose               Enable verbose output
    --quiet                 Suppress progress output
    --fail-on-gates         Exit with error if quality gates fail
    --ci                    CI-friendly output format
    --baseline BASELINE     Compare with baseline report

Examples:
    # Basic validation with console output
    python run_validation.py

    # Generate JSON report
    python run_validation.py --format json --output validation_results.json

    # Run in CI mode with strict quality gates
    python run_validation.py --ci --fail-on-gates

    # Compare with previous baseline
    python run_validation.py --baseline previous_validation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Import the validation framework
sys.path.append(str(Path(__file__).parent.parent.parent))
from test_suite_validation import TestSuiteValidator, print_validation_summary


class ValidationRunner:
    """Enhanced runner for test suite validation with configuration support."""

    def __init__(self, config_file: Path | None = None):
        """Initialize the validation runner with optional configuration."""
        self.config_file = (
            config_file or Path(__file__).parent / "validation_config.yaml"
        )
        self.config = self._load_config()
        self.validator = TestSuiteValidator()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"⚠️  Warning: Could not load config file {self.config_file}: {e}")
                return {}
        else:
            print(
                f"⚠️  Warning: Config file {self.config_file} not found, using defaults"
            )
            return {}

    def run_validation(
        self,
        output_format: str = "console",
        output_file: Path | None = None,
        verbose: bool = True,
        quiet: bool = False,
        baseline: Path | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive validation with specified options.

        Args:
            output_format: Output format (console, json, yaml)
            output_file: Output file path (if None, auto-generated)
            verbose: Enable verbose output
            quiet: Suppress progress output
            baseline: Baseline report to compare against

        Returns:
            Validation report dictionary
        """
        if not quiet:
            print("🚀 Starting Test Suite Validation")
            print(f"📁 Project: {self.validator.src_dir.parent.name}")
            print(f"⚙️  Config: {self.config_file.name}")
            print("=" * 60)

        # Apply configuration to validator
        self._apply_config_to_validator()

        # Run validation
        start_time = time.time()

        if not quiet:
            print("🔬 Running comprehensive validation...")

        try:
            report = self.validator.run_comprehensive_validation()

            # Add execution metadata
            execution_time = time.time() - start_time
            report["execution_metadata"] = {
                "execution_time_seconds": execution_time,
                "config_file": str(self.config_file),
                "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runner_version": "1.0.0",
            }

            # Compare with baseline if provided
            if baseline and baseline.exists():
                baseline_comparison = self._compare_with_baseline(report, baseline)
                report["baseline_comparison"] = baseline_comparison

            # Output results
            self._output_results(report, output_format, output_file, verbose, quiet)

            if not quiet:
                print(f"✅ Validation completed in {execution_time:.1f} seconds")

            return report

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            raise

    def _apply_config_to_validator(self):
        """Apply configuration settings to the validator."""
        # This would customize validator behavior based on config
        # For now, we'll store config for use during validation
        if hasattr(self.validator, "_config"):
            self.validator._config = self.config
        else:
            # Add config as attribute for validator to use
            self.validator._config = self.config

    def _compare_with_baseline(
        self, current_report: dict, baseline_file: Path
    ) -> dict[str, Any]:
        """Compare current report with baseline report."""
        try:
            with open(baseline_file, encoding="utf-8") as f:
                baseline_report = json.load(f)

            comparison = {
                "baseline_file": str(baseline_file),
                "baseline_timestamp": baseline_report.get(
                    "validation_timestamp", "unknown"
                ),
                "score_change": current_report.get("overall_test_suite_score", 0)
                - baseline_report.get("overall_test_suite_score", 0),
                "quality_gates_change": {
                    "current": current_report.get("quality_gates_passed", 0),
                    "baseline": baseline_report.get("quality_gates_passed", 0),
                    "change": current_report.get("quality_gates_passed", 0)
                    - baseline_report.get("quality_gates_passed", 0),
                },
                "category_changes": {},
            }

            # Compare category scores
            categories = [
                "coverage_analysis",
                "quality_metrics",
                "scientific_rigor",
                "completeness_analysis",
                "infrastructure_quality",
            ]
            for category in categories:
                current_score = current_report.get(category, {}).get(
                    f"overall_{category.split('_')[0]}_score", 0
                )
                baseline_score = baseline_report.get(category, {}).get(
                    f"overall_{category.split('_')[0]}_score", 0
                )
                comparison["category_changes"][category] = {
                    "current": current_score,
                    "baseline": baseline_score,
                    "change": current_score - baseline_score,
                }

            return comparison

        except Exception as e:
            return {
                "error": f"Could not compare with baseline: {e}",
                "baseline_file": str(baseline_file),
            }

    def _output_results(
        self,
        report: dict[str, Any],
        output_format: str,
        output_file: Path | None,
        verbose: bool,
        quiet: bool,
    ):
        """Output validation results in specified format."""

        if output_format == "console" and not quiet:
            print_validation_summary(report)

            # Show baseline comparison if available
            if "baseline_comparison" in report:
                self._print_baseline_comparison(report["baseline_comparison"])

        elif output_format in ["json", "yaml"]:
            # Determine output file
            if not output_file:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = Path(f"validation_report_{timestamp}.{output_format}")

            # Write report
            try:
                if output_format == "json":
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2, default=str)
                elif output_format == "yaml":
                    with open(output_file, "w", encoding="utf-8") as f:
                        yaml.dump(
                            report, f, default_flow_style=False, allow_unicode=True
                        )

                if not quiet:
                    print(f"📊 Report saved to: {output_file}")

            except Exception as e:
                print(f"❌ Error saving report: {e}")
                raise

    def _print_baseline_comparison(self, comparison: dict[str, Any]):
        """Print baseline comparison summary."""
        if "error" in comparison:
            print(f"\n⚠️  Baseline Comparison Error: {comparison['error']}")
            return

        print("\n📊 Baseline Comparison")
        print(f"Baseline: {comparison.get('baseline_timestamp', 'unknown')}")

        score_change = comparison.get("score_change", 0)
        if score_change > 0:
            print(f"📈 Overall Score: +{score_change:.1f} points (improved)")
        elif score_change < 0:
            print(f"📉 Overall Score: {score_change:.1f} points (declined)")
        else:
            print("➡️  Overall Score: No change")

        gates_change = comparison.get("quality_gates_change", {}).get("change", 0)
        if gates_change > 0:
            print(f"🚪 Quality Gates: +{gates_change} (improved)")
        elif gates_change < 0:
            print(f"🚪 Quality Gates: {gates_change} (declined)")
        else:
            print("🚪 Quality Gates: No change")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive test suite validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation with console output
    python run_validation.py

    # Generate JSON report
    python run_validation.py --format json --output validation_results.json

    # Run in CI mode with strict quality gates
    python run_validation.py --ci --fail-on-gates

    # Compare with previous baseline
    python run_validation.py --baseline previous_validation.json
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Configuration file path (default: validation_config.yaml)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["console", "json", "yaml"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (auto-generated if not specified)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    parser.add_argument(
        "--fail-on-gates",
        action="store_true",
        help="Exit with error code if quality gates fail",
    )

    parser.add_argument(
        "--ci", action="store_true", help="CI-friendly output (implies --quiet)"
    )

    parser.add_argument(
        "--baseline", "-b", type=Path, help="Baseline report file to compare against"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full validation (legacy compatibility flag)",
    )

    return parser


def main():
    """Main entry point for the validation runner."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # CI mode implies quiet
    if args.ci:
        args.quiet = True

    try:
        # Initialize runner
        runner = ValidationRunner(config_file=args.config)

        # Run validation
        report = runner.run_validation(
            output_format=args.format,
            output_file=args.output,
            verbose=args.verbose,
            quiet=args.quiet,
            baseline=args.baseline,
        )

        # Determine exit code
        if args.fail_on_gates or args.ci:
            quality_gates_passed = report.get("quality_gates_passed", 0)
            quality_gates_total = report.get("quality_gates_total", 0)

            if quality_gates_passed < quality_gates_total:
                if not args.quiet:
                    print(
                        f"\n❌ Quality gates failed: {quality_gates_passed}/{quality_gates_total} passed"
                    )
                sys.exit(1)
            else:
                if not args.quiet:
                    print(
                        f"\n✅ All quality gates passed: {quality_gates_passed}/{quality_gates_total}"
                    )
                sys.exit(0)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
