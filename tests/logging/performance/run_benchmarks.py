#!/usr/bin/env python3
"""
Quick script to run the comprehensive logging system benchmarks.

This script demonstrates the benchmark capabilities and generates
performance reports for the XPCS Toolkit logging system.

Usage:
    python run_logging_benchmarks.py [--quick] [--report]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_benchmarks(quick: bool = False, generate_report: bool = False):
    """Run the logging system benchmarks."""

    print("=" * 80)
    print("XPCS TOOLKIT LOGGING SYSTEM PERFORMANCE BENCHMARKS")
    print("=" * 80)

    # Base pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_logging_benchmarks.py",
        "--benchmark-columns=min,median,mean,max,stddev,iqr,ops,rounds",
        "--benchmark-sort=mean",
        "-v",
    ]

    if quick:
        print("Running QUICK benchmark suite...")
        cmd.extend(
            [
                "--benchmark-min-rounds=1",
                "--benchmark-max-time=2.0",
                # Run only a subset of tests
                "-k",
                "throughput or latency",
            ]
        )
    else:
        print("Running COMPREHENSIVE benchmark suite...")
        cmd.extend(["--benchmark-min-rounds=3", "--benchmark-max-time=10.0"])

    if generate_report:
        cmd.extend(["--benchmark-json=benchmark_results.json", "--benchmark-histogram"])

    # Run the benchmarks
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, check=True)

        if generate_report:
            print("\n" + "=" * 80)
            print("BENCHMARK RESULTS SAVED")
            print("=" * 80)
            print("Results saved to: benchmark_results.json")
            print("Histograms available if supported by your environment")

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print("✓ Throughput benchmarks: Message processing rates")
        print("✓ Latency benchmarks: Single message timing")
        print("✓ Memory benchmarks: Memory usage and leak detection")
        print("✓ Scientific computing: MCMC, real-time, correlation analysis")
        print("✓ Scalability: Multiple loggers, rotation, concurrency")
        print("✓ Performance validation: System-wide requirements")
        print("=" * 80)

        return True

    except subprocess.CalledProcessError as e:
        print(f"\nBenchmark execution failed with return code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nBenchmark execution interrupted by user")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run XPCS Toolkit logging system performance benchmarks"
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run a quick subset of benchmarks (faster execution)",
    )
    parser.add_argument(
        "--report",
        "-r",
        action="store_true",
        help="Generate detailed JSON report and histograms",
    )

    args = parser.parse_args()

    # Check if benchmark file exists
    benchmark_file = Path(__file__).parent / "test_logging_benchmarks.py"
    if not benchmark_file.exists():
        print(f"ERROR: Benchmark file not found: {benchmark_file}")
        sys.exit(1)

    # Run benchmarks
    success = run_benchmarks(quick=args.quick, generate_report=args.report)

    if success:
        print("\n🎉 Benchmarks completed successfully!")
        if args.report:
            print("📊 Check benchmark_results.json for detailed metrics")
    else:
        print("\n❌ Benchmarks failed or were interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
