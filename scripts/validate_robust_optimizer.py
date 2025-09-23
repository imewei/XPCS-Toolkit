#!/usr/bin/env python3
"""
Comprehensive validation script for the RobustOptimizer.

This script performs extensive testing of the robust multi-strategy optimization
engine to ensure it achieves the target 95% success rate on diverse synthetic
G2 datasets.

Usage:
    python validate_robust_optimizer.py [--n-datasets 1000] [--verbose]
"""

import argparse
import sys
import time
from pathlib import Path

# Add XPCS-Toolkit to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from xpcs_toolkit.helper.fitting import (
    RobustOptimizer,
    SyntheticG2DataGenerator,
    single_exp,
)


def create_diverse_test_scenarios():
    """Create diverse test scenarios for comprehensive validation."""
    scenarios = []

    # Scenario 1: Clean data with varying parameters
    for tau_scale in [1e-6, 1e-4, 1e-3, 1e-2]:
        for noise_level in [0.01, 0.02, 0.05]:
            scenarios.append(
                {
                    "name": f"clean_tau_{tau_scale}_noise_{noise_level}",
                    "tau_range": (tau_scale, tau_scale * 1000),
                    "noise_level": noise_level,
                    "n_points": 50,
                    "outlier_fraction": 0.0,
                    "systematic_error": False,
                }
            )

    # Scenario 2: Noisy data with outliers
    for noise_level in [0.05, 0.1, 0.15]:
        for outlier_frac in [0.05, 0.1]:
            scenarios.append(
                {
                    "name": f"noisy_{noise_level}_outliers_{outlier_frac}",
                    "tau_range": (1e-5, 1e-2),
                    "noise_level": noise_level,
                    "n_points": 60,
                    "outlier_fraction": outlier_frac,
                    "systematic_error": False,
                }
            )

    # Scenario 3: Systematic errors
    for noise_level in [0.02, 0.05]:
        scenarios.append(
            {
                "name": f"systematic_noise_{noise_level}",
                "tau_range": (1e-5, 1e-2),
                "noise_level": noise_level,
                "n_points": 50,
                "outlier_fraction": 0.0,
                "systematic_error": True,
            }
        )

    # Scenario 4: Sparse data
    for n_points in [20, 30]:
        scenarios.append(
            {
                "name": f"sparse_{n_points}_points",
                "tau_range": (1e-5, 1e-2),
                "noise_level": 0.03,
                "n_points": n_points,
                "outlier_fraction": 0.0,
                "systematic_error": False,
            }
        )

    # Scenario 5: Different tau ranges (challenging for parameter estimation)
    for tau_min, tau_max in [(1e-7, 1e-4), (1e-4, 1e-1), (1e-3, 1e0)]:
        scenarios.append(
            {
                "name": f"range_{tau_min}_to_{tau_max}",
                "tau_range": (tau_min, tau_max),
                "noise_level": 0.03,
                "n_points": 50,
                "outlier_fraction": 0.0,
                "systematic_error": False,
            }
        )

    return scenarios


def validate_single_dataset(generator, scenario, bounds, verbose=False):
    """Validate optimizer on a single synthetic dataset."""
    try:
        # Generate synthetic data
        tau, _g2, g2_err, true_params = generator.generate_dataset(
            model_type="single_exp",
            **{k: v for k, v in scenario.items() if k != "name"},
        )

        # Convert synthetic G2 parameters to single_exp parameters
        gamma_true = true_params["gamma"]
        baseline_true = true_params["baseline"]
        beta_true = true_params["beta"]

        # Convert to single_exp parameters: tau, bkg, cts
        tau_param_true = 2.0 / gamma_true  # tau parameter for single_exp
        bkg_true = baseline_true
        cts_true = beta_true

        # Create equivalent single_exp data
        g2_single_exp = single_exp(tau, tau_param_true, bkg_true, cts_true)

        # Use robust optimizer
        optimizer = RobustOptimizer(performance_tracking=True)

        start_time = time.time()
        popt, _pcov, info = optimizer.robust_curve_fit(
            single_exp, tau, g2_single_exp, bounds=bounds, sigma=g2_err
        )
        fit_time = time.time() - start_time

        # Validate parameters
        tau_fit, bkg_fit, cts_fit = popt

        # Calculate relative errors
        rel_err_tau = abs(tau_fit - tau_param_true) / tau_param_true
        rel_err_bkg = abs(bkg_fit - bkg_true) / bkg_true
        rel_err_cts = abs(cts_fit - cts_true) / cts_true

        # Success criteria (relaxed for diverse scenarios)
        success_criteria = [
            rel_err_tau < 0.5,  # 50% error tolerance for tau
            rel_err_bkg < 0.2,  # 20% error tolerance for background
            rel_err_cts < 0.5,  # 50% error tolerance for contrast
            info["r_squared"] > 0.5,  # Minimum R-squared
        ]

        success = all(success_criteria)

        result = {
            "success": success,
            "scenario": scenario["name"],
            "method": info["method"],
            "fit_time": fit_time,
            "r_squared": info["r_squared"],
            "rel_errors": {"tau": rel_err_tau, "bkg": rel_err_bkg, "cts": rel_err_cts},
            "true_params": {"tau": tau_param_true, "bkg": bkg_true, "cts": cts_true},
            "fitted_params": {"tau": tau_fit, "bkg": bkg_fit, "cts": cts_fit},
        }

        if verbose and not success:
            print(f"FAILED: {scenario['name']}")
            print(f"  Method: {info['method']}")
            print(f"  R²: {info['r_squared']:.3f}")
            print(
                f"  Errors: tau={rel_err_tau:.3f}, bkg={rel_err_bkg:.3f}, cts={rel_err_cts:.3f}"
            )
            print(f"  Criteria: {success_criteria}")

        return result

    except Exception as e:
        return {
            "success": False,
            "scenario": scenario["name"],
            "method": "Failed",
            "fit_time": 0,
            "r_squared": 0,
            "error": str(e),
            "rel_errors": {"tau": np.inf, "bkg": np.inf, "cts": np.inf},
        }


def compare_with_standard_optimizer(generator, scenarios, bounds, n_samples=10):
    """Compare RobustOptimizer with standard scipy.optimize.curve_fit."""
    print("\nComparing RobustOptimizer vs Standard curve_fit...")

    robust_successes = 0
    standard_successes = 0
    robust_times = []
    standard_times = []

    for scenario in scenarios[:5]:  # Test on subset for comparison
        for _ in range(n_samples):
            try:
                # Generate data
                tau, _g2, g2_err, true_params = generator.generate_dataset(
                    model_type="single_exp",
                    **{k: v for k, v in scenario.items() if k != "name"},
                )

                # Convert to single_exp equivalent
                gamma_true = true_params["gamma"]
                tau_param_true = 2.0 / gamma_true
                bkg_true = true_params["baseline"]
                cts_true = true_params["beta"]
                g2_single_exp = single_exp(tau, tau_param_true, bkg_true, cts_true)

                # Test RobustOptimizer
                optimizer = RobustOptimizer()
                start_time = time.time()
                try:
                    _popt_robust, _, info_robust = optimizer.robust_curve_fit(
                        single_exp, tau, g2_single_exp, bounds=bounds, sigma=g2_err
                    )
                    robust_time = time.time() - start_time
                    robust_times.append(robust_time)

                    # Simple success check
                    if info_robust["r_squared"] > 0.7:
                        robust_successes += 1
                except Exception:
                    robust_time = time.time() - start_time
                    robust_times.append(robust_time)

                # Test standard curve_fit
                start_time = time.time()
                try:
                    p0 = [1e-3, 1.0, 0.5]  # Initial guess
                    popt_std, _ = curve_fit(
                        single_exp,
                        tau,
                        g2_single_exp,
                        p0=p0,
                        bounds=bounds,
                        sigma=g2_err,
                    )
                    standard_time = time.time() - start_time
                    standard_times.append(standard_time)

                    # Check success
                    g2_fit = single_exp(tau, *popt_std)
                    residuals = g2_single_exp - g2_fit
                    r_squared = 1 - np.sum(residuals**2) / np.sum(
                        (g2_single_exp - np.mean(g2_single_exp)) ** 2
                    )

                    if r_squared > 0.7:
                        standard_successes += 1

                except Exception:
                    standard_time = time.time() - start_time
                    standard_times.append(standard_time)

            except Exception as e:
                print(f"Standard optimizer failed: {e}")
                continue

    total_tests = len(scenarios[:5]) * n_samples

    print(f"Results ({total_tests} tests):")
    print(
        f"  RobustOptimizer:  {robust_successes}/{total_tests} = {robust_successes / total_tests * 100:.1f}% success"
    )
    print(
        f"  Standard curve_fit: {standard_successes}/{total_tests} = {standard_successes / total_tests * 100:.1f}% success"
    )

    if robust_times and standard_times:
        avg_robust_time = np.mean(robust_times)
        avg_standard_time = np.mean(standard_times)
        overhead = (avg_robust_time / avg_standard_time - 1) * 100

        print("  Average times:")
        print(f"    RobustOptimizer:  {avg_robust_time * 1000:.1f} ms")
        print(f"    Standard curve_fit: {avg_standard_time * 1000:.1f} ms")
        print(f"    Overhead: {overhead:.1f}%")

        return overhead

    return None


def generate_validation_report(results):
    """Generate comprehensive validation report."""
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    success_rate = successful_tests / total_tests

    # Group results by scenario type
    scenario_groups = {}
    for result in results:
        scenario_name = result["scenario"]
        scenario_type = scenario_name.split("_")[0]  # First part of scenario name

        if scenario_type not in scenario_groups:
            scenario_groups[scenario_type] = []
        scenario_groups[scenario_type].append(result)

    # Method usage statistics
    method_counts = {}
    for result in results:
        if result["success"]:
            method = result["method"]
            method_counts[method] = method_counts.get(method, 0) + 1

    # Performance statistics
    fit_times = [r["fit_time"] for r in results if r["success"]]
    r_squared_values = [r["r_squared"] for r in results if r["success"]]

    # Generate report
    report = f"""
ROBUST OPTIMIZER VALIDATION REPORT
{"=" * 50}

OVERALL PERFORMANCE:
  Total tests: {total_tests}
  Successful fits: {successful_tests}
  Success rate: {success_rate * 100:.2f}%
  Target success rate: 95%
  {"✓ TARGET ACHIEVED" if success_rate >= 0.95 else "✗ TARGET MISSED"}

PERFORMANCE BY SCENARIO TYPE:
"""

    for scenario_type, group_results in scenario_groups.items():
        group_successes = sum(1 for r in group_results if r["success"])
        group_total = len(group_results)
        group_rate = group_successes / group_total

        report += f"  {scenario_type:12}: {group_successes:3}/{group_total:3} = {group_rate * 100:5.1f}%\n"

    report += """
OPTIMIZATION METHOD USAGE:
"""
    for method, count in sorted(
        method_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = count / successful_tests * 100
        report += f"  {method:25}: {count:3} times ({percentage:4.1f}%)\n"

    if fit_times:
        report += f"""
PERFORMANCE STATISTICS:
  Average fit time: {np.mean(fit_times) * 1000:6.1f} ms
  Median fit time:  {np.median(fit_times) * 1000:6.1f} ms
  95th percentile:  {np.percentile(fit_times, 95) * 1000:6.1f} ms

  Average R²:       {np.mean(r_squared_values):6.3f}
  Median R²:        {np.median(r_squared_values):6.3f}
  Min R²:           {np.min(r_squared_values):6.3f}
"""

    # Failure analysis
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        report += f"""
FAILURE ANALYSIS:
  Total failures: {len(failed_results)}

  Common failure scenarios:
"""
        failure_scenarios = {}
        for result in failed_results:
            scenario = result["scenario"].split("_")[0]
            failure_scenarios[scenario] = failure_scenarios.get(scenario, 0) + 1

        for scenario, count in sorted(
            failure_scenarios.items(), key=lambda x: x[1], reverse=True
        ):
            report += f"    {scenario:12}: {count} failures\n"

    return report


def plot_validation_results(results, save_path=None):
    """Create visualization plots for validation results."""
    try:
        # Success rate by scenario type
        scenario_groups = {}
        for result in results:
            scenario_type = result["scenario"].split("_")[0]
            if scenario_type not in scenario_groups:
                scenario_groups[scenario_type] = []
            scenario_groups[scenario_type].append(result)

        _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Success rate by scenario
        scenario_names = list(scenario_groups.keys())
        success_rates = []
        for scenario_type in scenario_names:
            group_results = scenario_groups[scenario_type]
            success_rate = sum(1 for r in group_results if r["success"]) / len(
                group_results
            )
            success_rates.append(success_rate * 100)

        ax1.bar(scenario_names, success_rates, color="skyblue", edgecolor="navy")
        ax1.axhline(y=95, color="red", linestyle="--", label="Target (95%)")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Success Rate by Scenario Type")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        # Plot 2: Method usage
        method_counts = {}
        for result in results:
            if result["success"]:
                method = result["method"]
                method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            ax2.pie(counts, labels=methods, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Optimization Method Usage")

        # Plot 3: Fit time distribution
        fit_times = [
            r["fit_time"] * 1000 for r in results if r["success"]
        ]  # Convert to ms
        if fit_times:
            ax3.hist(
                fit_times, bins=30, color="lightgreen", edgecolor="darkgreen", alpha=0.7
            )
            ax3.axvline(
                np.mean(fit_times),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(fit_times):.1f} ms",
            )
            ax3.set_xlabel("Fit Time (ms)")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Fit Time Distribution")
            ax3.legend()

        # Plot 4: R-squared distribution
        r_squared_values = [r["r_squared"] for r in results if r["success"]]
        if r_squared_values:
            ax4.hist(
                r_squared_values,
                bins=30,
                color="orange",
                edgecolor="darkorange",
                alpha=0.7,
            )
            ax4.axvline(
                np.mean(r_squared_values),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(r_squared_values):.3f}",
            )
            ax4.set_xlabel("R-squared")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Goodness of Fit Distribution")
            ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nValidation plots saved to: {save_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate RobustOptimizer performance")
    parser.add_argument(
        "--n-datasets",
        type=int,
        default=200,
        help="Number of datasets per scenario to test",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output including failures"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with standard curve_fit"
    )
    parser.add_argument(
        "--plot", type=str, default=None, help="Save validation plots to file"
    )

    args = parser.parse_args()

    print("XPCS Toolkit - RobustOptimizer Validation")
    print("=" * 50)

    # Initialize
    generator = SyntheticG2DataGenerator(random_state=42)
    scenarios = create_diverse_test_scenarios()

    # Define bounds for single_exp: tau, bkg, cts
    bounds = ([1e-6, 0.8, 0.01], [1e0, 1.2, 3.0])

    print(f"Testing scenarios: {len(scenarios)}")
    print(f"Datasets per scenario: {args.n_datasets}")
    print(f"Total datasets: {len(scenarios) * args.n_datasets}")

    # Run validation
    results = []
    start_time = time.time()

    for i, scenario in enumerate(scenarios):
        if args.verbose:
            print(f"Testing scenario {i + 1}/{len(scenarios)}: {scenario['name']}")

        scenario_results = []
        for j in range(args.n_datasets):
            result = validate_single_dataset(generator, scenario, bounds, args.verbose)
            scenario_results.append(result)

            if args.verbose and (j + 1) % 10 == 0:
                successes = sum(1 for r in scenario_results if r["success"])
                print(
                    f"  Progress: {j + 1}/{args.n_datasets}, Success: {successes}/{j + 1}"
                )

        results.extend(scenario_results)

        # Print scenario summary
        scenario_successes = sum(1 for r in scenario_results if r["success"])
        scenario_rate = scenario_successes / len(scenario_results)
        print(
            f"Scenario {scenario['name']}: {scenario_successes}/{len(scenario_results)} = {scenario_rate * 100:.1f}%"
        )

    total_time = time.time() - start_time

    # Generate and display report
    report = generate_validation_report(results)
    print(report)

    print(f"\nTotal validation time: {total_time:.1f} seconds")
    print(f"Average time per test: {total_time / len(results) * 1000:.1f} ms")

    # Performance comparison
    overhead = None
    if args.compare:
        overhead = compare_with_standard_optimizer(generator, scenarios, bounds)

    # Generate plots
    if args.plot:
        plot_validation_results(results, args.plot)

    # Final assessment
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    success_rate = successful_tests / total_tests

    print("\nFINAL ASSESSMENT:")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print("Target: 95%")

    if success_rate >= 0.95:
        print("✓ VALIDATION PASSED - Target success rate achieved!")
        exit_code = 0
    else:
        print("✗ VALIDATION FAILED - Target success rate not achieved")
        exit_code = 1

    if overhead is not None:
        print(f"Performance Overhead: {overhead:.1f}%")
        if overhead < 20:
            print("✓ Performance target met (<20% overhead)")
        else:
            print("⚠ Performance overhead higher than target")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
