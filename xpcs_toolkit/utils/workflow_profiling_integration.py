"""
Integration example for the XPCS Toolkit workflow profiling system.

This module demonstrates how to integrate the new CPU workflow profiling and
bottleneck identification systems with existing XPCS Toolkit components.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from .cpu_bottleneck_analyzer import analyze_recent_workflows
from .logging_config import get_logger
from .usage_pattern_miner import get_optimization_recommendations, usage_pattern_miner
from .workflow_optimization_report import (
    ReportFormat,
    export_optimization_report,
    generate_optimization_report,
    optimization_report_generator,
)
from .workflow_profiler import (
    profile_workflow,
    profile_workflow_function,
    profile_workflow_step,
    workflow_profiler,
)

logger = get_logger(__name__)


class WorkflowProfilingIntegration:
    """
    Integration layer for workflow profiling with existing XPCS components.

    This class provides easy integration points for existing XPCS Toolkit
    components to add workflow profiling capabilities.
    """

    def __init__(self):
        self.enabled = True
        self.current_session_id: Optional[str] = None
        self.auto_analysis_threshold = 10  # Auto-analyze after 10 workflows
        self.workflow_count = 0

        logger.info("WorkflowProfilingIntegration initialized")

    def enable_profiling(self):
        """Enable workflow profiling."""
        self.enabled = True
        logger.info("Workflow profiling enabled")

    def disable_profiling(self):
        """Disable workflow profiling."""
        self.enabled = False
        logger.info("Workflow profiling disabled")

    @contextmanager
    def profile_xpcs_workflow(self, workflow_type: str, **workflow_params):
        """
        Context manager for profiling XPCS workflows.

        Example usage:
            with profiling_integration.profile_xpcs_workflow('file_loading', file_path='data.h5'):
                # Your workflow code here
                xpcs_file = XpcsFile(file_path)
                data = xpcs_file.load_data()
        """
        if not self.enabled:
            yield None
            return

        with profile_workflow(workflow_type, **workflow_params) as session_id:
            self.current_session_id = session_id
            self.workflow_count += 1

            try:
                yield session_id
            finally:
                self.current_session_id = None

                # Auto-analyze if we've hit the threshold
                if self.workflow_count >= self.auto_analysis_threshold:
                    self._auto_analyze_workflows()
                    self.workflow_count = 0

    @contextmanager
    def profile_xpcs_step(self, step_name: str, **step_params):
        """
        Context manager for profiling individual steps within workflows.

        Example usage:
            with profiling_integration.profile_xpcs_step('g2_fitting', q_range=(0.1, 1.0)):
                # Your step code here
                result = fit_g2_data(data, q_range)
        """
        if not self.enabled or not self.current_session_id:
            yield
            return

        with profile_workflow_step(self.current_session_id, step_name, **step_params):
            yield

    def profile_xpcs_function(self, func_name: Optional[str] = None):
        """
        Decorator for profiling individual functions within workflows.

        Example usage:
            @profiling_integration.profile_xpcs_function('load_hdf5_data')
            def load_data(file_path):
                # Your function code here
                return data
        """
        if not self.enabled or not self.current_session_id:

            def passthrough_decorator(func):
                return func

            return passthrough_decorator

        return profile_workflow_function(self.current_session_id, func_name)

    def add_workflow_annotation(self, annotation: str, data: Any = None):
        """Add annotation to current workflow step."""
        if self.enabled and self.current_session_id:
            workflow_profiler.add_step_annotation(
                self.current_session_id, annotation, data
            )

    def _auto_analyze_workflows(self):
        """Automatically analyze recent workflows and log insights."""
        try:
            logger.info("Running automatic workflow analysis...")

            # Analyze recent bottlenecks
            bottlenecks = analyze_recent_workflows(count=self.auto_analysis_threshold)
            if bottlenecks:
                critical_bottlenecks = [
                    b for b in bottlenecks if b.severity.value == "critical"
                ]
                if critical_bottlenecks:
                    logger.warning(
                        f"Found {len(critical_bottlenecks)} critical performance bottlenecks"
                    )
                    for bottleneck in critical_bottlenecks:
                        logger.warning(
                            f"Critical bottleneck in {bottleneck.component}: {bottleneck.description}"
                        )

            # Get optimization recommendations
            recommendations = get_optimization_recommendations()
            cache_opts = recommendations.get("cache_optimizations", [])
            preload_opts = recommendations.get("preloading_recommendations", [])

            if cache_opts or preload_opts:
                logger.info(
                    f"Identified {len(cache_opts)} cache optimizations and "
                    f"{len(preload_opts)} preloading opportunities"
                )

        except Exception as e:
            logger.error(f"Error during automatic workflow analysis: {e}")

    def generate_performance_report(
        self,
        output_file: Optional[str] = None,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> Dict[str, Any]:
        """
        Generate and optionally export a comprehensive performance report.

        Args:
            output_file: Optional file path to export report
            format: Report format (JSON, HTML, MARKDOWN, TEXT)

        Returns:
            Report summary dictionary
        """
        try:
            logger.info("Generating comprehensive performance report...")

            # Generate the report
            report = generate_optimization_report(workflow_count=50)

            # Export if requested
            if output_file:
                export_optimization_report(report, output_file, format)
                logger.info(f"Performance report exported to {output_file}")

            # Return summary
            return optimization_report_generator.get_report_summary(report.report_id)

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}

    def get_current_bottlenecks(self, count: int = 5) -> Dict[str, Any]:
        """Get current performance bottlenecks."""
        try:
            bottlenecks = analyze_recent_workflows(count=20)

            # Categorize by severity
            critical = [b for b in bottlenecks if b.severity.value == "critical"]
            high = [b for b in bottlenecks if b.severity.value == "high"]

            return {
                "total_bottlenecks": len(bottlenecks),
                "critical_issues": len(critical),
                "high_priority_issues": len(high),
                "top_bottlenecks": [
                    {
                        "component": b.component,
                        "type": b.bottleneck_type.value,
                        "severity": b.severity.value,
                        "description": b.description,
                        "frequency": b.frequency,
                    }
                    for b in bottlenecks[:count]
                ],
            }

        except Exception as e:
            logger.error(f"Error analyzing current bottlenecks: {e}")
            return {"error": str(e)}

    def get_usage_insights(self) -> Dict[str, Any]:
        """Get insights about current usage patterns."""
        try:
            from .usage_pattern_miner import analyze_current_usage_patterns

            patterns = analyze_current_usage_patterns()
            return usage_pattern_miner.get_pattern_summary(patterns)

        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {e}")
            return {"error": str(e)}


# Global integration instance
profiling_integration = WorkflowProfilingIntegration()


# Integration functions for existing XPCS components


def integrate_with_xpcs_file():
    """
    Example integration with XpcsFile class.

    This function shows how to add profiling to the XpcsFile class
    without modifying the original implementation.
    """

    # This would typically be done through monkey patching or inheritance
    # Here's a conceptual example:

    original_init = None  # Would reference XpcsFile.__init__
    original_load_data = None  # Would reference XpcsFile.load_data

    def profiled_init(self, *args, **kwargs):
        """Profiled version of XpcsFile.__init__"""
        with profiling_integration.profile_xpcs_workflow(
            "file_initialization", file_path=args[0] if args else None
        ):
            return original_init(self, *args, **kwargs)

    def profiled_load_data(self, *args, **kwargs):
        """Profiled version of XpcsFile.load_data"""
        with profiling_integration.profile_xpcs_step("data_loading"):
            return original_load_data(self, *args, **kwargs)

    logger.info("XpcsFile profiling integration example prepared")


def integrate_with_viewer_kernel():
    """
    Example integration with ViewerKernel class.

    This function shows how to add profiling to workflow operations
    in the ViewerKernel.
    """

    # Example profiling for workflow operations
    def profiled_workflow_operation(operation_name: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                with profiling_integration.profile_xpcs_workflow(operation_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    logger.info("ViewerKernel profiling integration example prepared")


def integrate_with_analysis_modules():
    """
    Example integration with analysis modules (g2mod, saxs1d, etc.).

    This function shows how to add profiling to scientific computation modules.
    """

    def profiled_analysis_function(module_name: str, function_name: str):
        """Decorator for profiling analysis functions."""

        def decorator(func):
            @profiling_integration.profile_xpcs_function(
                f"{module_name}_{function_name}"
            )
            def wrapper(*args, **kwargs):
                # Add analysis-specific annotations
                profiling_integration.add_workflow_annotation(
                    f"Starting {module_name} {function_name}",
                    {"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                )

                result = func(*args, **kwargs)

                profiling_integration.add_workflow_annotation(
                    f"Completed {module_name} {function_name}",
                    {"result_type": type(result).__name__},
                )

                return result

            return wrapper

        return decorator

    logger.info("Analysis modules profiling integration example prepared")


# Example usage and demonstration functions


def demonstrate_workflow_profiling():
    """Demonstrate the workflow profiling system with example workflows."""

    logger.info("Starting workflow profiling demonstration...")

    # Example 1: File loading workflow
    with profiling_integration.profile_xpcs_workflow(
        "demo_file_loading", file_path="demo.h5"
    ):
        with profiling_integration.profile_xpcs_step("file_validation"):
            time.sleep(0.1)  # Simulate file validation

        with profiling_integration.profile_xpcs_step("metadata_extraction"):
            time.sleep(0.2)  # Simulate metadata extraction

        with profiling_integration.profile_xpcs_step("data_loading"):
            time.sleep(0.5)  # Simulate data loading
            profiling_integration.add_workflow_annotation("Loaded 1000x1000 array")

    # Example 2: Analysis workflow
    with profiling_integration.profile_xpcs_workflow(
        "demo_g2_analysis", q_range=(0.1, 1.0)
    ):
        with profiling_integration.profile_xpcs_step("g2_computation"):
            time.sleep(0.8)  # Simulate G2 computation

        with profiling_integration.profile_xpcs_step("g2_fitting"):
            time.sleep(0.3)  # Simulate fitting
            profiling_integration.add_workflow_annotation(
                "Fit converged", {"chi2": 1.234}
            )

    # Example 3: Visualization workflow
    with profiling_integration.profile_xpcs_workflow("demo_plotting"):
        with profiling_integration.profile_xpcs_step("data_preparation"):
            time.sleep(0.1)

        with profiling_integration.profile_xpcs_step("plot_generation"):
            time.sleep(0.4)
            profiling_integration.add_workflow_annotation("Generated 2D plot")

    logger.info("Workflow profiling demonstration completed")

    # Generate analysis
    time.sleep(1)  # Wait a moment for data to be processed
    bottlenecks = profiling_integration.get_current_bottlenecks()
    insights = profiling_integration.get_usage_insights()

    logger.info(
        f"Analysis results: {bottlenecks.get('total_bottlenecks', 0)} bottlenecks identified"
    )
    logger.info(
        f"Usage insights: {insights.get('optimization_opportunities', 0)} optimization opportunities"
    )


def demonstrate_performance_reporting():
    """Demonstrate the performance reporting system."""

    logger.info("Starting performance reporting demonstration...")

    # Run some demo workflows first
    demonstrate_workflow_profiling()

    # Generate performance report using secure temp directory
    import tempfile
    import os

    # Use system temp directory securely
    temp_dir = tempfile.gettempdir()
    output_file = os.path.join(temp_dir, "xpcs_performance_report.md")

    report_summary = profiling_integration.generate_performance_report(
        output_file=output_file, format=ReportFormat.MARKDOWN
    )

    logger.info("Performance report generated:")
    logger.info(
        f"  - Analyzed workflows: {report_summary.get('analyzed_workflows', 0)}"
    )
    logger.info(
        f"  - Total recommendations: {report_summary.get('total_recommendations', 0)}"
    )
    logger.info(
        f"  - Estimated improvement: {report_summary.get('estimated_total_improvement', 0):.1f}%"
    )

    # Show top recommendations
    top_recs = report_summary.get("top_recommendations", [])
    if top_recs:
        logger.info("Top optimization recommendations:")
        for i, rec in enumerate(top_recs, 1):
            logger.info(
                f"  {i}. {rec['title']} ({rec['improvement_percentage']:.1f}% improvement)"
            )


if __name__ == "__main__":
    """
    Run workflow profiling demonstrations.
    """
    logger.info("Starting XPCS Toolkit Workflow Profiling Integration Demonstration")

    # Enable profiling
    profiling_integration.enable_profiling()

    try:
        # Run demonstrations
        demonstrate_workflow_profiling()
        print("\n" + "=" * 60 + "\n")

        demonstrate_performance_reporting()
        print("\n" + "=" * 60 + "\n")

        logger.info("All workflow profiling demonstrations completed successfully")

    except Exception as e:
        logger.error(f"Error during demonstration: {e}")

    finally:
        # Clean up
        profiling_integration.disable_profiling()
        logger.info("Workflow profiling demonstrations finished")
