"""
Workflow optimization report generator for XPCS Toolkit.

This module generates comprehensive reports with CPU-focused optimization recommendations,
performance improvement estimates, and integration with existing optimization systems.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .cpu_bottleneck_analyzer import (
    BottleneckFinding,
    BottleneckSeverity,
    cpu_bottleneck_analyzer,
)
from .logging_config import get_logger
from .usage_pattern_miner import (
    CacheOptimization,
    PreloadingRecommendation,
    ThreadPoolRecommendation,
    usage_pattern_miner,
)
from .workflow_profiler import workflow_profiler

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


class OptimizationImpact(Enum):
    """Impact levels for optimization recommendations."""

    LOW = "low"  # <10% improvement
    MEDIUM = "medium"  # 10-25% improvement
    HIGH = "high"  # 25-50% improvement
    CRITICAL = "critical"  # >50% improvement


@dataclass
class PerformanceImprovement:
    """Estimated performance improvement from an optimization."""

    category: str
    improvement_percentage: float
    time_savings_seconds: float
    memory_savings_mb: float
    confidence: float
    implementation_effort: str  # "low", "medium", "high"

    @property
    def impact_level(self) -> OptimizationImpact:
        """Determine impact level based on improvement percentage."""
        if self.improvement_percentage >= 50:
            return OptimizationImpact.CRITICAL
        elif self.improvement_percentage >= 25:
            return OptimizationImpact.HIGH
        elif self.improvement_percentage >= 10:
            return OptimizationImpact.MEDIUM
        else:
            return OptimizationImpact.LOW


@dataclass
class OptimizationRecommendation:
    """Comprehensive optimization recommendation."""

    title: str
    category: str
    description: str
    implementation_steps: List[str]
    estimated_improvement: PerformanceImprovement
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    testing_requirements: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    priority_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "implementation_steps": self.implementation_steps,
            "estimated_improvement": asdict(self.estimated_improvement),
            "prerequisites": self.prerequisites,
            "risks": self.risks,
            "testing_requirements": self.testing_requirements,
            "affected_components": self.affected_components,
            "priority_score": self.priority_score,
        }


@dataclass
class WorkflowOptimizationReport:
    """Complete workflow optimization report."""

    report_id: str
    generation_timestamp: float
    analysis_period: Tuple[float, float]  # (start_time, end_time)
    analyzed_workflows: int

    # Analysis results
    bottleneck_findings: List[BottleneckFinding] = field(default_factory=list)
    optimization_recommendations: List[OptimizationRecommendation] = field(
        default_factory=list
    )
    usage_patterns: Dict[str, Any] = field(default_factory=dict)
    performance_baseline: Dict[str, Any] = field(default_factory=dict)

    # Summary statistics
    critical_issues: int = 0
    high_priority_optimizations: int = 0
    estimated_total_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "generation_timestamp": self.generation_timestamp,
            "analysis_period": self.analysis_period,
            "analyzed_workflows": self.analyzed_workflows,
            "bottleneck_findings": [f.to_dict() for f in self.bottleneck_findings],
            "optimization_recommendations": [
                r.to_dict() for r in self.optimization_recommendations
            ],
            "usage_patterns": self.usage_patterns,
            "performance_baseline": self.performance_baseline,
            "critical_issues": self.critical_issues,
            "high_priority_optimizations": self.high_priority_optimizations,
            "estimated_total_improvement": self.estimated_total_improvement,
        }


class WorkflowOptimizationReportGenerator:
    """
    Generates comprehensive workflow optimization reports.

    This class integrates bottleneck analysis, usage pattern mining, and existing
    optimization systems to produce actionable optimization reports.
    """

    def __init__(self):
        self.report_cache: Dict[str, WorkflowOptimizationReport] = {}
        self.baseline_metrics: Dict[str, Any] = {}

        # Integration with existing optimization systems
        self._load_baseline_metrics()

        logger.info("WorkflowOptimizationReportGenerator initialized")

    def _load_baseline_metrics(self):
        """Load baseline performance metrics."""
        try:
            # Try to load recent performance data from global profiler
            from .performance_profiler import global_profiler

            summary = global_profiler.get_performance_summary()

            self.baseline_metrics = {
                "performance_summary": summary,
                "baseline_timestamp": time.time(),
            }

            logger.info("Loaded baseline performance metrics")
        except Exception as e:
            logger.warning(f"Could not load baseline metrics: {e}")
            self.baseline_metrics = {"baseline_timestamp": time.time()}

    def generate_comprehensive_report(
        self,
        workflow_count: int = 50,
        include_patterns: bool = True,
        include_bottlenecks: bool = True,
        report_format: ReportFormat = ReportFormat.JSON,
    ) -> WorkflowOptimizationReport:
        """
        Generate a comprehensive optimization report.

        Args:
            workflow_count: Number of recent workflows to analyze
            include_patterns: Whether to include usage pattern analysis
            include_bottlenecks: Whether to include bottleneck analysis
            report_format: Output format for the report

        Returns:
            WorkflowOptimizationReport object
        """
        logger.info(
            f"Generating comprehensive optimization report for {workflow_count} workflows"
        )

        # Get recent workflow profiles
        profiles = workflow_profiler.get_recent_profiles(workflow_count)
        if not profiles:
            logger.warning("No workflow profiles available for analysis")
            return self._create_empty_report()

        # Create report
        report_id = f"optimization_report_{int(time.time())}"
        report = WorkflowOptimizationReport(
            report_id=report_id,
            generation_timestamp=time.time(),
            analysis_period=(profiles[-1].start_time, profiles[0].start_time),
            analyzed_workflows=len(profiles),
            performance_baseline=self.baseline_metrics.copy(),
        )

        # Perform bottleneck analysis
        if include_bottlenecks:
            logger.info("Performing bottleneck analysis...")
            report.bottleneck_findings = (
                cpu_bottleneck_analyzer.analyze_workflow_profiles(profiles)
            )
            report.critical_issues = sum(
                1
                for f in report.bottleneck_findings
                if f.severity == BottleneckSeverity.CRITICAL
            )

        # Perform usage pattern analysis
        if include_patterns:
            logger.info("Analyzing usage patterns...")
            report.usage_patterns = usage_pattern_miner.analyze_usage_patterns(profiles)

        # Generate optimization recommendations
        logger.info("Generating optimization recommendations...")
        report.optimization_recommendations = (
            self._generate_optimization_recommendations(
                report.bottleneck_findings, report.usage_patterns
            )
        )

        # Calculate summary statistics
        report.high_priority_optimizations = sum(
            1
            for r in report.optimization_recommendations
            if r.estimated_improvement.impact_level
            in [OptimizationImpact.HIGH, OptimizationImpact.CRITICAL]
        )

        report.estimated_total_improvement = self._calculate_total_improvement(
            report.optimization_recommendations
        )

        # Cache the report
        self.report_cache[report_id] = report

        logger.info(
            f"Generated optimization report {report_id} with "
            f"{len(report.optimization_recommendations)} recommendations"
        )

        return report

    def _create_empty_report(self) -> WorkflowOptimizationReport:
        """Create an empty report when no data is available."""
        return WorkflowOptimizationReport(
            report_id=f"empty_report_{int(time.time())}",
            generation_timestamp=time.time(),
            analysis_period=(time.time(), time.time()),
            analyzed_workflows=0,
        )

    def _generate_optimization_recommendations(
        self, bottlenecks: List[BottleneckFinding], usage_patterns: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []

        # Generate recommendations from bottleneck findings
        recommendations.extend(self._recommendations_from_bottlenecks(bottlenecks))

        # Generate recommendations from usage patterns
        if usage_patterns:
            recommendations.extend(self._recommendations_from_patterns(usage_patterns))

        # Sort recommendations by priority score
        for rec in recommendations:
            rec.priority_score = self._calculate_priority_score(rec)

        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def _recommendations_from_bottlenecks(
        self, bottlenecks: List[BottleneckFinding]
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations from bottleneck findings."""
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck.severity in [
                BottleneckSeverity.HIGH,
                BottleneckSeverity.CRITICAL,
            ]:
                rec = self._create_bottleneck_recommendation(bottleneck)
                if rec:
                    recommendations.append(rec)

        return recommendations

    def _create_bottleneck_recommendation(
        self, bottleneck: BottleneckFinding
    ) -> Optional[OptimizationRecommendation]:
        """Create optimization recommendation from bottleneck finding."""

        # Calculate estimated improvement
        improvement = self._estimate_bottleneck_improvement(bottleneck)

        # Generate implementation steps
        implementation_steps = bottleneck.recommendations.copy()
        implementation_steps.extend(
            [
                f"Profile {bottleneck.component} to identify specific optimization points",
                "Implement optimizations incrementally with benchmarking",
                "Test with representative datasets to ensure correctness",
            ]
        )

        # Determine prerequisites and risks
        prerequisites, risks = self._analyze_bottleneck_requirements(bottleneck)

        # Generate testing requirements
        testing_requirements = [
            f"Benchmark {bottleneck.component} before and after optimization",
            "Run regression tests to ensure functionality is preserved",
            "Test with various dataset sizes to verify scalability",
        ]

        return OptimizationRecommendation(
            title=f"Optimize {bottleneck.component} - {bottleneck.bottleneck_type.value}",
            category=bottleneck.bottleneck_type.value,
            description=f"{bottleneck.description}. "
            f"Affects {len(bottleneck.affected_workflows)} workflows.",
            implementation_steps=implementation_steps,
            estimated_improvement=improvement,
            prerequisites=prerequisites,
            risks=risks,
            testing_requirements=testing_requirements,
            affected_components=[bottleneck.component],
        )

    def _estimate_bottleneck_improvement(
        self, bottleneck: BottleneckFinding
    ) -> PerformanceImprovement:
        """Estimate performance improvement from fixing a bottleneck."""

        # Base improvement estimates by bottleneck type
        type_improvements = {
            "cpu_bound": (20, 40),  # 20-40% improvement
            "thread_contention": (15, 35),
            "memory_allocation": (25, 50),
            "algorithmic_inefficiency": (30, 70),
            "hot_path": (20, 45),
            "synchronization": (10, 30),
            "io_wait": (15, 35),
            "cache_miss": (25, 50),
        }

        min_improvement, max_improvement = type_improvements.get(
            bottleneck.bottleneck_type.value, (10, 25)
        )

        # Adjust based on severity
        severity_multipliers = {
            BottleneckSeverity.LOW: 0.5,
            BottleneckSeverity.MEDIUM: 0.75,
            BottleneckSeverity.HIGH: 1.0,
            BottleneckSeverity.CRITICAL: 1.3,
        }

        multiplier = severity_multipliers[bottleneck.severity]
        estimated_improvement = (min_improvement + max_improvement) / 2 * multiplier

        # Estimate time and memory savings
        time_savings = 0.0
        memory_savings = 0.0

        if "total_time" in bottleneck.metrics:
            time_savings = bottleneck.metrics["total_time"] * (
                estimated_improvement / 100.0
            )

        if "total_growth_mb" in bottleneck.metrics:
            memory_savings = bottleneck.metrics["total_growth_mb"] * (
                estimated_improvement / 100.0
            )

        # Determine implementation effort
        effort_by_type = {
            "cpu_bound": "medium",
            "thread_contention": "high",
            "memory_allocation": "medium",
            "algorithmic_inefficiency": "high",
            "hot_path": "low",
            "synchronization": "high",
            "io_wait": "low",
            "cache_miss": "low",
        }

        implementation_effort = effort_by_type.get(
            bottleneck.bottleneck_type.value, "medium"
        )

        return PerformanceImprovement(
            category=bottleneck.bottleneck_type.value,
            improvement_percentage=estimated_improvement,
            time_savings_seconds=time_savings,
            memory_savings_mb=memory_savings,
            confidence=bottleneck.confidence_score,
            implementation_effort=implementation_effort,
        )

    def _analyze_bottleneck_requirements(
        self, bottleneck: BottleneckFinding
    ) -> Tuple[List[str], List[str]]:
        """Analyze prerequisites and risks for bottleneck optimization."""

        prerequisites = [
            "Establish performance benchmarks for affected components",
            "Backup current implementation for rollback capability",
        ]

        risks = [
            "Optimization may introduce new bugs or edge cases",
            "Performance improvement may vary with different datasets",
        ]

        # Add type-specific requirements
        if bottleneck.bottleneck_type.value == "thread_contention":
            prerequisites.append("Thread-safe testing environment")
            risks.append("Changes to threading may introduce race conditions")

        elif bottleneck.bottleneck_type.value == "algorithmic_inefficiency":
            prerequisites.append("Mathematical verification of algorithm correctness")
            risks.append("Algorithm changes may affect numerical precision")

        elif bottleneck.bottleneck_type.value == "memory_allocation":
            prerequisites.append("Memory profiling tools and monitoring")
            risks.append("Memory optimizations may affect garbage collection")

        return prerequisites, risks

    def _recommendations_from_patterns(
        self, usage_patterns: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations from usage patterns."""
        recommendations = []

        # Cache optimization recommendations
        cache_optimizations = (
            usage_pattern_miner.generate_cache_optimization_recommendations(
                usage_patterns
            )
        )
        for cache_opt in cache_optimizations:
            rec = self._create_cache_recommendation(cache_opt)
            if rec:
                recommendations.append(rec)

        # Preloading recommendations
        preloading_recs = usage_pattern_miner.generate_preloading_recommendations(
            usage_patterns
        )
        for preload_rec in preloading_recs:
            rec = self._create_preloading_recommendation(preload_rec)
            if rec:
                recommendations.append(rec)

        # Thread pool recommendations
        thread_pool_recs = usage_pattern_miner.generate_thread_pool_recommendations(
            usage_patterns
        )
        for thread_rec in thread_pool_recs:
            rec = self._create_thread_pool_recommendation(thread_rec)
            if rec:
                recommendations.append(rec)

        return recommendations

    def _create_cache_recommendation(
        self, cache_opt: CacheOptimization
    ) -> OptimizationRecommendation:
        """Create recommendation from cache optimization."""

        improvement = PerformanceImprovement(
            category="caching",
            improvement_percentage=cache_opt.hit_rate_improvement * 100,
            time_savings_seconds=0.0,  # Would need more data to estimate
            memory_savings_mb=-cache_opt.cache_size_mb,  # Negative because cache uses memory
            confidence=0.8,
            implementation_effort="low" if cache_opt.cache_size_mb < 100 else "medium",
        )

        return OptimizationRecommendation(
            title=f"Implement {cache_opt.resource_type} caching",
            category="caching",
            description=cache_opt.description,
            implementation_steps=cache_opt.implementation_notes
            + [
                f"Allocate {cache_opt.cache_size_mb:.1f}MB for cache",
                "Implement cache key generation strategy",
                "Add cache hit/miss monitoring",
            ],
            estimated_improvement=improvement,
            prerequisites=[
                "Memory availability analysis",
                "Cache eviction strategy design",
            ],
            risks=["Increased memory usage", "Cache coherency issues"],
            testing_requirements=[
                "Test cache hit rates under typical workloads",
                "Verify cache eviction works correctly",
                "Measure memory impact of cache",
            ],
            affected_components=[cache_opt.resource_type],
        )

    def _create_preloading_recommendation(
        self, preload_rec: PreloadingRecommendation
    ) -> OptimizationRecommendation:
        """Create recommendation from preloading analysis."""

        improvement = PerformanceImprovement(
            category="preloading",
            improvement_percentage=preload_rec.predicted_benefit * 100,
            time_savings_seconds=0.0,
            memory_savings_mb=0.0,
            confidence=preload_rec.confidence,
            implementation_effort="medium",
        )

        return OptimizationRecommendation(
            title=f"Implement preloading for {preload_rec.resource_pattern}",
            category="preloading",
            description=preload_rec.description,
            implementation_steps=[
                f"Detect trigger condition: {preload_rec.trigger_condition}",
                "Implement background preloading mechanism",
                "Add preloading success/failure monitoring",
                "Test preloading with typical usage patterns",
            ],
            estimated_improvement=improvement,
            prerequisites=[
                "Background task scheduling system",
                "Resource prediction algorithm",
            ],
            risks=[
                "Unnecessary resource usage if prediction is wrong",
                "Race conditions with user actions",
            ],
            testing_requirements=[
                "Test preloading accuracy with various workflows",
                "Measure resource overhead of preloading",
                "Verify preloading doesn't interfere with user actions",
            ],
            affected_components=["resource_loading", "background_tasks"],
        )

    def _create_thread_pool_recommendation(
        self, thread_rec: ThreadPoolRecommendation
    ) -> OptimizationRecommendation:
        """Create recommendation from thread pool analysis."""

        improvement = PerformanceImprovement(
            category="threading",
            improvement_percentage=thread_rec.utilization_improvement * 100,
            time_savings_seconds=0.0,
            memory_savings_mb=0.0,
            confidence=0.7,
            implementation_effort="low",
        )

        return OptimizationRecommendation(
            title=f"Optimize thread pool for {thread_rec.workflow_type}",
            category="threading",
            description=thread_rec.reasoning,
            implementation_steps=[
                f"Adjust thread pool size to {thread_rec.recommended_pool_size}",
                "Monitor thread utilization after change",
                "Implement dynamic thread pool sizing if beneficial",
                "Add thread contention monitoring",
            ],
            estimated_improvement=improvement,
            prerequisites=[
                "Thread pool configuration access",
                "Thread utilization monitoring",
            ],
            risks=[
                "Thread pool size may not be optimal for all scenarios",
                "Changes may affect other workflows",
            ],
            testing_requirements=[
                f"Test {thread_rec.workflow_type} performance with new pool size",
                "Monitor thread contention metrics",
                "Verify no negative impact on other workflow types",
            ],
            affected_components=[f"thread_pool_{thread_rec.workflow_type}"],
        )

    def _calculate_priority_score(
        self, recommendation: OptimizationRecommendation
    ) -> float:
        """Calculate priority score for a recommendation."""

        # Base score from improvement percentage
        improvement_score = recommendation.estimated_improvement.improvement_percentage

        # Confidence multiplier
        confidence_multiplier = recommendation.estimated_improvement.confidence

        # Implementation effort penalty
        effort_penalties = {"low": 0, "medium": 5, "high": 15}
        effort_penalty = effort_penalties.get(
            recommendation.estimated_improvement.implementation_effort, 5
        )

        # Component impact bonus
        component_bonus = len(recommendation.affected_components) * 2

        # Calculate final score
        priority_score = (
            (improvement_score * confidence_multiplier)
            - effort_penalty
            + component_bonus
        )

        return max(0, priority_score)

    def _calculate_total_improvement(
        self, recommendations: List[OptimizationRecommendation]
    ) -> float:
        """Calculate estimated total improvement from all recommendations."""
        if not recommendations:
            return 0.0

        # Use weighted average based on confidence scores
        total_weighted_improvement = sum(
            rec.estimated_improvement.improvement_percentage
            * rec.estimated_improvement.confidence
            for rec in recommendations
        )

        total_confidence = sum(
            rec.estimated_improvement.confidence for rec in recommendations
        )

        if total_confidence == 0:
            return 0.0

        # Apply diminishing returns for multiple optimizations
        avg_improvement = total_weighted_improvement / total_confidence
        diminishing_factor = 1 - (
            0.1 * (len(recommendations) - 1)
        )  # Reduce by 10% per additional optimization

        return max(0, avg_improvement * max(0.3, diminishing_factor))

    def export_report(
        self,
        report: WorkflowOptimizationReport,
        file_path: str,
        format: ReportFormat = ReportFormat.JSON,
    ):
        """Export optimization report to file."""

        if format == ReportFormat.JSON:
            self._export_json_report(report, file_path)
        elif format == ReportFormat.HTML:
            self._export_html_report(report, file_path)
        elif format == ReportFormat.MARKDOWN:
            self._export_markdown_report(report, file_path)
        elif format == ReportFormat.TEXT:
            self._export_text_report(report, file_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")

        logger.info(
            f"Exported optimization report to {file_path} ({format.value} format)"
        )

    def _export_json_report(self, report: WorkflowOptimizationReport, file_path: str):
        """Export report as JSON."""
        with open(file_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def _export_html_report(self, report: WorkflowOptimizationReport, file_path: str):
        """Export report as HTML."""
        html_content = self._generate_html_report(report)
        with open(file_path, "w") as f:
            f.write(html_content)

    def _export_markdown_report(
        self, report: WorkflowOptimizationReport, file_path: str
    ):
        """Export report as Markdown."""
        markdown_content = self._generate_markdown_report(report)
        with open(file_path, "w") as f:
            f.write(markdown_content)

    def _export_text_report(self, report: WorkflowOptimizationReport, file_path: str):
        """Export report as plain text."""
        text_content = self._generate_text_report(report)
        with open(file_path, "w") as f:
            f.write(text_content)

    def _generate_html_report(self, report: WorkflowOptimizationReport) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XPCS Toolkit Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; }}
                .recommendation {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
                .critical {{ border-left: 5px solid #ff4444; }}
                .high {{ border-left: 5px solid #ff8800; }}
                .medium {{ border-left: 5px solid #ffbb33; }}
                .low {{ border-left: 5px solid #00aa00; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>XPCS Toolkit Workflow Optimization Report</h1>
                <p>Generated: {datetime.fromtimestamp(report.generation_timestamp).strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Report ID: {report.report_id}</p>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Analyzed Workflows:</strong> {report.analyzed_workflows}</p>
                <p><strong>Critical Issues:</strong> {report.critical_issues}</p>
                <p><strong>High Priority Optimizations:</strong> {report.high_priority_optimizations}</p>
                <p><strong>Estimated Total Improvement:</strong> {report.estimated_total_improvement:.1f}%</p>
            </div>
            
            <h2>Optimization Recommendations</h2>
        """

        for rec in report.optimization_recommendations[:10]:  # Top 10 recommendations
            impact_class = rec.estimated_improvement.impact_level.value
            html += f"""
            <div class="recommendation {impact_class}">
                <h3>{rec.title}</h3>
                <p><strong>Category:</strong> {rec.category}</p>
                <p><strong>Impact:</strong> {rec.estimated_improvement.improvement_percentage:.1f}% improvement</p>
                <p><strong>Implementation Effort:</strong> {rec.estimated_improvement.implementation_effort}</p>
                <p>{rec.description}</p>
                <details>
                    <summary>Implementation Details</summary>
                    <ul>
                        {"".join(f"<li>{step}</li>" for step in rec.implementation_steps)}
                    </ul>
                </details>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, report: WorkflowOptimizationReport) -> str:
        """Generate Markdown report content."""
        md = f"""# XPCS Toolkit Workflow Optimization Report

**Report ID:** {report.report_id}  
**Generated:** {datetime.fromtimestamp(report.generation_timestamp).strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

- **Analyzed Workflows:** {report.analyzed_workflows}
- **Critical Issues:** {report.critical_issues}
- **High Priority Optimizations:** {report.high_priority_optimizations}  
- **Estimated Total Improvement:** {report.estimated_total_improvement:.1f}%

## Top Optimization Recommendations

"""

        for i, rec in enumerate(report.optimization_recommendations[:10], 1):
            impact_emoji = {
                OptimizationImpact.CRITICAL: "🔴",
                OptimizationImpact.HIGH: "🟠",
                OptimizationImpact.MEDIUM: "🟡",
                OptimizationImpact.LOW: "🟢",
            }.get(rec.estimated_improvement.impact_level, "⚪")

            md += f"""### {i}. {impact_emoji} {rec.title}

**Category:** {rec.category}  
**Impact:** {rec.estimated_improvement.improvement_percentage:.1f}% improvement  
**Implementation Effort:** {rec.estimated_improvement.implementation_effort}  
**Priority Score:** {rec.priority_score:.1f}

{rec.description}

**Implementation Steps:**
"""
            for step in rec.implementation_steps:
                md += f"- {step}\n"

            md += "\n---\n\n"

        return md

    def _generate_text_report(self, report: WorkflowOptimizationReport) -> str:
        """Generate plain text report content."""
        text = f"""
XPCS Toolkit Workflow Optimization Report
==========================================

Report ID: {report.report_id}
Generated: {datetime.fromtimestamp(report.generation_timestamp).strftime("%Y-%m-%d %H:%M:%S")}

EXECUTIVE SUMMARY
-----------------
Analyzed Workflows: {report.analyzed_workflows}
Critical Issues: {report.critical_issues}
High Priority Optimizations: {report.high_priority_optimizations}
Estimated Total Improvement: {report.estimated_total_improvement:.1f}%

TOP OPTIMIZATION RECOMMENDATIONS
---------------------------------
"""

        for i, rec in enumerate(report.optimization_recommendations[:10], 1):
            text += f"""
{i}. {rec.title}
   Category: {rec.category}
   Impact: {rec.estimated_improvement.improvement_percentage:.1f}% improvement
   Implementation Effort: {rec.estimated_improvement.implementation_effort}
   Priority Score: {rec.priority_score:.1f}
   
   Description: {rec.description}
   
   Implementation Steps:
"""
            for step in rec.implementation_steps:
                text += f"   - {step}\n"

            text += "\n" + "=" * 50 + "\n"

        return text

    def get_report_summary(self, report_id: str) -> Dict[str, Any]:
        """Get a summary of a specific report."""
        if report_id not in self.report_cache:
            return {"error": f"Report {report_id} not found"}

        report = self.report_cache[report_id]

        return {
            "report_id": report_id,
            "generation_timestamp": report.generation_timestamp,
            "analyzed_workflows": report.analyzed_workflows,
            "total_recommendations": len(report.optimization_recommendations),
            "critical_issues": report.critical_issues,
            "high_priority_optimizations": report.high_priority_optimizations,
            "estimated_total_improvement": report.estimated_total_improvement,
            "top_recommendations": [
                {
                    "title": rec.title,
                    "category": rec.category,
                    "improvement_percentage": rec.estimated_improvement.improvement_percentage,
                    "priority_score": rec.priority_score,
                }
                for rec in report.optimization_recommendations[:5]
            ],
        }


# Global report generator instance
optimization_report_generator = WorkflowOptimizationReportGenerator()


def get_report_generator() -> WorkflowOptimizationReportGenerator:
    """
    Get the global workflow optimization report generator instance.

    Returns
    -------
    WorkflowOptimizationReportGenerator
        Global report generator instance
    """
    return optimization_report_generator


# Convenience functions


def generate_optimization_report(
    workflow_count: int = 50,
) -> WorkflowOptimizationReport:
    """Generate a comprehensive optimization report."""
    return optimization_report_generator.generate_comprehensive_report(workflow_count)


def export_optimization_report(
    report: WorkflowOptimizationReport,
    file_path: str,
    format: ReportFormat = ReportFormat.JSON,
):
    """Export optimization report to file."""
    optimization_report_generator.export_report(report, file_path, format)


def get_quick_optimization_summary() -> Dict[str, Any]:
    """Get a quick summary of current optimization opportunities."""
    report = generate_optimization_report(20)  # Analyze last 20 workflows
    return optimization_report_generator.get_report_summary(report.report_id)


def get_top_optimization_recommendations(count: int = 5) -> List[Dict[str, Any]]:
    """Get top optimization recommendations."""
    report = generate_optimization_report(30)

    return [
        {
            "title": rec.title,
            "category": rec.category,
            "description": rec.description,
            "improvement_percentage": rec.estimated_improvement.improvement_percentage,
            "implementation_effort": rec.estimated_improvement.implementation_effort,
            "priority_score": rec.priority_score,
        }
        for rec in report.optimization_recommendations[:count]
    ]
