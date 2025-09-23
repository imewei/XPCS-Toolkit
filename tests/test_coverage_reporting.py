#!/usr/bin/env python3
"""
Test Coverage Reporting System Validation.

This module validates the enhanced coverage reporting system with multiple
output formats and trend analysis capabilities.
"""

import contextlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from tests.coverage_manager import CoverageManager, CoverageMetrics, CoverageReport


class TestCoverageReporting(unittest.TestCase):
    """Test cases for coverage reporting system."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CoverageManager(Path(self.temp_dir))

        # Create sample coverage data
        self.sample_metrics = [
            CoverageMetrics(
                module_name="xpcs_toolkit/module/g2mod.py",
                statements=150,
                missing=20,
                branches=30,
                partial_branches=5,
                coverage_percentage=86.7,
                branch_coverage_percentage=83.3,
            ),
            CoverageMetrics(
                module_name="xpcs_toolkit/xpcs_file.py",
                statements=200,
                missing=40,
                branches=50,
                partial_branches=10,
                coverage_percentage=80.0,
                branch_coverage_percentage=80.0,
            ),
        ]

        self.sample_report = CoverageReport(
            timestamp=datetime.now(),
            overall_coverage=85.5,
            branch_coverage=82.1,
            total_statements=350,
            total_missing=60,
            module_metrics=self.sample_metrics,
            targets_met=["G2 correlation analysis: 86.7% >= 95.0%"],
            targets_missed=["Core data container: 80.0% < 85.0%"],
            regression_alerts=["Test coverage regression detected"],
        )

    def tearDown(self):
        """Clean up test environment and close database connections."""
        # Clean up CoverageManager and all its database connections
        if hasattr(self, "manager"):
            # Force garbage collection of the manager
            del self.manager

        # Force close any lingering database connections
        import gc
        import sqlite3

        # Close any lingering sqlite connections explicitly
        try:
            # Find all sqlite3.Connection objects and close them
            for obj in gc.get_objects():
                if isinstance(obj, sqlite3.Connection):
                    try:
                        obj.close()
                    except Exception:
                        pass  # Already closed or error closing
        except Exception:
            pass

        # Force garbage collection to clear any remaining references
        gc.collect()

        # Clean up temporary directory
        import shutil

        with contextlib.suppress(Exception):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_json_report_generation(self):
        """Test JSON report generation."""
        json_report = self.manager.generate_json_report(self.sample_report)

        # Validate JSON structure
        report_data = json.loads(json_report)

        self.assertIn("timestamp", report_data)
        self.assertIn("overall_metrics", report_data)
        self.assertIn("targets", report_data)
        self.assertIn("regression_alerts", report_data)
        self.assertIn("module_metrics", report_data)

        # Check overall metrics
        overall = report_data["overall_metrics"]
        self.assertEqual(overall["overall_coverage"], 85.5)
        self.assertEqual(overall["branch_coverage"], 82.1)
        self.assertEqual(overall["total_statements"], 350)
        self.assertEqual(overall["total_missing"], 60)
        self.assertEqual(overall["lines_covered"], 290)

        # Check targets
        targets = report_data["targets"]
        self.assertEqual(targets["total_targets"], 18)  # From manager's targets
        self.assertEqual(targets["targets_met"], 1)
        self.assertEqual(targets["targets_missed"], 1)

        # Check module metrics
        self.assertEqual(len(report_data["module_metrics"]), 2)
        module = report_data["module_metrics"][0]
        self.assertIn("module_name", module)
        self.assertIn("coverage_percentage", module)
        self.assertIn("target_info", module)

    def test_html_report_generation(self):
        """Test HTML report generation."""
        html_report = self.manager.generate_html_report(self.sample_report)

        # Validate HTML structure
        self.assertIn("<!DOCTYPE html>", html_report)
        self.assertIn("<title>XPCS Toolkit Coverage Report</title>", html_report)
        self.assertIn("Line Coverage", html_report)
        self.assertIn("Branch Coverage", html_report)
        self.assertIn("85.5%", html_report)  # Overall coverage
        self.assertIn("82.1%", html_report)  # Branch coverage

        # Check for module table
        self.assertIn('<table class="module-table">', html_report)
        self.assertIn("xpcs_toolkit/module/g2mod.py", html_report)
        self.assertIn("xpcs_toolkit/xpcs_file.py", html_report)

        # Check for regression alerts
        self.assertIn("Regression Alerts", html_report)
        self.assertIn("Test coverage regression detected", html_report)

    def test_xml_report_generation(self):
        """Test XML report generation."""
        xml_report = self.manager.generate_xml_report(self.sample_report)

        # Parse and validate XML
        root = ET.fromstring(xml_report)

        self.assertEqual(root.tag, "coverage_report")
        self.assertIn("timestamp", root.attrib)
        self.assertEqual(root.attrib["version"], "1.0")

        # Check overall metrics
        overall = root.find("overall_metrics")
        self.assertIsNotNone(overall)
        self.assertEqual(overall.find("overall_coverage").text, "85.5")
        self.assertEqual(overall.find("branch_coverage").text, "82.1")

        # Check targets
        targets = root.find("targets")
        self.assertIsNotNone(targets)
        self.assertEqual(targets.attrib["total"], "18")
        self.assertEqual(targets.attrib["met"], "1")
        self.assertEqual(targets.attrib["missed"], "1")

        # Check modules
        modules = root.find("modules")
        self.assertIsNotNone(modules)
        module_list = modules.findall("module")
        self.assertEqual(len(module_list), 2)

        # Check module attributes
        first_module = module_list[0]
        self.assertIn("name", first_module.attrib)
        self.assertIn("coverage", first_module.attrib)
        self.assertIn("statements", first_module.attrib)

    def test_trend_analysis_structure(self):
        """Test trend data export structure."""
        # This test validates the structure without actual database data
        trends = self.manager.export_coverage_trends(30)

        required_keys = [
            "timestamps",
            "overall_coverage",
            "branch_coverage",
            "target_success_rate",
            "profiles",
        ]

        for key in required_keys:
            self.assertIn(key, trends)
            self.assertIsInstance(trends[key], list)

    def test_text_report_generation(self):
        """Test detailed text report generation."""
        text_report = self.manager.generate_detailed_report(self.sample_report)

        # Validate text report content
        self.assertIn("XPCS TOOLKIT COVERAGE ANALYSIS REPORT", text_report)
        self.assertIn("OVERALL COVERAGE METRICS", text_report)
        self.assertIn("Line Coverage: 85.50%", text_report)
        self.assertIn("Branch Coverage: 82.10%", text_report)

        # Check targets section
        self.assertIn("COVERAGE TARGET ANALYSIS", text_report)
        self.assertIn("TARGETS MET:", text_report)
        self.assertIn("TARGETS MISSED:", text_report)

        # Check regression alerts
        self.assertIn("REGRESSION ALERTS:", text_report)
        self.assertIn("Test coverage regression detected", text_report)

        # Check module breakdown
        self.assertIn("MODULES WITH LOWEST COVERAGE", text_report)
        self.assertIn("CRITICAL MODULES STATUS", text_report)

    def test_report_file_saving(self):
        """Test report saving to files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON save
            json_file = Path(temp_dir) / "test_report.json"
            json_content = self.manager.generate_json_report(self.sample_report)

            with open(json_file, "w") as f:
                f.write(json_content)

            self.assertTrue(json_file.exists())

            # Validate saved JSON
            with open(json_file) as f:
                saved_data = json.load(f)
            self.assertIn("timestamp", saved_data)

            # Test HTML save
            html_file = Path(temp_dir) / "test_report.html"
            html_content = self.manager.generate_html_report(self.sample_report)

            with open(html_file, "w") as f:
                f.write(html_content)

            self.assertTrue(html_file.exists())
            self.assertGreater(html_file.stat().st_size, 1000)  # Should be substantial

    def test_module_target_matching(self):
        """Test module target matching logic."""
        json_report = self.manager.generate_json_report(self.sample_report)
        report_data = json.loads(json_report)

        # Check that modules have target information
        for module in report_data["module_metrics"]:
            if module["module_name"] == "xpcs_toolkit/module/g2mod.py":
                target_info = module["target_info"]
                self.assertIsNotNone(target_info)
                self.assertEqual(target_info["target_percentage"], 95.0)
                self.assertTrue(target_info["critical"])
            elif module["module_name"] == "xpcs_toolkit/xpcs_file.py":
                target_info = module["target_info"]
                self.assertIsNotNone(target_info)
                self.assertEqual(target_info["target_percentage"], 85.0)
                self.assertTrue(target_info["critical"])


if __name__ == "__main__":
    unittest.main()
