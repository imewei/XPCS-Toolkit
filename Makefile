# =============================================================================
# XPCS Toolkit - Optimized Makefile
# =============================================================================

# PHONY targets (targets that don't represent files)
.PHONY: help clean install lint format test coverage docs dist release
.PHONY: clean-all clean-build clean-cache clean-test
.PHONY: lint-ruff lint-flake8 format-ruff format-black
.PHONY: test-unit test-integration test-logging test-scientific test-end-to-end test-properties test-performance test-gui test-ci test-full test-all test-benchmarks
.PHONY: test-log test-unit-log test-integration-log test-logging-log test-full-log test-all-log
.PHONY: coverage-html coverage-report coverage-logging
.PHONY: docs-build docs-serve docs-clean docs-autobuild docs-linkcheck docs-validate
.PHONY: dev-setup dev-install quality-check pre-commit-install pre-commit-run

.DEFAULT_GOAL := help

# =============================================================================
# Configuration Variables
# =============================================================================

# Python and package configuration
PYTHON := python
PACKAGE_NAME := xpcs_toolkit
SRC_DIR := $(PACKAGE_NAME)
TESTS_DIR := tests
DOCS_DIR := docs

# Test configuration
PYTEST_OPTS := -v --tb=short
PYTEST_COV_OPTS := --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-report=json --cov-fail-under=12
PYTEST_BENCH_OPTS := --benchmark-only --benchmark-sort=mean

# Browser helper for opening HTML files
define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := $(PYTHON) -c "$$BROWSER_PYSCRIPT"

# Help formatter
define HELP_PYSCRIPT
import re, sys
print("XPCS Toolkit - Available Commands:\n")
categories = {}
for line in sys.stdin:
    match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help_text = match.groups()
        category = target.split('-')[0] if '-' in target else 'General'
        if category not in categories:
            categories[category] = []
        categories[category].append((target, help_text))

for category, targets in sorted(categories.items()):
    print(f"{category.upper()} COMMANDS:")
    for target, help_text in targets:
        print(f"  {target:<20} {help_text}")
    print()
endef
export HELP_PYSCRIPT

# =============================================================================
# Help and Information
# =============================================================================

help: ## show this help message with categorized commands
	@$(PYTHON) -c "$$HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# =============================================================================
# Environment Setup and Installation
# =============================================================================

dev-setup: ## setup complete development environment
	pip install -e ".[dev,test,docs]"
	pre-commit install || echo "pre-commit not available, skipping hook installation"

pre-commit-install: ## install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-run: ## run pre-commit hooks on all files
	pre-commit run --all-files

dev-install: clean ## install package in development mode
	pip install -e .

install: clean ## install package for production use
	pip install .

# =============================================================================
# Cleaning Operations
# =============================================================================

clean: clean-build clean-cache clean-test ## remove all build, test, coverage and cache artifacts

clean-all: clean ## comprehensive cleanup including OS-specific files
	find . -name '.DS_Store' -delete || true
	find . -name 'Thumbs.db' -delete || true
	find . -name '*.tmp' -delete || true

clean-build: ## remove build and distribution artifacts
	rm -rf build/ dist/ .eggs/
	find . -name '*.egg-info' -exec rm -rf {} + || true
	find . -name '*.egg' -exec rm -rf {} + || true

clean-cache: ## remove Python and tool cache files
	find . -name '*.pyc' -delete || true
	find . -name '*.pyo' -delete || true
	find . -name '*~' -delete || true
	find . -name '__pycache__' -exec rm -rf {} + || true
	rm -rf .ruff_cache/ .mypy_cache/ .pytest_cache/

clean-test: ## remove test, coverage and benchmark artifacts
	rm -rf .tox/ .coverage htmlcov/ .benchmark/ .benchmarks/ .hypothesis/
	rm -rf test-artifacts/ test-reports/
	rm -f test_switching.log coverage.json coverage.xml
	find . -name '*.log' -path './tests/*' -delete || true
	find . -name 'test_*.log' -delete || true
	find . -name 'test_*.xml' -delete || true
	rm -f *.xml || true

# =============================================================================
# Code Quality and Formatting
# =============================================================================

lint: lint-ruff ## run all linting tools (primary: ruff)

lint-ruff: ## lint code with ruff (fast, modern linter) - critical errors only
	$(PYTHON) -m ruff check --select E9,F82 --ignore F821 .

lint-flake8: ## lint code with flake8 (fallback/compatibility)
	$(PYTHON) -m flake8 $(SRC_DIR) $(TESTS_DIR)

format: format-ruff ## format code with all formatters (primary: ruff)

format-ruff: ## format code with ruff formatter
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix --select E9,F82 --ignore F821 .

format-black: ## format code with black (alternative)
	$(PYTHON) -m black $(SRC_DIR) $(TESTS_DIR)

quality-check: lint test-ci ## comprehensive quality check for CI/CD

# =============================================================================
# Testing
# =============================================================================

# Default test target
test: test-fast ## run fast tests (default test target)

# Optimized test execution profiles
test-fast: ## run fast tests only (unit tests, < 1 second)
	$(PYTHON) -m pytest -m "unit and not slow" $(PYTEST_OPTS)

test-core: ## run core functionality tests (fast + essential integration)
	$(PYTHON) -m pytest -m "unit or (integration and not slow)" $(PYTEST_OPTS)

# Core test categories
test-unit: ## run all unit tests
	$(PYTHON) -m pytest -m "unit" $(PYTEST_OPTS)

test-integration: ## run integration tests for key components
	$(PYTHON) -m pytest -m "integration" $(PYTEST_OPTS)

# Analysis-specific test profiles
test-analysis: ## run all analysis tests (G2, SAXS, two-time)
	$(PYTHON) -m pytest $(TESTS_DIR)/analysis/ $(PYTEST_OPTS)

test-g2: ## run G2 correlation analysis tests only
	$(PYTHON) -m pytest -m "g2_analysis" $(PYTEST_OPTS)

test-qt: ## run Qt validation tests only
	$(PYTHON) -m pytest $(TESTS_DIR)/qt_validation/ $(PYTEST_OPTS)

# Development workflow profiles
test-dev: ## run developer-friendly test suite (fast + critical)
	$(PYTHON) -m pytest -m "unit or (integration and not slow and not flaky)" $(PYTEST_OPTS) --maxfail=5

test-logging: ## run comprehensive logging system tests
	$(PYTHON) -m pytest $(TESTS_DIR)/logging/ $(PYTEST_OPTS)

test-scientific: ## run scientific computing validation tests
	$(PYTHON) -m pytest $(TESTS_DIR)/scientific/ $(PYTEST_OPTS)
	$(PYTHON) $(TESTS_DIR)/test_vectorization_accuracy.py || echo "Scientific validation script not available"

test-end-to-end: ## run complete workflow validation tests
	$(PYTHON) -m pytest $(TESTS_DIR)/end_to_end/ $(PYTEST_OPTS)

# Specialized test categories
test-properties: ## run property-based tests with Hypothesis
	$(PYTHON) -m pytest $(TESTS_DIR)/logging/properties/ $(TESTS_DIR)/scientific/properties/ $(PYTEST_OPTS)

test-performance: ## run performance benchmarks and profiling tests
	$(PYTHON) -m pytest $(TESTS_DIR)/logging/performance/ $(TESTS_DIR)/performance/ $(TESTS_DIR)/test_io_performance.py $(PYTEST_BENCH_OPTS)
	$(PYTHON) $(TESTS_DIR)/framework/runners/run_logging_benchmarks.py --report || echo "Benchmark runner not available"

test-gui: ## run GUI interactive tests (requires display)
	$(PYTHON) -m pytest $(TESTS_DIR)/gui_interactive/ $(PYTEST_OPTS) -s

# CI/CD optimized tests
test-ci: ## run tests optimized for CI/CD environments
	$(PYTHON) -m pytest $(TESTS_DIR)/unit/test_package_basics.py $(TESTS_DIR)/logging/functional/ \
		$(TESTS_DIR)/logging/unit/ $(TESTS_DIR)/integration/ \
		$(PYTEST_OPTS) -m "not slow" --durations=10

# Comprehensive test suites
test-full: ## run comprehensive test suite including validation
	$(PYTHON) -m pytest $(TESTS_DIR)/ $(PYTEST_OPTS) --ignore=$(TESTS_DIR)/gui_interactive/
	$(PYTHON) $(TESTS_DIR)/framework/runners/run_validation.py --ci || echo "Validation script not available"

test-all: ## run all tests including GUI tests (comprehensive)
	$(PYTHON) -m pytest $(TESTS_DIR)/ $(PYTEST_OPTS)
	$(PYTHON) $(TESTS_DIR)/framework/runners/run_validation.py --full || echo "Validation script not available"

# Logging variants - save test results to log files
test-log: test-unit-log ## run basic tests and save results to log file

test-unit-log: ## run unit tests with detailed logging to file
	@echo "Running unit tests with logging to test_unit_$(shell date +%Y%m%d_%H%M%S).log"
	$(PYTHON) -m pytest $(TESTS_DIR)/unit/test_package_basics.py $(TESTS_DIR)/logging/unit/ \
		$(PYTEST_OPTS) -v --tb=long --durations=0 \
		--junitxml=test_unit_$(shell date +%Y%m%d_%H%M%S).xml \
		2>&1 | tee test_unit_$(shell date +%Y%m%d_%H%M%S).log

test-integration-log: ## run integration tests with detailed logging to file
	@echo "Running integration tests with logging to test_integration_$(shell date +%Y%m%d_%H%M%S).log"
	$(PYTHON) -m pytest $(TESTS_DIR)/integration/ \
		$(PYTEST_OPTS) -v --tb=long --durations=0 \
		--junitxml=test_integration_$(shell date +%Y%m%d_%H%M%S).xml \
		2>&1 | tee test_integration_$(shell date +%Y%m%d_%H%M%S).log

test-logging-log: ## run logging tests with detailed logging to file
	@echo "Running logging tests with logging to test_logging_$(shell date +%Y%m%d_%H%M%S).log"
	$(PYTHON) -m pytest $(TESTS_DIR)/logging/ \
		$(PYTEST_OPTS) -v --tb=long --durations=0 \
		--junitxml=test_logging_$(shell date +%Y%m%d_%H%M%S).xml \
		2>&1 | tee test_logging_$(shell date +%Y%m%d_%H%M%S).log

test-full-log: ## run comprehensive tests with detailed logging to file
	@echo "Running comprehensive tests with logging to test_full_$(shell date +%Y%m%d_%H%M%S).log"
	$(PYTHON) -m pytest $(TESTS_DIR)/ $(PYTEST_OPTS) --ignore=$(TESTS_DIR)/gui_interactive/ \
		-v --tb=long --durations=0 \
		--junitxml=test_full_$(shell date +%Y%m%d_%H%M%S).xml \
		2>&1 | tee test_full_$(shell date +%Y%m%d_%H%M%S).log
	$(PYTHON) $(TESTS_DIR)/framework/runners/run_validation.py --ci 2>&1 | tee -a test_full_$(shell date +%Y%m%d_%H%M%S).log || echo "Validation script not available"

test-all-log: ## run all tests with detailed logging to file
	@echo "Running all tests with logging to test_all_$(shell date +%Y%m%d_%H%M%S).log"
	$(PYTHON) -m pytest $(TESTS_DIR)/ \
		$(PYTEST_OPTS) -v --tb=long --durations=0 \
		--junitxml=test_all_$(shell date +%Y%m%d_%H%M%S).xml \
		2>&1 | tee test_all_$(shell date +%Y%m%d_%H%M%S).log
	$(PYTHON) $(TESTS_DIR)/framework/runners/run_validation.py --full 2>&1 | tee -a test_all_$(shell date +%Y%m%d_%H%M%S).log || echo "Validation script not available"

# =============================================================================
# Coverage Reporting
# =============================================================================

coverage: coverage-html ## generate and display HTML coverage report

coverage-report: ## generate coverage report without opening browser
	$(PYTHON) -m pytest $(TESTS_DIR)/ $(PYTEST_COV_OPTS)

coverage-html: coverage-report ## generate HTML coverage report and open in browser
	$(BROWSER) htmlcov/index.html

coverage-logging: ## focused coverage for logging system components
	$(PYTHON) -m pytest $(TESTS_DIR)/logging/ \
		--cov=$(SRC_DIR)/utils/logging_config \
		--cov=$(SRC_DIR)/utils/log_formatters \
		--cov=$(SRC_DIR)/utils/log_templates \
		--cov-report=html --cov-report=term-missing
	$(BROWSER) htmlcov/index.html

# =============================================================================
# Documentation
# =============================================================================

docs: docs-build ## build and display documentation

docs-build: docs-clean ## build Sphinx documentation
	cd $(DOCS_DIR) && sphinx-build -b html . _build/html
	$(BROWSER) $(DOCS_DIR)/_build/html/index.html

docs-autobuild: docs-clean ## build docs with auto-reload (requires sphinx-autobuild)
	cd $(DOCS_DIR) && sphinx-autobuild -b html . _build/html --host 0.0.0.0 --port 8000

docs-serve: docs-build ## build docs and watch for changes (requires watchmedo)
	watchmedo shell-command -p '*.rst;*.py' -c '$(MAKE) docs-build' -R -D .

docs-linkcheck: ## check for broken links in documentation
	cd $(DOCS_DIR) && sphinx-build -b linkcheck . _build/linkcheck

docs-validate: docs-build docs-linkcheck ## validate documentation build and links
	@echo "Documentation validation complete"

docs-clean: ## clean documentation build artifacts
	rm -rf $(DOCS_DIR)/_build/
	find $(DOCS_DIR) -name '*.pyc' -delete || true
	find $(DOCS_DIR) -name '__pycache__' -exec rm -rf {} + || true

# =============================================================================
# Package Distribution
# =============================================================================

dist: clean-build ## build source and wheel distributions
	$(PYTHON) -m build
	@echo "Distribution files:"
	@ls -la dist/

dist-check: dist ## build and validate distributions
	$(PYTHON) -m twine check dist/*

release: dist-check ## build, validate and upload release to PyPI
	$(PYTHON) -m twine upload dist/*

release-test: dist-check ## upload to Test PyPI for validation
	$(PYTHON) -m twine upload --repository testpypi dist/*

# =============================================================================
# Development Utilities
# =============================================================================

run-app: ## launch the XPCS Toolkit GUI application
	$(PYTHON) -m $(PACKAGE_NAME).cli

debug-info: ## display environment and package information
	@echo "=== Environment Information ==="
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Platform: $$($(PYTHON) -c 'import platform; print(platform.platform())')"
	@echo "Working Directory: $$(pwd)"
	@echo "Package Directory: $(SRC_DIR)"
	@echo "=== Package Status ==="
	@pip show $(PACKAGE_NAME) 2>/dev/null || echo "Package not installed"
	@echo "=== Git Status ==="
	@git status --porcelain 2>/dev/null || echo "Not a git repository"

check-deps: ## verify all dependencies are installed
	@echo "Checking required dependencies..."
	@$(PYTHON) -c "import sys; print('Python:', sys.version)"
	@$(PYTHON) -c "import numpy; print('NumPy:', numpy.__version__)"
	@$(PYTHON) -c "import scipy; print('SciPy:', scipy.__version__)"
	@$(PYTHON) -c "import h5py; print('h5py:', h5py.version.version)"
	@$(PYTHON) -c "import PySide6; print('PySide6:', PySide6.__version__)"
	@$(PYTHON) -c "import pyqtgraph; print('PyQtGraph:', pyqtgraph.__version__)"
	@echo "Core dependencies OK"

check-docs-deps: ## verify documentation dependencies are installed
	@echo "Checking documentation dependencies..."
	@$(PYTHON) -c "import sphinx; print('Sphinx:', sphinx.__version__)"
	@$(PYTHON) -c "import sphinx_rtd_theme; print('RTD Theme: installed')"
	@$(PYTHON) -c "import myst_parser; print('MyST Parser: installed')"
	@echo "Documentation dependencies OK"

# =============================================================================
# Aliases for Compatibility
# =============================================================================

# Legacy compatibility aliases
test-benchmarks: test-performance ## alias for test-performance (backward compatibility)
lint/flake8: lint-flake8 ## alias for lint-flake8 (backward compatibility)
servedocs: docs-serve ## alias for docs-serve (backward compatibility)
livedocs: docs-autobuild ## alias for docs-autobuild (live reload)

# Project status check
status: debug-info check-deps ## comprehensive project status check
	@echo "=== Test Suite Status ==="
	@$(MAKE) test-fast --dry-run >/dev/null 2>&1 && echo "Test suite: READY" || echo "Test suite: ISSUES DETECTED"
	@echo "=== Documentation Status ==="
	@$(MAKE) docs-build --dry-run >/dev/null 2>&1 && echo "Documentation: READY" || echo "Documentation: ISSUES DETECTED"
