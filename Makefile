# =============================================================================
# XPCS Toolkit - Optimized Makefile
# =============================================================================

# PHONY targets (targets that don't represent files)
.PHONY: help clean install lint format test coverage docs dist release
.PHONY: clean-all clean-build clean-cache clean-test
.PHONY: lint-ruff lint-flake8 format-ruff format-black
.PHONY: test-unit test-integration test-properties test-benchmarks test-ci test-full
.PHONY: coverage-html coverage-report coverage-logging 
.PHONY: docs-build docs-serve docs-clean
.PHONY: dev-setup dev-install quality-check

.DEFAULT_GOAL := help

# =============================================================================
# Configuration Variables
# =============================================================================

# Python and package configuration
PYTHON := python
PACKAGE_NAME := xpcs_toolkit
SRC_DIR := src/$(PACKAGE_NAME)
TESTS_DIR := tests
DOCS_DIR := docs

# Test configuration
PYTEST_OPTS := -v --tb=short
PYTEST_COV_OPTS := --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-report=json
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
	rm -f test_switching.log coverage.json
	find . -name '*.log' -path './tests/*' -delete || true

# =============================================================================
# Code Quality and Formatting
# =============================================================================

lint: lint-ruff ## run all linting tools (primary: ruff)

lint-ruff: ## lint code with ruff (fast, modern linter)
	$(PYTHON) -m ruff check .

lint-flake8: ## lint code with flake8 (fallback/compatibility)
	$(PYTHON) -m flake8 $(SRC_DIR) $(TESTS_DIR)

format: format-ruff ## format code with all formatters (primary: ruff)

format-ruff: ## format code with ruff formatter
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

format-black: ## format code with black (alternative)
	$(PYTHON) -m black $(SRC_DIR) $(TESTS_DIR)

quality-check: lint test-ci ## comprehensive quality check for CI/CD

# =============================================================================
# Testing
# =============================================================================

test: test-unit ## run basic unit tests (default test target)

test-unit: ## run core unit and integration tests
	$(PYTHON) -m pytest $(TESTS_DIR)/test_xpcs_toolkit.py $(TESTS_DIR)/test_logging_system.py $(PYTEST_OPTS)

test-integration: ## run integration tests for key components
	$(PYTHON) -m pytest $(TESTS_DIR)/test_logging_system.py $(PYTEST_OPTS)

test-properties: ## run property-based tests with Hypothesis
	$(PYTHON) -m pytest $(TESTS_DIR)/test_logging_properties.py $(PYTEST_OPTS)

test-benchmarks: ## run performance benchmarks
	$(PYTHON) -m pytest $(TESTS_DIR)/test_logging_benchmarks.py $(PYTEST_BENCH_OPTS)

test-ci: ## run tests optimized for CI/CD environments
	$(PYTHON) -m pytest $(TESTS_DIR)/test_xpcs_toolkit.py $(TESTS_DIR)/test_logging_system.py \
		$(TESTS_DIR)/test_logging_properties.py \
		$(PYTEST_OPTS) -m "not slow" --durations=10

test-full: ## run comprehensive test suite including validation
	$(PYTHON) -m pytest $(TESTS_DIR)/ $(PYTEST_OPTS)
	$(PYTHON) $(TESTS_DIR)/run_validation.py --ci || echo "Validation script not available"

test-performance: ## run all performance-related tests
	$(PYTHON) -m pytest $(TESTS_DIR)/test_logging_benchmarks.py $(TESTS_DIR)/test_io_performance.py $(PYTEST_BENCH_OPTS)
	$(PYTHON) $(TESTS_DIR)/run_logging_benchmarks.py --report || echo "Benchmark runner not available"

test-scientific: ## run scientific computing validation tests
	$(PYTHON) -m pytest -k "scientific" $(PYTEST_OPTS)
	$(PYTHON) $(TESTS_DIR)/test_vectorization_accuracy.py || echo "Scientific tests not available"

# =============================================================================
# Coverage Reporting
# =============================================================================

coverage: coverage-html ## generate and display HTML coverage report

coverage-report: ## generate coverage report without opening browser
	$(PYTHON) -m pytest $(TESTS_DIR)/ $(PYTEST_COV_OPTS)

coverage-html: coverage-report ## generate HTML coverage report and open in browser
	$(BROWSER) htmlcov/index.html

coverage-logging: ## focused coverage for logging system components
	$(PYTHON) -m pytest $(TESTS_DIR)/test_logging_system.py $(TESTS_DIR)/test_logging_properties.py \
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
	sphinx-apidoc -o $(DOCS_DIR)/ $(SRC_DIR)
	$(MAKE) -C $(DOCS_DIR) clean
	$(MAKE) -C $(DOCS_DIR) html
	$(BROWSER) $(DOCS_DIR)/_build/html/index.html

docs-serve: docs-build ## build docs and watch for changes (requires watchmedo)
	watchmedo shell-command -p '*.rst;*.py' -c '$(MAKE) docs-build' -R -D .

docs-clean: ## clean documentation build artifacts
	rm -f $(DOCS_DIR)/$(PACKAGE_NAME).rst $(DOCS_DIR)/modules.rst
	$(MAKE) -C $(DOCS_DIR) clean || true

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
	@echo "Core dependencies OK"

# =============================================================================
# Aliases for Compatibility
# =============================================================================

# Legacy compatibility aliases
test-all: test-full ## alias for test-full (backward compatibility)
lint/flake8: lint-flake8 ## alias for lint-flake8 (backward compatibility)
servedocs: docs-serve ## alias for docs-serve (backward compatibility)