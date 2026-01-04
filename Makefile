# XPCSViewer Package Makefile
# ==============================
# Development Tools and Testing

.PHONY: help install install-dev env-info \
        test test-smoke test-fast test-ci test-full test-all test-coverage test-integration \
        test-parallel test-unit test-scientific test-gui test-gui-headless \
        clean clean-all clean-pyc clean-build clean-test clean-venv \
        format lint type-check check quick docs build publish info version \
        run-app check-deps

# Configuration
PYTHON := python
PYTEST := pytest
PACKAGE_NAME := xpcsviewer
SRC_DIR := xpcsviewer
TEST_DIR := tests
DOCS_DIR := docs
VENV := .venv

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

# Package manager detection (prioritize uv > conda/mamba > pip)
UV_AVAILABLE := $(shell command -v uv 2>/dev/null)
CONDA_PREFIX := $(shell echo $$CONDA_PREFIX)
MAMBA_AVAILABLE := $(shell command -v mamba 2>/dev/null)

# Determine package manager and commands
ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    INSTALL_CMD := uv pip install
    SYNC_CMD := uv sync
    RUN_CMD := uv run
else ifdef CONDA_PREFIX
    ifdef MAMBA_AVAILABLE
        PKG_MANAGER := mamba (using pip)
    else
        PKG_MANAGER := conda (using pip)
    endif
    PIP := pip
    INSTALL_CMD := pip install
    SYNC_CMD := pip install -e
    RUN_CMD :=
else
    PKG_MANAGER := pip
    PIP := pip
    INSTALL_CMD := pip install
    SYNC_CMD := pip install -e
    RUN_CMD :=
endif

# Colors for output
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m

# Default target
.DEFAULT_GOAL := help

# ===================
# Help target
# ===================
help:
	@echo "$(BOLD)$(BLUE)XPCSViewer Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)ENVIRONMENT$(RESET)"
	@echo "  $(CYAN)env-info$(RESET)         Show detailed environment information"
	@echo "  $(CYAN)info$(RESET)             Show project and environment info"
	@echo "  $(CYAN)version$(RESET)          Show package version"
	@echo "  $(CYAN)check-deps$(RESET)       Verify all dependencies are installed"
	@echo ""
	@echo "$(BOLD)$(GREEN)INSTALLATION$(RESET)"
	@echo "  $(CYAN)install$(RESET)          Install package in editable mode"
	@echo "  $(CYAN)install-dev$(RESET)      Install with development dependencies"
	@echo ""
	@echo "$(BOLD)$(GREEN)TESTING$(RESET)"
	@echo "  $(CYAN)test$(RESET)             Run all tests (default test suite)"
	@echo "  $(CYAN)test-smoke$(RESET)       Run smoke tests (quick sanity check, ~30s)"
	@echo "  $(CYAN)test-fast$(RESET)        Run fast tests excluding slow tests"
	@echo "  $(CYAN)test-unit$(RESET)        Run unit tests only"
	@echo "  $(CYAN)test-integration$(RESET) Run integration tests only"
	@echo "  $(CYAN)test-scientific$(RESET)  Run scientific validation tests"
	@echo "  $(CYAN)test-ci$(RESET)          Run CI test suite (matches GitHub Actions)"
	@echo "  $(CYAN)test-full$(RESET)        Run comprehensive test suite (excl. GUI)"
	@echo "  $(CYAN)test-all$(RESET)         Run ALL tests (parallel + GUI sequential)"
	@echo "  $(CYAN)test-parallel$(RESET)    Run tests in parallel (excl. GUI, 2-4x faster)"
	@echo "  $(CYAN)test-coverage$(RESET)    Run tests with coverage report"
	@echo "  $(CYAN)test-gui$(RESET)         Run GUI tests (requires display)"
	@echo "  $(CYAN)test-gui-headless$(RESET) Run GUI tests in headless mode"
	@echo ""
	@echo "$(BOLD)$(GREEN)CODE QUALITY$(RESET)"
	@echo "  $(CYAN)format$(RESET)           Format code with ruff"
	@echo "  $(CYAN)lint$(RESET)             Run linting checks (ruff)"
	@echo "  $(CYAN)type-check$(RESET)       Run type checking (mypy)"
	@echo "  $(CYAN)check$(RESET)            Run all checks (lint + type)"
	@echo "  $(CYAN)quick$(RESET)            Fast iteration: format + smoke tests"
	@echo ""
	@echo "$(BOLD)$(GREEN)DOCUMENTATION$(RESET)"
	@echo "  $(CYAN)docs$(RESET)             Build documentation with Sphinx"
	@echo "  $(CYAN)docs-serve$(RESET)       Build and serve docs with auto-reload"
	@echo ""
	@echo "$(BOLD)$(GREEN)BUILD & PUBLISH$(RESET)"
	@echo "  $(CYAN)build$(RESET)            Build distribution packages"
	@echo "  $(CYAN)publish$(RESET)          Publish to PyPI (requires credentials)"
	@echo ""
	@echo "$(BOLD)$(GREEN)APPLICATION$(RESET)"
	@echo "  $(CYAN)run-app$(RESET)          Launch the XPCS Toolkit GUI application"
	@echo ""
	@echo "$(BOLD)$(GREEN)CLEANUP$(RESET)"
	@echo "  $(CYAN)clean$(RESET)            Remove build artifacts and caches"
	@echo "  $(CYAN)clean-all$(RESET)        Deep clean of all caches"
	@echo "  $(CYAN)clean-pyc$(RESET)        Remove Python file artifacts"
	@echo "  $(CYAN)clean-build$(RESET)      Remove build artifacts"
	@echo "  $(CYAN)clean-test$(RESET)       Remove test and coverage artifacts"
	@echo "  $(CYAN)clean-venv$(RESET)       Remove virtual environment (use with caution)"
	@echo ""
	@echo "$(BOLD)Environment Detection:$(RESET)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo ""

# ===================
# Installation targets
# ===================
install:
	@echo "$(BOLD)$(BLUE)Installing $(PACKAGE_NAME) in editable mode...$(RESET)"
ifdef UV_AVAILABLE
	@$(SYNC_CMD)
else
	@$(INSTALL_CMD) -e .
endif
	@echo "$(BOLD)$(GREEN)✓ Package installed!$(RESET)"

install-dev: install
	@echo "$(BOLD)$(BLUE)Installing development dependencies...$(RESET)"
ifdef UV_AVAILABLE
	@$(SYNC_CMD) --dev --extra docs --extra test
else
	@$(INSTALL_CMD) -e ".[dev,test,docs]"
endif
	@pre-commit install 2>/dev/null || echo "pre-commit not available, skipping hook installation"
	@echo "$(BOLD)$(GREEN)✓ Dev dependencies installed!$(RESET)"

# ===================
# Environment info
# ===================
env-info:
	@echo "$(BOLD)$(BLUE)Environment Information$(RESET)"
	@echo "======================"
	@echo ""
	@echo "$(BOLD)Platform Detection:$(RESET)"
	@echo "  OS: $(UNAME_S)"
	@echo "  Platform: $(PLATFORM)"
	@echo ""
	@echo "$(BOLD)Python Environment:$(RESET)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo "  Python path: $(shell which $(PYTHON) 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "$(BOLD)Package Manager Detection:$(RESET)"
	@echo "  Active manager: $(PKG_MANAGER)"
ifdef UV_AVAILABLE
	@echo "  ✓ uv detected: $(UV_AVAILABLE)"
	@echo "    Install command: $(INSTALL_CMD)"
else
	@echo "  ✗ uv not found"
endif
ifdef CONDA_PREFIX
	@echo "  ✓ Conda environment detected"
	@echo "    CONDA_PREFIX: $(CONDA_PREFIX)"
ifdef MAMBA_AVAILABLE
	@echo "    Mamba available: $(MAMBA_AVAILABLE)"
else
	@echo "    Mamba: not found"
endif
else
	@echo "  ✗ Not in conda environment"
endif
	@echo "  pip: $(shell which pip 2>/dev/null || echo 'not found')"
	@echo ""

# ===================
# Testing targets
# ===================
test:
	@echo "$(BOLD)$(BLUE)Running all tests...$(RESET)"
	@echo "$(YELLOW)Note: GUI tests excluded from parallel - run 'make test-gui' separately$(RESET)"
	$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ -n auto --dist=loadscope

test-smoke:
	@echo "$(BOLD)$(BLUE)Running smoke tests (~30s)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "smoke" -n auto --dist=loadscope
	@echo "$(BOLD)$(GREEN)✓ Smoke tests passed!$(RESET)"

test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests (excluding slow tests)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "not slow" -n auto --dist=loadscope
	@echo "$(BOLD)$(GREEN)✓ Fast tests passed!$(RESET)"

test-unit:
	@echo "$(BOLD)$(BLUE)Running unit tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "unit" -n auto --dist=loadscope

test-integration:
	@echo "$(BOLD)$(BLUE)Running integration tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "integration"

test-scientific:
	@echo "$(BOLD)$(BLUE)Running scientific validation tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "scientific or numerical or validation"

test-parallel:
	@echo "$(BOLD)$(BLUE)Running tests in parallel (2-4x speedup)...$(RESET)"
	@echo "$(YELLOW)Note: GUI tests excluded - run 'make test-gui' separately$(RESET)"
	$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ -n auto --dist=loadscope

test-ci:
	@echo "$(BOLD)$(BLUE)Running CI test suite (matches GitHub Actions)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "not (slow or gui or stress or flaky)" -n auto --dist=loadscope --durations=10
	@echo "$(BOLD)$(GREEN)✓ CI test suite passed!$(RESET)"

test-full:
	@echo "$(BOLD)$(BLUE)Running comprehensive test suite...$(RESET)"
	$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ -n auto --dist=loadscope
	@echo "$(BOLD)$(GREEN)✓ Full test suite passed!$(RESET)"

test-all:
	@echo "$(BOLD)$(BLUE)Running all tests (parallel + GUI sequential)...$(RESET)"
	@echo "$(CYAN)Step 1/2: Running non-GUI tests in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ -n auto --dist=loadscope
	@echo "$(CYAN)Step 2/2: Running GUI tests sequentially...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/gui_interactive/ -p no:xdist
	@echo "$(BOLD)$(GREEN)✓ All tests passed!$(RESET)"

test-coverage:
	@echo "$(BOLD)$(BLUE)Running tests with coverage report...$(RESET)"
	$(RUN_CMD) $(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

test-gui:
	@echo "$(BOLD)$(BLUE)Running GUI tests (requires display)...$(RESET)"
	@echo "$(YELLOW)Note: GUI tests run sequentially (-p no:xdist) to prevent Qt segfaults$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/gui_interactive/ -s -p no:xdist

test-gui-headless:
	@echo "$(BOLD)$(BLUE)Running GUI tests in headless mode...$(RESET)"
	$(PYTHON) $(TEST_DIR)/gui_interactive/run_gui_tests.py quick --headless

# ===================
# Code quality targets
# ===================
format:
	@echo "$(BOLD)$(BLUE)Formatting code with ruff...$(RESET)"
	$(RUN_CMD) ruff format $(PACKAGE_NAME) $(TEST_DIR)
	$(RUN_CMD) ruff check --fix $(PACKAGE_NAME) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)✓ Code formatted!$(RESET)"

lint:
	@echo "$(BOLD)$(BLUE)Running linting checks...$(RESET)"
	$(RUN_CMD) ruff check $(PACKAGE_NAME) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)✓ No linting errors!$(RESET)"

type-check:
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(RUN_CMD) mypy $(PACKAGE_NAME)
	@echo "$(BOLD)$(GREEN)✓ Type checking passed!$(RESET)"

check: lint type-check
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"

quick: format test-smoke
	@echo "$(BOLD)$(GREEN)✓ Quick iteration complete!$(RESET)"

# ===================
# Documentation targets
# ===================
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && sphinx-build -b html . _build/html
	@echo "$(BOLD)$(GREEN)✓ Documentation built!$(RESET)"
	@echo "Open: $(DOCS_DIR)/_build/html/index.html"

docs-serve:
	@echo "$(BOLD)$(BLUE)Building and serving documentation with auto-reload...$(RESET)"
	cd $(DOCS_DIR) && sphinx-autobuild -b html . _build/html --host 0.0.0.0 --port 8000

docs-clean:
	@echo "$(BOLD)$(BLUE)Cleaning documentation build artifacts...$(RESET)"
	rm -rf $(DOCS_DIR)/_build/

# ===================
# Build and publish targets
# ===================
build: clean-build
	@echo "$(BOLD)$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(BOLD)$(GREEN)✓ Build complete!$(RESET)"
	@echo "Distributions in dist/"

publish: build
	@echo "$(BOLD)$(YELLOW)This will publish $(PACKAGE_NAME) to PyPI!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Publishing to PyPI...$(RESET)"; \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(BOLD)$(GREEN)✓ Published to PyPI!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

publish-test: build
	@echo "$(BOLD)$(BLUE)Publishing to Test PyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(BOLD)$(GREEN)✓ Published to Test PyPI!$(RESET)"

# ===================
# Application targets
# ===================
run-app:
	@echo "$(BOLD)$(BLUE)Launching XPCS Toolkit GUI...$(RESET)"
	$(PYTHON) -m $(PACKAGE_NAME).cli

# ===================
# Cleanup targets
# ===================
clean-build:
	@echo "$(BOLD)$(BLUE)Removing build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true

clean-pyc:
	@echo "$(BOLD)$(BLUE)Removing Python file artifacts...$(RESET)"
	find . -type d -name __pycache__ \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-delete 2>/dev/null || true

clean-test:
	@echo "$(BOLD)$(BLUE)Removing test and coverage artifacts...$(RESET)"
	find . -type d -name .pytest_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hypothesis \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .benchmarks \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage .coverage.* coverage.xml coverage.json
	rm -rf test-artifacts/ test-reports/
	find . -name '*.log' -path './tests/*' -delete 2>/dev/null || true
	find . -name 'test_*.log' -delete 2>/dev/null || true
	find . -name 'test_*.xml' -delete 2>/dev/null || true

clean: clean-build clean-pyc clean-test
	@echo "$(BOLD)$(BLUE)Removing temporary files...$(RESET)"
	find . -name '.DS_Store' -delete 2>/dev/null || true
	find . -name 'Thumbs.db' -delete 2>/dev/null || true
	find . -name '*.tmp' -delete 2>/dev/null || true
	find . -name '*~' -delete 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Cleaned!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/"

clean-all: clean
	@echo "$(BOLD)$(BLUE)Performing deep clean of additional caches...$(RESET)"
	rm -rf .tox/ 2>/dev/null || true
	rm -rf .nox/ 2>/dev/null || true
	rm -rf .eggs/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Deep clean complete!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/"

clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@echo "$(BOLD)You will need to recreate it manually.$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Removing virtual environment...$(RESET)"; \
		rm -rf $(VENV) venv; \
		echo "$(BOLD)$(GREEN)✓ Virtual environment removed!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Utility targets
# ===================
info:
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "===================="
	@echo "Project: $(PACKAGE_NAME)"
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Directory Structure$(RESET)"
	@echo "===================="
	@echo "Source: $(SRC_DIR)/"
	@echo "Tests: $(TEST_DIR)/"
	@echo "Docs: $(DOCS_DIR)/"

version:
	@$(PYTHON) -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)" 2>/dev/null || \
		echo "$(BOLD)$(RED)Error: Package not installed. Run 'make install' first.$(RESET)"

check-deps:
	@echo "$(BOLD)$(BLUE)Checking required dependencies...$(RESET)"
	@$(PYTHON) -c "import sys; print('Python:', sys.version)"
	@$(PYTHON) -c "import numpy; print('NumPy:', numpy.__version__)"
	@$(PYTHON) -c "import scipy; print('SciPy:', scipy.__version__)"
	@$(PYTHON) -c "import h5py; print('h5py:', h5py.version.version)"
	@$(PYTHON) -c "import PySide6; print('PySide6:', PySide6.__version__)"
	@$(PYTHON) -c "import pyqtgraph; print('PyQtGraph:', pyqtgraph.__version__)"
	@$(PYTHON) -c "import pandas; print('Pandas:', pandas.__version__)"
	@$(PYTHON) -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
	@echo "$(BOLD)$(GREEN)✓ Core dependencies OK$(RESET)"

# ===================
# Pre-commit targets
# ===================
pre-commit-install:
	@echo "$(BOLD)$(BLUE)Installing pre-commit hooks...$(RESET)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(BOLD)$(GREEN)✓ Pre-commit hooks installed!$(RESET)"

pre-commit-run:
	@echo "$(BOLD)$(BLUE)Running pre-commit hooks on all files...$(RESET)"
	pre-commit run --all-files
