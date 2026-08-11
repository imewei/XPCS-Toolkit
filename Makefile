# XPCSViewer Package Makefile
# ==============================
# Development Tools and Testing

.PHONY: help install install-dev env-info \
        test test-smoke test-fast test-ci test-full test-all test-coverage test-integration \
        test-parallel test-unit test-scientific test-gui test-gui-headless \
        clean clean-all clean-pyc clean-build clean-test clean-venv docs-clean \
        format lint type-check check quick docs docs-serve build publish publish-test \
        info version run-app check-deps verify verify-fast \
        install-hooks pre-commit-install pre-commit-run \
        install-jax-gpu install-jax-gpu-cuda12 install-jax-gpu-cuda13 gpu-check gpu-diagnose \
        _jax-gpu-install

# Configuration
PYTHON := python
PYTEST := pytest
PACKAGE_NAME := xpcsviewer
SRC_DIR := xpcsviewer
TEST_DIR := tests
DOCS_DIR := docs
VENV := .venv

# Common find exclusions (protects .venv, venv, .claude directories)
FIND_PRUNE := -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.claude/*"

# Parallel test options
PARALLEL_OPTS := -n auto --dist=loadscope

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
    UNINSTALL_CMD := uv pip uninstall -y
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
    UNINSTALL_CMD := pip uninstall -y
    SYNC_CMD := pip install -e
    RUN_CMD :=
else
    PKG_MANAGER := pip
    PIP := pip
    INSTALL_CMD := pip install
    UNINSTALL_CMD := pip uninstall -y
    SYNC_CMD := pip install -e
    RUN_CMD :=
endif

# All JAX/CUDA packages that must be removed before a clean install
JAX_ALL_PKGS := jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt

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
# Help
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
	@echo "$(BOLD)$(GREEN)PRE-PUSH VERIFICATION$(RESET)"
	@echo "  $(CYAN)verify$(RESET)           Run FULL local CI (lint + type + tests) - use before push"
	@echo "  $(CYAN)verify-fast$(RESET)      Quick verification (lint + type + fast tests)"
	@echo "  $(CYAN)install-hooks$(RESET)    Install pre-push hook to auto-run verify"
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
	@echo "$(BOLD)$(GREEN)GPU ACCELERATION (System CUDA)$(RESET)"
	@echo "  $(CYAN)install-jax-gpu$(RESET)         Auto-detect system CUDA and install JAX (Linux only)"
	@echo "  $(CYAN)install-jax-gpu-cuda13$(RESET)  Install JAX with system CUDA 13 (requires CUDA 13.x installed)"
	@echo "  $(CYAN)install-jax-gpu-cuda12$(RESET)  Install JAX with system CUDA 12 (requires CUDA 12.x installed)"
	@echo "  $(CYAN)gpu-check$(RESET)               Verify GPU backend, devices, SVD"
	@echo "  $(CYAN)gpu-diagnose$(RESET)            Diagnose common GPU issues (plugins, versions)"
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
# Installation
# ===================
install:
	@echo "$(BOLD)$(BLUE)Installing $(PACKAGE_NAME) in editable mode...$(RESET)"
ifdef UV_AVAILABLE
	@$(SYNC_CMD)
else
	@$(INSTALL_CMD) -e .
endif
	@echo "$(BOLD)$(GREEN)Done: Package installed$(RESET)"

install-dev: install
	@echo "$(BOLD)$(BLUE)Installing development dependencies...$(RESET)"
ifdef UV_AVAILABLE
	@$(SYNC_CMD) --dev
else
	@$(INSTALL_CMD) -e .
endif
	@pre-commit install 2>/dev/null || echo "pre-commit not available, skipping hook installation"
	@echo "$(BOLD)$(GREEN)Done: Dev dependencies installed$(RESET)"

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
	@echo "  uv: $(UV_AVAILABLE)"
	@echo "    Install command: $(INSTALL_CMD)"
else
	@echo "  uv: not found"
endif
ifdef CONDA_PREFIX
	@echo "  Conda: $(CONDA_PREFIX)"
ifdef MAMBA_AVAILABLE
	@echo "  Mamba: $(MAMBA_AVAILABLE)"
endif
else
	@echo "  Conda: not active"
endif
	@echo "  pip: $(shell which pip 2>/dev/null || echo 'not found')"
	@echo ""
	@$(MAKE) gpu-diagnose

# ===================
# Testing
# ===================

# Core parallel test runner (test, test-parallel, test-full are identical)
test test-parallel test-full:
	@echo "$(BOLD)$(BLUE)Running tests (parallel, excl. GUI)...$(RESET)"
	@echo "$(YELLOW)Note: GUI tests excluded - run 'make test-gui' separately$(RESET)"
	$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ $(PARALLEL_OPTS)
	@echo "$(BOLD)$(GREEN)Done: Tests passed$(RESET)"

test-smoke:
	@echo "$(BOLD)$(BLUE)Running smoke tests (~30s)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "smoke" $(PARALLEL_OPTS)
	@echo "$(BOLD)$(GREEN)Done: Smoke tests passed$(RESET)"

test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests (excluding slow)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "not slow" $(PARALLEL_OPTS)
	@echo "$(BOLD)$(GREEN)Done: Fast tests passed$(RESET)"

test-unit:
	@echo "$(BOLD)$(BLUE)Running unit tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "unit" $(PARALLEL_OPTS)

test-integration:
	@echo "$(BOLD)$(BLUE)Running integration tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "integration"

test-scientific:
	@echo "$(BOLD)$(BLUE)Running scientific validation tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "scientific or numerical or validation"

test-ci:
	@echo "$(BOLD)$(BLUE)Running CI test suite...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m "not (slow or gui or stress or flaky)" $(PARALLEL_OPTS) --durations=10
	@echo "$(BOLD)$(GREEN)Done: CI suite passed$(RESET)"

test-all:
	@echo "$(BOLD)$(BLUE)Running all tests (parallel + GUI sequential)...$(RESET)"
	@echo "$(CYAN)Step 1/2: Non-GUI tests (parallel)...$(RESET)"
	$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ $(PARALLEL_OPTS)
	@echo "$(CYAN)Step 2/2: GUI tests (sequential)...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/gui_interactive/ -p no:xdist
	@echo "$(BOLD)$(GREEN)Done: All tests passed$(RESET)"

test-coverage:
	@echo "$(BOLD)$(BLUE)Running tests with coverage...$(RESET)"
	$(RUN_CMD) $(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)Done: Coverage report at htmlcov/index.html$(RESET)"

test-gui:
	@echo "$(BOLD)$(BLUE)Running GUI tests (requires display)...$(RESET)"
	@echo "$(YELLOW)Note: GUI tests run sequentially (-p no:xdist) to prevent Qt segfaults$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/gui_interactive/ -s -p no:xdist

test-gui-headless:
	@echo "$(BOLD)$(BLUE)Running GUI tests in headless mode...$(RESET)"
	$(PYTHON) $(TEST_DIR)/gui_interactive/run_gui_tests.py quick --headless

# ===================
# Code quality
# ===================
format:
	@echo "$(BOLD)$(BLUE)Formatting code...$(RESET)"
	$(RUN_CMD) ruff format $(PACKAGE_NAME) $(TEST_DIR)
	$(RUN_CMD) ruff check --fix $(PACKAGE_NAME) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)Done: Code formatted$(RESET)"

lint:
	@echo "$(BOLD)$(BLUE)Running linting...$(RESET)"
	$(RUN_CMD) ruff check $(PACKAGE_NAME) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)Done: No lint errors$(RESET)"

type-check:
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(RUN_CMD) mypy $(PACKAGE_NAME)
	@echo "$(BOLD)$(GREEN)Done: Type checks passed$(RESET)"

check: lint type-check
	@echo "$(BOLD)$(GREEN)Done: All checks passed$(RESET)"

quick: format test-smoke
	@echo "$(BOLD)$(GREEN)Done: Quick iteration complete$(RESET)"

# ===================
# Documentation
# ===================
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && sphinx-build -b html . _build/html
	@echo "$(BOLD)$(GREEN)Done: $(DOCS_DIR)/_build/html/index.html$(RESET)"

docs-serve:
	@echo "$(BOLD)$(BLUE)Serving docs with auto-reload...$(RESET)"
	cd $(DOCS_DIR) && sphinx-autobuild -b html . _build/html --host 0.0.0.0 --port 8000

docs-clean:
	rm -rf $(DOCS_DIR)/_build/

# ===================
# Build and publish
# ===================
build: clean-build
	@echo "$(BOLD)$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(BOLD)$(GREEN)Done: Distributions in dist/$(RESET)"

publish: build
	@echo "$(BOLD)$(YELLOW)This will publish $(PACKAGE_NAME) to PyPI!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(BOLD)$(GREEN)Done: Published to PyPI$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

publish-test: build
	@echo "$(BOLD)$(BLUE)Publishing to Test PyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(BOLD)$(GREEN)Done: Published to Test PyPI$(RESET)"

# ===================
# Application
# ===================
run-app:
	@echo "$(BOLD)$(BLUE)Launching XPCS Toolkit GUI...$(RESET)"
	$(PYTHON) -m $(PACKAGE_NAME).cli

# ===================
# Cleanup
# ===================
clean-build:
	@echo "$(BOLD)$(BLUE)Removing build artifacts...$(RESET)"
	rm -rf build/ dist/ *.egg-info
	find . -type d \( -name "*.egg-info" -o -name "*.egg" \) \
		$(FIND_PRUNE) -exec rm -rf {} + 2>/dev/null || true

clean-pyc:
	@echo "$(BOLD)$(BLUE)Removing Python file artifacts...$(RESET)"
	find . -type d -name __pycache__ \
		$(FIND_PRUNE) -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) \
		$(FIND_PRUNE) -delete 2>/dev/null || true

clean-test:
	@echo "$(BOLD)$(BLUE)Removing test and coverage artifacts...$(RESET)"
	find . -type d \( -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache \
		-o -name htmlcov -o -name .hypothesis -o -name .benchmarks -o -name .nlsq_cache \) \
		$(FIND_PRUNE) -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage .coverage.* coverage.xml coverage.json test-artifacts/ test-reports/
	find . \( -name '*.log' -path './tests/*' -o -name 'test_*.log' -o -name 'test_*.xml' \) \
		-delete 2>/dev/null || true

clean: clean-build clean-pyc clean-test
	@echo "$(BOLD)$(BLUE)Removing temporary files...$(RESET)"
	find . \( -name '.DS_Store' -o -name 'Thumbs.db' -o -name '*.tmp' -o -name '*~' \) \
		-delete 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)Done: Cleaned (preserved .venv/, venv/, .claude/)$(RESET)"

clean-all: clean
	@echo "$(BOLD)$(BLUE)Deep clean...$(RESET)"
	rm -rf .tox/ .nox/ .eggs/ .cache/ 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)Done: Deep clean complete$(RESET)"

clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(VENV) venv; \
		echo "$(BOLD)$(GREEN)Done: Virtual environment removed$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Utility
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
	@echo "  Source: $(SRC_DIR)/  Tests: $(TEST_DIR)/  Docs: $(DOCS_DIR)/"

version:
	@$(PYTHON) -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)" 2>/dev/null || \
		echo "$(RED)Error: Package not installed. Run 'make install' first.$(RESET)"

check-deps:
	@echo "$(BOLD)$(BLUE)Checking dependencies...$(RESET)"
	@$(PYTHON) -c "\
	import sys; print(f'Python: {sys.version}'); \
	import numpy; print(f'NumPy: {numpy.__version__}'); \
	import scipy; print(f'SciPy: {scipy.__version__}'); \
	import h5py; print(f'h5py: {h5py.version.version}'); \
	import PySide6; print(f'PySide6: {PySide6.__version__}'); \
	import pyqtgraph; print(f'PyQtGraph: {pyqtgraph.__version__}'); \
	import pandas; print(f'Pandas: {pandas.__version__}'); \
	import matplotlib; print(f'Matplotlib: {matplotlib.__version__}'); \
	import jax; print(f'JAX: {jax.__version__}'); \
	import numpyro; print(f'NumPyro: {numpyro.__version__}'); \
	import arviz; print(f'ArviZ: {arviz.__version__}'); \
	import optimistix; print(f'Optimistix: {optimistix.__version__}'); \
	print('All dependencies OK')"

# ===================
# Pre-commit
# ===================
install-hooks pre-commit-install:
	@echo "$(BOLD)$(BLUE)Installing git hooks...$(RESET)"
	@pre-commit install
	@pre-commit install --hook-type commit-msg
	@rm -f .git/hooks/pre-push
	@echo "$(BOLD)$(GREEN)Done: Git hooks installed$(RESET)"
	@echo "  pre-commit: lint, format, type checks"
	@echo "  commit-msg: conventional commits check"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  git commit -m 'msg'  -> runs pre-commit hooks"
	@echo "  git push             -> triggers GitHub Actions CI"
	@echo "  make verify-fast     -> full local verification (optional)"

pre-commit-run:
	@echo "$(BOLD)$(BLUE)Running pre-commit on all files...$(RESET)"
	pre-commit run --all-files

# ===================
# Pre-push verification
# ===================
verify:
	@echo "$(BOLD)$(BLUE)====== FULL LOCAL CI VERIFICATION ======$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 1/3: Pre-commit hooks$(RESET)"
	@pre-commit run --all-files || (echo "$(RED)Pre-commit failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)Step 2/3: Type checking$(RESET)"
	@$(RUN_CMD) mypy $(PACKAGE_NAME) || (echo "$(RED)Type check failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)Step 3/3: Full test suite (excl. GUI & GPU)$(RESET)"
	@$(RUN_CMD) $(PYTEST) --ignore=$(TEST_DIR)/gui_interactive/ -m "not gpu" $(PARALLEL_OPTS) || (echo "$(RED)Tests failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)$(GREEN)====== ALL CHECKS PASSED - SAFE TO PUSH ======$(RESET)"

verify-fast:
	@echo "$(BOLD)$(BLUE)====== QUICK LOCAL CI VERIFICATION ======$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 1/3: Pre-commit hooks$(RESET)"
	@pre-commit run --all-files || (echo "$(RED)Pre-commit failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)Step 2/3: Type checking$(RESET)"
	@$(RUN_CMD) mypy $(PACKAGE_NAME) || (echo "$(RED)Type check failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)Step 3/3: Fast tests$(RESET)"
	@$(RUN_CMD) $(PYTEST) -m "not slow" $(PARALLEL_OPTS) -q || (echo "$(RED)Tests failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)$(GREEN)====== QUICK CHECKS PASSED ======$(RESET)"

# ===================
# GPU Acceleration (System CUDA)
# ===================

# Internal: Validate system CUDA + GPU, then install JAX
# Called with: $(MAKE) _jax-gpu-install CUDA_VER=<12|13> MIN_SM=<52|75> MIN_SM_DISP=<5.2|7.5> JAX_PKG=<pkg>
_jax-gpu-install:
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
ifeq ($(PLATFORM),linux)
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found - CUDA toolkit not installed or not in PATH$(RESET)"; \
		echo ""; \
		echo "Install CUDA toolkit:"; \
		echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"; \
		echo "  Or download: https://developer.nvidia.com/cuda-downloads"; \
		echo ""; \
		echo "After installation, ensure nvcc is in PATH:"; \
		echo "  export PATH=/usr/local/cuda/bin:\$$PATH"; \
		exit 1; \
	fi; \
	if [ "$$CUDA_VERSION" != "$(CUDA_VER)" ]; then \
		echo "$(RED)Error: System CUDA $$CUDA_FULL detected, but CUDA $(CUDA_VER).x required$(RESET)"; \
		echo "Either:"; \
		echo "  1. Install CUDA $(CUDA_VER).x toolkit"; \
		echo "  2. Use: make install-jax-gpu (auto-detect your CUDA version)"; \
		exit 1; \
	fi; \
	echo "System CUDA: $$CUDA_FULL"; \
	\
	SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	if [ -z "$$SM_VERSION" ]; then \
		echo "$(RED)Error: Could not detect GPU (nvidia-smi failed)$(RESET)"; \
		exit 1; \
	fi; \
	if [ "$$SM_VERSION" -lt $(MIN_SM) ]; then \
		echo "$(RED)Error: GPU $$GPU_NAME (SM $$SM_DISPLAY) requires SM >= $(MIN_SM_DISP) for CUDA $(CUDA_VER)$(RESET)"; \
		exit 1; \
	fi; \
	echo "GPU: $$GPU_NAME (SM $$SM_DISPLAY) - compatible with CUDA $(CUDA_VER)"
	@echo ""
	@echo "Step 1/3: Removing all existing JAX/CUDA packages..."
	@$(UNINSTALL_CMD) $(JAX_ALL_PKGS) 2>/dev/null || true
	@echo ""
	@echo "Step 2/3: Restoring jax/jaxlib from project lockfile..."
	@$(SYNC_CMD)
	@echo ""
	@echo "Step 3/3: Installing matching CUDA $(CUDA_VER) plugin (system CUDA)..."
	@JAXLIB_VER=$$($(RUN_CMD) python -c "import jaxlib; print(jaxlib.__version__)"); \
	echo "Command: $(INSTALL_CMD) jax-cuda$(CUDA_VER)-plugin==$$JAXLIB_VER jax-cuda$(CUDA_VER)-pjrt==$$JAXLIB_VER"; \
	$(INSTALL_CMD) jax-cuda$(CUDA_VER)-plugin==$$JAXLIB_VER jax-cuda$(CUDA_VER)-pjrt==$$JAXLIB_VER
	@echo ""
	@$(MAKE) gpu-check
	@echo ""
	@echo "$(BOLD)$(GREEN)JAX GPU support installed successfully$(RESET)"
	@echo "  Uses: System CUDA $(CUDA_VER).x installation"
else
	@echo "$(RED)Error: GPU acceleration only available on Linux$(RESET)"
	@echo "  Current platform: $(PLATFORM)"
	@echo ""
	@echo "Platform support:"
	@echo "  - Linux + NVIDIA GPU + System CUDA: Full GPU acceleration"
	@echo "  - Windows WSL2: Experimental (use Linux wheels)"
	@echo "  - macOS: CPU-only (no NVIDIA GPU support)"
	@echo "  - Windows native: CPU-only (no pre-built wheels)"
endif

install-jax-gpu:
	@echo "$(BOLD)$(BLUE)Installing JAX with GPU support (system CUDA auto-detect)...$(RESET)"
	@echo "============================================================"
ifeq ($(PLATFORM),linux)
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found - CUDA toolkit not installed or not in PATH$(RESET)"; \
		exit 1; \
	fi; \
	echo "Detected system CUDA: $$CUDA_FULL (major: $$CUDA_VERSION)"; \
	echo ""; \
	if [ "$$CUDA_VERSION" = "13" ]; then \
		$(MAKE) _jax-gpu-install CUDA_VER=13 MIN_SM=75 MIN_SM_DISP=7.5; \
	elif [ "$$CUDA_VERSION" = "12" ]; then \
		$(MAKE) _jax-gpu-install CUDA_VER=12 MIN_SM=52 MIN_SM_DISP=5.2; \
	else \
		echo "$(RED)Error: CUDA $$CUDA_VERSION not supported by JAX 0.8+$(RESET)"; \
		echo "JAX requires CUDA 12.x or 13.x"; \
		exit 1; \
	fi
else
	@echo "$(YELLOW)Error: GPU acceleration only available on Linux$(RESET)"
	@echo "  Current platform: $(PLATFORM)"
	@echo "  Keeping CPU-only installation"
endif

install-jax-gpu-cuda13:
	@echo "$(BOLD)$(BLUE)Installing JAX with system CUDA 13...$(RESET)"
	@echo "======================================"
	@$(MAKE) _jax-gpu-install CUDA_VER=13 MIN_SM=75 MIN_SM_DISP=7.5

install-jax-gpu-cuda12:
	@echo "$(BOLD)$(BLUE)Installing JAX with system CUDA 12...$(RESET)"
	@echo "======================================"
	@$(MAKE) _jax-gpu-install CUDA_VER=12 MIN_SM=52 MIN_SM_DISP=5.2

# GPU verification (backend + devices + SVD compute check)
gpu-check:
	@echo "$(BOLD)$(BLUE)GPU Verification$(RESET)"
	@echo "================"
	@$(PYTHON) -c "\
	import sys, jax, jax.numpy as jnp; \
	v = jax.__version__; \
	b = jax.default_backend(); \
	d = jax.devices(); \
	gpu = sum(1 for x in d if 'cuda' in str(x).lower()); \
	print(f'JAX {v}  backend={b}  devices={gpu} GPU'); \
	(b == 'gpu') or (print('WARNING: Not using GPU'), sys.exit(1)); \
	s = jnp.linalg.svd(jnp.eye(3))[1]; \
	print(f'SVD check: {s}'); \
	print('All checks passed')"

# GPU diagnostics (plugin conflicts, version mismatches, system CUDA)
gpu-diagnose:
	@echo "$(BOLD)$(BLUE)GPU Diagnostics$(RESET)"
	@echo "==============="
	@echo ""
	@echo "1. Installed JAX/CUDA packages:"
	@$(PYTHON) -m pip list 2>/dev/null | grep -iE "^(jax|cuda)" || echo "  (none found)"
	@echo ""
	@echo "2. Plugin conflict check:"
	@HAS12=$$($(PYTHON) -m pip list 2>/dev/null | grep -c "jax-cuda12-plugin"); \
	HAS13=$$($(PYTHON) -m pip list 2>/dev/null | grep -c "jax-cuda13-plugin"); \
	if [ "$$HAS12" -gt 0 ] && [ "$$HAS13" -gt 0 ]; then \
		echo "  $(RED)CONFLICT: Both cuda12 and cuda13 plugins installed!$(RESET)"; \
		echo "  Fix: make install-jax-gpu (will clean and reinstall)"; \
	elif [ "$$HAS12" -gt 0 ] || [ "$$HAS13" -gt 0 ]; then \
		echo "  $(GREEN)OK: Single plugin set installed$(RESET)"; \
	else \
		echo "  No CUDA plugins installed (CPU-only mode)"; \
	fi
	@echo ""
	@echo "3. Version match check:"
	@$(PYTHON) -c "\
	import importlib.metadata as md; \
	jaxlib_v = md.version('jaxlib'); \
	print(f'  jaxlib: {jaxlib_v}'); \
	for pkg in ['jax-cuda12-plugin','jax-cuda13-plugin','jax-cuda12-pjrt','jax-cuda13-pjrt']: \
	    try: \
	        v = md.version(pkg); \
	        match = 'OK' if v == jaxlib_v else 'MISMATCH'; \
	        print(f'  {pkg}: {v} [{match}]'); \
	    except md.PackageNotFoundError: pass" 2>/dev/null || echo "  (could not check)"
	@echo ""
	@echo "4. System CUDA:"
	@nvcc --version 2>/dev/null | grep "release" || echo "  nvcc not found"
	@echo ""
	@echo "5. GPU hardware:"
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null \
		|| echo "  nvidia-smi not found"
