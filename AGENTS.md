# Repository Guidelines

## Project Structure & Modules
- `xpcs_toolkit/`: core toolkit (Qt GUI, file I/O, analysis modules, helpers).
- `tests/`: pytest suites with markers (unit, integration, scientific, gui, performance, etc.).
- `docs/`: Sphinx sources; assets in `docs/images/`, UI grabs in `screenshots/`.
- `scripts/` and `validation/`: utilities/benchmarks and validation helpers; change only when needed.
- Packaging: entry points `xpcs-toolkit`, `pyxpcsviewer`, `run_viewer` defined in `pyproject.toml`.

## Build, Test, and Development Commands
- `make dev-setup` — installs editable deps with docs/test extras; prefers `uv sync --dev --extra docs --extra test` (falls back to pip).
- `make test-fast` — `pytest -m "unit and not slow" -v --tb=short` for quick feedback.
- `make test-core` — unit plus key integration tests.
- `make lint-ruff` / `make format-ruff` — lint or format via Ruff (critical errors fixed).
- `make docs-build` — build Sphinx HTML into `_build/html`.
- `make clean` — clear build, cache, coverage artifacts.

## Coding Style & Naming
- Python >=3.12 required; prefer `uv` as package manager for reproducible envs. Ruff enforced (line length 88); Black-compatible formatting.
- Type hints expected; mypy runs in strict mode. Prefer explicit returns and dataclasses where sensible.
- Naming: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants. Tests follow `test_<unit>.py` with `Test*` classes.
- Keep generated GUI resource files (`*_ui.py`, `icons_rc.py`) untouched; regenerate via Qt tools instead of manual edits.

## Testing Guidelines
- Pytest configured with coverage; reports in `htmlcov/` and `coverage.xml`. Global threshold is lenient (11%)—raise coverage for new/changed code.
- Use markers to scope runs, e.g. `pytest -m "smoke"`, `pytest -m "unit or integration"`. GUI/display tests (`-m gui`/`requires_display`) need a display or XVFB.
- Place regressions alongside relevant modules; property-based tests live under `tests/*/properties/`.

## Commit & Pull Request Guidelines
- Commit style follows Conventional Commit patterns seen in history (`feat:`, `fix:`, `chore:`, scoped like `fix(ci)`). Keep commits focused.
- In PRs, include: summary, testing commands run, linked issues (`#id`), and note risk areas. For UI/plot changes, add updated screenshots in `screenshots/` or describe visual impact.
- Update docs when user-facing behavior changes; align config references to Python >=3.12 and uv preference.
