# Repository Guidelines

## Project Structure & Modules
- `xpcs_toolkit/`: core toolkit code (Qt GUI, data I/O, analysis modules, helpers).
- `tests/`: pytest suites; markers cover unit, integration, scientific, gui, performance.
- `docs/`: Sphinx sources; images live in `docs/images/` and `screenshots/` for UI grabs.
- `scripts/` and `validation/`: utility/benchmarking and validation helpers; avoid modifying unless needed.
- Packaging: `pyproject.toml` defines entry points `xpcs-toolkit`, `pyxpcsviewer`, `run_viewer`.

## Build, Test, and Development Commands
- `make dev-setup` — install editable package with `[dev,test,docs]` extras and set up pre-commit hooks.
- `make test-fast` — quick feedback: `pytest -m "unit and not slow" -v --tb=short`.
- `make test-core` — unit + key integration coverage.
- `make lint-ruff` / `make format-ruff` — lint or auto-format with Ruff; fixes critical errors by default.
- `make docs-build` — build Sphinx HTML into `_build/html` (open with a browser).
- `make clean` — remove build/test/coverage caches; use before release packaging.

## Coding Style & Naming
- Python >=3.12 (project requires 3.12+); Ruff enforced (line length 88; see `pyproject.toml`). Black-compatible formatting.
- Type hints are expected; mypy runs in strict mode. Prefer dataclasses and explicit returns.
- Naming: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants. Tests follow `test_<unit>.py` with `Test*` classes.
- Keep GUI resource files generated; avoid hand-editing `*_ui.py` or `icons_rc.py`.

## Environment & Tooling
- Preferred package manager: `uv` (fast resolver); use `uv pip install -e .[dev,test,docs]` as an alternative to pip.

## Testing Guidelines
- Default suite uses pytest with coverage (fail-under 11% global; aim higher for new code). HTML/XML reports land in `htmlcov/` and `coverage.xml`.
- Use markers to scope runs, e.g. `pytest -m "smoke"` or `pytest -m "unit or integration"`. GUI tests (`-m gui`/`requires_display`) need a display or XVFB.
- Add regression cases near existing modules; property tests live under `tests/*/properties/`.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style when possible (`feat:`, `fix:`, `chore:`, scopes like `ci`, `docs`). Recent history shows scoped tags (e.g., `fix(ci)`), so keep consistency.
- Commits should be focused; include test command output in PR description. Link related issues (`#id`) and note risk areas.
- For UI/plot changes, attach updated screenshots (`screenshots/`) or describe visual impact. Update docs if user-facing behavior shifts.
