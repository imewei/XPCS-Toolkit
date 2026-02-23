# XPCS Viewer Documentation Structure

**Last Updated:** February 20, 2026

## Documentation Tree

```
docs/
├── conf.py                           # Sphinx configuration (Furo theme, intersphinx)
├── index.rst                         # Documentation home page
├── STRUCTURE.md                      # This file
├── authors.rst                       # Author information
├── history.rst                       # Changelog
├── contributing.rst                  # Contributing guidelines
├── usage.rst                         # CLI and GUI usage reference
│
├── _static/                          # Static assets
│   ├── xpcsviewer_logo.png           # Project logo
│   └── custom.css                    # Custom CSS (table striping, typography)
│
├── _includes/                        # Shared RST fragments (for .. include::)
│   ├── installation_snippet.rst      # Shared install commands
│   ├── mask_editor_overview.rst      # Shared mask editor description
│   ├── drawing_tools_table.rst       # Shared drawing tools reference table
│   └── geometry_parameters_table.rst # Shared geometry parameters table
│
├── tutorials/                        # Learning-oriented (Diataxis: Tutorials)
│   ├── index.rst                     # Tutorials landing page
│   ├── quickstart.rst                # 5-minute quick start
│   ├── getting_started.rst           # Full walkthrough: load, inspect, plot G2
│   ├── mask_editor.rst               # Mask and Q-map tutorial (Python API)
│   ├── fitting_guide.rst             # NLSQ + Bayesian fitting pipeline
│   ├── backend_selection.rst         # JAX vs NumPy backend tutorial
│   └── cookbook.rst                   # Common patterns and recipes
│
├── how-to/                           # Task-oriented (Diataxis: How-To Guides)
│   ├── index.rst                     # How-to landing page
│   ├── quickstart.rst                # Quick-reference task guide
│   ├── installation.rst              # Installation guide
│   ├── mask_editor_guide.rst         # GUI mask editor reference
│   └── examples.rst                  # Code examples (G2, SAXS, batch)
│
├── api/                              # API Reference (auto-generated)
│   ├── index.rst                     # API overview
│   ├── package.rst                   # XpcsFile main class
│   ├── backends.rst                  # Backend abstraction
│   ├── schemas.rst                   # Validated data structures
│   ├── io.rst                        # HDF5 facade
│   ├── fitting.rst                   # Fitting module
│   ├── modules.rst                   # Analysis modules (G2, SAXS, etc.)
│   ├── simplemask.rst                # SimpleMask module
│   ├── fileio.rst                    # Low-level HDF5 I/O
│   ├── cli.rst                       # CLI entry points
│   ├── plotting.rst                  # Plot handlers
│   ├── threading.rst                 # Threading system
│   ├── utils.rst                     # Utilities and logging
│   ├── constants.rst                 # Application constants
│   └── gui.rst                       # GUI components
│
├── explanation/                      # Understanding-oriented (Diataxis: Explanation)
│   └── index.rst                     # Explanation landing page
│
├── architecture/                     # Technical architecture
│   ├── index.rst                     # Architecture overview
│   ├── ADR-001-jax-migration.md      # ADR: JAX migration
│   ├── ADR-002-nlsq-migration.md     # ADR: NLSQ migration
│   ├── ADR-003-hdf5-facade.md        # ADR: HDF5 facade
│   ├── ADR-004-backend-abstraction.md # ADR: Backend abstraction
│   ├── system_overview.md            # System architecture diagrams
│   ├── data_flow.md                  # Data flow diagrams
│   ├── dependency_analysis.md        # Module dependency analysis
│   ├── dependency_diagram.md         # Dependency diagrams
│   ├── integration_catalog.md        # Integration point catalog
│   └── FACADE_INFRASTRUCTURE.md      # Facade infrastructure details
│
├── operations/                       # Administration-oriented
│   ├── index.rst                     # Operations landing page
│   ├── configuration.rst             # Configuration reference
│   ├── logging.rst                   # Logging setup and env vars
│   └── performance.rst               # Performance tuning
│
└── developer/                        # Retained content (referenced from explanation/)
    └── optimization.rst              # Performance profiling and optimization guide
```

## Diataxis Framework

This documentation follows the [Diataxis](https://diataxis.fr/) framework:

### Tutorials (Learning-Oriented)
- **Purpose:** Teach through guided exercises
- **Audience:** New users learning xpcsviewer
- **Content:** Step-by-step walkthroughs with explanations
- **Directory:** `tutorials/`

### How-To Guides (Task-Oriented)
- **Purpose:** Help accomplish specific tasks
- **Audience:** Users who know what they want to do
- **Content:** Goal-oriented instructions, minimal explanation
- **Directory:** `how-to/`

### API Reference (Information-Oriented)
- **Purpose:** Describe the technical interface
- **Audience:** Developers using the Python API
- **Content:** Auto-generated from docstrings
- **Directory:** `api/`

### Explanation (Understanding-Oriented)
- **Purpose:** Provide background and context
- **Audience:** Users wanting deeper understanding
- **Content:** Conceptual discussion, design rationale
- **Directory:** `explanation/`

## Shared Fragments

The `_includes/` directory contains RST fragments used by multiple pages via
`.. include::` directives. This eliminates content duplication between tutorials
and how-to guides that share reference material (e.g., drawing tools table,
geometry parameters).

## Building Documentation

```bash
cd /path/to/xpcsviewer

# Build HTML
uv run sphinx-build -b html docs docs/_build

# View locally
open docs/_build/index.html
```

## Key Features

- **Furo theme** with custom CSS, light/dark mode, brand colors
- **Intersphinx** linking to Python, NumPy, SciPy, Matplotlib, h5py, JAX, NumPyro, ArviZ
- **MyST Parser** for Markdown support (architecture docs, ADRs)
- **Mermaid diagrams** via sphinxcontrib-mermaid
- **Sphinx Design** for cards and grids
- **Copy button** on code blocks
