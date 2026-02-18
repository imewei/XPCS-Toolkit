# Docs Sprint Design

**Date:** 2026-02-17
**Status:** Approved

## Goal

Comprehensive documentation for xpcsviewer following the Diataxis framework, with reference-first priority.

## Decisions

- **Build on existing Sphinx + Furo** (not MkDocs)
- **Reference-first**: Complete API docs before tutorials
- **4-agent team**: docs-architect, technical-writer, tutorial-builder, architecture-writer

## Team

| Agent | Owns | Deliverables |
|-------|------|-------------|
| docs-architect | docs/ root, conf.py, navigation | Diataxis nav structure, Sphinx extensions, build pipeline |
| technical-writer | docs/api/ | Complete autodoc pages, docstring audit, example validation |
| tutorial-builder | docs/tutorials/, notebooks/ | Step-by-step tutorials, Jupyter notebooks, cookbook |
| architecture-writer | docs/architecture/, docs/operations/ | ADRs, Mermaid diagrams, config reference |

## Standard

Every public API: docstring + reference page + usage example.

## Existing Infrastructure

- Sphinx conf.py with Furo, MyST, autodoc, Napoleon, intersphinx
- Partial user guide, API stubs, architecture docs
- Source: xpcsviewer/ with backends, fitting, schemas, io, simplemask, gui, etc.
