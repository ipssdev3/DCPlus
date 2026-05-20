# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --all-groups

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/jax/test_lodf_jax.py

# Run a specific test
uv run pytest tests/jax/test_lodf_jax.py::test_lodf_jax_full_rank_update_compare_powsybl

# Run tests with coverage (>90% required for PRs)
uv run pytest --cov=src --cov-config=.coveragerc --cov-report term-missing

# Lint with ruff (config in ruff.toml, tests excluded from lint)
uv run ruff check --config=ruff.toml
uv run ruff format --config=ruff.toml

# Run all pre-commit hooks
uv run pre-commit run --all-files

# Build/serve docs
uv run mkdocs serve
```

## Architecture

DCPlus is a GPU-accelerated power flow solver implementing "One Step AC" — a linearization around N-0 AC loadflow that approximates network changes via one Newton-Raphson iteration step. Based on the paper "Voltage-sensitive distribution factors for contingency analysis and topology optimization."

### Core Pipeline

1. **Import** (`importing/`) — Load network data from Powsybl, PandaPower, or InterPSS into a unified schema (`import_schema.py` defines validated Pandera DataFrames for buses, branches, injections, shunts)
2. **Interface** (`interfaces/`) — Convert imported data into solver-ready structures: `StaticNetworkInformation` + `DynamicNetworkInformation` (dataclasses with typed arrays). Build the Jacobian matrix via `JacobianInterface`
3. **Preprocess** (`preprocess/`) — Prepare Jacobian data for BSDF (branch shift distribution factor) computations, handling bus splits for N-1 contingencies
4. **Solve** — Two implementations:
   - `jax/` — GPU-accelerated solver using JAX with `jax_dataclasses`. Computes LODF (line outage distribution factors) for branch flows and voltages. Main entry: `lodf_voltages.py` → `lodf_branches.py` → `SolverLoadflowResults`
   - `numpy/` — NumPy reference implementation for validation

### Key Data Flow

```
Network file → import_helpers → Schema DataFrames → network_information → JacobianInterface
    → preprocess_jacobian_bsdf → JAX solver → SolverLoadflowResults (post-contingency V, P, Q, I)
```

### Type System

- Uses `jaxtyping` for array shape annotations (e.g., `Float[Array, " n_outages n_buses"]`)
- `BusType` enum: SLACK=0, PV=1, PQ=2
- All values in per-unit system
- Ruff ignores `F722` to allow jaxtyping forward-reference syntax

## Conventions

- **Python 3.11 only** (strict: `>=3.11,<3.12`)
- **Conventional Commits** required with DCO sign-off (`git commit -s`). Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`
- **Branch strategy**: trunk-based (`main` only). Feature branches: `feat/`, fix branches: `fix/`
- **Docstring style**: NumPy convention (enforced by ruff pydocstyle)
- **License headers**: MPL 2.0 header on all source files (checked in CI via `nwa`)
- **Line length**: 125 characters
- **Notebooks**: Output stripped by pre-commit (`nb-clean` + `nbstripout`)
- **Tests excluded from lint** (`ruff.toml` excludes `tests/**/*` from lint rules; pre-commit also excludes `**/*tests`)

## Test Structure

- `tests/jax/` — JAX solver tests (main solver validation)
- `tests/numpy/` — NumPy reference tests
- `tests/preprocess/` — Jacobian/BSDF preprocessing tests
- `tests/importing/` — Import pipeline tests (powsybl, pandapower, interpss)
- `tests/interfaces/` — Network data interface tests
- `testdata/` — IEEE and PSSE format network files
- Tests have 300s timeout; `--new-first` flag runs recently modified tests first
