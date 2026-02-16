# Phase 1: Foundation & Core Infrastructure - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish project infrastructure with Python 3.10+ packaging, prescribed repository structure, development tooling (Ruff, mypy, pytest), CI pipeline, and pure Python implementations of all IR metrics (recall@k, precision@k, MRR, nDCG, hit rate). Define NumPy array interface contracts for future C integration. Ensure deterministic execution.

</domain>

<decisions>
## Implementation Decisions

### Project packaging
- Use pyproject.toml with setuptools or hatchling backend (no setup.py)
- Python 3.10+ minimum version
- Package name: berryeval
- Entry point: `berryeval` CLI command via console_scripts

### Repository structure
- Follow PRD-prescribed layout: berryeval/ (cli/, dataset/, retrievers/, metrics/, runner/, persistence/, config/), native/ (include/, src/, bindings/, CMakeLists.txt), tests/, benchmarks/
- All Python subpackages initialized with __init__.py
- native/ directory created but empty in Phase 1 (placeholder for Phase 5)

### Development tooling
- Ruff for linting and formatting (replaces Black + isort + flake8)
- mypy for type checking with strict mode
- pytest as test framework
- All enforced in CI — no code merges without passing checks

### Pure Python metrics
- Implement all 5 IR metrics in pure Python: recall@k, precision@k, MRR, nDCG, hit rate
- Functions accept NumPy arrays as input (establishes the interface contract for future C kernels)
- Input contract: contiguous NumPy arrays with int32/float32 types
- Functions must never mutate input arrays
- Each metric must have comprehensive test coverage against known reference values
- Deterministic: same inputs always produce identical outputs

### NumPy interface contracts
- Define clear function signatures that C kernels will later implement identically
- Pattern: Python implementation is the reference; C implementation must match exactly
- All data pre-validated by Python layer before passing to compute functions
- Ranking data represented as 2D arrays: rows = queries, columns = ranked document IDs

### CI pipeline
- GitHub Actions for Linux, macOS, Windows
- Matrix: Python 3.10, 3.11, 3.12
- Steps: lint (Ruff), type check (mypy), test (pytest)
- Phase 1 CI does not build C extensions (that's Phase 5)

### Claude's Discretion
- Exact Ruff rule configuration
- mypy strictness level details
- Test fixture design and organization
- Whether to use src/ layout or flat layout (both are valid with pyproject.toml)
- Specific GitHub Actions workflow structure
- Whether to include benchmarks/ scaffolding in Phase 1 or defer

</decisions>

<specifics>
## Specific Ideas

- PRD specifies the exact repository structure — follow it precisely
- The "system must remain fully functional in pure Python mode" principle means Phase 1 IS the full metrics engine (just without C acceleration)
- Configuration hash for deterministic execution should be established here so dataset generation in Phase 2 can use it
- Reference values for metric tests should come from established IR evaluation literature (known query-document relevance pairs with pre-computed metrics)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation-core-infrastructure*
*Context gathered: 2026-02-16*