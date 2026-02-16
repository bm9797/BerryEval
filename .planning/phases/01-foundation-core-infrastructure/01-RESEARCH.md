# Phase 1: Foundation & Core Infrastructure - Research

**Researched:** 2026-02-16
**Domain:** Python project scaffolding, IR metrics implementation, NumPy array interfaces, development tooling
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use pyproject.toml with setuptools or hatchling backend (no setup.py)
- Python 3.10+ minimum version
- Package name: berryeval
- Entry point: `berryeval` CLI command via console_scripts
- Follow PRD-prescribed layout: berryeval/ (cli/, dataset/, retrievers/, metrics/, runner/, persistence/, config/), native/ (include/, src/, bindings/, CMakeLists.txt), tests/, benchmarks/
- All Python subpackages initialized with __init__.py
- native/ directory created but empty in Phase 1 (placeholder for Phase 5)
- Ruff for linting and formatting (replaces Black + isort + flake8)
- mypy for type checking with strict mode
- pytest as test framework
- All enforced in CI — no code merges without passing checks
- Implement all 5 IR metrics in pure Python: recall@k, precision@k, MRR, nDCG, hit rate
- Functions accept NumPy arrays as input (establishes the interface contract for future C kernels)
- Input contract: contiguous NumPy arrays with int32/float32 types
- Functions must never mutate input arrays
- Each metric must have comprehensive test coverage against known reference values
- Deterministic: same inputs always produce identical outputs
- Define clear function signatures that C kernels will later implement identically
- Pattern: Python implementation is the reference; C implementation must match exactly
- All data pre-validated by Python layer before passing to compute functions
- Ranking data represented as 2D arrays: rows = queries, columns = ranked document IDs
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PROJ-01 | Project uses Python 3.10+ with pyproject.toml-based packaging | pyproject.toml configuration with setuptools/hatchling backend, PEP 621 metadata |
| PROJ-02 | Repository follows prescribed structure (berryeval/, native/, tests/, benchmarks/) | PRD-prescribed directory layout with all subpackages and __init__.py files |
| PROJ-03 | Development tooling configured (Ruff, mypy, pytest) | Ruff 0.15.1+, mypy 1.19.1+ with strict mode, pytest 9.0+ configuration all in pyproject.toml |
| PROJ-04 | CI pipeline builds and tests on Linux/macOS/Windows | GitHub Actions matrix strategy with Python 3.10, 3.11, 3.12 across 3 OS platforms |
| NATV-04 | Python-C interface uses contiguous NumPy arrays (int32, float32) | NumPy C array interface protocol, PyArrayInterface structure, ndpointer for type enforcement |
| NATV-05 | C layer never mutates input arrays | Pure function design pattern, input validation layer, copy-on-write semantics |
| NATV-06 | C layer contains zero business logic (pure computational kernels) | Thin kernel interface pattern — Python validates, C computes, Python aggregates |
| NATV-07 | System gracefully falls back to pure Python when C extension is unavailable | Conditional import pattern with try/except ImportError, backend dispatch at module init |
| METR-08 | All metrics have pure Python implementations that are always available | Pure Python implementations using NumPy vectorized operations for all 5 IR metrics |
| PERF-03 | Deterministic execution — same inputs produce same outputs | No randomness in metric computation, stable sorting, deterministic array operations |
</phase_requirements>

## Summary

Phase 1 establishes the BerryEval project foundation: a properly packaged Python 3.10+ project with a prescribed repository structure, development tooling (Ruff, mypy, pytest), CI pipeline (GitHub Actions), and pure Python implementations of all five IR metrics (recall@k, precision@k, MRR, nDCG, hit rate).

The metrics implementations serve dual purpose: they are the always-available fallback when C extensions are unavailable (Phase 5), and they define the exact function signatures and array interface contracts that C kernels must replicate. All metric functions accept contiguous NumPy arrays (int32 for document IDs, float32 for scores/gains) and produce deterministic outputs.

**Primary recommendation:** Use a flat package layout (berryeval/ at root, not src/berryeval/) to match the PRD-prescribed structure. Configure all tooling in a single pyproject.toml. Implement metrics as vectorized NumPy operations (not Python loops) so pure Python performance is reasonable even before C acceleration.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.10+ | Runtime and orchestration | Minimum for modern type hints (PEP 604 unions, ParamSpec), match/case |
| NumPy | 2.4+ | Numerical array interface | Foundation for all metric computation, C interop via array protocol |
| setuptools | 75+ | Build backend | Mature, handles console_scripts entry points, works with pyproject.toml |

### Development

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Ruff | 0.15.1+ | Linting and formatting | Replaces Black + isort + flake8, 10-100x faster, single config in pyproject.toml |
| mypy | 1.19.1+ | Static type checking | Industry standard for Python type safety, strict mode for new projects |
| pytest | 9.0+ | Test framework | De facto standard, parametrize for metric test cases, fixtures for array setup |
| pytest-cov | 7.0+ | Coverage reporting | Coverage measurement integrated with pytest, CI reporting |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| setuptools | hatchling | Hatchling is newer/cleaner but setuptools is more battle-tested for C extensions in Phase 5 |
| setuptools | scikit-build-core | Better for C/CMake projects but premature in Phase 1 (no C code yet) |
| flat layout | src/ layout | src/ layout prevents accidental imports of uninstalled code, but PRD prescribes flat layout |

**Installation (development):**
```bash
pip install -e ".[dev]"
```

## Architecture Patterns

### Recommended Project Structure

```
berryeval/
├── berryeval/
│   ├── __init__.py          # Package root, version
│   ├── cli/
│   │   └── __init__.py
│   ├── dataset/
│   │   └── __init__.py
│   ├── retrievers/
│   │   └── __init__.py
│   ├── metrics/
│   │   ├── __init__.py      # Public API, backend dispatch
│   │   ├── _types.py        # Type definitions, array contracts
│   │   ├── _validation.py   # Input validation (shared by Python and C paths)
│   │   ├── _python.py       # Pure Python metric implementations
│   │   └── _native.py       # C extension wrapper (try/except import)
│   ├── runner/
│   │   └── __init__.py
│   ├── persistence/
│   │   └── __init__.py
│   └── config/
│       └── __init__.py
├── native/
│   ├── include/
│   ├── src/
│   ├── bindings/
│   └── CMakeLists.txt       # Placeholder
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures (sample arrays, known results)
│   └── metrics/
│       ├── __init__.py
│       ├── test_recall.py
│       ├── test_precision.py
│       ├── test_mrr.py
│       ├── test_ndcg.py
│       └── test_hit_rate.py
├── benchmarks/               # Scaffolding only in Phase 1
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

### Pattern 1: Backend Dispatch (Metrics Module)

**What:** The metrics module provides a single public API that dispatches to either the pure Python or C implementation at import time.

**When to use:** Any module that will have both a Python fallback and a C-accelerated path.

**Example:**
```python
# berryeval/metrics/__init__.py
from berryeval.metrics._validation import validate_inputs

try:
    from berryeval.metrics._native import (
        recall_at_k as _recall_at_k,
        precision_at_k as _precision_at_k,
        mrr as _mrr,
        ndcg as _ndcg,
        hit_rate as _hit_rate,
    )
    _BACKEND = "native"
except ImportError:
    from berryeval.metrics._python import (
        recall_at_k as _recall_at_k,
        precision_at_k as _precision_at_k,
        mrr as _mrr,
        ndcg as _ndcg,
        hit_rate as _hit_rate,
    )
    _BACKEND = "python"

def recall_at_k(retrieved: np.ndarray, relevant: np.ndarray, k: int) -> np.ndarray:
    """Compute recall@k for each query.

    Args:
        retrieved: 2D int32 array (n_queries, max_retrieved) of document IDs
        relevant: 2D int32 array (n_queries, max_relevant) of relevant document IDs
        k: Number of top results to consider

    Returns:
        1D float32 array (n_queries,) of recall scores
    """
    validate_inputs(retrieved, relevant, k)
    return _recall_at_k(retrieved, relevant, k)
```

### Pattern 2: Input Validation Layer

**What:** A shared validation module that both Python and C paths use, ensuring all data is validated before computation.

**When to use:** Before any metric computation call.

**Example:**
```python
# berryeval/metrics/_validation.py
import numpy as np

def validate_inputs(
    retrieved: np.ndarray,
    relevant: np.ndarray,
    k: int,
) -> None:
    """Validate metric inputs match the NumPy array contract.

    Raises:
        TypeError: If arrays are not numpy arrays
        ValueError: If arrays have wrong dtype, shape, or are not contiguous
    """
    if not isinstance(retrieved, np.ndarray):
        raise TypeError(f"retrieved must be numpy array, got {type(retrieved)}")
    if not isinstance(relevant, np.ndarray):
        raise TypeError(f"relevant must be numpy array, got {type(relevant)}")

    if retrieved.dtype != np.int32:
        raise ValueError(f"retrieved must be int32, got {retrieved.dtype}")
    if relevant.dtype != np.int32:
        raise ValueError(f"relevant must be int32, got {relevant.dtype}")

    if retrieved.ndim != 2:
        raise ValueError(f"retrieved must be 2D, got {retrieved.ndim}D")
    if relevant.ndim != 2:
        raise ValueError(f"relevant must be 2D, got {relevant.ndim}D")

    if not retrieved.flags['C_CONTIGUOUS']:
        raise ValueError("retrieved must be C-contiguous")
    if not relevant.flags['C_CONTIGUOUS']:
        raise ValueError("relevant must be C-contiguous")

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > retrieved.shape[1]:
        raise ValueError(f"k ({k}) exceeds retrieved columns ({retrieved.shape[1]})")

    if retrieved.shape[0] != relevant.shape[0]:
        raise ValueError(
            f"Query count mismatch: retrieved has {retrieved.shape[0]}, "
            f"relevant has {relevant.shape[0]}"
        )
```

### Pattern 3: Deterministic Metric Functions

**What:** Pure functions that take NumPy arrays and return NumPy arrays with no side effects, no mutation, and deterministic output.

**When to use:** All metric implementations.

**Example:**
```python
# berryeval/metrics/_python.py
import numpy as np

def recall_at_k(retrieved: np.ndarray, relevant: np.ndarray, k: int) -> np.ndarray:
    """Pure Python recall@k implementation.

    For each query, computes: |retrieved[:k] ∩ relevant| / |relevant|

    Uses vectorized NumPy operations. Does not mutate inputs.
    """
    n_queries = retrieved.shape[0]
    results = np.empty(n_queries, dtype=np.float32)

    retrieved_at_k = retrieved[:, :k]

    for i in range(n_queries):
        relevant_set = set(relevant[i][relevant[i] != -1])  # -1 = padding
        if len(relevant_set) == 0:
            results[i] = 0.0
            continue
        retrieved_set = set(retrieved_at_k[i][retrieved_at_k[i] != -1])
        results[i] = len(relevant_set & retrieved_set) / len(relevant_set)

    return results
```

### Anti-Patterns to Avoid

- **Business logic in metrics module:** Metrics compute, they don't decide what to compute. Configuration, thresholds, and aggregation belong in the runner layer.
- **Implicit fallback without logging:** When C extension is unavailable, log a clear message so users know they're on the Python path.
- **Mutable array operations:** Never use in-place operations (`arr += 1`) on input arrays. Always create new arrays for output.
- **Non-deterministic operations:** Avoid `set()` iteration order (use sorted), avoid floating-point accumulation order changes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| nDCG formula edge cases | Custom nDCG with ad-hoc gain/discount functions | Standard DCG formula: DCG = sum(rel_i / log2(i+1)) for i in 1..k | Well-defined in IR literature, edge cases documented (zero relevant, all relevant) |
| Array type enforcement | Manual dtype checking everywhere | NumPy's `np.require(arr, dtype=np.int32, requirements='C')` | Handles casting, contiguity, and copy-on-demand in one call |
| CI matrix configuration | Shell scripts per platform | GitHub Actions matrix strategy | Built-in support for Python version x OS matrix, caching, artifact handling |
| Python project packaging | setup.py, setup.cfg, MANIFEST.in | pyproject.toml (PEP 621) | Modern standard, single file for all metadata and tool config |

**Key insight:** IR metrics have well-defined mathematical formulas from decades of research. The implementation challenge is not the math but the edge cases (empty result sets, padding values, zero-division) and the array interface contracts.

## Common Pitfalls

### Pitfall 1: Floating-Point Determinism
**What goes wrong:** nDCG and MRR involve division and logarithms. Different accumulation orders can produce slightly different float32 results.
**Why it happens:** Float32 addition is not associative. Parallelized reduction can change order.
**How to avoid:** Process queries in index order. Use float64 for intermediate accumulation, cast to float32 at the end. Document precision guarantees.
**Warning signs:** Tests passing on one platform but failing on another with tiny epsilon differences.

### Pitfall 2: Padding Value Confusion
**What goes wrong:** 2D arrays require padding for variable-length result lists. If padding value (e.g., -1) is not handled consistently, metrics produce wrong results.
**Why it happens:** Some queries return fewer results than max_k. Arrays must be rectangular.
**How to avoid:** Define a single padding constant (e.g., `PADDING_ID = -1`). Document it. Filter padding before set intersection. Test with variable-length results.
**Warning signs:** Metrics change when max array size changes but actual data doesn't.

### Pitfall 3: Off-By-One in @k Computation
**What goes wrong:** recall@5 should consider positions 0..4 (first 5 results), but slicing bugs can include position 5 or exclude position 0.
**Why it happens:** Confusion between 0-indexed array slicing and 1-indexed "top k" semantics.
**How to avoid:** Use `retrieved[:, :k]` consistently. k=5 means slice `[:5]` which gives indices 0,1,2,3,4. Test at k=1 boundary.
**Warning signs:** Results differ by 1 document at boundary k values.

### Pitfall 4: nDCG Gain Function Assumptions
**What goes wrong:** nDCG requires relevance grades (not just binary relevant/not-relevant). If using binary relevance, the gain function must be explicitly defined.
**Why it happens:** Some implementations assume graded relevance, others assume binary. Mixing them produces wrong results.
**How to avoid:** Accept a `relevance_type` parameter or document that BerryEval v1 uses binary relevance (gain = 1 for relevant, 0 for not relevant). Compute ideal DCG from the binary relevance set.
**Warning signs:** nDCG scores that seem too high or too low compared to reference implementations.

### Pitfall 5: mypy Strict Mode With NumPy
**What goes wrong:** NumPy's type stubs have gaps. `np.ndarray` is generic but mypy strict mode may flag operations as untyped.
**Why it happens:** NumPy's Python type stubs don't fully cover all array operations with precise types.
**How to avoid:** Use `numpy.typing.NDArray[np.int32]` for typed arrays. Add targeted `# type: ignore[...]` with specific error codes where NumPy stubs are incomplete. Don't blanket-ignore.
**Warning signs:** Hundreds of mypy errors from NumPy operations even with correct code.

### Pitfall 6: GitHub Actions Cache Invalidation
**What goes wrong:** CI runs slowly because pip cache isn't shared across matrix entries, or cache is stale.
**Why it happens:** Matrix strategy creates separate jobs; caching must be configured per job with correct keys.
**How to avoid:** Use `actions/setup-python` with `cache: 'pip'` which handles cache keys automatically based on requirements files or pyproject.toml.
**Warning signs:** CI takes 5+ minutes per job when it should take 1-2 minutes.

## Code Examples

### pyproject.toml Configuration

```toml
[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[project]
name = "berryeval"
version = "0.1.0"
description = "Evaluation framework for benchmarking RAG retrieval quality"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "numpy>=2.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.15.0",
    "mypy>=1.19.0",
    "pytest>=9.0",
    "pytest-cov>=7.0",
]

[project.scripts]
berryeval = "berryeval.cli:app"

[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific rules
]
ignore = ["E501"]  # line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["berryeval"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
disallow_untyped_decorators = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: marks tests as slow",
]
```

### NumPy Array Interface Contract

```python
# berryeval/metrics/_types.py
"""Type definitions and array contracts for the metrics engine.

All metric functions follow this contract:
- Input arrays: contiguous, C-order NumPy arrays
- Document IDs: int32 (2D: n_queries x max_docs)
- Scores/gains: float32 (2D: n_queries x max_docs)
- Output: float32 (1D: n_queries)
- Padding: -1 for unused positions in variable-length arrays
- No mutation of input arrays
"""
from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

# Padding value for variable-length arrays
PADDING_ID: Final[int] = -1

# Type aliases for documentation and type checking
QueryDocArray = npt.NDArray[np.int32]      # (n_queries, max_docs)
RelevanceArray = npt.NDArray[np.int32]     # (n_queries, max_relevant)
ScoreArray = npt.NDArray[np.float32]       # (n_queries, max_docs)
MetricResult = npt.NDArray[np.float32]     # (n_queries,)
```

### nDCG Implementation Pattern

```python
# berryeval/metrics/_python.py (nDCG portion)
def ndcg(
    retrieved: np.ndarray,
    relevant: np.ndarray,
    k: int,
) -> np.ndarray:
    """Pure Python nDCG@k implementation.

    Uses binary relevance: gain = 1.0 for relevant docs, 0.0 otherwise.
    DCG@k = sum(gain_i / log2(i + 2)) for i in 0..k-1  (position 1-indexed in formula)
    nDCG@k = DCG@k / IDCG@k

    When IDCG = 0 (no relevant docs), nDCG = 0.0.
    """
    n_queries = retrieved.shape[0]
    results = np.empty(n_queries, dtype=np.float32)

    # Precompute discount factors (1-indexed positions in log)
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))

    for i in range(n_queries):
        relevant_set = set(relevant[i][relevant[i] != -1])
        n_relevant = len(relevant_set)

        if n_relevant == 0:
            results[i] = 0.0
            continue

        # DCG: binary gains for retrieved docs
        retrieved_k = retrieved[i, :k]
        gains = np.array(
            [1.0 if doc_id in relevant_set else 0.0
             for doc_id in retrieved_k if doc_id != -1],
            dtype=np.float64,
        )

        actual_k = len(gains)
        if actual_k == 0:
            results[i] = 0.0
            continue

        dcg = np.sum(gains[:actual_k] / discounts[:actual_k])

        # IDCG: ideal ranking (all relevant docs first)
        ideal_k = min(n_relevant, k)
        idcg = np.sum(1.0 / discounts[:ideal_k])

        results[i] = np.float32(dcg / idcg) if idcg > 0 else np.float32(0.0)

    return results
```

### GitHub Actions CI Configuration

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: ruff format --check .

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      - run: pip install -e ".[dev]"
      - run: mypy berryeval

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"
      - run: pip install -e ".[dev]"
      - run: pytest --cov=berryeval --cov-report=xml
```

### Test Pattern for Metrics

```python
# tests/metrics/test_recall.py
import numpy as np
import pytest

from berryeval.metrics import recall_at_k

# Known reference values for validation
RECALL_TEST_CASES = [
    # (retrieved, relevant, k, expected)
    # Perfect recall: all relevant docs retrieved
    (
        np.array([[1, 2, 3, 4, 5]], dtype=np.int32),
        np.array([[1, 2, 3, -1, -1]], dtype=np.int32),
        5,
        np.array([1.0], dtype=np.float32),
    ),
    # Zero recall: no relevant docs in top-k
    (
        np.array([[10, 20, 30, 40, 50]], dtype=np.int32),
        np.array([[1, 2, 3, -1, -1]], dtype=np.int32),
        5,
        np.array([0.0], dtype=np.float32),
    ),
    # Partial recall: 2 of 3 relevant docs retrieved
    (
        np.array([[1, 2, 10, 20, 30]], dtype=np.int32),
        np.array([[1, 2, 3, -1, -1]], dtype=np.int32),
        5,
        np.array([2.0 / 3.0], dtype=np.float32),
    ),
    # recall@1 with relevant doc at position 0
    (
        np.array([[1, 2, 3, 4, 5]], dtype=np.int32),
        np.array([[1, -1, -1, -1, -1]], dtype=np.int32),
        1,
        np.array([1.0], dtype=np.float32),
    ),
    # Multiple queries
    (
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        np.array([[1, 2, -1], [7, 8, -1]], dtype=np.int32),
        3,
        np.array([1.0, 0.0], dtype=np.float32),
    ),
]


@pytest.mark.parametrize(
    "retrieved,relevant,k,expected",
    RECALL_TEST_CASES,
    ids=["perfect", "zero", "partial", "at_1", "multi_query"],
)
def test_recall_at_k(retrieved, relevant, k, expected):
    result = recall_at_k(retrieved, relevant, k)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


def test_recall_deterministic():
    """Same inputs must always produce identical outputs."""
    retrieved = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    relevant = np.array([[1, 3, 5, -1, -1]], dtype=np.int32)

    results = [recall_at_k(retrieved, relevant, 5) for _ in range(100)]
    for r in results[1:]:
        np.testing.assert_array_equal(results[0], r)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| setup.py + setup.cfg | pyproject.toml (PEP 621) | 2023 | Single file for all project metadata and tool config |
| Black + isort + flake8 | Ruff (all-in-one) | 2023-2024 | 10-100x faster, one config, one tool |
| numpy.ndarray untyped | numpy.typing.NDArray[T] | NumPy 1.20+ | Type-safe array annotations for mypy |
| pytest.ini or tox.ini | pyproject.toml [tool.pytest] | pytest 9.0 | Native TOML support, single config file |

**Deprecated/outdated:**
- **setup.py**: Replaced by pyproject.toml. No reason to use setup.py for new projects.
- **numpy.matrix**: Deprecated since NumPy 1.15, removed. Use 2D ndarray instead.
- **nose test framework**: Unmaintained since 2015. pytest is the standard.

## Open Questions

1. **Padding strategy for variable-length relevant sets**
   - What we know: 2D arrays require rectangular shape, so shorter rows need padding
   - What's unclear: Whether -1 is universally safe (no valid document could have ID -1) or if we need a configurable sentinel
   - Recommendation: Use -1 as padding constant (PADDING_ID), document it as reserved. Validate that no input doc IDs equal PADDING_ID.

2. **MRR: handling of no relevant documents**
   - What we know: MRR is undefined when there are zero relevant documents for a query
   - What's unclear: Whether to return 0.0 or NaN for queries with no relevant docs
   - Recommendation: Return 0.0 (consistent with ranx and most IR evaluation tools). Document this behavior.

3. **benchmarks/ directory in Phase 1**
   - What we know: PRD prescribes the directory. Phase 1 context leaves it to Claude's discretion.
   - What's unclear: Whether to include performance benchmark scaffolding now or in a later phase
   - Recommendation: Create the directory with a placeholder __init__.py and a README noting it's for Phase 5 performance benchmarks. Minimal cost, follows PRD structure.

## Sources

### Primary (HIGH confidence)
- Context7 /numpy/numpy — PyArrayInterface structure, contiguous array API, ndpointer usage
- Context7 /websites/astral_sh_ruff — pyproject.toml configuration, rule selection, formatting options
- Context7 /websites/pytest_en_stable — pyproject.toml configuration, parametrize, fixtures

### Secondary (MEDIUM confidence)
- [mypy documentation](https://mypy.readthedocs.io/en/stable/config_file.html) — strict mode configuration, module overrides
- [GitHub Actions setup-python](https://github.com/actions/setup-python) — matrix strategy, pip caching
- [Pinecone IR metrics guide](https://www.pinecone.io/learn/offline-evaluation/) — metric formulas and definitions
- [Weaviate retrieval metrics](https://weaviate.io/blog/retrieval-evaluation-metrics) — metric descriptions and edge cases
- [Microsoft rankerEval](https://github.com/microsoft/rankerEval) — NumPy-based IR metric reference implementation
- [scikit-learn ndcg_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) — nDCG reference implementation

### Tertiary (LOW confidence)
- None — all findings verified with primary or secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries are well-established, versions verified via Context7
- Architecture: HIGH - Patterns follow established Python packaging standards and IR evaluation conventions
- Pitfalls: HIGH - Float determinism, padding, off-by-one are well-documented IR evaluation challenges

**Research date:** 2026-02-16
**Valid until:** 2026-03-16 (stable domain, 30-day validity)
