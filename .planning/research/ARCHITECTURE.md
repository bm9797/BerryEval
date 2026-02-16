# Architecture Research

**Domain:** RAG Evaluation Framework with Python CLI and C Acceleration
**Researched:** 2026-02-16
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLI Layer (Click/Typer)                         │
│                    Entry points, argument parsing, config                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│   │   Dataset    │  │  Retriever   │  │   Runner     │                 │
│   │  Generator   │  │   Adapters   │  │ Orchestrator │                 │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│          │                  │                  │                         │
│          └──────────────────┴──────────────────┘                         │
│                              │                                           │
├──────────────────────────────┼───────────────────────────────────────────┤
│      Python Orchestration Layer (Pure Python - Always Available)         │
│                              │                                           │
│   ┌──────────────────────────┴──────────────────────────────┐           │
│   │            Metrics Engine (Python Wrapper)               │           │
│   │  • Dispatches to C or Python implementation             │           │
│   │  • NumPy array interface (py::array_t<double>)          │           │
│   │  • Fallback detection: try C import, except pure Python │           │
│   └──────────────┬────────────────────┬──────────────────────┘           │
│                  │                    │                                  │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  C Kernels       │  Pure Python       │    Persistence/Config            │
│  (pybind11)      │  Fallback          │                                  │
│                  │                    │                                  │
│  ┌─────────────┐ │  ┌─────────────┐  │  ┌──────────┐  ┌──────────┐     │
│  │ NDCG kernel │ │  │ NDCG (py)   │  │  │ Results  │  │ Config   │     │
│  │ MRR kernel  │ │  │ MRR (py)    │  │  │ Storage  │  │ Manager  │     │
│  │ Recall kern │ │  │ Recall (py) │  │  │          │  │          │     │
│  └─────────────┘ │  └─────────────┘  │  └──────────┘  └──────────┘     │
│                  │                    │                                  │
│  OPTIONAL        │  ALWAYS AVAILABLE  │                                  │
└──────────────────┴────────────────────┴──────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **CLI Layer** | Entry points, argument parsing, user interaction, config loading | Click or Typer framework with command decorators |
| **Dataset Generator** | Create synthetic test datasets, generate query-document pairs | Pure Python with LLM integration (optional) or template-based |
| **Retriever Adapters** | Normalize different retrieval system interfaces (Weaviate, Pinecone, etc.) | Adapter pattern with common base class |
| **Runner Orchestrator** | Coordinate evaluation runs, manage dataset→retriever→metrics pipeline | Pure Python coordinator |
| **Metrics Engine (Python)** | Dispatch to C or Python implementation, manage NumPy arrays | Try-except import pattern with fallback |
| **C Metric Kernels** | Performance-critical computational kernels (NDCG, MRR, Recall, etc.) | pybind11 with NumPy array interface |
| **Pure Python Fallback** | Identical metric implementations in pure Python | NumPy-based implementations for portability |
| **Persistence** | Store evaluation results, benchmarks, configurations | JSON/SQLite storage layer |
| **Config Manager** | Load/save configuration, environment detection | YAML/TOML parser with validation |

## Recommended Project Structure

```
berryeval/
├── cli/
│   ├── __init__.py         # CLI entry point registration
│   ├── main.py             # Main CLI group
│   ├── evaluate.py         # Evaluation commands
│   └── dataset.py          # Dataset generation commands
├── dataset/
│   ├── __init__.py
│   ├── generator.py        # Dataset generation logic
│   └── templates/          # Query/document templates
├── retrievers/
│   ├── __init__.py
│   ├── base.py             # Base retriever adapter interface
│   ├── weaviate.py         # Weaviate adapter
│   ├── pinecone.py         # Pinecone adapter
│   └── custom.py           # Custom/local retriever
├── metrics/
│   ├── __init__.py         # Auto-detection of C vs Python
│   ├── _metrics.py         # Shared metric interface
│   ├── _metrics_py.py      # Pure Python implementations
│   └── _metrics_c.pyi      # Type stubs for C extension
├── runner/
│   ├── __init__.py
│   ├── orchestrator.py     # Evaluation pipeline coordinator
│   └── parallel.py         # Parallel execution utilities
├── persistence/
│   ├── __init__.py
│   ├── storage.py          # Results storage
│   └── schemas.py          # Data schemas
├── config/
│   ├── __init__.py
│   └── manager.py          # Configuration management
└── utils/
    ├── __init__.py
    └── numpy_helpers.py    # NumPy utility functions

native/
├── include/
│   └── metrics/
│       ├── ndcg.hpp        # NDCG kernel header
│       ├── mrr.hpp         # MRR kernel header
│       └── recall.hpp      # Recall kernel header
├── src/
│   ├── ndcg.cpp            # NDCG implementation
│   ├── mrr.cpp             # MRR implementation
│   └── recall.cpp          # Recall implementation
├── bindings/
│   └── metrics_bindings.cpp  # pybind11 bindings
└── CMakeLists.txt          # CMake build configuration

tests/
├── unit/
│   ├── test_metrics_py.py  # Pure Python metric tests
│   ├── test_metrics_c.py   # C extension metric tests
│   └── test_dataset.py     # Dataset generator tests
├── integration/
│   └── test_pipeline.py    # End-to-end pipeline tests
└── benchmarks/
    └── benchmark_metrics.py  # Performance benchmarks

pyproject.toml              # PEP 517 build config
CMakeLists.txt              # Top-level CMake config
setup.py                    # Fallback for legacy installs
```

### Structure Rationale

- **berryeval/**: Main Python package with pure-Python implementations always available
- **native/**: Completely separate C++ code to enforce clean boundary between Python and C
- **metrics/ dual implementation**: Auto-detection pattern allows graceful degradation
- **CLI separation**: Command groups organized by functionality (evaluate, dataset, etc.)
- **Adapter pattern for retrievers**: Easy to add new retrieval systems without modifying core
- **Tests mirror source structure**: Separate tests for C and Python implementations ensure parity

## Architectural Patterns

### Pattern 1: Optional C Extension with Pure Python Fallback

**What:** Provide identical functionality in both C (for performance) and Python (for portability), with automatic selection at import time.

**When to use:** When performance is critical but platform/build compatibility must be guaranteed.

**Trade-offs:**
- PRO: Works everywhere Python runs, gets performance boost when C available
- PRO: Easier testing and debugging in pure Python
- CON: Must maintain two implementations in sync
- CON: Additional complexity in import logic

**Example:**
```python
# berryeval/metrics/__init__.py
try:
    # Try to import C extension (marked private with underscore)
    from berryeval._metrics_c import (
        ndcg, mrr, recall, precision
    )
    _USING_C_EXTENSION = True
except ImportError:
    # Fallback to pure Python implementation
    from berryeval.metrics._metrics_py import (
        ndcg, mrr, recall, precision
    )
    _USING_C_EXTENSION = False

# Expose a function to check which backend is active
def using_c_extension():
    return _USING_C_EXTENSION
```

### Pattern 2: NumPy Array Interface for C Bindings

**What:** Use pybind11's `py::array_t<T>` to pass NumPy arrays between Python and C++ with zero-copy semantics.

**When to use:** When computational kernels need to process large arrays efficiently.

**Trade-offs:**
- PRO: Zero-copy data transfer between Python and C++
- PRO: Type safety with `py::array_t<double>` enforces correct data types
- PRO: Automatic array shape/stride handling
- CON: Requires understanding of NumPy C API and memory layout
- CON: Must handle both C-contiguous and Fortran-contiguous arrays

**Example:**
```cpp
// native/bindings/metrics_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double ndcg_kernel(
    py::array_t<double> relevance_scores,
    py::array_t<int> rankings,
    int k
) {
    // Request buffer info from NumPy arrays
    auto relevance_buf = relevance_scores.request();
    auto rankings_buf = rankings.request();

    // Access raw pointers (zero-copy)
    double* relevance_ptr = static_cast<double*>(relevance_buf.ptr);
    int* rankings_ptr = static_cast<int*>(rankings_buf.ptr);

    // Compute NDCG using raw pointers
    // ... kernel implementation ...

    return ndcg_value;
}

PYBIND11_MODULE(_metrics_c, m) {
    m.def("ndcg", &ndcg_kernel,
          py::arg("relevance_scores"),
          py::arg("rankings"),
          py::arg("k") = 10,
          "Compute NDCG@k metric");
}
```

### Pattern 3: CMake Integration with setuptools/pyproject.toml

**What:** Use scikit-build or cmake-build-extension to bridge CMake (for C++ building) with Python packaging (PEP 517).

**When to use:** When building C++ extensions for Python packages with modern build standards.

**Trade-offs:**
- PRO: Standard CMake tooling for C++ developers
- PRO: PEP 517 compliant for modern Python packaging
- PRO: Handles platform-specific compilation automatically
- CON: Additional build system complexity
- CON: Requires CMake knowledge in addition to Python packaging

**Example:**
```toml
# pyproject.toml
[build-system]
requires = [
    "scikit-build-core>=0.10",
    "pybind11>=2.13",
    "numpy>=2.0"
]
build-backend = "scikit_build_core.build"

[project]
name = "berryeval"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.24",
    "click>=8.0",
]

[tool.scikit-build]
cmake.minimum-version = "3.18"
cmake.build-type = "Release"
wheel.py-api = "cp39"  # Python 3.9+ ABI compatibility
```

### Pattern 4: Adapter Pattern for Retriever Integration

**What:** Define a common interface for all retrieval systems, with concrete adapters for each supported backend.

**When to use:** When integrating multiple external systems with varying APIs into a unified interface.

**Trade-offs:**
- PRO: Easy to add new retrievers without modifying evaluation logic
- PRO: Consistent interface simplifies testing and mocking
- PRO: Decouples evaluation framework from specific retriever implementations
- CON: May not expose all capabilities of underlying systems
- CON: Adapter maintenance as retriever APIs evolve

**Example:**
```python
# berryeval/retrievers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class RetrieverAdapter(ABC):
    """Base interface for all retrieval system adapters."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for query.

        Returns:
            List of dicts with keys: 'doc_id', 'score', 'content'
        """
        pass

    @abstractmethod
    def batch_retrieve(self, queries: List[str], k: int = 10) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries in batch."""
        pass

# berryeval/retrievers/weaviate.py
class WeaviateAdapter(RetrieverAdapter):
    def __init__(self, client, index_name: str):
        self.client = client
        self.index_name = index_name

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        results = self.client.query.get(self.index_name) \
            .with_near_text({"concepts": [query]}) \
            .with_limit(k) \
            .do()

        # Normalize to standard format
        return [
            {
                'doc_id': r['_additional']['id'],
                'score': r['_additional']['certainty'],
                'content': r.get('content', '')
            }
            for r in results['data']['Get'][self.index_name]
        ]
```

## Data Flow

### Evaluation Pipeline Flow

```
User CLI Command
    ↓
CLI Parser (Click/Typer) → Config Validation
    ↓
Runner Orchestrator
    ↓
┌───────────────────────────────────────────┐
│   Dataset Generator (if needed)            │
│   • Load corpus                            │
│   • Generate synthetic queries             │
│   • Create ground truth relevance labels   │
└───────────────┬───────────────────────────┘
                ↓
┌───────────────────────────────────────────┐
│   Retriever Adapter                        │
│   • Normalize retriever interface          │
│   • Batch query execution                  │
│   • Collect rankings + scores              │
└───────────────┬───────────────────────────┘
                ↓
      NumPy Arrays Created
      (relevance_scores, rankings, k_values)
                ↓
┌───────────────────────────────────────────┐
│   Metrics Engine Dispatcher                │
│   • Auto-detect C vs Python backend        │
│   • Pass NumPy arrays to compute kernels   │
└───────────────┬───────────────────────────┘
                ↓
      ┌─────────┴──────────┐
      ↓                    ↓
┌──────────────┐    ┌─────────────┐
│ C Kernels    │    │ Python Impl │
│ (pybind11)   │    │ (pure NumPy)│
│ • NDCG       │    │ • NDCG      │
│ • MRR        │    │ • MRR       │
│ • Recall@k   │    │ • Recall@k  │
└──────┬───────┘    └──────┬──────┘
       └─────────┬──────────┘
                 ↓
         Computed Metrics
         (Python dicts)
                 ↓
┌───────────────────────────────────────────┐
│   Persistence Layer                        │
│   • Format results                         │
│   • Store to JSON/SQLite                   │
│   • Generate reports                       │
└───────────────────────────────────────────┘
                 ↓
         CLI Output Display
```

### NumPy Array Data Flow (Python ↔ C)

```
Python Layer
    ↓
NumPy Array Creation
    relevance = np.array([3.0, 2.0, 1.0, 0.0])
    rankings = np.array([0, 1, 2, 3], dtype=np.int32)
    ↓
Metrics Dispatcher
    metrics.ndcg(relevance, rankings, k=10)
    ↓
┌─────────────────────────────────────┐
│ Import Decision Point               │
│                                     │
│ if _metrics_c available:            │
│    → C Extension (zero-copy)        │
│ else:                               │
│    → Pure Python (NumPy ops)        │
└─────────┬───────────────────────────┘
          ↓
    C Extension Path
          ↓
pybind11 py::array_t<double> Wrapper
    ↓ (zero-copy)
C++ Pointer Access
    double* relevance_ptr = relevance.data()
    ↓
Computational Kernel (C++)
    for (int i = 0; i < n; ++i) {
        dcg += relevance_ptr[i] / log2(i + 2);
    }
    ↓
Return Scalar Result (double)
    ↓ (automatic conversion)
Python float object
    ↓
Return to caller
```

### Key Data Flows

1. **Dataset → Retriever → Metrics Pipeline:** Datasets generate queries → Retriever adapters fetch results → Metrics compute quality scores
2. **NumPy Zero-Copy Transfer:** Python NumPy arrays passed to C++ without copying via pybind11 buffer protocol
3. **Config → Runtime:** YAML/TOML config files loaded at CLI startup, validated, then passed to orchestrator
4. **Results → Storage:** Computed metrics aggregated into structured format, then persisted to JSON or SQLite

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **1-100 queries** | Single-threaded execution, pure Python fine, C extension provides 5-10x speedup but not critical |
| **100-10K queries** | Batch processing essential, C extension provides significant gains (100-1000ms → 10-100ms per batch), consider parallel retriever calls |
| **10K-1M queries** | Multi-process evaluation (Python multiprocessing), distribute retriever load, C kernels become critical (10-100x speedup), consider incremental result storage |
| **1M+ queries** | Distributed evaluation across machines, streaming dataset loading, C extensions mandatory, consider GPU kernels for massive parallelism |

### Scaling Priorities

1. **First bottleneck: Retriever latency** - Retrieval calls dominate total time; fix with batching, caching, async I/O, and parallel requests
2. **Second bottleneck: Metric computation** - After retrieval optimization, metric computation becomes significant; fix with C extensions and vectorization
3. **Third bottleneck: Dataset loading** - Large datasets may not fit in memory; fix with streaming/chunked loading and lazy evaluation

## Anti-Patterns

### Anti-Pattern 1: Tight Coupling Between Python and C

**What people do:** Implement business logic in C extensions, not just computational kernels

**Why it's wrong:**
- Makes debugging extremely difficult (no pdb, print debugging in C)
- Reduces portability and increases platform-specific build failures
- Hard to maintain synchronization between Python and C implementations
- Violates the principle of "Python for orchestration, C for computation"

**Do this instead:**
- Keep ALL business logic, orchestration, I/O, and error handling in Python
- C extensions should ONLY contain pure computational kernels
- C functions should have simple signatures: NumPy array in → scalar/array out
- Python wrapper handles validation, error messages, type conversion

### Anti-Pattern 2: Assuming C Extension Always Available

**What people do:** Write code that imports C extension directly without fallback handling

**Why it's wrong:**
- Fails on platforms where C compilation is difficult (Windows without MSVC, ARM, etc.)
- Breaks during development when C extension not yet built
- Makes testing more difficult (can't test in pure Python mode)
- Poor user experience when installation fails

**Do this instead:**
```python
# GOOD: Try-except with informative fallback
try:
    from berryeval._metrics_c import ndcg
    _backend = "C"
except ImportError:
    from berryeval._metrics_py import ndcg
    _backend = "Python"
    import warnings
    warnings.warn("C extension not available, using pure Python (slower)")
```

### Anti-Pattern 3: Copying NumPy Arrays in C Extensions

**What people do:** Convert NumPy arrays to std::vector or other C++ containers

**Why it's wrong:**
- Defeats the purpose of C acceleration (copying overhead negates computation savings)
- Doubles memory usage
- Adds unnecessary complexity

**Do this instead:**
```cpp
// GOOD: Work directly with NumPy buffer via raw pointers
double compute_ndcg(py::array_t<double> scores) {
    auto buf = scores.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];

    // Work directly with ptr, no copying
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += ptr[i] / log2(i + 2);
    }
    return result;
}
```

### Anti-Pattern 4: Monolithic Metrics Engine

**What people do:** Implement all metrics in a single large C++ file or Python module

**Why it's wrong:**
- Difficult to test individual metrics in isolation
- Increases build times (all metrics recompile when one changes)
- Hard to add new metrics without modifying existing code
- Violates single responsibility principle

**Do this instead:**
- Separate file for each metric family (ndcg.cpp, mrr.cpp, recall.cpp)
- Common header for shared utilities (discount functions, rank processing)
- Each metric independently importable
- Registration pattern for dynamic metric discovery

### Anti-Pattern 5: Inconsistent Metric Implementations

**What people do:** Python and C implementations produce different results due to floating-point handling or algorithm differences

**Why it's wrong:**
- Users get different results depending on platform
- Impossible to validate C implementation against Python
- Breaks reproducibility guarantees

**Do this instead:**
- Write comprehensive unit tests comparing C and Python outputs
- Use same algorithm and mathematical formulation in both
- Document any intentional differences (e.g., C uses float64, Python uses float32)
- Include numerical tolerance in tests (e.g., `assert abs(c_result - py_result) < 1e-10`)

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **Weaviate** | Adapter with weaviate-client SDK | Use batch queries, handle API rate limits |
| **Pinecone** | Adapter with pinecone-client SDK | Consider regional endpoints, index namespaces |
| **Elasticsearch** | Adapter with elasticsearch-py | Leverage scroll API for large result sets |
| **Milvus** | Adapter with pymilvus | Support for multiple distance metrics |
| **Custom/Local** | Adapter with file-based or HTTP API | Must implement retrieval interface |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **CLI ↔ Runner** | Direct Python function calls with validated config dicts | CLI validates inputs, Runner trusts them |
| **Runner ↔ Dataset** | Generator returns iterator/list of query-document tuples | Lazy evaluation for large datasets |
| **Runner ↔ Retriever** | Adapter pattern with standard retrieve() interface | Batch operations preferred |
| **Python ↔ C Metrics** | NumPy arrays via pybind11 buffer protocol (zero-copy) | C kernels return scalars or small arrays |
| **Metrics ↔ Persistence** | Python dicts/dataclasses → JSON serialization | Storage layer agnostic to metric computation |

## Build System Architecture

### CMake → Python Integration Pattern

```
pyproject.toml (PEP 517 entry point)
    ↓
scikit-build-core build backend
    ↓
CMakeLists.txt (top-level)
    ↓
┌─────────────────────────────────────┐
│ Configure pybind11                   │
│ Find Python, NumPy                   │
│ Set C++ standard (C++17)            │
└─────────────┬───────────────────────┘
              ↓
    native/CMakeLists.txt
              ↓
┌─────────────────────────────────────┐
│ Build C++ metric kernels             │
│ • ndcg.cpp → object file             │
│ • mrr.cpp → object file              │
│ • recall.cpp → object file           │
└─────────────┬───────────────────────┘
              ↓
    native/bindings/CMakeLists.txt
              ↓
┌─────────────────────────────────────┐
│ Build pybind11 module                │
│ • Link all kernel object files       │
│ • Create _metrics_c.so (Linux)       │
│ •        _metrics_c.pyd (Windows)    │
│ •        _metrics_c.dylib (macOS)    │
└─────────────┬───────────────────────┘
              ↓
    Wheel packaging
              ↓
berryeval-0.1.0-cp39-cp39-linux_x86_64.whl
    berryeval/
        _metrics_c.so  (included if build succeeded)
        metrics/
            _metrics_py.py  (always included)
```

### Conditional C Extension Build

The build system must gracefully handle cases where C compilation fails:

1. **Try to build C extension** (CMake + pybind11)
2. **If build fails** (missing compiler, incompatible platform):
   - Log warning but continue
   - Package only includes Python implementation
   - Installation succeeds
3. **If build succeeds**:
   - Package includes both C and Python implementations
   - Runtime auto-detection chooses C

This is achieved via scikit-build-core's optional extension configuration:

```toml
[tool.scikit-build]
cmake.minimum-version = "3.18"
cmake.build-type = "Release"
install.strip = false  # Preserve debug symbols in dev builds
wheel.py-api = "cp39"

# Continue installation even if CMake fails
[tool.scikit-build.cmake.define]
BERRYEVAL_REQUIRE_NATIVE = "OFF"  # OFF = optional, ON = required
```

## Component Build Order Dependencies

Based on the architecture, recommended build order for development:

### Phase 1: Core Python Infrastructure (No C dependencies)
1. **Config Manager** - Load YAML/TOML configurations
2. **CLI Scaffolding** - Basic Click/Typer commands (no-op implementations)
3. **Pure Python Metrics** - Implement NDCG, MRR, Recall in NumPy
4. **Unit Tests for Metrics** - Validate correctness of Python implementations

### Phase 2: Evaluation Pipeline (Pure Python mode)
5. **Dataset Generator** - Create synthetic test datasets
6. **Retriever Base Interface** - Define adapter abstract base class
7. **Runner Orchestrator** - Pipeline coordinator using Python metrics
8. **Simple Retriever Adapter** - File-based or in-memory retriever for testing

### Phase 3: C Acceleration Layer (Optional performance boost)
9. **CMake Build System** - Configure scikit-build-core
10. **C Metric Kernels** - Implement kernels in C++ (NDCG, MRR, Recall)
11. **pybind11 Bindings** - Wrap C++ kernels for Python
12. **Benchmark Tests** - Validate C implementations match Python, measure speedup

### Phase 4: Production Features
13. **Real Retriever Adapters** - Weaviate, Pinecone, Elasticsearch integrations
14. **Persistence Layer** - Results storage (JSON/SQLite)
15. **Advanced CLI Features** - Progress bars, parallel execution, batch modes
16. **Documentation & Examples** - User guides, API docs, example workflows

**Rationale:** This order ensures the framework is always functional (pure Python mode) before adding complexity. C extensions are built after Python implementations are validated, allowing C code to be tested against known-good Python baselines.

## Sources

### Python CLI & Build Systems
- [Python Packaging User Guide - Creating Command-Line Tools](https://packaging.python.org/en/latest/guides/creating-command-line-tools/)
- [10+ Best Python CLI Libraries 2026](https://medium.com/@wilson79/10-best-python-cli-libraries-for-developers-picking-the-right-one-for-your-project-cefb0bd41df1)
- [Building Python C Extension with CMake](https://martinopilia.com/posts/2018/09/15/building-python-extension.html)
- [cmake-build-extension on PyPI](https://pypi.org/project/cmake-build-extension/)
- [py-build-cmake on PyPI](https://pypi.org/project/py-build-cmake/)

### NumPy & C Extensions
- [NumPy C-API Documentation](https://numpy.org/doc/stable/reference/c-api/index.html)
- [pybind11 NumPy Documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
- [Guide to NumPy Arrays with Pybind11](https://scicoding.com/pybind11-numpy-compatible-arrays/)
- [Scipy Lecture Notes - Interfacing with C](https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html)

### Python Fallback Pattern
- [Optional C Extension Handling Discussion](https://discuss.python.org/t/optional-c-extension-handling/1595)
- [Python Developer's Guide - Extension Modules](https://devguide.python.org/developer-workflow/extension-modules/)

### RAG Evaluation Architecture
- [Retrieval Augmented Generation Evaluation Survey](https://arxiv.org/html/2504.14891v1)
- [Top 5 RAG Evaluation Platforms 2026](https://www.getmaxim.ai/articles/top-5-rag-evaluation-platforms-in-2026/)
- [Building Production RAG Systems 2026](https://brlikhon.engineer/blog/building-production-rag-systems-in-2026-complete-architecture-guide)

### Evaluation Metrics
- [Evaluation Metrics for Search - Weaviate](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [NDCG vs MRR - Ranking Metrics for RAG](https://blog.stackademic.com/ndcg-vs-mrr-ranking-metrics-for-information-retrieval-in-rags-2061b04298a6)
- [Evaluation Measures in Information Retrieval - Pinecone](https://www.pinecone.io/learn/offline-evaluation/)

### Dataset Generation
- [Generate Synthetic Data for RAG - AWS](https://aws.amazon.com/blogs/machine-learning/generate-synthetic-data-for-evaluating-rag-systems-using-amazon-bedrock/)
- [Ragas Test Data Generation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/)
- [RAGSynth Paper](https://arxiv.org/abs/2505.10989)

### Real-World Architecture Examples
- [scikit-learn Cython Best Practices](https://scikit-learn.org/stable/developers/cython.html)
- [pandas Internal Architecture](https://github.com/wesm/pandas2/blob/master/source/internal-architecture.rst)
- [pandas Extension Arrays](https://pandas.pydata.org/docs/development/extending.html)

---
*Architecture research for: BerryEval RAG Evaluation Framework*
*Researched: 2026-02-16*
*Confidence: HIGH - Based on official documentation, established patterns from scikit-learn/pandas, and current RAG evaluation ecosystem*
