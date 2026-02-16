# Project Research Summary

**Project:** BerryEval
**Domain:** RAG Evaluation Framework (Retrieval Quality Focus)
**Researched:** 2026-02-16
**Confidence:** HIGH

## Executive Summary

BerryEval is a retrieval-focused RAG evaluation framework implemented as a Python CLI with optional C acceleration for performance-critical metric computations. Unlike comprehensive RAG platforms (RAGAS, DeepEval) that evaluate end-to-end pipeline quality, BerryEval specializes in **retrieval quality evaluation** using information retrieval metrics (Precision@k, Recall@k, MRR, nDCG) combined with synthetic ground truth generation. The recommended technical approach uses Python 3.10+ with Typer for CLI, ranx for validated IR metrics, and nanobind-based C extensions for computational kernels, with pure Python fallbacks ensuring universal compatibility.

The architecture follows a proven pattern from scientific Python libraries (NumPy, scikit-learn): Python orchestration layer with optional C acceleration. The framework generates synthetic test datasets from user corpora using LLM-based query generation, executes retrieval through pluggable adapters (Weaviate, Pinecone, custom), computes IR metrics, and produces actionable evaluation reports. Critical success factors include **synthetic ground truth quality validation** (avoiding systematic bias), **cross-platform build robustness** (C extensions must gracefully degrade), and **component-level metric isolation** (debug retrieval separately from generation).

Key risks center on C extension complexity (reference counting errors, memory alignment issues), synthetic data quality degradation (LLM-generated queries drifting from real-world distribution), and abstraction leaks in retriever adapters (heterogeneous backends with incompatible capabilities). Mitigation strategies include establishing strict memory management discipline from day one, validating synthetic datasets against real queries, and designing adapter interfaces with explicit capability negotiation rather than assuming uniform functionality.

## Key Findings

### Recommended Stack

BerryEval's stack balances modern Python developer experience with performance requirements. The core framework uses **Python 3.10+ with Typer** (type-safe CLI framework), **Rich** (terminal formatting), and **Pydantic** (configuration validation). Computational kernels leverage **nanobind 2.11.0+** for Python-C++ bindings (4x faster compile, 10x lower runtime overhead vs pybind11), **NumPy 2.4.2+** for array operations, and **scikit-build-core** for PEP 517-compliant CMake integration. IR metrics use **ranx 0.3.21+** (Numba-accelerated, TREC-validated, 3-12x faster than alternatives). LLM integration relies on **OpenAI SDK 2.21.0+** and **Anthropic SDK 0.79.0+** for direct API access without abstraction overhead.

**Core technologies:**
- **Python 3.10+**: Minimum for modern type hints and match/case — balances compatibility with developer experience
- **Typer 0.23.2+**: Type-hint-driven CLI with auto-generated help — significantly cleaner than Click for type-safe CLIs
- **nanobind 2.11.0+**: Python-C++ bindings — 10x lower runtime overhead than pybind11, near-identical syntax
- **ranx 0.3.21+**: Numba-accelerated IR metrics — validated against TREC eval, 3-12x faster than pytrec_eval
- **Rich 14.3.2+**: Terminal output formatting — essential for metric visualization and user-friendly results display
- **Pydantic 2.12.5+**: Data validation — type-safe config, LLM output validation, Rust-core performance
- **scikit-build-core 0.11.6+**: CMake build backend — modern PEP 621 compliance, cleaner than setuptools

**Critical stack decisions:**
- **nanobind over pybind11**: Same syntax, measurably better performance (4x compile, 10x runtime for object passing)
- **ranx over pytrec_eval**: Pure Python (no C compilation), Numba JIT, same metrics, cleaner API
- **scikit-build-core over setuptools**: PEP 621 compliant, static config, required for modern C extensions
- **Direct LLM SDKs over LangChain**: BerryEval needs simple LLM calls for query generation, not agents/chains
- **Ruff over Black+flake8+isort**: One tool, 10-100x faster, single config, all-in-one linter/formatter

### Expected Features

RAG evaluation users expect **synthetic test data generation** (manual ground truth is expensive), **core retrieval metrics** (Precision@k, Recall@k, MRR, nDCG), **dataset management** (version and reuse evaluation datasets), and **CI/CD integration** (prevent quality regressions). BerryEval's differentiator is **pure retrieval focus** (not end-to-end RAG), **run comparison with regression detection** (side-by-side evaluation across commits), and **local-first operation** (no required cloud dependencies).

**Must have (table stakes):**
- Core Retrieval Metrics (Precision@k, Recall@k, MRR, nDCG) — all competitors provide these
- Synthetic Test Data Generation — automated testset creation from corpus reduces manual curation overhead
- Dataset Management — persist datasets with metadata (corpus source, generation params, timestamps)
- CI/CD Integration — pytest-style testing in pipelines to prevent quality regressions
- Evaluation Reports — clear, actionable output showing which queries failed and why
- Multiple Retriever Support — plugin/adapter pattern for different retriever implementations
- CLI Workflow — commands like `berryeval generate`, `berryeval evaluate`, `berryeval compare`

**Should have (competitive):**
- Synthetic Ground Truth Focus — pure retrieval evaluation (not end-to-end RAG), solves "just want to test my vector DB" use case
- Run Comparison & Regression Detection — side-by-side comparison with delta metrics, highlight regressions
- Zero-Config Quick Start — single command like `berryeval quick-start <corpus-dir>` beats multi-step setup
- Hit Rate & Latency Tracking — performance metrics (P50/P99 latency, QPS) alongside quality metrics
- Local-First Operation — runs entirely on-prem for security-sensitive teams, works offline with local LLMs
- Deterministic Test Cases — reproducible synthetic queries with seed control for debugging

**Defer (v2+):**
- End-to-End RAG Evaluation — mixing retrieval and generation concerns makes debugging harder (RAGAS does this well)
- Built-in Observability Platform — scope creep into Phoenix/LangSmith territory (provide export formats instead)
- Web UI / Dashboard — BerryEval targets ML/backend engineers who prefer CLI/code
- LLM Framework Integration — avoid lock-in to LangChain/LlamaIndex (provide simple adapter interface instead)

### Architecture Approach

The architecture follows the **optional C extension with pure Python fallback** pattern proven by NumPy and scikit-learn. Python handles orchestration (CLI, dataset generation, retriever coordination, persistence) while C extensions accelerate computational kernels (NDCG, MRR, Recall calculations). The metrics engine auto-detects C extension availability at import time and gracefully degrades to pure Python (NumPy-based) implementations when C compilation is unavailable or fails. This ensures universal compatibility while providing 5-100x performance gains when C extensions are available.

**Major components:**
1. **CLI Layer** (Typer) — entry points, argument parsing, user interaction, configuration loading
2. **Dataset Generator** — synthetic test dataset creation from corpus using LLM APIs (OpenAI/Anthropic)
3. **Retriever Adapters** — normalize different retrieval system interfaces (Weaviate, Pinecone, custom) via adapter pattern
4. **Runner Orchestrator** — pipeline coordinator managing dataset → retriever → metrics flow
5. **Metrics Engine** — dispatcher using try-except import pattern to select C or Python implementation
6. **C Metric Kernels** (nanobind) — performance-critical computations with NumPy array interface (zero-copy)
7. **Pure Python Fallback** — identical metric implementations in NumPy for portability
8. **Persistence Layer** — results storage (JSON/SQLite) and report generation

**Key architectural patterns:**
- **NumPy Array Interface**: Use `py::array_t<T>` for zero-copy data transfer between Python and C++
- **Adapter Pattern**: Common retriever interface with backend-specific implementations (Weaviate, Pinecone, etc.)
- **Graceful Degradation**: Try-except import selects C extension or Python fallback; framework always works
- **CMake Integration**: scikit-build-core bridges CMake (C++ building) with PEP 517 (Python packaging)
- **Separation of Concerns**: Python for orchestration/I/O/errors, C for pure computational kernels only

**Data flow:**
1. CLI validates inputs → Runner loads/generates dataset
2. Retriever adapter executes batch queries → creates NumPy arrays (relevance_scores, rankings)
3. Metrics engine dispatches to C kernels (if available) or Python implementation
4. Results aggregated → formatted → persisted (JSON/SQLite) → displayed (Rich terminal output)

### Critical Pitfalls

Research identified 10 critical pitfalls with HIGH recovery cost if not prevented early. The top 5 that must be addressed in Phase 1:

1. **Reference Counting Errors at Python/C Boundary** — Memory leaks or segfaults when reference counts get out of sync. Borrowed vs. owned references are commonly confused. PyList_Append() increments refcount, leading to leaks when lists are destroyed. **Mitigation:** Document reference ownership for every C API function, use Py_XDECREF() in error paths, run tests with Python debug build (--with-pydebug), never call Py_DECREF() on borrowed references from getters.

2. **NPY_ARRAY_OWNDATA Flag Mismanagement** — Double-free crashes or memory leaks when wrapping external C memory in NumPy arrays. PyArray_ENABLEFLAGS(NPY_OWNDATA) causes platform-specific crashes (works on Linux, crashes on Windows). **Mitigation:** Never use PyArray_ENABLEFLAGS(OWNDATA) with external memory. Always use PyArray_SetBaseObject() with a custom PyCapsule that has a destructor. Test on Windows, Linux, and macOS.

3. **Memory Alignment and Contiguity Assumptions** — C code assumes NumPy arrays are C-contiguous and aligned, but receives transposed/sliced/strided views. Accessing misaligned data triggers SIGBUS crashes. **Mitigation:** Always check PyArray_IS_C_CONTIGUOUS() before passing to C. Use PyArray_FROM_OTF() with NPY_ARRAY_C_CONTIGUOUS flag to force copy if needed. Test with transposed, sliced, and reshaped array views.

4. **Cross-Platform Build System Fragility** — C extensions build on developer machine but fail on CI or user systems due to platform-specific dependencies, compiler differences, or missing build tools. cibuildwheel generates pure Python wheels instead of binary wheels. **Mitigation:** Test builds on all target platforms from day one. Use cibuildwheel to standardize wheel building. Add CI jobs for Windows, macOS (Intel + ARM), and Linux. Test installation from built wheel, not editable install.

5. **Synthetic Ground Truth Bias and Quality Degradation** — LLM-generated synthetic data fails to represent real-world query complexity, introducing systematic biases that make evaluation metrics unreliable. High scores on synthetic data but poor performance on real queries. **Mitigation:** Validate synthetic data against real production queries. Create hybrid test sets (70% synthetic for coverage, 30% human-verified for quality). Include adversarial examples (typos, uncommon entities, ambiguous queries). Track generation parameters and version datasets.

**Additional critical pitfalls:**
- **Component-Level Evaluation Blind Spots** — Testing only end-to-end makes debugging impossible when metrics drop
- **Edge Case Metric Blindness** — Standard metrics fail on multi-source queries, typos, rare entities, overlapping intents
- **LLM Judge Bias in Evaluation** — Systematic biases (length preference, positional bias, self-preference) corrupt results
- **Retriever Adapter Abstraction Leaks** — Heterogeneous backends with incompatible capabilities break unified interface
- **Performance Testing with Toy Data** — O(n²) algorithms run quickly with 1K test docs but crawl with 1M production docs

## Implications for Roadmap

Based on research, the roadmap should prioritize **foundation before features**. C extension infrastructure, cross-platform builds, and memory management discipline must be established in Phase 1 before building higher-level functionality. Synthetic data generation (Phase 2) requires quality validation gates before being declared "done." Metric implementation (Phase 3) must include component-level isolation and edge case coverage from the start.

### Suggested Phase Structure

#### Phase 1: Core C Extension Foundation
**Rationale:** C extension complexity creates the highest technical risk. Establishing memory management discipline, cross-platform builds, and pure Python fallback patterns before building features prevents catastrophic rework. All pitfalls #1, #2, #3, #4, #6 must be addressed here.

**Delivers:**
- Pure Python metric implementations (NDCG, MRR, Recall@k, Precision@k) validated with known test cases
- C metric kernels with identical behavior to Python (verified with numerical tolerance tests)
- nanobind bindings with NumPy array interface (zero-copy data transfer)
- Cross-platform build system (scikit-build-core + CMake) tested on Windows, macOS, Linux
- Auto-detection import pattern (try C extension, fallback to Python)
- Reference counting discipline (documented ownership, Py_XDECREF in error paths)
- Memory ownership patterns (PyArray_SetBaseObject with PyCapsule destructors)

**Addresses:**
- Core Retrieval Metrics (table stakes feature)
- Critical pitfalls: reference counting errors, OWNDATA flag misuse, memory alignment, cross-platform builds

**Avoids:**
- Building features before infrastructure is proven stable
- Platform-specific failures discovered by users instead of CI
- Memory leaks and crashes requiring emergency patches

**Research flags:** Standard pattern (NumPy/scikit-learn). Skip research-phase. Follow established patterns from ARCHITECTURE.md.

---

#### Phase 2: Synthetic Ground Truth Generation
**Rationale:** Dataset generation is the core differentiator but requires LLM integration and quality validation. Must be built after infrastructure (Phase 1) but before retrieval integration (Phase 5) to provide test datasets for adapter development.

**Delivers:**
- Corpus ingestion with document chunking (chonkie for semantic/token chunking)
- LLM-based query generation (OpenAI/Anthropic SDK with prompt templates)
- Ground truth relevance labeling (query → expected document mappings)
- Dataset persistence (JSON format with metadata: corpus source, generation params, timestamps)
- Quality validation gates: diversity metrics, human review sampling, real-world validation
- Hybrid test set creation: 70% synthetic + 30% human-verified
- Deterministic seeding for reproducible generation

**Addresses:**
- Synthetic Test Data Generation (table stakes feature)
- Dataset Management (table stakes feature)
- Critical pitfall: synthetic ground truth bias and quality degradation

**Avoids:**
- Synthetic data bias reinforcing inequalities and creating overconfident incorrect conclusions
- Generation parameters tuned solely on synthetic data without real-world validation
- Test sets lacking linguistic diversity, long-tail patterns, adversarial examples

**Research flags:** **Needs research-phase.** LLM prompting strategies for query generation, chunking strategies for different document types, quality validation metrics for synthetic datasets.

---

#### Phase 3: Retrieval Metrics Implementation
**Rationale:** With infrastructure (Phase 1) and datasets (Phase 2) in place, implement the evaluation pipeline. Focus on component-level metric isolation and edge case coverage from the start to avoid "looks done but isn't" issues.

**Delivers:**
- Runner orchestrator (dataset → retriever → metrics pipeline coordinator)
- Evaluation execution engine (batch query processing, metric computation, result aggregation)
- Component-level metrics (retrieval quality measured independently)
- Edge case test suites (typos, rare entities, multi-hop queries, ambiguous intents)
- Position-aware metrics (Recall@5 vs Recall@20 to detect buried results)
- Latency tracking (P50/P99 latency, QPS alongside quality metrics)
- Evaluation reports (JSON/markdown with per-query breakdowns and aggregate metrics)

**Addresses:**
- Evaluation Execution (table stakes feature)
- Evaluation Reports (table stakes feature)
- Hit Rate & Latency Tracking (competitive differentiator)
- Critical pitfalls: component-level evaluation blind spots, edge case metric blindness

**Avoids:**
- End-to-end metrics only (debugging becomes impossible when metrics drop)
- Testing only clean benchmark queries (missing real-world messiness)
- Position-agnostic metrics (relevant docs at rank 20 are effectively invisible)

**Research flags:** Standard pattern (ranx metrics, established IR evaluation). Skip research-phase.

---

#### Phase 4: CLI Interface
**Rationale:** After core evaluation works (Phase 3), build the user-facing CLI. Typer makes this straightforward, but UX decisions (command structure, output formatting, progress indication) require user testing.

**Delivers:**
- CLI commands: `berryeval generate`, `berryeval evaluate`, `berryeval compare`
- Configuration management (YAML config loading, validation, environment detection)
- Rich terminal output (tables, progress bars, syntax highlighting)
- Zero-config quick-start mode (`berryeval quick-start <corpus-dir>`)
- Helpful error messages (component breadcrumbs, diagnostic steps, troubleshooting links)
- Progress indication for long-running operations (100K query benchmarks)

**Addresses:**
- CLI Workflow (table stakes feature)
- Zero-Config Quick Start (competitive differentiator)
- UX pitfalls: cryptic errors, no progress indication, overwhelming metric output

**Avoids:**
- Cryptic C extension import errors without troubleshooting guidance
- Users thinking process hung during long evaluations
- Configuration confusion across different retriever backends

**Research flags:** Standard pattern (Typer CLI, Rich formatting). Skip research-phase.

---

#### Phase 5: Retriever Integration Layer
**Rationale:** Build adapters for real retriever backends (Weaviate, Pinecone, Elasticsearch) after evaluation pipeline works with test data. Adapter interface design must accept abstraction leaks (heterogeneous capabilities) rather than forcing uniformity.

**Delivers:**
- Retriever adapter base interface with capability flags (`supports_filtering`, `supports_batch`, etc.)
- Weaviate adapter (batch queries, API rate limit handling)
- Pinecone adapter (regional endpoints, index namespaces)
- Elasticsearch adapter (scroll API for large result sets)
- Custom/local adapter (file-based or HTTP API)
- Normalized error handling (backend exceptions wrapped in framework exceptions)
- Adapter validation (smoke tests, health checks on first use)

**Addresses:**
- Multiple Retriever Support (table stakes feature)
- Critical pitfall: retriever adapter abstraction leaks

**Avoids:**
- Forcing lowest-common-denominator functionality (different backends have different strengths)
- Assuming uniform error types or performance characteristics
- Users bypassing adapter to call backend APIs directly

**Research flags:** **Needs research-phase for each new backend.** API patterns, rate limits, batch capabilities, error types vary significantly across vector databases.

---

#### Phase 6: Run Comparison & CI Integration
**Rationale:** After evaluation works end-to-end (Phases 1-5), add workflow features that make BerryEval useful in production. Run comparison enables regression detection; CI integration makes it actionable.

**Delivers:**
- Run comparison engine (side-by-side evaluation results, metric deltas, regression highlighting)
- Result versioning and storage (SQLite or JSON with run metadata)
- pytest integration (pytest decorators, in_ci flags, regression thresholds)
- Regression detection (alert when recall@5 drops below threshold)
- CI/CD examples (GitHub Actions, GitLab CI workflows)
- Report exports (HTML, markdown for sharing)

**Addresses:**
- Run Comparison & Regression Detection (competitive differentiator)
- CI/CD Integration (table stakes feature)

**Avoids:**
- Quality regressions discovered in production instead of CI
- No visibility into which commit caused metric degradation
- Manual comparison of evaluation runs

**Research flags:** Standard pattern (pytest integration, version control). Skip research-phase.

---

#### Phase 7: Performance Optimization
**Rationale:** After features are complete (Phases 1-6), optimize for the 100K queries in <30s performance target. Establish scale testing with realistic data volumes before declaring targets met.

**Delivers:**
- Realistic-scale test datasets (1M+ documents, diverse query distribution)
- Performance regression tests in CI at representative scale
- Batch processing optimizations (parallel retriever calls, async LLM API calls)
- Profiling and bottleneck identification (retriever latency vs metric computation)
- Algorithmic complexity validation (ensure O(n) not O(n²))
- Database query optimization (indexes, query plans)
- Stress tests (deliberately exceed target scale)

**Addresses:**
- Performance target: 100K queries in <30s
- Critical pitfall: performance testing with toy data

**Avoids:**
- O(n²) algorithms hidden by small test data (catastrophic at production scale)
- Database queries without indexes (acceptable at 10K rows, timeout at 10M)
- Batch vs single-item API performance differences ignored

**Research flags:** Standard pattern (profiling, optimization). Skip research-phase. Use established tools (cProfile, py-spy).

---

### Phase Ordering Rationale

**Foundation before features:** C extension infrastructure (Phase 1) must be solid before building dataset generation (Phase 2) or metrics (Phase 3). Memory management bugs discovered after features are built require painful refactoring.

**Datasets before integration:** Synthetic data generation (Phase 2) provides test datasets needed for developing retriever adapters (Phase 5). Building adapters first forces use of hand-crafted test data.

**Metrics before CLI:** Evaluation pipeline (Phase 3) should work as a Python API before building CLI (Phase 4). CLI is a thin wrapper around working functionality.

**Integration after core:** Retriever adapters (Phase 5) require working evaluation pipeline (Phase 3) to test against. Building adapters before pipeline forces mocking.

**Workflow features after foundation:** Run comparison and CI integration (Phase 6) add value only after evaluation works end-to-end (Phases 1-5).

**Optimization last:** Performance optimization (Phase 7) requires measuring realistic workloads. Optimizing before features are complete leads to premature optimization.

**Pitfall prevention:** Phase 1 addresses 4 critical pitfalls (reference counting, memory ownership, alignment, cross-platform builds). Phase 2 addresses synthetic data bias. Phase 3 addresses component isolation and edge cases. This prevents rework.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2 (Synthetic Ground Truth Generation):** LLM prompting strategies vary by domain. Query generation quality validation metrics need research. Chunking strategies for different document types (code, PDFs, markdown) require investigation.
- **Phase 5 (Retriever Integration Layer):** Each new retriever backend (Weaviate, Pinecone, Elasticsearch, Milvus) has unique API patterns, rate limits, batch capabilities, and error types. Backend-specific research required.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Core C Extension Foundation):** NumPy/scikit-learn patterns well-documented. Follow established nanobind + scikit-build-core approach.
- **Phase 3 (Retrieval Metrics Implementation):** ranx provides validated metrics. IR evaluation is well-established domain.
- **Phase 4 (CLI Interface):** Typer + Rich patterns well-documented. Standard CLI design.
- **Phase 6 (Run Comparison & CI Integration):** pytest integration well-established. Version control patterns standard.
- **Phase 7 (Performance Optimization):** Profiling and optimization tools (cProfile, py-spy) standard. Established patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All core technologies verified with official sources (PyPI releases, official docs). nanobind, ranx, Typer, Ruff, scikit-build-core all have recent stable releases. Version compatibility matrix validated. |
| Features | HIGH | Multiple high-quality sources (official docs for RAGAS, DeepEval, Phoenix). Competitive analysis based on official feature lists. Table stakes vs differentiators grounded in market analysis. |
| Architecture | HIGH | Based on official documentation (NumPy C API, pybind11/nanobind, Python packaging guides) and established patterns from scikit-learn/pandas. Component responsibilities and data flow patterns proven by scientific Python ecosystem. |
| Pitfalls | HIGH | All pitfalls sourced from official documentation (NumPy memory management, Python C API), academic research (synthetic ground truth bias), and community post-mortems (cross-platform builds, LLM judge bias). Recovery strategies validated against real-world examples. |

**Overall confidence:** HIGH

All research areas have multiple corroborating sources. Stack recommendations based on recent stable releases (Feb 2026 or later). Architecture patterns proven by NumPy/scikit-learn (10+ years of production use). Pitfalls documented in official guides and academic research. Feature analysis grounded in competitor official documentation.

### Gaps to Address

**During planning:**
- **LLM prompting strategies for query generation:** RAGAS provides one approach (knowledge graph enrichment), but BerryEval's retrieval-only focus may need different prompts. Validate during Phase 2 planning.
- **Retriever backend capabilities matrix:** Need to document which backends support filtering, batch operations, reranking, etc. Research during Phase 5 planning for each adapter.
- **Performance profiling at scale:** 100K queries target needs validation with realistic data. Establish baseline in Phase 7 planning before optimization.

**During execution:**
- **Synthetic data quality metrics:** What quantitative metrics indicate "good" synthetic data? Need to define thresholds during Phase 2 implementation.
- **Human validation sampling strategy:** How many synthetic queries need human review to validate quality? Cost-benefit analysis during Phase 2.
- **C extension optimization flags:** Which compiler flags (-O3, -march=native, -ffast-math) provide best performance without sacrificing correctness? Profile during Phase 1.

**No blockers identified.** Gaps are refinement questions, not missing critical information.

## Sources

### Primary (HIGH confidence)

**Technology Stack:**
- Typer: https://typer.tiangolo.com/, PyPI 0.23.2 (Feb 2026)
- nanobind: https://nanobind.readthedocs.io/, PyPI 2.11.0 (Jan 2026)
- ranx: https://github.com/AmenRa/ranx, PyPI 0.3.21 (Aug 2025)
- Ruff: https://docs.astral.sh/ruff/, PyPI 0.15.1 (Feb 2026)
- NumPy: PyPI 2.4.2 (Jan 2026)
- Rich: PyPI 14.3.2 (Feb 2026)
- Pydantic: PyPI 2.12.5 (Nov 2025)
- scikit-build-core: PyPI 0.11.6 (Jul 2024)
- OpenAI SDK: PyPI 2.21.0 (Feb 2026)
- Anthropic SDK: PyPI 0.79.0 (Feb 2026)

**RAG Evaluation Frameworks:**
- [Ragas Documentation](https://docs.ragas.io/en/stable/)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [Arize Phoenix GitHub](https://github.com/Arize-ai/phoenix)

**Python C Extensions:**
- [NumPy C-API Documentation](https://numpy.org/doc/stable/reference/c-api/index.html)
- [pybind11 NumPy Documentation](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
- [Python Extension Patterns: Reference Counting](https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html)
- [NumPy: Memory Management in C API](https://numpy.org/doc/stable/reference/c-api/data_memory.html)

### Secondary (MEDIUM confidence)

**Market Analysis:**
- [Top 5 RAG Evaluation Platforms in 2026](https://www.getmaxim.ai/articles/top-5-rag-evaluation-platforms-in-2026/)
- [RAG Evaluation: 2026 Metrics and Benchmarks](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation)
- [7 RAG Evaluation Tools You Must Know](https://www.iguazio.com/blog/best-rag-evaluation-tools/)

**Architectural Patterns:**
- [Building Python C Extension with CMake](https://martinopilia.com/posts/2018/09/15/building-python-extension.html)
- [Guide to NumPy Arrays with Pybind11](https://scicoding.com/pybind11-numpy-compatible-arrays/)
- [Scipy Lecture Notes - Interfacing with C](https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html)

**Performance & Pitfalls:**
- nanobind vs pybind11 benchmarks: https://nanobind.readthedocs.io/en/latest/benchmark.html
- ranx performance claims: Springer paper (ECIR 2022), GitHub repo
- [Evidently AI: Complete Guide to RAG Evaluation](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Pinecone: RAG Evaluation Best Practices](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)

### Tertiary (LOW confidence)

**Needs validation during implementation:**
- chonkie 85% speed claim (single WebSearch source, not verified in official docs)
- Exact ranx compatibility with Python 3.14 (likely works but not explicitly tested per search results)

---

*Research completed: 2026-02-16*
*Ready for roadmap: yes*
