# Requirements: BerryEval

**Defined:** 2026-02-16
**Core Value:** Engineers can reliably evaluate and benchmark their RAG retrieval systems through a deterministic, CLI-driven workflow

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Project Foundation

- [ ] **PROJ-01**: Project uses Python 3.10+ with pyproject.toml-based packaging
- [ ] **PROJ-02**: Repository follows prescribed structure (berryeval/, native/, tests/, benchmarks/)
- [ ] **PROJ-03**: Development tooling configured (Ruff, mypy, pytest)
- [ ] **PROJ-04**: CI pipeline builds and tests on Linux/macOS/Windows

### CLI Interface

- [ ] **CLI-01**: User can run `berryeval generate` to create synthetic evaluation datasets
- [ ] **CLI-02**: User can run `berryeval evaluate` to benchmark a retriever against a dataset
- [ ] **CLI-03**: User can run `berryeval compare` to diff two evaluation runs
- [ ] **CLI-04**: User can run `berryeval inspect` to examine dataset or run contents
- [ ] **CLI-05**: User can run `berryeval version` to see installed version info
- [ ] **CLI-06**: All commands produce machine-readable JSON output (--json flag)
- [ ] **CLI-07**: All commands produce human-readable terminal summary by default
- [ ] **CLI-08**: Exit codes are CI-compatible (0 = pass, non-zero = fail)

### Dataset Generation

- [ ] **DATA-01**: User can provide a corpus directory and generate a synthetic evaluation dataset
- [ ] **DATA-02**: Corpus is chunked with configurable chunk-size and overlap parameters
- [ ] **DATA-03**: LLM generates synthetic query-document pairs from corpus chunks
- [ ] **DATA-04**: Output dataset is versioned with a deterministic configuration hash
- [ ] **DATA-05**: Dataset is written in JSONL format with one record per line
- [ ] **DATA-06**: User can specify which LLM model to use for generation (--model flag)

### Retriever Adapters

- [ ] **RETR-01**: User configures retrievers via YAML configuration files
- [ ] **RETR-02**: Pluggable adapter interface allows adding new retriever backends
- [ ] **RETR-03**: Pinecone adapter ships as the first built-in retriever
- [ ] **RETR-04**: Adapter accepts a query string and returns ranked document list with scores

### Metrics Engine

- [ ] **METR-01**: System computes recall@k for configurable k values
- [ ] **METR-02**: System computes precision@k for configurable k values
- [ ] **METR-03**: System computes Mean Reciprocal Rank (MRR)
- [ ] **METR-04**: System computes normalized Discounted Cumulative Gain (nDCG)
- [ ] **METR-05**: System computes hit rate
- [ ] **METR-06**: System collects latency statistics (p50, p95, p99)
- [ ] **METR-07**: User can optionally view per-query metric breakdown
- [ ] **METR-08**: All metrics have pure Python implementations that are always available

### Run Comparison

- [ ] **COMP-01**: User can compare two evaluation runs and see metric deltas
- [ ] **COMP-02**: System flags regressions with warning indicators
- [ ] **COMP-03**: System shows pass/fail threshold indicators per metric

### CI Integration

- [ ] **CI-01**: User can set --fail-below thresholds (e.g., recall@10=0.80)
- [ ] **CI-02**: Threshold violations produce non-zero exit codes
- [ ] **CI-03**: Output is parseable by standard CI systems (JSON + exit codes)

### C Acceleration Layer

- [ ] **NATV-01**: C kernels accelerate recall@k batch computation
- [ ] **NATV-02**: C kernels accelerate precision@k, MRR, nDCG, hit rate computation
- [ ] **NATV-03**: C kernels accelerate ranking position lookup
- [ ] **NATV-04**: Python↔C interface uses contiguous NumPy arrays (int32, float32)
- [ ] **NATV-05**: C layer never mutates input arrays
- [ ] **NATV-06**: C layer contains zero business logic (pure computational kernels)
- [ ] **NATV-07**: System gracefully falls back to pure Python when C extension is unavailable
- [ ] **NATV-08**: C acceleration provides 5-10x improvement over pure Python loops

### Performance

- [ ] **PERF-01**: System handles 100K queries with top_k up to 50
- [ ] **PERF-02**: Full evaluation completes in <30 seconds on commodity hardware
- [ ] **PERF-03**: Deterministic execution — same inputs produce same outputs

### Persistence & Output

- [ ] **PERS-01**: Evaluation results are saved as JSON files
- [ ] **PERS-02**: Dataset files are self-contained and portable
- [ ] **PERS-03**: Configuration hash enables dataset reproducibility verification

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Metrics

- **ADVM-01**: LLM-as-judge reference-free evaluation metrics
- **ADVM-02**: Custom metric plugin system for user-defined metrics
- **ADVM-03**: Semantic similarity scoring between retrieved and expected documents

### Advanced Adapters

- **ADVR-01**: Weaviate retriever adapter
- **ADVR-02**: Elasticsearch retriever adapter
- **ADVR-03**: Chroma retriever adapter
- **ADVR-04**: Generic HTTP adapter for custom retriever APIs

### Reporting

- **REPT-01**: HTML report generation with visualizations
- **REPT-02**: Historical run tracking and trend analysis
- **REPT-03**: Export results to CSV/Parquet formats

### Multi-Provider LLM

- **MLLM-01**: Support for Anthropic models in synthetic generation
- **MLLM-02**: Support for local/open-source LLMs via Ollama
- **MLLM-03**: LiteLLM integration for unified provider interface

## Out of Scope

| Feature | Reason |
|---------|--------|
| Web UI or dashboard | CLI-first tool, v1 has no GUI — keep focused |
| Background service / daemon mode | Explicitly run tool, not a background library |
| Answer/generation quality evaluation | v1 focuses on retrieval quality only |
| End-to-end RAG evaluation (RAGAS territory) | Stay focused on retrieval, not generation |
| Observability platform features (Phoenix territory) | Different product category entirely |
| Framework lock-in (LangChain/LlamaIndex integration) | Keep framework-agnostic |
| Cloud-hosted evaluation service | Local and CI only for v1 |
| Real-time monitoring | Evaluation is batch-oriented, not streaming |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROJ-01 | Phase 1 | Pending |
| PROJ-02 | Phase 1 | Pending |
| PROJ-03 | Phase 1 | Pending |
| PROJ-04 | Phase 1 | Pending |
| CLI-01 | Phase 2 | Pending |
| CLI-02 | Phase 2 | Pending |
| CLI-03 | Phase 2 | Pending |
| CLI-04 | Phase 2 | Pending |
| CLI-05 | Phase 2 | Pending |
| CLI-06 | Phase 2 | Pending |
| CLI-07 | Phase 2 | Pending |
| CLI-08 | Phase 2 | Pending |
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| DATA-05 | Phase 2 | Pending |
| DATA-06 | Phase 2 | Pending |
| RETR-01 | Phase 3 | Pending |
| RETR-02 | Phase 3 | Pending |
| RETR-03 | Phase 3 | Pending |
| RETR-04 | Phase 3 | Pending |
| METR-01 | Phase 3 | Pending |
| METR-02 | Phase 3 | Pending |
| METR-03 | Phase 3 | Pending |
| METR-04 | Phase 3 | Pending |
| METR-05 | Phase 3 | Pending |
| METR-06 | Phase 3 | Pending |
| METR-07 | Phase 3 | Pending |
| METR-08 | Phase 1 | Pending |
| COMP-01 | Phase 4 | Pending |
| COMP-02 | Phase 4 | Pending |
| COMP-03 | Phase 4 | Pending |
| CI-01 | Phase 4 | Pending |
| CI-02 | Phase 4 | Pending |
| CI-03 | Phase 4 | Pending |
| NATV-01 | Phase 5 | Pending |
| NATV-02 | Phase 5 | Pending |
| NATV-03 | Phase 5 | Pending |
| NATV-04 | Phase 1 | Pending |
| NATV-05 | Phase 1 | Pending |
| NATV-06 | Phase 1 | Pending |
| NATV-07 | Phase 1 | Pending |
| NATV-08 | Phase 5 | Pending |
| PERF-01 | Phase 5 | Pending |
| PERF-02 | Phase 5 | Pending |
| PERF-03 | Phase 1 | Pending |
| PERS-01 | Phase 3 | Pending |
| PERS-02 | Phase 2 | Pending |
| PERS-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 50 total
- Mapped to phases: 50
- Unmapped: 0 ✓

**Phase distribution:**
- Phase 1 (Foundation & Core Infrastructure): 10 requirements
- Phase 2 (Dataset Generation & CLI): 16 requirements
- Phase 3 (Evaluation Engine): 12 requirements
- Phase 4 (Comparison & CI Integration): 6 requirements
- Phase 5 (Performance Acceleration): 6 requirements

---
*Requirements defined: 2026-02-16*
*Last updated: 2026-02-16 after roadmap creation*
