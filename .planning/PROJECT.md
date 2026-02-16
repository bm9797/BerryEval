# BerryEval

## What This Is

BerryEval is an open-source evaluation framework for benchmarking RAG retrieval quality using synthetic ground truth generation. It is a developer-facing CLI tool that ML engineers, backend engineers, and AI teams use to validate and benchmark their retrieval systems before production. The system is Python-first for orchestration with a C acceleration layer for high-performance metric computation at scale.

## Core Value

Engineers can reliably evaluate and benchmark their RAG retrieval systems through a deterministic, CLI-driven workflow that generates synthetic ground truth, computes standard retrieval metrics, and integrates into CI pipelines.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] CLI with subcommands: generate, evaluate, compare, inspect, version
- [ ] Synthetic evaluation dataset generation from a user-provided corpus
- [ ] Corpus chunking with configurable chunk-size and overlap
- [ ] LLM-powered synthetic ground truth creation (query-document pairs)
- [ ] Versioned evaluation datasets with deterministic configuration hashes
- [ ] YAML-based retriever configuration
- [ ] Pluggable retriever adapter system (Pinecone as first adapter)
- [ ] Retrieval metric computation: recall@k, precision@k, MRR, nDCG, hit rate
- [ ] Latency statistics collection (p50, p95, p99)
- [ ] Per-query breakdown output (optional)
- [ ] Run comparison with metric deltas and regression warnings
- [ ] CI integration with --fail-below thresholds and non-zero exit codes
- [ ] Machine-readable JSON output and human-readable terminal summary
- [ ] C acceleration layer for metric computation (5-10x improvement)
- [ ] Python/C interface via NumPy arrays (contiguous, primitive types)
- [ ] System must be fully functional in pure Python mode (C is optional)
- [ ] Performance target: 100K queries, top_k up to 50, <30s on commodity hardware

### Out of Scope

- Web UI or dashboard — CLI-first, v1 has no GUI
- Background service / daemon mode — BerryEval is explicitly run, not a background library
- Answer quality evaluation — v1 focuses on retrieval quality only, not generation quality
- Custom metric plugins — v1 ships with standard IR metrics only
- Cloud-hosted evaluation service — local and CI only

## Context

BerryEval targets the growing RAG ecosystem where teams need to validate retrieval quality before production deployment. The evaluation-first approach means teams generate synthetic ground truth from their own corpus, then benchmark retrievers against that ground truth using standard IR metrics.

The architecture follows a layered design:
- **CLI Layer** — User entry point, argument parsing, output formatting
- **Orchestration (Python Core)** — Dataset generation, retriever integration, benchmark orchestration, metrics configuration, result aggregation, persistence
- **Dataset Generator (Python)** — Corpus chunking, LLM-based synthetic query generation
- **Retriever Adapters (Python)** — Pluggable adapter pattern for different vector DBs
- **Metrics Engine (Python Wrapper)** — Metric configuration, dispatches to C or pure Python
- **C Metric Kernels (Acceleration Layer)** — Pure computational kernels, no business logic

Repository structure is prescribed:
```
berryeval/
├── berryeval/
│   ├── cli/
│   ├── dataset/
│   ├── retrievers/
│   ├── metrics/
│   ├── runner/
│   ├── persistence/
│   └── config/
├── native/
│   ├── include/
│   ├── src/
│   ├── bindings/
│   └── CMakeLists.txt
├── tests/
├── benchmarks/
├── pyproject.toml
└── README.md
```

## Constraints

- **Tech stack**: Python-first for orchestration, C for acceleration kernels only — ecosystem alignment with ML/AI tooling
- **Interface boundary**: All Python→C data must be contiguous NumPy arrays with primitive numeric types (int32, float32), pre-validated by Python
- **C layer scope**: Pure computational kernels only — no retriever logic, dataset logic, file I/O, LLM calls, or CLI logic. C must never mutate input arrays.
- **Fallback**: System must remain fully functional in pure Python mode; C acceleration is optional and pluggable
- **Performance**: 100K queries with top_k up to 50 in <30 seconds on commodity hardware; C must provide 5-10x improvement over pure Python loops
- **Output**: Deterministic outputs, machine-readable JSON, human-readable terminal summary, CI-compatible exit codes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python + C dual-layer architecture | Python for ecosystem alignment/extensibility, C for compute performance | — Pending |
| CLI as primary interface (no API/SDK in v1) | Developer-facing tool, not a library — explicit invocation model | — Pending |
| Synthetic ground truth generation | Enables evaluation without manual labeling; scalable to any corpus | — Pending |
| YAML-based retriever config | Declarative, version-controllable, supports multiple retriever backends | — Pending |
| NumPy as Python↔C bridge | Standard in ML ecosystem, contiguous memory layout for C interop | — Pending |

---
*Last updated: 2026-02-16 after initialization*