# Roadmap: BerryEval

## Overview

BerryEval is built in 5 phases that progress from foundational infrastructure through user-facing features to performance optimization. Phase 1 establishes project scaffolding and pure Python metrics. Phase 2 delivers the CLI and synthetic dataset generation capabilities. Phase 3 implements the evaluation engine with retriever adapters and metric computation. Phase 4 adds run comparison and CI integration for production workflows. Phase 5 accelerates performance with C extensions to meet the 100K queries in 30 seconds target.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation & Core Infrastructure** - Project scaffolding, pure Python metrics, fallback layer
- [ ] **Phase 2: Dataset Generation & CLI** - Synthetic ground truth generation, CLI interface, dataset versioning
- [ ] **Phase 3: Evaluation Engine** - Retriever adapters, metrics computation, evaluation orchestration
- [ ] **Phase 4: Comparison & CI Integration** - Run comparison, regression detection, threshold enforcement
- [ ] **Phase 5: Performance Acceleration** - C kernels for metrics, 100K query performance target

## Phase Details

### Phase 1: Foundation & Core Infrastructure
**Goal**: Establish project infrastructure with pure Python metric implementations that provide the always-available fallback layer
**Depends on**: Nothing (first phase)
**Requirements**: PROJ-01, PROJ-02, PROJ-03, PROJ-04, NATV-04, NATV-05, NATV-06, NATV-07, METR-08, PERF-03
**Success Criteria** (what must be TRUE):
  1. Developer can clone repository and run tests successfully on Linux, macOS, and Windows
  2. Pure Python implementations of all IR metrics (recall@k, precision@k, MRR, nDCG, hit rate) produce correct results against known test cases
  3. System executes deterministically with same inputs producing identical outputs across runs
  4. NumPy array interface contracts are defined and documented for future C integration
  5. Build system and development tooling (Ruff, mypy, pytest) are configured and enforced in CI
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Project scaffolding, pyproject.toml, directory structure, development tooling
- [ ] 01-02-PLAN.md — Pure Python IR metrics (recall@k, precision@k, MRR, nDCG, hit rate) with TDD
- [ ] 01-03-PLAN.md — GitHub Actions CI pipeline and C extension fallback mechanism

### Phase 2: Dataset Generation & CLI
**Goal**: Users can generate versioned synthetic evaluation datasets from their corpus and interact with BerryEval through a CLI
**Depends on**: Phase 1
**Requirements**: CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, CLI-07, CLI-08, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, PERS-02, PERS-03
**Success Criteria** (what must be TRUE):
  1. User can run `berryeval generate` with a corpus directory and receive a versioned JSONL dataset with synthetic query-document pairs
  2. User can run `berryeval version` and see current version information
  3. User can run `berryeval inspect` to examine dataset contents and metadata
  4. Dataset generation uses configurable chunking (chunk-size, overlap) and LLM parameters (model selection)
  5. Datasets are versioned with deterministic configuration hashes enabling reproducibility
  6. All commands produce both machine-readable JSON output (--json flag) and human-readable terminal summaries
  7. CLI provides CI-compatible exit codes (0 for success, non-zero for failures)
**Plans**: 4 plans

Plans:
- [ ] 02-01-PLAN.md — CLI framework setup (Typer app, global --json, version command, evaluate/compare stubs)
- [ ] 02-02-PLAN.md — Dataset generation core (chunker, hasher, JSONL writer/reader, data models)
- [ ] 02-03-PLAN.md — LLM query generator and generate CLI command wiring
- [ ] 02-04-PLAN.md — Inspect CLI command for dataset examination

### Phase 3: Evaluation Engine
**Goal**: Users can benchmark retriever systems against evaluation datasets with comprehensive metric computation and reporting
**Depends on**: Phase 2
**Requirements**: RETR-01, RETR-02, RETR-03, RETR-04, METR-01, METR-02, METR-03, METR-04, METR-05, METR-06, METR-07, PERS-01
**Success Criteria** (what must be TRUE):
  1. User can configure a retriever (initially Pinecone) via YAML and run `berryeval evaluate` against a dataset
  2. System computes all standard IR metrics (recall@k, precision@k, MRR, nDCG, hit rate) with configurable k values
  3. System collects and reports latency statistics (p50, p95, p99) alongside quality metrics
  4. User can view per-query metric breakdowns to debug specific retrieval failures
  5. Evaluation results are persisted as JSON files with full run metadata
  6. Retriever adapter interface supports pluggable backends beyond Pinecone
**Plans**: 4 plans

Plans:
- [ ] 03-01-PLAN.md — Retriever adapter ABC, config types, YAML loader with env var substitution
- [ ] 03-02-PLAN.md — Pinecone retriever adapter with mocked tests
- [ ] 03-03-PLAN.md — Evaluation runner, latency tracking, result persistence
- [ ] 03-04-PLAN.md — Evaluate CLI command replacing stub with full implementation

### Phase 4: Comparison & CI Integration
**Goal**: Users can compare evaluation runs, detect regressions, and integrate BerryEval into CI pipelines with threshold enforcement
**Depends on**: Phase 3
**Requirements**: COMP-01, COMP-02, COMP-03, CI-01, CI-02, CI-03
**Success Criteria** (what must be TRUE):
  1. User can run `berryeval compare` on two evaluation runs and see metric deltas with regression warnings
  2. System highlights pass/fail threshold indicators for each metric based on user-defined criteria
  3. User can set --fail-below thresholds (e.g., recall@10=0.80) that produce non-zero exit codes when violated
  4. CI systems can parse output (JSON + exit codes) to automatically fail builds on quality regressions
  5. Comparison output is actionable, showing which queries regressed and by how much
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md — Comparison engine, types, and threshold checker (compare_runs, parse_thresholds, check_thresholds)
- [ ] 04-02-PLAN.md — Compare CLI command and evaluate --fail-below extension with Rich output and CI exit codes

### Phase 5: Performance Acceleration
**Goal**: C acceleration layer delivers 5-10x performance improvement to handle 100K queries in under 30 seconds
**Depends on**: Phase 4
**Requirements**: NATV-01, NATV-02, NATV-03, NATV-08, PERF-01, PERF-02
**Success Criteria** (what must be TRUE):
  1. C kernels accelerate all metric computations (recall@k, precision@k, MRR, nDCG, hit rate) using NumPy array interface
  2. System gracefully falls back to pure Python when C extensions are unavailable (no build failures break installation)
  3. C acceleration provides measurable 5-10x improvement over pure Python on representative workloads
  4. System handles 100K queries with top_k up to 50 in under 30 seconds on commodity hardware
  5. C layer never mutates input arrays and contains zero business logic (pure computational kernels)
  6. All platforms (Linux, macOS, Windows) receive pre-built binary wheels from CI
**Plans**: 3 plans

Plans:
- [ ] 05-01-PLAN.md — Build system, C metric kernels (all 5 metrics + rank lookup), and Python C extension bindings
- [ ] 05-02-PLAN.md — Benchmark suite validating 5-10x speedup and 100K queries in <30 seconds
- [ ] 05-03-PLAN.md — CI binary wheel pipeline (cibuildwheel) and Python fallback verification

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Core Infrastructure | 0/3 | Planned | - |
| 2. Dataset Generation & CLI | 0/4 | Planned | - |
| 3. Evaluation Engine | 0/4 | Planned | - |
| 4. Comparison & CI Integration | 0/2 | Planned | - |
| 5. Performance Acceleration | 0/3 | Planned | - |
