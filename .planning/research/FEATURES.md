# Feature Research

**Domain:** RAG Evaluation Frameworks
**Researched:** 2026-02-16
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Core Retrieval Metrics (Precision@k, Recall@k, MRR, nDCG) | Standard IR metrics essential for measuring retrieval quality. All competitors provide these. | LOW | Well-established formulas. BerryEval already includes these. |
| Synthetic Test Data Generation | Manual ground truth is expensive. Users expect automated testset creation from corpus. | MEDIUM | RAGAS pioneered this. Critical for reducing manual curation overhead. |
| Dataset Management | Store, version, and reuse evaluation datasets. Prevents re-generating same data. | MEDIUM | Need to persist datasets with metadata (corpus source, generation params, timestamps). |
| CI/CD Integration | RAG systems must prevent quality regressions. Teams expect pytest-style testing in pipelines. | MEDIUM | DeepEval demonstrates this well. Requires pytest integration and in_ci flags. |
| Evaluation Reports | Clear, actionable output showing which queries failed and why. | LOW | JSON/markdown reports with per-query breakdowns. |
| Multiple Retriever Support | Evaluate different embedding models, vector DBs, or retrieval strategies. | LOW | Plugin/adapter pattern for different retriever implementations. |
| Reference-Free Evaluation | LLM-as-judge for quality metrics (faithfulness, relevance) without human labels. | MEDIUM | RAGAS popularized this. Requires LLM API integration but reduces annotation costs. |
| CLI Workflow | ML/backend engineers expect command-line tools, not just Python APIs. | LOW | Commands like `berryeval generate`, `berryeval evaluate`, `berryeval compare`. |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Synthetic Ground Truth Focus | Pure retrieval evaluation (not end-to-end RAG). Competitors bundle retrieval + generation. | MEDIUM | BerryEval's core differentiator. Solves "just want to test my vector DB" use case. |
| Run Comparison & Regression Detection | Side-by-side comparison of evaluation runs. Detect performance drops across commits. | MEDIUM | MLflow excels here. Show delta metrics, highlight regressions. Valuable for CI/CD. |
| Zero-Config Quick Start | Single command to generate dataset + evaluate without config files. | LOW | `berryeval quick-start <corpus-dir>` beats multi-step setup. Developer UX win. |
| Embedding Model Independence | Works with any embedding model (OpenAI, HuggingFace, Cohere, custom). | MEDIUM | Competitors often locked to specific providers. Abstract embedding interface. |
| Hit Rate & Latency Tracking | Performance metrics alongside quality metrics. Production readiness focus. | LOW | P50/P99 latency, QPS. Competitors focus only on quality. |
| Local-First Operation | No required cloud dependencies. Runs entirely on-prem for security-sensitive teams. | MEDIUM | Phoenix requires OTEL. BerryEval works offline with local LLMs. |
| Lightweight Framework (Not Platform) | Focused tool that integrates with existing stacks vs monolithic platform. | LOW | Lean alternative to heavy platforms like Maxim AI or LangSmith. |
| Deterministic Test Cases | Generate reproducible synthetic queries with seed control. | LOW | Enables exact result reproduction for debugging. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| End-to-End RAG Evaluation | "Evaluate the whole pipeline, not just retrieval" | Mixes retrieval and generation concerns. Debugging becomes harder when you can't isolate components. RAGAS/DeepEval already do this well. | Keep BerryEval retrieval-focused. Integrate with RAGAS for generation metrics if needed. |
| Built-in Observability Platform | "Monitor production RAG systems" | Scope creep into Arize Phoenix/LangSmith territory. Requires tracing infrastructure, dashboards, persistent storage. | Provide export formats (OpenTelemetry, JSON) for existing observability tools. |
| Web UI / Dashboard | "Visual interface for non-technical users" | BerryEval targets ML/backend engineers who prefer CLI/code. UI adds maintenance burden and slows iteration. | CLI + good report formatting (markdown/HTML exports). Teams can use existing tools for visualization. |
| LLM App Framework Integration (LangChain/LlamaIndex decorators) | "Auto-instrument my RAG app" | Lock-in to specific frameworks. Many teams build custom RAG without frameworks. | Provide simple adapter interface. Users wrap their retriever in <5 lines of code. |
| Proprietary Hosted Service | "SaaS platform for team collaboration" | Contradicts open-source focus. Competitors already dominate (Confident AI, Maxim, LangSmith). | Stay CLI/library. Users can self-host results in GitHub Pages, S3, etc. |
| Real-Time Streaming Evaluation | "Evaluate as queries come in" | Adds complexity for minimal value in offline evaluation use case. Production monitoring is different problem. | Batch evaluation mode. Export to real-time monitoring tools if needed. |

## Feature Dependencies

```
Synthetic Data Generation
    └──requires──> Corpus Ingestion
                       └──requires──> Document Chunking

Evaluation Execution
    └──requires──> Synthetic Dataset
    └──requires──> Retriever Adapter

Run Comparison
    └──requires──> Result Storage
    └──requires──> Evaluation Execution

CI/CD Integration
    └──requires──> Evaluation Execution
    └──enhances──> Run Comparison (detect regressions)

Reference-Free Metrics
    └──requires──> LLM API Integration
    └──optional-for──> Evaluation Execution (can use retrieval metrics only)

Latency Tracking
    └──enhances──> Evaluation Execution (adds timing instrumentation)
```

### Dependency Notes

- **Synthetic Data Generation requires Corpus Ingestion:** Can't generate queries without source documents. Need chunking strategy (size, overlap) before generating Q&A pairs.
- **Evaluation Execution requires Synthetic Dataset:** Dataset provides ground truth (query → expected document mappings). Can't evaluate without it.
- **Run Comparison requires Result Storage:** Need versioned results to compare. SQLite or JSON files for persistence.
- **CI/CD Integration enhances Run Comparison:** Regression detection becomes actionable when failing CI builds. Thresholds like "recall@5 must be > 0.85".
- **Reference-Free Metrics optional for Evaluation:** Can run pure IR metrics (precision, recall, MRR, nDCG) without LLM. LLM-as-judge adds faithfulness, relevance.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] **Core Retrieval Metrics** — Table stakes. Precision@k, Recall@k, MRR, nDCG, Hit Rate.
- [ ] **Synthetic Data Generation** — Core value prop. Generate query-document pairs from corpus using LLM.
- [ ] **Dataset Storage** — Persist generated datasets. JSON format with metadata.
- [ ] **Evaluation Execution** — Run retriever against dataset, calculate metrics.
- [ ] **CLI Interface** — Commands: `generate`, `evaluate`, `compare`.
- [ ] **Retriever Adapter Interface** — Simple Python class users implement (~5 methods).
- [ ] **Evaluation Reports** — Markdown/JSON output with per-query results and aggregate metrics.
- [ ] **Run Comparison** — Compare two evaluation runs, show metric deltas.

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **CI/CD Integration** — Add pytest integration once evaluation works. Trigger: user requests in GitHub issues.
- [ ] **Reference-Free Metrics** — LLM-as-judge for faithfulness/relevance. Add after synthetic generation is solid.
- [ ] **Latency Tracking** — P50/P99 latency metrics. Add when users ask "how fast is my retriever?"
- [ ] **Multiple Embedding Models** — Support for HuggingFace, Cohere beyond OpenAI. Add based on demand.
- [ ] **Deterministic Seeding** — Reproducible synthetic data generation. Add for debugging use cases.
- [ ] **HTML Report Export** — Pretty reports for sharing. Add when markdown feels insufficient.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Custom Metric Plugins** — User-defined metrics. Wait for clear plugin API patterns to emerge.
- [ ] **Benchmark Dataset Support** — Pre-built datasets (BEIR, MTEB). Nice-to-have but not critical initially.
- [ ] **Multi-Retriever Comparison** — Evaluate 3+ retrievers in one run. Defer until single retriever workflow is polished.
- [ ] **OpenTelemetry Export** — Integration with observability stacks. Wait for production monitoring requests.
- [ ] **Web Report Viewer** — Static site generator for results. Only if users consistently request it.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Core Retrieval Metrics | HIGH | LOW | P1 |
| Synthetic Data Generation | HIGH | MEDIUM | P1 |
| Dataset Storage | HIGH | LOW | P1 |
| Evaluation Execution | HIGH | MEDIUM | P1 |
| CLI Interface | HIGH | LOW | P1 |
| Run Comparison | HIGH | MEDIUM | P1 |
| Evaluation Reports | HIGH | LOW | P1 |
| Retriever Adapter Interface | HIGH | LOW | P1 |
| CI/CD Integration | MEDIUM | MEDIUM | P2 |
| Reference-Free Metrics | MEDIUM | MEDIUM | P2 |
| Latency Tracking | MEDIUM | LOW | P2 |
| Deterministic Seeding | MEDIUM | LOW | P2 |
| HTML Report Export | LOW | LOW | P2 |
| Multiple Embedding Models | MEDIUM | MEDIUM | P2 |
| Custom Metric Plugins | LOW | HIGH | P3 |
| Benchmark Dataset Support | LOW | MEDIUM | P3 |
| Multi-Retriever Comparison | LOW | MEDIUM | P3 |
| OpenTelemetry Export | LOW | MEDIUM | P3 |
| Web Report Viewer | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (MVP)
- P2: Should have, add when possible (v1.x)
- P3: Nice to have, future consideration (v2+)

## Competitor Feature Analysis

| Feature | RAGAS | DeepEval | Arize Phoenix | BerryEval Approach |
|---------|-------|----------|---------------|-------------------|
| Synthetic Data Generation | LLM-based, knowledge graph enrichment | LLM-based with custom templates | Limited, focuses on tracing | LLM-based from corpus, focus on retrieval (not end-to-end RAG) |
| Core Metrics | Context precision, recall, faithfulness, answer relevance | 50+ metrics (RAG, conversational, red-teaming) | Q&A accuracy, hallucination, toxicity | Pure retrieval metrics (P@k, R@k, MRR, nDCG, hit rate, latency) |
| CI/CD Integration | Pytest integration with `in_ci` flag | Native pytest compatibility with decorators | Export to monitoring tools | Pytest integration with regression detection |
| Developer UX | Python API + experiments tracking | Code-first, pytest-style tests | Tracing-focused, OpenTelemetry | CLI-first with zero-config quick-start |
| Framework Lock-In | Works with LangChain, LlamaIndex | Framework-agnostic with @observe decorator | OpenTelemetry instrumentation | Adapter interface, no framework requirement |
| Observability | Integrates with Arize, LangSmith | Confident AI platform integration | Full observability platform (tracing, datasets, experiments) | Export-focused (JSON, OpenTelemetry) - integrate with existing tools |
| Deployment | Library + cloud platform option | Library + Confident AI cloud | OSS + cloud (app.phoenix.arize.com) | Library only, local-first |
| Scope | End-to-end RAG evaluation | End-to-end LLM evaluation (RAG, agents, conversational) | Production observability + evaluation | Retrieval quality evaluation only |

## Sources

### Official Documentation (HIGH Confidence)
- [Ragas Documentation](https://docs.ragas.io/en/stable/) - Testset generation, metrics, integrations
- [DeepEval GitHub](https://github.com/confident-ai/deepeval) - Pytest integration, component-level evaluation
- [Arize Phoenix GitHub](https://github.com/Arize-ai/phoenix) - Tracing, OpenTelemetry, observability

### Market Analysis (MEDIUM Confidence)
- [Top 5 RAG Evaluation Platforms in 2026](https://www.getmaxim.ai/articles/top-5-rag-evaluation-platforms-in-2026/) - Platform comparison
- [RAG Evaluation: 2026 Metrics and Benchmarks](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation) - Metric overview
- [7 RAG Evaluation Tools You Must Know](https://www.iguazio.com/blog/best-rag-evaluation-tools/) - Feature comparison
- [RAG Monitoring Tools Benchmark in 2026](https://research.aimultiple.com/rag-monitoring/) - Monitoring vs evaluation

### Technical Deep Dives (HIGH Confidence)
- [Synthetic Test Data Generation for RAG](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/) - RAGAS testset generation
- [RAG Evaluation in CI/CD with DeepEval](https://www.confident-ai.com/blog/how-to-evaluate-rag-applications-in-ci-cd-pipelines-with-deepeval) - Pytest integration patterns
- [Evaluation Metrics for RAG Systems](https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/) - MRR, nDCG, precision, recall
- [RAG Evaluation Best Practices](https://www.evidentlyai.com/llm-guide/rag-evaluation) - Common pitfalls, evaluation methodology

### Benchmark Research (MEDIUM Confidence)
- [7 RAG Benchmarks](https://www.evidentlyai.com/blog/rag-benchmarks) - BEIR, CRAG, RAGBench, LegalBench-RAG
- [RAGBench Paper](https://arxiv.org/abs/2407.11005) - Benchmark dataset structure
- [Retrieval Augmented Generation Evaluation Survey](https://arxiv.org/html/2504.14891v1) - Comprehensive survey of RAG evaluation frameworks

### Developer Experience Research (MEDIUM Confidence)
- [The 5 Best RAG Evaluation Tools](https://www.braintrust.dev/articles/best-rag-evaluation-tools) - Developer experience analysis
- [15 Best Open-Source RAG Frameworks](https://apidog.com/blog/best-open-source-rag-frameworks/) - Framework comparison
- [RAG Evaluation CLI with Ragas](https://docs.ragas.io/en/stable/howtos/cli/rag_eval/) - CLI workflow patterns

---
*Feature research for: RAG Evaluation Frameworks*
*Researched: 2026-02-16*
