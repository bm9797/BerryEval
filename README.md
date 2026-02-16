# BerryEval

**Open-source evaluation framework for benchmarking RAG retrieval quality using synthetic ground truth generation.**

BerryEval provides an end-to-end pipeline for evaluating Retrieval-Augmented Generation (RAG) systems: generate synthetic evaluation datasets from your own corpus, run retrieval benchmarks against any backend, and compare results across configurations — all from a single CLI.

---

## Why BerryEval?

Evaluating RAG retrieval is hard. You need ground-truth query-document pairs, standardized metrics, and a way to compare changes over time. Most teams either skip evaluation entirely or build one-off scripts that can't be reused.

BerryEval solves this by providing:

- **Synthetic dataset generation** — Point it at your corpus and an LLM generates realistic search queries with known ground-truth relevance
- **Standard IR metrics** — Recall@k, Precision@k, MRR, nDCG, and Hit Rate computed per-query with NumPy
- **Pluggable retriever adapters** — Evaluate any retrieval backend through a simple adapter interface
- **Deterministic versioning** — SHA-256 config hashes ensure you know exactly which corpus + parameters produced each dataset
- **Dual-mode output** — Rich terminal tables for humans, `--json` for CI pipelines

## Architecture

```
berryeval/
├── cli/                 # Typer CLI with 5 commands
│   ├── _app.py          # App instance + global --json flag
│   ├── _output.py       # Dual-mode output helpers (Rich / JSON)
│   ├── generate.py      # Dataset generation command
│   ├── inspect_cmd.py   # Dataset inspection command
│   ├── version.py       # Version + environment info
│   ├── evaluate.py      # Evaluation runner (Phase 3)
│   └── compare.py       # Run comparison (Phase 4)
├── config/
│   └── types.py         # Chunk, DatasetMetadata, DatasetRecord dataclasses
├── dataset/
│   ├── chunker.py       # Text chunking with word-boundary respect
│   ├── generator.py     # LLM-based synthetic query generation (OpenAI)
│   ├── hasher.py        # Deterministic SHA-256 config hashing
│   ├── writer.py        # JSONL dataset writer with metadata header
│   └── reader.py        # JSONL dataset reader (metadata + record streaming)
├── metrics/
│   ├── _python.py       # Pure Python metric implementations (reference spec)
│   ├── _native.py       # C extension loader (Phase 5)
│   ├── _types.py        # PADDING_ID, type aliases (int32 arrays → float32 results)
│   └── _validation.py   # Input validation (dtype, shape, contiguity checks)
├── retrievers/
│   └── base.py          # RetrieverAdapter ABC + adapter registry
├── runner/              # Evaluation orchestration (Phase 3)
├── persistence/         # Result storage (Phase 3)
└── native/              # C acceleration kernels (Phase 5)
```

**Design principles:**
- Python-first with optional C acceleration for hot paths
- NumPy array contract: `int32` dtype, 2D, C-contiguous, `PADDING_ID=-1` for variable-length
- Backend dispatch via `try/except ImportError` — pure Python always available
- All metrics use `float64` intermediate computation, return `float32`

## Installation

### Requirements

- Python 3.10+
- An OpenAI API key (for dataset generation only)

### From source

```bash
git clone https://github.com/bm9797/BerryEval.git
cd BerryEval
pip install -e ".[dev]"
```

### Verify installation

```bash
berryeval version
```

```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component       ┃ Value                       ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ berryeval       │ 0.1.0                       │
│ python          │ 3.12.x                      │
│ metrics_backend │ python                      │
│ numpy           │ 2.x.x                       │
└─────────────────┴─────────────────────────────┘
```

## Usage

### Generate a dataset

Create synthetic evaluation data from a corpus of text files:

```bash
export OPENAI_API_KEY=your-key

berryeval generate \
  --corpus ./docs \
  --chunk-size 800 \
  --overlap 100 \
  --model gpt-4 \
  --output dataset.jsonl
```

This will:
1. Read all `.txt` and `.md` files from `./docs`
2. Split them into overlapping chunks (respecting word boundaries)
3. Generate a synthetic search query for each chunk via the OpenAI API
4. Write a versioned JSONL dataset with embedded metadata

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | *(required)* | Directory containing corpus documents |
| `--chunk-size` | `800` | Characters per chunk |
| `--overlap` | `100` | Character overlap between adjacent chunks |
| `--model` | `gpt-4` | LLM model for query generation |
| `--output`, `-o` | `dataset.jsonl` | Output file path |

### Inspect a dataset

Examine the metadata and statistics of a generated dataset:

```bash
berryeval inspect dataset.jsonl
```

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Field       ┃ Value               ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Config Hash │ a1b2c3d4e5f6...     │
│ Model       │ gpt-4               │
│ Chunk Size  │ 800                 │
│ Overlap     │ 100                 │
│ Timestamp   │ 2026-02-16T10:00:00 │
│ Version     │ 0.1.0               │
│ Query Count │ 42                  │
│ Corpus Files│ 5                   │
└─────────────┴─────────────────────┘
```

### JSON output

All commands support `--json` for machine-readable output (useful in CI):

```bash
berryeval --json version
berryeval --json inspect dataset.jsonl
```

### Evaluate retrieval (coming soon)

```bash
berryeval evaluate \
  --dataset dataset.jsonl \
  --retriever chromadb \
  --k 10
```

### Compare runs (coming soon)

```bash
berryeval compare run_a.json run_b.json
```

## Dataset Format

BerryEval datasets use JSONL (one JSON object per line) with a metadata header:

```jsonl
{"_type":"metadata","chunk_size":800,"config_hash":"a1b2c3...","model":"gpt-4","overlap":100,...}
{"_type":"record","chunk_text":"Paris is the capital...","metadata":{},"query":"What is the capital of France?","relevant_chunk_ids":["doc0000_chunk0000"]}
{"_type":"record","chunk_text":"The Eiffel Tower was...","metadata":{},"query":"When was the Eiffel Tower built?","relevant_chunk_ids":["doc0000_chunk0001"]}
```

**Metadata fields:**
- `config_hash` — Deterministic SHA-256 hash of corpus files + chunk parameters + model + prompt
- `model` — LLM used for query generation
- `chunk_size` / `overlap` — Chunking parameters
- `corpus_stats` — File count, chunk count, source files
- `timestamp` — ISO 8601 generation time
- `version` — BerryEval version that produced the dataset

**Record fields:**
- `query` — Synthetic search query
- `relevant_chunk_ids` — Ground-truth relevant chunk IDs
- `chunk_text` — The source text passage
- `metadata` — Source file and chunk ID

## Metrics

BerryEval implements five standard information retrieval metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Recall@k** | Fraction of relevant documents found in top-k | [0, 1] |
| **Precision@k** | Fraction of top-k that are relevant | [0, 1] |
| **MRR** | Reciprocal rank of first relevant document | [0, 1] |
| **nDCG** | Normalized discounted cumulative gain (binary relevance) | [0, 1] |
| **Hit Rate** | Whether any relevant document appears in top-k | {0, 1} |

All metrics operate on NumPy `int32` arrays and return `float32` results. The pure Python implementations serve as the reference specification; future C kernels will be validated against them.

```python
import numpy as np
from berryeval.metrics import recall_at_k, precision_at_k, mrr, ndcg, hit_rate

retrieved = np.array([[10, 20, 30, 40, 50]], dtype=np.int32)
relevant  = np.array([[10, 30, -1, -1, -1]], dtype=np.int32)  # -1 = padding

recall_at_k(retrieved, relevant, k=3)    # array([1.0], dtype=float32)
precision_at_k(retrieved, relevant, k=3) # array([0.667], dtype=float32)
mrr(retrieved, relevant, k=5)            # array([1.0], dtype=float32)
ndcg(retrieved, relevant, k=5)           # array([0.863], dtype=float32)
hit_rate(retrieved, relevant, k=1)       # array([1.0], dtype=float32)
```

## Retriever Adapters

BerryEval uses a pluggable adapter pattern. Implement the `RetrieverAdapter` interface to benchmark any retrieval backend:

```python
from berryeval.retrievers.base import RetrieverAdapter, RetrievedDocument, register_adapter

@register_adapter
class MyRetriever(RetrieverAdapter):
    name = "my-retriever"

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        # Call your retrieval backend
        results = my_backend.search(query, top_k)
        return [
            RetrievedDocument(doc_id=r.id, score=r.score)
            for r in results
        ]

    def close(self) -> None:
        self.my_backend.disconnect()
```

Adapters support the context manager protocol for automatic cleanup:

```python
with MyRetriever() as retriever:
    docs = retriever.retrieve("What is RAG?", top_k=10)
```

## Development

### Setup

```bash
git clone https://github.com/bm9797/BerryEval.git
cd BerryEval
pip install -e ".[dev]"
```

### Run tests

```bash
pytest -v
```

107 tests covering metrics, dataset operations, and CLI commands. All OpenAI interactions are mocked — no API key needed for testing.

### Code quality

```bash
ruff check .              # Linting
ruff format --check .     # Formatting
mypy berryeval            # Type checking
```

### CI

GitHub Actions runs on every push and PR to `main`:
- **Lint** — ruff check + format
- **Type check** — mypy strict mode
- **Test** — 3x3 matrix (Ubuntu/macOS/Windows x Python 3.10/3.11/3.12)

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation & Core Infrastructure (metrics, CI, project scaffolding) | Done |
| 2 | Dataset Generation & CLI (chunker, generator, JSONL, all 5 commands) | Done |
| 3 | Evaluation Engine (retriever adapters, metrics runner, evaluate command) | Planned |
| 4 | Comparison & CI Integration (run comparison, CI helpers) | Planned |
| 5 | Performance Acceleration (C kernels for metrics hot paths) | Planned |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | User error (bad input, missing file, invalid config) |
| `2` | System error (API failure, permission denied) |

## License

MIT
