# Technology Stack

**Project:** BerryEval
**Researched:** 2026-02-16
**Confidence:** HIGH

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ | Orchestration layer | Minimum 3.10 for modern type hints, match/case, and best compatibility with type checkers. 3.9+ supported by all dependencies but 3.10 recommended for development experience. |
| Typer | 0.23.2+ | CLI framework | Built on Click with type-hint-driven API, automatic help generation, minimal boilerplate. Released Feb 2026, actively maintained (FastAPI ecosystem). Significantly cleaner than Click for type-safe CLIs. |
| Rich | 14.3.2+ | Terminal output formatting | Professional CLI output with tables, progress bars, syntax highlighting. Essential for metric visualization and user-friendly results display. Released Feb 2026, widely adopted standard. |
| Pydantic | 2.12.5+ | Data validation & config | Type-safe configuration, LLM output validation, request/response models. Rust-core performance, 10B+ downloads, used by FastAPI/LangChain. De facto standard for data validation. |

### C Acceleration Layer

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| nanobind | 2.11.0+ | Python-C++ bindings | 4x faster compile, 5x smaller binaries, 10x lower runtime overhead vs pybind11. Near-identical syntax to pybind11 but superior performance. Python 3.9+ support, stable ABI targeting in 3.12+. Released Jan 2026. |
| NumPy | 2.4.2+ | Numerical arrays | Foundation for all numeric computation. Version 2.x has significant performance improvements. Used by ranx and all IR metric libraries. Released Jan 2026. |
| scikit-build-core | 0.11.6+ | Build backend (CMake) | Modern CMake-based build backend for C extensions. Static config in pyproject.toml, cleaner than setuptools. Used by PyTorch, adopted by scientific Python. |

**Why nanobind over alternatives:**
- **vs pybind11**: Same syntax, measurably faster (4x compile, 10x runtime for object passing), smaller binaries
- **vs Cython**: Better for wrapping C++ kernels, type-safe bindings, easier debugging, comparable performance
- **vs ctypes/cffi**: Type safety, automatic memory management, performance parity with native extensions

**Why scikit-build-core over alternatives:**
- **vs meson-python**: CMake more common for C++ projects, wider ecosystem, easier CI integration
- **vs setuptools**: Modern PEP 621 compliance, static config, cleaner separation of build logic
- **vs poetry/hatch native**: Required for compiled extensions, these don't handle C compilation

### LLM Integration

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| OpenAI SDK | 2.21.0+ | GPT-4 for synthetic queries | Official SDK, async support, structured outputs via Pydantic. Clean API, well-maintained. Released Feb 2026. |
| Anthropic SDK | 0.79.0+ | Claude for synthetic queries | Official SDK, prompt caching for cost efficiency, async support. Released Feb 2026. Alternative/fallback to OpenAI. |
| LiteLLM | Latest | Multi-provider abstraction (optional) | Unified interface for 100+ LLM providers, cost tracking, OpenAI-compatible format. Use if supporting multiple LLM backends; skip if OpenAI-only. |

**Recommendation:** Start with OpenAI SDK directly. Add LiteLLM only if users request multi-provider support (avoid premature abstraction).

### Information Retrieval Metrics

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| ranx | 0.3.21+ | IR metrics library | Numba-accelerated metrics (Recall@k, Precision@k, MRR, MAP, nDCG, hit rate). Validated against TREC eval, 3-12x faster than alternatives. Featured ECIR/CIKM/SIGIR. Released Aug 2025. |

**Why ranx over alternatives:**
- **vs pytrec_eval**: 3-10x faster (Numba JIT), pure Python (no C compilation), same metrics, cleaner API
- **vs custom implementation**: Validated against TREC, optimized, maintained, handles edge cases
- **vs RAGAS metrics**: RAGAS is for end-to-end RAG (answer quality), ranx is for retrieval (IR metrics). Different domains.

### Text Processing

| Technology | Version | Purpose | When to Use |
|------------|---------|---------|-------------|
| chonkie | 1.5.2+ | Semantic/token chunking | Lightweight chunking library with SemanticChunker, TokenChunker, SentenceChunker. 85% faster than semantic-text-splitter. Use for corpus preprocessing. Released Jan 2026. |
| tiktoken | Latest | OpenAI tokenization | Official OpenAI BPE tokenizer, fast Rust implementation. Use if chunk size needs to match OpenAI token limits exactly. |

**Recommendation:** Use chonkie for chunking (supports tiktoken tokenizer internally). Skip standalone tiktoken unless precise OpenAI token alignment required.

### Configuration & Data

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyYAML | 6.0.3+ | YAML config parsing | Human-readable config files (thresholds, adapters, LLM settings). Use yaml.safe_load() only. Python 3.14 compatible, released Sep 2025. |

**Why YAML over alternatives:**
- **vs JSON**: Human-friendly, comments, multi-line strings. Better for user-edited config.
- **vs TOML**: YAML more familiar to ML engineers, better for nested adapter configs.
- **vs environment variables**: Config files version-controlled, easier CI integration.

## Development Tools

### Code Quality

| Tool | Version | Purpose | Why Recommended |
|------|---------|---------|-----------------|
| Ruff | 0.15.1+ | Linting + formatting | All-in-one linter/formatter (replaces black, isort, flake8, pyupgrade). Rust-based, runs in milliseconds vs seconds. Actively developed by Astral (uv team). Released Feb 2026. |
| mypy | 1.19.1+ | Static type checking | Industry-standard type checker. Essential for maintaining type safety across Python/C boundary. Released Dec 2025. |

**Why Ruff over alternatives:**
- **vs Black + isort + flake8**: One tool, 10-100x faster, single config, same output quality
- **vs pylint**: Faster, modern rule set, better default config, less noise

### Testing

| Tool | Version | Purpose | Configuration |
|------|---------|---------|---------------|
| pytest | 9.0.2+ | Test framework | Use --import-mode=importlib in pyproject.toml. Released Dec 2025. |
| pytest-cov | 7.0.0+ | Coverage reporting | Integration with coverage.py for test coverage. Use with GitHub Actions for PR comments. Released Sep 2025. |

**CLI testing strategy:**
- Use Typer's CliRunner for isolated CLI invocation (no subprocesses)
- Test metric kernels with parametrized test cases (known inputs → expected outputs)
- Test C extensions with both Python unit tests and C++ unit tests (via CMake/CTest)

### CI/CD

| Tool | Purpose | Notes |
|------|---------|-------|
| GitHub Actions | CI/CD pipeline | Matrix testing across Python 3.10-3.14, multiple OS (Linux/macOS/Windows for C compilation) |
| pytest-cov + coverage comment | Coverage reporting | Automated PR comments with coverage delta |
| uv | Fast package manager (optional) | 10-100x faster than pip/poetry. Use if CI times become an issue. |

## Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | Latest | Cosine distance for semantic chunking | If implementing custom semantic chunking (scipy.spatial.distance.cosine). chonkie may handle this internally. |
| sentence-transformers | Latest | Embedding models for semantic chunking | If implementing custom semantic similarity chunking. chonkie may use this internally. Optional. |

**Recommendation:** Start without scipy/sentence-transformers. Add only if custom chunking logic needed beyond chonkie's capabilities.

## Alternatives Considered

| Category | Recommended | Alternative | When to Use Alternative |
|----------|-------------|-------------|-------------------------|
| CLI Framework | Typer | Click | Already using Click elsewhere; don't need type hints |
| CLI Framework | Typer | argparse | Zero dependencies required (stdlib only); very simple CLI |
| C Bindings | nanobind | pybind11 | Legacy project already using pybind11 (migration cost > perf gain) |
| C Bindings | nanobind | Cython | Need to write Python-like code for extensions (not wrapping C++) |
| Build Backend | scikit-build-core | meson-python | Project uses Meson already; scientific Python compliance prioritized |
| IR Metrics | ranx | pytrec_eval | Need exact TREC eval compatibility; don't care about speed |
| Formatting | Ruff | Black + isort | Team strongly prefers Black's specific opinionated style |
| LLM SDK | OpenAI/Anthropic direct | LiteLLM | Need multi-provider from day 1; cost tracking essential |
| Chunking | chonkie | Custom implementation | Highly specialized chunking logic; performance critical |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| RAGAS library | RAGAS is for end-to-end RAG evaluation (answer quality, faithfulness). BerryEval is for retrieval quality (IR metrics). Different domains. | ranx for IR metrics |
| LangChain for LLM calls | Adds abstraction overhead; BerryEval needs simple LLM calls for query generation, not agents/chains. | OpenAI/Anthropic SDK directly |
| setuptools for C extensions | Legacy build system; harder to configure; not PEP 621 compliant without setup.py. | scikit-build-core |
| distutils | Deprecated in Python 3.12+, removed in 3.14. | scikit-build-core |
| pybind11 (new projects) | nanobind is drop-in replacement with measurably better performance. Same author, actively maintained. | nanobind |
| flake8/black/isort separately | Ruff replaces all three, runs faster, single config. | Ruff |
| poetry for C extensions | Poetry's build system doesn't handle C compilation well; use as dependency manager but not build backend. | scikit-build-core as build backend |

## Installation

```bash
# Core dependencies
pip install typer[all] rich pydantic numpy ranx chonkie PyYAML openai anthropic

# Build dependencies (for C extensions)
pip install scikit-build-core[pyproject] nanobind

# Development dependencies
pip install ruff mypy pytest pytest-cov

# Optional: multi-provider LLM support
pip install litellm
```

**pyproject.toml structure:**

```toml
[build-system]
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"

[project]
name = "berryeval"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "typer[all]>=0.23.2",
    "rich>=14.3.2",
    "pydantic>=2.12.5",
    "numpy>=2.4.2",
    "ranx>=0.3.21",
    "chonkie>=1.5.2",
    "PyYAML>=6.0.3",
    "openai>=2.21.0",
    "anthropic>=0.79.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.15.1",
    "mypy>=1.19.1",
    "pytest>=9.0.2",
    "pytest-cov>=7.0.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "D"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
addopts = ["--import-mode=importlib", "--cov=berryeval", "--cov-report=term-missing"]
testpaths = ["tests"]

[tool.scikit-build]
wheel.packages = ["src/berryeval"]
cmake.build-type = "Release"
```

## Version Compatibility Matrix

| Component | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.14 |
|-----------|-------------|-------------|-------------|-------------|
| Typer | ✓ | ✓ | ✓ | ✓ |
| nanobind | ✓ | ✓ | ✓ (stable ABI) | ✓ |
| NumPy 2.4.2 | ✓ | ✓ | ✓ | ✓ |
| ranx | ✓ | ✓ | ✓ | Likely ✓ |
| PyYAML | ✓ | ✓ | ✓ | ✓ (6.0.3+) |
| Ruff | ✓ | ✓ | ✓ | ✓ |
| pytest 9.x | ✓ | ✓ | ✓ | ✓ |

**Recommendation:** Target Python 3.10-3.14 in CI matrix. Document Python 3.10+ as minimum requirement.

## Performance Considerations

### C Extension Optimization
- Compile with `-O3 -march=native` for 20-40% performance boost on metric kernels
- Use NumPy's BLAS/LAPACK backend (OpenBLAS recommended for cross-platform, MKL for Intel, Accelerate for Apple Silicon)
- Profile with `cProfile` before writing C extensions; many "slow" operations are I/O-bound, not CPU-bound

### Metric Computation
- ranx uses Numba JIT; first call slower (compilation), subsequent calls fast
- Batch metric computation when possible (ranx optimized for batch processing)
- For very large corpora (>1M docs), consider parallel evaluation across query batches

### LLM Cost Optimization
- Use prompt caching (Anthropic) for repeated corpus context
- Batch query generation requests when API supports batching
- Consider cheaper models for query generation vs evaluation (GPT-4o-mini vs GPT-4)

## Sources

**HIGH confidence sources (official docs, PyPI releases):**
- Typer: https://typer.tiangolo.com/, PyPI 0.23.2 (Feb 2026)
- nanobind: https://nanobind.readthedocs.io/, PyPI 2.11.0 (Jan 2026)
- ranx: https://github.com/AmenRa/ranx, PyPI 0.3.21 (Aug 2025)
- Ruff: https://docs.astral.sh/ruff/, PyPI 0.15.1 (Feb 2026)
- NumPy: PyPI 2.4.2 (Jan 2026)
- Rich: PyPI 14.3.2 (Feb 2026)
- Pydantic: PyPI 2.12.5 (Nov 2025)
- pytest: PyPI 9.0.2 (Dec 2025)
- mypy: PyPI 1.19.1 (Dec 2025)
- PyYAML: PyPI 6.0.3 (Sep 2025)
- OpenAI SDK: PyPI 2.21.0 (Feb 2026)
- Anthropic SDK: PyPI 0.79.0 (Feb 2026)
- pytest-cov: PyPI 7.0.0 (Sep 2025)
- scikit-build-core: PyPI 0.11.6 (Jul 2024)

**MEDIUM confidence sources (verified with multiple sources):**
- nanobind vs pybind11 benchmarks: https://nanobind.readthedocs.io/en/latest/benchmark.html
- ranx performance claims: Springer paper (ECIR 2022), GitHub repo
- Ruff adoption: https://simone-carolini.medium.com/modern-python-code-quality-setup-uv-ruff-and-mypy-8038c6549dcc
- chonkie: Libraries.io shows version 1.5.2 (Jan 2026), https://github.com/chonkie-inc/chonkie
- Python packaging trends 2025: https://medium.com/@dynamicy/python-build-backends-in-2025-what-to-use-and-why-uv-build-vs-hatchling-vs-poetry-core-94dd6b92248f

**LOW confidence / needs validation:**
- chonkie 85% speed claim (single WebSearch source, not verified in official docs)
- Exact ranx compatibility with Python 3.14 (likely works but not explicitly tested per search results)

---

*Stack research for: BerryEval RAG evaluation framework*
*Researched: 2026-02-16*
*Confidence: HIGH (all core technologies verified with official sources)*
