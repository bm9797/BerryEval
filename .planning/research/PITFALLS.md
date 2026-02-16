# Pitfalls Research

**Domain:** RAG Evaluation Framework (Python + C Extensions)
**Researched:** 2026-02-16
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Reference Counting Errors at Python/C Boundary

**What goes wrong:**
Memory leaks or segmentation faults occur when reference counts get out of sync between Python and C code. Two failure modes: reference surplus (memory leaks) and reference deficit (use-after-free crashes). When passing NumPy arrays between Python and C, incorrect handling of borrowed vs. owned references causes the Python process to leak memory or crash when objects are cleaned up while still being referenced.

**Why it happens:**
Developers misunderstand the difference between borrowed and new references. Functions like `PyTuple_GetItem()` and `PyList_GetItem()` return borrowed references (don't call `Py_DECREF()`), while `PyLong_FromLong()` returns a new reference (must call `Py_DECREF()`). `PyList_Append()` increments the reference count, so if you create an object with refcount=1 and append it, it becomes refcount=2, leading to leaks when the list is destroyed. Error handling paths that return NULL without cleaning up acquired references compound the problem.

**How to avoid:**
1. Document reference ownership for every C API function (returns new vs. borrowed)
2. Use `Py_XDECREF()` in error paths to safely clean up even if object is NULL
3. Never call `Py_DECREF()` on borrowed references from getters
4. For objects added to containers, either steal the reference or explicitly DECREF after adding
5. Run tests with Python debug build (`--with-pydebug`) to catch reference leaks early
6. Use reference count debugging tools to detect leaks before they reach production

**Warning signs:**
- Memory usage grows continuously during repeated operations
- Crashes occur when arrays are garbage collected
- Different behavior between development and production Python builds
- Valgrind reports "still reachable" memory blocks
- Test suite memory usage increases linearly with test count

**Phase to address:**
Phase 1 (Core C Extension Foundation) - Establish reference counting discipline from the start. Every C function must document reference semantics. Add reference leak tests to CI.

---

### Pitfall 2: NPY_ARRAY_OWNDATA Flag Mismanagement

**What goes wrong:**
When wrapping external C memory in NumPy arrays, incorrectly setting the `OWNDATA` flag causes double-free crashes or memory leaks. The flag controls whether NumPy deallocates the underlying buffer. Setting it incorrectly means NumPy tries to free memory it doesn't own (crash) or fails to free memory it should (leak). On Windows, `PyArray_ENABLEFLAGS(NPY_OWNDATA)` causes crashes that don't occur on Linux/macOS.

**Why it happens:**
Developers assume `PyArray_NewFromDescr()` with external data will respect the OWNDATA flag passed in, but the flag is automatically reset to 0 when data is non-NULL. Using `PyArray_ENABLEFLAGS()` seems like a fix but causes platform-specific crashes because NumPy will use its own deallocator (incompatible with malloc/custom allocators). The proper approach (`PyArray_SetBaseObject()`) is non-obvious and poorly documented.

**How to avoid:**
1. **Never use `PyArray_ENABLEFLAGS(NPY_OWNDATA)` with external memory**
2. Use `PyArray_SetBaseObject()` to set a Python object that owns the memory
3. Create a custom Python capsule with a destructor for C-allocated memory
4. Test on Windows, Linux, and macOS - memory ownership bugs are platform-specific
5. Document memory ownership clearly: "Who allocated? Who frees? When?"
6. For C-allocated buffers, create a PyCapsule with a destructor that calls the appropriate free function

**Warning signs:**
- Crashes during array deallocation (garbage collection)
- Different behavior across operating systems
- Valgrind reports double-free or invalid free
- Memory leaks when arrays are deleted
- Crashes that only appear under GC pressure

**Phase to address:**
Phase 1 (Core C Extension Foundation) - Establish memory ownership patterns before building higher-level features. Create wrapper utilities that encapsulate the correct `PyArray_SetBaseObject()` pattern.

---

### Pitfall 3: Synthetic Ground Truth Bias and Quality Degradation

**What goes wrong:**
Synthetic ground truth data fails to represent real-world query complexity, introducing systematic biases that make evaluation metrics unreliable. Generated datasets lack linguistic diversity, long-tail query patterns, and domain-specific edge cases. The framework reports high scores on synthetic data but performs poorly on real user queries. Worse, bias in synthetic data (representation, schema, diversity) reinforces existing inequalities and leads to overconfident incorrect conclusions about retrieval quality.

**Why it happens:**
LLM-generated synthetic data reflects the model's training distribution, not production query distribution. Simple template-based generation creates artificial patterns that are easier to retrieve than real queries. Developers skip validation of synthetic data quality because creating human-verified test sets is expensive (days to weeks of expert time). The "reference data is gold standard" assumption is false - synthetic ground truth is rarely perfect and errors compound during evaluation.

**How to avoid:**
1. **Validate synthetic data against real production queries** - measure distribution drift
2. Create hybrid test sets: 70% synthetic for coverage, 30% human-verified for quality anchoring
3. Explicitly test for bias dimensions: entity diversity, query complexity distribution, domain coverage
4. Include adversarial examples: typos, uncommon entities, overlapping intents, ambiguous queries
5. Track synthetic data generation parameters and version datasets
6. Regularly audit synthetic data for quality issues: inconsistent annotations, copy-pasted answers, insufficient sources
7. Never tune hyperparameters solely on synthetic data - require real-world validation

**Warning signs:**
- High metrics on synthetic data, low metrics on real queries
- Synthetic queries are suspiciously similar in structure
- Generated answers are shorter/simpler than real-world ground truth
- Metrics improve when adding more synthetic data but real performance stagnates
- Test set contains obvious patterns (all queries start with "What is...")

**Phase to address:**
Phase 2 (Synthetic Ground Truth Generation) - Build quality gates into generation pipeline. Add diversity metrics, human review sampling, and real-world validation before declaring generation "done."

---

### Pitfall 4: Memory Alignment and Contiguity Assumptions

**What goes wrong:**
C code assumes NumPy arrays are C-contiguous and properly aligned, but receives transposed, sliced, or strided views. Accessing misaligned data triggers `SIGBUS` crashes. Non-contiguous arrays passed to C functions expecting contiguous memory produce incorrect results or buffer overruns. Common in production pipelines where reshaping, transposing, or slicing creates non-contiguous views.

**Why it happens:**
NumPy operations like transpose, reshape, and slicing create views (not copies) with different memory layout. C code written for simple cases (`array[i][j]` pointer arithmetic) breaks with non-standard strides. The CPU requires data types to start at memory addresses that are multiples of their size - misalignment causes hardware exceptions. Developers test with freshly created arrays (always C-contiguous) and miss the edge cases until production.

**How to avoid:**
1. **Always check `PyArray_IS_C_CONTIGUOUS()` before passing to C**
2. Use `PyArray_FROM_OTF()` with `NPY_ARRAY_C_CONTIGUOUS` flag to force copy if needed
3. Check alignment with `PyArray_ISALIGNED()` before dereferencing pointers
4. Document function preconditions: "Requires C-contiguous, aligned arrays"
5. Add tests with transposed, sliced, and reshaped array views
6. Use `array.flags['C_CONTIGUOUS']` and `array.flags['ALIGNED']` in Python tests
7. Consider using NumPy iterators (`NpyIter`) which handle arbitrary strides correctly

**Warning signs:**
- Works with `np.array()` but crashes with `array.T` or `array[::2]`
- SIGBUS or SIGSEGV crashes in C code
- Platform-specific failures (alignment requirements differ)
- Incorrect results only with certain array shapes
- Crashes appear after adding preprocessing steps

**Phase to address:**
Phase 1 (Core C Extension Foundation) - Establish array validation patterns before implementing algorithms. Every C function accepting arrays must validate contiguity/alignment.

---

### Pitfall 5: Component-Level Evaluation Blind Spots

**What goes wrong:**
Evaluating only the end-to-end RAG pipeline makes debugging impossible. When metrics drop, you don't know if retrieval failed, generation failed, or the interaction between them broke. A perfect retriever paired with a hallucinating generator produces useless output, but you can't tell which component to fix. Testing complete pipelines without measuring individual components wastes debugging time and obscures root causes.

**Why it happens:**
End-to-end metrics are easier to implement (one score instead of separate retrieval/generation scores). Developers focus on the final answer quality and skip intermediate validation. Lack of proper component interfaces makes it hard to evaluate pieces independently. The pipeline is tightly coupled, making it difficult to substitute components for testing.

**How to avoid:**
1. **Separate metrics for retrieval and generation** - measure both independently
2. Track intermediate states: retrieved documents, reranked results, generated answers
3. Create "oracle" tests: perfect retrieval with real generator, real retrieval with oracle answers
4. Log component-level metrics alongside end-to-end metrics
5. Build interfaces that allow running each component in isolation
6. Add regression detection: alert when retrieval metrics drop even if end-to-end stays stable
7. Visualize metric breakdown: if end-to-end drops 10%, which component caused it?

**Warning signs:**
- Can't explain why metrics improved/degraded after changes
- Debugging requires re-running full pipeline repeatedly
- Small changes to one component affect all metrics unpredictably
- No visibility into which component is the bottleneck
- Cannot reproduce issues without full pipeline context

**Phase to address:**
Phase 3 (Retrieval Metrics Implementation) - Design metrics architecture with component separation from the start. Each metric should be computable on isolated components.

---

### Pitfall 6: Cross-Platform Build System Fragility

**What goes wrong:**
C extensions build successfully on the development machine but fail on CI or user systems due to platform-specific dependencies, compiler differences, or missing build tools. cibuildwheel generates pure Python wheels instead of binary wheels because the C extension wasn't detected. Tests pass locally but fail when run from the installed wheel. Windows-specific build failures don't appear until users report crashes.

**Why it happens:**
Cross-compilation support in Python packaging is unofficial and best-effort, prone to bitrot unless exercised regularly. Different platforms have different compilers (gcc vs. clang vs. MSVC) with different flags and behaviors. setuptools/distutils removal in Python 3.12 broke legacy build configurations. Custom `setup.cfg` options affect cibuildwheel's dependency installation. Platform-specific paths and library naming conventions create subtle failures. Developers test only on their local platform.

**How to avoid:**
1. **Test builds on all target platforms from day one** - don't wait for first release
2. Use cibuildwheel to standardize wheel building across platforms
3. Add CI jobs for Windows, macOS (Intel + ARM), and Linux (multiple distros)
4. Test installation from the built wheel, not just editable install
5. Use `CIBW_BUILD` to target specific Python versions during debugging
6. Document all build dependencies in `pyproject.toml` and CI configs
7. Set up local Docker testing for Linux builds before pushing to CI
8. Monitor for "pure Python wheel" warnings - indicates C extension wasn't compiled

**Warning signs:**
- "Works on my machine" but CI fails
- Platform-specific test failures
- Users report import errors or missing symbols
- Wheel size is suspiciously small (pure Python instead of binary)
- Build failures with "unsupported platform" errors

**Phase to address:**
Phase 1 (Core C Extension Foundation) - Set up cross-platform CI before writing substantial C code. Establish the build infrastructure early to catch issues fast.

---

### Pitfall 7: Edge Case Metric Blindness

**What goes wrong:**
Standard retrieval metrics (Recall@k, MRR, NDCG) fail on edge cases: multi-source queries where partial answers are in multiple documents, queries with typos, uncommon entities, overlapping intents, and near-duplicate documents. The framework reports high scores but misses critical failure modes. Position-based metrics ignore that LLMs have context length limits - if the best document is at position 10, the model might not see it.

**Why it happens:**
Developers test on clean, well-formed queries from academic benchmarks. Real-world queries are messy: typos, ambiguity, partial information needs. Standard metrics assume single correct answers; many complex questions have multiple valid answers or require synthesizing across documents. Ranking quality matters more than presence - a relevant document at rank 20 is effectively invisible. Test sets lack adversarial examples and edge cases because they're expensive to create.

**How to avoid:**
1. **Create edge case test suites explicitly** - typos, rare entities, multi-hop queries, ambiguous intents
2. Measure position-aware metrics: Recall@5 vs Recall@20 shows if good documents are buried
3. Test with near-duplicate documents to ensure ranking is robust
4. Include queries requiring multi-document synthesis
5. Add adversarial examples: queries designed to trigger failures
6. Track failure modes separately: "typo queries" vs "clean queries"
7. Use MRR to penalize when relevant docs appear late in ranking
8. Test with varying context window sizes to simulate LLM limitations

**Warning signs:**
- High metrics on benchmarks, user complaints about quality
- Performance degrades on real-world queries
- Metrics are stable but users report specific failure patterns
- Edge cases discovered during demos or production
- No differentiation between "clean" and "messy" query performance

**Phase to address:**
Phase 3 (Retrieval Metrics Implementation) - Build edge case testing into metric validation. Every metric should have a test case where it catches a known failure mode.

---

### Pitfall 8: LLM Judge Bias in Evaluation

**What goes wrong:**
When using LLMs to evaluate retrieval quality or answer correctness, systematic biases corrupt the results. LLM judges prefer longer responses, exhibit positional bias (favor first or last options), and show self-preference (rate their own outputs higher). Different judge models produce inconsistent scores. Using LLMs for ground truth generation and evaluation creates circular validation. Grading scales are inconsistent across use cases.

**Why it happens:**
LLMs optimize for human preference, which correlates with length and verbosity. Positional encoding in transformers creates position-dependent biases. Self-generated text is "closer" to the model's distribution, triggering higher scores. Different models have different calibrations and thresholds. Developers use LLM judges because human evaluation is expensive, but don't validate judge reliability. Lack of standardized grading scales makes cross-model comparison impossible.

**How to avoid:**
1. **Validate LLM judges against human judgments** - measure agreement and calibrate
2. Use multiple judge models and ensemble their scores
3. Randomize option positions to detect and correct positional bias
4. Normalize for response length when length shouldn't matter
5. Blind the judge to which model generated the response
6. Use structured grading rubrics, not free-form scoring
7. Track judge consistency: same input should yield same score
8. Never use the same LLM for generation and evaluation without human validation
9. Report inter-judge agreement alongside metrics

**Warning signs:**
- Judge scores correlate strongly with response length
- Different judge models rank options in opposite orders
- Scores change when options are reordered
- Judge prefers responses from specific models consistently
- Human reviewers disagree with judge assessments

**Phase to address:**
Phase 4 (Quality Metrics - LLM-as-Judge) - Build validation infrastructure before deploying LLM judges. Establish bias detection and correction as part of metric implementation.

---

### Pitfall 9: Retriever Adapter Abstraction Leaks

**What goes wrong:**
The adapter interface that wraps different retriever backends (Elasticsearch, Pinecone, custom vector DBs) leaks implementation details, forcing users to understand underlying complexity. Error handling differs across backends - PostgreSQL exceptions vs MySQL exceptions vs Pinecone errors - breaking the abstraction. Performance characteristics vary wildly (some backends support batch operations, others don't), making it impossible to write performant backend-agnostic code. Certain operations become inefficient when the adapter hides critical backend features.

**Why it happens:**
Different retriever backends have fundamentally different capabilities: some support filtering, some don't; some have batch APIs, some require per-item calls; some handle pagination, some require manual cursor management. Trying to create a common interface either limits functionality to the lowest common denominator or exposes backend-specific details. Error types are backend-specific. Developers want a simple API but retrievers are too heterogeneous.

**How to avoid:**
1. **Accept that perfect abstraction is impossible** - design for graceful degradation
2. Define capability flags: `supports_filtering`, `supports_batch`, `supports_reranking`
3. Provide backend-specific extension points for advanced features
4. Normalize error types into framework-specific exceptions with backend info attached
5. Document performance characteristics per backend: "Pinecone: batch=1000x faster"
6. Offer "fast path" and "compatible path" for operations
7. Test every adapter with the same test suite to verify behavioral consistency
8. Make the adapter layer thin - don't hide too much

**Warning signs:**
- Users bypass the adapter to call backend APIs directly
- Performance varies 10-100x across backends for same operation
- Error handling code is full of `if backend == "elasticsearch"` branches
- Feature requests are mostly "expose X from backend Y"
- Adapter interface grows continuously to accommodate backend features

**Phase to address:**
Phase 5 (Retriever Integration Layer) - Design adapter interface with explicit capability negotiation. Test all backends against the same behavioral contract from day one.

---

### Pitfall 10: Performance Testing with Toy Data

**What goes wrong:**
The framework meets the 100K queries in <30s target with test data (1K documents, simple queries), but crawls to a halt in production (50M documents, complex multi-hop queries). Quadratic algorithms are easy to create by mistake and run quickly with small test data but catastrophically slowly with real-world scale. Database queries without indexes perform acceptably at 10K rows but time out at 10M rows. No one tests at production scale until users complain.

**Why it happens:**
Test data is small (quick to generate, fast tests), production data is huge. Algorithmic complexity differences (O(n) vs O(n²)) don't matter at small scale. Database query plans are different for small vs large tables. Developers optimize for test suite speed, not real-world performance. Setting up realistic test data is expensive. Performance regressions are caught only when users complain.

**How to avoid:**
1. **Establish realistic scale test data from day one** - if production is 50M docs, test with 1M+
2. Generate data at multiple scales: 1K, 100K, 1M, 10M to detect scaling inflection points
3. Profile with realistic query distributions, not uniform random queries
4. Add performance regression tests to CI at representative scale
5. Use algorithmic complexity analysis - flag O(n²) before implementing
6. Test with production-like data characteristics: document length distribution, vocabulary size
7. Include "stress tests" that deliberately exceed target scale
8. Monitor metrics at different scales: 10K, 100K, 1M, 10M queries

**Warning signs:**
- Performance tests run on 1K documents but production has 1M+
- Test queries are simpler than production queries
- Algorithmic complexity is unknown or undocumented
- No performance tests in CI (only functional tests)
- Performance degrades non-linearly with scale
- Database query plans differ between test and production

**Phase to address:**
Phase 7 (Performance Optimization) - Establish scale testing before declaring performance goals met. Generate realistic-scale datasets and measure against actual targets.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip reference count documentation in C code | Faster initial development | Memory leaks that are nearly impossible to debug later | Never - document from day one |
| Use PyArray_ENABLEFLAGS(OWNDATA) instead of PyArray_SetBaseObject | Simpler code, fewer lines | Platform-specific crashes, memory corruption | Never - always use SetBaseObject |
| Generate synthetic data without validation | Fast test set creation | Biased evaluation, overconfident metrics | Only for initial prototyping, must validate before release |
| Test only on developer's platform | Faster iteration | Cross-platform failures discovered by users | Only in early exploration, set up CI within first week |
| Single end-to-end metric instead of component metrics | Simpler metric tracking | Impossible debugging when metrics degrade | Only in proof-of-concept phase |
| Use LLM judge without human validation | Automated evaluation at scale | Systematic bias, unreliable scores | Only for initial experiments, require validation before trusting |
| Small test datasets (1K docs) | Fast tests, quick CI | Scalability issues invisible until production | Acceptable for unit tests, unacceptable for performance validation |
| Assume C-contiguous arrays in C code | Cleaner C code | Crashes with transposed/sliced arrays | Only if explicitly validated at boundary |
| Single retriever backend during development | Faster initial implementation | Abstraction leaks discovered late | Acceptable for MVP, add 2nd backend before Phase 5 |
| Skip edge case test sets | Focus on happy path | Critical failures in production | Only in initial prototype, build edge cases by Phase 3 |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| NumPy C API | Treating all array references as owned | Document borrowed vs new references for every function, use Py_XDECREF in error paths |
| Retriever backends | Assuming identical error types | Normalize errors into framework exceptions, preserve backend details in exception metadata |
| LLM APIs for judges | Using same model for generation and evaluation | Use different models, validate against human judgments, blind evaluation |
| cibuildwheel | Assuming local builds predict wheel builds | Test actual wheel installation in CI, don't use editable installs in CI |
| NumPy memory layout | Assuming C-contiguous arrays | Check PyArray_IS_C_CONTIGUOUS before C calls, use PyArray_FROM_OTF with flags |
| Vector databases | Writing backend-agnostic code without capability flags | Define capability interface, test all backends with same test suite |
| Test data generation | Assuming synthetic data represents real queries | Validate distribution, include real-world samples, measure drift |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Linear scan instead of indexes | Acceptable test performance, production timeouts | Profile query plans at production scale, add indexes proactively | >100K documents or >10K queries |
| Quadratic complexity in retrieval scoring | Fast on test data (1K docs), crawls with real data | Analyze algorithmic complexity, use asymptotic notation | >10K documents |
| Per-item API calls instead of batch | Works fine with small test sets | Use batch APIs when available, measure batch vs single-item perf | >1K operations per second |
| Unbounded result sets | Small test data returns quickly | Always paginate, set max limits, use cursors for large result sets | >10K results or >1GB data |
| Memory copying at Python/C boundary | Negligible with small arrays | Use PyArray views when possible, minimize copies | Arrays >10MB, frequent calls |
| String concatenation in tight loops | Fast with short strings | Use buffer arrays or StringIO for accumulation | >1000 concatenations |
| No connection pooling for retrievers | Fine with single-threaded tests | Implement connection pooling, configure limits | >10 concurrent requests |
| Synchronous LLM API calls | Acceptable for sequential evaluation | Use async/await for LLM judge calls, batch when possible | >100 evaluations |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Passing unsanitized queries to retriever SQL backends | SQL injection via query strings | Use parameterized queries, validate/sanitize inputs, escape special chars |
| Allowing arbitrary Python code in synthetic generation prompts | Code injection, arbitrary execution | Sandbox LLM outputs, validate generated code, never eval() untrusted strings |
| Exposing raw retriever credentials in adapter errors | Credential leakage in logs/stack traces | Sanitize error messages, use credential management, redact sensitive fields |
| Storing ground truth with PII in version control | Data breach, compliance violations | Anonymize test data, use synthetic PII, add .gitignore rules |
| Unbounded memory allocation based on query input | DoS via memory exhaustion | Limit max query size, result set size, array dimensions |
| Including API keys in distributed wheels | Credential exposure | Use environment variables, config files outside package, rotate keys |
| Logging full LLM responses with user data | Privacy violations | Redact PII from logs, use sampling, implement log retention policies |

---

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Cryptic C extension import errors | Users can't diagnose build failures | Provide clear error messages, link to troubleshooting docs, detect platform issues |
| No progress indication for 100K query benchmarks | Users think process hung | Add progress bars, ETA calculation, intermediate logging |
| Overwhelming metric output (50+ numbers) | Users don't know which metrics matter | Highlight primary metrics, offer detailed vs summary views, explain metric meaning |
| Requiring deep NumPy knowledge to use C APIs | High barrier to entry, contribution friction | Provide Python wrapper utilities, document memory semantics, offer examples |
| Different retriever backends require different setup steps | Configuration confusion, setup failures | Unified config interface, auto-detection of available backends, clear error messages |
| No guidance on metric interpretation | Users don't know if scores are "good" | Provide baseline comparisons, score ranges, interpretation guidelines |
| Failure messages don't indicate which component failed | Difficult debugging | Include component breadcrumbs in errors, suggest diagnostic steps |
| No validation that retriever adapter is working correctly | Silent failures, incorrect results | Auto-validation on first use, smoke tests in adapter init, health checks |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **C Extension Memory Management:** Often missing Py_XDECREF in error paths - verify every return NULL has cleanup
- [ ] **NumPy Array Wrapping:** Often missing PyArray_SetBaseObject - verify memory ownership is explicit
- [ ] **Cross-Platform Builds:** Often missing Windows/macOS tests - verify CI includes all platforms
- [ ] **Synthetic Ground Truth:** Often missing human validation - verify quality metrics and real-world comparison
- [ ] **Component-Level Metrics:** Often missing isolation tests - verify each component can be evaluated independently
- [ ] **Edge Case Coverage:** Often missing adversarial examples - verify test set includes typos, rare entities, multi-hop
- [ ] **Performance at Scale:** Often missing realistic data volume - verify tests run at production scale (1M+ docs)
- [ ] **LLM Judge Validation:** Often missing human agreement measurement - verify judge correlates with human raters
- [ ] **Retriever Adapter Errors:** Often missing normalized error handling - verify backend exceptions are caught and wrapped
- [ ] **Memory Alignment Checks:** Often missing contiguity validation - verify PyArray_IS_C_CONTIGUOUS before C calls
- [ ] **Reference Count Documentation:** Often missing ownership semantics - verify every C function documents borrowed/new refs
- [ ] **Build from Wheel Tests:** Often missing installed package tests - verify CI tests installation, not just editable mode

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Reference counting errors | HIGH | Run with Python debug build, use reference leak detection tools, audit all Py_INCREF/DECREF pairs, add systematic review |
| OWNDATA flag misuse | MEDIUM | Replace PyArray_ENABLEFLAGS with PyArray_SetBaseObject, create PyCapsule with destructor, test on all platforms |
| Synthetic data bias | MEDIUM | Collect real queries, create human-verified subset, re-benchmark, adjust generation parameters, add diversity metrics |
| Memory alignment crashes | LOW | Add contiguity checks at boundaries, use PyArray_FROM_OTF with flags, test with views/slices/transposes |
| Component eval blindness | MEDIUM | Refactor to expose component interfaces, add intermediate logging, create oracle tests, track separate metrics |
| Cross-platform build failures | HIGH | Set up cibuildwheel CI, test on all platforms, fix platform-specific code, add build documentation |
| Edge case failures | MEDIUM | Create edge case test suite, measure per-category performance, add adversarial examples, tune for failures |
| LLM judge bias | MEDIUM | Validate against humans, use multiple judges, normalize for length/position, document bias characteristics |
| Abstraction leaks | LOW-MEDIUM | Add capability flags, expose extension points, document backend differences, accept leaky abstraction |
| Scale performance issues | HIGH | Profile at scale, identify bottlenecks (usually O(n²) or missing indexes), optimize hot paths, add scale tests |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Reference counting errors | Phase 1 - Core C Extension | Run Python debug build tests, reference leak detection passes, Valgrind clean |
| OWNDATA flag misuse | Phase 1 - Core C Extension | All platforms tested, PyCapsule pattern documented, memory tests pass |
| Synthetic data bias | Phase 2 - Synthetic Ground Truth | Human validation shows >90% agreement, diversity metrics within range, real-world comparison |
| Memory alignment crashes | Phase 1 - Core C Extension | Tests include transposed/sliced arrays, contiguity checks in place, no SIGBUS |
| Component eval blindness | Phase 3 - Retrieval Metrics | Each component has independent metrics, oracle tests exist, metric breakdown tracked |
| Cross-platform builds | Phase 1 - Core C Extension | CI passes on Windows/macOS/Linux, wheel installation tested, platform matrix complete |
| Edge case failures | Phase 3 - Retrieval Metrics | Edge case test suite exists, per-category metrics tracked, adversarial examples included |
| LLM judge bias | Phase 4 - Quality Metrics | Human agreement measured (>80%), multiple judges compared, bias correction applied |
| Abstraction leaks | Phase 5 - Retriever Integration | Capability flags defined, 2+ backends tested, error normalization works, documentation complete |
| Scale performance issues | Phase 7 - Performance Optimization | 100K queries in <30s target met, tested at 10x target scale, profiling shows no bottlenecks |

---

## Sources

### RAG Evaluation Framework Research
- [RAG Evaluation: 2026 Metrics and Benchmarks](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation)
- [DeepEval: RAG Evaluation Guide](https://deepeval.com/guides/guides-rag-evaluation)
- [Patronus: RAG Evaluation Metrics Best Practices](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
- [Evidently AI: Complete Guide to RAG Evaluation](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Pinecone: RAG Evaluation Best Practices](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [Superlinked: Evaluating RAG Framework for Assessment](https://superlinked.com/vectorhub/articles/evaluating-retrieval-augmented-generation-framework)
- [Braintrust: RAG Evaluation Metrics Guide](https://www.braintrust.dev/articles/rag-evaluation-metrics)

### Python C Extension Memory Management
- [NumPy: Memory Management in C API](https://numpy.org/doc/stable/reference/c-api/data_memory.html)
- [C Extension Tutorial: Common Issues and Bugs](https://llllllllll.github.io/c-extension-tutorial/common-issues.html)
- [Python Extension Patterns: Reference Counting](https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html)
- [PSU: Finding Reference-Counting Errors (PDF)](https://www.cse.psu.edu/~gxt29/papers/refcount.pdf)
- [NumPy GitHub: NPY_OWNDATA Crash Issue](https://github.com/numpy/numpy/issues/8253)
- [NumPy: Array API Documentation](https://numpy.org/doc/stable/reference/c-api/array.html)

### Synthetic Ground Truth Problems
- [Springer: Synthetic Ground Truth Counterfactuals](https://link.springer.com/chapter/10.1007/978-3-032-04984-1_52)
- [Sustainability Methods: Ground Truth Challenges](https://sustainabilitymethods.org/index.php/Ground_Truth)
- [Medium: How to Evaluate RAG Without Ground Truth Data](https://medium.com/data-science/how-to-evaluate-rag-if-you-dont-have-ground-truth-data-590697061d89)
- [ACM: Evaluating Model Explanations Without Ground Truth](https://dl.acm.org/doi/10.1145/3715275.3732219)
- [GitHub: Data Contamination Research](https://github.com/lyy1994/awesome-data-contamination)
- [arXiv: Detecting Benchmark Contamination](https://arxiv.org/html/2502.17259)
- [Hugging Face: RAG Evaluation Cookbook](https://huggingface.co/learn/cookbook/en/rag_evaluation)

### Cross-Platform Build Systems
- [PEP 720: Cross-Compiling Python Packages](https://peps.python.org/pep-0720/)
- [Python Docs: Building C Extensions](https://docs.python.org/3/extending/building.html)
- [EuroPython 2021: Python C Extensions and Cross-Platform Wheels](https://ep2021.europython.eu/talks/5gVwmkx-a-tale-of-python-c-extensions-and-cross-platform-wheels/)
- [pypackaging-native: Cross Compilation Guide](https://pypackaging-native.github.io/key-issues/cross_compilation/)
- [GitHub cibuildwheel: Issue #937 - macOS Platform Failures](https://github.com/pypa/cibuildwheel/issues/937)
- [GitHub cibuildwheel: Issue #1487 - setup.cfg Errors](https://github.com/pypa/cibuildwheel/issues/1487)

### Abstraction Leaks
- [Joel Spolsky: Law of Leaky Abstractions](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)
- [Wikipedia: Leaky Abstraction](https://en.wikipedia.org/wiki/Leaky_abstraction)
- [Embedded Artistry: Leaky Abstraction Definition](https://embeddedartistry.com/fieldmanual-terms/leaky-abstraction/)

### Performance Testing
- [Medium: 100 Backend Performance Bottlenecks Analysis](https://medium.com/@the_unwritten_algorithm/i-analyzed-100-backend-performance-bottlenecks-they-all-made-the-same-5-mistakes-c60877fde1e2)
- [Python Speed: Measuring Performance in Production](https://pythonspeed.com/articles/measure-performance-production/)
- [BrowserStack: Performance Bottleneck Guide](https://www.browserstack.com/guide/performance-bottleneck)
- [Gatling: Performance Bottlenecks and How to Avoid Them](https://gatling.io/blog/performance-bottlenecks-common-causes-and-how-to-avoid-them)

---
*Pitfalls research for: BerryEval RAG Evaluation Framework*
*Researched: 2026-02-16*
*Confidence: HIGH - Based on official documentation, academic research, and community post-mortems*
