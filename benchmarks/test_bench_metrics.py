"""Benchmark tests for BerryEval metrics performance.

Run with: pytest benchmarks/ -v --benchmark-enable
"""

import statistics
import time

import numpy as np
import pytest

from berryeval.metrics._python import hit_rate as py_hit_rate
from berryeval.metrics._python import mrr as py_mrr
from berryeval.metrics._python import ndcg as py_ndcg
from berryeval.metrics._python import precision_at_k as py_precision_at_k
from berryeval.metrics._python import recall_at_k as py_recall_at_k

from .conftest import SCALES, make_benchmark_data

try:
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        hit_rate as c_hit_rate,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        mrr as c_mrr,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        ndcg as c_ndcg,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        precision_at_k as c_precision_at_k,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        recall_at_k as c_recall_at_k,
    )

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

METRICS = ["recall_at_k", "precision_at_k", "mrr", "ndcg", "hit_rate"]

PY_BACKENDS = {
    "recall_at_k": py_recall_at_k,
    "precision_at_k": py_precision_at_k,
    "mrr": py_mrr,
    "ndcg": py_ndcg,
    "hit_rate": py_hit_rate,
}

C_BACKENDS: dict = {}
if HAS_NATIVE:
    C_BACKENDS = {
        "recall_at_k": c_recall_at_k,
        "precision_at_k": c_precision_at_k,
        "mrr": c_mrr,
        "ndcg": c_ndcg,
        "hit_rate": c_hit_rate,
    }

TOP_K = 10


@pytest.mark.benchmark
@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("scale_name", list(SCALES.keys()))
def test_bench_python_backend(benchmark, metric_name, scale_name):
    """Benchmark pure Python backend."""
    n_queries = SCALES[scale_name]
    retrieved, relevant = make_benchmark_data(n_queries)
    fn = PY_BACKENDS[metric_name]
    benchmark(fn, retrieved, relevant, TOP_K)


@pytest.mark.benchmark
@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("scale_name", list(SCALES.keys()))
def test_bench_native_backend(benchmark, metric_name, scale_name):
    """Benchmark C backend."""
    if not HAS_NATIVE:
        pytest.skip("Native C backend not available")
    n_queries = SCALES[scale_name]
    retrieved, relevant = make_benchmark_data(n_queries)
    fn = C_BACKENDS[metric_name]
    benchmark(fn, retrieved, relevant, TOP_K)


@pytest.mark.benchmark
@pytest.mark.parametrize("metric_name", METRICS)
def test_native_speedup_over_python(metric_name):
    """Assert C is >= 5x faster than Python on 10K queries."""
    if not HAS_NATIVE:
        pytest.skip("Native C backend not available")

    retrieved, relevant = make_benchmark_data(10_000)
    py_fn = PY_BACKENDS[metric_name]
    c_fn = C_BACKENDS[metric_name]

    # Warm up both backends
    py_fn(retrieved, relevant, TOP_K)
    c_fn(retrieved, relevant, TOP_K)

    n_runs = 3

    # Time Python backend
    py_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        py_fn(retrieved, relevant, TOP_K)
        py_times.append(time.perf_counter() - start)

    # Time C backend
    c_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        c_fn(retrieved, relevant, TOP_K)
        c_times.append(time.perf_counter() - start)

    py_median = statistics.median(py_times)
    c_median = statistics.median(c_times)
    speedup = py_median / c_median

    print(
        f"\n  {metric_name}: Python={py_median:.4f}s, C={c_median:.4f}s, "
        f"speedup={speedup:.1f}x"
    )

    assert speedup >= 5.0, (
        f"{metric_name}: Expected >= 5x speedup, got {speedup:.1f}x "
        f"(Python={py_median:.4f}s, C={c_median:.4f}s)"
    )


@pytest.mark.benchmark
def test_100k_queries_under_30_seconds(data_100k):
    """Assert all 5 metrics on 100K queries complete in <30 seconds total."""
    from berryeval.metrics import hit_rate, mrr, ndcg, precision_at_k, recall_at_k

    retrieved, relevant = data_100k
    top_k = 50

    # Warm up
    small_ret, small_rel = make_benchmark_data(100)
    for fn in [recall_at_k, precision_at_k, mrr, ndcg, hit_rate]:
        fn(small_ret, small_rel, 10)

    start = time.perf_counter()
    recall_at_k(retrieved, relevant, top_k)
    precision_at_k(retrieved, relevant, top_k)
    mrr(retrieved, relevant, top_k)
    ndcg(retrieved, relevant, top_k)
    hit_rate(retrieved, relevant, top_k)
    elapsed = time.perf_counter() - start

    print(f"\n  100K queries, all 5 metrics: {elapsed:.2f}s")

    assert elapsed < 30.0, (
        f"Expected all metrics on 100K queries in <30s, took {elapsed:.2f}s"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("metric_name", METRICS)
def test_backends_produce_same_results(metric_name):
    """Verify C and Python backends produce identical results on 1K data."""
    if not HAS_NATIVE:
        pytest.skip("Native C backend not available")

    retrieved, relevant = make_benchmark_data(1_000)
    py_fn = PY_BACKENDS[metric_name]
    c_fn = C_BACKENDS[metric_name]

    py_result = py_fn(retrieved, relevant, TOP_K)
    c_result = c_fn(retrieved, relevant, TOP_K)

    np.testing.assert_allclose(
        c_result,
        py_result,
        atol=1e-5,
        err_msg=f"{metric_name}: C and Python backends produce different results",
    )
