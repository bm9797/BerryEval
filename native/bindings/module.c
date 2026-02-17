#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "metrics.h"

/* ------------------------------------------------------------------ */
/* Helper: extract pointers and dimensions common to every wrapper.   */
/* ------------------------------------------------------------------ */

static int
parse_metric_args(PyObject *args, PyArrayObject **retrieved,
                  PyArrayObject **relevant, int *k, int *n_queries,
                  int *n_retrieved, int *n_relevant,
                  const int32_t **ret_data, const int32_t **rel_data)
{
    if (!PyArg_ParseTuple(args, "O!O!i",
                          &PyArray_Type, retrieved,
                          &PyArray_Type, relevant,
                          k))
        return -1;

    *n_queries   = (int)PyArray_DIM(*retrieved, 0);
    *n_retrieved = (int)PyArray_DIM(*retrieved, 1);
    *n_relevant  = (int)PyArray_DIM(*relevant, 1);
    *ret_data    = (const int32_t *)PyArray_DATA(*retrieved);
    *rel_data    = (const int32_t *)PyArray_DATA(*relevant);
    return 0;
}

static PyObject *
make_output(int n_queries)
{
    npy_intp dims[1] = { n_queries };
    return PyArray_SimpleNew(1, dims, NPY_FLOAT32);
}

/* ------------------------------------------------------------------ */
/* Wrappers                                                            */
/* ------------------------------------------------------------------ */

static PyObject *
py_recall_at_k(PyObject *self, PyObject *args)
{
    (void)self;
    PyArrayObject *retrieved, *relevant;
    int k, n_queries, n_retrieved, n_relevant;
    const int32_t *ret_data, *rel_data;

    if (parse_metric_args(args, &retrieved, &relevant, &k,
                          &n_queries, &n_retrieved, &n_relevant,
                          &ret_data, &rel_data) < 0)
        return NULL;

    PyObject *out = make_output(n_queries);
    if (!out) return NULL;
    float *out_data = (float *)PyArray_DATA((PyArrayObject *)out);

    berry_recall_at_k(ret_data, rel_data, out_data,
                      n_queries, n_retrieved, n_relevant, k);
    return out;
}

static PyObject *
py_precision_at_k(PyObject *self, PyObject *args)
{
    (void)self;
    PyArrayObject *retrieved, *relevant;
    int k, n_queries, n_retrieved, n_relevant;
    const int32_t *ret_data, *rel_data;

    if (parse_metric_args(args, &retrieved, &relevant, &k,
                          &n_queries, &n_retrieved, &n_relevant,
                          &ret_data, &rel_data) < 0)
        return NULL;

    PyObject *out = make_output(n_queries);
    if (!out) return NULL;
    float *out_data = (float *)PyArray_DATA((PyArrayObject *)out);

    berry_precision_at_k(ret_data, rel_data, out_data,
                         n_queries, n_retrieved, n_relevant, k);
    return out;
}

static PyObject *
py_mrr(PyObject *self, PyObject *args)
{
    (void)self;
    PyArrayObject *retrieved, *relevant;
    int k, n_queries, n_retrieved, n_relevant;
    const int32_t *ret_data, *rel_data;

    if (parse_metric_args(args, &retrieved, &relevant, &k,
                          &n_queries, &n_retrieved, &n_relevant,
                          &ret_data, &rel_data) < 0)
        return NULL;

    PyObject *out = make_output(n_queries);
    if (!out) return NULL;
    float *out_data = (float *)PyArray_DATA((PyArrayObject *)out);

    berry_mrr(ret_data, rel_data, out_data,
              n_queries, n_retrieved, n_relevant, k);
    return out;
}

static PyObject *
py_ndcg(PyObject *self, PyObject *args)
{
    (void)self;
    PyArrayObject *retrieved, *relevant;
    int k, n_queries, n_retrieved, n_relevant;
    const int32_t *ret_data, *rel_data;

    if (parse_metric_args(args, &retrieved, &relevant, &k,
                          &n_queries, &n_retrieved, &n_relevant,
                          &ret_data, &rel_data) < 0)
        return NULL;

    PyObject *out = make_output(n_queries);
    if (!out) return NULL;
    float *out_data = (float *)PyArray_DATA((PyArrayObject *)out);

    berry_ndcg(ret_data, rel_data, out_data,
               n_queries, n_retrieved, n_relevant, k);
    return out;
}

static PyObject *
py_hit_rate(PyObject *self, PyObject *args)
{
    (void)self;
    PyArrayObject *retrieved, *relevant;
    int k, n_queries, n_retrieved, n_relevant;
    const int32_t *ret_data, *rel_data;

    if (parse_metric_args(args, &retrieved, &relevant, &k,
                          &n_queries, &n_retrieved, &n_relevant,
                          &ret_data, &rel_data) < 0)
        return NULL;

    PyObject *out = make_output(n_queries);
    if (!out) return NULL;
    float *out_data = (float *)PyArray_DATA((PyArrayObject *)out);

    berry_hit_rate(ret_data, rel_data, out_data,
                   n_queries, n_retrieved, n_relevant, k);
    return out;
}

/* ------------------------------------------------------------------ */
/* Module definition                                                   */
/* ------------------------------------------------------------------ */

static PyMethodDef methods[] = {
    {"recall_at_k",    py_recall_at_k,    METH_VARARGS, "Compute recall@k."},
    {"precision_at_k", py_precision_at_k, METH_VARARGS, "Compute precision@k."},
    {"mrr",            py_mrr,            METH_VARARGS, "Compute MRR."},
    {"ndcg",           py_ndcg,           METH_VARARGS, "Compute nDCG."},
    {"hit_rate",       py_hit_rate,       METH_VARARGS, "Compute hit rate."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "berryeval.metrics._native",
    "C-accelerated IR metric kernels for BerryEval.",
    -1,
    methods,
};

PyMODINIT_FUNC
PyInit__native(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
