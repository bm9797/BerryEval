#ifndef BERRY_METRICS_H
#define BERRY_METRICS_H

#include <stdint.h>

#define BERRY_PADDING_ID (-1)

/*
 * All kernel functions operate on 2-D row-major arrays laid out as flat
 * buffers.  Each row has a fixed width (n_retrieved for retrieved,
 * n_relevant for relevant).  The output buffer `out` has length n_queries.
 */

/* Return the 1-based rank of the first element of `relevant` found in
   `retrieved` (up to position k).  Returns 0 when no match is found. */
int berry_rank_lookup(const int32_t *retrieved, int n_retrieved,
                      const int32_t *relevant, int n_relevant, int k);

void berry_recall_at_k(const int32_t *retrieved, const int32_t *relevant,
                       float *out, int n_queries, int n_retrieved,
                       int n_relevant, int k);

void berry_precision_at_k(const int32_t *retrieved, const int32_t *relevant,
                          float *out, int n_queries, int n_retrieved,
                          int n_relevant, int k);

void berry_mrr(const int32_t *retrieved, const int32_t *relevant, float *out,
               int n_queries, int n_retrieved, int n_relevant, int k);

void berry_ndcg(const int32_t *retrieved, const int32_t *relevant, float *out,
                int n_queries, int n_retrieved, int n_relevant, int k);

void berry_hit_rate(const int32_t *retrieved, const int32_t *relevant,
                    float *out, int n_queries, int n_retrieved, int n_relevant,
                    int k);

#endif /* BERRY_METRICS_H */
