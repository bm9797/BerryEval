#include "metrics.h"

#include <math.h>
#include <stdint.h>

/* ---------- helpers --------------------------------------------------- */

/* Check if val appears in arr[0..len-1], skipping PADDING_ID entries. */
static int is_in_relevant(int32_t val, const int32_t *arr, int len)
{
    for (int i = 0; i < len; i++) {
        if (arr[i] == BERRY_PADDING_ID) continue;
        if (arr[i] == val) return 1;
    }
    return 0;
}

/* Count non-PADDING entries in arr[0..len-1]. */
static int count_relevant(const int32_t *arr, int len)
{
    int c = 0;
    for (int i = 0; i < len; i++) {
        if (arr[i] != BERRY_PADDING_ID) c++;
    }
    return c;
}

/*
 * Count |set(retrieved[:k]) & set(relevant)|.
 *
 * The Python code builds set() for both sides then computes len(ret_set & rel_set).
 * We iterate over each relevant item (already unique by assumption) and check
 * whether it appears anywhere in retrieved[:k].  This naturally matches the
 * Python set-intersection semantics because each relevant item is counted at
 * most once.
 */
static int count_set_intersection(const int32_t *retrieved_row, int k,
                                  const int32_t *relevant_row, int n_relevant)
{
    int hits = 0;
    for (int r = 0; r < n_relevant; r++) {
        int32_t val = relevant_row[r];
        if (val == BERRY_PADDING_ID) continue;
        for (int j = 0; j < k; j++) {
            if (retrieved_row[j] == val) {
                hits++;
                break;
            }
        }
    }
    return hits;
}

/* ---------- rank lookup ----------------------------------------------- */

int berry_rank_lookup(const int32_t *retrieved, int n_retrieved,
                      const int32_t *relevant, int n_relevant, int k)
{
    int limit = k < n_retrieved ? k : n_retrieved;
    for (int j = 0; j < limit; j++) {
        if (is_in_relevant(retrieved[j], relevant, n_relevant)) {
            return j + 1;  /* 1-based rank */
        }
    }
    return 0;
}

/* ---------- recall@k -------------------------------------------------- */

void berry_recall_at_k(const int32_t *retrieved, const int32_t *relevant,
                       float *out, int n_queries, int n_retrieved,
                       int n_relevant, int k)
{
    for (int i = 0; i < n_queries; i++) {
        const int32_t *ret_row = retrieved + (long)i * n_retrieved;
        const int32_t *rel_row = relevant  + (long)i * n_relevant;
        int n_rel = count_relevant(rel_row, n_relevant);
        if (n_rel == 0) {
            out[i] = 0.0f;
            continue;
        }
        int hits = count_set_intersection(ret_row, k, rel_row, n_relevant);
        out[i] = (float)hits / (float)n_rel;
    }
}

/* ---------- precision@k ----------------------------------------------- */

void berry_precision_at_k(const int32_t *retrieved, const int32_t *relevant,
                          float *out, int n_queries, int n_retrieved,
                          int n_relevant, int k)
{
    for (int i = 0; i < n_queries; i++) {
        const int32_t *ret_row = retrieved + (long)i * n_retrieved;
        const int32_t *rel_row = relevant  + (long)i * n_relevant;
        int hits = count_set_intersection(ret_row, k, rel_row, n_relevant);
        out[i] = (float)hits / (float)k;
    }
}

/* ---------- MRR ------------------------------------------------------- */

void berry_mrr(const int32_t *retrieved, const int32_t *relevant, float *out,
               int n_queries, int n_retrieved, int n_relevant, int k)
{
    for (int i = 0; i < n_queries; i++) {
        const int32_t *ret_row = retrieved + (long)i * n_retrieved;
        const int32_t *rel_row = relevant  + (long)i * n_relevant;
        int n_rel = count_relevant(rel_row, n_relevant);
        if (n_rel == 0) {
            out[i] = 0.0f;
            continue;
        }
        int rank = berry_rank_lookup(ret_row, n_retrieved, rel_row, n_relevant, k);
        out[i] = (rank > 0) ? (1.0f / (float)rank) : 0.0f;
    }
}

/* ---------- nDCG ------------------------------------------------------ */

void berry_ndcg(const int32_t *retrieved, const int32_t *relevant, float *out,
                int n_queries, int n_retrieved, int n_relevant, int k)
{
    for (int i = 0; i < n_queries; i++) {
        const int32_t *ret_row = retrieved + (long)i * n_retrieved;
        const int32_t *rel_row = relevant  + (long)i * n_relevant;
        int n_rel = count_relevant(rel_row, n_relevant);
        if (n_rel == 0) {
            out[i] = 0.0f;
            continue;
        }

        /* DCG@k — iterate over retrieved positions */
        double dcg = 0.0;
        for (int j = 0; j < k; j++) {
            if (is_in_relevant(ret_row[j], rel_row, n_relevant)) {
                dcg += 1.0 / log2((double)(j + 2));
            }
        }

        /* IDCG@k — ideal ordering: all relevant at the top */
        int ideal_len = n_rel < k ? n_rel : k;
        double idcg = 0.0;
        for (int j = 0; j < ideal_len; j++) {
            idcg += 1.0 / log2((double)(j + 2));
        }

        out[i] = (idcg > 0.0) ? (float)(dcg / idcg) : 0.0f;
    }
}

/* ---------- hit rate -------------------------------------------------- */

void berry_hit_rate(const int32_t *retrieved, const int32_t *relevant,
                    float *out, int n_queries, int n_retrieved, int n_relevant,
                    int k)
{
    for (int i = 0; i < n_queries; i++) {
        const int32_t *ret_row = retrieved + (long)i * n_retrieved;
        const int32_t *rel_row = relevant  + (long)i * n_relevant;
        int n_rel = count_relevant(rel_row, n_relevant);
        if (n_rel == 0) {
            out[i] = 0.0f;
            continue;
        }
        int hits = count_set_intersection(ret_row, k, rel_row, n_relevant);
        out[i] = (hits > 0) ? 1.0f : 0.0f;
    }
}
