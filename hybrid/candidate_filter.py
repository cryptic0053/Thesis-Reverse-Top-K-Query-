import numpy as np
import math

TAU = 500

def approximate_candidate_filter(S, W, q_idx, k, c, T_table, THR):
    n_users, L, d = W.shape
    avg_users = W.mean(axis=1) 
    
    q = S[q_idx]
    s = avg_users @ q
    
    idx = (THR <= s[:, None]).sum(axis=1) - 1
    idx = np.clip(idx, 0, TAU - 2)
    
    lower = T_table[np.arange(n_users), idx + 1].astype(np.float32)
    upper = T_table[np.arange(n_users), idx].astype(np.float32)
    
    t0 = THR[np.arange(n_users), idx]
    t1 = THR[np.arange(n_users), idx + 1]
    denom = (t1 - t0)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    
    alpha = (s - t0) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    
    est_rank = upper + alpha * (lower - upper)
    
    fetch_count = math.ceil(k * c)
    fetch_count = min(fetch_count, n_users)
    
    topk_candidates = np.argpartition(est_rank, fetch_count - 1)[:fetch_count]
    return topk_candidates


def approximate_candidate_filter_union_windows(S, W, q_idx, k, c, T_table_list, THR_list):
    n_users, L, d = W.shape
    q = S[q_idx]
    fetch_count = math.ceil(k * c)
    fetch_count = min(fetch_count, n_users)

    all_candidates = set()

    for t in range(L):
        w_t = W[:, t, :]
        T_table = T_table_list[t]
        THR = THR_list[t]

        s = w_t @ q

        idx = (THR <= s[:, None]).sum(axis=1) - 1
        idx = np.clip(idx, 0, TAU - 2)

        arange_n = np.arange(n_users)
        lower = T_table[arange_n, idx + 1].astype(np.float32)
        upper = T_table[arange_n, idx].astype(np.float32)

        t0 = THR[arange_n, idx]
        t1 = THR[arange_n, idx + 1]
        denom = (t1 - t0)
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        alpha = (s - t0) / denom
        alpha = np.clip(alpha, 0.0, 1.0)

        est_rank = upper + alpha * (lower - upper)

        if fetch_count >= n_users:
            window_cands = np.arange(n_users)
        else:
            window_cands = np.argpartition(est_rank, fetch_count - 1)[:fetch_count]

        all_candidates.update(window_cands.tolist())

    return np.array(sorted(all_candidates), dtype=np.intp)


def legacy_max_window_rank_count_filter(S, W, q_idx, k, c, chunk_size=4000):
    """
    Legacy filter designed for the old smaller-is-better SSA behavior.
    It computes the maximum conventional rank count across all windows.
    Does not use [tb, te). Does not use tau_durable.
    Has no recall guarantee. Retained only for historical comparison.
    """
    n_items, d = S.shape
    n_users, L, _ = W.shape
    fetch_count = math.ceil(k * c)
    fetch_count = min(fetch_count, n_users)

    q_vec = S[q_idx]

    max_rank = np.zeros(n_users, dtype=np.int32)

    for t in range(L):
        wt = W[:, t, :]
        score_q = wt @ q_vec

        rank_t = np.zeros(n_users, dtype=np.int32)
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            chunk_scores = wt @ S[start:end].T
            rank_t += (chunk_scores > score_q[:, None]).sum(axis=1).astype(np.int32)

        max_rank = np.maximum(max_rank, rank_t)

    if fetch_count >= n_users:
        return np.arange(n_users, dtype=np.intp)

    candidates = np.argpartition(-max_rank, fetch_count - 1)[:fetch_count]
    return candidates

# Backward-compatible alias
approximate_candidate_filter_min_window_rank = legacy_max_window_rank_count_filter


def durable_quantile_rank_filter(S, W, q_idx, k, c, tb, te, tau_durable, T_table_list=None, THR_list=None):
    """
    Durability-aware candidate filter matching larger-is-better semantics.
    """
    n_items, d = S.shape
    n_users, L, _ = W.shape

    if not (0 <= tb < te <= L):
        raise ValueError("Invalid [tb, te)")
    if not (0 <= tau_durable <= 1):
        raise ValueError("Invalid tau_durable")
    if k <= 0 or c <= 0:
        raise ValueError("k and c must be positive")
    if not (0 <= q_idx < n_items):
        raise IndexError("Invalid q_idx")
        
    query_length = te - tb
    required_successes = math.ceil(tau_durable * query_length)
    if required_successes == 0:
        # If 0 successes required, all users trivially match
        fetch_count = math.ceil(k * c)
        return np.arange(min(fetch_count, n_users), dtype=np.intp)

    fetch_count = math.ceil(k * c)
    fetch_count = min(fetch_count, n_users)

    if fetch_count >= n_users:
        return np.arange(n_users, dtype=np.intp)

    q_vec = S[q_idx]
    est_ranks = np.zeros((n_users, query_length), dtype=np.float32)

    arange_n = np.arange(n_users)

    for i, t in enumerate(range(tb, te)):
        wt = W[:, t, :]
        s = wt @ q_vec

        if T_table_list is not None and THR_list is not None:
            T_table = T_table_list[t]
            THR = THR_list[t]

            idx = (THR <= s[:, None]).sum(axis=1) - 1
            idx = np.clip(idx, 0, TAU - 2)

            lower = T_table[arange_n, idx + 1].astype(np.float32)
            upper = T_table[arange_n, idx].astype(np.float32)

            t0 = THR[arange_n, idx]
            t1 = THR[arange_n, idx + 1]
            denom = (t1 - t0)
            denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

            alpha = (s - t0) / denom
            alpha = np.clip(alpha, 0.0, 1.0)

            est_ranks[:, i] = upper + alpha * (lower - upper)
        else:
            # Fallback to exact computation if tables not provided
            rank_t = np.zeros(n_users, dtype=np.int32)
            chunk_size = 4000
            for start in range(0, n_items, chunk_size):
                end = min(start + chunk_size, n_items)
                chunk_scores = wt @ S[start:end].T
                rank_t += (chunk_scores > s[:, None]).sum(axis=1).astype(np.int32)
            est_ranks[:, i] = rank_t + 1

    partition_index = required_successes - 1
    durable_rank = np.partition(est_ranks, partition_index, axis=1)[:, partition_index]

    candidates = np.argpartition(durable_rank, fetch_count - 1)[:fetch_count]
    return np.sort(candidates).astype(np.intp)

