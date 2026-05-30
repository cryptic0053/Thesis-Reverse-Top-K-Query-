import numpy as np

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
    
    # fetch proportion of candidates based on relaxation factor
    fetch_count = int(k * c)
    fetch_count = min(fetch_count, n_users)
    
    topk_candidates = np.argpartition(est_rank, fetch_count - 1)[:fetch_count]
    return topk_candidates


def approximate_candidate_filter_union_windows(S, W, q_idx, k, c, T_table_list, THR_list):
    """
    Per-window union candidate filter.

    Instead of averaging user vectors across time (which loses temporal
    preference info), this runs the approximate candidate filter independently
    for each time window using window-specific rank tables, then returns the
    union of all per-window candidate sets.

    Parameters
    ----------
    S : ndarray (n_items, d)
    W : ndarray (n_users, L, d)
    q_idx : int
    k : int
    c : float  – looseness / relaxation factor
    T_table_list : list[ndarray]  – one rank table per window, each (n_users, TAU)
    THR_list : list[ndarray]  – one threshold matrix per window, each (n_users, TAU)

    Returns
    -------
    candidates : ndarray of unique user ids (the union across windows)
    """
    n_users, L, d = W.shape
    q = S[q_idx]
    fetch_count = int(k * c)
    fetch_count = min(fetch_count, n_users)

    all_candidates = set()

    for t in range(L):
        w_t = W[:, t, :]          # (n_users, d)
        T_table = T_table_list[t] # (n_users, TAU)
        THR = THR_list[t]         # (n_users, TAU)

        s = w_t @ q               # (n_users,)

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


def approximate_candidate_filter_min_window_rank(S, W, q_idx, k, c, chunk_size=4000):
    """
    Direct per-window maximum-rank candidate filter.

    The SSA durable verifier uses bottom-k semantics: a user qualifies when
    query item q is among the k items with the SMALLEST preference scores.
    Therefore, a good candidate is a user where many other items score higher
    than q (i.e. q's rank-from-top is large).

    For each time window t this filter computes the rank of q for each user
    (= how many items score strictly higher than q).  It keeps the MAXIMUM
    rank across all windows — the window where q is ranked worst is the
    window most likely to place q in the bottom-k.

    Users with the highest max_rank are selected as candidates.

    Parameters
    ----------
    S : ndarray (n_items, d)
    W : ndarray (n_users, L, d)
    q_idx : int
    k : int
    c : float  – relaxation factor  (candidates = k * c)
    chunk_size : int – items processed per chunk to limit memory

    Returns
    -------
    candidates : ndarray of user ids
    """
    n_items, d = S.shape
    n_users, L, _ = W.shape
    fetch_count = int(k * c)
    fetch_count = min(fetch_count, n_users)

    q_vec = S[q_idx]   # (d,)

    # For each window, compute the rank of q for each user.
    # rank_of_q[u, t] = how many items score strictly higher than q for user u
    #                    at time t.  Higher rank = q scores poorly = more likely
    #                    to be in bottom-k.
    max_rank = np.zeros(n_users, dtype=np.int32)

    for t in range(L):
        wt = W[:, t, :]                     # (n_users, d)
        score_q = wt @ q_vec                # (n_users,)

        # Count how many items beat q for each user, processing in chunks
        rank_t = np.zeros(n_users, dtype=np.int32)
        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            chunk_scores = wt @ S[start:end].T   # (n_users, chunk)
            rank_t += (chunk_scores > score_q[:, None]).sum(axis=1).astype(np.int32)

        # Keep the maximum (worst) rank across windows
        max_rank = np.maximum(max_rank, rank_t)

    if fetch_count >= n_users:
        return np.arange(n_users, dtype=np.intp)

    # Select users with the HIGHEST max_rank (q scored worst for these users)
    # argpartition with negated values gives top-k largest
    candidates = np.argpartition(-max_rank, fetch_count - 1)[:fetch_count]
    return candidates

