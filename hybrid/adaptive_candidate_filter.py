import numpy as np
import math

TAU = 500


def compute_est_ranks_dqr(S, W, q_idx, tb, te, tau_durable, rank_tables, threshold_tables):
    """
    Precompute estimated ranks (per window) and DQR for all users.

    Returns
    -------
    est_ranks : ndarray, shape (n_users, query_length), dtype float32
    dqr       : ndarray, shape (n_users,), dtype float32
        DQR[u] = (required_successes)-th smallest estimated rank across windows.
        Smaller DQR means user is more likely to be durable (closer to top-k).
    """
    n_items, d = S.shape
    n_users, L, _ = W.shape
    query_length = te - tb
    required_successes = math.ceil(tau_durable * query_length)
    required_successes = max(required_successes, 1)

    q_vec = S[q_idx]
    est_ranks = np.zeros((n_users, query_length), dtype=np.float32)
    arange_n = np.arange(n_users)

    for i, t in enumerate(range(tb, te)):
        wt = W[:, t, :]
        s = wt @ q_vec  # (n_users,)

        T_table = rank_tables[t]
        THR = threshold_tables[t]

        idx = (THR <= s[:, None]).sum(axis=1) - 1
        idx = np.clip(idx, 0, TAU - 2)

        lower = T_table[arange_n, idx + 1].astype(np.float32)
        upper = T_table[arange_n, idx].astype(np.float32)

        t0 = THR[arange_n, idx]
        t1 = THR[arange_n, idx + 1]
        denom = t1 - t0
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

        alpha = (s - t0) / denom
        alpha = np.clip(alpha, 0.0, 1.0)

        est_ranks[:, i] = upper + alpha * (lower - upper)

    partition_index = min(required_successes - 1, query_length - 1)
    dqr = np.partition(est_ranks, partition_index, axis=1)[:, partition_index]

    return est_ranks, dqr


def select_candidates_from_dqr(dqr, k, n_users, c, min_candidates,
                                boundary_margin=None, max_candidates=None):
    """
    Fast candidate selection given precomputed DQR values.
    Avoids re-computing estimated ranks when sweeping over method configurations.

    candidate_count = max(ceil(k * c), min_candidates), capped at n_users.

    If boundary_margin is not None, additionally include users with DQR <= k + boundary_margin.
    If max_candidates is not None, cap the final set by DQR order.

    Returns sorted ndarray of candidate user indices (dtype intp).
    """
    candidate_count = max(math.ceil(k * c), min_candidates)
    candidate_count = min(candidate_count, n_users)

    if candidate_count >= n_users:
        return np.arange(n_users, dtype=np.intp)

    topn = np.argpartition(dqr, candidate_count - 1)[:candidate_count]
    candidate_set = set(topn.tolist())

    if boundary_margin is not None:
        boundary_threshold = k + boundary_margin
        boundary_mask = dqr <= boundary_threshold
        candidate_set.update(np.where(boundary_mask)[0].tolist())

    if max_candidates is not None and len(candidate_set) > max_candidates:
        arr = np.array(sorted(candidate_set), dtype=np.intp)
        dqr_sub = dqr[arr]
        keep = np.argpartition(dqr_sub, max_candidates - 1)[:max_candidates]
        candidate_set = set(arr[keep].tolist())

    return np.array(sorted(candidate_set), dtype=np.intp)


def adaptive_dqr_filter(
    S,
    W,
    q_idx,
    k,
    tb,
    te,
    tau_durable,
    rank_tables,
    threshold_tables,
    c=2.0,
    min_candidates=20,
    boundary_margin=None,
    max_candidates=None,
    return_debug=False,
):
    """
    Adaptive Durable Quantile Rank (DQR) candidate filter.

    Candidate count: max(ceil(k * c), min_candidates), capped at n_users.

    Optional boundary refinement: also include users where DQR <= k + boundary_margin.
    Optional hard cap: if max_candidates is set, sort final set by DQR and keep only max_candidates.

    The adaptive filter preserves larger-is-better semantics throughout:
    higher dot product = better rank for user u at time t.

    Parameters
    ----------
    S               : (n_items, d) item matrix
    W               : (n_users, L, d) user temporal preference tensor
    q_idx           : query item index
    k               : top-k parameter
    tb, te          : query interval [tb, te) over windows
    tau_durable     : durability threshold in [0, 1]
    rank_tables     : list of T_table arrays, one per window (from build_rank_table)
    threshold_tables: list of THR arrays, one per window (from build_rank_table)
    c               : expansion factor; candidate_count >= ceil(k * c)
    min_candidates  : floor on candidate count; candidate_count >= min_candidates
    boundary_margin : if set, include users with DQR <= k + boundary_margin
    max_candidates  : if set, cap final candidate set to max_candidates by DQR order
    return_debug    : if True, return (candidates, debug_dict) instead of just candidates

    Returns
    -------
    candidate_user_ids : sorted ndarray of candidate indices (dtype intp)
    debug              : dict (only when return_debug=True)
    """
    n_items, d = S.shape
    n_users, L, _ = W.shape

    if not (0 <= tb < te <= L):
        raise ValueError(f"Invalid [tb, te): got [{tb}, {te}), L={L}")
    if not (0 <= tau_durable <= 1):
        raise ValueError("tau_durable must be in [0, 1]")
    if k <= 0 or c <= 0:
        raise ValueError("k and c must be positive")
    if not (0 <= q_idx < n_items):
        raise IndexError(f"q_idx {q_idx} out of bounds for {n_items} items")

    query_length = te - tb
    required_successes = math.ceil(tau_durable * query_length)

    candidate_count = max(math.ceil(k * c), min_candidates)
    candidate_count = min(candidate_count, n_users)

    if candidate_count >= n_users:
        result = np.arange(n_users, dtype=np.intp)
        if return_debug:
            return result, {
                "all_users_selected": True,
                "candidate_count_topn": candidate_count,
                "n_users": n_users,
                "dqr_values": None,
                "topn_candidates": result,
                "boundary_candidates": np.array([], dtype=np.intp),
                "final_candidate_count": n_users,
            }
        return result

    if required_successes == 0:
        result = np.arange(candidate_count, dtype=np.intp)
        if return_debug:
            return result, {
                "all_trivially_qualify": True,
                "candidate_count_topn": candidate_count,
            }
        return result

    est_ranks, dqr = compute_est_ranks_dqr(
        S, W, q_idx, tb, te, tau_durable, rank_tables, threshold_tables
    )

    topn = np.argpartition(dqr, candidate_count - 1)[:candidate_count]
    topn_set = set(topn.tolist())

    boundary_indices = np.array([], dtype=np.intp)
    if boundary_margin is not None:
        boundary_threshold = k + boundary_margin
        boundary_mask = dqr <= boundary_threshold
        boundary_indices = np.where(boundary_mask)[0].astype(np.intp)
        candidate_set = topn_set | set(boundary_indices.tolist())
    else:
        candidate_set = topn_set

    if max_candidates is not None and len(candidate_set) > max_candidates:
        arr = np.array(sorted(candidate_set), dtype=np.intp)
        dqr_sub = dqr[arr]
        keep = np.argpartition(dqr_sub, max_candidates - 1)[:max_candidates]
        candidate_set = set(arr[keep].tolist())

    result = np.array(sorted(candidate_set), dtype=np.intp)

    if return_debug:
        debug = {
            "dqr_values": dqr,
            "est_ranks": est_ranks,
            "candidate_count_topn": candidate_count,
            "required_successes": required_successes,
            "query_length": query_length,
            "boundary_margin": boundary_margin,
            "max_candidates": max_candidates,
            "topn_candidates": np.sort(topn).astype(np.intp),
            "boundary_candidates": boundary_indices,
            "final_candidate_count": len(result),
        }
        return result, debug

    return result


def exact_rerank_candidates(
    S,
    W,
    candidate_user_ids,
    q_idx,
    k,
    tb,
    te,
    tau_durable,
    near_boundary_slack=0,
    chunk_size=4000,
    return_debug=False,
):
    """
    Two-stage exact reranking of the rough candidate pool.

    For each user in candidate_user_ids, compute the exact temporal rank of query item q
    across every window t in [tb, te):

        rank_t(u, q) = 1 + |{i : W[u,t]·S[i] > W[u,t]·S[q]}|

    A window t is a "success" for user u when rank_t(u, q) <= k.

        success_count(u) = sum_{t in [tb,te)} [rank_t(u,q) <= k]
        required_successes = ceil(tau_durable * (te - tb))

    Keep user u when:  success_count(u) >= required_successes - near_boundary_slack

    This is cheaper than full SSA over all users because it only processes the rough pool.
    Larger-is-better semantics are preserved: higher dot product = better rank.

    Parameters
    ----------
    S                  : (n_items, d) item matrix
    W                  : (n_users, L, d) user preference tensor
    candidate_user_ids : array-like of user indices (the rough pool)
    q_idx              : query item index
    k                  : top-k parameter
    tb, te             : query interval [tb, te)
    tau_durable        : durability threshold in [0, 1]
    near_boundary_slack: relax threshold by this many successes (0 = exact)
    chunk_size         : item chunk size for memory-efficient computation
    return_debug       : if True, return (refined, debug_dict)

    Returns
    -------
    refined_candidate_user_ids : sorted ndarray of retained candidate indices (dtype intp)
    debug                      : dict (only when return_debug=True)
    """
    n_items, d = S.shape
    n_users_total, L, _ = W.shape
    query_length = te - tb
    required_successes = math.ceil(tau_durable * query_length)

    cand_ids = np.asarray(candidate_user_ids, dtype=np.intp)
    n_cands = len(cand_ids)

    if n_cands == 0:
        empty = np.array([], dtype=np.intp)
        if return_debug:
            return empty, {
                "exact_ranks": np.zeros((0, query_length), dtype=np.int32),
                "success_counts": np.array([], dtype=np.int32),
                "required_successes": required_successes,
                "retained_global_ids": empty,
                "removed_global_ids": empty,
            }
        return empty

    W_sub = W[cand_ids]  # (n_cands, L, d)
    q_vec = S[q_idx]

    success_count = np.zeros(n_cands, dtype=np.int32)

    if return_debug:
        exact_ranks_all = np.zeros((n_cands, query_length), dtype=np.int32)

    for i, t in enumerate(range(tb, te)):
        wt = W_sub[:, t, :]         # (n_cands, d)
        score_q = wt @ q_vec        # (n_cands,) — score of query item for each candidate

        # rank starts at 1; each item strictly better than q increments rank by 1
        rank_t = np.ones(n_cands, dtype=np.int32)

        for start in range(0, n_items, chunk_size):
            end = min(start + chunk_size, n_items)
            chunk_scores = wt @ S[start:end].T  # (n_cands, chunk_size)
            rank_t += (chunk_scores > score_q[:, None]).sum(axis=1).astype(np.int32)

        if return_debug:
            exact_ranks_all[:, i] = rank_t

        success_count += (rank_t <= k).astype(np.int32)

    threshold = max(required_successes - near_boundary_slack, 0)
    keep_mask = success_count >= threshold

    retained_local = np.where(keep_mask)[0]
    removed_local = np.where(~keep_mask)[0]
    refined = cand_ids[retained_local]

    if return_debug:
        debug = {
            "exact_ranks": exact_ranks_all,
            "success_counts": success_count,
            "required_successes": required_successes,
            "near_boundary_slack": near_boundary_slack,
            "keep_threshold": threshold,
            "retained_local_indices": retained_local,
            "removed_local_indices": removed_local,
            "retained_global_ids": refined,
            "removed_global_ids": cand_ids[removed_local],
        }
        return refined, debug

    return refined
