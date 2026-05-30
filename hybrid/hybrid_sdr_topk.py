from hybrid.candidate_filter import (
    approximate_candidate_filter,
    approximate_candidate_filter_union_windows,
    approximate_candidate_filter_min_window_rank,
)
from hybrid.durable_verifier import ssa_on_candidates, pra_on_candidates

def hybrid_ssa_query(S, W, q_idx, k, c, tb, te, tau, T_table, THR, chunk_size=4000):
    candidate_user_ids = approximate_candidate_filter(S, W, q_idx, k, c, T_table, THR)
    
    result_user_ids = ssa_on_candidates(
        S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size
    )
    
    return {
        "query_item": q_idx,
        "num_total_users": W.shape[0],
        "num_candidates": len(candidate_user_ids),
        "candidate_user_ids": candidate_user_ids,
        "result_user_ids": result_user_ids,
    }

def hybrid_pra_query(S, W, q_idx, k, c, tb, te, tau, T_table, THR, chunk_size=4000):
    candidate_user_ids = approximate_candidate_filter(S, W, q_idx, k, c, T_table, THR)
    
    result_user_ids = pra_on_candidates(
        S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size
    )
    
    return {
        "query_item": q_idx,
        "num_total_users": W.shape[0],
        "num_candidates": len(candidate_user_ids),
        "candidate_user_ids": candidate_user_ids,
        "result_user_ids": result_user_ids,
    }


# ── Union-window variants ──────────────────────────────────────────────

def hybrid_ssa_query_union(S, W, q_idx, k, c, tb, te, tau,
                           T_table_list, THR_list, chunk_size=4000):
    """Hybrid SSA with per-window union candidate filter."""
    candidate_user_ids = approximate_candidate_filter_union_windows(
        S, W, q_idx, k, c, T_table_list, THR_list
    )

    result_user_ids = ssa_on_candidates(
        S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size
    )

    return {
        "query_item": q_idx,
        "num_total_users": W.shape[0],
        "num_candidates": len(candidate_user_ids),
        "candidate_user_ids": candidate_user_ids,
        "result_user_ids": result_user_ids,
    }


def hybrid_pra_query_union(S, W, q_idx, k, c, tb, te, tau,
                            T_table_list, THR_list, chunk_size=4000):
    """Hybrid PRA with per-window union candidate filter."""
    candidate_user_ids = approximate_candidate_filter_union_windows(
        S, W, q_idx, k, c, T_table_list, THR_list
    )

    result_user_ids = pra_on_candidates(
        S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size
    )

    return {
        "query_item": q_idx,
        "num_total_users": W.shape[0],
        "num_candidates": len(candidate_user_ids),
        "candidate_user_ids": candidate_user_ids,
        "result_user_ids": result_user_ids,
    }


# -- Min-window-rank variants (exact per-window rank) --------------------

def hybrid_ssa_query_minrank(S, W, q_idx, k, c, tb, te, tau, chunk_size=4000):
    """Hybrid SSA with per-window minimum-rank candidate filter."""
    candidate_user_ids = approximate_candidate_filter_min_window_rank(
        S, W, q_idx, k, c, chunk_size
    )

    result_user_ids = ssa_on_candidates(
        S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size
    )

    return {
        "query_item": q_idx,
        "num_total_users": W.shape[0],
        "num_candidates": len(candidate_user_ids),
        "candidate_user_ids": candidate_user_ids,
        "result_user_ids": result_user_ids,
    }


def hybrid_pra_query_minrank(S, W, q_idx, k, c, tb, te, tau, chunk_size=4000):
    """Hybrid PRA with per-window minimum-rank candidate filter."""
    candidate_user_ids = approximate_candidate_filter_min_window_rank(
        S, W, q_idx, k, c, chunk_size
    )

    result_user_ids = pra_on_candidates(
        S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size
    )

    return {
        "query_item": q_idx,
        "num_total_users": W.shape[0],
        "num_candidates": len(candidate_user_ids),
        "candidate_user_ids": candidate_user_ids,
        "result_user_ids": result_user_ids,
    }
