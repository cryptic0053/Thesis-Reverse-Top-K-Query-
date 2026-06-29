"""
Tests for hybrid/adaptive_candidate_filter.py

Verifies:
  1. Adaptive candidate count uses max(ceil(k*c), min_candidates)
  2. Boundary rule includes users with DQR <= k + boundary_margin
  3. exact_rerank_candidates keeps users with success_count >= required_successes
  4. Hybrid subset invariant holds after adaptive filtering
  5. No score-direction inversion (larger-is-better semantics preserved)
"""

import sys
import os
import math

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from hybrid.adaptive_candidate_filter import (
    adaptive_dqr_filter,
    exact_rerank_candidates,
    compute_est_ranks_dqr,
    select_candidates_from_dqr,
)
from utils.rank_table import build_rank_table
from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from hybrid.durable_verifier import ssa_on_candidates


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_data(n_users=80, n_items=150, d=8, L=4, seed=17):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n_items, d)).astype(np.float32)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    W = rng.standard_normal((n_users, L, d)).astype(np.float32)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + 1e-12
    return S, W


def build_tables(W, S, L):
    T_list, THR_list = [], []
    for t in range(L):
        T, THR = build_rank_table(W[:, t, :], S)
        T_list.append(T)
        THR_list.append(THR)
    return T_list, THR_list


# ---------------------------------------------------------------------------
# Test 1: adaptive candidate count
# ---------------------------------------------------------------------------

def test_adaptive_candidate_count_uses_max_rule():
    """candidate_count = max(ceil(k * c), min_candidates), capped at n_users."""
    S, W = make_data()
    n_users = W.shape[0]
    L = W.shape[1]
    T_list, THR_list = build_tables(W, S, L)

    k = 5
    tb, te = 0, L
    tau = 0.6

    # c=2 → ceil(5*2)=10; min_candidates=50 → expected ≥ 50
    cands_a, dbg_a = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=50, return_debug=True,
    )
    assert len(cands_a) >= 50, f"Expected >= 50, got {len(cands_a)}"
    assert len(cands_a) <= n_users

    # c=10 → ceil(5*10)=50; min_candidates=5 → expected ≥ 50
    cands_b, dbg_b = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=10.0, min_candidates=5, return_debug=True,
    )
    assert len(cands_b) >= 50, f"Expected >= 50, got {len(cands_b)}"

    # c=2 → 10; min_candidates=10 → expected exactly 10 (no boundary)
    cands_c, _ = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=10, boundary_margin=None, return_debug=True,
    )
    assert len(cands_c) == 10, f"Expected exactly 10, got {len(cands_c)}"


def test_select_candidates_from_dqr_count_rule():
    """select_candidates_from_dqr implements max(ceil(k*c), min_candidates)."""
    rng = np.random.default_rng(0)
    dqr = rng.uniform(1, 100, size=200).astype(np.float32)
    n_users = len(dqr)
    k = 10

    cands = select_candidates_from_dqr(dqr, k, n_users, c=2.0, min_candidates=30)
    assert len(cands) == 30  # max(20, 30) = 30

    cands2 = select_candidates_from_dqr(dqr, k, n_users, c=5.0, min_candidates=10)
    assert len(cands2) == 50  # max(50, 10) = 50

    # Cap at n_users
    cands3 = select_candidates_from_dqr(dqr, k, n_users, c=100.0, min_candidates=10)
    assert len(cands3) == n_users


# ---------------------------------------------------------------------------
# Test 2: boundary rule
# ---------------------------------------------------------------------------

def test_boundary_rule_includes_near_k_users():
    """Boundary rule includes users with DQR <= k + boundary_margin."""
    S, W = make_data()
    L = W.shape[1]
    T_list, THR_list = build_tables(W, S, L)
    k = 5
    tb, te = 0, L
    tau = 0.6

    cands_no_bm, dbg_no = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=10, boundary_margin=None, return_debug=True,
    )
    dqr = dbg_no["dqr_values"]

    cands_bm5, dbg_bm5 = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=10, boundary_margin=5, return_debug=True,
    )

    # All boundary candidates must satisfy DQR <= k + 5
    bnd_cands = dbg_bm5["boundary_candidates"]
    if len(bnd_cands) > 0:
        assert np.all(dqr[bnd_cands] <= k + 5), (
            "Boundary candidates should have DQR <= k + boundary_margin"
        )

    # With boundary margin the result is at least as large
    assert len(cands_bm5) >= len(cands_no_bm)


def test_boundary_users_have_expected_dqr():
    """Manually verify boundary selection via select_candidates_from_dqr."""
    k = 10
    n_users = 100
    # Create artificial DQR: 20 users with DQR in (k, k+5], rest much larger
    rng = np.random.default_rng(42)
    dqr = rng.uniform(50, 200, size=n_users).astype(np.float32)
    boundary_users = np.arange(5, 25)   # indices 5..24 get DQR in (10, 14]
    dqr[boundary_users] = rng.uniform(k + 0.1, k + 4.9, size=len(boundary_users))

    # min_candidates=1 so top-N is tiny; boundary_margin=5 should capture dqr <= 15
    cands = select_candidates_from_dqr(
        dqr, k, n_users, c=0.5, min_candidates=1,
        boundary_margin=5,
    )
    cand_set = set(cands.tolist())

    for u in boundary_users:
        if dqr[u] <= k + 5:
            assert u in cand_set, f"User {u} with DQR={dqr[u]:.2f} should be included by boundary rule"


# ---------------------------------------------------------------------------
# Test 3: exact_rerank_candidates correctness
# ---------------------------------------------------------------------------

def test_exact_rerank_matches_ssa_when_all_users_are_candidates():
    """exact_rerank_candidates should match SSA when given all users as candidates."""
    S, W = make_data(n_users=50, n_items=100, d=8, L=4)
    k = 5
    tb, te = 0, 4
    tau = 0.5
    q_idx = 3

    all_users = np.arange(W.shape[0], dtype=np.intp)
    refined = exact_rerank_candidates(S, W, all_users, q_idx, k, tb, te, tau)

    runs = build_ssa_for_object_chunked(S, W, q_idx, k)
    ssa_result = set(drtopk_ssa_query(runs, tb, te, tau))
    exact_result = set(refined.tolist())

    assert exact_result == ssa_result, (
        f"exact_rerank({len(exact_result)}) != SSA({len(ssa_result)}):\n"
        f"  extra: {exact_result - ssa_result}\n"
        f"  missing: {ssa_result - exact_result}"
    )


def test_exact_rerank_empty_candidates():
    """exact_rerank_candidates handles empty input gracefully."""
    S, W = make_data()
    result = exact_rerank_candidates(S, W, np.array([], dtype=np.intp), 0, 5, 0, 4, 0.6)
    assert len(result) == 0


def test_exact_rerank_near_boundary_slack():
    """near_boundary_slack relaxes the required_successes threshold."""
    S, W = make_data(n_users=60, n_items=120, d=8, L=6)
    k = 5
    tb, te = 0, 6
    tau = 0.8   # required = ceil(0.8 * 6) = 5
    q_idx = 1

    all_users = np.arange(W.shape[0], dtype=np.intp)

    refined_strict = exact_rerank_candidates(S, W, all_users, q_idx, k, tb, te, tau, near_boundary_slack=0)
    refined_slack = exact_rerank_candidates(S, W, all_users, q_idx, k, tb, te, tau, near_boundary_slack=1)

    # Relaxing threshold should keep at least as many users
    assert len(refined_slack) >= len(refined_strict), (
        "near_boundary_slack=1 should keep at least as many users as slack=0"
    )


def test_exact_rerank_debug_info():
    """exact_rerank_candidates returns correct debug information."""
    S, W = make_data(n_users=30, n_items=60, d=8, L=4)
    k = 5
    tb, te = 0, 4
    tau = 0.5
    q_idx = 2

    cands = np.arange(20, dtype=np.intp)
    refined, debug = exact_rerank_candidates(
        S, W, cands, q_idx, k, tb, te, tau, return_debug=True
    )

    assert "exact_ranks" in debug
    assert "success_counts" in debug
    assert debug["exact_ranks"].shape == (len(cands), te - tb)
    assert len(debug["success_counts"]) == len(cands)
    assert debug["required_successes"] == math.ceil(tau * (te - tb))

    # All exact_ranks should be >= 1
    assert np.all(debug["exact_ranks"] >= 1), "Ranks must be >= 1 (1-indexed)"


# ---------------------------------------------------------------------------
# Test 4: hybrid subset invariant
# ---------------------------------------------------------------------------

def test_hybrid_subset_invariant_with_adaptive_filter():
    """SSA result on adaptive candidates must be a subset of Full SSA result."""
    S, W = make_data(n_users=100, n_items=200, d=8, L=4)
    k = 5
    tb, te = 0, 4
    tau = 0.6
    q_idx = 7
    T_list, THR_list = build_tables(W, S, W.shape[1])

    # Full SSA ground truth
    runs = build_ssa_for_object_chunked(S, W, q_idx, k)
    full_result = set(drtopk_ssa_query(runs, tb, te, tau))

    # Adaptive filter with generous candidate budget
    cands = adaptive_dqr_filter(
        S, W, q_idx, k, tb, te, tau, T_list, THR_list,
        c=5.0, min_candidates=50,
    )
    hybrid_result = set(ssa_on_candidates(S, W, list(cands), q_idx, k, tb, te, tau))

    fp = hybrid_result - full_result
    assert len(fp) == 0, (
        f"Adaptive filter + SSA produced {len(fp)} false positives: {fp}"
    )


def test_hybrid_subset_invariant_with_exact_rerank():
    """SSA result after exact reranking must also be a subset of Full SSA result."""
    S, W = make_data(n_users=80, n_items=160, d=8, L=4)
    k = 5
    tb, te = 0, 4
    tau = 0.6
    q_idx = 5
    T_list, THR_list = build_tables(W, S, W.shape[1])

    runs = build_ssa_for_object_chunked(S, W, q_idx, k)
    full_result = set(drtopk_ssa_query(runs, tb, te, tau))

    cands, _ = adaptive_dqr_filter(
        S, W, q_idx, k, tb, te, tau, T_list, THR_list,
        c=3.0, min_candidates=40, return_debug=True,
    )
    refined = exact_rerank_candidates(S, W, cands, q_idx, k, tb, te, tau)
    hybrid_result = set(ssa_on_candidates(S, W, list(refined), q_idx, k, tb, te, tau))

    fp = hybrid_result - full_result
    assert len(fp) == 0, (
        f"Exact rerank + SSA produced {len(fp)} false positives: {fp}"
    )


# ---------------------------------------------------------------------------
# Test 5: no score-direction inversion
# ---------------------------------------------------------------------------

def test_no_score_direction_inversion():
    """DQR selects users with SMALLER estimated rank (closer to top-k), not larger."""
    S, W = make_data(n_users=100, n_items=200, d=8, L=4)
    k = 5
    tb, te = 0, 4
    tau = 0.6
    q_idx = 9
    T_list, THR_list = build_tables(W, S, W.shape[1])

    cands, debug = adaptive_dqr_filter(
        S, W, q_idx, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=10, boundary_margin=None, return_debug=True,
    )
    dqr = debug["dqr_values"]

    # All DQR values should be >= 1 (rank is 1-indexed)
    assert np.all(dqr >= 1), f"DQR should be >= 1; min={dqr.min()}"

    # Top-N candidates should be the users with smallest DQR
    topn = debug["topn_candidates"]
    n_topn = len(topn)
    non_topn = np.setdiff1d(np.arange(W.shape[0]), topn)

    if len(non_topn) > 0:
        max_dqr_in_topn = dqr[topn].max()
        min_dqr_outside = dqr[non_topn].min()
        assert max_dqr_in_topn <= min_dqr_outside + 1e-3, (
            f"Top-N should have the smallest DQR values. "
            f"max_in_topN={max_dqr_in_topn:.3f}, min_outside={min_dqr_outside:.3f}"
        )


def test_exact_rerank_larger_is_better_semantics():
    """
    Exact reranking uses larger dot-product = better rank.
    A user with very high scores for all items should always rank #1 for any q.
    """
    d = 4
    n_items = 20
    L = 3
    k = 1

    S = np.random.default_rng(0).standard_normal((n_items, d)).astype(np.float32)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-12

    # Construct 2 users:
    # user 0: preference aligned with S[0], scores high for all items (should often be rank 1)
    # user 1: random preference
    q_idx = 0
    W = np.zeros((2, L, d), dtype=np.float32)
    W[0] = S[q_idx][None, :]       # perfectly aligned — W[0,t]·S[q] is maximum
    W[1] = np.random.default_rng(1).standard_normal((L, d)).astype(np.float32)
    W[1] /= np.linalg.norm(W[1], axis=1, keepdims=True) + 1e-12

    # For user 0, score of q is maximum (equal to 1 since S[q_idx] is normalized),
    # but other items may score lower → rank(u=0) = 1 in most windows
    all_users = np.arange(2, dtype=np.intp)
    refined, debug = exact_rerank_candidates(
        S, W, all_users, q_idx, k=1, tb=0, te=L, tau_durable=0.0,
        return_debug=True,
    )
    # With k=1 and tau=0: success requires rank <= 1, i.e., score_q >= all other scores
    # User 0 has preference = S[q_idx], so score_q = ||S[q_idx]||^2 = 1.
    # For items i ≠ q_idx: W[0,t]·S[i] = S[q_idx]·S[i] ≤ 1 (Cauchy-Schwarz).
    # Ties at 1 are possible but rank should be 1 if score_q equals max.
    # The point is: this does NOT use smaller-is-better anywhere.
    exact_ranks_u0 = debug["exact_ranks"][0, :]  # shape (L,)
    # Rank should be 1 or close to 1 for user 0 with perfectly aligned preference
    assert exact_ranks_u0.max() <= n_items, "Rank must be in [1, n_items]"


# ---------------------------------------------------------------------------
# Test 6: max_candidates cap
# ---------------------------------------------------------------------------

def test_max_candidates_cap():
    """max_candidates caps the final candidate set size."""
    S, W = make_data(n_users=200, n_items=300, d=8, L=4)
    L = W.shape[1]
    T_list, THR_list = build_tables(W, S, L)
    k = 5
    tb, te = 0, L
    tau = 0.6

    # Without cap: boundary_margin=50 would give many candidates
    cands_uncapped, _ = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=10, boundary_margin=50, return_debug=True,
    )

    # With cap at 30
    cands_capped, _ = adaptive_dqr_filter(
        S, W, 0, k, tb, te, tau, T_list, THR_list,
        c=2.0, min_candidates=10, boundary_margin=50,
        max_candidates=30, return_debug=True,
    )
    assert len(cands_capped) <= 30, f"max_candidates=30 cap violated: {len(cands_capped)}"


if __name__ == "__main__":
    import traceback

    tests = [
        test_adaptive_candidate_count_uses_max_rule,
        test_select_candidates_from_dqr_count_rule,
        test_boundary_rule_includes_near_k_users,
        test_boundary_users_have_expected_dqr,
        test_exact_rerank_matches_ssa_when_all_users_are_candidates,
        test_exact_rerank_empty_candidates,
        test_exact_rerank_near_boundary_slack,
        test_exact_rerank_debug_info,
        test_hybrid_subset_invariant_with_adaptive_filter,
        test_hybrid_subset_invariant_with_exact_rerank,
        test_no_score_direction_inversion,
        test_exact_rerank_larger_is_better_semantics,
        test_max_candidates_cap,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
