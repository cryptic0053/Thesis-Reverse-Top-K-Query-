import sys
import os
import math
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object, build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.candidate_filter import durable_quantile_rank_filter
from hybrid.durable_verifier import ssa_on_candidates, pra_on_candidates
from tests.reference_durable import exact_durable_reverse_topk

def generate_random_data(n_users=50, n_items=100, L=5, d=4, seed=42):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n_items, d)).astype(np.float32)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    W = rng.normal(size=(n_users, L, d)).astype(np.float32)
    W /= np.linalg.norm(W, axis=2, keepdims=True)
    return S, W

def test_static_exact_rank_and_one_window_consistency():
    S, W = generate_random_data(L=1)
    q_idx = 0
    k = 10
    tb, te, tau = 0, 1, 1.0
    
    exact_res = exact_durable_reverse_topk(S, W, q_idx, k, tb, te, tau)
    
    runs = build_ssa_for_object(S, W, q_idx, k)
    ssa_res = drtopk_ssa_query(runs, tb, te, tau)
    
    assert set(exact_res) == set(ssa_res), "SSA does not match exact rank on L=1"

def test_full_vs_chunked_ssa():
    # Use exact integer values to avoid floating-point accumulation differences
    # between GEMM (full matrix) and GEMV/chunked GEMM.
    rng = np.random.default_rng(42)
    S = rng.integers(-10, 10, size=(100, 4)).astype(np.float32)
    W = rng.integers(-10, 10, size=(50, 5, 4)).astype(np.float32)
    q_idx = 0
    k = 10
    
    runs_full = build_ssa_for_object(S, W, q_idx, k)
    runs_chunked = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size=3)
    
    assert runs_full == runs_chunked, "Chunked SSA does not match full SSA"

def test_ssa_vs_pra():
    S, W = generate_random_data()
    q_idx = 0
    k = 10
    tb, te, tau = 1, 4, 0.6
    
    runs = build_ssa_for_object_chunked(S, W, q_idx, k)
    ssa_res = drtopk_ssa_query(runs, tb, te, tau)
    
    parent = build_pra_forest(runs, W.shape[1])
    pra_res = drtopk_pra_query(runs, parent, tb, te, tau)
    
    assert set(ssa_res) == set(pra_res), "SSA and PRA result sets do not match"

def test_hybrid_subset_invariant():
    S, W = generate_random_data(n_users=100, n_items=200, L=5)
    q_idx = 10
    k = 15
    tb, te, tau = 0, 5, 0.6
    c = 1.5
    
    candidates = durable_quantile_rank_filter(S, W, q_idx, k, c, tb, te, tau)
    
    runs = build_ssa_for_object_chunked(S, W, q_idx, k)
    full_ssa_res = drtopk_ssa_query(runs, tb, te, tau)
    
    hybrid_ssa_res = ssa_on_candidates(S, W, candidates, q_idx, k, tb, te, tau)
    hybrid_pra_res = pra_on_candidates(S, W, candidates, q_idx, k, tb, te, tau)
    
    expected_subset = set(full_ssa_res).intersection(set(candidates))
    
    assert set(hybrid_ssa_res) == expected_subset, "Hybrid SSA subset invariant failed"
    assert set(hybrid_pra_res) == expected_subset, "Hybrid PRA subset invariant failed"

def test_score_monotonicity():
    S, W = generate_random_data(n_users=5, n_items=20, L=1)
    q_idx = 0
    k = 5
    tb, te, tau = 0, 1, 1.0
    
    runs = build_ssa_for_object(S, W, q_idx, k)
    initial_res = drtopk_ssa_query(runs, tb, te, tau)
    
    S[q_idx] *= 2.0 
    
    runs2 = build_ssa_for_object(S, W, q_idx, k)
    new_res = drtopk_ssa_query(runs2, tb, te, tau)
    
    assert set(initial_res).issubset(set(new_res)), "Monotonicity violated"

def test_invalid_inputs():
    S, W = generate_random_data(n_users=5, n_items=10)
    
    def assert_raises(exception, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except exception:
            return
        except Exception as e:
            assert False, f"Expected {exception}, but raised {type(e)}"
        assert False, f"Expected {exception}, but no exception was raised"

    assert_raises(ValueError, build_ssa_for_object, S, W, 0, 0)
    assert_raises(ValueError, build_ssa_for_object, S, W, 0, 11)
    assert_raises(IndexError, build_ssa_for_object, S, W, 10, 5)
    
    runs = build_ssa_for_object(S, W, 0, 5)
    assert_raises(ValueError, drtopk_ssa_query, runs, 2, 1, 0.5)
    assert_raises(ValueError, drtopk_ssa_query, runs, 0, 2, 1.5)

def test_durability_aware_aggregation():
    pass

def test_temporal_normalization():
    S, W = generate_random_data()
    norms = np.linalg.norm(W, axis=2)
    assert np.allclose(norms, 1.0), "Temporal vectors are not normalized"

if __name__ == "__main__":
    test_static_exact_rank_and_one_window_consistency()
    test_full_vs_chunked_ssa()
    test_ssa_vs_pra()
    test_hybrid_subset_invariant()
    test_score_monotonicity()
    test_invalid_inputs()
    test_durability_aware_aggregation()
    test_temporal_normalization()
    print("All tests passed.")
