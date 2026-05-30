import os
import sys
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import hybrid_ssa_query, hybrid_pra_query

def main():
    OUT = os.path.join(BASE_DIR, "outputs")
    u_path = os.path.join(OUT, "U_user_vectors.npy")
    p_path = os.path.join(OUT, "P_item_vectors.npy")
    
    U = np.load(u_path)  # Static preferences acting as basis
    P = np.load(p_path)  # Static items
    
    n_users, d = U.shape
    n_items, _ = P.shape
    L = 10  # Evaluate across 10 temporal timestamps
    
    # Simulate W (temporal preferences)
    rng = np.random.default_rng(42)
    noise = rng.normal(scale=0.03, size=(n_users, L, d)).astype(np.float32)
    W = U[:, None, :] + noise
    W /= np.linalg.norm(W, axis=2, keepdims=True)  # maintain unit vector norms
    
    S = P
    
    q_idx = 42
    k = 50
    c = 1.5
    tb = 0
    te = L
    tau = 0.8
    chunk_size = 4000
    
    print(f"--- COMPARISON DEMO: Query Item {q_idx} ---")
    print(f"Params: k={k}, cand_relax={c}, interval=({tb}, {te}), tau={tau}\n")
    
    # ---------------- 1. FULL SSA BASELINE ----------------
    t0 = time.time()
    runs_all = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size=chunk_size)
    full_ssa_results = drtopk_ssa_query(runs_all, tb, te, tau)
    t1 = time.time()
    time_full_ssa = t1 - t0
    
    print(f"[1. Full SSA] Evaluated Users: {n_users}")
    print(f"[1. Full SSA] Matches Found: {len(full_ssa_results)}")
    print(f"[1. Full SSA] Runtime: {time_full_ssa:.4f} sec\n")
    
    # ---------------- 2. FULL PRA BASELINE ----------------
    t0 = time.time()
    runs_all_pra = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size=chunk_size)
    parent_all = build_pra_forest(runs_all_pra, L, do_verify=True)
    full_pra_results = drtopk_pra_query(runs_all_pra, parent_all, tb, te, tau)
    t1 = time.time()
    time_full_pra = t1 - t0
    
    print(f"[2. Full PRA] Evaluated Users: {n_users}")
    print(f"[2. Full PRA] Matches Found: {len(full_pra_results)}")
    print(f"[2. Full PRA] Runtime: {time_full_pra:.4f} sec\n")
    
    # ---------------- 3. HYBRID SSA -----------------------
    t0 = time.time()
    hybrid_ssa_res = hybrid_ssa_query(S, W, q_idx, k, c, tb, te, tau, chunk_size=chunk_size)
    hybrid_ssa_results = hybrid_ssa_res["result_user_ids"]
    t1 = time.time()
    time_hybrid_ssa = t1 - t0
    
    print(f"[3. Hybrid + SSA] Candidates Filtered: {hybrid_ssa_res['num_candidates']} / {n_users} total")
    print(f"[3. Hybrid + SSA] Matches Found: {len(hybrid_ssa_results)}")
    print(f"[3. Hybrid + SSA] Runtime: {time_hybrid_ssa:.4f} sec\n")
    
    # ---------------- 4. HYBRID PRA -----------------------
    t0 = time.time()
    hybrid_pra_res = hybrid_pra_query(S, W, q_idx, k, c, tb, te, tau, chunk_size=chunk_size)
    hybrid_pra_results = hybrid_pra_res["result_user_ids"]
    t1 = time.time()
    time_hybrid_pra = t1 - t0
    
    print(f"[4. Hybrid + PRA] Candidates Filtered: {hybrid_pra_res['num_candidates']} / {n_users} total")
    print(f"[4. Hybrid + PRA] Matches Found: {len(hybrid_pra_results)}")
    print(f"[4. Hybrid + PRA] Runtime: {time_hybrid_pra:.4f} sec\n")
    
    # ---------------- OVERLAP COMPARISONS -----------------
    set_full_ssa = set(full_ssa_results)
    
    def recall(test_set):
        overlap = len(set_full_ssa & test_set)
        return (overlap / len(set_full_ssa) * 100) if len(set_full_ssa) > 0 else 100
    
    print(f"--- OUTCOMES VS FULL SSA ---")
    print(f"Recall Full PRA:      {recall(set(full_pra_results)):.2f}%")
    print(f"Recall Hybrid + SSA:  {recall(set(hybrid_ssa_results)):.2f}% (Speedup vs Full SSA: {time_full_ssa/time_hybrid_ssa if time_hybrid_ssa > 0 else 0:.2f}x)")
    print(f"Recall Hybrid + PRA:  {recall(set(hybrid_pra_results)):.2f}% (Speedup vs Full PRA: {time_full_pra/time_hybrid_pra if time_hybrid_pra > 0 else 0:.2f}x)")

if __name__ == "__main__":
    main()
