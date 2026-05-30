import os
import sys
import time
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import hybrid_ssa_query, hybrid_pra_query
from data.common_data import load_data
from utils.rank_table import build_rank_table

def main():
    OUT = os.path.join(BASE_DIR, "outputs")
    csv_path = os.path.join(OUT, "csv", "real", "real_validation.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    print(f"--- REAL VALIDATION ---")
    
    # Using L=10 matching the realistic load defaults
    S, W = load_data(mode="real_embeddings", L=10)
    
    n_items = S.shape[0]
    n_users = W.shape[0]
    L = W.shape[1]
    
    T_table, THR = build_rank_table(W.mean(axis=1), S)
    
    rng = np.random.default_rng(42)
    q_idx = int(rng.choice(n_items))
    k, c, tau = 10, 1.5, 0.6
    chunk = 4000
    
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "Method", "Runtime", "Candidates", "Results", "PruningRatio", "Recall"])
        
        # SSA
        t0 = time.time()
        runs_all = build_ssa_for_object_chunked(S, W, q_idx, k, chunk)
        full_res = drtopk_ssa_query(runs_all, 0, L, tau)
        t_full = time.time() - t0
        set_full = set(full_res)
        writer.writerow(["real", "N/A", "Full_SSA", t_full, n_users, len(full_res), 0.0, 1.0])
        print(f"Full SSA Runtime: {t_full:.4f}s")
        
        # Full PRA
        t0 = time.time()
        runs_all_pra = build_ssa_for_object_chunked(S, W, q_idx, k, chunk)
        parent_all = build_pra_forest(runs_all_pra, L, do_verify=True)
        full_pra_results = drtopk_pra_query(runs_all_pra, parent_all, 0, L, tau)
        t_full_pra = time.time() - t0
        writer.writerow(["real", "N/A", "Full_PRA", t_full_pra, n_users, len(full_pra_results), 0.0, 1.0])
        print(f"Full PRA Runtime: {t_full_pra:.4f}s")
        
        # Hybrid SSA
        t0 = time.time()
        h_res = hybrid_ssa_query(S, W, q_idx, k, c, 0, L, tau, T_table, THR, chunk)
        t_hybrid = time.time() - t0
        cur_set = set(h_res['result_user_ids'])
        recall = len(set_full & cur_set) / len(set_full) if set_full else 1.0
        pr = 1.0 - (h_res['num_candidates'] / n_users)
        writer.writerow(["real", "N/A", "Hybrid_SSA", t_hybrid, h_res['num_candidates'], len(cur_set), pr, recall])
        print(f"Hybrid SSA Runtime: {t_hybrid:.4f}s | PR: {pr:.2f} | Recall: {recall:.2f}")
        
        # Hybrid PRA
        t0 = time.time()
        h_pra = hybrid_pra_query(S, W, q_idx, k, c, 0, L, tau, T_table, THR, chunk)
        t_h_pra = time.time() - t0
        cur_set_pra = set(h_pra['result_user_ids'])
        recall_pra = len(set_full & cur_set_pra) / len(set_full) if set_full else 1.0
        pr_pra = 1.0 - (h_pra['num_candidates'] / n_users)
        writer.writerow(["real", "N/A", "Hybrid_PRA", t_h_pra, h_pra['num_candidates'], len(cur_set_pra), pr_pra, recall_pra])
        print(f"Hybrid PRA Runtime: {t_h_pra:.4f}s | PR: {pr_pra:.2f} | Recall: {recall_pra:.2f}")

if __name__ == "__main__":
    main()
