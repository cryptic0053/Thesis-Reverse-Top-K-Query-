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
    rng = np.random.default_rng(42)
    n_users, n_items, d, L = 3000, 1000, 16, 10
    
    for dist in ["UN", "CL"]:
        csv_path = os.path.join(OUT, "csv", f"synthetic_{dist.lower()}", "runtime_compare.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Pull matching shapes from upgraded dynamic synthetic generators
        S, W = load_data(mode="synthetic_paper", n=n_users, m=n_items, d=d, L=L, dist=dist)
        T_table, THR = build_rank_table(W.mean(axis=1), S)
        
        queries = rng.choice(n_items, size=5, replace=False)
        k, c, tb, te, tau, chunk_size = 10, 1.5, 0, L, 0.6, 4000
        
        print(f"--- RUNTIME COMPARE: {dist} ---")
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Query", "Method", "Runtime", "Candidates", "Results", "Recall", "PruningRatio"])
            
            for q_idx in queries:
                # Full SSA
                t0 = time.time()
                runs_all = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size)
                full_ssa_results = drtopk_ssa_query(runs_all, tb, te, tau)
                t_full_ssa = time.time() - t0
                set_full_ssa = set(full_ssa_results)
                writer.writerow([q_idx, "Full_SSA", t_full_ssa, n_users, len(full_ssa_results), 1.0, 0.0])
                
                # Full PRA
                t0 = time.time()
                runs_all_pra = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size)
                parent_all = build_pra_forest(runs_all_pra, L, do_verify=True)
                full_pra_results = drtopk_pra_query(runs_all_pra, parent_all, tb, te, tau)
                t_full_pra = time.time() - t0
                recall_pra = len(set_full_ssa & set(full_pra_results)) / len(set_full_ssa) if set_full_ssa else 1.0
                writer.writerow([q_idx, "Full_PRA", t_full_pra, n_users, len(full_pra_results), recall_pra, 0.0])
                
                # Hybrid SSA
                t0 = time.time()
                h_ssa = hybrid_ssa_query(S, W, q_idx, k, c, tb, te, tau, T_table, THR, chunk_size)
                t_h_ssa = time.time() - t0
                recall_h_ssa = len(set_full_ssa & set(h_ssa['result_user_ids'])) / len(set_full_ssa) if set_full_ssa else 1.0
                pr = 1.0 - (h_ssa['num_candidates'] / n_users)
                writer.writerow([q_idx, "Hybrid_SSA", t_h_ssa, h_ssa['num_candidates'], len(h_ssa['result_user_ids']), recall_h_ssa, pr])
                
                # Hybrid PRA
                t0 = time.time()
                h_pra = hybrid_pra_query(S, W, q_idx, k, c, tb, te, tau, T_table, THR, chunk_size)
                t_h_pra = time.time() - t0
                recall_h_pra = len(set_full_ssa & set(h_pra['result_user_ids'])) / len(set_full_ssa) if set_full_ssa else 1.0
                pr2 = 1.0 - (h_pra['num_candidates'] / n_users)
                writer.writerow([q_idx, "Hybrid_PRA", t_h_pra, h_pra['num_candidates'], len(h_pra['result_user_ids']), recall_h_pra, pr2])
        print(f"Saved {csv_path}")

if __name__ == "__main__":
    main()
