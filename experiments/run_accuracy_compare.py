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
        csv_path = os.path.join(OUT, "csv", f"synthetic_{dist.lower()}", "parameter_sweep.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        print(f"--- ACCURACY SWEEP: {dist} ---")
        
        S, W = load_data(mode="synthetic_paper", n=n_users, m=n_items, d=d, L=L, dist=dist)
        T_table, THR = build_rank_table(W.mean(axis=1), S)
        q_idx = int(rng.choice(n_items))
        
        K_list = [10, 50, 100]
        TAU_list = [0.5, 0.8, 1.0]
        C_list = [1.0, 1.5, 2.0]
        
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["k", "tau", "c", "Method", "Runtime", "Recall", "Candidates"])
            
            for k in K_list:
                for tau in TAU_list:
                    runs_all = build_ssa_for_object_chunked(S, W, q_idx, k)
                    t0 = time.time()
                    full_res = drtopk_ssa_query(runs_all, 0, L, tau)
                    t_full = time.time() - t0
                    set_full = set(full_res)
                    writer.writerow([k, tau, "-", "Full_SSA", t_full, 1.0, n_users])
                    
                    t0 = time.time()
                    parent_all = build_pra_forest(runs_all, L, do_verify=True)
                    full_pra_results = drtopk_pra_query(runs_all, parent_all, 0, L, tau)
                    t_full_pra = time.time() - t0
                    recall_pra = len(set_full & set(full_pra_results)) / len(set_full) if set_full else 1.0
                    writer.writerow([k, tau, "-", "Full_PRA", t_full_pra, recall_pra, n_users])
                    
                    for c in C_list:
                        t0 = time.time()
                        h_res = hybrid_ssa_query(S, W, q_idx, k, c, 0, L, tau, T_table, THR)
                        t_h = time.time() - t0
                        cur_set = set(h_res['result_user_ids'])
                        recall = len(set_full & cur_set) / len(set_full) if set_full else 1.0
                        writer.writerow([k, tau, c, "Hybrid_SSA", t_h, recall, h_res['num_candidates']])
                        
                        t0 = time.time()
                        h_pra = hybrid_pra_query(S, W, q_idx, k, c, 0, L, tau, T_table, THR)
                        t_h_pra = time.time() - t0
                        cur_set_pra = set(h_pra['result_user_ids'])
                        recall_pra_h = len(set_full & cur_set_pra) / len(set_full) if set_full else 1.0
                        writer.writerow([k, tau, c, "Hybrid_PRA", t_h_pra, recall_pra_h, h_pra['num_candidates']])
                        
                        print(f"Sweep {dist} | k={k}, tau={tau}, c={c} | H_SSA Recall: {recall:.2f} | H_PRA Recall: {recall_pra_h:.2f}")

if __name__ == "__main__":
    main()
