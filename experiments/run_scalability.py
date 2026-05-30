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

def run_experiment(writer, dist, param_name, param_val, n, m, d, L):
    S, W = load_data(mode="synthetic_paper", n=n, m=m, d=d, L=L, dist=dist)
    
    U_avg = W.mean(axis=1)
    T_table, THR = build_rank_table(U_avg, S)
    
    q_idx = int(np.random.default_rng(42).choice(m))
    k, c, tau = 10, 1.5, 0.6
    chunk = 4000
    
    # SSA
    t0 = time.time()
    runs_all = build_ssa_for_object_chunked(S, W, q_idx, k, chunk)
    full_res = drtopk_ssa_query(runs_all, 0, L, tau)
    t_full = time.time() - t0
    set_full = set(full_res)
    writer.writerow([param_name, param_val, "Full_SSA", t_full, n, len(full_res), 0.0, 1.0])
    
    # Full PRA
    t0 = time.time()
    runs_all_pra = build_ssa_for_object_chunked(S, W, q_idx, k, chunk)
    parent_all = build_pra_forest(runs_all_pra, L, do_verify=True)
    full_pra_results = drtopk_pra_query(runs_all_pra, parent_all, 0, L, tau)
    t_full_pra = time.time() - t0
    writer.writerow([param_name, param_val, "Full_PRA", t_full_pra, n, len(full_pra_results), 0.0, 1.0])
    
    # Hybrid SSA
    t0 = time.time()
    h_res = hybrid_ssa_query(S, W, q_idx, k, c, 0, L, tau, T_table, THR, chunk)
    t_hybrid = time.time() - t0
    cur_set = set(h_res['result_user_ids'])
    recall = len(set_full & cur_set) / len(set_full) if set_full else 1.0
    pr = 1.0 - (h_res['num_candidates'] / n)
    writer.writerow([param_name, param_val, "Hybrid_SSA", t_hybrid, h_res['num_candidates'], len(cur_set), pr, recall])
    
    # Hybrid PRA
    t0 = time.time()
    h_pra = hybrid_pra_query(S, W, q_idx, k, c, 0, L, tau, T_table, THR, chunk)
    t_h_pra = time.time() - t0
    cur_set_pra = set(h_pra['result_user_ids'])
    recall_pra = len(set_full & cur_set_pra) / len(set_full) if set_full else 1.0
    pr_pra = 1.0 - (h_pra['num_candidates'] / n)
    writer.writerow([param_name, param_val, "Hybrid_PRA", t_h_pra, h_pra['num_candidates'], len(cur_set_pra), pr_pra, recall_pra])
    print(f"[{dist}] Scale {param_name}={param_val} | Hybrid Speedup: {t_full/t_hybrid if t_hybrid>0 else 0:.2f}x | PR: {pr:.2f}")

def main():
    OUT = os.path.join(BASE_DIR, "outputs")
    for dist in ["UN", "CL"]:
        csv_path = os.path.join(OUT, "csv", f"synthetic_{dist.lower()}", "scalability_results.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        print(f"--- RUNNING SCALABILITY: {dist} ---")
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value", "Method", "Runtime", "Candidates", "Results", "PruningRatio", "Recall"])
            
            for n in [1000, 3000, 5000]:
                run_experiment(writer, dist, "users", n, n=n, m=1000, d=16, L=10)
            for m in [500, 1000, 5000]:
                run_experiment(writer, dist, "items", m, n=3000, m=m, d=16, L=10)
            for L in [5, 10, 20]:
                run_experiment(writer, dist, "timestamps", L, n=3000, m=1000, d=16, L=L)

if __name__ == "__main__":
    main()
