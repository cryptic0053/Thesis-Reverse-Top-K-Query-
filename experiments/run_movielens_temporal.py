import os
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.common_data import load_data
from baselines.approximate.build_rank_table import build_table
from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import hybrid_ssa_query, hybrid_pra_query

OUT_CSV = os.path.join(BASE_DIR, "outputs", "csv", "movielens_temporal")
OUT_PLOTS = os.path.join(BASE_DIR, "outputs", "plots", "movielens_temporal")
os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

D = 32
L = 5
K = 10
TAU = 0.6
C = 1.5
NUM_QUERIES = 5
SEED = 42

rng = np.random.default_rng(SEED)

def main():
    print("--- 1. Loading MovieLens Temporal Data ---")
    S, W = load_data(mode="movielens_temporal", L=L, d=D)
    
    n_users, L_windows, d_f = W.shape
    n_items, d_s = S.shape
    
    # Required prints
    print(f"Shape of S: {S.shape}")
    print(f"Shape of W: {W.shape}")
    print(f"Number of users: {n_users}")
    print(f"Number of items: {n_items}")
    print(f"Dimension d: {d_f}")
    print(f"Number of time windows L: {L_windows}")
    print(f"Number of query items used: {NUM_QUERIES}")
    
    print("\n--- 2. Building Static Rank Tables using W mean ---")
    U_avg = W.mean(axis=1)
    
    # Re-normalize just to be safe for table build thresholds
    eps = 1e-12
    norms = np.linalg.norm(U_avg, axis=1, keepdims=True)
    U_avg_norm = U_avg / np.maximum(norms, eps)
    
    T_table, THR = build_table(U_avg_norm, S, out_dir=os.path.join(BASE_DIR, "outputs"), tau=500, sample_m=4000, seed=SEED)
    
    queries = rng.choice(n_items, size=NUM_QUERIES, replace=False)
    
    print(f"\n--- 3. Running Methods for {NUM_QUERIES} queries ---")
    
    query_log = []
    
    tb, te = 0, L
    
    total_ssa_time = 0
    total_pra_time = 0
    total_hssa_time = 0
    total_hpra_time = 0
    
    total_cand = 0
    total_hssa_recall = 0
    total_hpra_recall = 0
    
    for _, q_idx in enumerate(queries):
        q_idx = int(q_idx)
        print(f"Evaluating Query {q_idx}...")
        
        # 1. Full SSA
        t_ssa_start = time.time()
        runs_ssa = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=4000)
        res_ssa = drtopk_ssa_query(runs_ssa, tb, te, TAU)
        t_ssa = time.time() - t_ssa_start
        set_ssa = set(res_ssa)
        
        # 2. Full PRA
        t_pra_start = time.time()
        # Full PRA needs SSA runs first to build the tree
        runs_pra = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=4000)
        parent_pra = build_pra_forest(runs_pra, L_windows, do_verify=True)
        res_pra = drtopk_pra_query(runs_pra, parent_pra, tb, te, TAU)
        t_pra = time.time() - t_pra_start
        
        # 3. Hybrid SSA
        t_hssa_start = time.time()
        res_hssa_dict = hybrid_ssa_query(S, W, q_idx, K, C, tb, te, TAU, T_table, THR, chunk_size=4000)
        t_hssa = time.time() - t_hssa_start
        set_hssa = set(res_hssa_dict["result_user_ids"])
        
        # 4. Hybrid PRA
        t_hpra_start = time.time()
        res_hpra_dict = hybrid_pra_query(S, W, q_idx, K, C, tb, te, TAU, T_table, THR, chunk_size=4000)
        t_hpra = time.time() - t_hpra_start
        set_hpra = set(res_hpra_dict["result_user_ids"])
        
        cands = res_hssa_dict["num_candidates"]
        pruning_ratio = 1.0 - (cands / n_users)
        
        denom = max(len(set_ssa), 1)
        # However, recall is typically measured relative to the total possible return size or the true result size.
        # Here we compare to Full SSA (true result)
        if len(set_ssa) == 0:
            recall_hssa = 1.0 if len(set_hssa) == 0 else 0.0
            recall_hpra = 1.0 if len(set_hpra) == 0 else 0.0
        else:
            recall_hssa = len(set_ssa.intersection(set_hssa)) / len(set_ssa)
            recall_hpra = len(set_ssa.intersection(set_hpra)) / len(set_ssa)
            
        # Logging
        query_log.append({
            "Query_ID": q_idx,
            "Full_SSA_Time": t_ssa,
            "Full_PRA_Time": t_pra,
            "Hybrid_SSA_Time": t_hssa,
            "Hybrid_PRA_Time": t_hpra,
            "Candidates": cands,
            "Pruning_Ratio": pruning_ratio,
            "Recall_Hybrid_SSA": recall_hssa,
            "Recall_Hybrid_PRA": recall_hpra
        })
        
        total_ssa_time += t_ssa
        total_pra_time += t_pra
        total_hssa_time += t_hssa
        total_hpra_time += t_hpra
        total_cand += cands
        total_hssa_recall += recall_hssa
        total_hpra_recall += recall_hpra
        
    avg_ssa = total_ssa_time / NUM_QUERIES
    avg_pra = total_pra_time / NUM_QUERIES
    avg_hssa = total_hssa_time / NUM_QUERIES
    avg_hpra = total_hpra_time / NUM_QUERIES
    avg_cand = total_cand / NUM_QUERIES
    avg_pruning = 1.0 - (avg_cand / n_users)
    avg_recall_hssa = total_hssa_recall / NUM_QUERIES
    avg_recall_hpra = total_hpra_recall / NUM_QUERIES
    avg_hybrid_recall = (avg_recall_hssa + avg_recall_hpra) / 2.0
    
    # 1. Save detailed query log
    df_log = pd.DataFrame(query_log)
    per_query_csv = os.path.join(OUT_CSV, "per_query_results.csv")
    df_log.to_csv(per_query_csv, index=False)
    
    # 2. Save averages
    summary_df = pd.DataFrame({
        "Method": ["Full SSA", "Full PRA", "Hybrid SSA", "Hybrid PRA"],
        "Runtime_sec": [avg_ssa, avg_pra, avg_hssa, avg_hpra]
    })
    runtime_csv = os.path.join(OUT_CSV, "runtime_compare.csv")
    summary_df.to_csv(runtime_csv, index=False)
    
    # Plots
    # Plot 1: Runtime Bar
    plt.figure(figsize=(8, 5))
    bars = plt.bar(["Full SSA", "Full PRA", "Hybrid SSA", "Hybrid PRA"], 
                   [avg_ssa, avg_pra, avg_hssa, avg_hpra], 
                   color=['#ff9999', '#ffcc99', '#66b3ff', '#99ff99'])
    plt.ylabel("Time (seconds)")
    plt.title("Runtime Comparison (Temporal)")
    for bar, val in zip(bars, [avg_ssa, avg_pra, avg_hssa, avg_hpra]):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.4f}s", ha='center', va='bottom')
    plt.tight_layout()
    plot_runtime = os.path.join(OUT_PLOTS, "runtime_bar.png")
    plt.savefig(plot_runtime)
    plt.close()
    
    # Plot 2: Recall Bar
    plt.figure(figsize=(6, 4))
    bars2 = plt.bar(["Hybrid SSA", "Hybrid PRA"], 
                    [avg_recall_hssa, avg_recall_hpra], color=['#c2c2f0','#ffb3e6'])
    plt.ylabel("Recall Ratio vs Full SSA")
    plt.title("Hybrid Recall Performance")
    plt.ylim(0, 1.1)
    for bar, val in zip(bars2, [avg_recall_hssa, avg_recall_hpra]):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plot_recall = os.path.join(OUT_PLOTS, "recall_bar.png")
    plt.savefig(plot_recall)
    plt.close()
    
    # Plot 3: Pruning Bar
    pruning_vals = df_log["Pruning_Ratio"].values
    query_labels = [f"Q{i}" for i in range(1, NUM_QUERIES+1)]
    plt.figure(figsize=(7, 4))
    plt.bar(query_labels, pruning_vals, color='#ffcc99')
    plt.axhline(y=avg_pruning, color='r', linestyle='--', label=f"Avg: {avg_pruning:.4f}")
    plt.ylabel("Pruning Ratio")
    plt.title("Candidate Pruning Ratio per Query")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plot_pruning = os.path.join(OUT_PLOTS, "pruning_bar.png")
    plt.savefig(plot_pruning)
    plt.close()
    
    print("\n--- Summary ---")
    print(f"average runtime of Full SSA: {avg_ssa:.4f}s")
    print(f"average runtime of Full PRA: {avg_pra:.4f}s")
    print(f"average runtime of Hybrid + SSA: {avg_hssa:.4f}s")
    print(f"average runtime of Hybrid + PRA: {avg_hpra:.4f}s")
    print(f"recall of hybrid methods vs Full SSA: {avg_hybrid_recall:.4f}")
    print(f"average pruning ratio: {avg_pruning:.4f}")
    print(f"\nCSV Paths:\n - {per_query_csv}\n - {runtime_csv}")
    print(f"\nPlot Paths:\n - {plot_runtime}\n - {plot_recall}\n - {plot_pruning}")
   
if __name__ == "__main__":
    main()
