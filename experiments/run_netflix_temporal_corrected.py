import os
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.common_data import load_data
from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import (
    hybrid_ssa_query_durable_rank, hybrid_pra_query_durable_rank,
    hybrid_ssa_query_minrank, hybrid_pra_query_minrank
)
from utils.rank_table import build_rank_table

OUT_CSV   = os.path.join(BASE_DIR, "outputs", "csv",   "corrected_netflix_temporal")
OUT_PLOTS = os.path.join(BASE_DIR, "outputs", "plots", "corrected")
os.makedirs(OUT_CSV,   exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

D = 32
L = 5
K = 10
TAU = 0.6
C = 2.0
NUM_QUERIES = 50
SEED = 42

MAX_USERS = 5000
MAX_ITEMS = 3000

rng = np.random.default_rng(SEED)

def calc_metrics(baseline_set, approx_set):
    tp = len(baseline_set & approx_set)
    fp = len(approx_set - baseline_set)
    fn = len(baseline_set - approx_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = (baseline_set == approx_set)
    
    return tp, fp, fn, precision, recall, f1, exact_match

def main():
    print("=" * 64)
    print("  Corrected Netflix Temporal Evaluation")
    print("=" * 64)

    S, W = load_data(mode="netflix_temporal", L=L, d=D, max_users=MAX_USERS, max_items=MAX_ITEMS)
    n_users, L_win, d_f = W.shape
    n_items, d_s = S.shape

    queries = rng.choice(n_items, size=NUM_QUERIES, replace=False)
    tb, te = 0, L
    
    # Precompute rank tables
    t0 = time.time()
    T_table_list, THR_list = [], []
    for t in range(L):
        T_t, THR_t = build_rank_table(W[:, t, :], S)
        T_table_list.append(T_t)
        THR_list.append(THR_t)
    print(f"Tables built in {time.time()-t0:.2f}s")

    query_log = []
    
    methods = [
        "Full_SSA", "Full_PRA",
        "Hybrid_SSA_DurableRank", "Hybrid_PRA_DurableRank",
        "Hybrid_SSA_LegacyMinRank", "Hybrid_PRA_LegacyMinRank"
    ]

    for qi, q_idx in enumerate(queries):
        q_idx = int(q_idx)
        print(f"  Query {qi+1}/{NUM_QUERIES}  (item {q_idx})")
        
        row = {"query_item": q_idx}
        
        # 1. Full SSA
        t0 = time.time()
        runs_ssa = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=4000)
        res_ssa = drtopk_ssa_query(runs_ssa, tb, te, TAU)
        t_ssa = time.time() - t0
        set_ssa = set(res_ssa)
        
        row["Full_SSA_time"] = t_ssa
        row["Full_SSA_count"] = len(set_ssa)
        baseline_empty = (len(set_ssa) == 0)
        row["baseline_empty"] = baseline_empty
        
        # 2. Full PRA
        t0 = time.time()
        runs_pra = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=4000)
        parent = build_pra_forest(runs_pra, L_win)
        res_pra = drtopk_pra_query(runs_pra, parent, tb, te, TAU)
        t_pra = time.time() - t0
        set_pra = set(res_pra)
        
        row["Full_PRA_time"] = t_pra
        row["Full_PRA_count"] = len(set_pra)
        
        hybrid_calls = [
            ("Hybrid_SSA_DurableRank", lambda: hybrid_ssa_query_durable_rank(S, W, q_idx, K, C, tb, te, TAU, T_table_list, THR_list)),
            ("Hybrid_PRA_DurableRank", lambda: hybrid_pra_query_durable_rank(S, W, q_idx, K, C, tb, te, TAU, T_table_list, THR_list)),
            ("Hybrid_SSA_LegacyMinRank", lambda: hybrid_ssa_query_minrank(S, W, q_idx, K, C, tb, te, TAU)),
            ("Hybrid_PRA_LegacyMinRank", lambda: hybrid_pra_query_minrank(S, W, q_idx, K, C, tb, te, TAU))
        ]
        
        for name, func in hybrid_calls:
            t0 = time.time()
            res = func()
            t_hyb = time.time() - t0
            
            cands = res["num_candidates"]
            set_hyb = set(res["result_user_ids"])
            
            tp, fp, fn, prec, rec, f1, exact = calc_metrics(set_ssa, set_hyb)
            
            row[f"{name}_time"] = t_hyb
            row[f"{name}_cands"] = cands
            row[f"{name}_pruning"] = 1.0 - cands / n_users
            row[f"{name}_tp"] = tp
            row[f"{name}_fp"] = fp
            row[f"{name}_fn"] = fn
            row[f"{name}_prec"] = prec
            row[f"{name}_recall"] = rec
            row[f"{name}_f1"] = f1
            row[f"{name}_exact"] = exact
            
        query_log.append(row)

    df = pd.DataFrame(query_log)
    df.to_csv(os.path.join(OUT_CSV, "per_query_results.csv"), index=False)
    
    df_non_empty = df[~df["baseline_empty"]]
    num_non_empty = len(df_non_empty)
    print(f"\nSummary ({num_non_empty} queries with non-empty baseline out of {NUM_QUERIES}):")
    
    summary = []
    for name in methods:
        if name.startswith("Full"):
            avg_time = df[f"{name}_time"].mean()
            print(f"  {name:25s} | Time: {avg_time:.4f}s")
        else:
            avg_time = df[f"{name}_time"].mean()
            avg_rec = df_non_empty[f"{name}_recall"].mean() if num_non_empty > 0 else float('nan')
            avg_prec = df_non_empty[f"{name}_prec"].mean() if num_non_empty > 0 else float('nan')
            avg_prun = df[f"{name}_pruning"].mean()
            avg_exact = df[f"{name}_exact"].mean()
            print(f"  {name:25s} | Time: {avg_time:.4f}s | Rec: {avg_rec:.4f} | Prec: {avg_prec:.4f} | Pruning: {avg_prun:.4f} | Exact: {avg_exact:.4f}")
            summary.append({
                "Method": name,
                "Time": avg_time,
                "Recall": avg_rec,
                "Precision": avg_prec,
                "Pruning": avg_prun,
                "ExactMatch": avg_exact
            })
            
    pd.DataFrame(summary).to_csv(os.path.join(OUT_CSV, "summary.csv"), index=False)
    
if __name__ == "__main__":
    main()
