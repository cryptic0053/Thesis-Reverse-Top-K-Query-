"""
run_temporal_filter_compare.py
==============================
Compare candidate-generation strategies for the MovieLens temporal pipeline.

Strategies evaluated
--------------------
1. Global-average filter  c = 1.5  (current default)
2. Global-average filter  c = 2.0
3. Global-average filter  c = 3.0
4. Per-window union filter  c = 1.5

For each strategy we run:
  - Full SSA   (ground truth)
  - Full PRA
  - Hybrid + SSA  (using the strategy's candidate set)
  - Hybrid + PRA  (using the strategy's candidate set)

Outputs
-------
  outputs/csv/movielens_temporal_filter_compare/filter_strategy_compare.csv
  outputs/plots/movielens_temporal_filter_compare/recall_vs_strategy.png
  outputs/plots/movielens_temporal_filter_compare/runtime_vs_strategy.png
  outputs/plots/movielens_temporal_filter_compare/pruning_vs_strategy.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.common_data import load_data
from baselines.approximate.build_rank_table import build_table
from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import (
    hybrid_ssa_query, hybrid_pra_query,
    hybrid_ssa_query_union, hybrid_pra_query_union,
    hybrid_ssa_query_minrank, hybrid_pra_query_minrank,
)

# ── output dirs ──
OUT_CSV = os.path.join(BASE_DIR, "outputs", "csv", "movielens_temporal_filter_compare")
OUT_PLOTS = os.path.join(BASE_DIR, "outputs", "plots", "movielens_temporal_filter_compare")
os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# ── hyper-parameters (fixed) ──
D = 32
L = 5
K = 10
TAU_DUR = 0.6          # durability threshold
NUM_QUERIES = 5
SEED = 42
TAU_TABLE = 500        # rank-table resolution

rng = np.random.default_rng(SEED)


# ── helpers ──────────────────────────────────────────────────────────────

def _build_global_avg_table(W, S):
    """Build one rank table using the time-averaged user vector."""
    eps = 1e-12
    U_avg = W.mean(axis=1)
    norms = np.linalg.norm(U_avg, axis=1, keepdims=True)
    U_avg_norm = U_avg / np.maximum(norms, eps)
    T_table, THR = build_table(
        U_avg_norm, S,
        out_dir=os.path.join(BASE_DIR, "outputs"),
        tau=TAU_TABLE, sample_m=4000, seed=SEED
    )
    return T_table, THR


def _build_per_window_tables(W, S):
    """Build one rank table per time window using W[:, t, :]."""
    eps = 1e-12
    n_users, L_w, d = W.shape
    T_list, THR_list = [], []
    for t in range(L_w):
        wt = W[:, t, :].copy()
        norms = np.linalg.norm(wt, axis=1, keepdims=True)
        wt_norm = wt / np.maximum(norms, eps)
        T_t, THR_t = build_table(
            wt_norm, S,
            out_dir=os.path.join(BASE_DIR, "outputs"),
            tau=TAU_TABLE, sample_m=4000, seed=SEED
        )
        T_list.append(T_t)
        THR_list.append(THR_t)
    return T_list, THR_list


def _recall(ground_truth_set, predicted_set):
    if len(ground_truth_set) == 0:
        return 1.0 if len(predicted_set) == 0 else 0.0
    return len(ground_truth_set & predicted_set) / len(ground_truth_set)


# ── main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MovieLens Temporal - Candidate Filter Strategy Comparison")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading MovieLens temporal data ...")
    S, W = load_data(mode="movielens_temporal", L=L, d=D)
    n_users, L_win, d_f = W.shape
    n_items = S.shape[0]
    print(f"  S: {S.shape}  W: {W.shape}")

    # 2. Build rank tables
    print("\n[2/4] Building rank tables ...")
    print("  -> global-average table ...")
    T_global, THR_global = _build_global_avg_table(W, S)

    print("  -> per-window tables (L tables) ...")
    T_list, THR_list = _build_per_window_tables(W, S)

    # 3. Select query items
    queries = rng.choice(n_items, size=NUM_QUERIES, replace=False)
    tb, te = 0, L

    # 4. Define strategies
    #    With only 610 users and k=10, c=1.5 gives 15 candidates,
    #    c=10 gives 100 candidates, c=30 gives 300 candidates (about half the users).
    strategies = [
        {"name": "GlobalAvg c=1.5",   "kind": "global",  "c": 1.5},
        {"name": "GlobalAvg c=2.0",   "kind": "global",  "c": 2.0},
        {"name": "GlobalAvg c=3.0",   "kind": "global",  "c": 3.0},
        {"name": "GlobalAvg c=5.0",   "kind": "global",  "c": 5.0},
        {"name": "GlobalAvg c=10.0",  "kind": "global",  "c": 10.0},
        {"name": "GlobalAvg c=20.0",  "kind": "global",  "c": 20.0},
        {"name": "GlobalAvg c=30.0",  "kind": "global",  "c": 30.0},
        {"name": "UnionWin c=1.5",    "kind": "union",   "c": 1.5},
        {"name": "UnionWin c=3.0",    "kind": "union",   "c": 3.0},
        {"name": "UnionWin c=5.0",    "kind": "union",   "c": 5.0},
        {"name": "UnionWin c=10.0",   "kind": "union",   "c": 10.0},
        {"name": "UnionWin c=20.0",   "kind": "union",   "c": 20.0},
        {"name": "MinRank c=1.5",     "kind": "minrank", "c": 1.5},
        {"name": "MinRank c=3.0",     "kind": "minrank", "c": 3.0},
        {"name": "MinRank c=5.0",     "kind": "minrank", "c": 5.0},
        {"name": "MinRank c=10.0",    "kind": "minrank", "c": 10.0},
        {"name": "MinRank c=20.0",    "kind": "minrank", "c": 20.0},
    ]

    # Collect per-query results
    all_rows = []

    print(f"\n[3/4] Evaluating {len(strategies)} strategies x {NUM_QUERIES} queries ...\n")

    for strat in strategies:
        strat_name = strat["name"]
        c_val = strat["c"]
        print(f"  Strategy: {strat_name}")

        strat_recall_hssa = []
        strat_recall_hpra = []
        strat_cand_counts = []
        strat_times = {"Full SSA": [], "Full PRA": [], "Hybrid SSA": [], "Hybrid PRA": []}

        for qi, q_idx in enumerate(queries):
            q_idx = int(q_idx)

            # ── Full SSA (ground truth) ──
            t0 = time.time()
            runs_ssa = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=4000)
            res_ssa = drtopk_ssa_query(runs_ssa, tb, te, TAU_DUR)
            t_ssa = time.time() - t0
            set_ssa = set(res_ssa)

            # ── Full PRA ──
            t0 = time.time()
            runs_pra = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=4000)
            parent_pra = build_pra_forest(runs_pra, L_win, do_verify=True)
            res_pra = drtopk_pra_query(runs_pra, parent_pra, tb, te, TAU_DUR)
            t_pra = time.time() - t0

            # ── Hybrid SSA ──
            t0 = time.time()
            if strat["kind"] == "global":
                res_hssa = hybrid_ssa_query(
                    S, W, q_idx, K, c_val, tb, te, TAU_DUR,
                    T_global, THR_global, chunk_size=4000
                )
            elif strat["kind"] == "union":
                res_hssa = hybrid_ssa_query_union(
                    S, W, q_idx, K, c_val, tb, te, TAU_DUR,
                    T_list, THR_list, chunk_size=4000
                )
            else:  # minrank
                res_hssa = hybrid_ssa_query_minrank(
                    S, W, q_idx, K, c_val, tb, te, TAU_DUR, chunk_size=4000
                )
            t_hssa = time.time() - t0
            set_hssa = set(res_hssa["result_user_ids"])

            # ── Hybrid PRA ──
            t0 = time.time()
            if strat["kind"] == "global":
                res_hpra = hybrid_pra_query(
                    S, W, q_idx, K, c_val, tb, te, TAU_DUR,
                    T_global, THR_global, chunk_size=4000
                )
            elif strat["kind"] == "union":
                res_hpra = hybrid_pra_query_union(
                    S, W, q_idx, K, c_val, tb, te, TAU_DUR,
                    T_list, THR_list, chunk_size=4000
                )
            else:  # minrank
                res_hpra = hybrid_pra_query_minrank(
                    S, W, q_idx, K, c_val, tb, te, TAU_DUR, chunk_size=4000
                )
            t_hpra = time.time() - t0
            set_hpra = set(res_hpra["result_user_ids"])

            cands = res_hssa["num_candidates"]
            pruning = 1.0 - cands / n_users
            recall_hssa = _recall(set_ssa, set_hssa)
            recall_hpra = _recall(set_ssa, set_hpra)

            strat_recall_hssa.append(recall_hssa)
            strat_recall_hpra.append(recall_hpra)
            strat_cand_counts.append(cands)
            strat_times["Full SSA"].append(t_ssa)
            strat_times["Full PRA"].append(t_pra)
            strat_times["Hybrid SSA"].append(t_hssa)
            strat_times["Hybrid PRA"].append(t_hpra)

            all_rows.append({
                "Strategy":       strat_name,
                "Query_ID":       q_idx,
                "Full_SSA_Time":  t_ssa,
                "Full_PRA_Time":  t_pra,
                "Hybrid_SSA_Time": t_hssa,
                "Hybrid_PRA_Time": t_hpra,
                "Candidates":     cands,
                "Pruning_Ratio":  pruning,
                "Recall_HSSA":    recall_hssa,
                "Recall_HPRA":    recall_hpra,
                "FullSSA_Count":  len(set_ssa),
                "HSSA_Count":     len(set_hssa),
                "HPRA_Count":     len(set_hpra),
            })

        avg_recall_hssa = np.mean(strat_recall_hssa)
        avg_recall_hpra = np.mean(strat_recall_hpra)
        avg_cands = np.mean(strat_cand_counts)
        avg_pruning = 1.0 - avg_cands / n_users
        avg_t_hssa = np.mean(strat_times["Hybrid SSA"])
        avg_t_hpra = np.mean(strat_times["Hybrid PRA"])

        print(f"    recall HSSA: {avg_recall_hssa:.4f}   recall HPRA: {avg_recall_hpra:.4f}")
        print(f"    candidates: {avg_cands:.0f}/{n_users}   pruning: {avg_pruning:.4f}")
        print(f"    HSSA time: {avg_t_hssa:.4f}s   HPRA time: {avg_t_hpra:.4f}s\n")

    # ── Save CSV ──
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_CSV, "filter_strategy_compare.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved -> {csv_path}")

    # ── Build summary per strategy for plots ──
    summary_rows = []
    for strat_name in df["Strategy"].unique():
        sub = df[df["Strategy"] == strat_name]
        summary_rows.append({
            "Strategy":          strat_name,
            "Recall_HSSA":       sub["Recall_HSSA"].mean(),
            "Recall_HPRA":       sub["Recall_HPRA"].mean(),
            "Pruning_Ratio":     sub["Pruning_Ratio"].mean(),
            "Candidates":        sub["Candidates"].mean(),
            "Hybrid_SSA_Time":   sub["Hybrid_SSA_Time"].mean(),
            "Hybrid_PRA_Time":   sub["Hybrid_PRA_Time"].mean(),
            "Full_SSA_Time":     sub["Full_SSA_Time"].mean(),
            "Full_PRA_Time":     sub["Full_PRA_Time"].mean(),
        })
    sdf = pd.DataFrame(summary_rows)

    # ── Plots ──
    print("\n[4/4] Generating plots ...")

    strategies_labels = sdf["Strategy"].tolist()
    x = np.arange(len(strategies_labels))
    width = 0.35

    # --- Recall ---
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, sdf["Recall_HSSA"], width, label="Hybrid SSA", color="#5b8def")
    b2 = ax.bar(x + width/2, sdf["Recall_HPRA"], width, label="Hybrid PRA", color="#f26d6d")
    ax.set_ylabel("Recall vs Full SSA")
    ax.set_title("Recall by Filter Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    for bar_set in [b1, b2]:
        for bar in bar_set:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    recall_path = os.path.join(OUT_PLOTS, "recall_vs_strategy.png")
    fig.savefig(recall_path, dpi=150)
    plt.close(fig)
    print(f"  {recall_path}")

    # --- Runtime ---
    fig, ax = plt.subplots(figsize=(9, 5))
    w4 = 0.2
    b1 = ax.bar(x - 1.5*w4, sdf["Full_SSA_Time"],    w4, label="Full SSA",    color="#ff9999")
    b2 = ax.bar(x - 0.5*w4, sdf["Full_PRA_Time"],    w4, label="Full PRA",    color="#ffcc99")
    b3 = ax.bar(x + 0.5*w4, sdf["Hybrid_SSA_Time"],  w4, label="Hybrid SSA",  color="#66b3ff")
    b4 = ax.bar(x + 1.5*w4, sdf["Hybrid_PRA_Time"],  w4, label="Hybrid PRA",  color="#99ff99")
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime by Filter Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_labels, rotation=15, ha="right")
    ax.legend(fontsize=8)
    for bar_set in [b1, b2, b3, b4]:
        for bar in bar_set:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    runtime_path = os.path.join(OUT_PLOTS, "runtime_vs_strategy.png")
    fig.savefig(runtime_path, dpi=150)
    plt.close(fig)
    print(f"  {runtime_path}")

    # --- Pruning ---
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(strategies_labels, sdf["Pruning_Ratio"], color="#c9a0dc")
    ax.set_ylabel("Pruning Ratio")
    ax.set_title("Pruning Ratio by Filter Strategy")
    ax.set_ylim(0, 1.15)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.4f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    pruning_path = os.path.join(OUT_PLOTS, "pruning_vs_strategy.png")
    fig.savefig(pruning_path, dpi=150)
    plt.close(fig)
    print(f"  {pruning_path}")

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(sdf[["Strategy", "Recall_HSSA", "Recall_HPRA",
               "Pruning_Ratio", "Candidates",
               "Hybrid_SSA_Time", "Hybrid_PRA_Time"]].to_string(index=False))
    print("=" * 60)

    # Pick best balance (highest avg recall with reasonable pruning)
    sdf["Avg_Recall"] = (sdf["Recall_HSSA"] + sdf["Recall_HPRA"]) / 2
    sdf["Avg_Hybrid_Time"] = (sdf["Hybrid_SSA_Time"] + sdf["Hybrid_PRA_Time"]) / 2
    best_idx = sdf["Avg_Recall"].idxmax()
    best = sdf.loc[best_idx]
    print(f"\n  * Best recall strategy: {best['Strategy']}")
    print(f"    avg recall = {best['Avg_Recall']:.4f}")
    print(f"    pruning    = {best['Pruning_Ratio']:.4f}")
    print(f"    avg hybrid time = {best['Avg_Hybrid_Time']:.4f}s")
    print()


if __name__ == "__main__":
    main()
