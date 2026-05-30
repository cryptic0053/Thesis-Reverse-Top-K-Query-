"""
run_movielens_temporal_robustness.py
====================================
Robustness evaluation of the MinRank-based temporal hybrid method.

Sweeps k, tau, L, d, c individually (one-at-a-time) while holding the
other parameters at their defaults.  Records runtime, recall vs Full SSA,
pruning ratio, and average candidate count.

Outputs
-------
  outputs/csv/movielens_temporal_robustness/robustness_all.csv
  outputs/csv/movielens_temporal_robustness/robustness_<sweep>.csv
  outputs/csv/movielens_temporal_final/final_summary_table.csv
  outputs/plots/movielens_temporal_robustness/runtime_vs_k.png
  outputs/plots/movielens_temporal_robustness/runtime_vs_tau.png
  outputs/plots/movielens_temporal_robustness/runtime_vs_L.png
  outputs/plots/movielens_temporal_robustness/runtime_vs_d.png
  outputs/plots/movielens_temporal_robustness/recall_vs_k.png
  outputs/plots/movielens_temporal_robustness/recall_vs_tau.png
  outputs/plots/movielens_temporal_robustness/pruning_vs_c.png
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

from data.common_data import (
    load_data,
    generate_movielens_static,
    generate_movielens_temporal,
)
from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import (
    hybrid_ssa_query_minrank,
    hybrid_pra_query_minrank,
)

# ── output dirs ─────────────────────────────────────────────────────────
OUT_CSV   = os.path.join(BASE_DIR, "outputs", "csv",   "movielens_temporal_robustness")
OUT_PLOTS = os.path.join(BASE_DIR, "outputs", "plots", "movielens_temporal_robustness")
OUT_FINAL = os.path.join(BASE_DIR, "outputs", "csv",   "movielens_temporal_final")
os.makedirs(OUT_CSV,   exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_FINAL, exist_ok=True)

# ── defaults ────────────────────────────────────────────────────────────
D_DEF   = 32
L_DEF   = 5
K_DEF   = 10
TAU_DEF = 0.6
C_DEF   = 1.5

NUM_QUERIES = 5
SEED        = 42

# ── sweep ranges ────────────────────────────────────────────────────────
K_VALS   = [5, 10, 20]
TAU_VALS = [0.4, 0.6, 0.8]
L_VALS   = [3, 5, 10]
D_VALS   = [16, 32, 64]
C_VALS   = [1.5, 2.0, 3.0]

# ────────────────────────────────────────────────────────────────────────
#  core evaluator
# ────────────────────────────────────────────────────────────────────────
def evaluate(S, W, queries, k, tau, c, L_win):
    """Run Full SSA/PRA + Hybrid SSA/PRA (MinRank) and return avg metrics."""
    n_users = W.shape[0]
    tb, te = 0, L_win
    nq = len(queries)

    t_ssa = t_pra = t_hssa = t_hpra = 0.0
    sum_cand = 0
    sum_recall_hssa = sum_recall_hpra = 0.0

    for q_idx in queries:
        q_idx = int(q_idx)

        # Full SSA
        t0 = time.time()
        runs = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size=4000)
        res_ssa = drtopk_ssa_query(runs, tb, te, tau)
        t_ssa += time.time() - t0
        set_ssa = set(res_ssa)

        # Full PRA
        t0 = time.time()
        runs = build_ssa_for_object_chunked(S, W, q_idx, k, chunk_size=4000)
        parent = build_pra_forest(runs, L_win, do_verify=True)
        res_pra = drtopk_pra_query(runs, parent, tb, te, tau)
        t_pra += time.time() - t0

        # Hybrid SSA (MinRank)
        t0 = time.time()
        rh = hybrid_ssa_query_minrank(S, W, q_idx, k, c, tb, te, tau,
                                      chunk_size=4000)
        t_hssa += time.time() - t0
        set_hssa = set(rh["result_user_ids"])

        # Hybrid PRA (MinRank)
        t0 = time.time()
        rh2 = hybrid_pra_query_minrank(S, W, q_idx, k, c, tb, te, tau,
                                       chunk_size=4000)
        t_hpra += time.time() - t0
        set_hpra = set(rh2["result_user_ids"])

        sum_cand += rh["num_candidates"]

        if len(set_ssa) == 0:
            sum_recall_hssa += (1.0 if len(set_hssa) == 0 else 0.0)
            sum_recall_hpra += (1.0 if len(set_hpra) == 0 else 0.0)
        else:
            sum_recall_hssa += len(set_ssa & set_hssa) / len(set_ssa)
            sum_recall_hpra += len(set_ssa & set_hpra) / len(set_ssa)

    avg_cand = sum_cand / nq
    return {
        "avg_ssa_time":    t_ssa    / nq,
        "avg_pra_time":    t_pra    / nq,
        "avg_hssa_time":   t_hssa   / nq,
        "avg_hpra_time":   t_hpra   / nq,
        "avg_recall_hssa": sum_recall_hssa / nq,
        "avg_recall_hpra": sum_recall_hpra / nq,
        "avg_candidates":  avg_cand,
        "avg_pruning":     1.0 - avg_cand / n_users,
        "n_users":         n_users,
    }

# ────────────────────────────────────────────────────────────────────────
#  plotting helpers
# ────────────────────────────────────────────────────────────────────────
METHOD_STYLES = [
    ("avg_ssa_time",  "Full SSA",        "#e74c3c", "o"),
    ("avg_pra_time",  "Full PRA",        "#e67e22", "s"),
    ("avg_hssa_time", "Hybrid SSA (MR)", "#3498db", "^"),
    ("avg_hpra_time", "Hybrid PRA (MR)", "#2ecc71", "D"),
]

def _plot_runtime(df, xcol, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label, color, marker in METHOD_STYLES:
        ax.plot(df[xcol], df[col], marker=marker, color=color,
                label=label, linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(f"Runtime vs {xlabel} (MinRank Filter)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _plot_recall(df, xcol, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[xcol], df["avg_recall_hssa"], marker="^", color="#3498db",
            label="Hybrid SSA (MR)", linewidth=2, markersize=7)
    ax.plot(df[xcol], df["avg_recall_hpra"], marker="D", color="#2ecc71",
            label="Hybrid PRA (MR)", linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Recall vs Full SSA", fontsize=12)
    ax.set_title(f"Recall vs {xlabel} (MinRank Filter)", fontsize=13)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _plot_pruning(df, xcol, xlabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[xcol], df["avg_pruning"], marker="o", color="#9b59b6",
            linewidth=2, markersize=8)
    for x, y in zip(df[xcol], df["avg_pruning"]):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Pruning Ratio", fontsize=12)
    ax.set_title(f"Pruning vs {xlabel} (MinRank Filter)", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  Robustness Evaluation -- MinRank Temporal Hybrid")
    print("=" * 64)

    all_rows = []

    # load default data & fix query set
    print("\nLoading default data (d=%d, L=%d) ..." % (D_DEF, L_DEF))
    S_def, W_def = load_data(mode="movielens_temporal", L=L_DEF, d=D_DEF)
    n_items = S_def.shape[0]
    rng = np.random.default_rng(SEED)
    queries = rng.choice(n_items, size=NUM_QUERIES, replace=False)
    print(f"  Query items: {queries.tolist()}")

    # ── 1. sweep k ──────────────────────────────────────────────────────
    print("\n[1/5] Sweeping k ...")
    for k_val in K_VALS:
        print(f"  k={k_val} ", end="", flush=True)
        m = evaluate(S_def, W_def, queries, k=k_val, tau=TAU_DEF,
                     c=C_DEF, L_win=L_DEF)
        m.update(sweep="k", param_value=k_val,
                 k=k_val, tau=TAU_DEF, L=L_DEF, d=D_DEF, c=C_DEF)
        all_rows.append(m)
        print(f" recall={m['avg_recall_hssa']:.2f}/{m['avg_recall_hpra']:.2f}"
              f"  prune={m['avg_pruning']:.4f}"
              f"  hssa={m['avg_hssa_time']:.4f}s")

    # ── 2. sweep tau ────────────────────────────────────────────────────
    print("\n[2/5] Sweeping tau ...")
    for tau_val in TAU_VALS:
        print(f"  tau={tau_val} ", end="", flush=True)
        m = evaluate(S_def, W_def, queries, k=K_DEF, tau=tau_val,
                     c=C_DEF, L_win=L_DEF)
        m.update(sweep="tau", param_value=tau_val,
                 k=K_DEF, tau=tau_val, L=L_DEF, d=D_DEF, c=C_DEF)
        all_rows.append(m)
        print(f" recall={m['avg_recall_hssa']:.2f}/{m['avg_recall_hpra']:.2f}"
              f"  prune={m['avg_pruning']:.4f}"
              f"  hssa={m['avg_hssa_time']:.4f}s")

    # ── 3. sweep c ──────────────────────────────────────────────────────
    print("\n[3/5] Sweeping c ...")
    for c_val in C_VALS:
        print(f"  c={c_val} ", end="", flush=True)
        m = evaluate(S_def, W_def, queries, k=K_DEF, tau=TAU_DEF,
                     c=c_val, L_win=L_DEF)
        m.update(sweep="c", param_value=c_val,
                 k=K_DEF, tau=TAU_DEF, L=L_DEF, d=D_DEF, c=c_val)
        all_rows.append(m)
        print(f" recall={m['avg_recall_hssa']:.2f}/{m['avg_recall_hpra']:.2f}"
              f"  prune={m['avg_pruning']:.4f}"
              f"  cands={m['avg_candidates']:.0f}"
              f"  hssa={m['avg_hssa_time']:.4f}s")

    # ── 4. sweep L ──────────────────────────────────────────────────────
    print("\n[4/5] Sweeping L ...")
    for L_val in L_VALS:
        print(f"  L={L_val} ", end="", flush=True)
        S_l, W_l = load_data(mode="movielens_temporal", L=L_val, d=D_DEF)
        m = evaluate(S_l, W_l, queries, k=K_DEF, tau=TAU_DEF,
                     c=C_DEF, L_win=L_val)
        m.update(sweep="L", param_value=L_val,
                 k=K_DEF, tau=TAU_DEF, L=L_val, d=D_DEF, c=C_DEF)
        all_rows.append(m)
        print(f" recall={m['avg_recall_hssa']:.2f}/{m['avg_recall_hpra']:.2f}"
              f"  prune={m['avg_pruning']:.4f}"
              f"  hssa={m['avg_hssa_time']:.4f}s")

    # ── 5. sweep d ──────────────────────────────────────────────────────
    print("\n[5/5] Sweeping d ...")
    for d_val in D_VALS:
        print(f"  d={d_val} ", end="", flush=True)
        # Re-generate static vectors with the new d, then rebuild temporal
        generate_movielens_static(d=d_val)
        S_d, W_d = generate_movielens_temporal(L=L_DEF)
        m = evaluate(S_d, W_d, queries, k=K_DEF, tau=TAU_DEF,
                     c=C_DEF, L_win=L_DEF)
        m.update(sweep="d", param_value=d_val,
                 k=K_DEF, tau=TAU_DEF, L=L_DEF, d=d_val, c=C_DEF)
        all_rows.append(m)
        print(f" recall={m['avg_recall_hssa']:.2f}/{m['avg_recall_hpra']:.2f}"
              f"  prune={m['avg_pruning']:.4f}"
              f"  hssa={m['avg_hssa_time']:.4f}s")

    # restore default d=32
    print("\n  Restoring default d=32 ...")
    generate_movielens_static(d=D_DEF)
    generate_movielens_temporal(L=L_DEF)

    # ── save CSVs ───────────────────────────────────────────────────────
    print("\nSaving CSVs ...")
    df_all = pd.DataFrame(all_rows)
    all_csv = os.path.join(OUT_CSV, "robustness_all.csv")
    df_all.to_csv(all_csv, index=False)
    print(f"  {all_csv}")

    for sweep_name in ["k", "tau", "c", "L", "d"]:
        df_s = df_all[df_all["sweep"] == sweep_name]
        p = os.path.join(OUT_CSV, f"robustness_{sweep_name}.csv")
        df_s.to_csv(p, index=False)
        print(f"  {p}")

    # ── final summary table ─────────────────────────────────────────────
    # Best setting = default (k=10, tau=0.6, c=1.5, L=5, d=32) from main eval
    best_row = df_all[
        (df_all["k"] == K_DEF) & (df_all["tau"] == TAU_DEF) &
        (df_all["c"] == C_DEF) & (df_all["L"] == L_DEF) &
        (df_all["d"] == D_DEF)
    ].iloc[0]

    summary_table = pd.DataFrame([{
        "Best_Method": "Hybrid + PRA (MinRank)",
        "k": K_DEF, "tau": TAU_DEF, "c": C_DEF, "L": L_DEF, "d": D_DEF,
        "Recall_HSSA": best_row["avg_recall_hssa"],
        "Recall_HPRA": best_row["avg_recall_hpra"],
        "Pruning_Ratio": best_row["avg_pruning"],
        "Avg_Candidates": best_row["avg_candidates"],
        "Avg_HSSA_Time": best_row["avg_hssa_time"],
        "Avg_HPRA_Time": best_row["avg_hpra_time"],
        "Avg_SSA_Time": best_row["avg_ssa_time"],
        "Avg_PRA_Time": best_row["avg_pra_time"],
        "Speedup_HSSA": best_row["avg_ssa_time"] / max(best_row["avg_hssa_time"], 1e-9),
        "Speedup_HPRA": best_row["avg_pra_time"] / max(best_row["avg_hpra_time"], 1e-9),
    }])
    final_csv = os.path.join(OUT_FINAL, "final_summary_table.csv")
    summary_table.to_csv(final_csv, index=False)
    print(f"  {final_csv}")

    # ── plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots ...")

    # runtime_vs_k
    df_k = df_all[df_all["sweep"] == "k"].sort_values("param_value")
    p = os.path.join(OUT_PLOTS, "runtime_vs_k.png")
    _plot_runtime(df_k, "param_value", "k (top-k)", p)
    print(f"  {p}")

    # runtime_vs_tau
    df_tau = df_all[df_all["sweep"] == "tau"].sort_values("param_value")
    p = os.path.join(OUT_PLOTS, "runtime_vs_tau.png")
    _plot_runtime(df_tau, "param_value", "τ (durability threshold)", p)
    print(f"  {p}")

    # runtime_vs_L
    df_L = df_all[df_all["sweep"] == "L"].sort_values("param_value")
    p = os.path.join(OUT_PLOTS, "runtime_vs_L.png")
    _plot_runtime(df_L, "param_value", "L (time windows)", p)
    print(f"  {p}")

    # runtime_vs_d
    df_d = df_all[df_all["sweep"] == "d"].sort_values("param_value")
    p = os.path.join(OUT_PLOTS, "runtime_vs_d.png")
    _plot_runtime(df_d, "param_value", "d (embedding dimension)", p)
    print(f"  {p}")

    # recall_vs_k
    p = os.path.join(OUT_PLOTS, "recall_vs_k.png")
    _plot_recall(df_k, "param_value", "k (top-k)", p)
    print(f"  {p}")

    # recall_vs_tau
    p = os.path.join(OUT_PLOTS, "recall_vs_tau.png")
    _plot_recall(df_tau, "param_value", "τ (durability threshold)", p)
    print(f"  {p}")

    # pruning_vs_c
    df_c = df_all[df_all["sweep"] == "c"].sort_values("param_value")
    p = os.path.join(OUT_PLOTS, "pruning_vs_c.png")
    _plot_pruning(df_c, "param_value", "c (relaxation factor)", p)
    print(f"  {p}")

    # ── summary to stdout ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  ROBUSTNESS SUMMARY")
    print("=" * 64)

    # check if recall is always 1.0
    all_recall_hssa = df_all["avg_recall_hssa"].values
    all_recall_hpra = df_all["avg_recall_hpra"].values
    perfect = (all_recall_hssa >= 0.9999).all() and (all_recall_hpra >= 0.9999).all()
    if perfect:
        print("  [OK]  Recall = 1.0000 across ALL parameter settings")
    else:
        low_hssa = df_all.loc[all_recall_hssa < 0.9999,
                              ["sweep", "param_value", "avg_recall_hssa"]]
        low_hpra = df_all.loc[all_recall_hpra < 0.9999,
                              ["sweep", "param_value", "avg_recall_hpra"]]
        print("  [!!]  Some settings have recall < 1.0:")
        if len(low_hssa) > 0:
            print(low_hssa.to_string(index=False))
        if len(low_hpra) > 0:
            print(low_hpra.to_string(index=False))

    print(f"\n  Best method  : Hybrid + PRA (MinRank)")
    print(f"  Best params  : k={K_DEF}, tau={TAU_DEF}, c={C_DEF}, L={L_DEF}, d={D_DEF}")
    print(f"  Recall HSSA  : {best_row['avg_recall_hssa']:.4f}")
    print(f"  Recall HPRA  : {best_row['avg_recall_hpra']:.4f}")
    print(f"  Pruning      : {best_row['avg_pruning']:.4f}")
    print(f"  Candidates   : {best_row['avg_candidates']:.0f}")

    sp_hssa = best_row["avg_ssa_time"] / max(best_row["avg_hssa_time"], 1e-9)
    sp_hpra = best_row["avg_pra_time"] / max(best_row["avg_hpra_time"], 1e-9)
    print(f"  Speedup HSSA : {sp_hssa:.1f}x")
    print(f"  Speedup HPRA : {sp_hpra:.1f}x")
    print("=" * 64)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
