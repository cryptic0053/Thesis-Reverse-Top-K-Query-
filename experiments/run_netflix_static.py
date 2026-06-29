"""
run_netflix_static.py
=====================
Netflix Prize static validation: approximate reverse k-ranks.

Uses a controlled subset (top 5000 users, top 3000 items by rating count)
to keep runtime practical.
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
from baselines.approximate.reverse_k_rank_bruteforce import reverse_k_ranks
from baselines.approximate.reverse_k_rank_approx import approx_reverse_k_ranks

OUT_CSV   = os.path.join(BASE_DIR, "outputs", "csv",   "netflix_static")
OUT_PLOTS = os.path.join(BASE_DIR, "outputs", "plots", "netflix_static")
os.makedirs(OUT_CSV,   exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

D           = 32
C           = 1.5
DEFAULT_K   = 10
NUM_QUERIES = 5
SEED        = 42

rng = np.random.default_rng(SEED)


def main():
    print("=" * 64)
    print("  Netflix Static Validation -- Approximate Reverse k-Ranks")
    print("=" * 64)

    # ---- 1. Data ----
    print("\n[1/5] Loading Netflix static data ...")
    U, P = load_data(mode="netflix_static", d=D, max_users=5000, max_items=3000)
    n_users, d_f = U.shape
    n_items, _   = P.shape

    print(f"  U (users)  : {U.shape}")
    print(f"  P (items)  : {P.shape}")
    print(f"  Dimension  : {d_f}")

    # ---- 2. Rank table ----
    print("\n[2/5] Building rank table ...")
    t0 = time.time()
    T, THR = build_table(U, P, out_dir=os.path.join(BASE_DIR, "outputs"),
                         tau=500, sample_m=min(4000, n_items), seed=SEED)
    build_time = time.time() - t0
    print(f"  Build time : {build_time:.4f}s")

    # ---- 3. Queries ----
    queries = rng.choice(n_items, size=NUM_QUERIES, replace=False)

    print(f"\n[3/5] Running reverse {DEFAULT_K}-ranks for {NUM_QUERIES} queries ...")
    bf_times   = []
    apx_times  = []
    exact_res  = {}
    approx_res = {}

    for q_idx in queries:
        q = int(q_idx)
        ts = time.time()
        ubf, _ = reverse_k_ranks(q, DEFAULT_K, U, P, batch=2048)
        bf_times.append(time.time() - ts)
        exact_res[q] = set(ubf)

        ts = time.time()
        uapx, _ = approx_reverse_k_ranks(q, DEFAULT_K, C, U, P, T, THR, 500)
        apx_times.append(time.time() - ts)
        approx_res[q] = set(uapx)

    avg_bf  = np.mean(bf_times)
    avg_apx = np.mean(apx_times)
    speedup = avg_bf / avg_apx if avg_apx > 0 else float("inf")

    recalls = []
    for q in queries:
        q = int(q)
        inter = len(exact_res[q] & approx_res[q])
        recalls.append(inter / DEFAULT_K)
    avg_recall = np.mean(recalls)

    print(f"  Brute-force avg : {avg_bf:.4f}s")
    print(f"  Approximate avg : {avg_apx:.4f}s")
    print(f"  Speedup         : {speedup:.1f}x")
    print(f"  Recall (k={DEFAULT_K}) : {avg_recall:.4f}")

    # ---- 4. Recall vs k ----
    print("\n[4/5] Evaluating recall vs k ...")
    k_vals = [5, 10, 20, 50, 100]
    recall_vs_k = []
    for k in k_vals:
        tmp = []
        for q_idx in queries:
            q = int(q_idx)
            ubf, _ = reverse_k_ranks(q, k, U, P, batch=2048)
            uapx, _ = approx_reverse_k_ranks(q, k, C, U, P, T, THR, 500)
            tmp.append(len(set(ubf) & set(uapx)) / k)
        recall_vs_k.append(np.mean(tmp))
        print(f"  k={k:3d}  recall={recall_vs_k[-1]:.4f}")

    # ---- 5. Save ----
    print("\n[5/5] Saving CSVs and plots ...")

    # CSVs
    pd.DataFrame({
        "Algorithm": ["Brute-Force", "Approximate"],
        "Average_Time_sec": [avg_bf, avg_apx],
    }).to_csv(os.path.join(OUT_CSV, "runtime_compare.csv"), index=False)

    pd.DataFrame({
        "k": k_vals,
        "Recall": recall_vs_k,
    }).to_csv(os.path.join(OUT_CSV, "accuracy_compare.csv"), index=False)

    # runtime bar
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Brute-Force", "Approximate"], [avg_bf, avg_apx],
                  color=["#e74c3c", "#3498db"], edgecolor="#333", linewidth=0.6)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title(f"Netflix Static -- Runtime (k={DEFAULT_K})", fontsize=13)
    for bar, val in zip(bars, [avg_bf, avg_apx]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.4f}s", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    runtime_png = os.path.join(OUT_PLOTS, "runtime_bar.png")
    fig.savefig(runtime_png, dpi=150)
    plt.close(fig)

    # recall vs k
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_vals, recall_vs_k, marker="o", color="#9b59b6", linewidth=2)
    ax.set_xlabel("k (top-k)", fontsize=12)
    ax.set_ylabel("Recall (overlap)", fontsize=12)
    ax.set_title("Netflix Static -- Recall vs k", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    recall_png = os.path.join(OUT_PLOTS, "recall_vs_k.png")
    fig.savefig(recall_png, dpi=150)
    plt.close(fig)

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("  NETFLIX STATIC RESULTS")
    print("=" * 64)
    print(f"  U shape               : {U.shape}")
    print(f"  P shape               : {P.shape}")
    print(f"  Factorisation dim (d) : {d_f}")
    print(f"  Dataset               : Netflix subset (top 5000 users, top 3000 items)")
    print(f"  Rank table build time : {build_time:.4f}s")
    print(f"  Brute-force avg       : {avg_bf:.4f}s")
    print(f"  Approximate avg       : {avg_apx:.4f}s")
    print(f"  Speedup               : {speedup:.1f}x")
    print(f"  Recall (k={DEFAULT_K})       : {avg_recall:.4f}")
    print(f"\n  CSV:  {os.path.join(OUT_CSV, 'runtime_compare.csv')}")
    print(f"  CSV:  {os.path.join(OUT_CSV, 'accuracy_compare.csv')}")
    print(f"  Plot: {runtime_png}")
    print(f"  Plot: {recall_png}")
    print("=" * 64)

    # report file
    with open(os.path.join(BASE_DIR, "outputs", "netflix_static_report.txt"), "w") as f:
        f.write(f"Shape of U: {U.shape}\n")
        f.write(f"Shape of P: {P.shape}\n")
        f.write(f"Factorization dimension: {d_f}\n")
        f.write(f"Dataset: Netflix subset (top 5000 users, top 3000 items)\n")
        f.write(f"Rank table build time: {build_time:.4f}s\n")
        f.write(f"Brute-force average runtime: {avg_bf:.4f}s\n")
        f.write(f"Approximate average runtime: {avg_apx:.4f}s\n")
        f.write(f"Speedup: {speedup:.1f}x\n")
        f.write(f"Recall / overlap at k={DEFAULT_K}: {avg_recall:.4f}\n")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
