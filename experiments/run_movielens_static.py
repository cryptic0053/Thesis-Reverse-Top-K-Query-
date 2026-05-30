import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data.common_data import load_data
from baselines.approximate.build_rank_table import build_table
from baselines.approximate.reverse_k_rank_bruteforce import reverse_k_ranks
from baselines.approximate.reverse_k_rank_approx import approx_reverse_k_ranks

OUT_CSV = os.path.join(BASE_DIR, "outputs", "csv", "movielens_static")
OUT_PLOTS = os.path.join(BASE_DIR, "outputs", "plots", "movielens_static")
os.makedirs(OUT_CSV, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

# Configurations
D = 32
C = 1.5
DEFAULT_K = 10
NUM_QUERIES = 5
SEED = 42

rng = np.random.default_rng(SEED)

def main():
    print("--- 1. Loading / Generating MovieLens Static Data ---")
    # Returns generated normalized U and P
    U, P = load_data(mode="movielens_static", d=D)
    
    n_users, d_f = U.shape
    n_items, _ = P.shape
    
    print(f"User Vector Space (U): {U.shape}")
    print(f"Item Vector Space (P): {P.shape}")
    print(f"Factorization Dimension: {d_f}")
    
    print("\n--- 2. Building Rank Tables ---")
    t0 = time.time()
    T, THR = build_table(U, P, out_dir=os.path.join(BASE_DIR, "outputs"), tau=500, sample_m=4000, seed=SEED)
    t1 = time.time()
    build_time = t1 - t0
    print(f"Rank Table Build Time: {build_time:.4f}s")
    
    queries = rng.choice(n_items, size=NUM_QUERIES, replace=False)
    
    print(f"\n--- 3. Running Reverse {DEFAULT_K}-Ranks for {NUM_QUERIES} queries ---")
    
    bf_times = []
    approx_times = []
    
    exact_results = {}
    approx_results = {}
    
    for q_idx in queries:
        # Brute-force
        ts = time.time()
        users_bf, _ = reverse_k_ranks(int(q_idx), DEFAULT_K, U, P, batch=2048)
        bf_times.append(time.time() - ts)
        exact_results[q_idx] = set(users_bf)
        
        # Approximate
        ts = time.time()
        users_apx, _ = approx_reverse_k_ranks(int(q_idx), DEFAULT_K, C, U, P, T, THR, 500)
        approx_times.append(time.time() - ts)
        approx_results[q_idx] = set(users_apx)
        
    avg_bf = np.mean(bf_times)
    avg_apx = np.mean(approx_times)
    
    # Calculate recall / overlap
    recalls = []
    for q in queries:
        intersection = len(exact_results[q].intersection(approx_results[q]))
        recalls.append(intersection / DEFAULT_K)
    
    avg_recall = np.mean(recalls)
    
    print(f"Average Brute-Force Runtime: {avg_bf:.4f}s")
    print(f"Average Approximate Runtime: {avg_apx:.4f}s")
    print(f"Average Recall (k={DEFAULT_K}): {avg_recall:.4f}")
    
    # Save runtime CSV
    runtime_df = pd.DataFrame({
        "Algorithm": ["Brute-Force", "Approximate"],
        "Average_Time_sec": [avg_bf, avg_apx]
    })
    runtime_csv_path = os.path.join(OUT_CSV, "runtime_compare.csv")
    runtime_df.to_csv(runtime_csv_path, index=False)
    
    # Run tests for different K
    print("\n--- 4. Evaluating Recall vs K ---")
    k_vals = [5, 10, 20, 50, 100]
    recall_vs_k = []
    
    for k in k_vals:
        tmp_recalls = []
        for q_idx in queries:
            ubf, _ = reverse_k_ranks(int(q_idx), k, U, P, batch=2048)
            uapx, _ = approx_reverse_k_ranks(int(q_idx), k, C, U, P, T, THR, 500)
            tmp_recalls.append(len(set(ubf).intersection(set(uapx))) / k)
        recall_vs_k.append(np.mean(tmp_recalls))
        
    acc_df = pd.DataFrame({
        "k": k_vals,
        "Recall": recall_vs_k
    })
    acc_csv_path = os.path.join(OUT_CSV, "accuracy_compare.csv")
    acc_df.to_csv(acc_csv_path, index=False)
    
    # Plots
    # 1. Runtime bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(["Brute-Force", "Approximate"], [avg_bf, avg_apx], color=['#ff9999','#66b3ff'])
    plt.ylabel("Time (seconds)")
    plt.title(f"Runtime Comparison (k={DEFAULT_K})")
    for i, v in enumerate([avg_bf, avg_apx]):
        plt.text(i, v + 0.001, f"{v:.4f}s", ha='center')
    plt.tight_layout()
    runtime_plot_path = os.path.join(OUT_PLOTS, "runtime_bar.png")
    plt.savefig(runtime_plot_path)
    plt.close()
    
    # 2. Recall vs K plot
    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, recall_vs_k, marker='o', linestyle='-', color='purple')
    plt.xlabel("k (Top-k)")
    plt.ylabel("Recall (Overlap %)")
    plt.title("Recall vs K")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    recall_plot_path = os.path.join(OUT_PLOTS, "recall_vs_k.png")
    plt.savefig(recall_plot_path)
    plt.close()
    
    print("\n--- Summary ---")
    print("Files successfully generated:")
    print(f" - Mappings & Vectors: {os.path.join(BASE_DIR, 'data', 'processed', 'movielens')}")
    print(f" - CSV outputs: {OUT_CSV}")
    print(f" - Plot outputs: {OUT_PLOTS}")
    
    # Create an artifact file for final stats viewing
    with open(os.path.join(BASE_DIR, "outputs", "movielens_static_report.txt"), "w") as f:
        f.write(f"Shape of U: {U.shape}\n")
        f.write(f"Shape of P: {P.shape}\n")
        f.write(f"Factorization dimension: {d_f}\n")
        f.write(f"Rank table build time: {build_time:.4f}s\n")
        f.write(f"Brute-force average runtime: {avg_bf:.4f}s\n")
        f.write(f"Approximate average runtime: {avg_apx:.4f}s\n")
        f.write(f"Recall / overlap at k={DEFAULT_K}: {avg_recall:.4f}\n")
        f.write(f"Exact CSV paths:\n")
        f.write(f"  - {runtime_csv_path}\n")
        f.write(f"  - {acc_csv_path}\n")
        f.write(f"Exact plot paths:\n")
        f.write(f"  - {runtime_plot_path}\n")
        f.write(f"  - {recall_plot_path}\n")

if __name__ == "__main__":
    main()
