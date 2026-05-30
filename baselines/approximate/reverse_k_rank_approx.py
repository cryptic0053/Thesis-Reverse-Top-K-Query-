import os, time
import numpy as np

def approx_reverse_k_ranks(q_idx, k, c, U, P, T, THR, tau):
    q = P[q_idx]
    s = U @ q   # (n_users,) scores u·q
    n_users = U.shape[0]

    # For each user i, find j such that THR[i,j] <= s[i] <= THR[i,j+1]
    idx = (THR <= s[:, None]).sum(axis=1) - 1
    idx = np.clip(idx, 0, tau - 2)

    lower = T[np.arange(n_users), idx + 1].astype(np.float32)
    upper = T[np.arange(n_users), idx].astype(np.float32)

    t0_vals = THR[np.arange(n_users), idx]
    t1_vals = THR[np.arange(n_users), idx + 1]
    denom = (t1_vals - t0_vals)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)

    alpha = (s - t0_vals) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    est_rank = upper + alpha * (lower - upper)  # between upper and lower

    topk = np.argpartition(est_rank, k)[:k]
    topk = topk[np.argsort(est_rank[topk])]
    return topk, est_rank[topk]

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(BASE_DIR, "outputs")
    
    U = np.load(os.path.join(OUT, "U_user_vectors.npy"))  # (n, d)
    P = np.load(os.path.join(OUT, "P_item_vectors.npy"))  # (m, d)
    
    TAU = 500
    
    T = np.load(os.path.join(OUT, f"rank_table_T_tau{TAU}.npy"))          # (n, TAU) int
    THR = np.load(os.path.join(OUT, f"thresholds_tau{TAU}.npy"))          # (n, TAU) float
    
    n_users, d = U.shape
    n_items, _ = P.shape
    
    k = 200
    c = 1.5                 
    num_queries = 5
    SEED = 42
    
    rng = np.random.default_rng(SEED)
    query_items = rng.choice(n_items, size=num_queries, replace=False)
    
    t0 = time.time()
    for qi in query_items:
        users, est = approx_reverse_k_ranks(int(qi), k, c, U, P, T, THR, TAU)
        print(f"Query item {qi}: est best={est[0]:.2f}, est worst in topk={est[-1]:.2f}")
    t1 = time.time()
    
    print(f"Approx done. {num_queries} queries, k={k}, tau={TAU}, time={t1-t0:.2f}s")
