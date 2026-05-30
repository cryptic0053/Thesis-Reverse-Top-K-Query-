import os, time
import numpy as np

def reverse_k_ranks(q_idx, k, U, P, batch=2048):
    q = P[q_idx]
    uq = U @ q
    n_users = U.shape[0]
    n_items = P.shape[0]

    ranks = np.ones(n_users, dtype=np.int32)

    for s in range(0, n_items, batch):
        e = min(s + batch, n_items)
        scores = U @ P[s:e].T
        ranks += (scores > uq[:, None]).sum(axis=1)

    topk = np.argpartition(ranks, k)[:k]
    topk = topk[np.argsort(ranks[topk])]
    return topk, ranks[topk]

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(BASE_DIR, "outputs")
    
    U = np.load(os.path.join(OUT, "U_user_vectors.npy"))
    P = np.load(os.path.join(OUT, "P_item_vectors.npy"))
    
    k = 200
    num_queries = 5
    BATCH = 2048
    
    n_users, d = U.shape
    n_items, _ = P.shape
    
    print("Users:", n_users, "Items:", n_items)
    
    rng = np.random.default_rng(42)
    query_items = rng.choice(n_items, size=num_queries, replace=False)
    
    t0 = time.time()
    for qi in query_items:
        users, ranks = reverse_k_ranks(int(qi), k, U, P, batch=BATCH)
        print(f"Query item {qi}: best rank={ranks[0]}, worst rank in top-k={ranks[-1]}")
    t1 = time.time()
    
    print(f"Done. Time = {t1 - t0:.2f}s")
    
    with open(os.path.join(OUT, "baseline_bruteforce_log.txt"), "w") as f:
        f.write(f"U shape: {U.shape}\n")
        f.write(f"P shape: {P.shape}\n")
        f.write(f"k={k}, num_queries={num_queries}, time={t1-t0:.2f}s\n")
