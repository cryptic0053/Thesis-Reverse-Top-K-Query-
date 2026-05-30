import os
import numpy as np

def build_table(U, P, out_dir, tau=500, sample_m=8000, seed=42):
    n, d = U.shape
    m, _ = P.shape

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(m, size=min(sample_m, m), replace=False)
    P_sample = P[sample_idx]  # (SAMPLE_M, d)

    print("U:", U.shape, "P:", P.shape, "P_sample:", P_sample.shape)

    # compute score range per user using sampled items
    scores_sample = U @ P_sample.T  # (n, SAMPLE_M)

    fmax = scores_sample.max(axis=1)  # (n,)
    fmin = scores_sample.min(axis=1)  # (n,)

    # create thresholds per user (n x TAU)
    j = np.arange(tau, dtype=np.float32)
    T_thr = fmin[:, None] + (fmax - fmin)[:, None] * (j[None, :] / (tau - 1))

    T = np.empty((n, tau), dtype=np.int32)

    print("Building rank table...")
    for col in range(tau):
        thr = T_thr[:, col][:, None]                  # (n,1)
        cnt = (scores_sample > thr).sum(axis=1)       # (n,)
        est = 1 + (m * cnt / scores_sample.shape[1])  # scale up to full m
        T[:, col] = est.astype(np.int32)

    return T, T_thr.astype(np.float32)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUT = os.path.join(BASE_DIR, "outputs")
    
    U_path = os.path.join(OUT, "U_user_vectors.npy")
    P_path = os.path.join(OUT, "P_item_vectors.npy")
    
    U = np.load(U_path)  # (n, d)
    P = np.load(P_path)  # (m, d)
    
    TAU = 500
    SAMPLE_M = 8000
    SEED = 42
    
    T, T_thr = build_table(U, P, out_dir=OUT, tau=TAU, sample_m=SAMPLE_M, seed=SEED)
    
    np.save(os.path.join(OUT, f"rank_table_T_tau{TAU}.npy"), T)
    np.save(os.path.join(OUT, f"thresholds_tau{TAU}.npy"), T_thr)
    
    print("Saved:")
    print(" -", os.path.join(OUT, f"rank_table_T_tau{TAU}.npy"), T.shape)
    print(" -", os.path.join(OUT, f"thresholds_tau{TAU}.npy"), T_thr.shape)
