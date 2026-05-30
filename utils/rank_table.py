import numpy as np

def build_rank_table(U, P, TAU=500):
    """
    Builds the Table T and Thresholds in-memory directly to save disk IO!
    """
    n, d = U.shape
    m, _ = P.shape
    SAMPLE_M = min(8000, m)
    rng = np.random.default_rng(42)
    P_sample = P[rng.choice(m, SAMPLE_M, replace=False)]
    
    scores = U @ P_sample.T
    fmax, fmin = scores.max(axis=1), scores.min(axis=1)
    
    j = np.arange(TAU, dtype=np.float32)
    T_thr = fmin[:, None] + (fmax - fmin)[:, None] * (j[None, :] / (TAU - 1))
    
    T = np.empty((n, TAU), dtype=np.int32)
    for col in range(TAU):
        thr = T_thr[:, col][:, None]
        cnt = (scores > thr).sum(axis=1)
        T[:, col] = (1 + (m * cnt / SAMPLE_M)).astype(np.int32)
        
    return T, T_thr.astype(np.float32)
