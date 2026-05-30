import numpy as np

# ---------- RUN BUILDER ----------
def bool_to_runs(h: np.ndarray):
    runs = []
    L = len(h)
    i = 0
    while i < L:
        if not h[i]:
            i += 1
            continue
        start = i
        while i + 1 < L and h[i + 1]:
            i += 1
        end = i
        runs.append((start, end))
        i += 1
    return runs


# ---------- OVERLAP COUNTER ----------
def overlap_count(runs, tb, te_excl):
    if tb >= te_excl or not runs:
        return 0

    ql, qr = tb, te_excl - 1
    cnt = 0
    for s, e in runs:
        if e < ql:
            continue
        if s > qr:
            break
        left = max(s, ql)
        right = min(e, qr)
        if left <= right:
            cnt += (right - left + 1)
    return cnt


# ---------- ORIGINAL SSA (BIG MEMORY) ----------
def build_ssa_for_object(S, W, o_idx, k):
    n, d = S.shape
    m, L, d2 = W.shape
    assert d == d2

    runs_per_user = [None] * m
    h_all = np.zeros((m, L), dtype=bool)
    S_T = S.T

    for t in range(L):
        scores_t = W[:, t, :] @ S_T
        kth_t = np.partition(scores_t, k-1, axis=1)[:, k-1]
        score_o = scores_t[:, o_idx]
        h_all[:, t] = (score_o <= kth_t)

    for u in range(m):
        runs_per_user[u] = bool_to_runs(h_all[u])

    return runs_per_user


# ---------- CHUNKED SSA (LOW MEMORY) ----------
def build_ssa_for_object_chunked(S, W, o_idx, k, chunk_size=4000):
    n, d = S.shape
    m, L, d2 = W.shape
    assert d == d2

    h_all = np.zeros((m, L), dtype=np.uint8)

    for t in range(L):
        wt = W[:, t, :]  # (m,d)

        score_o = wt @ S[o_idx]  # (m,)

        best = np.full((m, k), np.inf, dtype=np.float64)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_scores = wt @ S[start:end].T

            merged = np.concatenate([best, chunk_scores], axis=1)
            best = np.partition(merged, k-1, axis=1)[:, :k]

        kth_t = np.max(best, axis=1)
        h_all[:, t] = (score_o <= kth_t).astype(np.uint8)

    runs_per_user = []
    for u in range(m):
        runs = []
        i = 0
        while i < L:
            if h_all[u, i] == 0:
                i += 1
                continue
            s = i
            while i + 1 < L and h_all[u, i + 1] == 1:
                i += 1
            e = i
            runs.append((s, e))
            i += 1
        runs_per_user.append(runs)

    return runs_per_user


# ---------- QUERY ----------
def drtopk_ssa_query(runs_per_user, tb, te, tau):
    L = te - tb
    if L <= 0:
        return []

    ans = []
    for u, runs in enumerate(runs_per_user):
        ones = overlap_count(runs, tb, te)
        if (ones / L) >= tau:
            ans.append(u)
    return ans