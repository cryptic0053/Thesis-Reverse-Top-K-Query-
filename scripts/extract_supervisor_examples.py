"""
Extract real numerical examples from MovieLens and Netflix processed arrays
for the supervisor-ready IEEE-style PDF.
Saves output to outputs/supervisor_examples.json
"""
import os, sys, math, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from utils.rank_table import build_rank_table

OUT_JSON = os.path.join(BASE_DIR, "outputs", "supervisor_examples.json")

# ── parameters ───────────────────────────────────────────────────────────────
K, TAU, C, L = 10, 0.6, 2.0, 5
TB, TE = 0, 5
REQUIRED = math.ceil(TAU * (TE - TB))   # 3
SEED = 42

# ── load processed arrays ────────────────────────────────────────────────────
ML  = os.path.join(BASE_DIR, "data", "processed", "movielens")
NF  = os.path.join(BASE_DIR, "data", "processed", "netflix")

print("Loading MovieLens …")
U_ml = np.load(os.path.join(ML, "U_user_vectors.npy"))
P_ml = np.load(os.path.join(ML, "P_item_vectors.npy"))
W_ml = np.load(os.path.join(ML, "W_temporal_vectors.npy"))

print("Loading Netflix …")
U_nf = np.load(os.path.join(NF, "U_user_vectors.npy"))
P_nf = np.load(os.path.join(NF, "P_item_vectors.npy"))
W_nf = np.load(os.path.join(NF, "W_temporal_vectors.npy"))

n_ml, d_ml = U_ml.shape
m_ml = P_ml.shape[0]
n_nf, d_nf = U_nf.shape
m_nf = P_nf.shape[0]

print(f"ML  U={U_ml.shape} P={P_ml.shape} W={W_ml.shape}")
print(f"NF  U={U_nf.shape} P={P_nf.shape} W={W_nf.shape}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Find non-empty MovieLens query and its qualifying user
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rng = np.random.default_rng(SEED)
query_items_ml = rng.choice(m_ml, size=50, replace=False).tolist()

ml_query = None
ml_user  = None
ml_temporal_rows = []
ml_static_rows = []
ml_rank_table_rows = []

for q_idx in query_items_ml:
    runs = build_ssa_for_object_chunked(P_ml, W_ml, q_idx, K, chunk_size=4000)
    result = drtopk_ssa_query(runs, TB, TE, TAU)
    if result:
        ml_query = int(q_idx)
        ml_user  = int(result[0])
        print(f"MovieLens: query={ml_query}  qualifying users={result}")
        break

if ml_query is None:
    print("No non-empty query found in first 50 – using item 0 for illustration")
    ml_query = int(query_items_ml[0])
    ml_user  = 0

# ── 1a. Per-window temporal example ──────────────────────────────────────────
u = ml_user
for t in range(L):
    wt    = W_ml[u, t, :]
    scores = wt @ P_ml.T          # (m_ml,)
    q_score = float(scores[ml_query])
    part_idx = m_ml - K
    kth_largest = float(np.partition(scores, part_idx)[part_idx])
    rank_t  = int(1 + np.sum(scores > q_score))
    passes  = bool(q_score >= kth_largest)
    ml_temporal_rows.append({
        "t": t,
        "q_score": round(q_score, 6),
        "kth_largest": round(kth_largest, 6),
        "rank": rank_t,
        "pass": passes
    })

successes = sum(r["pass"] for r in ml_temporal_rows)
dur       = successes / L

print(f"  User {ml_user}: successes={successes}/{L}  dur={dur:.2f}  required={REQUIRED}")
for r in ml_temporal_rows:
    print(f"  t={r['t']}  q_score={r['q_score']:.6f}  kth={r['kth_largest']:.6f}"
          f"  rank={r['rank']}  pass={r['pass']}")

# ── 1b. Static score / rank example for the same user ────────────────────────
user_scores_static = U_ml[ml_user] @ P_ml.T
q_score_static     = float(user_scores_static[ml_query])
exact_rank_static  = int(1 + np.sum(user_scores_static > q_score_static))

# Show 6 competing items (highest-scoring, excluding q itself)
top_idx  = np.argsort(-user_scores_static)
comp_idx = [int(i) for i in top_idx if i != ml_query][:5]
comp_rows = []
for ci in comp_idx:
    comp_rows.append({"item": ci, "score": round(float(user_scores_static[ci]), 6)})

ml_static_rows = {
    "user": ml_user,
    "query_item": ml_query,
    "q_score_static": round(q_score_static, 6),
    "exact_rank_static": exact_rank_static,
    "competing_items": comp_rows
}
print(f"\nStatic example – user={ml_user} query={ml_query}"
      f" score={q_score_static:.6f} rank={exact_rank_static}")
for cr in comp_rows:
    print(f"  item {cr['item']}  score={cr['score']:.6f}")

# ── 1c. Rank-table example (build for window t=0, show a few thresholds) ─────
T0, THR0 = build_rank_table(W_ml[:, 0, :], P_ml, TAU=500)
rng2 = np.random.default_rng(SEED)
SAMPLE_M = min(8000, m_ml)
samp_idx = rng2.choice(m_ml, SAMPLE_M, replace=False)
P_samp   = P_ml[samp_idx]
wt0_u    = W_ml[ml_user, 0, :]
s_u      = float(wt0_u @ P_ml[ml_query])     # score of query at t=0

# find bracket
idx_arr  = (THR0[ml_user] <= s_u).sum() - 1
idx_arr  = int(np.clip(idx_arr, 0, 498))
thr_lo   = float(THR0[ml_user, idx_arr])
thr_hi   = float(THR0[ml_user, idx_arr + 1])
rank_lo  = int(T0[ml_user, idx_arr])
rank_hi  = int(T0[ml_user, idx_arr + 1])
denom    = thr_hi - thr_lo if abs(thr_hi - thr_lo) > 1e-12 else 1.0
alpha    = (s_u - thr_lo) / denom
alpha    = float(np.clip(alpha, 0.0, 1.0))
r_hat    = rank_lo + alpha * (rank_hi - rank_lo)

# show 4 threshold rows around the bracket
show_idxs = [max(0, idx_arr - 1), idx_arr, idx_arr + 1, min(499, idx_arr + 2)]
thr_rows  = []
for ji in show_idxs:
    thr_rows.append({
        "j": int(ji),
        "theta_j": round(float(THR0[ml_user, ji]), 6),
        "T_j":     int(T0[ml_user, ji])
    })

ml_rank_table_rows = {
    "user": ml_user,
    "query_item": ml_query,
    "window": 0,
    "score_at_query": round(s_u, 6),
    "bracket_idx": idx_arr,
    "theta_lo": round(thr_lo, 6),
    "theta_hi": round(thr_hi, 6),
    "T_lo": rank_lo,
    "T_hi": rank_hi,
    "alpha": round(alpha, 6),
    "r_hat": round(r_hat, 2),
    "threshold_rows": thr_rows
}
print(f"\nRank-table – user={ml_user} window=0 score={s_u:.6f}"
      f" bracket=[{thr_lo:.6f}, {thr_hi:.6f}]"
      f" T=[{rank_lo},{rank_hi}] alpha={alpha:.4f} r_hat={r_hat:.2f}")

# ── 1d. DQR example for MovieLens (3 users: qualifying, another, and a third) ─
T_list_ml, THR_list_ml = [], []
for t in range(L):
    Tt, THRt = build_rank_table(W_ml[:, t, :], P_ml, TAU=500)
    T_list_ml.append(Tt)
    THR_list_ml.append(THRt)

# Pick 3 users: the qualifying user + 2 others
# Compute DQR for every user to find the non-qualifying ones near the boundary
q_vec_ml = P_ml[ml_query]
est_ranks_ml = np.zeros((n_ml, L), dtype=np.float32)
arange_n = np.arange(n_ml)
for ti in range(L):
    wt = W_ml[:, ti, :]
    s  = wt @ q_vec_ml
    T_t, THR_t = T_list_ml[ti], THR_list_ml[ti]
    idx = (THR_t <= s[:, None]).sum(axis=1) - 1
    idx = np.clip(idx, 0, 498)
    lower = T_t[arange_n, idx + 1].astype(np.float32)
    upper = T_t[arange_n, idx].astype(np.float32)
    t0v   = THR_t[arange_n, idx]
    t1v   = THR_t[arange_n, idx + 1]
    denom = (t1v - t0v)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    alpha_v = np.clip((s - t0v) / denom, 0.0, 1.0)
    est_ranks_ml[:, ti] = upper + alpha_v * (lower - upper)

dqr_ml = np.partition(est_ranks_ml, REQUIRED - 1, axis=1)[:, REQUIRED - 1]
NC_ml  = math.ceil(C * K)  # 20
cand_idx_ml = np.argpartition(dqr_ml, NC_ml - 1)[:NC_ml]

# show 3 users: the qualifying user + one just inside + one just outside
in_cands  = set(cand_idx_ml.tolist())
not_in    = [u for u in range(n_ml) if u not in in_cands and dqr_ml[u] > K][:1]
also_in   = [u for u in cand_idx_ml if u != ml_user][:1]

show_users = [ml_user] + also_in + not_in
dqr_rows_ml = []
for uu in show_users:
    ranks = [round(float(est_ranks_ml[uu, ti]), 1) for ti in range(L)]
    sorted_r = sorted(ranks)
    dqr_val  = round(float(dqr_ml[uu]), 1)
    is_cand  = uu in in_cands
    # SSA verify (only for candidates)
    ssa_qual = None
    if is_cand:
        runs_u = build_ssa_for_object_chunked(P_ml, W_ml[[uu]], ml_query, K, chunk_size=4000)
        res_u  = drtopk_ssa_query(runs_u, TB, TE, TAU)
        ssa_qual = len(res_u) > 0
    dqr_rows_ml.append({
        "user": uu,
        "est_ranks": ranks,
        "sorted_ranks": sorted_r,
        "DQR": dqr_val,
        "candidate": is_cand,
        "ssa_verified": ssa_qual
    })
    print(f"  ML DQR user={uu}  ranks={ranks}  DQR={dqr_val}"
          f"  cand={is_cand}  ssa={ssa_qual}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Netflix false-negative example
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n── Netflix: searching for false-negative example ──")

# Load CSV for quick check
NF_CSV = os.path.join(BASE_DIR, "outputs", "csv", "corrected_netflix_temporal",
                      "per_query_results.csv")
df_nf  = pd.read_csv(NF_CSV)
fn_rows = df_nf[df_nf["Hybrid_SSA_DurableRank_fn"] > 0]
print(f"  Queries with fn>0 in Netflix: {len(fn_rows)}")

nf_query   = None
nf_user    = None
nf_fn_user = None
nf_temporal_rows = []
nf_fn_temporal_rows = []
nf_dqr_rows = []

if len(fn_rows) > 0:
    nf_query = int(fn_rows.iloc[0]["query_item"])
    print(f"  Netflix false-negative query item: {nf_query}")

    # Build full SSA to find all qualifying users
    runs_nf = build_ssa_for_object_chunked(P_nf, W_nf, nf_query, K, chunk_size=4000)
    all_qual = drtopk_ssa_query(runs_nf, TB, TE, TAU)
    print(f"  Full SSA qualifying users: {all_qual[:10]}")

    # Compute DQR for all NF users
    q_vec_nf = P_nf[nf_query]
    est_ranks_nf = np.zeros((n_nf, L), dtype=np.float32)
    T_list_nf, THR_list_nf = [], []
    for ti in range(L):
        Tt, THRt = build_rank_table(W_nf[:, ti, :], P_nf, TAU=500)
        T_list_nf.append(Tt)
        THR_list_nf.append(THRt)

    arange_nf = np.arange(n_nf)
    for ti in range(L):
        wt   = W_nf[:, ti, :]
        s    = wt @ q_vec_nf
        T_t, THR_t = T_list_nf[ti], THR_list_nf[ti]
        idx  = (THR_t <= s[:, None]).sum(axis=1) - 1
        idx  = np.clip(idx, 0, 498)
        lower = T_t[arange_nf, idx + 1].astype(np.float32)
        upper = T_t[arange_nf, idx].astype(np.float32)
        t0v   = THR_t[arange_nf, idx]
        t1v   = THR_t[arange_nf, idx + 1]
        dv    = (t1v - t0v)
        dv    = np.where(np.abs(dv) < 1e-12, 1.0, dv)
        av    = np.clip((s - t0v) / dv, 0.0, 1.0)
        est_ranks_nf[:, ti] = upper + av * (lower - upper)

    dqr_nf = np.partition(est_ranks_nf, REQUIRED - 1, axis=1)[:, REQUIRED - 1]
    NC_nf  = math.ceil(C * K)   # 20
    cand_idx_nf = set(np.argpartition(dqr_nf, NC_nf - 1)[:NC_nf].tolist())

    # Find false-negative user (in all_qual but NOT in candidates)
    fn_users = [uu for uu in all_qual if uu not in cand_idx_nf]
    tp_users = [uu for uu in all_qual if uu in cand_idx_nf]
    print(f"  True positives: {tp_users}  False negatives: {fn_users}")

    if fn_users:
        nf_fn_user = int(fn_users[0])
        nf_user    = int(tp_users[0]) if tp_users else int(all_qual[0])

        # temporal rows for fn user
        uu = nf_fn_user
        for t in range(L):
            wt     = W_nf[uu, t, :]
            scores = wt @ P_nf.T
            q_sc   = float(scores[nf_query])
            pidx   = m_nf - K
            kth    = float(np.partition(scores, pidx)[pidx])
            rank_t = int(1 + np.sum(scores > q_sc))
            passes = bool(q_sc >= kth)
            nf_fn_temporal_rows.append({
                "t": t, "q_score": round(q_sc, 6),
                "kth_largest": round(kth, 6), "rank": rank_t, "pass": passes
            })
        fn_succ = sum(r["pass"] for r in nf_fn_temporal_rows)
        print(f"  FN user {uu}: successes={fn_succ}")

        # temporal for tp user (if exists)
        if tp_users:
            uu2 = tp_users[0]
            for t in range(L):
                wt     = W_nf[uu2, t, :]
                scores = wt @ P_nf.T
                q_sc   = float(scores[nf_query])
                pidx   = m_nf - K
                kth    = float(np.partition(scores, pidx)[pidx])
                rank_t = int(1 + np.sum(scores > q_sc))
                passes = bool(q_sc >= kth)
                nf_temporal_rows.append({
                    "t": t, "q_score": round(q_sc, 6),
                    "kth_largest": round(kth, 6), "rank": rank_t, "pass": passes
                })

        # DQR rows for 3 users: tp, fn, and one non-qualifying
        non_qual = [uu for uu in range(n_nf) if uu not in set(all_qual)][:1]
        show_nf  = (tp_users[:1] if tp_users else []) + fn_users[:1] + non_qual
        for uu in show_nf:
            ranks    = [round(float(est_ranks_nf[uu, ti]), 1) for ti in range(L)]
            sorted_r = sorted(ranks)
            dqr_val  = round(float(dqr_nf[uu]), 1)
            is_cand  = uu in cand_idx_nf
            is_qual  = uu in set(all_qual)
            nf_dqr_rows.append({
                "user": uu,
                "est_ranks": ranks,
                "sorted_ranks": sorted_r,
                "DQR": dqr_val,
                "candidate": is_cand,
                "truly_durable": is_qual,
                "note": "TP" if (is_qual and is_cand) else
                        ("FN" if (is_qual and not is_cand) else "non-qualifying")
            })
            print(f"  NF DQR user={uu}  DQR={dqr_val}  cand={is_cand}  qual={is_qual}  note={nf_dqr_rows[-1]['note']}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Assemble JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
out = {
    "params": {"K": K, "tau": TAU, "c": C, "L": L,
               "tb": TB, "te": TE, "required_successes": REQUIRED,
               "N_C": math.ceil(C * K)},
    "movielens": {
        "U_shape": list(U_ml.shape),
        "P_shape": list(P_ml.shape),
        "W_shape": list(W_ml.shape),
        "query_item": ml_query,
        "qualifying_user": ml_user,
        "temporal_rows": ml_temporal_rows,
        "successes": int(sum(r["pass"] for r in ml_temporal_rows)),
        "durability": round(sum(r["pass"] for r in ml_temporal_rows) / L, 4),
        "static_example": ml_static_rows,
        "rank_table_example": ml_rank_table_rows,
        "dqr_rows": dqr_rows_ml,
    },
    "netflix": {
        "U_shape": list(U_nf.shape),
        "P_shape": list(P_nf.shape),
        "W_shape": list(W_nf.shape),
        "query_item": nf_query,
        "tp_user": nf_user,
        "fn_user": nf_fn_user,
        "temporal_rows_tp": nf_temporal_rows,
        "fn_temporal_rows": nf_fn_temporal_rows,
        "dqr_rows": nf_dqr_rows,
    }
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {OUT_JSON}")
