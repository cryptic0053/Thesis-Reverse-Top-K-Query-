"""Extract Netflix false-negative example."""
import os, sys, math, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from utils.rank_table import build_rank_table

K, TAU, C, L = 10, 0.6, 2.0, 5
TB, TE = 0, 5
REQUIRED = math.ceil(TAU * (TE - TB))   # 3

NF  = os.path.join(BASE_DIR, "data", "processed", "netflix")
P_nf = np.load(os.path.join(NF, "P_item_vectors.npy"))
W_nf = np.load(os.path.join(NF, "W_temporal_vectors.npy"))
U_nf = np.load(os.path.join(NF, "U_user_vectors.npy"))
n_nf, d_nf = U_nf.shape
m_nf = P_nf.shape[0]
print(f"NF U={U_nf.shape} P={P_nf.shape} W={W_nf.shape}")

NF_CSV = os.path.join(BASE_DIR, "outputs", "csv", "corrected_netflix_temporal",
                      "per_query_results.csv")
df_nf = pd.read_csv(NF_CSV)
fn_rows = df_nf[df_nf["Hybrid_SSA_DurableRank_fn"] > 0]
print(f"Queries with fn>0: {len(fn_rows)}")
print(fn_rows[["query_item","Full_SSA_count","Hybrid_SSA_DurableRank_tp",
               "Hybrid_SSA_DurableRank_fp","Hybrid_SSA_DurableRank_fn"]].head(5).to_string())

nf_query = int(fn_rows.iloc[0]["query_item"])
print(f"\nUsing Netflix query item: {nf_query}")

# Full SSA
runs_nf = build_ssa_for_object_chunked(P_nf, W_nf, nf_query, K, chunk_size=4000)
all_qual = drtopk_ssa_query(runs_nf, TB, TE, TAU)
print(f"Full SSA qualifying users: {all_qual}")

# Build DQR for all NF users
q_vec_nf = P_nf[nf_query]
est_ranks_nf = np.zeros((n_nf, L), dtype=np.float32)
arange_nf = np.arange(n_nf)

for ti in range(L):
    print(f"  Building rank table t={ti}...")
    Tt, THRt = build_rank_table(W_nf[:, ti, :], P_nf, TAU=500)
    wt = W_nf[:, ti, :]
    s  = wt @ q_vec_nf
    idx  = (THRt <= s[:, None]).sum(axis=1) - 1
    idx  = np.clip(idx, 0, 498)
    lower = Tt[arange_nf, idx + 1].astype(np.float32)
    upper = Tt[arange_nf, idx].astype(np.float32)
    t0v   = THRt[arange_nf, idx]
    t1v   = THRt[arange_nf, idx + 1]
    dv    = (t1v - t0v)
    dv    = np.where(np.abs(dv) < 1e-12, 1.0, dv)
    av    = np.clip((s - t0v) / dv, 0.0, 1.0)
    est_ranks_nf[:, ti] = upper + av * (lower - upper)

dqr_nf = np.partition(est_ranks_nf, REQUIRED - 1, axis=1)[:, REQUIRED - 1]
NC_nf  = math.ceil(C * K)   # 20
cand_idx_nf = set(np.argpartition(dqr_nf, NC_nf - 1)[:NC_nf].tolist())

fn_users = [uu for uu in all_qual if uu not in cand_idx_nf]
tp_users = [uu for uu in all_qual if uu in cand_idx_nf]
print(f"TP users: {tp_users}  FN users: {fn_users}")

result = {
    "nf_query": nf_query,
    "qualifying_users": all_qual,
    "tp_users": tp_users,
    "fn_users": fn_users,
    "dqr_rows": []
}

# build temporal rows for fn and tp users
def get_temporal_rows(uu):
    rows = []
    for t in range(L):
        wt     = W_nf[uu, t, :]
        scores = wt @ P_nf.T
        q_sc   = float(scores[nf_query])
        pidx   = m_nf - K
        kth    = float(np.partition(scores, pidx)[pidx])
        rank_t = int(1 + np.sum(scores > q_sc))
        passes = bool(q_sc >= kth)
        rows.append({"t": t, "q_score": round(q_sc, 6),
                     "kth_largest": round(kth, 6), "rank": rank_t, "pass": passes})
    return rows

nf_temporal_rows_tp = []
nf_fn_temporal_rows = []

if tp_users:
    uu2 = tp_users[0]
    nf_temporal_rows_tp = get_temporal_rows(uu2)
    succ = sum(r["pass"] for r in nf_temporal_rows_tp)
    print(f"TP user {uu2}: successes={succ}/5")
    for r in nf_temporal_rows_tp:
        print(f"  t={r['t']} q_score={r['q_score']:.6f} kth={r['kth_largest']:.6f} rank={r['rank']} pass={r['pass']}")

if fn_users:
    uu = fn_users[0]
    nf_fn_temporal_rows = get_temporal_rows(uu)
    succ = sum(r["pass"] for r in nf_fn_temporal_rows)
    print(f"FN user {uu}: successes={succ}/5")
    for r in nf_fn_temporal_rows:
        print(f"  t={r['t']} q_score={r['q_score']:.6f} kth={r['kth_largest']:.6f} rank={r['rank']} pass={r['pass']}")

# DQR rows for 3 users
non_qual_u = [uu for uu in range(n_nf) if uu not in set(all_qual) and dqr_nf[uu] > K][:1]
show_nf = (tp_users[:1] if tp_users else []) + (fn_users[:1] if fn_users else []) + non_qual_u

dqr_rows = []
for uu in show_nf:
    ranks    = [round(float(est_ranks_nf[uu, ti]), 1) for ti in range(L)]
    sorted_r = sorted(ranks)
    dqr_val  = round(float(dqr_nf[uu]), 1)
    is_cand  = uu in cand_idx_nf
    is_qual  = uu in set(all_qual)
    note     = ("TP" if (is_qual and is_cand) else
                ("FN" if (is_qual and not is_cand) else "non-qualifying"))
    dqr_rows.append({
        "user": uu,
        "est_ranks": ranks,
        "sorted_ranks": sorted_r,
        "DQR": dqr_val,
        "candidate": is_cand,
        "truly_durable": is_qual,
        "note": note
    })
    print(f"  NF DQR user={uu} ranks={ranks} DQR={dqr_val} cand={is_cand} qual={is_qual} note={note}")

result["dqr_rows"] = dqr_rows
result["temporal_rows_tp"] = nf_temporal_rows_tp
result["fn_temporal_rows"] = nf_fn_temporal_rows

OUT = os.path.join(BASE_DIR, "outputs", "netflix_supervisor_example.json")
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved -> {OUT}")
