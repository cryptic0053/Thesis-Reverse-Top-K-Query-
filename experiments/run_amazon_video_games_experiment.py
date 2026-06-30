"""
Amazon Video Games – HyDART-RQ Temporal Reverse Top-k Experiment
================================================================
Dataset  : Amazon Reviews 2023 Video Games (medium subset, 5-core)
Source   : data/raw/amazon_video_games/amazon_video_games_medium.csv
Outputs  : outputs/amazon_video_games/

Evaluated methods
  Full_SSA            : brute-force SSA durability (ground truth)
  Full_PRA            : brute-force PRA durability (ground truth cross-check)
  Hybrid_SSA_GlobalAvg: candidate filter via global-avg rank table + SSA verify
  Hybrid_PRA_GlobalAvg: candidate filter via global-avg rank table + PRA verify
  Hybrid_SSA_UnionWin : union-window rank filter + SSA verify
  Hybrid_PRA_UnionWin : union-window rank filter + PRA verify
  Hybrid_SSA_DurableRank (HyDART-RQ): durable-quantile filter + SSA verify
  Hybrid_PRA_DurableRank (HyDART-RQ): durable-quantile filter + PRA verify
  Hybrid_SSA_LegacyMinRank: legacy min-window rank filter + SSA verify
  Hybrid_PRA_LegacyMinRank: legacy min-window rank filter + PRA verify
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query
from hybrid.hybrid_sdr_topk import (
    hybrid_ssa_query, hybrid_pra_query,
    hybrid_ssa_query_union, hybrid_pra_query_union,
    hybrid_ssa_query_minrank, hybrid_pra_query_minrank,
    hybrid_ssa_query_durable_rank, hybrid_pra_query_durable_rank,
)
from utils.rank_table import build_rank_table

# ---------------------------------------------------------------------------
# Experiment hyper-parameters
# ---------------------------------------------------------------------------
D           = 32     # latent dimension
K           = 10     # top-k
TAU         = 0.6    # durability threshold (fraction of windows)
C           = 1.5    # candidate expansion factor
NUM_QUERIES = 100    # number of query items
L           = 5      # time windows
SEED        = 42
CHUNK       = 4000   # SSA chunk size

CSV_PATH  = os.path.join(BASE_DIR, "data", "raw", "amazon_video_games",
                         "amazon_video_games_medium.csv")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed", "amazon_video_games")
OUT_DIR   = os.path.join(BASE_DIR, "outputs", "amazon_video_games")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)


# ---------------------------------------------------------------------------
# Metric helper (same as other experiment scripts)
# ---------------------------------------------------------------------------
def calc_metrics(baseline_set, approx_set):
    tp  = len(baseline_set & approx_set)
    fp  = len(approx_set - baseline_set)
    fn  = len(baseline_set - approx_set)
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    exact      = baseline_set == approx_set
    return tp, fp, fn, precision, recall, f1, exact


# ---------------------------------------------------------------------------
# Static snapshot reverse top-k (brute force, single time window)
# ---------------------------------------------------------------------------
def static_reverse_topk_brute(S, U_snap, q_idx, k, chunk_size=4000):
    """Find all users who have item q_idx in their top-k at a single snapshot."""
    n_items = S.shape[0]
    n_users = U_snap.shape[0]
    score_q = U_snap @ S[q_idx]               # (n_users,)
    best    = np.full((n_users, k), -np.inf, dtype=np.float32)
    for start in range(0, n_items, chunk_size):
        end     = min(start + chunk_size, n_items)
        chunk   = U_snap @ S[start:end].T
        merged  = np.concatenate([best, chunk], axis=1)
        best    = np.partition(merged, merged.shape[1] - k, axis=1)[:, -k:]
    kth        = np.min(best, axis=1)
    return np.where(score_q >= kth)[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 64
    print(sep)
    print("  HyDART-RQ  |  Amazon Video Games Experiment")
    print(sep)
    print(f"  D={D}, K={K}, TAU={TAU}, C={C}, L={L}, queries={NUM_QUERIES}")
    print(sep)

    exp_start = time.time()

    # ── [1/8] Load dataset ────────────────────────────────────────────────
    print(f"\n[1/8] Loading Amazon Video Games dataset")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at:\n       {CSV_PATH}")
        print("Fix: run  python scripts/download_amazon_video_games_medium.py")
        sys.exit(1)

    df_raw = pd.read_csv(CSV_PATH)

    required_cols = ["user_id", "item_id", "rating", "timestamp"]
    missing_cols  = [c for c in required_cols if c not in df_raw.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        print(f"       Found: {list(df_raw.columns)}")
        sys.exit(1)

    n_rows_raw = len(df_raw)
    print(f"  Rows    : {n_rows_raw:,}")
    print(f"  Users   : {df_raw['user_id'].nunique():,}")
    print(f"  Items   : {df_raw['item_id'].nunique():,}")
    print(f"  Ratings : {df_raw['rating'].min()} – {df_raw['rating'].max()}")
    ts_min_ms = df_raw["timestamp"].min()
    ts_max_ms = df_raw["timestamp"].max()
    yr_min = datetime.fromtimestamp(ts_min_ms / 1000).year
    yr_max = datetime.fromtimestamp(ts_max_ms / 1000).year
    print(f"  Time    : {yr_min} – {yr_max}  ({L} windows of ~{(yr_max-yr_min)//L} years each)")

    # ── [2/8] Build rating matrix ─────────────────────────────────────────
    print(f"\n[2/8] Building rating matrix")
    df = df_raw.copy()
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["item_id"].astype("category").cat.codes

    n_users = int(df["user_idx"].nunique())
    n_items = int(df["item_idx"].nunique())

    user_map = df[["user_id", "user_idx"]].drop_duplicates()
    item_map = df[["item_id", "item_idx"]].drop_duplicates()
    user_map.to_csv(os.path.join(PROC_DIR, "user_id_map.csv"), index=False)
    item_map.to_csv(os.path.join(PROC_DIR, "item_id_map.csv"), index=False)

    R = sp.coo_matrix(
        (df["rating"].values.astype(np.float32),
         (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    ).asfptype()
    density = n_rows_raw / (n_users * n_items) * 100
    print(f"  Matrix  : {n_users} users x {n_items} items")
    print(f"  Non-zeros: {n_rows_raw:,}  (density {density:.3f}%)")

    # ── [3/8] SVD ─────────────────────────────────────────────────────────
    print(f"\n[3/8] Running SVD  (latent_dim = {D})")
    svd_k = min(D, min(n_users, n_items) - 1)
    if svd_k < D:
        print(f"  WARNING: matrix too small for D={D}; using k={svd_k}")

    t0_svd = time.time()
    u_svd, s_svd, vt_svd = svds(R, k=svd_k)
    s_sqrt = np.diag(np.sqrt(s_svd))
    U_static = (u_svd @ s_sqrt).astype(np.float64)
    P_mat    = ((s_sqrt @ vt_svd).T).astype(np.float64)

    eps = 1e-12
    U_static /= np.maximum(np.linalg.norm(U_static, axis=1, keepdims=True), eps)
    P_mat    /= np.maximum(np.linalg.norm(P_mat,    axis=1, keepdims=True), eps)
    U_static  = U_static.astype(np.float32)
    S         = P_mat.astype(np.float32)          # item embeddings (n_items, D)

    np.save(os.path.join(PROC_DIR, "U_user_vectors.npy"), U_static)
    np.save(os.path.join(PROC_DIR, "P_item_vectors.npy"), S)
    svd_time = time.time() - t0_svd
    print(f"  U: {U_static.shape}  S (items): {S.shape}  [{svd_time:.2f}s]")

    # ── [4/8] Temporal windows ────────────────────────────────────────────
    print(f"\n[4/8] Creating temporal windows  (L = {L})")
    ts_arr = df["timestamp"].values.astype(np.float64)
    bins   = np.linspace(ts_arr.min(), ts_arr.max() + 1.0, L + 1)
    tw_arr = np.clip(np.digitize(ts_arr, bins) - 1, 0, L - 1)
    df["time_window"] = tw_arr

    for t in range(L):
        yr_lo = datetime.fromtimestamp(bins[t]   / 1000).year
        yr_hi = datetime.fromtimestamp(bins[t+1] / 1000).year
        cnt   = int((tw_arr == t).sum())
        print(f"  Window {t}  [{yr_lo} – {yr_hi}]: {cnt:,} ratings")

    # Build W: temporal user preference tensor (n_users, L, D)
    S_d     = S.astype(np.float64)
    u_arr   = df["user_idx"].values.astype(int)
    i_arr   = df["item_idx"].values.astype(int)
    w_arr   = tw_arr.astype(int)

    overall_sums   = np.zeros((n_users, D),    dtype=np.float64)
    overall_counts = np.zeros(n_users,          dtype=np.int32)
    window_sums    = np.zeros((n_users, L, D), dtype=np.float64)
    window_counts  = np.zeros((n_users, L),    dtype=np.int32)

    t0_w = time.time()
    np.add.at(overall_sums,   u_arr, S_d[i_arr])
    np.add.at(overall_counts, u_arr, 1)
    for t in range(L):
        mask = w_arr == t
        if mask.any():
            np.add.at(window_sums[:, t, :], u_arr[mask], S_d[i_arr[mask]])
            np.add.at(window_counts[:, t],  u_arr[mask], 1)

    overall_avg = overall_sums / np.maximum(overall_counts[:, None], 1.0)
    nrm = np.linalg.norm(overall_avg, axis=1, keepdims=True)
    overall_avg /= np.maximum(nrm, eps)
    overall_avg[overall_counts == 0] = 0.0

    W = np.zeros((n_users, L, D), dtype=np.float32)
    for uu in range(n_users):
        for t in range(L):
            if window_counts[uu, t] > 0:
                vec = window_sums[uu, t] / window_counts[uu, t]
                nrm_v = np.linalg.norm(vec)
                W[uu, t] = vec / max(nrm_v, eps)
            else:
                W[uu, t] = W[uu, t - 1] if t > 0 else overall_avg[uu]

    np.save(os.path.join(PROC_DIR, "W_temporal_vectors.npy"), W)
    print(f"  W: {W.shape}  (built in {time.time()-t0_w:.2f}s)")

    # ── [5/8] Pre-compute rank tables ─────────────────────────────────────
    print(f"\n[5/8] Pre-computing per-window rank tables")
    t0_tab = time.time()
    T_table_list, THR_list = [], []
    for t in range(L):
        T_t, THR_t = build_rank_table(W[:, t, :], S)
        T_table_list.append(T_t)
        THR_list.append(THR_t)
    avg_users    = W.mean(axis=1)                  # (n_users, D)
    T_global, THR_global = build_rank_table(avg_users, S)
    tab_time = time.time() - t0_tab
    print(f"  {L} per-window tables + 1 global-avg table  [{tab_time:.2f}s]")

    # ── [6/8] Run static brute-force baseline ─────────────────────────────
    print(f"\n[6/8] Running static reverse top-k baseline  (snapshot at window 0)")
    rng         = np.random.default_rng(SEED)
    queries     = rng.choice(n_items, size=NUM_QUERIES, replace=False)
    U_snap      = W[:, 0, :]                       # first time window

    static_rows = []
    t0_stat = time.time()
    for qi, q_idx in enumerate(queries):
        q_idx = int(q_idx)
        t0 = time.time()
        res = static_reverse_topk_brute(S, U_snap, q_idx, K, CHUNK)
        static_rows.append({
            "query_item":   q_idx,
            "n_qualifiers": len(res),
            "query_time_s": time.time() - t0,
        })
        if (qi + 1) % 20 == 0:
            print(f"  Static baseline: {qi+1}/{NUM_QUERIES} queries done")

    df_static = pd.DataFrame(static_rows)
    df_static.to_csv(os.path.join(OUT_DIR, "amazon_static_results.csv"), index=False)
    static_total = time.time() - t0_stat
    print(f"  Done in {static_total:.2f}s  |  avg qualifiers/query: "
          f"{df_static['n_qualifiers'].mean():.1f}")

    # ── [7/8] Run all temporal methods ────────────────────────────────────
    print(f"\n[7/8] Running temporal methods over {NUM_QUERIES} queries")
    print(f"  (Full_SSA / Full_PRA / 8 hybrid variants)")

    tb, te = 0, L

    HYBRID_METHODS = [
        ("Hybrid_SSA_GlobalAvg",
         lambda: hybrid_ssa_query(S, W, q_idx, K, C, tb, te, TAU, T_global, THR_global)),
        ("Hybrid_PRA_GlobalAvg",
         lambda: hybrid_pra_query(S, W, q_idx, K, C, tb, te, TAU, T_global, THR_global)),
        ("Hybrid_SSA_UnionWin",
         lambda: hybrid_ssa_query_union(S, W, q_idx, K, C, tb, te, TAU, T_table_list, THR_list)),
        ("Hybrid_PRA_UnionWin",
         lambda: hybrid_pra_query_union(S, W, q_idx, K, C, tb, te, TAU, T_table_list, THR_list)),
        ("Hybrid_SSA_DurableRank",
         lambda: hybrid_ssa_query_durable_rank(S, W, q_idx, K, C, tb, te, TAU, T_table_list, THR_list)),
        ("Hybrid_PRA_DurableRank",
         lambda: hybrid_pra_query_durable_rank(S, W, q_idx, K, C, tb, te, TAU, T_table_list, THR_list)),
        ("Hybrid_SSA_LegacyMinRank",
         lambda: hybrid_ssa_query_minrank(S, W, q_idx, K, C, tb, te, TAU)),
        ("Hybrid_PRA_LegacyMinRank",
         lambda: hybrid_pra_query_minrank(S, W, q_idx, K, C, tb, te, TAU)),
    ]
    ALL_METHODS = (["Full_SSA", "Full_PRA"]
                   + [name for name, _ in HYBRID_METHODS])

    query_log = []

    for qi, q_idx in enumerate(queries):
        q_idx = int(q_idx)

        row = {"query_item": q_idx}

        # Full SSA (ground truth)
        t0 = time.time()
        runs_ssa = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=CHUNK)
        res_ssa  = drtopk_ssa_query(runs_ssa, tb, te, TAU)
        row["Full_SSA_time"]  = time.time() - t0
        row["Full_SSA_count"] = len(res_ssa)
        set_ssa = set(res_ssa)
        row["baseline_empty"] = len(set_ssa) == 0

        # Full PRA (cross-check ground truth)
        t0 = time.time()
        runs_pra = build_ssa_for_object_chunked(S, W, q_idx, K, chunk_size=CHUNK)
        parent   = build_pra_forest(runs_pra, L)
        res_pra  = drtopk_pra_query(runs_pra, parent, tb, te, TAU)
        row["Full_PRA_time"]  = time.time() - t0
        row["Full_PRA_count"] = len(res_pra)

        # All hybrid methods
        for name, func in HYBRID_METHODS:
            t0  = time.time()
            res = func()
            t_h = time.time() - t0

            cands   = res["num_candidates"]
            set_hyb = set(res["result_user_ids"])
            tp, fp, fn, prec, rec, f1, exact = calc_metrics(set_ssa, set_hyb)

            row[f"{name}_time"]    = t_h
            row[f"{name}_cands"]   = cands
            row[f"{name}_pruning"] = 1.0 - cands / n_users
            row[f"{name}_tp"]      = tp
            row[f"{name}_fp"]      = fp
            row[f"{name}_fn"]      = fn
            row[f"{name}_prec"]    = prec
            row[f"{name}_recall"]  = rec
            row[f"{name}_f1"]      = f1
            row[f"{name}_exact"]   = exact

        query_log.append(row)

        if (qi + 1) % 10 == 0 or qi == 0:
            n_qual = row["Full_SSA_count"]
            print(f"  Query {qi+1:3d}/{NUM_QUERIES}  item={q_idx:4d}  "
                  f"durable_users={n_qual}  "
                  f"SSA={row['Full_SSA_time']:.3f}s  "
                  f"HySSA_DR={row['Hybrid_SSA_DurableRank_time']:.3f}s")

    # ── [8/8] Save results ────────────────────────────────────────────────
    print(f"\n[8/8] Saving results to {OUT_DIR}")

    df_temporal = pd.DataFrame(query_log)
    df_temporal.to_csv(os.path.join(OUT_DIR, "amazon_temporal_results.csv"), index=False)

    # Aggregate statistics
    df_ne = df_temporal[~df_temporal["baseline_empty"]]   # non-empty baseline queries
    n_ne  = len(df_ne)
    n_emp = NUM_QUERIES - n_ne
    print(f"\n  Non-empty baseline queries : {n_ne}/{NUM_QUERIES}  "
          f"(empty: {n_emp})")

    summary_rows = []
    for name in ALL_METHODS:
        col_t = f"{name}_time"
        avg_t = df_temporal[col_t].mean()
        if name.startswith("Full"):
            summary_rows.append({
                "Method": name, "AvgTime_s": avg_t,
                "AvgRecall": "N/A", "AvgPrecision": "N/A",
                "AvgPruning": "N/A", "AvgExactMatch": "N/A",
                "AvgCandidates": "N/A",
                "SpeedupVsSSA": "1.00x" if name == "Full_SSA" else "N/A",
            })
        else:
            avg_rec  = df_ne[f"{name}_recall"].mean()  if n_ne > 0 else float("nan")
            avg_prec = df_ne[f"{name}_prec"].mean()    if n_ne > 0 else float("nan")
            avg_prun = df_temporal[f"{name}_pruning"].mean()
            avg_ex   = df_temporal[f"{name}_exact"].mean()
            avg_can  = df_temporal[f"{name}_cands"].mean()
            t_ssa    = df_temporal["Full_SSA_time"].mean()
            speedup  = t_ssa / avg_t if avg_t > 0 else float("nan")
            summary_rows.append({
                "Method": name, "AvgTime_s": avg_t,
                "AvgRecall": avg_rec, "AvgPrecision": avg_prec,
                "AvgPruning": avg_prun, "AvgExactMatch": avg_ex,
                "AvgCandidates": avg_can,
                "SpeedupVsSSA": f"{speedup:.2f}x",
            })

    df_summary = pd.DataFrame(summary_rows)

    # amazon_temporal_results already saved; save HyDART-RQ specific summary
    df_hydart = df_summary[df_summary["Method"].str.startswith("Hybrid")]
    df_hydart.to_csv(os.path.join(OUT_DIR, "amazon_hydart_rq_results.csv"), index=False)

    # Full summary (all methods)
    df_summary.to_csv(os.path.join(OUT_DIR, "amazon_summary.csv"), index=False)

    # Print comparison table
    print(f"\n{'Method':<30} {'Time(s)':>8} {'Recall':>8} {'Precision':>10}"
          f" {'Pruning':>8} {'Exact':>6} {'Speedup':>8}")
    print("-" * 84)
    for r in summary_rows:
        def fmt(v, width, decimals=4):
            return f"{v:{width}.{decimals}f}" if isinstance(v, float) else f"{str(v):>{width}}"
        t_str    = f"{r['AvgTime_s']:8.4f}"
        rec_str  = fmt(r["AvgRecall"],    8)
        prec_str = fmt(r["AvgPrecision"], 10)
        prun_str = fmt(r["AvgPruning"],   8)
        ex_str   = fmt(r["AvgExactMatch"],6)
        sp_str   = f"{str(r['SpeedupVsSSA']):>8}"
        print(f"  {r['Method']:<28} {t_str} {rec_str} {prec_str} {prun_str} {ex_str} {sp_str}")

    # ── Text summary file ─────────────────────────────────────────────────
    t_ssa_avg = df_temporal["Full_SSA_time"].mean()
    t_hdr_avg = df_temporal["Hybrid_SSA_DurableRank_time"].mean()
    speedup_hdr = t_ssa_avg / t_hdr_avg if t_hdr_avg > 0 else float("nan")
    recall_hdr  = df_ne["Hybrid_SSA_DurableRank_recall"].mean() if n_ne > 0 else float("nan")
    pruning_hdr = df_temporal["Hybrid_SSA_DurableRank_pruning"].mean()
    cands_hdr   = df_temporal["Hybrid_SSA_DurableRank_cands"].mean()
    cand_ratio  = 1.0 - cands_hdr / n_users

    total_time = time.time() - exp_start

    lines = [
        "=" * 64,
        "  HyDART-RQ Amazon Video Games Experiment Summary",
        "=" * 64,
        "",
        "Dataset",
        f"  Name         : Amazon Reviews 2023 - Video Games",
        f"  Path         : {CSV_PATH}",
        f"  Total rows   : {n_rows_raw:,}",
        f"  Users        : {n_users:,}",
        f"  Items        : {n_items:,}",
        f"  Year range   : {yr_min} - {yr_max}",
        "",
        "Experiment Parameters",
        f"  latent_dim   : {D}",
        f"  k            : {K}",
        f"  tau          : {TAU}",
        f"  c            : {C}",
        f"  time_windows : {L}",
        f"  query_items  : {NUM_QUERIES}",
        f"  seed         : {SEED}",
        "",
        "Query Statistics",
        f"  Non-empty baseline : {n_ne}/{NUM_QUERIES} queries",
        f"  Empty baseline     : {n_emp}/{NUM_QUERIES} queries",
        f"  Avg durable users  : {df_temporal['Full_SSA_count'].mean():.2f} per query",
        "",
        "Method Performance (averages over all queries)",
    ]
    for r in summary_rows:
        rec  = (f"{r['AvgRecall']:.4f}"    if isinstance(r["AvgRecall"], float)    else r["AvgRecall"])
        prec = (f"{r['AvgPrecision']:.4f}" if isinstance(r["AvgPrecision"], float) else r["AvgPrecision"])
        prun = (f"{r['AvgPruning']:.4f}"   if isinstance(r["AvgPruning"], float)   else r["AvgPruning"])
        ex   = (f"{r['AvgExactMatch']:.4f}"if isinstance(r["AvgExactMatch"], float) else r["AvgExactMatch"])
        lines.append(
            f"  {r['Method']:<30} time={r['AvgTime_s']:.4f}s  rec={rec}"
            f"  prec={prec}  pruning={prun}  exact={ex}"
            f"  speedup={r['SpeedupVsSSA']}"
        )
    lines += [
        "",
        "HyDART-RQ (Hybrid_SSA_DurableRank) Key Metrics",
        f"  Avg time           : {t_hdr_avg:.4f}s",
        f"  Speedup vs SSA     : {speedup_hdr:.2f}x",
        f"  Avg recall         : {recall_hdr:.4f}",
        f"  Avg pruning ratio  : {pruning_hdr:.4f}",
        f"  Candidate reduction: {cand_ratio:.4f}",
        f"  Avg candidates     : {cands_hdr:.1f} / {n_users} users",
        "",
        "Output Files",
        f"  {os.path.join(OUT_DIR, 'amazon_static_results.csv')}",
        f"  {os.path.join(OUT_DIR, 'amazon_temporal_results.csv')}",
        f"  {os.path.join(OUT_DIR, 'amazon_hydart_rq_results.csv')}",
        f"  {os.path.join(OUT_DIR, 'amazon_summary.csv')}",
        f"  {os.path.join(OUT_DIR, 'amazon_experiment_summary.txt')}",
        "",
        "Observations",
        (f"  HyDART-RQ achieves {speedup_hdr:.1f}x speedup over brute-force SSA "
         f"with {recall_hdr:.4f} recall on Amazon Video Games."),
        (f"  The durable-quantile candidate filter reduces the user space by "
         f"{cand_ratio*100:.1f}%, examining only {cands_hdr:.0f} of {n_users} users."),
        (f"  With tau={TAU} and L={L}, {n_ne} of {NUM_QUERIES} sampled items "
         f"have at least one durable user in their reverse top-{K}."),
        "",
        f"Total experiment runtime : {total_time:.1f}s",
        "=" * 64,
    ]

    summary_text = "\n".join(lines)
    summary_path = os.path.join(OUT_DIR, "amazon_experiment_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n  Saved: amazon_static_results.csv")
    print(f"  Saved: amazon_temporal_results.csv")
    print(f"  Saved: amazon_hydart_rq_results.csv")
    print(f"  Saved: amazon_summary.csv")
    print(f"  Saved: amazon_experiment_summary.txt")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        hybrid_names   = [r["Method"] for r in summary_rows if r["Method"].startswith("Hybrid")]
        hybrid_recalls = [r["AvgRecall"] for r in summary_rows if r["Method"].startswith("Hybrid")]
        hybrid_times   = [r["AvgTime_s"] for r in summary_rows if r["Method"].startswith("Hybrid")]

        if n_ne > 0 and any(isinstance(v, float) for v in hybrid_recalls):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Amazon Video Games – HyDART-RQ Evaluation", fontsize=13)

            short = [n.replace("Hybrid_", "").replace("_", "\n") for n in hybrid_names]
            recalls_clean = [v if isinstance(v, float) else 0.0 for v in hybrid_recalls]

            axes[0].barh(short, recalls_clean, color="steelblue")
            axes[0].set_xlabel("Average Recall")
            axes[0].set_title("Recall by Method")
            axes[0].set_xlim(0, 1.05)

            axes[1].barh(short, hybrid_times, color="coral")
            axes[1].set_xlabel("Avg Time (s)")
            axes[1].set_title("Query Time by Method")

            plt.tight_layout()
            plot_path = os.path.join(OUT_DIR, "amazon_method_comparison.png")
            plt.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Saved: amazon_method_comparison.png")
    except Exception as e:
        print(f"  (Plot skipped: {e})")

    print(f"\n{sep}")
    print(f"  DONE  |  Total runtime: {total_time:.1f}s")
    print(f"{sep}")
    print(f"\n  HyDART-RQ speedup  : {speedup_hdr:.2f}x")
    print(f"  Recall             : {recall_hdr:.4f}")
    print(f"  Candidate pruning  : {pruning_hdr*100:.1f}%")
    print(f"\n  Results directory  : {OUT_DIR}\n")


if __name__ == "__main__":
    main()
