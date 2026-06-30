import os
import urllib.request
import zipfile
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import numpy as np

def generate_dummy_data(n=3000, m=15000, d=5, seed=42):
    """
    Kept backward-compatible: generates U and P static arrays to disk.
    Existing scripts that manually read the .npy dumps will continue to work flawlessly.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUT, exist_ok=True)
    
    u_path = os.path.join(OUT, "U_user_vectors.npy")
    p_path = os.path.join(OUT, "P_item_vectors.npy")
    
    rng = np.random.default_rng(seed)
    
    U = rng.normal(size=(n, d)).astype(np.float32)
    P = rng.normal(size=(m, d)).astype(np.float32)
    
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    
    np.save(u_path, U)
    np.save(p_path, P)
    
    return U, P

def generate_synthetic_paper(n=3000, m=15000, d=5, L=10, dist='UN', p=0.03, sigma=0.05, seed=42):
    """
    Aligned with Durable Reverse Top-k queries on time-varying preference paper.
    dist: 'UN' (Uniform) or 'CL' (Clustered)
    p: probability that a user's preference changes significantly at a given timestamp
    sigma: variance of the random walk / noise added over time
    """
    rng = np.random.default_rng(seed)
    
    # 1. Base Items (S)
    if dist == 'CL':
        # Clustered items around a few centroids
        num_clusters = min(10, max(2, m // 1000))
        centroids = rng.normal(size=(num_clusters, d)).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        cluster_assignments = rng.choice(num_clusters, size=m)
        offsets = rng.normal(scale=0.1, size=(m, d)).astype(np.float32)
        S = centroids[cluster_assignments] + offsets
    else:
        # Uniform mapping
        S = rng.uniform(-1, 1, size=(m, d)).astype(np.float32)
        
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    
    # 2. Base Users (U) Initial Positions
    if dist == 'CL':
        num_clusters = min(10, max(2, n // 1000))
        centroids = rng.normal(size=(num_clusters, d)).astype(np.float32)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        cluster_assignments = rng.choice(num_clusters, size=n)
        offsets = rng.normal(scale=0.1, size=(n, d)).astype(np.float32)
        U = centroids[cluster_assignments] + offsets
    else:
        U = rng.uniform(-1, 1, size=(n, d)).astype(np.float32)
        
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    
    # 3. Temporal user preferences (W) via random walk with sudden shifts
    W = np.empty((n, L, d), dtype=np.float32)
    W[:, 0, :] = U
    
    for t in range(1, L):
        step = rng.normal(scale=sigma, size=(n, d)).astype(np.float32)
        
        # sudden shift modifier (chance p per user to suddenly move much further)
        mask = rng.random(size=n) < p
        shift = rng.normal(scale=0.5, size=(n, d)).astype(np.float32)
        step[mask] += shift[mask]
        
        curr_w = W[:, t-1, :] + step
        # Maintain bounds/normalization
        curr_w /= np.linalg.norm(curr_w, axis=1, keepdims=True)
        W[:, t, :] = curr_w
        
    return S, W

def load_real_embeddings(npy_path_items=None, npy_path_users=None, L=10):
    """
    Placeholder pipeline to read actual vector representations.
    Reads from .npy drops directly if paths exist.
    """
    m, n, d = 1000, 500, 10
    
    if npy_path_items and os.path.exists(npy_path_items):
        S = np.load(npy_path_items)
    else:
        S = np.random.rand(m, d).astype(np.float32)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        
    if npy_path_users and os.path.exists(npy_path_users):
        W = np.load(npy_path_users)
        # Assuming shape loaded is naturally (n, L, d) or transformed
    else:
        W = np.random.rand(n, L, d).astype(np.float32)
        W /= np.linalg.norm(W, axis=2, keepdims=True)
        
    return S, W

def load_data(mode="dummy", **kwargs):
    """
    Unified interface returning S (items matrix) and W (temporal users tensor).
    """
    if mode == "dummy":
        # Keep old format perfectly identical
        U, P = generate_dummy_data(
            n=kwargs.get('n', 3000), 
            m=kwargs.get('m', 15000), 
            d=kwargs.get('d', 5)
        )
        L = kwargs.get('L', 10)
        rng = np.random.default_rng(42)
        noise = rng.normal(scale=0.03, size=(U.shape[0], L, U.shape[1])).astype(np.float32)
        W = U[:, None, :] + noise
        W /= np.linalg.norm(W, axis=2, keepdims=True)
        return P, W
        
    elif mode == "synthetic_paper":
        return generate_synthetic_paper(
            n=kwargs.get('n', 3000),
            m=kwargs.get('m', 15000),
            d=kwargs.get('d', 5),
            L=kwargs.get('L', 10),
            dist=kwargs.get('dist', 'UN'),
            p=kwargs.get('p', 0.03),
            sigma=kwargs.get('sigma', 0.05)
        )
        
    elif mode == "real_embeddings":
        return load_real_embeddings(
            npy_path_items=kwargs.get('items_path', None),
            npy_path_users=kwargs.get('users_path', None),
            L=kwargs.get('L', 10)
        )
    elif mode == "movielens_static":
        return generate_movielens_static(d=kwargs.get('d', 32))
    elif mode == "movielens_temporal":
        return generate_movielens_temporal(L=kwargs.get('L', 5))
    elif mode == "netflix_static":
        return generate_netflix_static(
            d=kwargs.get('d', 32),
            max_users=kwargs.get('max_users', 5000),
            max_items=kwargs.get('max_items', 3000),
        )
    elif mode == "netflix_temporal":
        return generate_netflix_temporal(
            L=kwargs.get('L', 5),
            max_users=kwargs.get('max_users', 5000),
            max_items=kwargs.get('max_items', 3000),
        )
    elif mode == "amazon_video_games_static":
        return generate_amazon_video_games_static(d=kwargs.get('d', 32))
    elif mode == "amazon_video_games_temporal":
        return generate_amazon_video_games_temporal(L=kwargs.get('L', 5))
    else:
        raise ValueError(f"Unknown generator mode: {mode}")

def generate_movielens_static(d=32):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "movielens")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "movielens")
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    csv_path = os.path.join(RAW_DIR, "ratings.csv")
    if not os.path.exists(csv_path):
        print("Downloading MovieLens ml-latest-small...")
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = os.path.join(RAW_DIR, "ml-latest-small.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        
        # move files out of ml-latest-small to RAW_DIR
        extracted_dir = os.path.join(RAW_DIR, "ml-latest-small")
        for file_name in os.listdir(extracted_dir):
            os.rename(os.path.join(extracted_dir, file_name), os.path.join(RAW_DIR, file_name))
        os.rmdir(extracted_dir)
        os.remove(zip_path)

    print("Loading ratings data...")
    df = pd.read_csv(csv_path)
    
    # Map user/item ids to contiguous indices
    df['user_idx'] = df['userId'].astype('category').cat.codes
    df['item_idx'] = df['movieId'].astype('category').cat.codes
    
    # Save id maps
    user_map = df[['userId', 'user_idx']].drop_duplicates()
    item_map = df[['movieId', 'item_idx']].drop_duplicates()
    user_map.to_csv(os.path.join(PROCESSED_DIR, "user_id_map.csv"), index=False)
    item_map.to_csv(os.path.join(PROCESSED_DIR, "item_id_map.csv"), index=False)
    
    n_users = df['user_idx'].nunique()
    n_items = df['item_idx'].nunique()
    
    print(f"Building sparse matrix for {n_users} users and {n_items} items...")
    R = sp.coo_matrix((df['rating'], (df['user_idx'], df['item_idx'])), shape=(n_users, n_items)).asfptype()
    
    print(f"Running Matrix Factorization (SVD) with d={d}...")
    # k must be strictly less than min(R.shape)
    u, s, vt = svds(R, k=d)
    
    # u is (n_users, d), s is (d,), vt is (d, n_items)
    # Absorb sqrt(s) into u and vt
    s_sqrt = np.diag(np.sqrt(s))
    U = u @ s_sqrt
    P = (s_sqrt @ vt).T  # (n_items, d)
    
    # Optionally normalize them (some algorithms assume normalized vectors)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    
    u_path = os.path.join(PROCESSED_DIR, "U_user_vectors.npy")
    p_path = os.path.join(PROCESSED_DIR, "P_item_vectors.npy")
    
    np.save(u_path, U.astype(np.float32))
    np.save(p_path, P.astype(np.float32))
    
    return U.astype(np.float32), P.astype(np.float32)

def generate_movielens_temporal(L=5):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "movielens")
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "movielens")
    
    p_path = os.path.join(PROCESSED_DIR, "P_item_vectors.npy")
    csv_path = os.path.join(RAW_DIR, "ratings.csv")
    user_map_path = os.path.join(PROCESSED_DIR, "user_id_map.csv")
    item_map_path = os.path.join(PROCESSED_DIR, "item_id_map.csv")
    
    if not os.path.exists(p_path):
        raise FileNotFoundError("Level 1 P_item_vectors.npy not found!")
        
    P = np.load(p_path)
    df = pd.read_csv(csv_path)
    user_map = pd.read_csv(user_map_path)
    item_map = pd.read_csv(item_map_path)
    
    # Map raw IDs to internal indices
    df = df.merge(user_map, on="userId").merge(item_map, on="movieId")
    
    # Sort by timestamp
    df = df.sort_values("timestamp")
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].min() + (df['timestamp'].max() - df['timestamp'].min()) + 1 # +1 to inclusive upper
    
    # Create L equal bins based on time ranges
    bins = np.linspace(min_ts, max_ts, L + 1)
    df['time_window'] = np.digitize(df['timestamp'], bins) - 1
    # clamp out-of-bounds just in case
    df['time_window'] = df['time_window'].clip(0, L - 1)
    
    n_users = user_map['user_idx'].max() + 1
    d = P.shape[1]
    
    W = np.zeros((n_users, L, d), dtype=np.float32)
    
    # Compute overall average user vectors (for fallback)
    overall_sums = np.zeros((n_users, d), dtype=np.float64)
    overall_counts = np.zeros(n_users, dtype=np.int32)
    
    # Window-specific sums
    window_sums = np.zeros((n_users, L, d), dtype=np.float64)
    window_counts = np.zeros((n_users, L), dtype=np.int32)
    
    P_double = P.astype(np.float64)
    
    for _, row in df.iterrows():
        u = int(row['user_idx'])
        i = int(row['item_idx'])
        w = int(row['time_window'])
        
        vec = P_double[i]
        overall_sums[u] += vec
        overall_counts[u] += 1
        
        window_sums[u, w] += vec
        window_counts[u, w] += 1
        
    # Calculate fallbacks
    # Overall user averages
    eps = 1e-12
    overall_avg = overall_sums / np.maximum(overall_counts[:, None], 1.0)
    
    # Compute vector lengths for normalization logic
    norms = np.linalg.norm(overall_avg, axis=1, keepdims=True)
    overall_avg /= np.maximum(norms, eps)
    # Zero out those who never rated anything (though with Movielens, counts >= 20 usually)
    overall_avg[overall_counts == 0] = 0.0
    
    # Build W applying fallback logic sequentially
    for u in range(n_users):
        for t in range(L):
            if window_counts[u, t] > 0:
                vec = window_sums[u, t] / window_counts[u, t]
                vec_norm = np.linalg.norm(vec)
                W[u, t] = vec / max(vec_norm, eps)
            else:
                if t > 0:
                    W[u, t] = W[u, t - 1]
                else: # t == 0
                    W[u, t] = overall_avg[u]
                    
    # Save temporal user vectors
    w_path = os.path.join(PROCESSED_DIR, "W_temporal_vectors.npy")
    np.save(w_path, W)
    
    return P.astype(np.float32), W.astype(np.float32)


# ====================================================================
#  Netflix Prize dataset
# ====================================================================

def _iter_netflix_raw(raw_dir):
    """Yield (movieId, userId, rating, date_str) from combined_data_*.txt."""
    for part in range(1, 5):
        fpath = os.path.join(raw_dir, f"combined_data_{part}.txt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {fpath}")
        current_movie = None
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie = int(line[:-1])
                else:
                    parts = line.split(",")
                    yield current_movie, int(parts[0]), int(parts[1]), parts[2]
        print(f"  [pass] combined_data_{part}.txt")


def generate_netflix_static(d=32, max_users=5000, max_items=3000):
    """Build SVD embeddings from Netflix Prize data.

    Uses a memory-efficient two-pass approach:
      Pass 1 -- count ratings per user/item to find the top subset.
      Pass 2 -- collect only subset ratings into a DataFrame.
    """
    from collections import Counter

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "Netflix")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "netflix")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    subset_path = os.path.join(PROCESSED_DIR, "ratings_subset.csv")

    # If subset already cached, load and rebuild SVD from it
    if os.path.exists(subset_path):
        print("Loading cached Netflix subset ...")
        df = pd.read_csv(subset_path)
        df["date"] = pd.to_datetime(df["date"])
    else:
        # ---- Pass 1: count frequencies ----
        print("Pass 1: counting user/item frequencies ...")
        user_counts = Counter()
        item_counts = Counter()
        for mid, uid, rating, date_str in _iter_netflix_raw(RAW_DIR):
            user_counts[uid] += 1
            item_counts[mid] += 1

        top_users = set(u for u, _ in user_counts.most_common(max_users))
        top_items = set(m for m, _ in item_counts.most_common(max_items))
        print(f"  top users: {len(top_users)}, top items: {len(top_items)}")

        # ---- Pass 2: collect subset ratings ----
        print("Pass 2: collecting subset ratings ...")
        rows = []
        for mid, uid, rating, date_str in _iter_netflix_raw(RAW_DIR):
            if uid in top_users and mid in top_items:
                rows.append((mid, uid, rating, date_str))

        df = pd.DataFrame(rows, columns=["movieId", "userId", "rating", "date"])
        df["date"] = pd.to_datetime(df["date"])
        print(f"  subset: {len(df):,} ratings")

        # Build index maps and save
        df["user_idx"] = df["userId"].astype("category").cat.codes
        df["item_idx"] = df["movieId"].astype("category").cat.codes

        user_map = df[["userId", "user_idx"]].drop_duplicates()
        item_map = df[["movieId", "item_idx"]].drop_duplicates()
        user_map.to_csv(os.path.join(PROCESSED_DIR, "user_id_map.csv"), index=False)
        item_map.to_csv(os.path.join(PROCESSED_DIR, "item_id_map.csv"), index=False)

        df.to_csv(subset_path, index=False)
        print(f"  cached subset to {subset_path}")

    # Ensure idx columns exist (for cached load)
    if "user_idx" not in df.columns:
        df["user_idx"] = df["userId"].astype("category").cat.codes
        df["item_idx"] = df["movieId"].astype("category").cat.codes

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    print(f"Building sparse matrix for {n_users} users and {n_items} items ...")
    R = sp.coo_matrix(
        (df["rating"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    ).asfptype()

    svd_k = min(d, min(n_users, n_items) - 1)
    print(f"Running SVD with d={svd_k} ...")
    u, s, vt = svds(R, k=svd_k)

    s_sqrt = np.diag(np.sqrt(s))
    U = u @ s_sqrt
    P = (s_sqrt @ vt).T  # (n_items, d)

    U /= np.linalg.norm(U, axis=1, keepdims=True)
    P /= np.linalg.norm(P, axis=1, keepdims=True)

    np.save(os.path.join(PROCESSED_DIR, "U_user_vectors.npy"), U.astype(np.float32))
    np.save(os.path.join(PROCESSED_DIR, "P_item_vectors.npy"), P.astype(np.float32))

    print(f"  U: {U.shape}  P: {P.shape}")
    return U.astype(np.float32), P.astype(np.float32)


def generate_netflix_temporal(L=5, max_users=5000, max_items=3000):
    """Build temporal user-preference tensor from Netflix dates."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "netflix")

    p_path = os.path.join(PROCESSED_DIR, "P_item_vectors.npy")
    subset_path = os.path.join(PROCESSED_DIR, "ratings_subset.csv")
    user_map_path = os.path.join(PROCESSED_DIR, "user_id_map.csv")

    if not os.path.exists(p_path):
        print("Netflix P_item_vectors.npy not found, running static first ...")
        generate_netflix_static(max_users=max_users, max_items=max_items)

    P = np.load(p_path)
    d = P.shape[1]

    if os.path.exists(subset_path):
        print("Loading Netflix subset ratings ...")
        df = pd.read_csv(subset_path)
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise FileNotFoundError("ratings_subset.csv not found! Run netflix_static first.")

    user_map = pd.read_csv(user_map_path)
    n_users = user_map["user_idx"].max() + 1

    # ---- time windows from Netflix dates ----
    df = df.sort_values("date")
    min_ts = df["date"].min().timestamp()
    max_ts = df["date"].max().timestamp() + 1
    bins = np.linspace(min_ts, max_ts, L + 1)
    df["time_window"] = np.digitize(df["date"].apply(lambda x: x.timestamp()), bins) - 1
    df["time_window"] = df["time_window"].clip(0, L - 1)

    print(f"Building W temporal tensor ({n_users} users, L={L}, d={d}) ...")

    P_double = P.astype(np.float64)

    # vectorised accumulation
    overall_sums = np.zeros((n_users, d), dtype=np.float64)
    overall_counts = np.zeros(n_users, dtype=np.int32)
    window_sums = np.zeros((n_users, L, d), dtype=np.float64)
    window_counts = np.zeros((n_users, L), dtype=np.int32)

    u_arr = df["user_idx"].values
    i_arr = df["item_idx"].values
    w_arr = df["time_window"].values

    for idx in range(len(df)):
        u = u_arr[idx]
        i = i_arr[idx]
        w = w_arr[idx]
        vec = P_double[i]
        overall_sums[u] += vec
        overall_counts[u] += 1
        window_sums[u, w] += vec
        window_counts[u, w] += 1

    eps = 1e-12
    overall_avg = overall_sums / np.maximum(overall_counts[:, None], 1.0)
    norms = np.linalg.norm(overall_avg, axis=1, keepdims=True)
    overall_avg /= np.maximum(norms, eps)
    overall_avg[overall_counts == 0] = 0.0

    W = np.zeros((n_users, L, d), dtype=np.float32)
    for u in range(n_users):
        for t in range(L):
            if window_counts[u, t] > 0:
                vec = window_sums[u, t] / window_counts[u, t]
                vec_norm = np.linalg.norm(vec)
                W[u, t] = vec / max(vec_norm, eps)
            else:
                if t > 0:
                    W[u, t] = W[u, t - 1]
                else:
                    W[u, t] = overall_avg[u]

    w_path = os.path.join(PROCESSED_DIR, "W_temporal_vectors.npy")
    np.save(w_path, W)
    print(f"  W: {W.shape}")

    return P.astype(np.float32), W.astype(np.float32)


# ====================================================================
#  Amazon Reviews 2023 Video Games dataset
# ====================================================================

def generate_amazon_video_games_static(d=32):
    """Build SVD embeddings from Amazon Video Games medium subset.

    Returns (U, P): user vectors (n_users, d) and item vectors (n_items, d).
    Both are L2-normalised float32 arrays.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "amazon_video_games",
                           "amazon_video_games_medium.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "amazon_video_games")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Amazon dataset not found: {RAW_CSV}\n"
            "Run: python scripts/download_amazon_video_games_medium.py"
        )

    print("Loading Amazon Video Games ratings ...")
    df = pd.read_csv(RAW_CSV)

    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available: {list(df.columns)}"
        )

    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["item_id"].astype("category").cat.codes

    user_map = df[["user_id", "user_idx"]].drop_duplicates()
    item_map = df[["item_id", "item_idx"]].drop_duplicates()
    user_map.to_csv(os.path.join(PROCESSED_DIR, "user_id_map.csv"), index=False)
    item_map.to_csv(os.path.join(PROCESSED_DIR, "item_id_map.csv"), index=False)

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()
    print(f"  {n_users} users, {n_items} items, {len(df):,} ratings")

    print(f"Building sparse rating matrix ({n_users} x {n_items}) ...")
    R = sp.coo_matrix(
        (df["rating"].values.astype(np.float32),
         (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    ).asfptype()

    svd_k = min(d, min(n_users, n_items) - 1)
    print(f"Running SVD with k={svd_k} ...")
    u_svd, s_svd, vt_svd = svds(R, k=svd_k)

    s_sqrt = np.diag(np.sqrt(s_svd))
    U = u_svd @ s_sqrt
    P = (s_sqrt @ vt_svd).T  # (n_items, d)

    eps = 1e-12
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), eps)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), eps)

    np.save(os.path.join(PROCESSED_DIR, "U_user_vectors.npy"), U.astype(np.float32))
    np.save(os.path.join(PROCESSED_DIR, "P_item_vectors.npy"), P.astype(np.float32))
    print(f"  U: {U.shape}  P: {P.shape}")

    return U.astype(np.float32), P.astype(np.float32)


def generate_amazon_video_games_temporal(L=5):
    """Build temporal user-preference tensor W from Amazon review timestamps.

    Returns (P, W):
      P : item vectors  (n_items, d)  -- static SVD embeddings
      W : user tensor   (n_users, L, d) -- average item vector per time window
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "amazon_video_games",
                           "amazon_video_games_medium.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "amazon_video_games")

    p_path       = os.path.join(PROCESSED_DIR, "P_item_vectors.npy")
    user_map_path = os.path.join(PROCESSED_DIR, "user_id_map.csv")
    item_map_path = os.path.join(PROCESSED_DIR, "item_id_map.csv")

    if not os.path.exists(p_path):
        print("P_item_vectors.npy not found; running static embedding first ...")
        generate_amazon_video_games_static()

    P = np.load(p_path)
    d = P.shape[1]

    df = pd.read_csv(RAW_CSV)
    user_map = pd.read_csv(user_map_path)
    item_map = pd.read_csv(item_map_path)
    df = df.merge(user_map, on="user_id").merge(item_map, on="item_id")

    # Amazon timestamps are Unix milliseconds; bin into L equal time windows
    ts = df["timestamp"].values.astype(np.float64)
    bins = np.linspace(ts.min(), ts.max() + 1.0, L + 1)
    df["time_window"] = np.clip(np.digitize(ts, bins) - 1, 0, L - 1)

    n_users = int(user_map["user_idx"].max()) + 1
    P_d = P.astype(np.float64)
    eps = 1e-12

    u_arr = df["user_idx"].values.astype(int)
    i_arr = df["item_idx"].values.astype(int)
    w_arr = df["time_window"].values.astype(int)

    overall_sums   = np.zeros((n_users, d), dtype=np.float64)
    overall_counts = np.zeros(n_users,      dtype=np.int32)
    window_sums    = np.zeros((n_users, L, d), dtype=np.float64)
    window_counts  = np.zeros((n_users, L),    dtype=np.int32)

    print(f"Building temporal preference tensor ({n_users} users, L={L}, d={d}) ...")
    np.add.at(overall_sums,   u_arr, P_d[i_arr])
    np.add.at(overall_counts, u_arr, 1)
    for t in range(L):
        mask = w_arr == t
        if mask.any():
            np.add.at(window_sums[:, t, :], u_arr[mask], P_d[i_arr[mask]])
            np.add.at(window_counts[:, t],  u_arr[mask], 1)

    overall_avg = overall_sums / np.maximum(overall_counts[:, None], 1.0)
    norms = np.linalg.norm(overall_avg, axis=1, keepdims=True)
    overall_avg /= np.maximum(norms, eps)
    overall_avg[overall_counts == 0] = 0.0

    W = np.zeros((n_users, L, d), dtype=np.float32)
    for uu in range(n_users):
        for t in range(L):
            if window_counts[uu, t] > 0:
                vec = window_sums[uu, t] / window_counts[uu, t]
                nrm = np.linalg.norm(vec)
                W[uu, t] = vec / max(nrm, eps)
            else:
                W[uu, t] = W[uu, t - 1] if t > 0 else overall_avg[uu]

    w_path = os.path.join(PROCESSED_DIR, "W_temporal_vectors.npy")
    np.save(w_path, W)
    print(f"  W: {W.shape}")

    return P.astype(np.float32), W.astype(np.float32)


if __name__ == "__main__":
    generate_dummy_data()
