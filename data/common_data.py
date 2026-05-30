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

if __name__ == "__main__":
    generate_dummy_data()
