import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from baselines.durable.ssa import build_ssa_for_object_chunked, drtopk_ssa_query
from baselines.durable.pra import build_pra_forest, drtopk_pra_query

def ssa_on_candidates(S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size=4000):
    """
    S: all items (n_items, d)
    W: all users' temporal preferences (n_users, L, d)
    candidate_user_ids: List of candidates obtained from filter
    Returns mapping of verified true user IDs.
    """
    # 1. Isolate the W corresponding to candidates only for durable verifications
    W_subset = W[candidate_user_ids, :, :]
    
    # 2. Build SSA on candidates chunks vs all elements
    runs_per_user = build_ssa_for_object_chunked(S, W_subset, q_idx, k, chunk_size)
    
    # 3. Assess durable queries within interval tb to te
    result_subset_indices = drtopk_ssa_query(runs_per_user, tb, te, tau)
    
    # 4. Map subset indices from drtopk_ssa_query back to global user ID values
    result_user_ids = [candidate_user_ids[idx] for idx in result_subset_indices]
    
    return result_user_ids

def pra_on_candidates(S, W, candidate_user_ids, q_idx, k, tb, te, tau, chunk_size=4000):
    """
    S: all items (n_items, d)
    W: all users' temporal preferences (n_users, L, d)
    candidate_user_ids: List of candidates obtained from filter
    Returns mapping of verified true user IDs.
    """
    # 1. Isolate the W corresponding to candidates only for durable verifications
    W_subset = W[candidate_user_ids, :, :]
    L = W.shape[1]
    
    # 2. Extract runs via SSA for candidates
    runs_per_user = build_ssa_for_object_chunked(S, W_subset, q_idx, k, chunk_size)
    
    # 3. Build PRA Forest on the candidates' runs
    parent = build_pra_forest(runs_per_user, L)
    
    # 4. Assess durable queries via PRA tree
    result_subset_indices = drtopk_pra_query(runs_per_user, parent, tb, te, tau)
    
    # 5. Map subset indices back to global user ID values
    result_user_ids = [candidate_user_ids[idx] for idx in result_subset_indices]
    
    return result_user_ids
