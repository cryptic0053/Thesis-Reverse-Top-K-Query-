import numpy as np
import math

def exact_durable_reverse_topk(S, W, q_idx, k, tb, te, tau):
    """
    Exact reference implementation for Durable Reverse Top-k.
    Computes conventional rank per window (larger score = better).
    Returns all users who qualify.
    """
    n_items, d = S.shape
    n_users, L, _ = W.shape
    
    query_length = te - tb
    if query_length <= 0:
        return []
        
    required_successes = math.ceil(tau * query_length)
    
    q_vec = S[q_idx]
    
    qualifying_users = []
    
    for u in range(n_users):
        successes = 0
        for t in range(tb, te):
            u_vec = W[u, t, :]
            scores = S @ u_vec
            q_score = scores[q_idx]
            
            # conventional rank: 1 + number of items strictly better
            rank = 1 + np.sum(scores > q_score)
            if rank <= k:
                successes += 1
                
        if successes >= required_successes:
            qualifying_users.append(u)
            
    return qualifying_users
