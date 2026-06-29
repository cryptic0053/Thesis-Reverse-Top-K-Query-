import random

def total_ones(runs):
    return sum(e - s + 1 for s, e in runs)

def includes(runsA, runsB):
    if not runsB:
        return True
    if not runsA:
        return False

    i = 0
    for sb, eb in runsB:
        while i < len(runsA) and runsA[i][1] < sb:
            i += 1
        if i == len(runsA):
            return False
        sa, ea = runsA[i]
        if sa > sb or ea < eb:
            return False
    return True

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


def build_pra_forest(runs_per_user, L):
    """
    Builds the PRA containment forest.
    O(m^2) distance computation based on run inclusions.
    """
    m = len(runs_per_user)
    bit_counts = [total_ones(r) for r in runs_per_user]
    active = [u for u in range(m) if bit_counts[u] > 0]

    parent = [-1] * m

    active_sorted = sorted(active, key=lambda u: bit_counts[u], reverse=True)

    for v in active_sorted:
        best_p = -1
        best_dist = None

        for p in active_sorted:
            if p == v:
                continue
            if bit_counts[p] < bit_counts[v]:
                break
            
            # Tie-breaker to prevent cycles
            if bit_counts[p] == bit_counts[v] and p > v:
                continue

            if includes(runs_per_user[p], runs_per_user[v]):
                dist = bit_counts[p] - bit_counts[v]
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_p = p

        parent[v] = best_p

    return parent

def build_children(parent):
    m = len(parent)
    children = [[] for _ in range(m)]
    roots = []
    for v, p in enumerate(parent):
        if p == -1:
            roots.append(v)
        else:
            children[p].append(v)
    return roots, children

def drtopk_pra_query(runs_per_user, parent, tb, te, tau):
    Lq = te - tb
    if Lq <= 0:
        raise ValueError("Invalid interval [tb, te)")
    if not (0 <= tau <= 1):
        raise ValueError("tau_durable must be between 0 and 1")

    roots, children = build_children(parent)
    ans = []

    def dfs(v):
        ones = overlap_count(runs_per_user[v], tb, te)
        dur = ones / Lq
        if dur >= tau:
            ans.append(v)
            for c in children[v]:
                dfs(c)
        else:
            return

    for r in roots:
        dfs(r)

    return ans
