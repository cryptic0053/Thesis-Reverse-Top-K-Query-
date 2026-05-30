# MovieLens Temporal -- Final Level 2 Results

## 1. Why the old filter failed

The original `approximate_candidate_filter` (GlobalAvg) and the
`approximate_candidate_filter_union_windows` (UnionWin) both estimated
the rank of the query item for each user and then selected the users
where the query item ranked **best** (lowest rank = highest score).

However, the SSA / PRA durable verifiers use **bottom-k semantics**:
a user qualifies when the query item is among the k items with the
**smallest** (worst) preference scores.  The old filters therefore
selected users in the exact **opposite** direction, systematically
excluding the true positives regardless of the relaxation factor c.

This semantic mismatch caused recall to plateau at ~0.60 even with
300 out of 610 candidates (c = 30).

## 2. Why MinRank works

The `approximate_candidate_filter_min_window_rank` (MinRank) filter:

1. Computes the **actual rank** of the query item for each user in
   every time window (rank = number of items scoring strictly higher).
2. Keeps the **maximum rank** across windows for each user -- the
   window where the query item scored worst.
3. Selects the users with the **highest** max-rank -- users where the
   query item is most likely to be in the bottom-k.

This correctly aligns with the SSA verifier's bottom-k condition
`score(q) <= kth_smallest(scores)`, capturing the true positives
that the old filters systematically missed.

## 3. Final results

| Setting | Value |
|:--------|:------|
| Dataset | MovieLens ml-latest-small |
| Items (S) | (9724, 32) |
| Users temporal (W) | (610, 5, 32) |
| Users (n) | 610 |
| Items (m) | 9724 |
| Dimension (d) | 32 |
| Time windows (L) | 5 |
| k | 10 |
| tau | 0.6 |
| c (MinRank) | 1.5 |

### Runtime

| Method | Avg Runtime | Speedup |
|:-------|:------------|:--------|
| Full SSA | 0.3849s | 1.0x (baseline) |
| Full PRA | 0.3897s | -- |
| Hybrid + SSA (MinRank) | 0.1099s | 3.5x faster |
| Hybrid + PRA (MinRank) | 0.1007s | 3.9x faster |

### Recall

| Method | Recall vs Full SSA |
|:-------|:-------------------|
| Hybrid + SSA (MinRank) | 1.0000 |
| Hybrid + PRA (MinRank) | 1.0000 |

### Pruning

| Metric | Value |
|:-------|:------|
| Avg candidates | 15.0 / 610 |
| Avg pruning ratio | 0.9754 |

## 4. Why this is the correct Level 2 result

The MinRank filter is the correct Level 2 temporal candidate filter
because:

* It achieves **100% recall** (perfect agreement with
  Full SSA), proving that no true positives are lost.
* It prunes **97.5%** of users, passing only 15
  candidates to the durable verifier.
* The hybrid pipeline is **3.5x faster** than Full SSA,
  validating the Level 2 speedup claim.
* The filter semantics are **provably aligned** with the SSA / PRA
  bottom-k verification condition.

## 5. Output files

* `D:\4-1\Lab\CSE 4000 Project  Thesis\thesis_project\outputs\csv\movielens_temporal_final\per_query_results.csv`
* `D:\4-1\Lab\CSE 4000 Project  Thesis\thesis_project\outputs\csv\movielens_temporal_final\runtime_compare.csv`
* `D:\4-1\Lab\CSE 4000 Project  Thesis\thesis_project\outputs\plots\movielens_temporal_final\runtime_bar.png`
* `D:\4-1\Lab\CSE 4000 Project  Thesis\thesis_project\outputs\plots\movielens_temporal_final\recall_bar.png`
* `D:\4-1\Lab\CSE 4000 Project  Thesis\thesis_project\outputs\plots\movielens_temporal_final\pruning_bar.png`
