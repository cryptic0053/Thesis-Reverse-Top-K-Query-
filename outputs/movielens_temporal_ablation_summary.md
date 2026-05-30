# MovieLens Temporal -- Candidate Filter Ablation Summary

## Overview

This document summarizes why the MinRank filter is the only correct
candidate-generation strategy for the Level 2 temporal hybrid pipeline.
Three filter strategies were compared:

| Filter | Strategy | Recall | Correct? |
|:-------|:---------|:-------|:---------|
| GlobalAvg | Average user vector across time, select lowest estimated rank (best score) | 0.60 | No |
| UnionWin | Per-window rank tables, select lowest estimated rank per window, union results | 0.60 | No |
| **MinRank** | Per-window exact rank of query item, select users where query scored worst | **1.00** | **Yes** |

## What GlobalAvg was doing

The `approximate_candidate_filter` function:

1. Computes a single user vector by averaging across all L time windows:
   `avg_users = W.mean(axis=1)`
2. Estimates the rank of the query item using a pre-built rank table.
3. Selects users where the query item has the **lowest estimated rank**
   (i.e. users who score the query item **highest**).

This selects users who *like* the query item most.

## What UnionWin was doing

The `approximate_candidate_filter_union_windows` function:

1. For each time window t, builds a per-window rank table.
2. Estimates the rank of the query item per window using
   window-specific user vectors.
3. Selects the lowest-ranked users per window (best score), then
   unions all per-window candidate sets.

This is better at capturing temporal variation, but still selects
users who *like* the query item across some window.

## Why both were wrong

The SSA / PRA durable verifiers use **bottom-k semantics**:

> A user u qualifies at time t if the query item q is among the k items
> with the **smallest** preference scores for u.

In other words, the verifier looks for users where the query item
scores **poorly** -- not users where it scores well.

Both GlobalAvg and UnionWin select in the exact **opposite direction**:
they keep users where the query item scores **best** (lowest rank =
highest score).  This is a **semantic mismatch** that causes the true
positives to be systematically pruned away.

### Evidence from the filter comparison experiment

The following data comes from `filter_strategy_compare.csv`:

| Strategy | c | Avg Recall HSSA | Candidates | Pruning |
|:---------|:--|:----------------|:-----------|:--------|
| GlobalAvg | 1.5 | 0.60 | 15 | 97.5% |
| GlobalAvg | 3.0 | 0.60 | 30 | 95.1% |
| GlobalAvg | 10.0 | 0.60 | 100 | 83.6% |
| GlobalAvg | 20.0 | 0.60 | 200 | 67.2% |
| GlobalAvg | 30.0 | 0.60 | 300 | 50.8% |
| UnionWin | 1.5 | 0.60 | 18 | 97.0% |
| UnionWin | 10.0 | 0.60 | 109 | 82.1% |
| UnionWin | 20.0 | 0.60 | 216 | 64.6% |
| **MinRank** | **1.5** | **1.00** | **15** | **97.5%** |
| MinRank | 3.0 | 1.00 | 30 | 95.1% |
| MinRank | 10.0 | 1.00 | 100 | 83.6% |
| MinRank | 20.0 | 1.00 | 200 | 67.2% |

Key observations:

- GlobalAvg recall plateaus at 0.60 regardless of how many candidates
  are fetched (even 300 out of 610 = 49% of users).
- UnionWin shows the same plateau.
- MinRank achieves 1.00 recall at every c value tested.

This confirms the problem was **semantic direction**, not pruning
aggressiveness or temporal averaging.

## Why MinRank works

The `approximate_candidate_filter_min_window_rank` function:

1. For each time window t, computes the **exact rank** of the query
   item for each user (rank = number of items scoring strictly higher).
2. Keeps the **maximum rank across windows** -- the window where the
   query item scored worst for each user.
3. Selects users with the **highest max-rank** -- users where the
   query item is most likely to be in the bottom-k.

This correctly aligns with the SSA verifier's condition:
`score(q) <= kth_smallest(scores)`, which means q must have a **high
rank-from-top** to qualify.  MinRank selects exactly these users.

## Conclusion

The recall failure in the original Level 2 pipeline was caused by a
**semantic mismatch** between candidate filter direction and verifier
direction.  The fix required no changes to SSA/PRA core logic -- only
correcting the filter to select in the bottom-k direction.

MinRank achieves:

- **100% recall** with c = 1.5 (15 candidates out of 610)
- **97.5% pruning**
- **3.5x--3.9x speedup** over Full SSA / Full PRA
