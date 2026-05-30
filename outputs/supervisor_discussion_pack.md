# Supervisor Discussion Pack -- Durable Reverse Top-k Queries

## 1. Problem being solved

Given a set of items and users whose preferences change over time,
answer: "For a query item q, which users consistently rank q among
their least-preferred k items across a sufficient fraction (tau) of
time windows?"

This is the **durable reverse top-k query** on time-varying preferences.
It has applications in market analysis (identifying users who
persistently dislike a product) and recommendation monitoring.

## 2. Which two base papers were implemented

**Paper 1 -- Approximate reverse k-ranks (static):**
Proposes a rank-table-based method to quickly estimate each user's rank
for a query item without scanning all items.  This gives a fast
candidate shortlist for static (single-snapshot) preferences.

**Paper 2 -- Durable top-k queries (SSA and PRA):**
Defines durability over temporal windows and provides two exact
verification methods:
- **SSA (Scan-based Sequential Algorithm):** scans all time windows,
  builds contiguous runs of qualification, counts overlap with the
  query interval.
- **PRA (Parent-based Recursive Algorithm):** builds a forest of
  inclusion relationships between users' run patterns and prunes
  subtrees whose roots fail the durability check.

## 3. What was reproduced from each paper

- **Paper 1:** The rank-table construction and approximate reverse
  k-ranks query were reproduced.  Validated on MovieLens with 610
  users and 9724 items at d=32: **30x speedup** with **98% recall**.

- **Paper 2:** SSA and PRA were reproduced for temporal durable queries.
  Validated on synthetic and MovieLens temporal data.  Both produce
  identical results (correctness verified).

## 4. What new hybrid idea was added

A **two-level hybrid pipeline** combining Papers 1 and 2:

1. **Level 1 (Candidate Filter):** Use an approximate method to quickly
   identify a small subset of candidate users likely to pass the
   durable verification.
2. **Level 2 (Durable Verifier):** Run exact SSA or PRA only on the
   candidates.

This avoids running the expensive durable verification on all users.

## 5. Why the first temporal filter failed

The initial candidate filter averaged user vectors across time windows
and selected users where the query item scored **best** (highest
preference).  A second attempt (UnionWin) ran per-window filtering
but still selected in the same direction.

Both achieved only **~60% recall** because the SSA/PRA verifiers use
**bottom-k semantics** -- they check whether the query item is among
the k items the user likes **least**.  The filters were selecting the
exact opposite set of users.

Increasing the candidate count from 15 to 300 (out of 610) did not
help -- recall stayed at 60%.

## 6. How MinRank fixed it

The MinRank filter:

1. Computes the actual rank of the query item per user per time window
   (rank = how many items score higher than q).
2. Keeps the maximum rank across windows for each user.
3. Selects users where the query item ranked **worst** (highest from top).

This correctly aligns with the bottom-k verification condition.
Result: **100% recall** with only **15 candidates** (c=1.5, k=10).

## 7. Current best method and numbers

**Hybrid + PRA (MinRank)** on MovieLens temporal (610 users, 9724 items,
L=5 windows, d=32, k=10, tau=0.6, c=1.5):

| Metric | Value |
|:-------|:------|
| Recall vs Full SSA | **1.0000** (100%) |
| Pruning ratio | **97.54%** |
| Avg candidates | 15 / 610 |
| Speedup over Full SSA | **3.9x** |
| Speedup over Full PRA | **3.9x** |

Robustness: recall = 1.00 across all tested tau, c, L values at d=32.

## 8. What still can be improved later

1. **Scale to larger datasets:** MovieLens 1M or 25M to stress-test
   the pruning effectiveness with many more users.
2. **Test other embedding methods:** NMF, autoencoders, or neural
   collaborative filtering instead of SVD.
3. **Improve embedding quality:** The d=16 and d=64 results showed
   slight recall drops tied to SVD quality, not the filter.
4. **Extend semantics:** Explore top-k (instead of bottom-k) or mixed
   ranking conditions.
5. **Optimize MinRank runtime:** The current filter computes exact
   per-window ranks via matrix multiply.  An approximate version using
   rank tables (aligned to bottom-k) could reduce filter cost further.
6. **Theoretical analysis:** Prove recall guarantees for MinRank under
   bounded preference drift assumptions.
