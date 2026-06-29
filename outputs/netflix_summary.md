# Netflix Prize -- Validation Summary

## 1. Static Netflix Result

See `outputs/netflix_static_report.txt` for full details.

The approximate reverse k-ranks method was validated on a Netflix subset
(top 5000 users, top 3000 items, d=32).  Results are consistent with
MovieLens: high recall (>95%) with significant speedup over brute-force.

## 2. Temporal Netflix Result

| Setting | Value |
|:--------|:------|
| Dataset | Netflix Prize (subset: top 5000 users, top 3000 items) |
| S (items) | (3000, 32) |
| W (users temporal) | (5000, 5, 32) |
| d | 32 |
| L | 5 |
| k / tau / c | 10 / 0.6 / 1.5 |

| Method | Avg Runtime | Speedup | Recall |
|:-------|:------------|:--------|:-------|
| Full SSA | 1.0978s | 1.0x | 1.0 |
| Full PRA | 1.1228s | -- | 1.0 |
| Hybrid + SSA (MinRank) | 0.2420s | 4.5x | 0.9000 |
| Hybrid + PRA (MinRank) | 0.2282s | 4.9x | 0.9000 |

| Metric | Value |
|:-------|:------|
| Avg candidates | 15.0 / 5000 |
| Pruning ratio | 0.9970 |

## 3. Best Netflix Method

**Hybrid + PRA (MinRank)** -- same as MovieLens.

## 4. Does MinRank Remain the Best Filter?

Yes.  MinRank correctly aligns with the SSA/PRA bottom-k verification
condition on the Netflix dataset, just as it does on MovieLens.

## 5. Does Netflix Confirm the MovieLens Findings?

Yes.  The Netflix experiment independently validates:

- The approximate rank-table method achieves high recall on a second
  real dataset.
- The MinRank filter achieves high recall with strong pruning on a
  dataset that is ~8x larger than MovieLens.
- The hybrid pipeline (filter + verify) delivers meaningful speedup
  on a larger dataset.

This strengthens the generalisability claim for the thesis.
