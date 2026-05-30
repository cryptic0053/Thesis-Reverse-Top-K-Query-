# Final Thesis Summary: Reverse Top-k Queries

## Main Findings
- **Fastest Method**: `Hybrid_PRA` slightly outperforms `Hybrid_SSA` by avoiding overlapping exact match recalculations, proving to be the overall fastest query processor across both Synthetic and Real distributions. Both Hybrid models are orders of magnitude faster than Full models.
- **Recall Preservation**: Recall varies highly based on data distribution (Uniform vs Clustered) and the approximation multiplier (`c`). In realistic dense datasets (`real_embeddings`), the spatial locality is strong enough that `Hybrid_SSA` and `Hybrid_PRA` achieved **1.00 (100%)** recall relative to the exact `Full` counterparts. 
- **Pruning Factor**: The approximation filter applies immense pruning. In the real dataset evaluation, the algorithm achieved a **0.97 Pruning Ratio** (removing 97% of users from verification). In synthetic tests with $n = 3000$, it hit **0.995** (99.5% discarded). Only the top $c \times k$ candidates proceed to the durable verifier. 
- **Scalability**: While the Full `SSA` and `PRA` scale linearly (and often punishingly) with the number of users ($n$) and duration ($L$), the `Hybrid` approach remains virtually flat, demonstrating exceptional scalability decoupled from database size.
- **Uniform (UN) vs Clustered (CL)**: Clustered datasets showed marginally denser candidate distributions around centroids, making the `c` parameter slightly more forgiving than Uniform, where random walk displacements scattered base users rapidly.

## Best Configurations
- **Best Performing Method**: Hybrid_PRA (combining rapid candidate filtering with hierarchical prefix tree verification).
- **Best Parameters**: 
    - `c = 1.5` offers an ideal baseline to guarantee robust candidate margins. 
    - `k = 10` is an ideal constraint.
    - `tau = 0.6` matches realistic durable spatial thresholds.
