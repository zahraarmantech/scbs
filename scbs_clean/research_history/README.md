# Research History

The complete record of all approaches tried during the development of SCBS. Each folder contains the code and test scripts as they existed when measured.

This is preserved for transparency and reproducibility — not as production code. **For actual use, see `src/scbs/`** which is the clean implementation of the final architecture (Approach 15).

## Approach summary

| Folder | Status | Notes |
|--------|--------|-------|
| `approach_3_baseline/` | Frozen | Initial hard-clustering system. Failed at scale on real text. |
| `approach_5_bucket_matrix/` | Discarded | Bucket assignment refinement; marginal improvement only. |
| `approach_6_threshold_sweep/` | Discarded | Proved distance threshold is a dead lever; ruled out a class of fixes. |
| `approach_9_probabilistic_neighborhoods/` | Discarded | Soft cluster membership; worked at 1K, collapsed at 5K. |
| `approach_10_energy_based/` | Discarded | Energy diffusion on cluster graph; 21% of BM25 on Cranfield. |
| `approach_11_directed_flow/` | Discarded | Genericity penalty + directed flow on broken clusters; no change. |
| `approach_12_sdsf/` | Superseded | NMF + diffusion. First real improvement (50% of BM25). |
| `approach_13_hard_gated/` | Superseded | NMF + hard gating, no diffusion. 67% of BM25. |
| `approach_14_late_interaction/` | Falsified | Late interaction on NMF basis. Regression (59%). Useful negative result. |
| `approach_15_overlapping_dict/` | **Production version (in src/)** | Sparse dictionary + late interaction. 96% of BM25, beats P@1. |
| `standard_benchmark/` | Archived | The original synthetic test corpus generator. Numbers from this turned out to be inflated. |

See `docs/JOURNEY.md` in the repository root for the full narrative of how the architecture evolved.
