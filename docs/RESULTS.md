# Measured Results

All numbers in this document come from actual test runs. Each line corresponds to a script in `experiments/` that can reproduce it.

**Test setup:** synthetic 6-domain corpus (DevOps, Finance, HR, Security, Customer Support, Engineering), keyword-overlap relevance, 7-100 queries per test. Seed = 42.

---

## Approach 3 — Hard Buckets + Distance (Baseline)

### @ 1K corpus (test_approach_3 in approach_3_baseline)
| Metric | Value |
|--------|-------|
| P@1    | 100%  |
| P@10   | 96%   |
| Recall | 23%   |
| F1     | 29%   |
| p50    | 0.4ms |

### @ 5K corpus (standard_benchmark/test_5k_scale.py with Store)
| Metric  | Value  |
|---------|--------|
| P@1     | 58%    |
| P@10    | 58%    |
| Recall  | 0.1%   |
| p50     | 3.7ms  |

**Conclusion:** Severe degradation at 5× scale. Bucket filter is the binding constraint.

---

## Approach 6 — Threshold Sweep

Tested distance threshold values: 50, 100, 200, 500, 1000, 5000, 9000.

All values produced **identical** P@10, F1, and Recall. Conclusion: threshold parameter is a dead lever in the current architecture. The bucket filter makes selection decisions before distance comparison happens.

---

## Approach 9 — Probabilistic Neighborhoods + 5-Component Ranking

### @ 1K corpus (test_approach_9.py)
| Metric  | Value  | Δ vs Approach 3 |
|---------|--------|-----------------|
| P@1     | 86%    | -14pp           |
| P@10    | 87%    | -9pp            |
| Recall  | 81%    | **+58pp**       |
| F1      | 36%    | +7pp            |
| p50     | 8.2ms  | +7.8ms          |

### @ 5K corpus (test_approach_9_5k.py)
| Metric  | Value  |
|---------|--------|
| P@10    | 92%    |
| Recall  | 0.2%   |
| p50     | 60ms   |

**Conclusion:** Recall ceiling broken at small scale, but the centroid-based neighborhood assignment doesn't generalize to 5K. Speed degraded significantly.

---

## Approach 10 — Energy-Based Retrieval (Final)

### Complete scaling table

| Scale | P@1  | P@10 | NDCG@10 | MRR   | Recall | p50    | p95     | p99     |
|-------|------|------|---------|-------|--------|--------|---------|---------|
| 1K    | 86%  | 86%  | —       | —     | 52%    | 0.1ms  | —       | 0.3ms   |
| 5K    | 98%  | 98%  | 0.979   | —     | 14%    | 0.4ms  | 1.8ms   | 2.4ms   |
| 10K   | 98%  | 97%  | 0.971   | 0.984 | 7%     | 0.8ms  | 4.3ms   | 13.2ms  |
| 50K   | 97%  | 97%  | 0.969   | 0.977 | 1.4%   | 5.6ms  | 22.6ms  | 33.9ms  |

### Build times
| Scale | Indexing time | Doc/sec |
|-------|---------------|---------|
| 1K    | ~1.3s         | 794     |
| 5K    | ~22s          | 230     |
| 10K   | ~44s          | 227     |
| 50K   | ~214s         | 233     |

### Speed scaling analysis
- 1K → 5K (5× scale): p50 0.1 → 0.4ms (4× slowdown — sub-linear)
- 5K → 10K (2× scale): p50 0.4 → 0.8ms (2× slowdown — linear)
- 10K → 50K (5× scale): p50 0.8 → 5.6ms (7× slowdown — slightly super-linear)

Diffusion is constant-time. The growing latency comes from document scoring iterating through more docs per activated cluster.

### Concept activation behavior
- Initial query injects energy at 2-7 concept nodes (varies by query length)
- After 3 diffusion iterations with damping 0.7, typically all 14 clusters have non-zero energy
- Energy distribution is naturally sparse — top 2-4 concepts hold most energy
- Documents in those highly-activated clusters dominate the score ranking

---

## Failed Optimization: Generic Energy Threshold

Attempted to filter scored documents to only those in concepts above mean activated energy. Hypothesis: this would speed up scoring at scale without quality loss.

### @ 5K
| Version    | P@10 | NDCG@10 | p50     | p95     |
|------------|------|---------|---------|---------|
| Original   | 98%  | 0.979   | 0.40ms  | 1.80ms  |
| Optimized  | 98%  | 0.979   | 0.86ms  | 3.22ms  |

### @ 20K
| Version    | P@10 | NDCG@10 | p50     | p95      | p99      |
|------------|------|---------|---------|----------|----------|
| Original   | 97%  | 0.968   | 3.48ms  | 17.70ms  | 176.13ms |
| Optimized  | 97%  | 0.969   | 5.66ms  | 15.62ms  | 40.56ms  |

### @ 50K
| Version    | P@10 | NDCG@10 | p50     | p95      |
|------------|------|---------|---------|----------|
| Original   | 97%  | 0.969   | 5.60ms  | 22.60ms  |
| Optimized  | 97%  | 0.969   | 8.85ms  | 40.33ms  |

**Conclusion:** Quality preserved exactly, but optimized version is slower than original at p50 across all scales. p99 improved at 20K but degraded at 50K.

**Why it failed:** Energy decay during diffusion already produces a sparse distribution. Adding an explicit threshold filter added computation overhead that exceeded any savings. The original architecture was already efficient.

**Status:** Optimization discarded. Original Approach 10 retained.

---

## Cross-Approach Comparison at 5K

| Approach | P@10 | Recall | p50 |
|----------|------|--------|-----|
| Approach 3 baseline    | 58%  | 0.1%  | 3.7ms |
| Approach 9 (prob neighborhoods) | 92%  | 0.2%  | 60ms  |
| **Approach 10 (energy)**        | **98%**  | **14%**   | **0.4ms** |

The energy-based approach was the only one that retained both precision and reasonable recall at 5K scale.

---

## What These Numbers Don't Prove

The synthetic corpus has properties that may inflate results:
- ~600-word vocabulary (real corpora have millions)
- Template-generated documents (real text has more variance)
- Keyword-overlap relevance ("shares ≥1 keyword" → relevant)
- 100 query maximum per test

Real benchmark results would require BEIR, MS MARCO, or TREC with their human-judged relevance and larger scale. The next step for any production claim is running on those datasets.
