# Changelog

## [2.0.0] — Energy-based retrieval

### Added
- **Approach 10**: Energy-based retrieval via graph diffusion (final architecture)
  - Builds semantic graph from word co-occurrence
  - Query injects energy at concept nodes
  - Energy diffuses through graph with damping
  - Documents rank by accumulated activation
- `experiments/approach_10_energy_based/energy_store.py` — clean reusable implementation
- Scale tests at 1K / 5K / 10K / 50K
- `docs/JOURNEY.md` — full development log
- `docs/RESULTS.md` — all measured numbers
- Realistic 6-domain corpus generator (`experiments/standard_benchmark/test_5k_scale.py`)
- 100-query evaluation harness with NDCG@10, MRR, Recall@100, P@1/5/10

### Measured (synthetic corpus, keyword-overlap relevance)
- @ 50K: P@10=97%, NDCG@10=0.969, p50=5.6ms, p95=22.6ms
- @ 10K: P@10=97%, NDCG@10=0.971, p50=0.8ms
- @ 5K:  P@10=98%, NDCG@10=0.979, p50=0.4ms

### Documented failures (kept for transparency)
- Approach 5 (bucket matrix refinement) — marginal improvement only
- Approach 6 (threshold sweep) — proved distance threshold is a dead lever
- Approach 9 (probabilistic neighborhoods) — worked at 1K, collapsed at 5K
- Generic energy-threshold optimization for Approach 10 — added overhead, no speedup

### Honest limitations recorded in README
- Synthetic test corpus
- Permissive relevance definition
- No standard benchmark (BEIR/MS MARCO) validation yet

---

## [1.0.0] — Initial release (Approach 3 baseline)

### Added
- 10-slot blueprint extraction
- 930-word vocabulary with sub-clusters
- Hybrid signature + zone index
- TF-IDF weighted distance
- Co-occurrence clustering
- Unified Encoder and Store API
- Unit and integration tests
- Architecture documentation

### Performance (1K corpus only)
- Search latency: 0.4ms p50
- P@10: 96%

### Known issue (later discovered)
- Performance degrades at scale: P@10 dropped to 58% at 5K corpus.
  This motivated the iteration that led to Approach 10.
