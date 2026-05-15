# Changelog

## [1.0.0] — Initial release

Sparse Compositional Basis Search — a retrieval system using sparse overcomplete dictionary learning with late interaction (MaxSim) scoring.

### Architecture
- TF-IDF vectorization
- Sparse dictionary learning (`MiniBatchDictionaryLearning`, positive constraints) with overlapping atoms
- Atom-to-atom affinity matrix computed once at index time
- Late interaction scoring: for each query atom, find the best matching document atom via MaxSim
- Hard top-K query gating, atom-level IDF weighting

### Benchmark results (Cranfield, 1,400 docs, 225 queries, human relevance)
- NDCG@10: 0.3378 (96% of BM25)
- MAP: 0.2533 (97%)
- MRR: 0.5086 (99%)
- **P@1: 0.3867 (119% — beats BM25)**
- P@10: 0.2178 (100%)
- Recall@100: 0.6274 (91%)
- p50 latency: 11.86ms

### Public API
- `scbs.Retriever` — main retriever class
- `scbs.BM25` — baseline for comparison
- `scbs.metrics` — standard IR metrics (NDCG@k, MAP, MRR, P@k, Recall@k)

### Includes
- Reproducible benchmark on Cranfield collection (data + script)
- Full research history (15 approach iterations preserved in `research_history/`)
- Architecture documentation, journey log, LLM comparison

### Honest limitations
- Tested on one benchmark only (Cranfield, small specialized aerospace corpus)
- Quality vs learned dense retrievers not measured
- Speed is ~2× slower than BM25 (~12ms vs ~6ms p50)
- Recall trails BM25 by ~10pp
