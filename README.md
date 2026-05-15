# SCBS — Semantic Cluster-Based Search

A retrieval system built around the question: *what if we stopped treating semantic search as a nearest-neighbor problem?*

This repository documents an iterative research project that started with a hard-bucket distance-based approach and ended with energy diffusion on a semantic graph. Each approach was measured honestly; failures are documented alongside successes.

---

## TL;DR

The final system (**Approach 10**) performs retrieval via spreading activation on a small semantic graph instead of computing geometric distance. On the test corpus, it maintains stable precision across a 50× scale increase:

| Scale | P@10 | NDCG@10 | p50 latency | p95 latency |
|-------|------|---------|-------------|-------------|
| 1K    | 86%  | —       | 0.1ms       | —           |
| 5K    | 98%  | 0.979   | 0.4ms       | 1.8ms       |
| 10K   | 97%  | 0.971   | 0.8ms       | 4.3ms       |
| 50K   | 97%  | 0.969   | 5.6ms       | 22.6ms      |

The graph remains constant-size regardless of corpus, so diffusion cost is O(1) per query. Document scoring is O(activated_docs), sub-linear in corpus size.

**These numbers are on a synthetic 6-domain test corpus with keyword-overlap relevance.** They are not standard benchmark results. Validating against BEIR or MS MARCO is the obvious next step.

---

## Repository Structure

```
scbs/
├── src/scbs/                          # Core package (Approach 3 baseline)
│   ├── encoder.py                     # Text → slot rows
│   ├── blueprint.py                   # Blueprint encoder variant
│   ├── clustering.py                  # Word co-occurrence clustering
│   ├── distance.py                    # IDF-weighted distance computation
│   ├── matrix_index.py                # Zone-based candidate filtering
│   ├── domain_voting.py               # Per-domain slot weights
│   ├── store.py                       # Index + search API
│   └── vocabulary.py                  # Vocabulary management
│
├── experiments/                       # Iteration history
│   ├── approach_3_baseline/           # Frozen reference snapshot
│   ├── approach_5_bucket_matrix/      # Partial fix to bucket filter
│   ├── approach_6_threshold_sweep/    # Proved threshold is a dead lever
│   ├── approach_9_probabilistic_neighborhoods/   # Soft membership replacement
│   ├── approach_10_energy_based/      # Final system — energy diffusion
│   │   ├── energy_store.py            # Clean implementation
│   │   ├── test_approach_10.py        # 1K benchmark
│   │   ├── test_approach_10_5k.py     # 5K scale test
│   │   ├── test_approach_10_10k.py    # 10K scale test
│   │   └── test_approach_10_50k.py    # 50K scale test
│   └── standard_benchmark/            # Realistic corpus generator + metrics
│
├── docs/
│   ├── ARCHITECTURE.md                # Original architecture notes
│   ├── JOURNEY.md                     # Full development log
│   └── RESULTS.md                     # All measured numbers in one place
│
├── tests/                             # Unit tests for Approach 3 baseline
├── examples/                          # Usage examples
├── README.md                          # This file
├── CHANGELOG.md                       # Version history
├── LICENSE
└── pyproject.toml
```

---

## How Approach 10 Works

Retrieval is reframed from "find the nearest vector" to "what concepts does this query activate, and what documents live in those activated concepts?"

### Index time
1. **Cluster words** by co-occurrence — words that appear together in the same documents end up in the same concept cluster
2. **Build semantic graph** where nodes are clusters and edges are co-occurrence strength between clusters
3. **Index documents** by which clusters their words belong to (inverted index from cluster → documents)

### Query time
1. **Inject energy** at the query's concept nodes (initial activation)
2. **Diffuse energy** through the graph for N iterations — energy flows along edges with damping
3. **Score documents** by summing the final energy at each document's concept nodes
4. **Rank** by total accumulated energy

The graph has a fixed number of nodes (default 14 clusters) regardless of corpus size, so diffusion is constant-time. Scoring touches only documents in activated clusters.

```python
from scbs.blueprint import BlueprintEncoder
from scbs.matrix_index import make_row
from experiments.approach_10_energy_based.energy_store import EnergyStore

encoder = BlueprintEncoder("vocab.json")
store = EnergyStore(n_clusters=14, diffusion_iterations=3, damping=0.7)

# Index
store.learn(corpus_texts)
for text in corpus_texts:
    _, bp, _ = encoder.encode(text)
    store.add(make_row(bp), text)
store.build()

# Search
_, qbp, _ = encoder.encode("my query")
results, stats = store.search(make_row(qbp), "my query", top_k=10)
```

---

## The Iteration Journey

Each approach was implemented, measured, and either kept or rejected based on real numbers. Failure modes are documented because they shaped what came next.

### Approach 3 — Baseline (Hard Buckets + Distance)
- Document assigned to exactly one cluster
- Search compares query to documents in matching cluster only
- **Result @ 1K:** P@10=96%, Recall=23%, p50=0.4ms
- **Result @ 5K:** P@10=58%, Recall=0.1%, p50=3.7ms
- **Verdict:** Looks good at small scale, breaks at 5K. The bucket filter is the bottleneck.

### Approach 5 — Bucket Matrix Refinement
- Attempt to improve bucket assignment quality
- **Verdict:** Marginal improvement, didn't address root cause.

### Approach 6 — Threshold Sweep
- Tested distance thresholds from 50 to 9000
- **Verdict:** Identical metrics across all thresholds. Proved the threshold parameter is a dead lever — the bucket filter is the binding constraint, not the distance cutoff.

### Approach 9 — Probabilistic Neighborhoods
- Replace hard bucket assignment with soft membership in top-K neighborhoods
- Each document belongs to multiple concept regions with membership scores
- **Result @ 1K:** P@10=87%, Recall=81%, p50=8.2ms — recall tripled
- **Result @ 5K:** P@10=92%, Recall=0.2%, p50=60ms — neighborhoods collapsed at scale
- **Verdict:** Concept worked at small scale but didn't generalize. Showed the bucket filter wasn't the only issue.

### Approach 10 — Energy-Based Retrieval (Final)
- Abandon distance computation entirely
- Build semantic graph from co-occurrence, run energy diffusion
- Documents rank by accumulated activation, not geometric distance
- **Result @ 50K:** P@10=97%, NDCG@10=0.969, p50=5.6ms
- **Verdict:** Stable at scale. The architectural shift to spreading activation broke the recall-precision-speed tradeoff that constrained earlier approaches.

---

## Honest Limitations

This is research-quality code with real measurements, but the limitations are equally real:

1. **Test corpus is synthetic.** Documents are template-generated across 6 domains. Real-world text has more variance and structure.

2. **Relevance definition is permissive.** "Relevant" means "shares at least one keyword with query." This inflates absolute precision numbers. Human-judged relevance (as in BEIR) would be stricter.

3. **No standard benchmark results.** We attempted to download BEIR SciFact but the host returned 403. Validating these numbers against MS MARCO, BEIR, or TREC is the next required step before any production claim.

4. **Custom encoder.** BlueprintEncoder works for our test data. Its behavior on arbitrary long-form text is unproven.

5. **Small vocabulary.** Our corpus has ~600 distinct words. Real corpora have orders of magnitude more, which may require different clustering strategies.

6. **Recall trades off with scale.** Because the system aggressively prioritizes high-activation matches, recall@100 drops as corpus grows (52% at 1K → 1.4% at 50K). For top-K precision-focused retrieval this is acceptable; for exhaustive recall it is not.

---

## Why This Might Be Interesting

If the numbers translate to real benchmarks (untested), three properties stand out:

1. **Architectural novelty.** Most production retrieval uses learned vector similarity (cosine, dot product) or BM25. Spreading activation on concept graphs is well-studied in cognitive science but uncommon in deployed IR systems.

2. **Constant-size graph.** The graph has 14 nodes whether the corpus has 1K or 1M documents. Diffusion cost does not grow with corpus.

3. **Stable scaling.** P@10 stayed within 1 percentage point across a 10× scale increase (5K → 50K). Most retrievers degrade more aggressively.

Whether these properties hold on real benchmarks is an open question, not a claim.

---

## Running the Experiments

```bash
# Install dependencies
pip install -e .

# Approach 10 @ 1K (fastest sanity check)
cd experiments/approach_10_energy_based
python test_approach_10.py

# Approach 10 @ 5K (~30 sec)
python test_approach_10_5k.py

# Approach 10 @ 10K (~1 min)
python test_approach_10_10k.py

# Approach 10 @ 50K (~5 min build)
python test_approach_10_50k.py
```

---

## What Would Validate or Falsify This

**Run on standard benchmarks:**
- BEIR SciFact (300 queries, 5K docs)
- BEIR TREC-COVID (50 queries, 171K docs)
- MS MARCO passage dev-small (~7K queries, 8.8M passages)

**Compare against:**
- BM25 (the long-standing baseline — typical NDCG@10 ≈ 0.65-0.70)
- Dense retrievers (DPR, SBERT, ColBERT — typical NDCG@10 ≈ 0.70-0.80)

If Approach 10 reaches BM25-level NDCG@10 on these datasets with its current latency profile, that is a real result. If it falls below, the synthetic-corpus numbers were inflated by the permissive relevance definition.

Either outcome is informative.

---

## License

See `LICENSE`.

## Notes

This project is the result of many iterations and honest measurement. The path from Approach 3 to Approach 10 was not predetermined — each approach was kept or rejected based on what the numbers showed. The experiments folder preserves the failed approaches because they are part of how the final design was found.
