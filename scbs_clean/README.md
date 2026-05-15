# SCBS

**Sparse Compositional Basis Search** — a retrieval system that matches BM25 on real benchmarks using sparse dictionary learning with late interaction. No learned embeddings. No GPU. No external services.

```
                              BM25       SCBS
NDCG@10                       0.3509     0.3378       96% of BM25
MAP                           0.2610     0.2533       97%
MRR                           0.5135     0.5086       99%
P@1                           0.3244     0.3867       119%  (beats BM25)
P@10                          0.2187     0.2178       100%
Recall@100                    0.6907     0.6274       91%
p50 latency                   5.82ms     11.86ms
```

*Measured on the Cranfield collection: 1,400 aerospace papers, 225 real queries, 1,837 human relevance judgments.*

---

## What it does

SCBS represents documents and queries as sparse activations over a learned dictionary of semantic atoms, then ranks via late interaction (MaxSim). The system:

- **Matches BM25** on standard IR metrics (NDCG, MAP, MRR)
- **Beats BM25** on top-1 precision by ~19%
- **Runs on CPU** with no learned models, no GPU, no API keys
- **Is fully interpretable** — you can inspect which atoms matched what
- **Has constant-cost atom affinity** computed once at index time

It is not a replacement for dense neural retrievers. Modern learned models (SBERT, BGE, ColBERT, etc.) typically beat BM25 on most BEIR datasets, and likely beat SCBS too — but at significantly higher cost in compute, memory, and operational complexity. SCBS is a strong option when those costs matter.

---

## Install

```bash
pip install scbs
```

Or from source:
```bash
git clone https://github.com/zahraarmantech/scbs.git
cd scbs
pip install -e .
```

Requirements: Python 3.9+, numpy, scikit-learn.

---

## Quickstart

```python
from scbs import Retriever

corpus = [
    "Kafka consumer group lag increased beyond limits in production.",
    "Kubernetes pod evicted due to memory pressure on node.",
    "Wire transfer flagged for manual fraud review by compliance team.",
    "Customer reported duplicate charge on their account statement.",
    # ... more documents
]
doc_ids = [f"doc-{i}" for i in range(len(corpus))]

retriever = Retriever(n_atoms=256)
retriever.fit(corpus, doc_ids)

results = retriever.search("fraud detection in payment systems", top_k=5)
for r in results:
    print(f"{r.doc_id}\t{r.score:.3f}\t{r.text[:60]}")
```

See `examples/quickstart.py` for a complete runnable demo.

---

## How it works

```
INDEX TIME
    Corpus  →  TF-IDF matrix
            →  Sparse Dictionary Learning (overcomplete, overlapping atoms)
            →  Per-document top-M sparse representation
            →  Atom-to-atom affinity matrix (cosine over atom vectors)
            →  Per-atom IDF

QUERY TIME
    Query   →  TF-IDF vector
            →  Project to atom space (sparse coding)
            →  Top-K query atoms (hard gate)
            →  Late interaction:
                  for each query atom i:
                      score += weight(i) · max_j (doc_atom_j · affinity(i, j))
            →  Rank documents by accumulated score
```

The key idea: **overlapping atoms enable late interaction**. NMF basis are nearly orthogonal (atom affinity ≈ 0.035 in our tests), which destroys cross-atom interaction signal. Dictionary learning produces atoms that genuinely overlap (mean affinity ≈ 0.13, max 0.64), which is what makes the MaxSim scoring work.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full derivation and [`docs/JOURNEY.md`](docs/JOURNEY.md) for the path from clustering through 15 iterations to this design.

---

## Benchmarks

Reproduce the result:

```bash
python benchmarks/run_cranfield.py
```

Cranfield data is included in the repository. The benchmark builds both BM25 and SCBS indexes, runs all 225 queries, and prints standard IR metrics with comparisons.

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the full result table and methodology.

---

## When to use SCBS

**Good fit:**
- You need retrieval quality close to BM25 with similar deployment simplicity
- Top-1 precision matters more than recall (SCBS beats BM25 on P@1)
- Interpretability matters (you can inspect which atoms matched)
- Resource constraints rule out neural models (no GPU, low memory, offline)
- You want a stronger sparse baseline before reaching for dense retrievers

**Not a fit:**
- You need state-of-the-art retrieval quality — use a dense model
- Your corpus relies heavily on synonyms/paraphrasing — TF-IDF foundation won't capture that as well as learned embeddings
- You need cross-lingual retrieval
- Sub-millisecond latency is required and you can tune BM25 properly

---

## API

### `Retriever(n_atoms=256, top_k_query=10, top_m_doc=15, sparsity_alpha=0.1, ...)`

The main retrieval class.

- **`n_atoms`** — Size of the dictionary. Should be overcomplete relative to corpus intrinsic dimensionality. 256 works for thousands of docs; consider 512+ for larger corpora.
- **`top_k_query`** — Number of strongest query atoms used during scoring. Hard-gates weaker atoms out.
- **`top_m_doc`** — Number of top atoms stored per document. Larger = better recall, more memory.
- **`sparsity_alpha`** — L1 regularization. Lower = denser codes = more atom overlap = better late interaction signal (but slower).

### `retriever.fit(documents, doc_ids=None, verbose=False)`
Build the index.

### `retriever.search(query, top_k=10) -> list[SearchResult]`
Retrieve top-K matching documents.

### `retriever.get_atom_stats() -> dict`
Diagnostic info: atom affinity statistics, IDF range.

### `BM25(k1=1.5, b=0.75)`
Standard BM25 implementation included for benchmark comparison.

---

## Project structure

```
scbs/
├── src/scbs/                      # The library
│   ├── retriever.py               # Main SCBS retriever (sparse dict + late interaction)
│   ├── bm25.py                    # BM25 baseline for comparison
│   └── metrics.py                 # Standard IR metrics (NDCG, MAP, MRR, etc.)
├── benchmarks/
│   ├── cranfield/                 # Cranfield IR test collection (1968)
│   └── run_cranfield.py           # Reproducible benchmark script
├── docs/
│   ├── ARCHITECTURE.md            # How the system works
│   ├── BENCHMARKS.md              # Full benchmark results
│   ├── COMPARISON_WITH_LLM.md     # SCBS vs LLM-based retrieval
│   └── JOURNEY.md                 # The research path (15 iterations)
├── examples/
│   └── quickstart.py              # Minimal usage example
├── research_history/              # All 15 approaches preserved
└── README.md
```

---

## Limitations

- **Tested on one benchmark.** Cranfield is small (1,400 docs) and specialized (aerospace). Results on BEIR or MS MARCO would be more general but were not run.
- **No comparison vs dense retrievers.** SBERT/DPR/ColBERT likely beat both BM25 and SCBS but were out of scope.
- **Speed is ~2× slower than BM25.** 11.86ms p50 vs 5.82ms. Production-fast but BM25 is faster.
- **Recall trails BM25 by ~10pp.** Sparse representations lose some terms BM25 catches.

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

If this work is useful in your research:

```
@software{scbs,
  title  = {SCBS: Sparse Compositional Basis Search},
  year   = {2025},
  url    = {https://github.com/zahraarmantech/scbs}
}
```
